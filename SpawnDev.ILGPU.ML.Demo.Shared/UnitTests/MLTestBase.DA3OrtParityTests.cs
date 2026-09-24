using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.Tensors;
using SpawnDev.UnitTesting;
using System.Text;
using System.Text.Json;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// DAv3 against onnxruntime on the SAME input tensor, for every output the export has
/// (predicted_depth, confidence, extrinsics, intrinsics), at every input shape SpawnScene and the
/// official DA3 pipeline produce: square letterboxed 518/672/896, the official NON-SQUARE shapes,
/// joint multi-view N=2/4/6, and batch_size=2.
///
/// Before this, "DAv3 is bit-exact vs ORT" rested on a June scratchpad console that ran CUDA, one
/// square 224 input, depth only. Nothing compared WebGPU, a non-square grid, a multi-view depth
/// map or a camera pose against a reference, and SpawnScene uses all four.
///
/// References: <c>python tools/dav3/dav3_reference.py</c> writes <c>test-refs/dav3/</c> (gitignored,
/// ~130 MB) from native onnxruntime on the byte-identical model the hub serves. Our raw outputs
/// are POSTed to PMT's <c>__pmt/out/</c> sink (<c>_mldump/test-out/dav3/&lt;backend&gt;/</c>) so
/// <c>tools/dav3/dav3_compare.py</c> can render the three engines side by side.
/// </summary>
public abstract partial class MLTestBase
{
    // relRMS = ||ours - ref|| / ||ref||, per view. float32 through ~2,600 nodes lands at 1e-6..1e-4;
    // every real DAv3 defect found so far (bilinear-for-bicubic pos-embed: 1.3%, the RoPE shape
    // collapse: 11%, the WebGPU Slice sentinel: 18% range error) sat at 1e-2 or above.
    const double Da3DepthRelRmsGate = 1e-3;
    // confidence goes through an Exp in the head, which amplifies input noise - MEASURED 2026-07-02
    // (the one residual WebGPU divergent node was /head/Exp_1).
    const double Da3ConfRelRmsGate = 5e-3;
    // extrinsics/intrinsics: max |diff| relative to the reference's own max |value|.
    const double Da3CameraRelGate = 1e-3;

    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task<string> DA3_OrtParity_SingleView() => await RunTest(async accelerator =>
        await Da3OrtParity(accelerator, "s518_truck", "off_truck", "s672_bath", "off_bath", "s896_bath"));

    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task<string> DA3_OrtParity_MultiView() => await RunTest(async accelerator =>
        await Da3OrtParity(accelerator, "mv2_drj_off", "mv4_temple_off", "mv4_temple_672", "mv6_temple_518"));

    [TestMethod(Timeout = 900000, Category = "HeavyModel")]
    public async Task<string> DA3_OrtParity_Batch2() => await RunTest(async accelerator =>
        await Da3OrtParity(accelerator, "b2n2_temple_off"));

    /// <summary>
    /// The "ours" references were made from a numpy emulation of <see cref="Kernels.ImagePreprocessKernel"/>
    /// (letterbox, preserveAspect=true). This pushes the same decoded pixels through the REAL kernel, so
    /// a parity pass on those cases is a statement about what SpawnScene actually feeds the model.
    /// </summary>
    [TestMethod(Timeout = 300000, Category = "HeavyModel")]
    public async Task<string> DA3_Preprocess_Letterbox_MatchesReference() => await RunTest(async accelerator =>
    {
        var (http, manifest) = await Da3Manifest();
        var pre = new Kernels.ImagePreprocessKernel(accelerator);
        var report = new StringBuilder();
        var failures = new List<string>();
        foreach (var name in new[] { "s518_truck", "mv4_temple_672", "mv6_temple_518" })
        {
            var c = manifest.GetProperty("cases").GetProperty(name);
            var shape = Da3Ints(c.GetProperty("input_shape"));
            int side = shape[^1], chw = 3 * side * side;
            var input = await F32(http, $"test-refs/dav3/{name}.input.f32");
            var dims = c.GetProperty("rgba").EnumerateArray().Select(d => Da3Ints(d)).ToArray();
            for (int v = 0; v < dims.Length; v++)
            {
                var bytes = await http.GetByteArrayAsync($"test-refs/dav3/{name}.v{v}.rgba");
                var packed = new int[bytes.Length / 4];
                Buffer.BlockCopy(bytes, 0, packed, 0, bytes.Length);   // R in the low byte, as the kernel reads
                using var src = accelerator.Allocate1D(packed);
                using var dst = accelerator.Allocate1D<float>(chw);
                pre.Forward(src.View, dst.View, dims[v][0], dims[v][1], side, side, preserveAspect: true);
                var got = await dst.View.CopyToHostAsync();
                var (rel, _, maxAbs) = Da3Compare(got, input.AsSpan(v * chw, chw));
                if (!(rel <= 1e-4 && maxAbs <= 5e-3)) failures.Add($"{name}/v{v} relRMS={rel:E2} maxAbs={maxAbs:E2}");
                report.Append($"{name}/v{v} maxAbs={maxAbs:E2} relRMS={rel:E2}; ");

                // The gate must be able to fail: the STRETCH path (no letterbox) differs from the
                // reference only in geometry, and has to be rejected by the same thresholds.
                if (v == 0)
                {
                    pre.Forward(src.View, dst.View, dims[v][0], dims[v][1], side, side, preserveAspect: false);
                    var (relStretch, _, _) = Da3Compare(await dst.View.CopyToHostAsync(), input.AsSpan(0, chw));
                    if (relStretch <= 1e-4) failures.Add($"{name}: stretch path PASSED the letterbox gate (relRMS {relStretch:E2}) - gate cannot discriminate");
                    report.Append($"(stretch control relRMS={relStretch:E2}) ");
                }
            }
        }
        // Thresholds are about sample COORDINATES, not values. WGSL f32 division is not correctly
        // rounded, so a ~500 px source coordinate carries a few ulp (~1e-4 px) of error, and on the
        // steepest edge (normalised step ~4 per pixel) that moves a value by ~1e-4 - MEASURED 2026-09-23:
        // maxAbs 1.4e-4, relRMS 1e-6 on every view. A half-pixel or letterbox-rect defect moves values by
        // ~1e-1 and relRMS by ~1e-2.
        if (failures.Count > 0) throw new Exception($"preprocess kernel != reference emulation: {string.Join("; ", failures)} | {report}");
        return $"PASSED. {report}";
    });

    async Task<(HttpClient http, JsonElement manifest)> Da3Manifest()
    {
        var http = GetHttpClient() ?? throw new UnsupportedTestException("HttpClient not available");
        string json;
        try { json = await http.GetStringAsync("test-refs/dav3/manifest.json"); }
        catch (Exception ex)
        {
            throw new UnsupportedTestException(
                $"DAv3 ORT references missing - run `python tools/dav3/dav3_reference.py` first ({ex.Message})");
        }
        return (http, JsonDocument.Parse(json).RootElement.Clone());
    }

    static int[] Da3Ints(JsonElement a) => a.EnumerateArray().Select(e => e.GetInt32()).ToArray();

    async Task<string> Da3OrtParity(Accelerator accelerator, params string[] cases)
    {
        // Heavy multi-second-per-forward model: the fast lanes carry correctness (ML CLAUDE.md,
        // backend priority). The Wasm/WebGL/CPU lanes are the same kernels at 10-100x the wall time.
        if (accelerator.AcceleratorType is AcceleratorType.WebGL or AcceleratorType.Wasm or AcceleratorType.CPU)
            throw new UnsupportedTestException($"{accelerator.AcceleratorType}: DAv3 ORT parity runs on CUDA/OpenCL/WebGPU");

        var (http, manifest) = await Da3Manifest();
        var onnxBytes = await InferenceSession.DownloadBytesChunkedAsync(http,
            HuggingFaceClient.GetDownloadUrl(ModelHub.KnownModels.DepthAnythingV3Small, "onnx/model.onnx"));
        var extBytes = await InferenceSession.DownloadBytesChunkedAsync(http,
            HuggingFaceClient.GetDownloadUrl(ModelHub.KnownModels.DepthAnythingV3Small, "onnx/model.onnx_data"));

        string backend = accelerator.AcceleratorType.ToString();
        var report = new StringBuilder();
        var failures = new List<string>();
        int dumped = 0, dumpFailed = 0;
        var timing = new Dictionary<string, Dictionary<string, long>>();

        foreach (var name in cases)
        {
            var c = manifest.GetProperty("cases").GetProperty(name);
            var shape = Da3Ints(c.GetProperty("input_shape"));
            var input = await F32(http, $"test-refs/dav3/{name}.input.f32");

            var sw = System.Diagnostics.Stopwatch.StartNew();
            using var session = InferenceSession.CreateFromOnnx(accelerator, onnxBytes,
                inputShapes: new Dictionary<string, int[]> { ["pixel_values"] = shape },
                externalData: extBytes);
            long createMs = sw.ElapsedMilliseconds;

            using var inBuf = accelerator.Allocate1D(input);
            var feed = new Dictionary<string, Tensor> { [session.InputNames[0]] = new Tensor(inBuf.View, shape) };

            // Cold forward (kernel compile + first run), then warm forwards for the steady-state number.
            sw.Restart();
            var outputs = await session.RunAsync(feed);
            await accelerator.SynchronizeAsync();
            long coldMs = sw.ElapsedMilliseconds;

            var line = new StringBuilder($"[{name} {string.Join("x", shape)}] ");

            // Same gates for every path that produces DAv3 outputs (direct forward, captured replay).
            async Task Check(Dictionary<string, Tensor> outs, string tag, bool dump)
            {
                if (tag.Length > 0) line.Append($"{tag}: ");
                foreach (var o in c.GetProperty("outputs").EnumerateObject())
                {
                    var refShape = Da3Ints(o.Value.GetProperty("shape"));
                    if (!outs.TryGetValue(o.Name, out var t))
                    {
                        failures.Add($"{name}{tag}.{o.Name}: output missing");
                        continue;
                    }
                    if (!t.Shape.SequenceEqual(refShape))
                    {
                        failures.Add($"{name}{tag}.{o.Name}: shape [{string.Join(",", t.Shape)}] != ORT [{string.Join(",", refShape)}]");
                        continue;
                    }
                    var reference = await F32(http, $"test-refs/dav3/{name}.{o.Name}.f32");
                    var ours = await t.Data.SubView(0, t.ElementCount).CopyToHostAsync();

                    if (dump)
                    {
                        try
                        {
                            var bytes = new byte[ours.Length * 4];
                            Buffer.BlockCopy(ours, 0, bytes, 0, bytes.Length);
                            using var resp = await http.PostAsync($"__pmt/out/dav3/{backend}/{name}.{o.Name}.f32", new ByteArrayContent(bytes));
                            if (resp.IsSuccessStatusCode) dumped++; else dumpFailed++;
                        }
                        catch { dumpFailed++; }   // no sink outside PMT (e.g. the demo's /tests page): the comparison below still runs
                    }

                    if (o.Name is "predicted_depth" or "confidence")
                    {
                        // Per VIEW: a batch/view indexing defect leaves view 0 exact and corrupts the rest
                        // (MEASURED 2026-07-01, three "computes only batch 0" kernels), which a whole-tensor
                        // statistic averages away.
                        int views = refShape[0] * refShape[1], per = ours.Length / views;
                        double gate = o.Name == "predicted_depth" ? Da3DepthRelRmsGate : Da3ConfRelRmsGate;
                        line.Append($"{o.Name}:");
                        for (int v = 0; v < views; v++)
                        {
                            var (rel, corr, _) = Da3Compare(ours.AsSpan(v * per, per), reference.AsSpan(v * per, per));
                            line.Append($" v{v} rel={rel:E1} r={corr:F6}");
                            if (!(rel <= gate)) failures.Add($"{name}{tag}.{o.Name} view {v}: relRMS {rel:E2} > {gate:E0} (corr {corr:F6})");
                        }
                        line.Append("; ");
                        // The gate must be able to fail: two different views of one scene are the closest
                        // WRONG answer there is, and must be rejected by the same threshold.
                        if (views > 1 && o.Name == "predicted_depth")
                        {
                            var (relX, _, _) = Da3Compare(ours.AsSpan(0, per), reference.AsSpan(per, per));
                            if (relX <= gate) failures.Add($"{name}{tag}: view 0 vs view 1's reference PASSED the gate (relRMS {relX:E2}) - gate cannot discriminate");
                        }
                    }
                    else
                    {
                        var (_, _, maxAbs) = Da3Compare(ours, reference);
                        double refMax = reference.Max(MathF.Abs);
                        double rel = maxAbs / Math.Max(1e-12, refMax);
                        line.Append($"{o.Name}: maxAbs={maxAbs:E1} (rel {rel:E1}); ");
                        if (!(rel <= Da3CameraRelGate)) failures.Add($"{name}{tag}.{o.Name}: max|diff| {maxAbs:E2} = {rel:E2} of max|ref| > {Da3CameraRelGate:E0}");
                    }
                }
            }

            await Check(outputs, "", dump: true);
            session.ReturnOutputs(outputs);

            var warm = new List<long>();
            for (int i = 0; i < 3; i++)
            {
                sw.Restart();
                var o2 = await session.RunAsync(feed);
                await accelerator.SynchronizeAsync();
                warm.Add(sw.ElapsedMilliseconds);
                session.ReturnOutputs(o2);
            }
            warm.Sort();

            // The captured-plan path (what DepthEstimationPipeline.EstimateGpuRawAsync uses for a single
            // view by default). EstimateMultiViewGpuAsync skips capture, so these rows answer whether the
            // JOINT path could have it: is a replay at a multi-view shape both correct and fast?
            long captureMs = -1, replayMs = -1;
            if (accelerator.AcceleratorType is AcceleratorType.WebGPU or AcceleratorType.Cuda)
            {
                sw.Restart();
                IDisposable? cap;
                Func<Dictionary<string, Tensor>, Task<Dictionary<string, Tensor>>>? replay = null;
                string? refused = null;
                if (accelerator.AcceleratorType == AcceleratorType.WebGPU)
                {
                    var w = await WebGPUGraphCapture.TryCaptureAsync(session, feed);
                    cap = w; if (w != null) replay = w.ReplayAsync;
                }
                else
                {
                    var cg = await CudaGraphCapture.TryCaptureAsync(session, feed);
                    cap = cg; if (cg != null) replay = cg.ReplayAsync; else refused = CudaGraphCapture.LastRefusalReason;
                }
                captureMs = sw.ElapsedMilliseconds;
                if (replay == null)
                {
                    failures.Add($"{name}: graph capture refused on {backend} ({refused ?? "TryCaptureAsync returned null"})");
                }
                else
                {
                    using (cap)
                    {
                        var r = await replay(feed);
                        await accelerator.SynchronizeAsync();
                        await Check(r, " replay", dump: false);   // capture-owned outputs: never ReturnOutputs
                        var rt = new List<long>();
                        for (int i = 0; i < 3; i++)
                        {
                            sw.Restart();
                            await replay(feed);
                            await accelerator.SynchronizeAsync();
                            rt.Add(sw.ElapsedMilliseconds);
                        }
                        rt.Sort();
                        replayMs = rt[1];
                    }
                }
            }

            int ortMs = c.GetProperty("ort_cpu_ms").GetInt32();
            timing[name] = new() { ["create"] = createMs, ["cold"] = coldMs, ["warm"] = warm[1], ["capture"] = captureMs, ["replay"] = replayMs };
            line.Append($"time: create={createMs}ms cold={coldMs}ms warm={warm[1]}ms capture={captureMs}ms replay={replayMs}ms (ORT-CPU {ortMs}ms)");
            Console.WriteLine($"[DA3-ORT] {backend} {line}");
            report.AppendLine(line.ToString());
        }

        try
        {
            // PMT's results JSON does not carry resultText, so the timings travel like the tensors.
            // One file per test method (keyed by its first case) so the three do not overwrite.
            using var resp = await http.PostAsync($"__pmt/out/dav3/{backend}/timing-{cases[0]}.json",
                new StringContent(JsonSerializer.Serialize(timing)));
            if (!resp.IsSuccessStatusCode) dumpFailed++;
        }
        catch { dumpFailed++; }
        string summary = $"{backend} dumped={dumped} dumpFailed={dumpFailed}\n{report}";
        if (failures.Count > 0)
            throw new Exception($"DAv3 diverges from onnxruntime ({failures.Count}):\n  {string.Join("\n  ", failures)}\n{summary}");
        return $"PASSED. {summary}";
    }

    /// <summary>relRMS (||a-b||/||b||), Pearson correlation, max |a-b|. Doubles throughout.</summary>
    static (double relRms, double corr, double maxAbs) Da3Compare(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        if (a.Length != b.Length) throw new ArgumentException($"length {a.Length} != {b.Length}");
        double ma = 0, mb = 0;
        for (int i = 0; i < a.Length; i++) { ma += a[i]; mb += b[i]; }
        ma /= a.Length; mb /= b.Length;
        double d2 = 0, b2 = 0, cab = 0, caa = 0, cbb = 0, maxAbs = 0;
        bool nan = false;
        for (int i = 0; i < a.Length; i++)
        {
            double d = a[i] - (double)b[i];
            d2 += d * d; b2 += (double)b[i] * b[i];
            double da = a[i] - ma, db = b[i] - mb;
            cab += da * db; caa += da * da; cbb += db * db;
            double ad = Math.Abs(d);
            if (double.IsNaN(ad)) nan = true;
            else if (ad > maxAbs) maxAbs = ad;
        }
        // Any NaN makes every statistic NaN, and NaN fails every `!(x <= gate)` check.
        if (nan) return (double.NaN, double.NaN, double.NaN);
        return (Math.Sqrt(d2 / Math.Max(1e-300, b2)), cab / Math.Sqrt(Math.Max(1e-300, caa * cbb)), maxAbs);
    }
}
