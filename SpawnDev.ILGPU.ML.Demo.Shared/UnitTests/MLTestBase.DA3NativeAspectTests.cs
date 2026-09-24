using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.UnitTesting;
using System.Text;
using System.Text.Json;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// <see cref="DepthResizeMode.NativeAspect"/> - Depth Anything 3's own preprocessing on the device -
/// against the Python reference (<c>tools/dav3/dav3_reference.py</c>, a port of the DA3 repo's
/// input_processor.py running real OpenCV), then the whole pipeline against onnxruntime.
/// </summary>
public abstract partial class MLTestBase
{
    // Every "official" reference case, with its process_res. Between them they take every stage path:
    // area->area (truck), area->copy (bath 8x, temple), area->cubic (drjohnson 331->336),
    // cubic->copy (temple grown to 896), multi-view and batch.
    // Cubic = the case has a CUBIC stage (drjohnson grows 331->336; temple896 grows 640->896).
    static readonly (string Name, int LongSide, bool Cubic)[] Da3OfficialCases =
    {
        ("off_truck", 504, false), ("off_bath", 504, false), ("off896_temple", 896, true),
        ("mv2_drj_off", 504, true), ("mv4_temple_off", 504, false), ("b2n2_temple_off", 280, false),
    };

    /// <summary>
    /// The device preprocessing reproduces the reference tensor. Unit = one uint8 LSB after
    /// normalisation (1/(255 std), ~0.0175): the only admissible difference is a pixel landing on the
    /// other side of a rounding boundary because a backend rounds a float operation differently
    /// (summation order, or a division that is not correctly rounded - see the gate below).
    /// Cheap kernels, so every backend runs it.
    /// </summary>
    [TestMethod(Timeout = 300000)]
    public async Task<string> DA3_NativeAspect_Preprocess_MatchesReference() => await RunTest(async accelerator =>
    {
        var (http, manifest) = await Da3Manifest();
        var pre = new Kernels.ImagePreprocessKernel(accelerator);
        var report = new StringBuilder();
        var failures = new List<string>();
        const float Lsb = 1f / (255f * 0.224f);   // the largest normalised step of one uint8 (G channel)

        foreach (var (name, longSide, cubic) in Da3OfficialCases)
        {
            var c = manifest.GetProperty("cases").GetProperty(name);
            var shape = Da3Ints(c.GetProperty("input_shape"));
            int h = shape[3], w = shape[4], chw = 3 * h * w;
            var input = await F32(http, $"test-refs/dav3/{name}.input.f32");
            var dims = c.GetProperty("rgba").EnumerateArray().Select(d => Da3Ints(d)).ToArray();
            for (int v = 0; v < dims.Length; v++)
            {
                int sw = dims[v][0], sh = dims[v][1];
                var bytes = await http.GetByteArrayAsync($"test-refs/dav3/{name}.v{v}.rgba");
                var packed = new int[bytes.Length / 4];
                Buffer.BlockCopy(bytes, 0, packed, 0, bytes.Length);
                using var src = accelerator.Allocate1D(packed);
                using var dst = accelerator.Allocate1D<float>(chw);
                int scratchLen = Kernels.ImagePreprocessKernel.NativeAspectScratchLength(sw, sh, longSide);
                using var scratch = accelerator.Allocate1D<int>(Math.Max(1, scratchLen));
                var (gw, gh) = pre.ForwardNativeAspect(src.View, sw, sh, dst.View, scratch.View, longSide);
                if (gw != w || gh != h)
                {
                    failures.Add($"{name}/v{v}: sized {gw}x{gh}, reference {w}x{h}");
                    continue;
                }
                var got = await dst.View.CopyToHostAsync();
                var refView = new ArraySegment<float>(input, v * chw, chw);   // a Span cannot cross the awaits below
                try
                {
                    // Our device tensor, so tools/dav3 can run onnxruntime on it and measure how far the
                    // model's own outputs move for these LSB flips (the floor for the pipeline gates).
                    var tb = new byte[got.Length * 4];
                    Buffer.BlockCopy(got, 0, tb, 0, tb.Length);
                    using var resp = await http.PostAsync($"__pmt/out/dav3/{accelerator.AcceleratorType}/native/{name}.v{v}.input.f32", new ByteArrayContent(tb));
                }
                catch { /* no sink outside PMT */ }
                if (scratchLen > 0)
                {
                    // Stage 1's packed RGBA too, so a defect can be pinned to one stage against cv2's own
                    // stage-1 image (tools/dav3/check_cv2_port.py computes it).
                    try
                    {
                        var s1 = await scratch.View.SubView(0, scratchLen).CopyToHostAsync();
                        var sb = new byte[s1.Length * 4];
                        Buffer.BlockCopy(s1, 0, sb, 0, sb.Length);
                        using var resp = await http.PostAsync($"__pmt/out/dav3/{accelerator.AcceleratorType}/native/{name}.v{v}.stage1.rgba", new ByteArrayContent(sb));
                    }
                    catch { /* no sink outside PMT */ }
                }
                var (rel, _, maxAbs) = Da3Compare(got, refView);
                int off = 0;
                for (int i = 0; i < chw; i++) if (MathF.Abs(got[i] - refView[i]) > 0.5f * Lsb) off++;
                double offPct = 100.0 * off / chw;
                report.Append($"{name}/v{v} {sw}x{sh}->{w}x{h} maxLSB={maxAbs / Lsb:F2} off={offPct:F3}% relRMS={rel:E1}; ");
                // Never more than one LSB anywhere. The reference is OpenCV's own algorithm (IPP off -
                // see dav3_reference.py). AREA is exact integer arithmetic, identical on every backend:
                // 0.000-0.016% off OpenCV (its own float rounding). CUBIC reproduces cv2's float fraction
                // and 11-bit taps, but that fraction comes from a float DIVISION, which OpenCL, WGSL and
                // GLSL do not round correctly: one ulp there moves ~0.1% of values by one LSB (MEASURED
                // 2026-09-23: CUDA/CPU/Wasm 0.014%, OpenCL/WebGPU 0.10%, WebGL 0.33% on temple896;
                // tools/dav3 ulp check reproduces 0.10-0.14% from a 1-ulp divide alone). For scale,
                // Intel IPP's cubic is 1.3% off OpenCV's own - "official" is not bit-defined at this level.
                double offGate = cubic ? 0.5 : 0.02;
                if (!(maxAbs <= 1.01 * Lsb && offPct <= offGate))
                    failures.Add($"{name}/v{v}: maxLSB {maxAbs / Lsb:F2}, {offPct:F3}% of values off");

                // The gate must be able to fail: plain bilinear to the SAME size (what the letterbox
                // kernel does, minus the pad) differs from the area/cubic reference by many LSBs.
                if (v == 0)
                {
                    pre.Forward(src.View, dst.View, sw, sh, w, h, preserveAspect: false);
                    var bil = await dst.View.CopyToHostAsync();
                    var (_, _, bilMax) = Da3Compare(bil, refView);
                    int bilOff = 0;
                    for (int i = 0; i < chw; i++) if (MathF.Abs(bil[i] - refView[i]) > 0.5f * Lsb) bilOff++;
                    double bilPct = 100.0 * bilOff / chw;
                    report.Append($"(bilinear control maxLSB={bilMax / Lsb:F1} off={bilPct:F1}%) ");
                    if (bilMax <= 1.01 * Lsb && bilPct <= (cubic ? 0.5 : 0.02))
                        failures.Add($"{name}: bilinear control PASSED the gate - it cannot discriminate");
                }
            }
        }
        if (failures.Count > 0)
            throw new Exception($"NativeAspect preprocessing != DA3 reference: {string.Join("; ", failures)} | {report}");
        return $"PASSED. {report}";
    });

    /// <summary>
    /// <see cref="DepthEstimationPipeline"/> in <see cref="DepthResizeMode.NativeAspect"/>, fed raw
    /// pictures, against onnxruntime run on the reference's own tensors: single view through the
    /// default capture/replay path (twice per picture, and a bigger picture mid-run to force the
    /// stable-buffer regrow), joint multi-view with poses and intrinsics, and the mixed-shape refusal.
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task<string> DA3_NativeAspect_Pipeline_MatchesOrt() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is AcceleratorType.WebGL or AcceleratorType.Wasm or AcceleratorType.CPU)
            throw new UnsupportedTestException($"{accelerator.AcceleratorType}: DAv3 pipeline gates run on CUDA/OpenCL/WebGPU");

        var (http, manifest) = await Da3Manifest();
        var onnxBytes = await InferenceSession.DownloadBytesChunkedAsync(http,
            HuggingFaceClient.GetDownloadUrl(ModelHub.KnownModels.DepthAnythingV3Small, "onnx/model.onnx"));
        var extBytes = await InferenceSession.DownloadBytesChunkedAsync(http,
            HuggingFaceClient.GetDownloadUrl(ModelHub.KnownModels.DepthAnythingV3Small, "onnx/model.onnx_data"));
        // Bound at 518 square, as SpawnScene binds it: NativeAspect must drive its own shapes from there.
        using var session = InferenceSession.CreateFromOnnx(accelerator, onnxBytes,
            inputShapes: new Dictionary<string, int[]> { ["pixel_values"] = new[] { 1, 1, 3, 518, 518 } },
            externalData: extBytes);
        using var pipe = new DepthEstimationPipeline(session, accelerator) { ResizeMode = DepthResizeMode.NativeAspect };

        var report = new StringBuilder();
        var failures = new List<string>();

        async Task<(int[] packed, int w, int h)> Rgba(string name, int v, JsonElement c)
        {
            var d = Da3Ints(c.GetProperty("rgba")[v]);
            var bytes = await http.GetByteArrayAsync($"test-refs/dav3/{name}.v{v}.rgba");
            var packed = new int[bytes.Length / 4];
            Buffer.BlockCopy(bytes, 0, packed, 0, bytes.Length);
            return (packed, d[0], d[1]);
        }

        void Gate(string what, float[] ours, ReadOnlySpan<float> reference, double gate)
        {
            var (rel, corr, _) = Da3Compare(ours, reference);
            report.Append($"{what} rel={rel:E1} r={corr:F6}; ");
            if (!(rel <= gate)) failures.Add($"{what}: relRMS {rel:E2} > {gate:E0}");
        }

        // ---- single view, capture/replay path; 504 -> 896 (regrow) -> 504 ----
        foreach (var (name, longSide) in new[] { ("off_truck", 504), ("off896_temple", 896), ("off_bath", 504) })
        {
            var c = manifest.GetProperty("cases").GetProperty(name);
            var shape = Da3Ints(c.GetProperty("input_shape"));
            int h = shape[3], w = shape[4];
            pipe.ProcessResolution = longSide;
            var (packed, sw, sh) = await Rgba(name, 0, c);
            if (pipe.ModelInputSize(sw, sh) != (w, h))
                failures.Add($"{name}: ModelInputSize {pipe.ModelInputSize(sw, sh)} != reference {w}x{h}");
            var reference = await F32(http, $"test-refs/dav3/{name}.predicted_depth.f32");
            for (int pass = 0; pass < 2; pass++)   // pass 0 records the plan, pass 1 replays it
            {
                var (raw, _, _, ow, oh) = await pipe.EstimateGpuRawAsync(packed, sw, sh, w, h);
                using (raw)
                {
                    if (ow != w || oh != h) { failures.Add($"{name}: output {ow}x{oh}"); continue; }
                    Gate($"{name}#{pass}", await raw.View.CopyToHostAsync(), reference, Da3DepthRelRmsGate);
                }
            }
        }

        // ---- joint multi-view: depth per view + poses + intrinsics ----
        pipe.ProcessResolution = 504;
        foreach (var name in new[] { "mv2_drj_off", "mv4_temple_off" })
        {
            var c = manifest.GetProperty("cases").GetProperty(name);
            var shape = Da3Ints(c.GetProperty("input_shape"));
            int n = shape[1], h = shape[3], w = shape[4];
            var frames = new List<int[]>(); var ws = new List<int>(); var hs = new List<int>();
            for (int v = 0; v < n; v++) { var (p, sw, sh) = await Rgba(name, v, c); frames.Add(p); ws.Add(sw); hs.Add(sh); }
            var depthRef = await F32(http, $"test-refs/dav3/{name}.predicted_depth.f32");
            // Pass 0 runs direct; the joint path captures only when a shape REPEATS, so pass 1 records and
            // replays and pass 2 is a pure replay. All three must match onnxruntime.
            for (int pass = 0; pass < 3; pass++)
            {
                using var mv = await pipe.EstimateMultiViewGpuAsync(frames, ws, hs, w, h);
                for (int v = 0; v < n; v++)
                    Gate($"{name}#{pass}/v{v}", await mv.Views[v].RawDepth.View.CopyToHostAsync(), depthRef.AsSpan(v * w * h, w * h), Da3DepthRelRmsGate);
                // The gate must be able to fail: view 0 against view 1's reference.
                var (relX, _, _) = Da3Compare(await mv.Views[0].RawDepth.View.CopyToHostAsync(), depthRef.AsSpan(w * h, w * h));
                if (relX <= Da3DepthRelRmsGate) failures.Add($"{name}: view 0 vs view 1's reference PASSED - gate cannot discriminate");

                foreach (var (outName, per, got) in new[] { ("extrinsics", 12, mv.Extrinsics), ("intrinsics", 9, mv.Intrinsics) })
                {
                    if (got == null) { failures.Add($"{name}#{pass}: {outName} missing"); continue; }
                    var r = await F32(http, $"test-refs/dav3/{name}.{outName}.f32");
                    double maxAbs = 0, refMax = 0;
                    for (int v = 0; v < n; v++)
                        for (int k = 0; k < per; k++)
                        {
                            maxAbs = Math.Max(maxAbs, Math.Abs(got[v][k] - r[v * per + k]));
                            refMax = Math.Max(refMax, Math.Abs(r[v * per + k]));
                        }
                    double rel = maxAbs / Math.Max(1e-12, refMax);
                    report.Append($"{name}#{pass}.{outName} rel={rel:E1}; ");
                    // Preprocessing differs from the reference by the odd uint8 LSB, which the camera
                    // heads see as input noise: 10x the tensor-identical parity gate.
                    if (!(rel <= 10 * Da3CameraRelGate)) failures.Add($"{name}#{pass}.{outName}: max|diff| {rel:E2} of max|ref|");
                }
            }
        }

        // ---- mixed shapes must be refused, not centre-cropped into misaligned depth ----
        {
            var ct = manifest.GetProperty("cases").GetProperty("off_truck");
            var cm = manifest.GetProperty("cases").GetProperty("mv4_temple_off");
            var (pt, twd, tht) = await Rgba("off_truck", 0, ct);
            var (pm, mwd, mht) = await Rgba("mv4_temple_off", 0, cm);
            try
            {
                using var bad = await pipe.EstimateMultiViewGpuAsync(new[] { pt, pm }, new[] { twd, mwd }, new[] { tht, mht });
                failures.Add("mixed-shape joint pass was accepted");
            }
            catch (ArgumentException ex) { report.Append($"mixed shapes refused: {ex.Message.Split(';')[0]}; "); }
        }

        if (failures.Count > 0)
            throw new Exception($"NativeAspect pipeline != onnxruntime ({failures.Count}): {string.Join("; ", failures)} | {report}");
        return $"PASSED. {report}";
    });
}
