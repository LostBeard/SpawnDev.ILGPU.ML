using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.UnitTesting;
using System.Text;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// A joint DAv3 pass hands back EVERYTHING in the caller's pixel grid, not the model's.
/// </summary>
public abstract partial class MLTestBase
{
    /// <summary>
    /// Letterboxed joint pass (TempleRing 640x480 x4 into 672x672, the reference's own letterbox:
    /// content 672x504 at y=84) against onnxruntime:
    /// <list type="bullet">
    /// <item>At an output grid EQUAL to the content rectangle the crop is 1:1, so depth AND confidence must
    /// equal the reference planes' rows 84..587 exactly. Confidence used to be resized from the whole plane,
    /// pad included - squashed against depth.</item>
    /// <item><c>ModelIntrinsics</c> is the model's K (onnxruntime parity); <c>Intrinsics</c> is K in the output
    /// grid: at 1:1 that is the reference K with cy - 84. Handing out the model's K put SpawnScene's focal in the
    /// wrong units by the input/source scale.</item>
    /// <item>At 640x480 and 1280x960 K scales exactly with the grid, and DAv3's centred principal point lands
    /// on the output centre (it lands 84 px low in 672 units if the pad is not subtracted).</item>
    /// </list>
    /// Negative controls prove the plane gate can see a 1-row shift and the K gate a missing pad offset.
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task<string> DA3_MultiView_OutputGrid_IntrinsicsAndConfidence() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is AcceleratorType.WebGL or AcceleratorType.Wasm or AcceleratorType.CPU)
            throw new UnsupportedTestException($"{accelerator.AcceleratorType}: DAv3 pipeline gates run on CUDA/OpenCL/WebGPU");

        var (http, manifest) = await Da3Manifest();
        const string name = "mv4_temple_672";
        var c = manifest.GetProperty("cases").GetProperty(name);
        var shape = Da3Ints(c.GetProperty("input_shape"));
        int n = shape[1], mh = shape[3], mw = shape[4];
        var lb = Da3Ints(c.GetProperty("letterbox_rects")[0]);   // [contentW, contentH, padX, padY]
        int cw = lb[0], ch = lb[1], px = lb[2], py = lb[3];

        var onnxBytes = await InferenceSession.DownloadBytesChunkedAsync(http,
            HuggingFaceClient.GetDownloadUrl(ModelHub.KnownModels.DepthAnythingV3Small, "onnx/model.onnx"));
        var extBytes = await InferenceSession.DownloadBytesChunkedAsync(http,
            HuggingFaceClient.GetDownloadUrl(ModelHub.KnownModels.DepthAnythingV3Small, "onnx/model.onnx_data"));
        using var session = InferenceSession.CreateFromOnnx(accelerator, onnxBytes,
            inputShapes: new Dictionary<string, int[]> { ["pixel_values"] = new[] { 1, 1, 3, mh, mw } },
            externalData: extBytes);
        using var pipe = new DepthEstimationPipeline(session, accelerator);   // Letterbox, the default

        var report = new StringBuilder();
        var failures = new List<string>();

        var frames = new List<int[]>(); var ws = new List<int>(); var hs = new List<int>();
        for (int v = 0; v < n; v++)
        {
            var d = Da3Ints(c.GetProperty("rgba")[v]);
            var bytes = await http.GetByteArrayAsync($"test-refs/dav3/{name}.v{v}.rgba");
            var packed = new int[bytes.Length / 4];
            Buffer.BlockCopy(bytes, 0, packed, 0, bytes.Length);
            frames.Add(packed); ws.Add(d[0]); hs.Add(d[1]);
        }
        int sw = ws[0], sh = hs[0];

        var rect = pipe.ModelContentRect(sw, sh);
        if (rect != (px, py, cw, ch))
            failures.Add($"ModelContentRect {rect} != reference letterbox ({px}, {py}, {cw}, {ch})");

        var depthRef = await F32(http, $"test-refs/dav3/{name}.predicted_depth.f32");
        var confRef = await F32(http, $"test-refs/dav3/{name}.confidence.f32");
        var kRef = await F32(http, $"test-refs/dav3/{name}.intrinsics.f32");

        // Reference plane v, content rows only (rowShift = negative control).
        float[] Crop(float[] planes, int v, int rowShift)
        {
            var o = new float[cw * ch];
            for (int y = 0; y < ch; y++)
                Array.Copy(planes, (long)v * mw * mh + (long)(py + y + rowShift) * mw + px, o, y * cw, cw);
            return o;
        }

        void Plane(string what, float[] ours, float[] reference, double gate)
        {
            var (rel, corr, _) = Da3Compare(ours, reference);
            report.Append($"{what} rel={rel:E1} r={corr:F6}; ");
            if (!(rel <= gate)) failures.Add($"{what}: relRMS {rel:E2} > {gate:E0}");
        }

        double KRel(float[] got, float[] want)
        {
            double maxAbs = 0, refMax = 0;
            for (int k = 0; k < 9; k++)
            {
                maxAbs = Math.Max(maxAbs, Math.Abs(got[k] - want[k]));
                refMax = Math.Max(refMax, Math.Abs(want[k]));
            }
            return maxAbs / Math.Max(1e-12, refMax);
        }
        float[] RefK(int v) => kRef.AsSpan(v * 9, 9).ToArray();
        const double KGate = 10 * Da3CameraRelGate;   // preprocessing LSB noise, as in the NativeAspect gate

        // ---- 1:1 grid: output == content rectangle ----
        using (var mv = await pipe.EstimateMultiViewGpuAsync(frames, ws, hs, cw, ch))
        {
            if (mv.ConfidenceMaps == null) failures.Add("no confidence maps");
            if (mv.Intrinsics == null || mv.ModelIntrinsics == null) failures.Add("no intrinsics");
            for (int v = 0; v < n; v++)
            {
                var (raw, _, _, ow, oh) = mv.Views[v];
                if (ow != cw || oh != ch) { failures.Add($"v{v}: depth {ow}x{oh}, expected {cw}x{ch}"); continue; }
                Plane($"depth/v{v}", await raw.View.CopyToHostAsync(), Crop(depthRef, v, 0), Da3DepthRelRmsGate);
                if (mv.ConfidenceMaps != null)
                {
                    var conf = await mv.ConfidenceMaps[v].View.CopyToHostAsync();
                    Plane($"conf/v{v}", conf, Crop(confRef, v, 0), Da3ConfRelRmsGate);
                    // The gate must see a one-row misregistration.
                    var (relShift, _, _) = Da3Compare(conf, Crop(confRef, v, 1));
                    if (relShift <= Da3ConfRelRmsGate) failures.Add($"conf/v{v}: a 1-row shift PASSED the gate - it cannot discriminate");
                }
                if (mv.Intrinsics == null || mv.ModelIntrinsics == null) continue;

                double modelRel = KRel(mv.ModelIntrinsics[v], RefK(v));
                if (!(modelRel <= KGate)) failures.Add($"ModelIntrinsics/v{v}: {modelRel:E2} of max|ref| vs onnxruntime");

                var want = RefK(v);
                want[2] -= px; want[5] -= py;             // scale 1: only the pad moves
                double gridRel = KRel(mv.Intrinsics[v], want);
                if (!(gridRel <= KGate)) failures.Add($"Intrinsics/v{v} @1:1: {gridRel:E2} vs reference K minus the pad");
                if (KRel(RefK(v), want) <= KGate) failures.Add($"K/v{v}: the pad offset is below the gate - it cannot discriminate");
                report.Append($"K/v{v} model={modelRel:E1} grid={gridRel:E1}; ");
            }
        }

        // ---- source grid and 2x: K follows the grid exactly; the principal point is centred ----
        float[][]? k1 = null;
        foreach (var scale in new[] { 1, 2 })
        {
            int ow = sw * scale, oh = sh * scale;
            using var mv = await pipe.EstimateMultiViewGpuAsync(frames, ws, hs, ow, oh);
            if (mv.Intrinsics == null) { failures.Add($"{ow}x{oh}: no intrinsics"); continue; }
            if (mv.ConfidenceMaps == null || mv.ConfidenceMaps[0].Length != (long)ow * oh)
                failures.Add($"{ow}x{oh}: confidence is not in the depth grid");
            for (int v = 0; v < n; v++)
            {
                var k = mv.Intrinsics[v];
                double cxOff = Math.Abs(k[2] - ow / 2.0) / ow, cyOff = Math.Abs(k[5] - oh / 2.0) / oh;
                report.Append($"{ow}x{oh}/v{v} f={k[0]:F1},{k[4]:F1} c={k[2]:F1},{k[5]:F1}; ");
                if (cxOff > 0.05 || cyOff > 0.05)
                    failures.Add($"{ow}x{oh}/v{v}: principal point ({k[2]:F1}, {k[5]:F1}) is {cxOff:P1}/{cyOff:P1} off the centre");
                if (scale == 2 && k1 != null)
                {
                    var twice = k1[v].ToArray();
                    twice[0] *= 2; twice[1] *= 2; twice[2] *= 2; twice[4] *= 2; twice[5] *= 2;
                    double rel2 = KRel(k, twice);
                    if (!(rel2 <= 1e-5)) failures.Add($"v{v}: K at 2x is not 2x K ({rel2:E2})");
                }
            }
            if (scale == 1)
            {
                k1 = mv.Intrinsics;
                // Negative control: the model's K read as if it were already in this grid - what SpawnScene did.
                var m = mv.ModelIntrinsics![0];
                if (Math.Abs(m[5] - oh / 2.0) / oh <= 0.05 && Math.Abs(m[0] - k1[0][0]) / k1[0][0] <= 1e-3)
                    failures.Add("the unconverted model K passes the centring check - it cannot discriminate");
            }
        }

        // ---- a non-square binding is refused: Letterbox used to read it as its WIDTH, silently ----
        {
            using var wide = InferenceSession.CreateFromOnnx(accelerator, onnxBytes,
                inputShapes: new Dictionary<string, int[]> { ["pixel_values"] = new[] { 1, 1, 3, 602, 448 } },
                externalData: extBytes);
            try
            {
                using var bad = new DepthEstimationPipeline(wide, accelerator);
                failures.Add("a 448x602 binding was accepted (it would letterbox into 448x448)");
            }
            catch (ArgumentException ex) { report.Append($"non-square binding refused: {ex.Message.Split('.')[0]}; "); }
        }

        if (failures.Count > 0)
            throw new Exception($"DAv3 output grid ({failures.Count}): {string.Join("; ", failures)} | {report}");
        return $"PASSED. {report}";
    });
}
