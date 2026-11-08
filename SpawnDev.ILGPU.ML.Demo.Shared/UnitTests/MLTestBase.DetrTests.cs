using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.Tensors;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// The DETR family (D-FINE, RT-DETRv2: Apache-2.0 detectors, AssistZ's YOLOv8 replacement). Three engine defects kept them
/// from running or from matching onnxruntime (Riker, 2026-11-06; fixed 2026-11-08). The small fixtures gate each defect on
/// every backend in milliseconds (tools/gen_detr_ops_reference.py); the HeavyModel tests run the real models against
/// onnxruntime's outputs on the YOLOv8 cat fixture (references/&lt;model&gt;/, from python onnxruntime 1.26 on the CPU).
/// </summary>
public abstract partial class MLTestBase
{
    /// <summary>
    /// <c>Add(x, zeros[1,1,1,1])</c> raises x's rank with its values unchanged. StrengthReduce rewrote it to Identity, so
    /// D-FINE's anchor-grid Concat came out [3] instead of [4] ("Shapes [3] and [4] are not broadcastable").
    /// </summary>
    [TestMethod(Timeout = 180000)]
    public async Task Detr_AddZeroRankPromotion_MatchesOnnxRuntime() => await FusionFixture("add_zero_rank_promote", _ => null);

    /// <summary>
    /// ONNX rank-1 MatMul operands ([..,K] x [K] -> [..], [K] x [K,N] -> [N]) as MatMul and as FusedScaledMatMul, plus a
    /// FusedScaledMatMul whose 2-D B is shared by a batched A. D-FINE's box integral MatMul(softmax [1200,33], [33])
    /// came out [1200,33], and the 33 became a batch two decoder layers later.
    /// </summary>
    [TestMethod(Timeout = 180000)]
    public async Task Detr_MatMulVectorOperands_MatchesOnnxRuntime() => await FusionFixture("matmul_vector",
        ops => ops.Contains("FusedScaledMatMul") ? null : "no FusedScaledMatMul: the fixture no longer exercises the fused form");

    /// <summary>
    /// MatMul -> Add -> Clip(-0.5, 0.5). A fused Clip ran as ReLU (D-FINE's query_pos_head Clip(-10, 10) lost every
    /// negative value: logits off by 4). The Clip must survive as its own node after the fused Linear.
    /// </summary>
    [TestMethod(Timeout = 180000)]
    public async Task Detr_LinearThenClip_MatchesOnnxRuntime() => await FusionFixture("linear_clip",
        ops => !ops.Contains("FusedLinear") ? "MatMul+Add did not fuse into FusedLinear"
             : !ops.Contains("Clip") ? "the Clip was fused away" : null);

    [TestMethod(Timeout = 600000, Category = "HeavyModel")]
    public async Task Detr_DFineS_MatchesOnnxRuntime() => await RunTest(async accelerator =>
        await DetrOrtParity(accelerator, "onnx-community/dfine_s_coco-ONNX", "dfine_s_coco"));

    [TestMethod(Timeout = 600000, Category = "HeavyModel")]
    public async Task Detr_DFineN_MatchesOnnxRuntime() => await RunTest(async accelerator =>
        await DetrOrtParity(accelerator, "onnx-community/dfine_n_coco-ONNX", "dfine_n_coco"));

    [TestMethod(Timeout = 600000, Category = "HeavyModel")]
    public async Task Detr_RtDetrV2R18_MatchesOnnxRuntime() => await RunTest(async accelerator =>
        await DetrOrtParity(accelerator, "onnx-community/rtdetr_v2_r18vd-ONNX", "rtdetr_v2_r18vd"));

    async Task<(int[] Pixels, int W, int H)> StreetRgba(HttpClient http)
    {
        var bin = await http.GetByteArrayAsync("samples/street_rgba.bin");
        int w = BitConverter.ToInt32(bin, 0), h = BitConverter.ToInt32(bin, 4);
        var px = new int[w * h];
        Buffer.BlockCopy(bin, 8, px, 0, w * h * 4);
        return (px, w, h);
    }

    /// <summary>
    /// <see cref="Kernels.ImagePreprocessKernel.ForwardPil"/> reproduces Pillow's <c>Image.resize(BILINEAR)</c> to the byte: a whole-image
    /// 970x646 -> 160x160 shrink (antialiased, 7-tap support) and a crop(box).resize to 96x64 (one axis shrinking, one growing).
    /// References are Pillow's own output (tools/gen_detr_pipeline_reference.py). Plain 4-tap bilinear differs from both.
    /// </summary>
    [TestMethod(Timeout = 180000)]
    public async Task Detr_PilResize_MatchesPillowExactly() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient() ?? throw new UnsupportedTestException("HttpClient not available");
        var (px, w, h) = await StreetRgba(http);
        using var rgba = accelerator.Allocate1D(px);
        using var pre = new Kernels.ImagePreprocessKernel(accelerator);
        foreach (var (name, dw, dh, crop) in new[] { ("whole_160x160", 160, 160, (0, 0, 0, 0)), ("crop_96x64", 96, 64, (37, 21, 150, 110)) })
        {
            var expected = await http.GetByteArrayAsync($"references/pil_resize/{name}.bin");   // HWC RGB
            using var outBuf = accelerator.Allocate1D<float>(3 * dw * dh);
            pre.ForwardPil(rgba.View, outBuf.View, w, h, dw, dh, srcX: crop.Item1, srcY: crop.Item2, regionW: crop.Item3, regionH: crop.Item4);
            var got = await outBuf.CopyToHostAsync<float>(0, 3 * dw * dh);
            int bad = 0, worst = 0, firstBad = -1;
            for (int c = 0; c < 3; c++)
                for (int i = 0; i < dw * dh; i++)
                {
                    int v = (int)MathF.Round(got[c * dw * dh + i] * 255f);
                    int d = Math.Abs(v - expected[i * 3 + c]);
                    if (d != 0) { bad++; worst = Math.Max(worst, d); if (firstBad < 0) firstBad = c * dw * dh + i; }
                }
            Console.WriteLine($"[DETR] PIL resize {name}: {bad} of {3 * dw * dh} bytes differ (worst {worst}) on {BackendName}");
            if (bad != 0)
                throw new Exception($"PIL resize {name}: {bad} of {3 * dw * dh} values differ from Pillow (worst {worst}, first at {firstBad}) on {BackendName}");
        }
    });

    /// <summary>
    /// <see cref="Pipelines.DetrDetectionPipeline"/> on D-FINE-S (AssistZ's model) and the street picture gives transformers' answer:
    /// onnxruntime on Pillow's 640 input + <c>post_process_object_detection</c> at 0.5 (26 people/cars/trucks/a handbag). Then the same
    /// with graph capture, the replayed frames identical to the first. Detections within 0.01 of the threshold may differ (float order).
    /// </summary>
    [TestMethod(Timeout = 600000, Category = "HeavyModel")]
    public async Task Detr_Pipeline_DFineS_Street_MatchesTransformers() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient() ?? throw new UnsupportedTestException("HttpClient not available");
        var (px, w, h) = await StreetRgba(http);
        var onnxBytes = await InferenceSession.DownloadBytesChunkedAsync(http, HuggingFaceClient.GetDownloadUrl("onnx-community/dfine_s_coco-ONNX", "onnx/model.onnx"));
        using var refDoc = System.Text.Json.JsonDocument.Parse(await http.GetStringAsync("references/detr_pipeline/dfine_s_street.json"));
        var expected = refDoc.RootElement.GetProperty("detections").EnumerateArray().Select(d => (
            Score: (float)d.GetProperty("score").GetDouble(), Cls: d.GetProperty("cls").GetInt32(),
            X: (float)d.GetProperty("x").GetDouble(), Y: (float)d.GetProperty("y").GetDouble(),
            W: (float)d.GetProperty("w").GetDouble(), H: (float)d.GetProperty("h").GetDouble())).ToList();

        bool canCapture = accelerator.AcceleratorType is AcceleratorType.Cuda or AcceleratorType.WebGPU;
        foreach (bool capture in canCapture ? new[] { false, true } : new[] { false })
        {
            // The pipeline owns the session (as ObjectDetectionPipeline): disposing it frees the capture, its buffers and the session.
            using var pipeline = new Pipelines.DetrDetectionPipeline(InferenceSession.CreateFromOnnx(accelerator, onnxBytes), accelerator) { EnableGraphCapture = capture };
            using var rgba = accelerator.Allocate1D(px);
            DetectedObjectLike[]? first = null;
            int frames = capture ? 4 : 1;
            for (int f = 0; f < frames; f++)
            {
                var r = await pipeline.DetectAsync(rgba.View, w, h, threshold: 0.5f);
                var got = r.Objects.Select(o => new DetectedObjectLike(o.ClassId, o.Confidence, o.X, o.Y, o.Width, o.Height)).ToArray();
                Console.WriteLine($"[DETR] pipeline capture={capture} frame {f}: {got.Length} detections in {r.InferenceTimeMs:F1} ms, {pipeline.CaptureStatus} on {BackendName}");
                if (f == 0)
                {
                    first = got;
                    // Every reference detection found (same class, score within 0.01, box within 1 px), except one within 0.01 of the bar.
                    foreach (var e in expected)
                    {
                        var m = got.FirstOrDefault(g => g.Cls == e.Cls && MathF.Abs(g.X - e.X) < 1f && MathF.Abs(g.Y - e.Y) < 1f && MathF.Abs(g.W - e.W) < 1f && MathF.Abs(g.H - e.H) < 1f);
                        if (m == null && e.Score > 0.51f)
                            throw new Exception($"capture={capture}: transformers' class {e.Cls} {e.Score:F3} at ({e.X:F1},{e.Y:F1},{e.W:F1}x{e.H:F1}) not found on {BackendName}");
                        if (m != null && MathF.Abs(m.Score - e.Score) > 0.01f)
                            throw new Exception($"capture={capture}: class {e.Cls} at ({e.X:F1},{e.Y:F1}) scored {m.Score:F4}, transformers {e.Score:F4} on {BackendName}");
                    }
                    foreach (var g in got)
                        if (g.Score > 0.51f && !expected.Any(e => e.Cls == g.Cls && MathF.Abs(g.X - e.X) < 1f && MathF.Abs(g.Y - e.Y) < 1f))
                            throw new Exception($"capture={capture}: extra class {g.Cls} {g.Score:F3} at ({g.X:F1},{g.Y:F1}) that transformers does not report on {BackendName}");
                }
                else
                {
                    if (got.Length != first!.Length) throw new Exception($"captured frame {f}: {got.Length} detections vs {first.Length} on {BackendName}");
                    for (int i = 0; i < got.Length; i++)
                        if (got[i].Cls != first[i].Cls || MathF.Abs(got[i].Score - first[i].Score) > 1e-4f || MathF.Abs(got[i].X - first[i].X) > 0.05f || MathF.Abs(got[i].Y - first[i].Y) > 0.05f)
                            throw new Exception($"captured frame {f} detection {i} differs from frame 0 on {BackendName}");
                }
            }
            if (capture && !pipeline.CaptureStatus.StartsWith("live"))
                throw new Exception($"capture requested but not live: {pipeline.CaptureStatus} on {BackendName}");
        }
    });

    /// <summary>
    /// D-FINE-S on WebGPU: warm direct forward wall time and dispatches per op type (GraphExecutor.OpProfile), then the pipeline with graph
    /// capture (median of 10 replayed frames). Prints only; the numbers are what to compare (PMT_CONSOLE_LOG=DETR).
    /// </summary>
    [TestMethod(Timeout = 900000, Category = "HeavyModel")]
    public async Task Bench_DFineS_WebGPU_Profile() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType != AcceleratorType.WebGPU)
            throw new UnsupportedTestException("the browser WebGPU forward is what is measured");
        var http = GetHttpClient() ?? throw new UnsupportedTestException("HttpClient not available");
        var (px, w, h) = await StreetRgba(http);
        var onnxBytes = await InferenceSession.DownloadBytesChunkedAsync(http, HuggingFaceClient.GetDownloadUrl("onnx-community/dfine_s_coco-ONNX", "onnx/model.onnx"));
        using var rgba = accelerator.Allocate1D(px);
        foreach (bool capture in new[] { false, true })
        {
            using var pipeline = new Pipelines.DetrDetectionPipeline(InferenceSession.CreateFromOnnx(accelerator, onnxBytes), accelerator) { EnableGraphCapture = capture };
            var times = new List<double>();
            for (int f = 0; f < 12; f++)
            {
                bool prof = !capture && f == 3;
                if (prof) { Graph.GraphExecutor.OpProfile = new(); SpawnDev.ILGPU.WebGPU.Backend.WebGPUBackend.EnableDispatchProfiling = true; SpawnDev.ILGPU.WebGPU.Backend.WebGPUBackend.ResetDispatchProfiling(); }
                var r = await pipeline.DetectAsync(rgba.View, w, h, threshold: 0.5f);
                if (f >= 2) times.Add(r.InferenceTimeMs);
                if (prof && Graph.GraphExecutor.OpProfile is { } p)
                {
                    Graph.GraphExecutor.OpProfile = null;
                    SpawnDev.ILGPU.WebGPU.Backend.WebGPUBackend.EnableDispatchProfiling = false;
                    Console.WriteLine($"[DETR] profile frame: {r.InferenceTimeMs:F1} ms, {p.Values.Sum(e => e.Dispatches)} dispatches, {p.Values.Sum(e => e.Count)} nodes");
                    foreach (var (op, e) in p.OrderByDescending(kv => kv.Value.Dispatches).Take(15))
                        Console.WriteLine($"[DETR] OP {op,-22} n={e.Count,5} dispatches={e.Dispatches,6} wall={e.WallMs,7:F1} ms");
                }
            }
            times.Sort();
            Console.WriteLine($"[DETR] D-FINE-S WebGPU capture={capture}: median {times[times.Count / 2]:F1} ms (min {times[0]:F1}, max {times[^1]:F1}) over {times.Count} frames, {pipeline.CaptureStatus}");
        }
    });

    /// <summary>
    /// GraphExecutor's capture flags belong to the async flow that set them (Riker, AssistZ: a YOLOv8 capture with the video and audio loops
    /// running put the other loops' forwards in capture mode). A flow started BEFORE the claim (a concurrent loop) sees them false; the claiming
    /// flow sees them true; a second flow trying to claim while the first holds them throws; clearing releases the claim.
    /// </summary>
    [TestMethod(Timeout = 60000)]
    public async Task Capture_Flags_AreScopedToTheirAsyncFlow() => await RunTest(async accelerator =>
    {
        var gate = new TaskCompletionSource();
        async Task<(bool Suppress, bool Slots, bool Attn, string? Claim)> OtherLoop()
        {
            await gate.Task;
            string? claim = null;
            try { Graph.GraphExecutor.SuppressDrains = true; claim = "claimed"; }
            catch (InvalidOperationException ex) { claim = ex.Message; }
            return (Graph.GraphExecutor.SuppressDrains, Graph.GraphExecutor.UseCaptureParamSlots, Kernels.FusedAttentionKernel.UseStableCaptureSlots, claim);
        }
        var other = OtherLoop();   // a "concurrent loop": its flow was captured before this one claims the capture
        try
        {
            Kernels.FusedAttentionKernel.UseStableCaptureSlots = true;
            Graph.GraphExecutor.UseCaptureParamSlots = true;
            Graph.GraphExecutor.SuppressDrains = true;
            gate.SetResult();
            var seen = await other;
            bool mine = Graph.GraphExecutor.SuppressDrains && Graph.GraphExecutor.UseCaptureParamSlots && Kernels.FusedAttentionKernel.UseStableCaptureSlots;
            Console.WriteLine($"[CaptureFlow] owner sees {mine}; other loop sees suppress={seen.Suppress} slots={seen.Slots} attn={seen.Attn}, its claim: {seen.Claim}");
            if (!mine) throw new Exception("the flow that set the capture flags does not see them");
            if (seen.Suppress || seen.Slots || seen.Attn) throw new Exception("a concurrent flow sees the capture flags");
            if (seen.Claim == "claimed") throw new Exception("a second flow claimed the capture while the first held it");
        }
        finally
        {
            Graph.GraphExecutor.SuppressDrains = false;
            Graph.GraphExecutor.UseCaptureParamSlots = false;
            Kernels.FusedAttentionKernel.UseStableCaptureSlots = false;
        }
        // Released: another flow can claim now.
        async Task<bool> NextCapture() { await Task.Yield(); Graph.GraphExecutor.SuppressDrains = true; bool s = Graph.GraphExecutor.SuppressDrains; Graph.GraphExecutor.SuppressDrains = false; return s; }
        if (!await NextCapture()) throw new Exception("after release, a new flow's claim did not take");
    });

    sealed record DetectedObjectLike(int Cls, float Score, float X, float Y, float W, float H);

    /// <summary>
    /// Runs a DETR model on the cat fixture ([1,3,640,640], 0..1, the model's own rescale-only preprocessing) and compares
    /// both outputs with onnxruntime's. MEASURED on CUDA/OpenCL after the fixes: logits within 1.4e-3, boxes within 7.3e-5;
    /// before them D-FINE threw (two shape errors) and then, with the Clip fused as ReLU, was off by 4.05 / 0.32.
    /// </summary>
    async Task DetrOrtParity(Accelerator accelerator, string repo, string refDir)
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");
        var onnxBytes = await InferenceSession.DownloadBytesChunkedAsync(http, HuggingFaceClient.GetDownloadUrl(repo, "onnx/model.onnx"));
        using var session = InferenceSession.CreateFromOnnx(accelerator, onnxBytes,
            inputShapes: new Dictionary<string, int[]> { ["pixel_values"] = new[] { 1, 3, 640, 640 } });
        var inputBytes = await http.GetByteArrayAsync("references/yolov8n/cat_input_nchw.bin");
        var input = new float[inputBytes.Length / 4];
        Buffer.BlockCopy(inputBytes, 0, input, 0, inputBytes.Length);
        using var inputBuf = accelerator.Allocate1D(input);
        var outs = await session.RunAsync(new Dictionary<string, Tensor> { ["pixel_values"] = new Tensor(inputBuf.View, new[] { 1, 3, 640, 640 }) });
        try
        {
            foreach (var (name, tol) in new[] { ("logits", 1e-2f), ("pred_boxes", 1e-3f) })
            {
                var refBytes = await http.GetByteArrayAsync($"references/{refDir}/{name}.bin");
                var expected = new float[refBytes.Length / 4];
                Buffer.BlockCopy(refBytes, 0, expected, 0, refBytes.Length);
                var t = outs[name];
                if (t.ElementCount != expected.Length)
                    throw new Exception($"{refDir} {name}: [{string.Join(",", t.Shape)}] = {t.ElementCount} values, onnxruntime has {expected.Length} on {BackendName}");
                using var host = accelerator.Allocate1D<float>(expected.Length);
                await host.View.CopyFromAsync(t.Data.SubView(0, expected.Length));
                await accelerator.SynchronizeAsync();
                var got = await host.CopyToHostAsync<float>(0, expected.Length);
                float worst = 0; int at = 0;
                for (int i = 0; i < expected.Length; i++) { var d = MathF.Abs(got[i] - expected[i]); if (!(d <= worst)) { worst = d; at = i; } }
                Console.WriteLine($"[DETR] {refDir} {name} max|diff| {worst:E2} at {at} (ours {got[at]:G6}, ORT {expected[at]:G6}) on {BackendName}");
                if (!(worst <= tol))
                    throw new Exception($"{refDir} {name}: max|diff| {worst:E2} at {at} (ours {got[at]:G6}, onnxruntime {expected[at]:G6}) > {tol:E0} on {BackendName}");
            }
        }
        finally { session.ReturnOutputs(outs); }
    }
}
