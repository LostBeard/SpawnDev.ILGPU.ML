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
