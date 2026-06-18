using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.GGUF;
using SpawnDev.ILGPU.ML.Tensors;

namespace SpawnDev.ILGPU.ML.Operators;

/// <summary>
/// Mixture-of-Experts feed-forward (gpt-oss / OpenAI-MoE shape). One fused node instead of decomposing the
/// data-dependent top-k routing into static graph nodes. EXACT port of llama.cpp build_moe_ffn for the
/// gpt-oss config: SOFTMAX_WEIGHT gating (select top-k by raw router logits, softmax over the selected k),
/// norm_w = false, per-expert gate/up/down with biases, SwiGLU-OAI activation (alpha 1.702, limit 7.0),
/// weighted combine.
///
/// Inputs (ggml gpt-oss order):
///   0 x            [T, n_embd]                      fp32 activation
///   1 ffn_gate_inp [n_embd, n_expert]   (router)    fp32 or quantized
///   2 ffn_gate_inp.bias [n_expert]                  fp32
///   3 ffn_gate_exps [n_embd, n_ff, n_expert]        fp32 or quantized (MXFP4)
///   4 ffn_gate_exps.bias [n_ff, n_expert]           fp32
///   5 ffn_up_exps   [n_embd, n_ff, n_expert]        fp32 or quantized
///   6 ffn_up_exps.bias [n_ff, n_expert]             fp32
///   7 ffn_down_exps [n_ff, n_embd, n_expert]        fp32 or quantized
///   8 ffn_down_exps.bias [n_embd, n_expert]         fp32
/// Attributes: n_expert, n_expert_used (top-k), n_ff (expert hidden), alpha=1.702, limit=7.0, w_scale=1.
/// Output 0: [T, n_embd].
///
/// WEIGHT ORIENTATION: each expert weight is stored ggml-contiguous [.., n_expert] with the per-expert slice
/// laid out [N rows][K] (ne1 rows of ne0) - the SAME transposed-read contract FusedDequantMatMul uses, so a
/// quantized expert slice feeds straight in (reads [N][K], out[m,n]=Σ_k x[m,k]·W[n,k]). The router uses the
/// identical [N=n_expert][K=n_embd] layout.
/// </summary>
public class MoEOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "MoE";

    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
    {
        // out has x's shape (last dim n_embd preserved).
        return new[] { (int[])inputs[0].Clone() };
    }

    private readonly record struct Dims(int T, int NEmbd, int NExpert, int TopK, int NFf, float Alpha, float Limit, float WScale);

    private Dims ReadDims(OnnxOpContext ctx)
    {
        var x = ctx.Inputs[0];
        int nEmbd = x.Shape[^1];
        int T = x.ElementCount / nEmbd;
        int nExpert = ctx.GetInt("n_expert");
        int topK = ctx.GetInt("n_expert_used");
        int nFf = ctx.GetInt("n_ff");
        if (nExpert <= 0 || topK <= 0 || nFf <= 0)
            throw new InvalidOperationException($"MoE: bad attrs n_expert={nExpert} n_expert_used={topK} n_ff={nFf}.");
        if (topK > nExpert) topK = nExpert;
        return new Dims(T, nEmbd, nExpert, topK, nFf,
            ctx.GetFloat("alpha", 1.702f), ctx.GetFloat("limit", 7.0f), ctx.GetFloat("w_scale", 1.0f));
    }

    /// <summary>Router matmul + bias into a pool tensor [T, n_expert] (caller returns it).</summary>
    private Tensor BuildLogits(OnnxOpContext ctx, in Dims d)
    {
        var logits = ctx.Pool.Rent(new[] { d.T, d.NExpert }, "_moe_logits");
        // router weight is [n_expert][n_embd] (same transposed-read layout as an expert), so it's one linear.
        ExpertLinear(ctx, ctx.Inputs[0].Data, 1, d.NEmbd, d.NExpert, d.T, d.NEmbd, d.NExpert, logits.Data);
        if (ctx.Inputs.Length > 2 && ctx.Inputs[2] != null)
            reg.ElementWise.AddBias(logits.Data, ctx.Inputs[2].Data, d.T * d.NExpert, d.NExpert);
        return logits;
    }

    // Synchronous path (CPU/CUDA/OpenCL): sync GPU->CPU readback of the tiny router logits.
    public void Execute(OnnxOpContext ctx)
    {
        var d = ReadDims(ctx);
        reg.ElementWise.Fill(ctx.Outputs[0].Data, d.T * d.NEmbd, 0f);
        var logits = BuildLogits(ctx, d);
        float[] host;
        using (var rb = reg.Accelerator.Allocate1D<float>(d.T * d.NExpert))
        {
            rb.View.CopyFrom(logits.Data.SubView(0, d.T * d.NExpert));
            host = rb.GetAsArray1D();
        }
        ctx.Pool.Return(logits);
        RunExperts(ctx, d, host);
    }

    // Async path (GraphExecutor.RunAsync, browser-safe): the router-logits readback uses CopyToHostAsync,
    // which works on WebGPU/WebGL/Wasm (sync CopyToCPU throws there). Everything else is identical.
    public async Task ExecuteAsync(OnnxOpContext ctx)
    {
        var d = ReadDims(ctx);
        reg.ElementWise.Fill(ctx.Outputs[0].Data, d.T * d.NEmbd, 0f);
        var logits = BuildLogits(ctx, d);
        float[] host;
        using (var rb = reg.Accelerator.Allocate1D<float>(d.T * d.NExpert))
        {
            rb.View.CopyFrom(logits.Data.SubView(0, d.T * d.NExpert));
            host = await rb.CopyToHostAsync<float>(0, d.T * d.NExpert).ConfigureAwait(false);
        }
        ctx.Pool.Return(logits);
        RunExperts(ctx, d, host);
    }

    /// <summary>Host-side top-k + softmax-over-selected per token; GPU per-expert FFN; weighted combine.
    /// No GPU readback here (routing weights already on host) - all expert work stays on the GPU.</summary>
    private void RunExperts(OnnxOpContext ctx, in Dims d, float[] logitsHost)
    {
        var pool = ctx.Pool;
        var x = ctx.Inputs[0];
        var outData = ctx.Outputs[0].Data;

        var gateBuf = pool.Rent(new[] { 1, d.NFf }, "_moe_gate");
        var upBuf = pool.Rent(new[] { 1, d.NFf }, "_moe_up");
        var actBuf = pool.Rent(new[] { 1, d.NFf }, "_moe_act");
        var downBuf = pool.Rent(new[] { 1, d.NEmbd }, "_moe_down");
        var sel = new int[d.TopK];
        var selW = new float[d.TopK];
        try
        {
            for (int t = 0; t < d.T; t++)
            {
                TopKSoftmax(logitsHost, t * d.NExpert, d.NExpert, d.TopK, sel, selW);
                var xt = x.Data.SubView((long)t * d.NEmbd, d.NEmbd);
                var outT = outData.SubView((long)t * d.NEmbd, d.NEmbd);

                for (int s = 0; s < d.TopK; s++)
                {
                    int e = sel[s];
                    float w = selW[s] * d.WScale;

                    ExpertLinear(ctx, xt, 3, d.NEmbd, d.NFf, 1, d.NEmbd, d.NFf, gateBuf.Data, e);
                    AddExpertBias(ctx, 4, e, d.NFf, gateBuf.Data);
                    ExpertLinear(ctx, xt, 5, d.NEmbd, d.NFf, 1, d.NEmbd, d.NFf, upBuf.Data, e);
                    AddExpertBias(ctx, 6, e, d.NFf, upBuf.Data);

                    reg.MoE.SwiGluOai(gateBuf.Data, upBuf.Data, actBuf.Data, d.NFf, d.Alpha, d.Limit);

                    ExpertLinear(ctx, actBuf.Data, 7, d.NFf, d.NEmbd, 1, d.NFf, d.NEmbd, downBuf.Data, e);
                    AddExpertBias(ctx, 8, e, d.NEmbd, downBuf.Data);

                    reg.ElementWise.ScaleInPlace(downBuf.Data, d.NEmbd, w);
                    reg.ElementWise.AddInPlace(outT, downBuf.Data, d.NEmbd);
                }
            }
        }
        finally
        {
            pool.Return(gateBuf); pool.Return(upBuf); pool.Return(actBuf); pool.Return(downBuf);
        }
    }

    /// <summary>top-k of logits[off..off+n] by value (descending), then softmax over the selected k.</summary>
    private static void TopKSoftmax(float[] logits, int off, int n, int k, int[] sel, float[] w)
    {
        // simple partial selection (k is small: gpt-oss top-4 of 32)
        var used = new bool[n];
        for (int i = 0; i < k; i++)
        {
            int best = -1; float bestV = float.NegativeInfinity;
            for (int j = 0; j < n; j++)
                if (!used[j] && logits[off + j] > bestV) { bestV = logits[off + j]; best = j; }
            used[best] = true; sel[i] = best; w[i] = bestV;
        }
        // softmax over selected logits
        float m = float.NegativeInfinity;
        for (int i = 0; i < k; i++) m = MathF.Max(m, w[i]);
        float sum = 0f;
        for (int i = 0; i < k; i++) { w[i] = MathF.Exp(w[i] - m); sum += w[i]; }
        float inv = 1f / sum;
        for (int i = 0; i < k; i++) w[i] *= inv;
    }

    /// <summary>
    /// out[M,N] = A[M,K] · W^T where W is the expert <paramref name="inputIdx"/>'s slice stored [N rows][K]
    /// (ggml ne1 rows of ne0). expertIdx selects the slice within a 3D expert tensor (-1 = whole tensor, for
    /// the 2D router). Quantized experts go through FusedDequantMatMul (reads [N][K] natively); fp32 experts
    /// transpose the slice to [K,N] and use the plain matmul.
    /// </summary>
    private void ExpertLinear(OnnxOpContext ctx, ArrayView1D<float, Stride1D.Dense> a,
        int inputIdx, int K, int N, int M, int kCheck, int nCheck,
        ArrayView1D<float, Stride1D.Dense> outView, int expertIdx = -1)
    {
        string? name = inputIdx < ctx.InputNames.Length ? ctx.InputNames[inputIdx] : null;
        long perExpertElems = (long)K * N;

        if (name != null && ctx.QuantizedWeights != null && ctx.QuantizedWeights.TryGetValue(name, out var qAll))
        {
            var qType = ctx.Registry?.QuantizedWeightTypes != null
                && ctx.Registry.QuantizedWeightTypes.TryGetValue(name, out var qt) ? qt
                : throw new InvalidOperationException($"MoE: quantized expert '{name}' has no GGMLType registered.");
            long bytesPerExpert = GGMLTypes.TypeSize(qType, perExpertElems);
            var qView = expertIdx < 0 ? qAll : qAll.SubView(expertIdx * bytesPerExpert, bytesPerExpert);
            reg.FusedDequant.Forward(a, qView, outView, M, K, N, qType);
            return;
        }

        // fp32 expert weight slice: [N rows][K] row-major -> transpose to [K, N] for the plain matmul.
        var wT = ctx.Inputs[inputIdx];
        var slice = expertIdx < 0 ? wT.Data : wT.Data.SubView(expertIdx * perExpertElems, perExpertElems);
        var bT = ctx.Pool.Rent(new[] { K, N }, "_moe_wT");
        reg.Transpose.Transpose(slice, bT.Data, new[] { N, K }, new[] { 1, 0 });
        reg.MatMul.MatMul(a, bT.Data, outView, M, K, N);
        ctx.Pool.Return(bT);
    }

    /// <summary>data[1,len] += expert <paramref name="expertIdx"/>'s bias slice (input <paramref name="inputIdx"/>,
    /// stored [len, n_expert] -> slice of length len at expertIdx*len). Skips when the bias input is absent.</summary>
    private void AddExpertBias(OnnxOpContext ctx, int inputIdx, int expertIdx, int len,
        ArrayView1D<float, Stride1D.Dense> data)
    {
        if (inputIdx >= ctx.Inputs.Length || ctx.Inputs[inputIdx] == null) return;
        var bias = ctx.Inputs[inputIdx].Data.SubView((long)expertIdx * len, len);
        reg.ElementWise.AddBias(data, bias, len, len);
    }
}
