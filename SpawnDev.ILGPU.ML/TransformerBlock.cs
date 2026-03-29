using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML;

/// <summary>
/// DAv3 ViT Transformer Block: LayerNorm → MHSA → LayerScale → Residual → LayerNorm → MLP → LayerScale → Residual.
///
/// Architecture (per block):
///   x = input + ls1 * Attention(LayerNorm1(input))
///   x = x + ls2 * MLP(LayerNorm2(x))
///
/// Where Attention = QKV projection → split heads → Q×K^T/√d → softmax → ×V → concat → projection
/// And MLP = fc1 → GELU → fc2
///
/// Constants: T=1369 tokens (37×37 patches), C=384 embedding dim, H=6 heads, D=64 head dim
///
/// Future home: SpawnDev.ILGPU.ML
/// </summary>
public class TransformerBlock
{
    public const int C = 384;       // Embedding dimension
    public const int H = 6;         // Number of attention heads
    public const int D = 64;        // Head dimension (C / H)
    public const int MLP_DIM = 1536; // MLP hidden dimension (4 * C)

    private readonly MatMulKernel _matMul;
    private readonly LayerNormKernel _layerNorm;
    private readonly SoftmaxKernel _softmax;
    private readonly ElementWiseKernels _elementWise;
    private readonly AttentionKernels _attention;
    private readonly Kernels.FusedAttentionKernel? _fusedAttention;

    /// <summary>When true, uses single-dispatch fused attention kernel instead of multi-dispatch.</summary>
    public bool UseFusedAttention { get; set; }

    public TransformerBlock(
        MatMulKernel matMul,
        LayerNormKernel layerNorm,
        SoftmaxKernel softmax,
        ElementWiseKernels elementWise,
        AttentionKernels attention,
        Kernels.FusedAttentionKernel? fusedAttention = null)
    {
        _matMul = matMul;
        _layerNorm = layerNorm;
        _softmax = softmax;
        _elementWise = elementWise;
        _attention = attention;
        _fusedAttention = fusedAttention;
    }

    /// <summary>
    /// Weight references for one transformer block.
    /// Resolved from WeightLoader by name.
    /// </summary>
    public class BlockWeights
    {
        // LayerNorm 1
        public required ArrayView1D<float, Stride1D.Dense> Norm1Weight; // [C]
        public required ArrayView1D<float, Stride1D.Dense> Norm1Bias;   // [C]

        // Attention
        public required ArrayView1D<float, Stride1D.Dense> QkvWeight;   // [C, 3*C] = [384, 1152]
        public required ArrayView1D<float, Stride1D.Dense> QkvBias;     // [3*C] = [1152]
        public required ArrayView1D<float, Stride1D.Dense> ProjWeight;  // [C, C] = [384, 384]
        public required ArrayView1D<float, Stride1D.Dense> ProjBias;    // [C]

        // LayerScale 1
        public required ArrayView1D<float, Stride1D.Dense> Ls1Gamma;    // [C]

        // LayerNorm 2
        public required ArrayView1D<float, Stride1D.Dense> Norm2Weight; // [C]
        public required ArrayView1D<float, Stride1D.Dense> Norm2Bias;   // [C]

        // MLP
        public required ArrayView1D<float, Stride1D.Dense> Fc1Weight;   // [C, MLP_DIM] = [384, 1536]
        public required ArrayView1D<float, Stride1D.Dense> Fc1Bias;     // [MLP_DIM]
        public required ArrayView1D<float, Stride1D.Dense> Fc2Weight;   // [MLP_DIM, C] = [1536, 384]
        public required ArrayView1D<float, Stride1D.Dense> Fc2Bias;     // [C]

        // LayerScale 2
        public required ArrayView1D<float, Stride1D.Dense> Ls2Gamma;    // [C]
    }

    /// <summary>
    /// Resolve block weights from the WeightLoader for a given block index.
    /// </summary>
    public static BlockWeights ResolveWeights(WeightLoader loader, int blockIdx)
    {
        string bp = $"backbone.pretrained.blocks.{blockIdx}";
        int[] matMulIds = MatMulIdsPerBlock[blockIdx];

        return new BlockWeights
        {
            Norm1Weight = loader.GetView($"{bp}.norm1.weight"),
            Norm1Bias = loader.GetView($"{bp}.norm1.bias"),
            QkvWeight = loader.GetView($"onnx::MatMul_{matMulIds[0]}"),
            QkvBias = loader.GetView($"{bp}.attn.qkv.bias"),
            ProjWeight = loader.GetView($"onnx::MatMul_{matMulIds[1]}"),
            ProjBias = loader.GetView($"{bp}.attn.proj.bias"),
            Ls1Gamma = loader.GetView($"{bp}.ls1.gamma"),
            Norm2Weight = loader.GetView($"{bp}.norm2.weight"),
            Norm2Bias = loader.GetView($"{bp}.norm2.bias"),
            Fc1Weight = loader.GetView($"onnx::MatMul_{matMulIds[2]}"),
            Fc1Bias = loader.GetView($"{bp}.mlp.fc1.bias"),
            Fc2Weight = loader.GetView($"onnx::MatMul_{matMulIds[3]}"),
            Fc2Bias = loader.GetView($"{bp}.mlp.fc2.bias"),
            Ls2Gamma = loader.GetView($"{bp}.ls2.gamma"),
        };
    }

    /// <summary>
    /// Run one transformer block forward pass.
    /// input: [T, C], output: [T, C] (can be same buffer for in-place).
    /// Temp buffers are pre-allocated by caller for reuse across blocks.
    /// </summary>
    public void Forward(
        ArrayView1D<float, Stride1D.Dense> input,   // [T*C]
        ArrayView1D<float, Stride1D.Dense> output,   // [T*C]
        BlockWeights w,
        int T,  // number of tokens (1369 for 518×518 input)
        TempBuffers tmp)
    {
        int TC = T * C;

        // ── Attention branch ──

        // LayerNorm1: input → tmp.norm
        _layerNorm.Forward(input, tmp.Norm, w.Norm1Weight, w.Norm1Bias, T, C);

        // QKV projection: [T, C] × [C, 3C] → [T, 3C]
        _matMul.MatMul(tmp.Norm, w.QkvWeight, tmp.Qkv, T, C, 3 * C);
        _elementWise.AddBias(tmp.Qkv, w.QkvBias, T * 3 * C, 3 * C);

        // Attention: Q×K^T → softmax → ×V
        // QKV layout: [T, 3C] where 3C = [Q_h0..Q_h5, K_h0..K_h5, V_h0..V_h5]
        // Actually ONNX DAv3 uses [T, 3, H, D] → need to handle head splitting
        // For now: batched matmul for Q×K^T per head, softmax, ×V per head
        RunAttention(tmp.Qkv, tmp.AttnOut, T, tmp);

        // Projection: [T, C] × [C, C] → [T, C]
        _matMul.MatMul(tmp.AttnOut, w.ProjWeight, tmp.ProjOut, T, C, C);
        _elementWise.AddBias(tmp.ProjOut, w.ProjBias, TC, C);

        // LayerScale1 + Residual: output = input + ls1 * proj_out
        LayerScaleMul(tmp.ProjOut, w.Ls1Gamma, tmp.Scaled, TC, C);
        _elementWise.Add(input, tmp.Scaled, output, TC);

        // ── MLP branch ──

        // LayerNorm2: output → tmp.norm
        _layerNorm.Forward(output, tmp.Norm, w.Norm2Weight, w.Norm2Bias, T, C);

        // MLP fc1: [T, C] × [C, MLP_DIM] → [T, MLP_DIM]
        _matMul.MatMul(tmp.Norm, w.Fc1Weight, tmp.MlpHidden, T, C, MLP_DIM);
        _elementWise.AddBias(tmp.MlpHidden, w.Fc1Bias, T * MLP_DIM, MLP_DIM);

        // GELU
        _elementWise.GELUInPlace(tmp.MlpHidden, T * MLP_DIM);

        // MLP fc2: [T, MLP_DIM] × [MLP_DIM, C] → [T, C]
        _matMul.MatMul(tmp.MlpHidden, w.Fc2Weight, tmp.ProjOut, T, MLP_DIM, C);
        _elementWise.AddBias(tmp.ProjOut, w.Fc2Bias, TC, C);

        // LayerScale2 + Residual: output += ls2 * fc2_out (in-place to avoid aliasing)
        LayerScaleMul(tmp.ProjOut, w.Ls2Gamma, tmp.Scaled, TC, C);
        _elementWise.AddInPlace(output, tmp.Scaled, TC);
    }

    /// <summary>
    /// Diagnostic: run block forward with per-step logging. SLOW — syncs after each step.
    /// Logs first 5 values of CLS token (position 0) and first patch (position 1) at each step.
    /// </summary>
    public async Task ForwardDiagnosticAsync(
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output,
        BlockWeights w, int T, TempBuffers tmp,
        Accelerator accelerator, int blockIdx)
    {
        int TC = T * C;
        async Task Log(string step, MemoryBuffer1D<float, Stride1D.Dense>? buf, int offset, int count)
        {
            await accelerator.SynchronizeAsync();
            if (buf == null) return;
            var data = await buf.CopyToHostAsync<float>(offset, Math.Min(count, (int)buf.Length - offset));
            float min = float.MaxValue, max = float.MinValue;
            double sum = 0, sumSq = 0;
            for (int i = 0; i < data.Length; i++) { float v = data[i]; if (v < min) min = v; if (v > max) max = v; sum += v; sumSq += (double)v * v; }
            double std = Math.Sqrt(sumSq / data.Length - (sum / data.Length) * (sum / data.Length));
            var first5 = string.Join(", ", data.Take(5).Select(v => v.ToString("F4")));
            if (InferenceSession.VerboseLogging) Console.WriteLine($"[Block{blockIdx}] {step}: min={min:F4} max={max:F4} std={std:F4} first5=[{first5}]");
        }

        // ── Attention branch ──
        _layerNorm.Forward(input, tmp.Norm, w.Norm1Weight, w.Norm1Bias, T, C);
        await Log("LN1", tmp.NormBuf, 0, TC);

        _matMul.MatMul(tmp.Norm, w.QkvWeight, tmp.Qkv, T, C, 3 * C);
        _elementWise.AddBias(tmp.Qkv, w.QkvBias, T * 3 * C, 3 * C);
        await Log("QKV+bias", tmp.QkvBuf, 0, T * 3 * C);

        RunAttention(tmp.Qkv, tmp.AttnOut, T, tmp);
        await Log("Attn", tmp.AttnOutBuf, 0, TC);

        _matMul.MatMul(tmp.AttnOut, w.ProjWeight, tmp.ProjOut, T, C, C);
        _elementWise.AddBias(tmp.ProjOut, w.ProjBias, TC, C);
        await Log("Proj+bias", tmp.ProjOutBuf, 0, TC);

        LayerScaleMul(tmp.ProjOut, w.Ls1Gamma, tmp.Scaled, TC, C);
        await Log("LS1*Proj", tmp.ScaledBuf, 0, TC);

        _elementWise.Add(input, tmp.Scaled, output, TC);
        await Log("Resid1", null, 0, 0); // output is a view, can't easily read — skip for now

        // ── MLP branch ──
        _layerNorm.Forward(output, tmp.Norm, w.Norm2Weight, w.Norm2Bias, T, C);
        await Log("LN2", tmp.NormBuf, 0, TC);

        _matMul.MatMul(tmp.Norm, w.Fc1Weight, tmp.MlpHidden, T, C, MLP_DIM);
        _elementWise.AddBias(tmp.MlpHidden, w.Fc1Bias, T * MLP_DIM, MLP_DIM);
        await Log("FC1+bias", tmp.MlpHiddenBuf, 0, T * MLP_DIM);

        _elementWise.GELUInPlace(tmp.MlpHidden, T * MLP_DIM);
        await Log("GELU", tmp.MlpHiddenBuf, 0, T * MLP_DIM);

        _matMul.MatMul(tmp.MlpHidden, w.Fc2Weight, tmp.ProjOut, T, MLP_DIM, C);
        _elementWise.AddBias(tmp.ProjOut, w.Fc2Bias, TC, C);
        await Log("FC2+bias", tmp.ProjOutBuf, 0, TC);

        LayerScaleMul(tmp.ProjOut, w.Ls2Gamma, tmp.Scaled, TC, C);
        await Log("LS2*FC2", tmp.ScaledBuf, 0, TC);

        _elementWise.AddInPlace(output, tmp.Scaled, TC);
        // Can't easily log output view here
        if (InferenceSession.VerboseLogging) Console.WriteLine($"[Block{blockIdx}] Done");
    }

    /// <summary>
    /// LayerScale: out[i] = input[i] * gamma[i % C] — broadcast gamma across rows.
    /// </summary>
    private void LayerScaleMul(
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> gamma,
        ArrayView1D<float, Stride1D.Dense> output,
        int totalElements, int channelDim)
    {
        _elementWise.BroadcastMul(input, gamma, output, totalElements, channelDim);
    }

    /// <summary>
    /// Multi-head self-attention:
    ///   1. Split QKV [T, 3C] → Q[H,T,D], K[H,T,D], V[H,T,D]
    ///   2. Batched MatMul: Q × K^T → scores[H, T, T]
    ///   3. Scale by 1/√D (= 1/8 for D=64)
    ///   4. Softmax per row
    ///   5. Batched MatMul: scores × V → attnOut[H, T, D]
    ///   6. Merge heads [H,T,D] → [T, C]
    /// </summary>
    private void RunAttention(
        ArrayView1D<float, Stride1D.Dense> qkv,      // [T, 3*C]
        ArrayView1D<float, Stride1D.Dense> output,    // [T, C]
        int T,
        TempBuffers tmp)
    {
        // Step 1: Split QKV into separate Q, K, V with heads as batch dim
        // Q, K, V each [H, T, D] = [6, T, 64] contiguous
        var Q = tmp.Q;
        var K = tmp.K;
        var V = tmp.V;
        _attention.SplitHeads(qkv, Q, K, V, T);

        if (UseFusedAttention && _fusedAttention != null)
        {
            // Fused path: single dispatch replaces steps 2-5.
            // softmax(Q @ K^T / sqrt(D)) @ V computed in one kernel per thread.
            // No intermediate scores materialization, no multi-dispatch overhead.
            _fusedAttention.Forward(Q, K, V, tmp.AttnHeadOut, H, T, T, D);
        }
        else
        {
            // Unfused path: 4 dispatches (transpose, matmul, scale+softmax, matmul)
            TransposeLastTwo(K, tmp.KTransposed, H, T, D);
            _matMul.BatchedMatMul(Q, tmp.KTransposed, tmp.AttnScores, H, T, D, T);

            float scale = 1f / MathF.Sqrt(D);
            _elementWise.ScaleInPlace(tmp.AttnScores, H * T * T, scale);
            _softmax.Forward(tmp.AttnScores, H * T, T);

            _matMul.BatchedMatMul(tmp.AttnScores, V, tmp.AttnHeadOut, H, T, T, D);
        }

        // Step 6: Merge heads [H, T, D] → [T, C]
        _attention.MergeHeads(tmp.AttnHeadOut, output, T);
    }

    /// <summary>
    /// Transpose the last two dimensions: [batch, rows, cols] → [batch, cols, rows].
    /// Auto-grouped, one thread per element.
    /// </summary>
    private void TransposeLastTwo(
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output,
        int batch, int rows, int cols)
    {
        // Total elements = batch * rows * cols
        // For each element at [b, r, c]: output[b * cols * rows + c * rows + r] = input[b * rows * cols + r * cols + c]
        _elementWise.TransposeLastTwo(input, output, batch, rows, cols);
    }

    /// <summary>
    /// Pre-allocated temporary buffers for one transformer block.
    /// Reused across all 12 blocks to minimize GPU memory allocation.
    /// </summary>
    public class TempBuffers : IDisposable
    {
        public MemoryBuffer1D<float, Stride1D.Dense> NormBuf { get; }
        public MemoryBuffer1D<float, Stride1D.Dense> QkvBuf { get; }
        public MemoryBuffer1D<float, Stride1D.Dense> AttnOutBuf { get; }
        public MemoryBuffer1D<float, Stride1D.Dense> ProjOutBuf { get; }
        public MemoryBuffer1D<float, Stride1D.Dense> ScaledBuf { get; }
        public MemoryBuffer1D<float, Stride1D.Dense> MlpHiddenBuf { get; }
        public MemoryBuffer1D<float, Stride1D.Dense> AttnScoresBuf { get; }
        // Attention head buffers
        public MemoryBuffer1D<float, Stride1D.Dense> QBuf { get; }
        public MemoryBuffer1D<float, Stride1D.Dense> KBuf { get; }
        public MemoryBuffer1D<float, Stride1D.Dense> VBuf { get; }
        public MemoryBuffer1D<float, Stride1D.Dense> KTransposedBuf { get; }
        public MemoryBuffer1D<float, Stride1D.Dense> AttnHeadOutBuf { get; }

        public ArrayView1D<float, Stride1D.Dense> Norm => NormBuf.View;
        public ArrayView1D<float, Stride1D.Dense> Qkv => QkvBuf.View;
        public ArrayView1D<float, Stride1D.Dense> AttnOut => AttnOutBuf.View;
        public ArrayView1D<float, Stride1D.Dense> ProjOut => ProjOutBuf.View;
        public ArrayView1D<float, Stride1D.Dense> Scaled => ScaledBuf.View;
        public ArrayView1D<float, Stride1D.Dense> MlpHidden => MlpHiddenBuf.View;
        public ArrayView1D<float, Stride1D.Dense> AttnScores => AttnScoresBuf.View;
        public ArrayView1D<float, Stride1D.Dense> Q => QBuf.View;
        public ArrayView1D<float, Stride1D.Dense> K => KBuf.View;
        public ArrayView1D<float, Stride1D.Dense> V => VBuf.View;
        public ArrayView1D<float, Stride1D.Dense> KTransposed => KTransposedBuf.View;
        public ArrayView1D<float, Stride1D.Dense> AttnHeadOut => AttnHeadOutBuf.View;

        public TempBuffers(Accelerator accelerator, int T)
        {
            NormBuf = accelerator.Allocate1D<float>(T * C);
            QkvBuf = accelerator.Allocate1D<float>(T * 3 * C);
            AttnOutBuf = accelerator.Allocate1D<float>(T * C);
            ProjOutBuf = accelerator.Allocate1D<float>(T * C);
            ScaledBuf = accelerator.Allocate1D<float>(T * C);
            MlpHiddenBuf = accelerator.Allocate1D<float>(T * MLP_DIM);
            AttnScoresBuf = accelerator.Allocate1D<float>(H * T * T);
            QBuf = accelerator.Allocate1D<float>(H * T * D);
            KBuf = accelerator.Allocate1D<float>(H * T * D);
            VBuf = accelerator.Allocate1D<float>(H * T * D);
            KTransposedBuf = accelerator.Allocate1D<float>(H * D * T); // [H, D, T]
            AttnHeadOutBuf = accelerator.Allocate1D<float>(H * T * D);
        }

        public void Dispose()
        {
            NormBuf.Dispose(); QkvBuf.Dispose(); AttnOutBuf.Dispose();
            ProjOutBuf.Dispose(); ScaledBuf.Dispose(); MlpHiddenBuf.Dispose();
            AttnScoresBuf.Dispose(); QBuf.Dispose(); KBuf.Dispose();
            VBuf.Dispose(); KTransposedBuf.Dispose(); AttnHeadOutBuf.Dispose();
        }
    }

    // ── Weight ID mapping (extracted from ONNX graph) ──

    /// <summary>
    /// ONNX MatMul weight tensor IDs for each block: [qkv, proj, fc1, fc2].
    /// </summary>
    public static readonly int[][] MatMulIdsPerBlock = new[]
    {
        new[] { 6357, 6360, 6361, 6362 }, // Block 0
        new[] { 6363, 6366, 6367, 6368 }, // Block 1
        new[] { 6369, 6372, 6373, 6374 }, // Block 2
        new[] { 6375, 6378, 6379, 6380 }, // Block 3
        new[] { 6394, 6425, 6426, 6427 }, // Block 4
        new[] { 6428, 6459, 6460, 6461 }, // Block 5
        new[] { 6462, 6493, 6494, 6495 }, // Block 6
        new[] { 6496, 6527, 6528, 6529 }, // Block 7
        new[] { 6530, 6561, 6562, 6563 }, // Block 8
        new[] { 6564, 6595, 6596, 6597 }, // Block 9
        new[] { 6598, 6629, 6630, 6631 }, // Block 10
        new[] { 6632, 6663, 6664, 6665 }, // Block 11
    };
}
