using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Training;

/// <summary>
/// Backward (gradient) kernels for the graph autograd (Plans/training-autograd-lora-2026-11-04.md, M1.3). Each mirrors a
/// forward kernel's exact math (the forward it differentiates is named on each method). Correctness first: the attention
/// backward is the naive O(seq^2 * head_dim) recompute, one thread per output element, and is replaced by a flash-style
/// kernel at M3 (llama3.1:8b). Every method only DISPATCHES; scratch comes from the caller.
/// </summary>
public sealed partial class BackwardKernels : IDisposable
{
    private readonly Accelerator _acc;

    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, float>? _rmsRstd;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, int, int>? _rmsMean;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int>? _rmsNorm;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int>? _ceLse;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, int, int>? _ceRowLoss;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, int, int, int, int>? _ceReduce;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int>? _ceLogProb;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>? _swiglu;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        int, int, float, int, int, int, int>? _rope;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int>? _ceCount;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, int, int>? _ceGrad;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, AttnDims>? _attnStats;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, AttnDims>? _attnDq;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, AttnDims>? _attnDkDv;

    private MemoryBuffer1D<float, Stride1D.Dense>? _ones;   // freqFactors stand-in when a RoPE has none (the kernel reads [0] = 1)

    public BackwardKernels(Accelerator accelerator) => _acc = accelerator;

    // ── RMSNorm: y = x * rstd * w, rstd = 1/sqrt(mean(x^2) + eps)  (NormalizationKernels.RMSNorm) ──
    /// <summary>
    /// dx = rstd * (w*dy - xhat * mean(xhat * w * dy)) per row of C, xhat = x * rstd. Two passes - per-row statistics into
    /// <paramref name="scratch"/> (2 * rows floats: rstd, then the mean term), then one thread per element - so every thread
    /// stores ONE element (WebGL's transform feedback cannot capture a thread writing a whole row) and the element pass is parallel.
    /// </summary>
    public void RMSNorm(ArrayView1D<float, Stride1D.Dense> x, ArrayView1D<float, Stride1D.Dense> w, ArrayView1D<float, Stride1D.Dense> dy,
        ArrayView1D<float, Stride1D.Dense> dx, ArrayView1D<float, Stride1D.Dense> scratch, int rows, int C, float eps)
    {
        _rmsRstd ??= _acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, float>(RMSRstdImpl);
        _rmsMean ??= _acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int>(RMSMeanImpl);
        _rmsNorm ??= _acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int>(RMSNormImpl);
        // ONE binding for both statistics (stats[row] = rstd, stats[rows + row] = the mean term): WebGPU forbids two views of
        // one buffer in a dispatch (MEASURED: "Storage buffer aliasing detected"), and sub-view offsets need 256-byte alignment.
        _rmsRstd(rows, x, scratch, C, eps);
        _rmsMean(rows, x, w, dy, scratch, C, rows);
        _rmsNorm(rows * C, x, w, dy, scratch, dx, C, rows);
    }

    private static void RMSRstdImpl(Index1D row, ArrayView1D<float, Stride1D.Dense> x, ArrayView1D<float, Stride1D.Dense> rstd, int C, float eps)
    {
        int b = row * C;
        float ss = 0f;
        for (int c = 0; c < C; c++) { float v = x[b + c]; ss += v * v; }
        rstd[row] = 1f / MathF.Sqrt(ss / C + eps);
    }

    private static void RMSMeanImpl(Index1D row, ArrayView1D<float, Stride1D.Dense> x, ArrayView1D<float, Stride1D.Dense> w,
        ArrayView1D<float, Stride1D.Dense> dy, ArrayView1D<float, Stride1D.Dense> stats, int C, int rows)
    {
        int b = row * C;
        float r = stats[row];
        float dot = 0f;   // sum(xhat * w * dy)
        for (int c = 0; c < C; c++) dot += x[b + c] * r * w[c] * dy[b + c];
        stats[rows + row] = dot / C;
    }

    private static void RMSNormImpl(Index1D i, ArrayView1D<float, Stride1D.Dense> x, ArrayView1D<float, Stride1D.Dense> w,
        ArrayView1D<float, Stride1D.Dense> dy, ArrayView1D<float, Stride1D.Dense> stats, ArrayView1D<float, Stride1D.Dense> dx, int C, int rows)
    {
        int row = i / C, c = i - row * C;
        float r = stats[row];
        dx[i] = r * (w[c] * dy[i] - x[i] * r * stats[rows + row]);
    }

    // ── SwiGLU: y = silu(g) * u  (ActivationKernels.SwiGLU) ──
    /// <summary>dg = dy * u * silu'(g), silu'(g) = s(g) * (1 + g * (1 - s(g))); du = dy * silu(g).</summary>
    public void SwiGLU(ArrayView1D<float, Stride1D.Dense> g, ArrayView1D<float, Stride1D.Dense> u, ArrayView1D<float, Stride1D.Dense> dy,
        ArrayView1D<float, Stride1D.Dense> dg, ArrayView1D<float, Stride1D.Dense> du, int n)
    {
        _swiglu ??= _acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(SwiGLUImpl);
        _swiglu(n, g, u, dy, dg, du);
    }

    private static void SwiGLUImpl(Index1D i, ArrayView1D<float, Stride1D.Dense> g, ArrayView1D<float, Stride1D.Dense> u,
        ArrayView1D<float, Stride1D.Dense> dy, ArrayView1D<float, Stride1D.Dense> dg, ArrayView1D<float, Stride1D.Dense> du)
    {
        float gv = g[i];
        float s = 1f / (1f + MathF.Exp(-gv));
        float silu = gv * s;
        dg[i] = dy[i] * u[i] * s * (1f + gv * (1f - s));
        du[i] = dy[i] * silu;
    }

    // ── RoPE (RoPEKernel): out = R(theta) x on each pair; backward = R(-theta) dy ──
    /// <summary>The adjoint rotation of <c>RoPEKernel.Apply</c> for the same attributes (no frequency factors yet).</summary>
    public void RoPE(ArrayView1D<float, Stride1D.Dense> dy, ArrayView1D<float, Stride1D.Dense> dx, ArrayView1D<float, Stride1D.Dense>? freqFactors,
        int rows, int headDim, int startPos, float ropeBase, int rotaryDim, bool interleaved, int rowsPerPosition)
    {
        bool hasFactors = freqFactors != null;
        _ones ??= _acc.Allocate1D(new[] { 1f });
        _rope ??= _acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, int, int, float, int, int, int, int>(RoPEImpl);
        _rope(rows * headDim, dy, dx, freqFactors ?? _ones.View, headDim, startPos, ropeBase, rotaryDim, interleaved ? 1 : 0, rowsPerPosition, hasFactors ? 1 : 0);
    }

    private static void RoPEImpl(Index1D idx, ArrayView1D<float, Stride1D.Dense> dy, ArrayView1D<float, Stride1D.Dense> dx,
        ArrayView1D<float, Stride1D.Dense> freqFactors, int D, int startPos, float ropeBase, int rotDim, int interleaved, int rowsPerPos, int hasFactors)
    {
        int k = idx % D;
        int pos = (idx / D) / rowsPerPos + startPos;
        int rowStart = (idx / D) * D;
        if (k >= rotDim) { dx[idx] = dy[idx]; return; }
        int half = rotDim / 2;
        int laneNeoX = k >= half ? k - half : k;
        int laneJ = k >> 1;
        int i = interleaved == 1 ? laneJ : laneNeoX;
        int x0Idx = interleaved == 1 ? 2 * laneJ : laneNeoX;
        int x1Idx = interleaved == 1 ? 2 * laneJ + 1 : laneNeoX + half;
        bool second = interleaved == 1 ? (k & 1) == 1 : k >= half;
        float invFreq = 1f / MathF.Pow(ropeBase, 2f * i / (float)rotDim);
        float ff = freqFactors[i * hasFactors];
        float theta = pos * invFreq / ff;
        float c = MathF.Cos(theta), s = MathF.Sin(theta);
        float y0 = dy[rowStart + x0Idx], y1 = dy[rowStart + x1Idx];
        // R(theta)^T = [[c, s], [-s, c]]
        dx[idx] = second ? (-y0 * s + y1 * c) : (y0 * c + y1 * s);
    }

    // ── Softmax cross-entropy (ONNX SoftmaxCrossEntropyLoss, reduction=mean, ignore_index) ──
    /// <summary>
    /// dlogits[r, c] = (softmax(logits[r])[c] - onehot(label[r])[c]) / count for rows whose label is not
    /// <paramref name="ignoreIndex"/> (0 for ignored rows); count = the number of such rows. <paramref name="scratch"/> holds
    /// 1 + rows floats: the count, then each row's log-sum-exp (computed on the device, so no readback). One store per thread in
    /// every pass (WebGL). Labels are float-held class indices, as the engine stores ids.
    /// </summary>
    public void SoftmaxCrossEntropy(ArrayView1D<float, Stride1D.Dense> logits, ArrayView1D<float, Stride1D.Dense> labels,
        ArrayView1D<float, Stride1D.Dense> dlogits, ArrayView1D<float, Stride1D.Dense> scratch, int rows, int C, int ignoreIndex)
    {
        _ceCount ??= _acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int>(CECountImpl);
        _ceLse ??= _acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int>(CELseImpl);
        _ceGrad ??= _acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int>(CEGradImpl);
        // ONE binding for the count (stats[0]) and each row's log-sum-exp (stats[1 + r]); see RMSNorm for why.
        _ceCount(1, labels, scratch, rows, ignoreIndex);
        _ceLse(rows, logits, scratch, C);
        _ceGrad(rows * C, logits, labels, scratch, dlogits, C, ignoreIndex);
    }

    /// <summary>
    /// The FORWARD loss of ONNX SoftmaxCrossEntropyLoss on [rows, C] scores, on the device: per-row log-sum-exp into
    /// <paramref name="scratch"/> (1 + rows floats, same layout as <see cref="SoftmaxCrossEntropy"/>), then
    /// <paramref name="loss"/>: reduction 0 = none (one value per row, 0 for ignored rows), 1 = mean over the rows not ignored,
    /// 2 = sum. Optional <paramref name="logProb"/> [rows, C] = logits - lse. Pass <c>int.MinValue</c> to ignore nothing.
    /// </summary>
    public void SoftmaxCrossEntropyLoss(ArrayView1D<float, Stride1D.Dense> logits, ArrayView1D<float, Stride1D.Dense> labels,
        ArrayView1D<float, Stride1D.Dense> loss, ArrayView1D<float, Stride1D.Dense>? logProb, ArrayView1D<float, Stride1D.Dense> scratch,
        int rows, int C, int ignoreIndex, int reduction)
    {
        _ceCount ??= _acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int>(CECountImpl);
        _ceLse ??= _acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int>(CELseImpl);
        _ceRowLoss ??= _acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int>(CERowLossImpl);
        _ceReduce ??= _acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int, int, int>(CEReduceImpl);
        _ceLogProb ??= _acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, int>(CELogProbImpl);
        _ceCount(1, labels, scratch, rows, ignoreIndex);
        _ceLse(rows, logits, scratch, C);
        if (reduction == 0) _ceRowLoss(rows, logits, labels, scratch, loss, C, ignoreIndex);
        else _ceReduce(1, logits, labels, scratch, loss, rows, C, ignoreIndex, reduction);
        if (logProb is { } lp) _ceLogProb(rows * C, logits, scratch, lp, C);
    }

    private static void CERowLossImpl(Index1D r, ArrayView1D<float, Stride1D.Dense> logits, ArrayView1D<float, Stride1D.Dense> labels,
        ArrayView1D<float, Stride1D.Dense> stats, ArrayView1D<float, Stride1D.Dense> loss, int C, int ignoreIndex)
    {
        int label = (int)labels[r];
        loss[r] = label == ignoreIndex ? 0f : stats[1 + r] - logits[r * C + label];
    }

    // Single thread, fixed order: deterministic, and the rows of one training example are few (the targets of one sequence).
    private static void CEReduceImpl(Index1D t, ArrayView1D<float, Stride1D.Dense> logits, ArrayView1D<float, Stride1D.Dense> labels,
        ArrayView1D<float, Stride1D.Dense> stats, ArrayView1D<float, Stride1D.Dense> loss, int rows, int C, int ignoreIndex, int reduction)
    {
        float s = 0f;
        for (int r = 0; r < rows; r++)
        {
            int label = (int)labels[r];
            if (label != ignoreIndex) s += stats[1 + r] - logits[r * C + label];
        }
        loss[0] = reduction == 1 ? s / stats[0] : s;
    }

    private static void CELogProbImpl(Index1D i, ArrayView1D<float, Stride1D.Dense> logits, ArrayView1D<float, Stride1D.Dense> stats,
        ArrayView1D<float, Stride1D.Dense> logProb, int C)
    {
        logProb[i] = logits[i] - stats[1 + i / C];
    }

    private static void CECountImpl(Index1D t, ArrayView1D<float, Stride1D.Dense> labels, ArrayView1D<float, Stride1D.Dense> count, int rows, int ignoreIndex)
    {
        float n = 0f;
        for (int r = 0; r < rows; r++) if ((int)labels[r] != ignoreIndex) n += 1f;
        count[0] = n;
    }

    private static void CELseImpl(Index1D r, ArrayView1D<float, Stride1D.Dense> logits, ArrayView1D<float, Stride1D.Dense> stats, int C)
    {
        int b = r * C;
        float m = logits[b];
        for (int c = 1; c < C; c++) m = MathF.Max(m, logits[b + c]);
        float sum = 0f;
        for (int c = 0; c < C; c++) sum += MathF.Exp(logits[b + c] - m);
        stats[1 + r] = m + MathF.Log(sum);
    }

    private static void CEGradImpl(Index1D i, ArrayView1D<float, Stride1D.Dense> logits, ArrayView1D<float, Stride1D.Dense> labels,
        ArrayView1D<float, Stride1D.Dense> stats, ArrayView1D<float, Stride1D.Dense> dlogits, int C, int ignoreIndex)
    {
        int r = i / C, c = i - r * C;
        int label = (int)labels[r];
        float invN = 1f / stats[0];
        dlogits[i] = label == ignoreIndex ? 0f : (MathF.Exp(logits[i] - stats[1 + r]) - (c == label ? 1f : 0f)) * invN;
    }

    // ── Input gradient through a frozen GGUF-quantized weight (FusedDequantMatMul's layout: [N rows][K], value k of row n) ──
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        int, int, int, int>? _matmulQuantT;

    /// <summary>
    /// dx[m, k] = sum_n dy[m, n] * W[k, n] for the forward <c>y = x W</c> with W [K, N] held as raw GGUF blocks (<paramref name="weightQuant"/>,
    /// any type <c>FusedDequantMatMul.DecodeOf</c> decodes, padded to a 4-byte multiple as the loader uploads it). One thread per dx element,
    /// decoding W in-register: the base weights stay compressed. Correct first; a tiled kernel replaces it when it is the bottleneck.
    /// </summary>
    public void MatMulTransposedQuant(ArrayView1D<float, Stride1D.Dense> dy, ArrayView1D<byte, Stride1D.Dense> weightQuant,
        ArrayView1D<float, Stride1D.Dense> dx, int M, int K, int N, GGUF.GGMLType type)
    {
        _matmulQuantT ??= _acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, int, int, int, int>(MatMulTransposedQuantImpl);
        int code = Kernels.FusedDequantMatMul.TypeCode(type);
        int rowBytes = Kernels.FusedDequantMatMul.RowBytesOf(code, K);
        if (UseTensorCoreQuantT && M >= 16 && N % Trellis2.MmaGemm.BK == 0 && K % 8 == 0 && K >= 64 && Kernels.CudaCaps.TensorCores(_acc))
        {
            MatMulTransposedQuantTensorCore(dy, weightQuant, dx, M, K, N, code, rowBytes);
            return;
        }
        if (UseTiledQuantT && _acc.AcceleratorType != AcceleratorType.WebGL && _acc.MaxNumThreadsPerGroup >= TqThreads)
        {
            _matmulQuantTTiled ??= _acc.LoadStreamKernel<ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>, int, int, int, int, int, int>(MatMulTransposedQuantTiledImpl);
            int tilesK = (K + TqK - 1) / TqK;
            _matmulQuantTTiled(new KernelConfig(((M + TqM - 1) / TqM) * tilesK, TqThreads), dy, weightQuant.Cast<byte, int>(), dx, M, K, N, code, rowBytes, tilesK);
            return;
        }
        _matmulQuantT(M * K, dy, weightQuant.Cast<byte, int>(), dx, K, N, code, rowBytes);
    }

    /// <summary>The shared-memory tiled kernel for <see cref="MatMulTransposedQuant"/> (default on; false = the per-element kernel, the A/B control).</summary>
    public static bool UseTiledQuantT { get; set; } = true;

    /// <summary>
    /// CUDA sm_80+: <see cref="MatMulTransposedQuant"/> on the tensor cores (default on; false = the SIMT tiled kernel, the A/B control). W is decoded
    /// and transposed ONCE per call into a bf16 scratch [K, N], then <c>Trellis2.MmaGemm.KernelBf16Dense</c> computes dx = dy W^T with dy rounded
    /// to bf16 and fp32 accumulation. bf16, not f16: gradients can sit far below f16's normal range (6e-5); bf16 keeps fp32's exponent.
    /// </summary>
    public static bool UseTensorCoreQuantT { get; set; } = true;
    private MemoryBuffer1D<ushort, Stride1D.Dense>? _wtBf16;
    private readonly List<MemoryBuffer1D<ushort, Stride1D.Dense>> _retiredWt = new();
    private MemoryBuffer1D<BFloat16, Stride1D.Dense>? _noBias;
    private MemoryBuffer1D<int, Stride1D.Dense>? _noNbr;
    private Action<KernelConfig, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<ushort, Stride1D.Dense>, int, int, int, int, int>? _dequantTBf16;
    private Action<KernelConfig, ArrayView1D<Trellis2.F4, Stride1D.Dense>, ArrayView1D<Trellis2.U4, Stride1D.Dense>, ArrayView1D<BFloat16, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, int, int, int, int, int, int, int>? _mmaBf16;

    private void MatMulTransposedQuantTensorCore(ArrayView1D<float, Stride1D.Dense> dy, ArrayView1D<byte, Stride1D.Dense> weightQuant,
        ArrayView1D<float, Stride1D.Dense> dx, int M, int K, int N, int code, int rowBytes)
    {
        long need = (long)K * N;
        if (_wtBf16 == null || _wtBf16.Length < need)
        {
            if (_wtBf16 != null) _retiredWt.Add(_wtBf16);   // a queued launch may still read it
            _wtBf16 = _acc.Allocate1D<ushort>(need);
        }
        _noBias ??= _acc.Allocate1D<BFloat16>(8);
        _noNbr ??= _acc.Allocate1D<int>(1);
        _dequantTBf16 ??= _acc.LoadStreamKernel<ArrayView1D<int, Stride1D.Dense>, ArrayView1D<ushort, Stride1D.Dense>, int, int, int, int, int>(DequantTransposeBf16Impl);
        int tilesK = (K + 31) / 32;
        _dequantTBf16(new KernelConfig(((N + 31) / 32) * tilesK, 256), weightQuant.Cast<byte, int>(), _wtBf16.View, K, N, code, rowBytes, tilesK);
        _mmaBf16 ??= _acc.LoadStreamKernel<ArrayView1D<Trellis2.F4, Stride1D.Dense>, ArrayView1D<Trellis2.U4, Stride1D.Dense>, ArrayView1D<BFloat16, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, int, int, int, int, int, int, int>(Trellis2.MmaGemm.KernelBf16Dense);
        // MmaGemm: C[M, Nm] = A[M, Km] * W[Nm, Km]^T. Here A = dy [M, N], W = the transposed weight [K, N], C = dx [M, K].
        int tiles = ((M + Trellis2.MmaGemm.BM - 1) / Trellis2.MmaGemm.BM) * ((K + Trellis2.MmaGemm.BN - 1) / Trellis2.MmaGemm.BN);
        _mmaBf16(new KernelConfig(tiles, Trellis2.MmaGemm.Threads), dy.SubView(0, (long)M * N).Cast<float, Trellis2.F4>(),
            _wtBf16.View.SubView(0, need).Cast<ushort, Trellis2.U4>(), _noBias.View, dx, dx, _noNbr.View, M, N, K, 0, 0, 0, 0);
    }

    // One group = a 32 (stored rows n) x 32 (values k) tile of the GGUF weight: 256 threads decode it with consecutive threads on consecutive k
    // (one row's bytes), stage it in shared memory (33-float rows: no bank conflicts on the transposed read), then write out[k, n] with
    // consecutive threads on consecutive n. bf16 by round to nearest even.
    private static void DequantTransposeBf16Impl(ArrayView1D<int, Stride1D.Dense> w, ArrayView1D<ushort, Stride1D.Dense> o, int K, int N, int code, int rowBytes, int tilesK)
    {
        var sh = SharedMemory.Allocate<float>(32 * 33);
        int t = Grid.IdxX;
        int n0 = (t / tilesK) * 32, k0 = (t % tilesK) * 32;
        int tx = Group.IdxX & 31, ty = Group.IdxX >> 5;
        for (int r = 0; r < 4; r++)
        {
            int nn = ty + 8 * r, n = n0 + nn, k = k0 + tx;
            sh[nn * 33 + tx] = n < N && k < K ? Kernels.FusedDequantMatMul.DecodeOf(code, w, n * rowBytes, k) : 0f;
        }
        Group.Barrier();
        for (int r = 0; r < 4; r++)
        {
            int kk = ty + 8 * r, k = k0 + kk, n = n0 + tx;
            if (k < K && n < N)
            {
                uint b = (uint)Interop.FloatAsInt(sh[tx * 33 + kk]);
                b += 0x7FFFu + ((b >> 16) & 1u);
                o[(long)k * N + n] = (ushort)(b >> 16);
            }
        }
    }
    // MEASURED 2026-11-04 (llama3.1:8b training step, 1,600 tokens, per-op sync): with TqM = 16 the transposed matmul was 60% of the step
    // (25.5 s, 16x the forward GEMMs): each decoded weight served only 16 rows, and every one of the 100 row tiles decoded the same weight
    // tile again - decode-bound. TqM = 64 with a 4 x 4 register block per thread: each decode serves 64 rows.
    private const int TqM = 64, TqK = 64, TqN = 16, TqThreads = 256;
    private Action<KernelConfig, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        int, int, int, int, int, int>? _matmulQuantTTiled;

    // One group = a TqM x TqK (64 x 64) tile of dx; thread (tr, tc) of the 16 x 16 group owns rows tr*4..+3 and columns tc*4..+3 (a 4 x 4
    // register block). Per chunk of TqN weight rows: the group stages dy[TqM, TqN] and DECODES W[TqN rows, TqK cols] once into shared
    // memory (each W value is decoded once per 64 output rows), then every thread accumulates its 16 outputs. Bounds are zero-filled.
    private static void MatMulTransposedQuantTiledImpl(ArrayView1D<float, Stride1D.Dense> dy, ArrayView1D<int, Stride1D.Dense> w,
        ArrayView1D<float, Stride1D.Dense> dx, int M, int K, int N, int code, int rowBytes, int tilesK)
    {
        var dyT = SharedMemory.Allocate<float>(TqM * TqN);
        var wT = SharedMemory.Allocate<float>(TqN * TqK);
        int tile = Grid.IdxX;
        int m0 = (tile / tilesK) * TqM, k0 = (tile % tilesK) * TqK;
        int tid = Group.IdxX;
        int tr = tid / 16, tc = tid % 16;
        int mb = tr * 4, kb = tc * 4;
        float c00 = 0f, c01 = 0f, c02 = 0f, c03 = 0f, c10 = 0f, c11 = 0f, c12 = 0f, c13 = 0f;
        float c20 = 0f, c21 = 0f, c22 = 0f, c23 = 0f, c30 = 0f, c31 = 0f, c32 = 0f, c33 = 0f;
        for (int n0 = 0; n0 < N; n0 += TqN)
        {
            for (int e = tid; e < TqM * TqN; e += TqThreads)
            {
                int r = e / TqN, c = e - r * TqN;
                int m = m0 + r, n = n0 + c;
                dyT[e] = m < M && n < N ? dy[m * N + n] : 0f;
            }
            for (int e = tid; e < TqN * TqK; e += TqThreads)
            {
                int r = e / TqK, c = e - r * TqK;
                int n = n0 + r, k = k0 + c;
                wT[e] = n < N && k < K ? Kernels.FusedDequantMatMul.DecodeOf(code, w, n * rowBytes, k) : 0f;
            }
            Group.Barrier();
            for (int nn = 0; nn < TqN; nn++)
            {
                float d0 = dyT[(mb + 0) * TqN + nn], d1 = dyT[(mb + 1) * TqN + nn], d2 = dyT[(mb + 2) * TqN + nn], d3 = dyT[(mb + 3) * TqN + nn];
                int wb = nn * TqK + kb;
                float w0 = wT[wb], w1 = wT[wb + 1], w2 = wT[wb + 2], w3 = wT[wb + 3];
                c00 += d0 * w0; c01 += d0 * w1; c02 += d0 * w2; c03 += d0 * w3;
                c10 += d1 * w0; c11 += d1 * w1; c12 += d1 * w2; c13 += d1 * w3;
                c20 += d2 * w0; c21 += d2 * w1; c22 += d2 * w2; c23 += d2 * w3;
                c30 += d3 * w0; c31 += d3 * w1; c32 += d3 * w2; c33 += d3 * w3;
            }
            Group.Barrier();
        }
        int ko = k0 + kb;
        for (int i = 0; i < 4; i++)
        {
            int mo = m0 + mb + i;
            if (mo >= M) break;
            float a0 = i == 0 ? c00 : i == 1 ? c10 : i == 2 ? c20 : c30;
            float a1 = i == 0 ? c01 : i == 1 ? c11 : i == 2 ? c21 : c31;
            float a2 = i == 0 ? c02 : i == 1 ? c12 : i == 2 ? c22 : c32;
            float a3 = i == 0 ? c03 : i == 1 ? c13 : i == 2 ? c23 : c33;
            int ob = mo * K + ko;
            if (ko < K) dx[ob] = a0;
            if (ko + 1 < K) dx[ob + 1] = a1;
            if (ko + 2 < K) dx[ob + 2] = a2;
            if (ko + 3 < K) dx[ob + 3] = a3;
        }
    }

    private static void MatMulTransposedQuantImpl(Index1D i, ArrayView1D<float, Stride1D.Dense> dy, ArrayView1D<int, Stride1D.Dense> w,
        ArrayView1D<float, Stride1D.Dense> dx, int K, int N, int code, int rowBytes)
    {
        int m = i / K, k = i - m * K;
        int dyBase = m * N;
        float acc = 0f;
        for (int n = 0; n < N; n++)
            acc += dy[dyBase + n] * Kernels.FusedDequantMatMul.DecodeOf(code, w, n * rowBytes, k);
        dx[i] = acc;
    }

    // ── Gather of rows along the sequence axis: backward = scatter the selected rows' gradients into zeros ──
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>>? _zero;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int>? _scatterRows;

    /// <summary>
    /// dx [rows, C] = 0 except dx[positions[j]] = dy[j] for j &lt; n: the gradient of gathering n DISTINCT rows of x (the training loss's
    /// labelled positions). Two dispatches: zero-fill, then one thread per selected value. A scatter: not on WebGL (one store per thread at its own slot).
    /// </summary>
    public void GatherRowsGrad(ArrayView1D<float, Stride1D.Dense> dy, ArrayView1D<float, Stride1D.Dense> positions, ArrayView1D<float, Stride1D.Dense> dx,
        int rows, int n, int C)
    {
        if (_acc.AcceleratorType == AcceleratorType.WebGL)
            throw new NotSupportedException("GatherRowsGrad scatters rows, which WebGL's transform feedback cannot (one store per thread at its own slot)");
        _zero ??= _acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>>(ZeroImpl);
        _scatterRows ??= _acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, int>(ScatterRowsImpl);
        _zero(rows * C, dx);
        _scatterRows(n * C, dy, positions, dx, C);
    }

    private static void ZeroImpl(Index1D i, ArrayView1D<float, Stride1D.Dense> y) => y[i] = 0f;

    private static void ScatterRowsImpl(Index1D i, ArrayView1D<float, Stride1D.Dense> dy, ArrayView1D<float, Stride1D.Dense> positions,
        ArrayView1D<float, Stride1D.Dense> dx, int C)
    {
        int j = i / C, c = i - j * C;
        dx[(int)positions[j] * C + c] = dy[i];
    }

    // ── y = a * y + b * x (gradient accumulation) ──
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, float, float>? _axpby;

    /// <summary>y[i] = a * y[i] + b * x[i] for i &lt; n (a = 0 overwrites y).</summary>
    public void Axpby(ArrayView1D<float, Stride1D.Dense> y, ArrayView1D<float, Stride1D.Dense> x, int n, float a, float b)
    {
        _axpby ??= _acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, float, float>(AxpbyImpl);
        _axpby(n, y, x, a, b);
    }

    private static void AxpbyImpl(Index1D i, ArrayView1D<float, Stride1D.Dense> y, ArrayView1D<float, Stride1D.Dense> x, float a, float b)
        => y[i] = (a == 0f ? 0f : a * y[i]) + b * x[i];

    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, float>? _scale;

    /// <summary>y[i] *= a for i &lt; n, in place.</summary>
    public void Scale(ArrayView1D<float, Stride1D.Dense> y, int n, float a)
    {
        _scale ??= _acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, float>(ScaleImpl);
        _scale(n, y, a);
    }

    private static void ScaleImpl(Index1D i, ArrayView1D<float, Stride1D.Dense> y, float a) => y[i] = y[i] * a;

    // ── AdamW (torch.optim.AdamW, decoupled weight decay) ──
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, AdamWStep>? _adamw;

    /// <summary>One AdamW step's scalars; the bias corrections are computed on the host from the step number.</summary>
    public struct AdamWStep
    {
        public float Lr, Beta1, Beta2, Eps, WeightDecay, BiasCorrection1, SqrtBiasCorrection2;
    }

    /// <summary>
    /// In place, element-wise, exactly torch.optim.AdamW's order of operations: p *= 1 - lr*wd; m = b1 m + (1-b1) g;
    /// v = b2 v + (1-b2) g^2; p -= (lr / bc1) * m / (sqrt(v) / sqrt(bc2) + eps), with bc_k = 1 - beta_k^step.
    /// </summary>
    public void AdamW(ArrayView1D<float, Stride1D.Dense> p, ArrayView1D<float, Stride1D.Dense> g, ArrayView1D<float, Stride1D.Dense> m,
        ArrayView1D<float, Stride1D.Dense> v, int n, float lr, float beta1, float beta2, float eps, float weightDecay, int step)
    {
        _adamw ??= _acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, AdamWStep>(AdamWImpl);
        var s = new AdamWStep
        {
            Lr = lr, Beta1 = beta1, Beta2 = beta2, Eps = eps, WeightDecay = weightDecay,
            BiasCorrection1 = (float)(1 - Math.Pow(beta1, step)), SqrtBiasCorrection2 = (float)Math.Sqrt(1 - Math.Pow(beta2, step)),
        };
        _adamw(n, p, g, m, v, s);
    }

    private static void AdamWImpl(Index1D i, ArrayView1D<float, Stride1D.Dense> p, ArrayView1D<float, Stride1D.Dense> g,
        ArrayView1D<float, Stride1D.Dense> m, ArrayView1D<float, Stride1D.Dense> v, AdamWStep s)
    {
        float gi = g[i];
        float pi = p[i] * (1f - s.Lr * s.WeightDecay);
        float mi = s.Beta1 * m[i] + (1f - s.Beta1) * gi;
        float vi = s.Beta2 * v[i] + (1f - s.Beta2) * gi * gi;
        m[i] = mi;
        v[i] = vi;
        p[i] = pi - (s.Lr / s.BiasCorrection1) * mi / (MathF.Sqrt(vi) / s.SqrtBiasCorrection2 + s.Eps);
    }

    // ── Attention (FusedAttentionKernel, seq-major q/k/v/out [seq, heads, hd], causal, GQA, window, kv_offset) ──
    /// <summary>Attention geometry for the backward kernels.</summary>
    public struct AttnDims
    {
        public int Heads, KvHeads, SeqQ, SeqKV, HeadDim, Causal, Window, KvOffset;
        public float Scale;
    }

    /// <summary>
    /// dq, dk, dv of <c>out = softmax(q k^T * scale + mask) v</c> given dout and the forward's out (for D = rowsum(dout * out)).
    /// <paramref name="stats"/> is scratch of 2 * Heads * SeqQ floats (log-sum-exp and D per query row).
    /// </summary>
    public void Attention(ArrayView1D<float, Stride1D.Dense> q, ArrayView1D<float, Stride1D.Dense> k, ArrayView1D<float, Stride1D.Dense> v,
        ArrayView1D<float, Stride1D.Dense> outp, ArrayView1D<float, Stride1D.Dense> dout,
        ArrayView1D<float, Stride1D.Dense> dq, ArrayView1D<float, Stride1D.Dense> dk, ArrayView1D<float, Stride1D.Dense> dv,
        ArrayView1D<float, Stride1D.Dense> stats, AttnDims d)
    {
        _attnStats ??= _acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, AttnDims>(AttnStatsImpl);
        _attnDq ??= _acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, AttnDims>(AttnDqImpl);
        _attnDkDv ??= _acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, AttnDims>(AttnDkDvImpl);
        int rowsQ = d.Heads * d.SeqQ;
        if (UseGemmAttentionBackward && d.Heads % d.KvHeads == 0 && _acc.AcceleratorType != AcceleratorType.WebGL && _acc.MaxNumThreadsPerGroup >= 256)
        {
            AttentionGemm(q, k, v, outp, dout, dq, dk, dv, stats, d);
            return;
        }
        _attnStats(rowsQ, q, k, outp, dout, stats, d);
        if (UseBlockedAttentionBackward && d.HeadDim <= FaMaxHd && _acc.AcceleratorType != AcceleratorType.WebGL && _acc.MaxNumThreadsPerGroup >= FaThreads)
        {
            _faDkDv ??= _acc.LoadStreamKernel<ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>, AttnDims>(FaDkDvImpl);
            _faDq ??= _acc.LoadStreamKernel<ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, AttnDims>(FaDqImpl);
            _faDkDv(new KernelConfig(d.KvHeads * ((d.SeqKV + FaB - 1) / FaB), FaThreads), q, k, v, dout, stats, dk, dv, d);
            _faDq(new KernelConfig(d.Heads * ((d.SeqQ + FaB - 1) / FaB), FaThreads), q, k, v, dout, stats, dq, d);
            return;
        }
        _attnDq(rowsQ * d.HeadDim, q, k, v, dout, stats, dq, d);
        _attnDkDv(d.KvHeads * d.SeqKV * d.HeadDim, q, k, v, dout, stats, dk, dv, d);
    }

    /// <summary>
    /// The blocked attention backward (default on; false = the per-element recompute, the A/B control). The per-element kernels cost
    /// O(seq^2 * head_dim) PER OUTPUT VALUE: minutes per step at Vertex's ~2,100-token prompts on an 8B model. The blocked kernels keep a block
    /// of keys (or queries) in shared memory and walk the other side once, so each score is computed once per block, not once per output.
    /// </summary>
    public static bool UseBlockedAttentionBackward { get; set; } = true;
    // FaB = the block a group keeps resident (keys for dk/dv, query rows for dq); FaS = the rows of the OTHER side processed per iteration.
    // MEASURED 2026-11-04 (llama3.1:8b, 1,600 tokens): one row per iteration (3 barriers each, 16 of 256 threads scoring) made attention
    // backward 52% of a training step (13.3 s, ~415 ms per layer). FaS = 8: 128 threads score 8 x 16 pairs per iteration, 8x fewer rounds.
    private const int FaB = 16, FaS = 8, FaThreads = 256, FaMaxHd = 128;
    private Action<KernelConfig, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, AttnDims>? _faDkDv;
    private Action<KernelConfig, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, AttnDims>? _faDq;

    // 8 accumulators per thread (FaB * FaMaxHd / FaThreads): slot t holds output e = tid + t * FaThreads.
    private static float Slot(int t, float a0, float a1, float a2, float a3, float a4, float a5, float a6, float a7)
        => t == 0 ? a0 : t == 1 ? a1 : t == 2 ? a2 : t == 3 ? a3 : t == 4 ? a4 : t == 5 ? a5 : t == 6 ? a6 : a7;

    // Group = (kv head, block of FaB keys), resident in shared memory with dk/dv accumulators in registers. The group walks the query rows of
    // every head sharing this kv head, FaS rows per iteration: 128 threads score the FaS x FaB (row, key) pairs (p and dS), then every thread
    // adds the FaS rows' contributions to its outputs.
    private static void FaDkDvImpl(ArrayView1D<float, Stride1D.Dense> q, ArrayView1D<float, Stride1D.Dense> k, ArrayView1D<float, Stride1D.Dense> v,
        ArrayView1D<float, Stride1D.Dense> dout, ArrayView1D<float, Stride1D.Dense> stats, ArrayView1D<float, Stride1D.Dense> dk,
        ArrayView1D<float, Stride1D.Dense> dv, AttnDims d)
    {
        var kB = SharedMemory.Allocate<float>(FaB * FaMaxHd);
        var vB = SharedMemory.Allocate<float>(FaB * FaMaxHd);
        var qS = SharedMemory.Allocate<float>(FaS * FaMaxHd);
        var doS = SharedMemory.Allocate<float>(FaS * FaMaxHd);
        var pS = SharedMemory.Allocate<float>(FaS * FaB);
        var dsS = SharedMemory.Allocate<float>(FaS * FaB);
        int hd = d.HeadDim;
        int blocks = (d.SeqKV + FaB - 1) / FaB;
        int kvh = Grid.IdxX / blocks, j0 = (Grid.IdxX % blocks) * FaB;
        int tid = Group.IdxX;
        int nOut = FaB * hd;
        for (int e = tid; e < nOut; e += FaThreads)
        {
            int jj = e / hd, c = e - jj * hd, j = j0 + jj;
            kB[e] = j < d.SeqKV ? k[(j * d.KvHeads + kvh) * hd + c] : 0f;
            vB[e] = j < d.SeqKV ? v[(j * d.KvHeads + kvh) * hd + c] : 0f;
        }
        float k0 = 0f, k1 = 0f, k2 = 0f, k3 = 0f, k4 = 0f, k5 = 0f, k6 = 0f, k7 = 0f;
        float v0 = 0f, v1 = 0f, v2 = 0f, v3 = 0f, v4 = 0f, v5 = 0f, v6 = 0f, v7 = 0f;
        Group.Barrier();
        int group = d.Heads / d.KvHeads;
        // causal: row i sees key j only when j <= i + kv_offset; rows below j0 - kv_offset see none of this block (a uniform bound)
        int iStart = d.Causal == 1 ? Math.Max(0, j0 - d.KvOffset) : 0;
        for (int g = 0; g < group; g++)
        {
            int h = kvh * group + g;
            for (int i0 = iStart; i0 < d.SeqQ; i0 += FaS)
            {
                for (int e = tid; e < FaS * hd; e += FaThreads)
                {
                    int ii = e / hd, c = e - ii * hd, i = i0 + ii;
                    int ob = (i * d.Heads + h) * hd + c;
                    qS[e] = i < d.SeqQ ? q[ob] : 0f;
                    doS[e] = i < d.SeqQ ? dout[ob] : 0f;
                }
                Group.Barrier();
                if (tid < FaS * FaB)
                {
                    int ii = tid / FaB, jj = tid - ii * FaB, i = i0 + ii, j = j0 + jj;
                    float p = 0f, ds = 0f;
                    if (i < d.SeqQ && j < d.SeqKV && Allowed(i, j, d))
                    {
                        float sc = 0f, dp = 0f;
                        for (int c = 0; c < hd; c++) { sc += qS[ii * hd + c] * kB[jj * hd + c]; dp += doS[ii * hd + c] * vB[jj * hd + c]; }
                        int sr = h * d.SeqQ + i;
                        p = MathF.Exp(sc * d.Scale - stats[sr]);
                        ds = p * (dp - stats[d.Heads * d.SeqQ + sr]);
                    }
                    pS[tid] = p; dsS[tid] = ds;
                }
                Group.Barrier();
                for (int t = 0; t < 8; t++)
                {
                    int e = tid + t * FaThreads;
                    if (e >= nOut) break;
                    int jj = e / hd, c = e - jj * hd;
                    float addK = 0f, addV = 0f;
                    for (int ii = 0; ii < FaS; ii++)
                    {
                        addK += dsS[ii * FaB + jj] * qS[ii * hd + c];
                        addV += pS[ii * FaB + jj] * doS[ii * hd + c];
                    }
                    switch (t)
                    {
                        case 0: k0 += addK; v0 += addV; break;
                        case 1: k1 += addK; v1 += addV; break;
                        case 2: k2 += addK; v2 += addV; break;
                        case 3: k3 += addK; v3 += addV; break;
                        case 4: k4 += addK; v4 += addV; break;
                        case 5: k5 += addK; v5 += addV; break;
                        case 6: k6 += addK; v6 += addV; break;
                        default: k7 += addK; v7 += addV; break;
                    }
                }
                Group.Barrier();
            }
        }
        for (int t = 0; t < 8; t++)
        {
            int e = tid + t * FaThreads;
            if (e >= nOut) break;
            int jj = e / hd, c = e - jj * hd, j = j0 + jj;
            if (j >= d.SeqKV) continue;
            int o = (j * d.KvHeads + kvh) * hd + c;
            dk[o] = Slot(t, k0, k1, k2, k3, k4, k5, k6, k7) * d.Scale;
            dv[o] = Slot(t, v0, v1, v2, v3, v4, v5, v6, v7);
        }
    }

    // Group = (head, block of FaB query rows), resident in shared memory with dq accumulators in registers. The group walks the keys FaS per
    // iteration: 128 threads compute the FaB x FaS (row, key) dS values, then every thread adds them times the keys to its outputs.
    private static void FaDqImpl(ArrayView1D<float, Stride1D.Dense> q, ArrayView1D<float, Stride1D.Dense> k, ArrayView1D<float, Stride1D.Dense> v,
        ArrayView1D<float, Stride1D.Dense> dout, ArrayView1D<float, Stride1D.Dense> stats, ArrayView1D<float, Stride1D.Dense> dq, AttnDims d)
    {
        var qB = SharedMemory.Allocate<float>(FaB * FaMaxHd);
        var doB = SharedMemory.Allocate<float>(FaB * FaMaxHd);
        var kS = SharedMemory.Allocate<float>(FaS * FaMaxHd);
        var vS = SharedMemory.Allocate<float>(FaS * FaMaxHd);
        var dsS = SharedMemory.Allocate<float>(FaB * FaS);
        int hd = d.HeadDim;
        int blocks = (d.SeqQ + FaB - 1) / FaB;
        int h = Grid.IdxX / blocks, i0 = (Grid.IdxX % blocks) * FaB;
        int kvh = h / (d.Heads / d.KvHeads);
        int tid = Group.IdxX;
        int nOut = FaB * hd;
        for (int e = tid; e < nOut; e += FaThreads)
        {
            int ii = e / hd, c = e - ii * hd, i = i0 + ii;
            qB[e] = i < d.SeqQ ? q[(i * d.Heads + h) * hd + c] : 0f;
            doB[e] = i < d.SeqQ ? dout[(i * d.Heads + h) * hd + c] : 0f;
        }
        float a0 = 0f, a1 = 0f, a2 = 0f, a3 = 0f, a4 = 0f, a5 = 0f, a6 = 0f, a7 = 0f;
        Group.Barrier();
        // causal: the block's last row sees keys up to (i0 + FaB - 1) + kv_offset (a uniform bound)
        int jEnd = d.Causal == 1 ? Math.Min(d.SeqKV, i0 + FaB + d.KvOffset) : d.SeqKV;
        for (int j0 = 0; j0 < jEnd; j0 += FaS)
        {
            for (int e = tid; e < FaS * hd; e += FaThreads)
            {
                int jj = e / hd, c = e - jj * hd, j = j0 + jj;
                int kb = (j * d.KvHeads + kvh) * hd + c;
                kS[e] = j < d.SeqKV ? k[kb] : 0f;
                vS[e] = j < d.SeqKV ? v[kb] : 0f;
            }
            Group.Barrier();
            if (tid < FaB * FaS)
            {
                int ii = tid / FaS, jj = tid - ii * FaS, i = i0 + ii, j = j0 + jj;
                float ds = 0f;
                if (i < d.SeqQ && j < d.SeqKV && Allowed(i, j, d))
                {
                    float sc = 0f, dp = 0f;
                    for (int c = 0; c < hd; c++) { sc += qB[ii * hd + c] * kS[jj * hd + c]; dp += doB[ii * hd + c] * vS[jj * hd + c]; }
                    int sr = h * d.SeqQ + i;
                    float p = MathF.Exp(sc * d.Scale - stats[sr]);
                    ds = p * (dp - stats[d.Heads * d.SeqQ + sr]);
                }
                dsS[tid] = ds;
            }
            Group.Barrier();
            for (int t = 0; t < 8; t++)
            {
                int e = tid + t * FaThreads;
                if (e >= nOut) break;
                int ii = e / hd, c = e - ii * hd;
                float add = 0f;
                for (int jj = 0; jj < FaS; jj++) add += dsS[ii * FaS + jj] * kS[jj * hd + c];
                switch (t)
                {
                    case 0: a0 += add; break;
                    case 1: a1 += add; break;
                    case 2: a2 += add; break;
                    case 3: a3 += add; break;
                    case 4: a4 += add; break;
                    case 5: a5 += add; break;
                    case 6: a6 += add; break;
                    default: a7 += add; break;
                }
            }
            Group.Barrier();
        }
        for (int t = 0; t < 8; t++)
        {
            int e = tid + t * FaThreads;
            if (e >= nOut) break;
            int ii = e / hd, c = e - ii * hd, i = i0 + ii;
            if (i >= d.SeqQ) continue;
            dq[(i * d.Heads + h) * hd + c] = Slot(t, a0, a1, a2, a3, a4, a5, a6, a7) * d.Scale;
        }
    }

    private static bool Allowed(int i, int j, AttnDims d)
    {
        int qPos = i + d.KvOffset;
        if (d.Causal == 1 && j > qPos) return false;
        if (d.Window > 0 && qPos - j >= d.Window) return false;
        return true;
    }

    private static float QK(ArrayView1D<float, Stride1D.Dense> q, ArrayView1D<float, Stride1D.Dense> k, int h, int i, int kvh, int j, AttnDims d)
    {
        int qb = (i * d.Heads + h) * d.HeadDim, kb = (j * d.KvHeads + kvh) * d.HeadDim;
        float s = 0f;
        for (int e = 0; e < d.HeadDim; e++) s += q[qb + e] * k[kb + e];
        return s * d.Scale;
    }

    // stats[h*SeqQ + i] = logsumexp_j(score), stats[Heads*SeqQ + h*SeqQ + i] = D = sum_e dout * out
    private static void AttnStatsImpl(Index1D t, ArrayView1D<float, Stride1D.Dense> q, ArrayView1D<float, Stride1D.Dense> k,
        ArrayView1D<float, Stride1D.Dense> outp, ArrayView1D<float, Stride1D.Dense> dout, ArrayView1D<float, Stride1D.Dense> stats, AttnDims d)
    {
        int h = t / d.SeqQ, i = t % d.SeqQ;
        int kvh = h / (d.Heads / d.KvHeads);
        float m = float.NegativeInfinity;
        for (int j = 0; j < d.SeqKV; j++) if (Allowed(i, j, d)) m = MathF.Max(m, QK(q, k, h, i, kvh, j, d));
        float sum = 0f;
        for (int j = 0; j < d.SeqKV; j++) if (Allowed(i, j, d)) sum += MathF.Exp(QK(q, k, h, i, kvh, j, d) - m);
        stats[t] = m + MathF.Log(sum);
        int ob = (i * d.Heads + h) * d.HeadDim;
        float dd = 0f;
        for (int e = 0; e < d.HeadDim; e++) dd += dout[ob + e] * outp[ob + e];
        stats[d.Heads * d.SeqQ + t] = dd;
    }

    // dq[i, h, e] = scale * sum_j P_ij (dout_i . v_j - D_i) k_j[e]
    private static void AttnDqImpl(Index1D t, ArrayView1D<float, Stride1D.Dense> q, ArrayView1D<float, Stride1D.Dense> k,
        ArrayView1D<float, Stride1D.Dense> v, ArrayView1D<float, Stride1D.Dense> dout, ArrayView1D<float, Stride1D.Dense> stats,
        ArrayView1D<float, Stride1D.Dense> dq, AttnDims d)
    {
        int e = t % d.HeadDim, row = t / d.HeadDim;
        int i = row / d.Heads, h = row % d.Heads;             // seq-major element order [i, h, e]
        int kvh = h / (d.Heads / d.KvHeads);
        int sr = h * d.SeqQ + i;
        float lse = stats[sr], D = stats[d.Heads * d.SeqQ + sr];
        int ob = (i * d.Heads + h) * d.HeadDim;
        float acc = 0f;
        for (int j = 0; j < d.SeqKV; j++)
        {
            if (!Allowed(i, j, d)) continue;
            float p = MathF.Exp(QK(q, k, h, i, kvh, j, d) - lse);
            int vb = (j * d.KvHeads + kvh) * d.HeadDim;
            float dp = 0f;
            for (int f = 0; f < d.HeadDim; f++) dp += dout[ob + f] * v[vb + f];
            acc += p * (dp - D) * k[vb + e];
        }
        dq[t] = acc * d.Scale;
    }

    // dk[j, kvh, e] = scale * sum_{h in group} sum_i P_ij (dout_i . v_j - D_i) q_i[e];  dv[j, kvh, e] = sum P_ij dout_i[e]
    private static void AttnDkDvImpl(Index1D t, ArrayView1D<float, Stride1D.Dense> q, ArrayView1D<float, Stride1D.Dense> k,
        ArrayView1D<float, Stride1D.Dense> v, ArrayView1D<float, Stride1D.Dense> dout, ArrayView1D<float, Stride1D.Dense> stats,
        ArrayView1D<float, Stride1D.Dense> dk, ArrayView1D<float, Stride1D.Dense> dv, AttnDims d)
    {
        int e = t % d.HeadDim, row = t / d.HeadDim;
        int j = row / d.KvHeads, kvh = row % d.KvHeads;
        int group = d.Heads / d.KvHeads;
        int vb = (j * d.KvHeads + kvh) * d.HeadDim;
        float accK = 0f, accV = 0f;
        for (int g = 0; g < group; g++)
        {
            int h = kvh * group + g;
            for (int i = 0; i < d.SeqQ; i++)
            {
                if (!Allowed(i, j, d)) continue;
                int sr = h * d.SeqQ + i;
                float p = MathF.Exp(QK(q, k, h, i, kvh, j, d) - stats[sr]);
                int ob = (i * d.Heads + h) * d.HeadDim;
                float dp = 0f;
                for (int f = 0; f < d.HeadDim; f++) dp += dout[ob + f] * v[vb + f];
                accK += p * (dp - stats[d.Heads * d.SeqQ + sr]) * q[ob + e];
                accV += p * dout[ob + e];
            }
        }
        dk[t] = accK * d.Scale;
        dv[t] = accV;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        _ones?.Dispose(); _ones = null;
        _wtBf16?.Dispose(); _wtBf16 = null;
        foreach (var b in _retiredWt) b.Dispose();
        _retiredWt.Clear();
        _noBias?.Dispose(); _noBias = null;
        _noNbr?.Dispose(); _noNbr = null;
        DisposeAttentionGemm();
    }
}
