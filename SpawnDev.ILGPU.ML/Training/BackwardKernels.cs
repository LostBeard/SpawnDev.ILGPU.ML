using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Training;

/// <summary>
/// Backward (gradient) kernels for the graph autograd (Plans/training-autograd-lora-2026-11-04.md, M1.3). Each mirrors a
/// forward kernel's exact math (the forward it differentiates is named on each method). Correctness first: the attention
/// backward is the naive O(seq^2 * head_dim) recompute, one thread per output element, and is replaced by a flash-style
/// kernel at M3 (llama3.1:8b). Every method only DISPATCHES; scratch comes from the caller.
/// </summary>
public sealed class BackwardKernels : IDisposable
{
    private readonly Accelerator _acc;

    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, float>? _rmsRstd;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, int, int>? _rmsMean;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int>? _rmsNorm;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int>? _ceLse;
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
        _attnStats(rowsQ, q, k, outp, dout, stats, d);
        _attnDq(rowsQ * d.HeadDim, q, k, v, dout, stats, dq, d);
        _attnDkDv(d.KvHeads * d.SeqKV * d.HeadDim, q, k, v, dout, stats, dk, dv, d);
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
    public void Dispose() { _ones?.Dispose(); _ones = null; }
}
