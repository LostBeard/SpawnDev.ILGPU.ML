using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Flux2;

using FV = ArrayView1D<float, Stride1D.Dense>;

/// <summary>
/// The kernels FLUX.2 klein needs beside <see cref="Trellis2.DiTKernels"/> (linear, layer norm, flash attention, residual) and <see cref="SkinTokens.SkinTokensKernels"/> (RMS norm, rotary
/// embedding, SwiGLU): attention for the padded text encoder, column copies, and the DiT's interleaved rotary embedding, half-split SwiGLU, sinusoidal timestep embedding and
/// modulation helpers. fp32, one thread per output element (or per row where a reduction is needed).
/// </summary>
public sealed class Flux2Kernels
{
    private readonly Accelerator _acc;
    public Flux2Kernels(Accelerator accelerator) => _acc = accelerator;

    // ───────────────────────────────────────────── causal GQA attention over a padded sequence ──────────

    private Action<Index1D, FV, FV, FV, int, int, int, int, int, float>? _scores;
    private static void ScoresImpl(Index1D i, FV qkv, FV kc, FV scores, int T, int real, int H, int D, int rep, float scale)
    {
        int j = i % T, rest = i / T, t = rest % T, h = rest / T;
        int qCols = H * D, kvCols = qCols / rep, rowStride = qCols + 2 * kvCols;
        float s = -1e30f;                                                     // masked: a later key, or a padding key
        if (j <= t && j < real)
        {
            float acc = 0f;
            long qo = (long)t * rowStride + h * D, ko = (long)j * kvCols + (h / rep) * D;
            for (int d = 0; d < D; d++) acc += qkv[qo + d] * kc[ko + d];
            s = acc * scale;
        }
        scores[i] = s;
    }
    private Action<Index1D, FV, int>? _softmax;
    private static void SoftmaxImpl(Index1D row, FV scores, int nk)
    {
        long o = (long)(int)row * nk;
        float max = scores[o];
        for (int j = 1; j < nk; j++) max = MathF.Max(max, scores[o + j]);
        float sum = 0f;
        for (int j = 0; j < nk; j++) { float e = MathF.Exp(scores[o + j] - max); scores[o + j] = e; sum += e; }
        float inv = 1f / sum;
        for (int j = 0; j < nk; j++) scores[o + j] *= inv;
    }
    private Action<Index1D, FV, FV, FV, int, int, int, int>? _pv;
    private static void PvImpl(Index1D i, FV scores, FV vc, FV o, int T, int H, int D, int rep)
    {
        int C = H * D, t = i / C, c = i % C, h = c / D, d = c % D, kvCols = C / rep;
        long so = ((long)h * T + t) * T;
        float acc = 0f;
        for (int j = 0; j < T; j++) acc += scores[so + j] * vc[(long)j * kvCols + (h / rep) * D + d];
        o[i] = acc;
    }

    /// <summary>
    /// Causal attention of a T-token sequence whose last T - <paramref name="real"/> tokens are padding: query t attends to keys j with j &lt;= t and j &lt; real (HF: causal mask combined with
    /// the padding mask), so padding rows still see the real prompt. q is columns 0..H*D of the fused [T, H*D + 2*kv] projection, k and v are the caches [T, kv] (kv = H*D/rep).
    /// <paramref name="scores"/> holds H*T*T floats; o is [T, H*D].
    /// </summary>
    public void PaddedCausalGqa(FV qkv, FV kc, FV vc, FV scores, FV o, int T, int real, int H, int D, int rep)
    {
        _scores ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, FV, int, int, int, int, int, float>(ScoresImpl);
        _softmax ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, int>(SoftmaxImpl);
        _pv ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, FV, int, int, int, int>(PvImpl);
        _scores((int)((long)H * T * T), qkv, kc, scores, T, real, H, D, rep, 1f / MathF.Sqrt(D));
        _softmax(H * T, scores, T);
        _pv(T * H * D, scores, vc, o, T, H, D, rep);
    }

    // ───────────────────────────────────────────── column copy ──────────────────────────

    private Action<Index1D, FV, FV, int, int, int, int>? _copyCols;
    private static void CopyColsImpl(Index1D i, FV src, FV dst, int width, int srcStride, int dstStride, int dstOff)
    {
        int r = i / width, c = i % width;
        dst[(long)r * dstStride + dstOff + c] = src[(long)r * srcStride + c];
    }
    /// <summary>dst[r, dstOff + c] = src[r, c] for the first <paramref name="width"/> columns of every row.</summary>
    public void CopyCols(FV src, FV dst, int rows, int width, int srcStride, int dstStride, int dstOff)
    {
        _copyCols ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, int, int, int, int>(CopyColsImpl);
        _copyCols(rows * width, src, dst, width, srcStride, dstStride, dstOff);
    }

    // ───────────────────────────────────────────── the DiT's rotary embedding and SwiGLU ──────────────────

    private Action<Index1D, FV, FV, FV, int, int, int, int>? _ropeInterleaved;
    private static void RopeInterleavedImpl(Index1D i, FV buf, FV cos, FV sin, int stride, int off, int H, int D)
    {
        int half = D / 2;
        int p = i % half, rest = i / half, h = rest % H, n = rest / H;
        long b = (long)n * stride + off + h * D;
        long t = (long)n * D + 2 * p;
        float x0 = buf[b + 2 * p], x1 = buf[b + 2 * p + 1];                  // (real, imaginary) are neighbours: diffusers apply_rotary_emb with use_real_unbind_dim = -1
        buf[b + 2 * p] = x0 * cos[t] - x1 * sin[t];
        buf[b + 2 * p + 1] = x1 * cos[t + 1] + x0 * sin[t + 1];
    }
    /// <summary>In place interleaved-pair rotary embedding of the H head vectors of width D at columns off.. of rows [n, stride]; cos and sin are [rows, D] (each frequency repeated for its pair).</summary>
    public void RopeInterleaved(FV buf, FV cos, FV sin, int rows, int stride, int off, int H, int D)
    {
        _ropeInterleaved ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, FV, int, int, int, int>(RopeInterleavedImpl);
        _ropeInterleaved(rows * H * (D / 2), buf, cos, sin, stride, off, H, D);
    }

    private Action<Index1D, FV, FV, int, int, int, int, int>? _swigluCols;
    private static void SwigluColsImpl(Index1D i, FV src, FV dst, int srcStride, int srcOff, int dstStride, int dstOff, int I)
    {
        int r = i / I, c = i % I;
        float g = src[(long)r * srcStride + srcOff + c], u = src[(long)r * srcStride + srcOff + I + c];
        dst[(long)r * dstStride + dstOff + c] = g / (1f + MathF.Exp(-g)) * u;
    }
    /// <summary>dst[r, dstOff + c] = silu(src[r, srcOff + c]) * src[r, srcOff + I + c]: Flux2SwiGLU on a slice of a wider row, written into a slice of another.</summary>
    public void SwigluCols(FV src, int srcStride, int srcOff, FV dst, int dstStride, int dstOff, int rows, int I)
    {
        _swigluCols ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, int, int, int, int, int>(SwigluColsImpl);
        _swigluCols(rows * I, src, dst, srcStride, srcOff, dstStride, dstOff, I);
    }
}
