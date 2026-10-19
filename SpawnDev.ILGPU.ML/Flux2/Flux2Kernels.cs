using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Flux2;

using FV = ArrayView1D<float, Stride1D.Dense>;
using UV = ArrayView1D<uint, Stride1D.Dense>;
using IV = ArrayView1D<int, Stride1D.Dense>;

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

    private Action<Index1D, FV, FV, IV, int, int, int, int>? _ropeHalf;
    private static void RopeHalfImpl(Index1D i, FV buf, FV invFreq, IV positions, int stride, int off, int H, int D)
    {
        int half = D / 2;
        int k = i % half, rest = i / half, h = rest % H, n = rest / H;
        float ang = positions[n] * invFreq[k];
        float c = MathF.Cos(ang), s = MathF.Sin(ang);
        long b = (long)n * stride + off + h * D;
        float x0 = buf[b + k], x1 = buf[b + half + k];                     // rotate_half: element k pairs with element k + D/2
        buf[b + k] = x0 * c - x1 * s;
        buf[b + half + k] = x1 * c + x0 * s;
    }
    /// <summary>
    /// In place rotary embedding (HF rotate_half convention) of the H head vectors of width D at columns off.. of rows [n, stride] with an explicit position per row
    /// (<paramref name="positions"/>[row]; FLUX 3's text encoder puts its padding rows at position 1, not at their index).
    /// </summary>
    public void RopeHalfPositions(FV buf, FV invFreq, IV positions, int rows, int stride, int off, int H, int D)
    {
        _ropeHalf ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, IV, int, int, int, int>(RopeHalfImpl);
        _ropeHalf(rows * H * (D / 2), buf, invFreq, positions, stride, off, H, D);
    }

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

    // ───────────────────────────────────────────── 4-bit weight blocks ────────────────────────────────

    private static float HalfToFloat(uint h)
    {
        uint sign = (h >> 15) & 1u, exp = (h >> 10) & 0x1Fu, man = h & 0x3FFu;
        float r = exp == 0u ? man * (1f / 16777216f)                                      // subnormal: man * 2^-24
                : exp == 31u ? 65504f                                                    // inf / nan never occur in a scale
                : Interop.IntAsFloat(((exp + 112u) << 23) | (man << 13));
        return sign == 1u ? -r : r;
    }
    private static uint Bf16Bits(float f)
    {
        uint bits = Interop.FloatAsInt(f);
        return (bits + 0x7FFFu + ((bits >> 16) & 1u)) >> 16;
    }
    private static float DequantOne(UV q, long row, int col, int K)
    {
        long b5 = (row * (K / 32) + col / 32) * 5;
        uint h = q[b5];
        float d = HalfToFloat(h & 0xFFFFu), m = HalfToFloat(h >> 16);
        int e = col % 32;
        return d * ((q[b5 + 1 + e / 8] >> (4 * (e % 8))) & 15u) + m;
    }

    private static float DequantOne8(UV q, long row, int col, int K)
    {
        long b9 = (row * (K / 32) + col / 32) * 9;
        uint h = q[b9];
        float d = HalfToFloat(h & 0xFFFFu), m = HalfToFloat(h >> 16);
        int e = col % 32;
        return d * ((q[b9 + 1 + e / 4] >> (8 * (e % 4))) & 255u) + m;
    }

    private Action<Index1D, UV, UV, int>? _dequant8;
    private static void Dequant8Impl(Index1D i, UV q, UV dst, int K)
    {
        long e0 = 2L * i;
        long row = e0 / K; int col = (int)(e0 % K);
        dst[i] = Bf16Bits(DequantOne8(q, row, col, K)) | (Bf16Bits(DequantOne8(q, row, col + 1, K)) << 16);
    }
    /// <summary>Dequantise a [N, K] matrix of 8-bit blocks (quantize_flux2.py ... 8: 9 words per block of 32) into bf16 pairs packed in <paramref name="dst"/> (N*K/2 words).</summary>
    public void DequantQ8(UV q, UV dst, int N, int K)
    {
        _dequant8 ??= _acc.LoadAutoGroupedStreamKernel<Index1D, UV, UV, int>(Dequant8Impl);
        _dequant8((int)((long)N * K / 2), q, dst, K);
    }

    private Action<Index1D, UV, ArrayView1D<int, Stride1D.Dense>, FV, int>? _dequantRows8;
    private static void DequantRows8Impl(Index1D i, UV q, ArrayView1D<int, Stride1D.Dense> ids, FV o, int K)
    {
        int t = i / K, c = i % K;
        o[i] = DequantOne8(q, ids[t], c, K);
    }
    /// <summary>The embedding gather for an 8-bit [N, K] matrix.</summary>
    public void DequantRows8(UV q, ArrayView1D<int, Stride1D.Dense> ids, FV o, int count, int K)
    {
        _dequantRows8 ??= _acc.LoadAutoGroupedStreamKernel<Index1D, UV, ArrayView1D<int, Stride1D.Dense>, FV, int>(DequantRows8Impl);
        _dequantRows8(count * K, q, ids, o, K);
    }

    private Action<Index1D, UV, UV, int>? _dequant;
    private static void DequantImpl(Index1D i, UV q, UV dst, int K)
    {
        long e0 = 2L * i;
        long row = e0 / K; int col = (int)(e0 % K);
        dst[i] = Bf16Bits(DequantOne(q, row, col, K)) | (Bf16Bits(DequantOne(q, row, col + 1, K)) << 16);
    }
    /// <summary>Dequantise a [N, K] matrix of 4-bit blocks (see tools/unirig-harness/reference/quantize_flux2.py) into bf16 pairs packed in <paramref name="dst"/> (N*K/2 words).</summary>
    public void DequantQ4(UV q, UV dst, int N, int K)
    {
        _dequant ??= _acc.LoadAutoGroupedStreamKernel<Index1D, UV, UV, int>(DequantImpl);
        _dequant((int)((long)N * K / 2), q, dst, K);
    }

    private Action<Index1D, UV, ArrayView1D<int, Stride1D.Dense>, FV, int>? _dequantRows;
    private static void DequantRowsImpl(Index1D i, UV q, ArrayView1D<int, Stride1D.Dense> ids, FV o, int K)
    {
        int t = i / K, c = i % K;
        o[i] = DequantOne(q, ids[t], c, K);
    }
    /// <summary>o[t, :] = dequantised row ids[t] of a [N, K] 4-bit matrix (the embedding gather), fp32.</summary>
    public void DequantRows(UV q, ArrayView1D<int, Stride1D.Dense> ids, FV o, int count, int K)
    {
        _dequantRows ??= _acc.LoadAutoGroupedStreamKernel<Index1D, UV, ArrayView1D<int, Stride1D.Dense>, FV, int>(DequantRowsImpl);
        _dequantRows(count * K, q, ids, o, K);
    }
}
