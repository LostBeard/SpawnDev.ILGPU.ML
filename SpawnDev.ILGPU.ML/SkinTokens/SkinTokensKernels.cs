using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.SkinTokens;

using FV = ArrayView1D<float, Stride1D.Dense>;
using IV = ArrayView1D<int, Stride1D.Dense>;
using BV = ArrayView1D<BFloat16, Stride1D.Dense>;

/// <summary>
/// The small kernels SkinTokens needs beside <see cref="Trellis2.DiTKernels"/> (linear, layer norm, flash attention, residual add): RMS norm, rotary embedding,
/// grouped-query causal attention over a KV cache, SwiGLU, the VAE's phase-shifted Fourier point embedding, FSQ code projection and the skin head. fp32, one thread
/// per output element (or per row/vector where a reduction is needed).
/// </summary>
public sealed class SkinTokensKernels
{
    private readonly Accelerator _acc;
    public SkinTokensKernels(Accelerator accelerator) => _acc = accelerator;

    // ───────────────────────────────────────────── norms ─────────────────────────────────────────────

    private Action<Index1D, FV, FV, BV, int, float>? _rms;
    private static void RmsImpl(Index1D row, FV x, FV o, BV w, int C, float eps)
    {
        long b = (long)(int)row * C;
        float ss = 0f;
        for (int c = 0; c < C; c++) { float v = x[b + c]; ss += v * v; }
        float inv = 1f / MathF.Sqrt(ss / C + eps);
        for (int c = 0; c < C; c++) o[b + c] = x[b + c] * inv * PrecisionConvert.ConvertToSingle(w[c]);
    }
    /// <summary>o[r, :] = x[r, :] / sqrt(mean(x[r, :]^2) + eps) * w   (torch.nn.RMSNorm / Qwen3RMSNorm; x and o may be the same view).</summary>
    public void RmsNorm(FV x, FV o, BV w, int rows, int C, float eps)
    {
        _rms ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, BV, int, float>(RmsImpl);
        _rms(rows, x, o, w, C, eps);
    }

    private Action<Index1D, FV, BV, int, int, int, int, float>? _rmsHeads;
    private static void RmsHeadsImpl(Index1D i, FV buf, BV w, int stride, int off, int H, int D, float eps)
    {
        int h = i % H, n = i / H;
        long b = (long)n * stride + off + h * D;
        float ss = 0f;
        for (int d = 0; d < D; d++) { float v = buf[b + d]; ss += v * v; }
        float inv = 1f / MathF.Sqrt(ss / D + eps);
        for (int d = 0; d < D; d++) buf[b + d] = buf[b + d] * inv * PrecisionConvert.ConvertToSingle(w[d]);
    }
    /// <summary>In place RMS norm of every D-wide head vector of rows [n, stride] starting at column off (Qwen3's q_norm / k_norm: one weight of D shared by all heads).</summary>
    public void RmsNormHeads(FV buf, BV w, int rows, int stride, int off, int H, int D, float eps)
    {
        _rmsHeads ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, BV, int, int, int, int, float>(RmsHeadsImpl);
        _rmsHeads(rows * H, buf, w, stride, off, H, D, eps);
    }

    // ───────────────────────────────────────────── rotary embedding ─────────────────────────────────────────────

    /// <summary>
    /// inv_freq[k] = 1 / theta^(2k/D), rounded to bf16. HF computes this table in float32 at construction as a non-persistent buffer, and the shipped model is then
    /// converted to bf16, which rounds it (inv_freq[1] 0.80584 -> 0.8046875): the shipped rotary embedding, and the float32 golden, use the ROUNDED values.
    /// </summary>
    public static float[] RopeTable(float theta, int D)
    {
        var t = new float[D / 2];
        for (int k = 0; k < t.Length; k++) t[k] = BF16Round(1f / MathF.Pow(theta, (2f * k) / D));
        return t;
    }

    private Action<Index1D, FV, FV, int, int, int, int, int>? _rope;
    private static void RopeImpl(Index1D i, FV buf, FV invFreq, int stride, int off, int H, int D, int p0)
    {
        int half = D / 2;
        int k = i % half, rest = i / half, h = rest % H, n = rest / H;
        float ang = (p0 + n) * invFreq[k];
        float c = MathF.Cos(ang), s = MathF.Sin(ang);
        long b = (long)n * stride + off + h * D;
        float x0 = buf[b + k], x1 = buf[b + half + k];                     // rotate_half pairs element k with element k + D/2
        buf[b + k] = x0 * c - x1 * s;
        buf[b + half + k] = x1 * c + x0 * s;
    }
    /// <summary>In place rotary embedding (HF rotate_half convention) of the H head vectors of width D at columns off.. of rows [n, stride], row n at position p0 + n; <paramref name="invFreq"/> is <see cref="RopeTable"/>.</summary>
    public void Rope(FV buf, FV invFreq, int rows, int stride, int off, int H, int D, int p0)
    {
        _rope ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, int, int, int, int, int>(RopeImpl);
        _rope(rows * H * (D / 2), buf, invFreq, stride, off, H, D, p0);
    }

    // ───────────────────────────────────────────── grouped-query causal attention ─────────────────────────────────────────────

    private Action<Index1D, FV, FV, FV, int, int, int, int, int, int, float>? _kvStore;
    private static void KvStoreImpl(Index1D i, FV qkv, FV kc, FV vc, int p0, int qCols, int kvCols, int rowStride, int unused1, int unused2, float unused3)
    {
        int t = i / kvCols, c = i % kvCols;
        kc[(long)(p0 + t) * kvCols + c] = qkv[(long)t * rowStride + qCols + c];
        vc[(long)(p0 + t) * kvCols + c] = qkv[(long)t * rowStride + qCols + kvCols + c];
    }
    /// <summary>Append the k and v columns of a fused [T, qCols + 2*kvCols] projection to the layer's caches at positions p0..p0+T-1.</summary>
    public void StoreKv(FV qkv, FV kc, FV vc, int p0, int T, int qCols, int kvCols)
    {
        _kvStore ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, FV, int, int, int, int, int, int, float>(KvStoreImpl);
        _kvStore(T * kvCols, qkv, kc, vc, p0, qCols, kvCols, qCols + 2 * kvCols, 0, 0, 0f);
    }

    private Action<Index1D, FV, FV, FV, int, int, int, int, int, int, float>? _scores;
    private static void ScoresImpl(Index1D i, FV qkv, FV kc, FV scores, int p0, int T, int nk, int H, int D, int rep, float scale)
    {
        int j = i % nk, rest = i / nk, t = rest % T, h = rest / T;
        int qCols = H * D, kvCols = qCols / rep, rowStride = qCols + 2 * kvCols;
        float s = -1e30f;                                                         // masked: a later key
        if (j <= p0 + t)
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
    private Action<Index1D, FV, FV, FV, int, int, int, int, int>? _pv;
    private static void PvImpl(Index1D i, FV scores, FV vc, FV o, int T, int nk, int H, int D, int rep)
    {
        int C = H * D, t = i / C, c = i % C, h = c / D, d = c % D, kvCols = C / rep;
        long so = ((long)h * T + t) * nk;
        float acc = 0f;
        for (int j = 0; j < nk; j++) acc += scores[so + j] * vc[(long)j * kvCols + (h / rep) * D + d];
        o[i] = acc;
    }
    /// <summary>
    /// Causal attention of T new queries (positions p0..p0+T-1; q columns 0..H*D of the fused [T, H*D + 2*kv] projection) over caches 0..p0+T-1 with
    /// <paramref name="rep"/> query heads per key/value head. <paramref name="scores"/> holds H*T*(p0+T) floats; o is [T, H*D].
    /// </summary>
    public void CausalGqa(FV qkv, FV kc, FV vc, FV scores, FV o, int p0, int T, int H, int D, int rep)
    {
        int nk = p0 + T;
        _scores ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, FV, int, int, int, int, int, int, float>(ScoresImpl);
        _softmax ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, int>(SoftmaxImpl);
        _pv ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, FV, int, int, int, int, int>(PvImpl);
        _scores((int)((long)H * T * nk), qkv, kc, scores, p0, T, nk, H, D, rep, 1f / MathF.Sqrt(D));
        _softmax(H * T, scores, nk);
        _pv(T * H * D, scores, vc, o, T, nk, H, D, rep);
    }

    // ───────────────────────────────────────────── small element-wise ─────────────────────────────────────────────

    private Action<Index1D, FV, FV, int>? _swiglu;
    private static void SwigluImpl(Index1D i, FV gateUp, FV o, int I)
    {
        int r = i / I, c = i % I;
        float g = gateUp[(long)r * 2 * I + c], u = gateUp[(long)r * 2 * I + I + c];
        o[i] = g / (1f + MathF.Exp(-g)) * u;
    }
    /// <summary>o[r, c] = silu(gateUp[r, c]) * gateUp[r, I + c] for the fused [rows, 2I] gate/up projection.</summary>
    public void Swiglu(FV gateUp, FV o, int rows, int I)
    {
        _swiglu ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, int>(SwigluImpl);
        _swiglu(rows * I, gateUp, o, I);
    }

    private Action<Index1D, IV, BV, FV, int>? _embed;
    private static void EmbedImpl(Index1D i, IV ids, BV table, FV o, int C)
    {
        int t = i / C, c = i % C;
        o[i] = PrecisionConvert.ConvertToSingle(table[(long)ids[t] * C + c]);
    }
    /// <summary>o[t, :] = embedding table row ids[t].</summary>
    public void Embed(IV ids, BV table, FV o, int count, int C)
    {
        _embed ??= _acc.LoadAutoGroupedStreamKernel<Index1D, IV, BV, FV, int>(EmbedImpl);
        _embed(count * C, ids, table, o, C);
    }

    // ───────────────────────────────────────────── the VAE's point embedding ─────────────────────────────────────────────

    /// <summary>Width of the VAE's per-point input: xyz (3) + phase-shifted sin and cos at 8 frequencies per axis (2 * 3 * 8) + the 3 feature channels (normal).</summary>
    public const int PointWidth = 54;

    /// <summary>
    /// The embedding's two 8-entry tables, as the SHIPPED model holds them. frequencies = 2^f * pi and phase[f] = (8^(1 - (f+1)/8) + (f+1)/8) * 2 pi are non-persistent
    /// buffers computed in float32 at construction, and the shipped model is then converted to bf16, which rounds them to bf16 (pi * 2^7 moves by ~0.1 rad). The
    /// reference's float32 golden keeps those rounded values, so does this port. Layout: [freq 0..7, phase 0..7].
    /// </summary>
    public static readonly float[] PointTables = BuildPointTables();
    private static float BF16Round(float f)
    {
        uint bits = (uint)BitConverter.SingleToInt32Bits(f);
        bits = (bits + 0x7FFFu + ((bits >> 16) & 1u)) >> 16;                 // round to nearest even
        return BitConverter.Int32BitsToSingle((int)(bits << 16));
    }
    private static float[] BuildPointTables()
    {
        var t = new float[16]; float pi = (float)Math.PI;
        for (int f = 0; f < 8; f++)
        {
            t[f] = BF16Round((float)(1 << f) * pi);
            t[8 + f] = BF16Round((MathF.Pow(8f, 1f - (f + 1) / 8f) + (f + 1) / 8f) * ((float)(Math.PI * 2)));
        }
        return t;
    }

    private Action<Index1D, FV, FV, FV, FV>? _pmpe;
    private MemoryBuffer1D<float, Stride1D.Dense>? _pointTables;
    private static void PmpeImpl(Index1D i, FV pts, FV feats, FV o, FV tables)
    {
        int row = i / PointWidth, c = i % PointWidth;
        float v;
        if (c < 3) v = pts[row * 3 + c];
        else if (c < 51)
        {
            bool cos = c >= 27;
            int j = (cos ? c - 27 : c - 3), axis = j / 8, f = j % 8;                  // coordinate-major, 8 frequencies per axis
            float x = pts[row * 3 + axis];
            float e = x * tables[f], p = x * 3.14159274f * 0.5f + tables[8 + f];
            v = cos ? MathF.Cos(e) + MathF.Cos(p) : MathF.Sin(e) + MathF.Sin(p);
        }
        else v = feats[row * 3 + (c - 51)];
        o[i] = v;
    }
    /// <summary>[n,3] points and [n,3] features to the VAE's [n,54] input: the FrequencyPositionalEmbedding with pmpe phases (embed_frequency 8, include_pi), input kept, features appended.</summary>
    public void PointEmbed(FV pts, FV feats, FV o, int n)
    {
        _pmpe ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, FV, FV>(PmpeImpl);
        _pointTables ??= _acc.Allocate1D(PointTables);
        _pmpe(n * PointWidth, pts, feats, o, _pointTables.View);
    }

    private Action<Index1D, FV, IV, FV, int>? _gather;
    private static void GatherImpl(Index1D i, FV src, IV rows, FV o, int C) => o[i] = src[(long)rows[i / C] * C + i % C];
    /// <summary>o[m, :] = src[rows[m], :].</summary>
    public void GatherRows(FV src, IV rows, FV o, int count, int C)
    {
        _gather ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, IV, FV, int>(GatherImpl);
        _gather(count * C, src, rows, o, C);
    }

    // ───────────────────────────────────────────── FSQ and the skin head ─────────────────────────────────────────────

    private Action<Index1D, IV, BV, BV, FV, int, int>? _fsq;
    private static void FsqImpl(Index1D i, IV idx, BV w, BV bias, FV o, int outDim, int levels)
    {
        int t = i / outDim, n = i % outDim;
        // five levels of 8: digit d of the index in base 8 (basis 1, 8, 64, ...), code = (digit - 4) / 4; project_out Linear(5 -> outDim)
        int code = idx[t];
        float acc = PrecisionConvert.ConvertToSingle(bias[n]);
        for (int d = 0; d < 5; d++)
        {
            int digit = code % levels; code /= levels;
            acc += (digit - levels / 2) / (float)(levels / 2) * PrecisionConvert.ConvertToSingle(w[n * 5 + d]);
        }
        o[i] = acc;
    }
    /// <summary>FSQ indices (0..32767) to the VAE's latent vectors [count, outDim]: digits base <paramref name="levels"/>, centred and scaled, then project_out.</summary>
    public void FsqCodes(IV indices, BV projectOutWeight, BV projectOutBias, FV o, int count, int outDim, int levels = 8)
    {
        _fsq ??= _acc.LoadAutoGroupedStreamKernel<Index1D, IV, BV, BV, FV, int, int>(FsqImpl);
        _fsq(count * outDim, indices, projectOutWeight, projectOutBias, o, outDim, levels);
    }

    private Action<Index1D, FV, BV, BV, FV, int, int, int>? _head;
    private static void HeadImpl(Index1D r, FV x, BV w, BV bias, FV o, int C, int column, int columns)
    {
        long b = (long)(int)r * C;
        float acc = PrecisionConvert.ConvertToSingle(bias[0]);
        for (int c = 0; c < C; c++) acc += x[b + c] * PrecisionConvert.ConvertToSingle(w[c]);
        o[(long)(int)r * columns + column] = 1f / (1f + MathF.Exp(-acc));
    }
    /// <summary>skin[r, column] = sigmoid(x[r, :] . w + bias) for r in [0, rows): the VAE decoder's proj_out + sigmoid, written into a column of the [rows0, columns] skin matrix (o already offset to the chunk's first row).</summary>
    public void SkinHead(FV x, BV w, BV bias, FV o, int rows, int C, int column, int columns)
    {
        _head ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, BV, BV, FV, int, int, int>(HeadImpl);
        _head(rows, x, w, bias, o, C, column, columns);
    }
}
