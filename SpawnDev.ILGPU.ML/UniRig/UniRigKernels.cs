using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.UniRig;

using FV = ArrayView1D<float, Stride1D.Dense>;
using IV = ArrayView1D<int, Stride1D.Dense>;
using BV = ArrayView1D<BFloat16, Stride1D.Dense>;

/// <summary>
/// The small kernels UniRig needs beside <see cref="Trellis2.DiTKernels"/> (linear, layer norm, flash attention, residual add): the perceiver's Fourier point
/// embedding and a row gather. One thread per output element, fp32.
/// </summary>
public sealed class UniRigKernels
{
    private readonly Accelerator _acc;
    public UniRigKernels(Accelerator accelerator) => _acc = accelerator;

    /// <summary>Width of the encoder's per-point input: xyz (3) + sin and cos of xyz * 2^k for k = 0..7 (2 * 3 * 8) + normal (3).</summary>
    public const int FourierWidth = 54;

    private Action<Index1D, FV, FV, FV>? _fourier;
    private static void FourierImpl(Index1D i, FV pts, FV nrm, FV o)
    {
        int row = i / FourierWidth, c = i % FourierWidth;
        float v;
        if (c < 3) v = pts[row * 3 + c];
        else if (c < 27) { int j = c - 3; v = MathF.Sin(pts[row * 3 + j / 8] * (float)(1 << (j % 8))); }     // coordinate-major: x*2^0..2^7, y*..., z*...
        else if (c < 51) { int j = c - 27; v = MathF.Cos(pts[row * 3 + j / 8] * (float)(1 << (j % 8))); }
        else v = nrm[row * 3 + (c - 51)];
        o[i] = v;
    }

    /// <summary>[n,3] points and [n,3] normals to [n,54]: the encoder's input features (reference: Fourier embedding with 8 frequencies 2^k, input and normal kept).</summary>
    public void FourierEmbed(FV pts, FV nrm, FV o, int n)
    {
        _fourier ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, FV>(FourierImpl);
        _fourier(n * FourierWidth, pts, nrm, o);
    }

    // ───────────────────────────────────────── OPT decoder kernels ─────────────────────────────────────────

    private Action<Index1D, FV, BV, int, int>? _addPositions;
    private static void AddPositionsImpl(Index1D index, FV x, BV pos, int p0, int C)
    {
        int i = index, t = i / C, c = i % C;
        x[i] += PrecisionConvert.ConvertToSingle(pos[(long)(p0 + t + 2) * C + c]);                    // OPT's learned positions start at table row 2
    }

    /// <summary>x[t, :] += position table row (p0 + t + 2), for t in [0, count). (HF OPTLearnedPositionalEmbedding: offset 2.)</summary>
    public void AddPositions(FV x, BV pos, int p0, int count, int C)
    {
        _addPositions ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, BV, int, int>(AddPositionsImpl);
        _addPositions(count * C, x, pos, p0, C);
    }

    private Action<Index1D, IV, BV, BV, FV, int, int>? _embed;
    private static void EmbedImpl(Index1D i, IV ids, BV table, BV pos, FV o, int p0, int C)
    {
        int t = i / C, c = i % C;
        o[i] = PrecisionConvert.ConvertToSingle(table[(long)ids[t] * C + c]) + PrecisionConvert.ConvertToSingle(pos[(long)(p0 + t + 2) * C + c]);
    }

    /// <summary>o[t, :] = token embedding(ids[t]) + position row (p0 + t + 2).</summary>
    public void Embed(IV ids, BV table, BV pos, FV o, int p0, int count, int C)
    {
        _embed ??= _acc.LoadAutoGroupedStreamKernel<Index1D, IV, BV, BV, FV, int, int>(EmbedImpl);
        _embed(count * C, ids, table, pos, o, p0, C);
    }

    private Action<Index1D, FV>? _relu;
    private static void ReluImpl(Index1D i, FV x) { float v = x[i]; x[i] = v > 0f ? v : 0f; }
    public void Relu(FV x, int count)
    {
        _relu ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV>(ReluImpl);
        _relu(count, x);
    }

    private Action<Index1D, FV, FV, FV, int, int>? _storeKv;
    private static void StoreKvImpl(Index1D i, FV qkv, FV kc, FV vc, int p0, int C)
    {
        int t = i / C, c = i % C;
        kc[(long)(p0 + t) * C + c] = qkv[(long)t * 3 * C + C + c];
        vc[(long)(p0 + t) * C + c] = qkv[(long)t * 3 * C + 2 * C + c];
    }

    /// <summary>Append the K and V columns of a fused [T, 3C] projection (rows [q | k | v]) to the layer's caches at positions p0..p0+T-1.</summary>
    public void StoreKv(FV qkv, FV kc, FV vc, int p0, int T, int C)
    {
        _storeKv ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, FV, int, int>(StoreKvImpl);
        _storeKv(T * C, qkv, kc, vc, p0, C);
    }

    private Action<Index1D, FV, FV, FV, int, int, int, int, int, float>? _scores;
    private static void ScoresImpl(Index1D i, FV qkv, FV kc, FV scores, int p0, int T, int nk, int H, int D, float scale)
    {
        int j = i % nk, rest = i / nk, t = rest % T, h = rest / T, C = H * D;
        float s = -1e30f;                                                         // masked: a later key (causal)
        if (j <= p0 + t)
        {
            float acc = 0f;
            long qo = (long)t * 3 * C + h * D, ko = (long)j * C + h * D;
            for (int d = 0; d < D; d++) acc += qkv[qo + d] * kc[ko + d];
            s = acc * scale;
        }
        scores[i] = s;
    }

    private Action<Index1D, FV, int>? _softmaxRows;
    private static void SoftmaxRowsImpl(Index1D index, FV scores, int nk)
    {
        int row = index;
        long o = (long)row * nk;
        float max = scores[o];
        for (int j = 1; j < nk; j++) max = MathF.Max(max, scores[o + j]);
        float sum = 0f;
        for (int j = 0; j < nk; j++) { float e = MathF.Exp(scores[o + j] - max); scores[o + j] = e; sum += e; }
        float inv = 1f / sum;
        for (int j = 0; j < nk; j++) scores[o + j] *= inv;
    }

    private Action<Index1D, FV, FV, FV, int, int, int, int>? _pv;
    private static void PvImpl(Index1D i, FV scores, FV vc, FV o, int T, int nk, int H, int D)
    {
        int C = H * D, t = i / C, c = i % C, h = c / D;
        long so = ((long)h * T + t) * nk;
        float acc = 0f;
        for (int j = 0; j < nk; j++) acc += scores[so + j] * vc[(long)j * C + c];
        o[i] = acc;
    }

    /// <summary>
    /// Causal multi-head attention of T new queries (absolute positions p0..p0+T-1, taken from the q columns of the fused [T, 3C] projection) over keys/values
    /// 0..p0+T-1 in the layer's caches. <paramref name="scores"/> holds H*T*(p0+T) floats. Materialised scores: prefill is 16 x 1k x 1k, a decode step 16 x 1 x 3k.
    /// </summary>
    public void CausalAttention(FV qkv, FV kc, FV vc, FV scores, FV o, int p0, int T, int H, int D)
    {
        int nk = p0 + T;
        _scores ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, FV, int, int, int, int, int, float>(ScoresImpl);
        _softmaxRows ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, int>(SoftmaxRowsImpl);
        _pv ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, FV, int, int, int, int>(PvImpl);
        _scores((int)((long)H * T * nk), qkv, kc, scores, p0, T, nk, H, D, 1f / MathF.Sqrt(D));
        _softmaxRows(H * T, scores, nk);
        _pv(T * H * D, scores, vc, o, T, nk, H, D);
    }

    private Action<Index1D, FV, IV, FV, int>? _gather;
    private static void GatherImpl(Index1D index, FV src, IV rows, FV o, int C)
    {
        int i = index;
        o[i] = src[(long)rows[i / C] * C + i % C];
    }

    /// <summary>o[m, :] = src[rows[m], :] for m in [0, count).</summary>
    public void GatherRows(FV src, IV rows, FV o, int count, int C)
    {
        _gather ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, IV, FV, int>(GatherImpl);
        _gather(count * C, src, rows, o, C);
    }
}
