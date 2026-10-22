using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Wan;

using FV = ArrayView1D<float, Stride1D.Dense>;
using BV = ArrayView1D<BFloat16, Stride1D.Dense>;

/// <summary>
/// The kernels of the Wan 2.2 DiT that the TRELLIS.2 flow transformer's <see cref="Trellis2.DiTKernels"/> do not have: the RMS norm over a whole row (Wan
/// normalises q and k across all heads together, TRELLIS per head), RoPE applied from a precomputed phase table, the patch gather and its inverse
/// (the patch embedding is a Conv3d with kernel = stride = (1, 2, 2), i.e. a linear layer on patches), and a broadcast add for the head modulation.
/// Like the DiT kernels these are the correctness lane first: one tensor layout, no tricks.
/// </summary>
public sealed class WanKernels
{
    private readonly Accelerator _acc;
    public WanKernels(Accelerator accelerator) => _acc = accelerator;

    // ---------------------------------------------------------------------------------------------------
    // x[n, off + c] = x[n, off + c] * rsqrt(mean_c(x^2) + eps) * gamma[c]      over C columns of every row, IN PLACE (WanRMSNorm).
    // ---------------------------------------------------------------------------------------------------
    private const int NormGroup = 256;
    private Action<KernelConfig, FV, BV, int, int, int, float>? _rmsGroup;
    private static void RowRmsNormGroupImpl(FV buf, BV gamma, int stride, int off, int C, float eps)
    {
        int row = Grid.IdxX, t = Group.IdxX;
        long b = (long)row * stride + off;
        var sh = SharedMemory.Allocate<float>(NormGroup);
        float s = 0f;
        for (int c = t; c < C; c += NormGroup) { float v = buf[b + c]; s += v * v; }
        sh[t] = s;
        Group.Barrier();
        for (int st = NormGroup / 2; st > 0; st >>= 1) { if (t < st) sh[t] += sh[t + st]; Group.Barrier(); }
        float inv = 1f / MathF.Sqrt(sh[0] / C + eps);
        for (int c = t; c < C; c += NormGroup) buf[b + c] = buf[b + c] * inv * PrecisionConvert.ConvertToSingle(gamma[c]);
    }
    private Action<Index1D, FV, BV, int, int, int, float>? _rms;
    private static void RowRmsNormImpl(Index1D row, FV buf, BV gamma, int stride, int off, int C, float eps)
    {
        long b = (long)(int)row * stride + off;
        float s = 0f;
        for (int c = 0; c < C; c++) { float v = buf[b + c]; s += v * v; }
        float inv = 1f / MathF.Sqrt(s / C + eps);
        for (int c = 0; c < C; c++) buf[b + c] = buf[b + c] * inv * PrecisionConvert.ConvertToSingle(gamma[c]);
    }
    /// <summary>RMS-normalise C columns starting at <paramref name="off"/> of each of <paramref name="rows"/> rows (row stride <paramref name="stride"/>), scaled by the bf16 gamma[C].</summary>
    public void RowRmsNorm(FV buf, BV gamma, int rows, int stride, int off, int C, float eps)
    {
        if (_acc.MaxGroupSize.X >= NormGroup)
        {
            _rmsGroup ??= _acc.LoadStreamKernel<FV, BV, int, int, int, float>(RowRmsNormGroupImpl);
            _rmsGroup(new KernelConfig(rows, NormGroup), buf, gamma, stride, off, C, eps);
            return;
        }
        _rms ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, BV, int, int, int, float>(RowRmsNormImpl);
        _rms(rows, buf, gamma, stride, off, C, eps);
    }

    // ---------------------------------------------------------------------------------------------------
    // RoPE from a phase table: phases = [N, D/2, 2] (cos, sin); every (token, head) vector at buf[n*stride + off + h*D ..) has its adjacent pairs
    // (x[2p], x[2p+1]) rotated, IN PLACE (the complex multiply of rope_apply).
    // ---------------------------------------------------------------------------------------------------
    private Action<Index1D, FV, FV, int, int, int, int>? _rope;
    private static void RopeApplyImpl(Index1D i, FV buf, FV phases, int stride, int off, int H, int D)
    {
        int pairs = D / 2;
        int p = i % pairs, h = (i / pairs) % H, n = i / (pairs * H);
        long b = (long)n * stride + off + h * D + 2 * p;
        long pi = ((long)n * pairs + p) * 2;
        float c = phases[pi], s = phases[pi + 1];
        float x0 = buf[b], x1 = buf[b + 1];
        buf[b] = x0 * c - x1 * s;
        buf[b + 1] = x0 * s + x1 * c;
    }
    public void RopeApply(FV buf, FV phases, int N, int stride, int off, int H, int D)
    {
        _rope ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, int, int, int, int>(RopeApplyImpl);
        _rope(N * H * (D / 2), buf, phases, stride, off, H, D);
    }

    // ---------------------------------------------------------------------------------------------------
    // patch gather: latent [Cin, Fz, h, w] -> patches [N, Cin*4], N = Fz * (h/2) * (w/2), token order (f, i, j), column (c, kh, kw)
    // ---------------------------------------------------------------------------------------------------
    private Action<Index1D, FV, FV, int, int, int, int>? _patchify;
    private static void PatchifyImpl(Index1D i, FV latent, FV patches, int Cin, int Fz, int h, int w)
    {
        int cols = Cin * 4, n = i / cols, col = i % cols;
        int c = col / 4, kh = (col % 4) / 2, kw = col % 2;
        int hp = h / 2, wp = w / 2;
        int f = n / (hp * wp), rem = n % (hp * wp), pi = rem / wp, pj = rem % wp;
        patches[i] = latent[(((long)c * Fz + f) * h + 2 * pi + kh) * w + 2 * pj + kw];
    }
    public void Patchify(FV latent, FV patches, int Cin, int Fz, int h, int w)
    {
        _patchify ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, int, int, int, int>(PatchifyImpl);
        _patchify(Fz * (h / 2) * (w / 2) * Cin * 4, latent, patches, Cin, Fz, h, w);
    }

    // unpatchify: head output [N, 4*Cout] with column ((q*2 + r) * Cout + c) -> latent [Cout, Fz, h, w] (height 2*i + q, width 2*j + r)
    private Action<Index1D, FV, FV, int, int, int, int>? _unpatchify;
    private static void UnpatchifyImpl(Index1D i, FV headOut, FV latent, int Cout, int Fz, int h, int w)
    {
        int x = i % w, y = (i / w) % h, f = (i / (w * h)) % Fz, c = i / (w * h * Fz);
        int hp = h / 2, wp = w / 2;
        int n = f * hp * wp + (y / 2) * wp + x / 2;
        int q = y % 2, r = x % 2;
        latent[i] = headOut[(long)n * (4 * Cout) + (q * 2 + r) * Cout + c];
    }
    public void Unpatchify(FV headOut, FV latent, int Cout, int Fz, int h, int w)
    {
        _unpatchify ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, int, int, int, int>(UnpatchifyImpl);
        _unpatchify(Cout * Fz * h * w, headOut, latent, Cout, Fz, h, w);
    }

    // out[i] = a[i] + b[i % period]   (the head's [2, C] modulation table plus the time embedding e)
    private Action<Index1D, BV, FV, FV, int>? _addBroadcast;
    private static void AddBroadcastImpl(Index1D i, BV a, FV b, FV o, int period) => o[i] = PrecisionConvert.ConvertToSingle(a[i]) + b[i % period];
    public void AddBroadcastBf16(BV a, FV b, FV o, int count, int period)
    {
        _addBroadcast ??= _acc.LoadAutoGroupedStreamKernel<Index1D, BV, FV, FV, int>(AddBroadcastImpl);
        _addBroadcast(count, a, b, o, period);
    }

    // ---------------------------------------------------------------------------------------------------
    // host helpers
    // ---------------------------------------------------------------------------------------------------

    /// <summary>sinusoidal_embedding_1d(freq_dim, t): [cos(t * f_j), sin(t * f_j)], f_j = 10000^(-j / half), computed in double like the reference.</summary>
    public static float[] SinusoidalEmbedding(double t, int dim = 256)
    {
        int half = dim / 2;
        var e = new float[dim];
        for (int j = 0; j < half; j++)
        {
            double a = t * Math.Pow(10000.0, -(double)j / half);
            e[j] = (float)Math.Cos(a);
            e[half + j] = (float)Math.Sin(a);
        }
        return e;
    }

    /// <summary>
    /// The 3-D RoPE phase table of Wan: head dim D split into (frame, height, width) pair counts [D/2 - 2 * (D/2/3), D/2/3, D/2/3] (22, 21, 21 for D = 128),
    /// pair j of an axis with 2m pairs rotating by position * 10000^(-2j / (2m)); tokens in (f, i, j) order. [N, D/2, 2] (cos, sin), angles in double.
    /// </summary>
    public static float[] RopePhases3D(int Fz, int hp, int wp, int D)
    {
        int pairs = D / 2, per = pairs / 3, pf = pairs - 2 * per;     // pair counts: frame, height, width
        int dimF = 2 * pf, dimH = 2 * per, dimW = 2 * per;
        var fF = new double[pf]; var fH = new double[per]; var fW = new double[per];
        for (int j = 0; j < pf; j++) fF[j] = Math.Pow(10000.0, -(2.0 * j) / dimF);
        for (int j = 0; j < per; j++) { fH[j] = Math.Pow(10000.0, -(2.0 * j) / dimH); fW[j] = Math.Pow(10000.0, -(2.0 * j) / dimW); }
        int N = Fz * hp * wp;
        var t = new float[(long)N * pairs * 2];
        int n = 0;
        for (int f = 0; f < Fz; f++)
            for (int i = 0; i < hp; i++)
                for (int j = 0; j < wp; j++, n++)
                {
                    long o = (long)n * pairs * 2;
                    for (int p = 0; p < pairs; p++)
                    {
                        double a = p < pf ? f * fF[p] : p < pf + per ? i * fH[p - pf] : j * fW[p - pf - per];
                        t[o + 2 * p] = (float)Math.Cos(a);
                        t[o + 2 * p + 1] = (float)Math.Sin(a);
                    }
                }
        return t;
    }
}
