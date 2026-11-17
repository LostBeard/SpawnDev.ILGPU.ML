using ILGPU;
using ILGPU.Runtime;
using FV = ILGPU.Runtime.ArrayView1D<float, ILGPU.Stride1D.Dense>;
using BV = ILGPU.Runtime.ArrayView1D<ILGPU.BFloat16, ILGPU.Stride1D.Dense>;

namespace SpawnDev.ILGPU.ML.Ltx2;

/// <summary>
/// LTX-2 kernels the other models do not have (diffusers' LTX-2, Apache-2.0): the "split" rotary embedding with a frequency table per head, the RMS norm without a weight,
/// and the text connectors' register fill.
/// </summary>
public sealed class Ltx2Kernels
{
    private readonly Accelerator _acc;
    public Ltx2Kernels(Accelerator accelerator) => _acc = accelerator;

    private Action<Index1D, FV, FV, int, float>? _rms;
    private static void RmsImpl(Index1D row, FV x, FV o, int C, float eps)
    {
        long b = (long)(int)row * C;
        float ss = 0f;
        for (int c = 0; c < C; c++) { float v = x[b + c]; ss += v * v; }
        float inv = 1f / MathF.Sqrt(ss / C + eps);
        for (int c = 0; c < C; c++) o[b + c] = x[b + c] * inv;
    }
    /// <summary>o[r, :] = x[r, :] / sqrt(mean(x[r, :]^2) + eps)  (torch RMSNorm without elementwise affine; x and o may be the same view).</summary>
    public void RmsNormPlain(FV x, FV o, int rows, int C, float eps)
    {
        _rms ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, int, float>(RmsImpl);
        _rms(rows, x, o, C, eps);
    }

    private Action<Index1D, FV, FV, FV, int, int, int, int>? _rope;
    private static void RopeImpl(Index1D i, FV buf, FV cos, FV sin, int stride, int off, int H, int D)
    {
        int half = D / 2;
        int k = i % half, rest = i / half, h = rest % H, n = rest / H;
        long t = (long)n * H * half + h * half + k;
        float c = cos[t], s = sin[t];
        long b = (long)n * stride + off + h * D;
        float x0 = buf[b + k], x1 = buf[b + half + k];
        buf[b + k] = x0 * c - x1 * s;
        buf[b + half + k] = x1 * c + x0 * s;
    }
    /// <summary>
    /// In place "split" rotary embedding (diffusers' apply_split_rotary_emb): each D-wide head vector of rows [n, stride] from column off is two halves, and pair k of head h
    /// turns by <c>cos[n, h D/2 + k]</c>, <c>sin[..]</c>: tables [rows, H D / 2] (the frequency list cut per head).
    /// </summary>
    public void RopeSplit(FV buf, FV cos, FV sin, int rows, int stride, int off, int H, int D)
    {
        if (cos.Length < (long)rows * H * (D / 2) || sin.Length < (long)rows * H * (D / 2)) throw new ArgumentException($"tables need rows x {H * (D / 2)} values");
        _rope ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, FV, int, int, int, int>(RopeImpl);
        _rope(rows * H * (D / 2), buf, cos, sin, stride, off, H, D);
    }

    private Action<Index1D, FV, BV, FV, int, int, int>? _fill;
    private static void FillImpl(Index1D i, FV input, BV registers, FV o, int n, int C, int R)
    {
        int p = i / C, c = i % C;
        o[i] = p < n ? input[(long)p * C + c] : PrecisionConvert.ConvertToSingle(registers[(long)(p % R) * C + c]);
    }
    /// <summary>o [T, C]: rows 0..n-1 the input's rows, row p &gt;= n the learnable register p mod R (registers [R, C]) — the connectors' padding (diffusers LTX2ConnectorTransformer1d).</summary>
    public void FillRegisters(FV input, BV registers, FV o, int n, int T, int C, int R)
    {
        _fill ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, BV, FV, int, int, int>(FillImpl);
        _fill((int)((long)T * C), input, registers, o, n, C, R);
    }

    /// <summary>
    /// The split-RoPE tables for one axis of normalised positions: frequency j of F is <c>theta^(j / (F - 1)) pi / 2</c> (computed in float64, used as float32), the angle
    /// <c>(2 frac - 1) freq</c> in float32, cos / sin evaluated accurately; <paramref name="pad"/> leading entries per row are cos 1 / sin 0. Rows [rows, pad + F].
    /// </summary>
    public static (float[] Cos, float[] Sin) SplitRopeTables(float[] frac, int F, int pad, double theta)
    {
        var freq = new float[F];
        for (int j = 0; j < F; j++) freq[j] = (float)(Math.Pow(theta, F == 1 ? 0.0 : (double)j / (F - 1)) * Math.PI / 2.0);
        int W = pad + F, rows = frac.Length;
        var c = new float[rows * W]; var s = new float[rows * W];
        for (int r = 0; r < rows; r++)
        {
            for (int j = 0; j < pad; j++) { c[r * W + j] = 1f; s[r * W + j] = 0f; }
            float g = frac[r] * 2f - 1f;
            for (int j = 0; j < F; j++) { float a = g * freq[j]; c[r * W + pad + j] = (float)Math.Cos(a); s[r * W + pad + j] = (float)Math.Sin(a); }
        }
        return (c, s);
    }
}
