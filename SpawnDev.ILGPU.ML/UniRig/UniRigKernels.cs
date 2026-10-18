using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.UniRig;

using FV = ArrayView1D<float, Stride1D.Dense>;
using IV = ArrayView1D<int, Stride1D.Dense>;

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

    private Action<Index1D, FV, IV, FV, int>? _gather;
    private static void GatherImpl(Index1D i, FV src, IV rows, FV o, int C) => o[i] = src[(long)rows[i / C] * C + i % C];

    /// <summary>o[m, :] = src[rows[m], :] for m in [0, count).</summary>
    public void GatherRows(FV src, IV rows, FV o, int count, int C)
    {
        _gather ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, IV, FV, int>(GatherImpl);
        _gather(count * C, src, rows, o, C);
    }
}
