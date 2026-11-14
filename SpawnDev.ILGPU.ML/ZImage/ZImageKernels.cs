using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.ZImage;

using FV = ArrayView1D<float, Stride1D.Dense>;
using BV = ArrayView1D<BFloat16, Stride1D.Dense>;

/// <summary>The element-wise kernels Z-Image needs beyond the shared transformer kernels.</summary>
public sealed class ZImageKernels
{
    private readonly Accelerator _acc;
    public ZImageKernels(Accelerator accelerator) => _acc = accelerator;

    private Action<Index1D, FV, int>? _adjustMod;
    private static void AdjustModImpl(Index1D i, FV mod, int dim)
    {
        // (scale_msa, gate_msa, scale_mlp, gate_mlp): scales become 1 + s, gates tanh(g)
        int part = i / dim;
        float v = mod[i];
        mod[i] = (part & 1) == 0 ? 1f + v : MathF.Tanh(v);
    }
    /// <summary>A block's modulation [4 * dim] in place: (1 + scale_msa, tanh(gate_msa), 1 + scale_mlp, tanh(gate_mlp)).</summary>
    public void AdjustMod(FV mod, int dim)
    {
        _adjustMod ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, int>(AdjustModImpl);
        _adjustMod(4 * dim, mod, dim);
    }

    private Action<Index1D, FV, FV, int, int>? _scaleCols;
    private static void ScaleColsImpl(Index1D i, FV x, FV mod, int off, int C) => x[i] *= mod[off + (int)(i % C)];
    /// <summary>x[r, c] *= mod[off + c] over [rows, C].</summary>
    public void ScaleCols(FV x, FV mod, int off, int rows, int C)
    {
        _scaleCols ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, int, int>(ScaleColsImpl);
        _scaleCols(rows * C, x, mod, off, C);
    }

    private Action<Index1D, FV, BV, int, int>? _fillRows;
    private static void FillRowsImpl(Index1D i, FV x, BV token, int rowStart, int C)
        => x[(long)rowStart * C + i] = PrecisionConvert.ConvertToSingle(token[(int)(i % C)]);
    /// <summary>Rows [rowStart, rowStart + rows) of x [*, C] := token [C] (the learned pad token).</summary>
    public void FillRows(FV x, BV token, int rowStart, int rows, int C)
    {
        if (rows <= 0) return;
        _fillRows ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, BV, int, int>(FillRowsImpl);
        _fillRows(rows * C, x, token, rowStart, C);
    }

    private Action<Index1D, FV, FV, int, int, int>? _patchify;
    private static void PatchifyImpl(Index1D idx, FV lat, FV tokens, int rows, int cols, int padded)
    {
        // tokens [padded, 64]: token (y, x) channel (py * 2 + px) * 16 + c = latent c at (2y + py, 2x + px); rows past rows * cols repeat the last token (diffusers' padding)
        int ch = idx % 64, cell = idx / 64;
        int real = rows * cols;
        if (cell >= real) cell = real - 1;
        int y = cell / cols, x = cell % cols, c = ch % 16, p = ch / 16, py = p / 2, px = p % 2;
        tokens[idx] = lat[(c * (2 * rows) + 2 * y + py) * (2 * cols) + 2 * x + px];
    }
    /// <summary>The model-space latent [16, 2 rows, 2 cols] to image tokens [padded, 64] in Z-Image's patch order (ph, pw, c).</summary>
    public void Patchify(FV latent, FV tokens, int rows, int cols, int padded)
    {
        _patchify ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, int, int, int>(PatchifyImpl);
        _patchify(padded * 64, latent, tokens, rows, cols, padded);
    }

    private Action<Index1D, FV, FV, FV, float, int, int>? _denoised;
    private static void DenoisedImpl(Index1D idx, FV x, FV outTokens, FV o, float factor, int rows, int cols)
    {
        // idx over the latent [16, 2 rows, 2 cols]; o = x + factor * output (the output unpatchified from tokens [*, 64] in (ph, pw, c) order)
        int W = 2 * cols, H = 2 * rows;
        int xx = idx % W, rest = idx / W, yy = rest % H, c = rest / H;
        int cell = (yy / 2) * cols + xx / 2, ch = ((yy % 2) * 2 + xx % 2) * 16 + c;
        o[idx] = x[idx] + factor * outTokens[cell * 64 + ch];
    }
    /// <summary>o = x + factor * (the transformer's output, unpatchified): the denoised estimate is x + sigma * output (the model predicts minus the velocity).</summary>
    public void Denoised(FV x, FV outTokens, FV o, float factor, int rows, int cols)
    {
        _denoised ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, FV, float, int, int>(DenoisedImpl);
        _denoised(16 * 4 * rows * cols, x, outTokens, o, factor, rows, cols);
    }
}
