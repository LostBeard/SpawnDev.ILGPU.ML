using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Flux1;

using FV = ArrayView1D<float, Stride1D.Dense>;

/// <summary>The few element-wise kernels FLUX.1 needs beyond the shared transformer kernels: the latent's 2 x 2 patches and the denoised estimate.</summary>
public sealed class Flux1Kernels
{
    private readonly Accelerator _acc;
    public Flux1Kernels(Accelerator accelerator) => _acc = accelerator;

    private Action<Index1D, FV, FV>? _addInto;
    private static void AddIntoImpl(Index1D i, FV acc, FV x) => acc[i] += x[i];
    /// <summary>acc += x over <paramref name="count"/> values.</summary>
    public void AddInto(FV acc, FV x, int count)
    {
        _addInto ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV>(AddIntoImpl);
        _addInto(count, acc, x);
    }

    private Action<Index1D, FV, ArrayView1D<BFloat16, Stride1D.Dense>, int, int>? _addRowVector;
    private static void AddRowVectorImpl(Index1D i, FV o, ArrayView1D<BFloat16, Stride1D.Dense> v, int N, int act)
        => o[i] = Trellis2.DiTKernels.Activate(o[i] + PrecisionConvert.ConvertToSingle(v[(int)(i % N)]), act);
    /// <summary>o[r, c] = act(o[r, c] + v[c]) over [rows, N] (a bias's bf16 remainder, before the layer's activation: StreamedModelWeights keeps fp16 biases as hi + lo; act 0 = none).</summary>
    public void AddRowVector(FV o, ArrayView1D<BFloat16, Stride1D.Dense> v, int rows, int N, int act)
    {
        _addRowVector ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, ArrayView1D<BFloat16, Stride1D.Dense>, int, int>(AddRowVectorImpl);
        _addRowVector(rows * N, o, v, N, act);
    }

    private Action<Index1D, FV, FV, int, int>? _patchify;
    private static void PatchifyImpl(Index1D idx, FV lat, FV tokens, int rows, int cols)
    {
        // tokens [rows * cols, 64]: channel c * 4 + py * 2 + px of cell (y, x) = latent channel c at (2y + py, 2x + px)   (einops "c (h ph) (w pw) -> (h w) (c ph pw)")
        int ch = idx % 64, cell = idx / 64, y = cell / cols, x = cell % cols;
        int c = ch / 4, py = (ch / 2) % 2, px = ch % 2;
        tokens[idx] = lat[(c * (2 * rows) + 2 * y + py) * (2 * cols) + 2 * x + px];
    }
    /// <summary>The model-space latent [16, 2 rows, 2 cols] to the transformer's image tokens [rows * cols, 64].</summary>
    public void Patchify(FV latent, FV tokens, int rows, int cols)
    {
        _patchify ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, int, int>(PatchifyImpl);
        _patchify(rows * cols * 64, latent, tokens, rows, cols);
    }

    private Action<Index1D, FV, FV, FV, float, int, int>? _denoised;
    private static void DenoisedImpl(Index1D idx, FV x, FV velocity, FV o, float sigma, int rows, int cols)
    {
        // idx over the latent [16, 2 rows, 2 cols]; the velocity is the transformer's output tokens [rows * cols, 64]
        int W = 2 * cols, H = 2 * rows;
        int xx = idx % W, rest = idx / W, yy = rest % H, c = rest / H;
        int cell = (yy / 2) * cols + xx / 2, ch = c * 4 + (yy % 2) * 2 + xx % 2;
        o[idx] = x[idx] - sigma * velocity[cell * 64 + ch];
    }
    /// <summary>The flow model's denoised estimate, ComfyUI's model output for FLUX.1: o = x - sigma * v, with v unpatchified from the output tokens; all [16, 2 rows, 2 cols].</summary>
    public void Denoised(FV x, FV velocityTokens, FV o, float sigma, int rows, int cols)
    {
        _denoised ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, FV, float, int, int>(DenoisedImpl);
        _denoised(16 * 4 * rows * cols, x, velocityTokens, o, sigma, rows, cols);
    }
}
