using ILGPU;
using ILGPU.Runtime;
using FV = ILGPU.Runtime.ArrayView1D<float, ILGPU.Stride1D.Dense>;

namespace SpawnDev.ILGPU.ML;

/// <summary>
/// ComfyUI's IMAGE layout on the device, both ways (rule 4: pictures stay on the GPU): [frames, H, W, 3] channel-last in 0..1 against the VAEs' channel-first
/// [3, frames, H, W] in [-1, 1]. To an IMAGE: clamp((x + 1) / 2, 0, 1) (ComfyUI's VAEDecode); from one: x * 2 - 1 (ComfyUI's VAE.process_input).
/// </summary>
public sealed class ImageTensorKernels
{
    private readonly Accelerator _acc;
    private Action<Index1D, FV, FV, int, int, int>? _to;
    private Action<Index1D, FV, FV, int, int>? _from;

    public ImageTensorKernels(Accelerator accelerator) => _acc = accelerator;

    // one value of the IMAGE per thread: o[(first + f) * pixels + p, c] = clamp((video[c, f, p] + 1) / 2, 0, 1)
    internal static void ToImage01Impl(Index1D i, FV video, FV o, int frames, int pixels, int first)
    {
        int c = i % 3, p = (i / 3) % pixels, f = i / (3 * pixels);
        float q = (video[(long)c * frames * pixels + (long)f * pixels + p] + 1f) * 0.5f;
        o[((long)(first + f) * pixels + p) * 3 + c] = q < 0f ? 0f : (q > 1f ? 1f : q);
    }
    private static void FromImage01Impl(Index1D i, FV image, FV o, int frames, int pixels)
    {
        int c = i % 3, p = (i / 3) % pixels, f = i / (3 * pixels);
        o[(long)c * frames * pixels + (long)f * pixels + p] = image[i] * 2f - 1f;
    }

    /// <summary>
    /// Channel-first [3, <paramref name="frames"/>, pixels] in [-1, 1] into the IMAGE <paramref name="o"/> [.., pixels, 3] in 0..1 at frame <paramref name="first"/>
    /// (a chunk of a longer video lands at its offset).
    /// </summary>
    public void ToImage01(FV video, FV o, int frames, int pixels, int first = 0)
    {
        _to ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, int, int, int>(ToImage01Impl);
        _to(3 * frames * pixels, video, o, frames, pixels, first);
    }

    /// <summary>The IMAGE <paramref name="image"/> [<paramref name="frames"/>, pixels, 3] in 0..1 into channel-first <paramref name="o"/> [3, frames, pixels] in [-1, 1].</summary>
    public void FromImage01(FV image, FV o, int frames, int pixels)
    {
        _from ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, int, int>(FromImage01Impl);
        _from(3 * frames * pixels, image, o, frames, pixels);
    }
}
