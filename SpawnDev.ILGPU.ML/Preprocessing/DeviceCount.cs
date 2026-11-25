using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Preprocessing;

using IV = ArrayView1D<int, Stride1D.Dense>;

/// <summary>
/// Counts the elements of a buffer that satisfy a predicate on the device, with no atomics and no shared memory, so it runs on EVERY backend (WebGL has no Atomic.Max
/// and one store per thread). <see cref="Partials"/> threads stride through the elements and write their own partial, then one thread adds the partials up: the shape
/// <c>ElementWiseKernels.CompareOnGpuAsync</c> uses. Only the one number crosses to the host.
/// </summary>
internal static class DeviceCount
{
    /// <summary>
    /// The launch width of the partial kernel AND its stride: always exactly this many threads, also for fewer elements (the surplus threads count nothing). A kernel launched
    /// with fewer would leave elements nobody visits, and the count would be silently too small.
    /// </summary>
    public const int Partials = 1024;

    private static void FoldImpl(Index1D i, IV partials, IV result)
    {
        int sum = 0;
        for (int p = 0; p < Partials; p++) sum += partials[p];
        result[i] = sum;
    }

    /// <summary>
    /// <paramref name="launchPartials"/> must launch a kernel with exactly <see cref="Partials"/> threads that writes each thread's count of the elements
    /// <c>p, p + Partials, p + 2 * Partials, ...</c> to the view at index <c>p</c>. Returns the total.
    /// </summary>
    public static async Task<int> SumAsync(Accelerator accelerator, Action<IV> launchPartials)
    {
        using var partials = accelerator.Allocate1D<int>(Partials);
        using var total = accelerator.Allocate1D<int>(1);
        launchPartials(partials.View);
        accelerator.LoadAutoGroupedStreamKernel<Index1D, IV, IV>(FoldImpl)(1, partials.View, total.View);
        await accelerator.SynchronizeAsync().ConfigureAwait(false);      // a real drain: the buffers above are disposed after this method
        return (await total.View.CopyToHostAsync().ConfigureAwait(false))[0];
    }
}
