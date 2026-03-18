using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML;

/// <summary>
/// Cross-backend extension methods for GPU buffer readback.
/// Provides CopyToHostAsync(offset, count) that works on all accelerator backends.
/// On WebGPU/WebGL/Wasm: uses SpawnDev's async readback.
/// On CPU/CUDA/OpenCL: uses synchronous CopyToCPU.
/// </summary>
public static class AcceleratorExtensions
{
    /// <summary>
    /// Copy a range of elements from a GPU buffer to a CPU array.
    /// Works on all accelerator backends (CPU, CUDA, OpenCL, WebGPU, WebGL, Wasm).
    /// </summary>
    public static async Task<T[]> CopyToHostAsync<T>(this MemoryBuffer1D<T, Stride1D.Dense> buffer, long offset, long count) where T : unmanaged
    {
        // Use SpawnDev's cross-backend CopyToHostAsync (copies entire buffer)
        // then slice to the requested range
        var all = await SpawnDev.ILGPU.SpawnDevContextExtensions.CopyToHostAsync<T>(buffer);
        if (offset == 0 && count == all.Length)
            return all;
        var result = new T[(int)count];
        Array.Copy(all, offset, result, 0, count);
        return result;
    }

    /// <summary>
    /// Synchronize the accelerator (await all pending GPU work).
    /// Works on all backends.
    /// </summary>
    public static async Task SynchronizeAsync(this Accelerator accelerator)
    {
        await SpawnDev.ILGPU.SpawnDevContextExtensions.SynchronizeAsync(accelerator);
    }
}
