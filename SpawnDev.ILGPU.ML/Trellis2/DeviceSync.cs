using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Trellis2;

/// <summary>
/// The two kinds of "wait" the pipeline needs, made browser-safe. On the browser backends a synchronous wait THROWS (WebGPU's Synchronize
/// is the completion wait, which a single browser thread cannot honour); only readbacks need real completion, and those are awaited.
/// </summary>
internal static class DeviceSync
{
    public static bool IsBrowser(this Accelerator a)
        => a.AcceleratorType is AcceleratorType.WebGPU or AcceleratorType.WebGL or AcceleratorType.Wasm;

    /// <summary>
    /// End of a block of work whose temporaries are about to be released: submit what is queued (WebGPU keeps a released buffer alive until
    /// the submitted work that uses it is done, and a flushed batch never grows past the browser's GPU watchdog); desktop backends also wait,
    /// which keeps per-stage timings meaningful.
    /// </summary>
    public static void Fence(this Accelerator a)
    {
        if (a.IsBrowser()) a.Flush();
        else a.Synchronize();
    }

    /// <summary>Real completion, every backend (browser: SynchronizeAsync; desktop: the synchronous wait).</summary>
    public static Task WaitAsync(this Accelerator a)
    {
        if (a.IsBrowser()) return a.SynchronizeAsync();
        a.Synchronize();
        return Task.CompletedTask;
    }
}
