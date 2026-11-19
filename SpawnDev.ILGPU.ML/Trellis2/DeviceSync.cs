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

    /// <summary>
    /// Browser: end the current command buffer here (a submit, no wait); desktop: nothing (each launch is its own submission). Called after every block
    /// of a model: Windows resets a GPU whose one submission runs past ~2 s (TDR, DXGI_ERROR_DEVICE_HUNG), and a whole forward in one command buffer
    /// was ~1 s on an RTX 4070 but several seconds on a slower card (2026-11-19, Lore: TRELLIS.2 lost the device on an RX 590 in Chrome). A submit per
    /// block costs microseconds.
    /// </summary>
    public static void Submit(this Accelerator a)
    {
        if (a.IsBrowser()) a.Flush();
    }

    /// <summary>Real completion, every backend (browser: SynchronizeAsync; desktop: the synchronous wait).</summary>
    public static Task WaitAsync(this Accelerator a)
    {
        if (a.IsBrowser()) return a.SynchronizeAsync();
        a.Synchronize();
        return Task.CompletedTask;
    }
}
