using ILGPU;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// A Context created and disposed must not retain browser interop handles.
/// </summary>
/// <remarks>
/// <para>
/// ⚠️ WHAT SHIPPED BROKEN. WebGPU device enumeration calls <c>RequestAdapter</c>, and the resulting
/// <c>GPUAdapter</c> is held by the registered ILGPU Device for the life of the Context. Nothing released
/// it: ILGPU's <c>Device</c> is not IDisposable upstream, because a DESKTOP device is a description - a
/// CUDA ordinal, an OpenCL device id - that owns nothing. A BROWSER device owns a live JS handle. So every
/// <c>Context.Create()</c> abandoned one adapter, pinning real GPU driver resources for the life of the
/// page. MEASURED in the demo: +1 rooted GPUAdapter per test, surviving a forced GC, on the way to the
/// Wasm lane's "Garbage collector could not allocate 16384u bytes" failure.
/// </para>
/// <para>
/// ⚠️ WHY THE GATE COMPARES TWO ROUND COUNTS INSTEAD OF ASSERTING A NUMBER. Startup and first-use caches
/// legitimately add a few permanent slots, so "delta must be zero" would be flaky and "delta under N"
/// would silently tolerate a real per-Context leak as soon as N exceeded the round count. Running the SAME
/// loop at two sizes and requiring the extra rounds to add nothing measures the property that actually
/// matters - growth PER CONTEXT - and is immune to any constant hold.
/// </para>
/// <para>
/// Desktop lanes skip: their devices own no JS handle and there is no slot table to count. That is not a
/// coverage gap, it is the reason the defect was browser-only.
/// </para>
/// <para>
/// ⚠️ A NOTE ON FINDING THIS ONE, because it cost a day. The fix looked correct and changed nothing,
/// through several rebuild-and-remeasure cycles. <c>Context</c> lives in the SpawnDev.ILGPU.Fork
/// assembly, while the package being rebuilt and shipped was the SpawnDev.ILGPU wrapper - which carries
/// the same commit hash in its ProductVersion, so every version check agreed the fix was present. What
/// broke the tie was <c>Context.ContextDisposeTrace</c> printing NOTHING, not even its entry line: a
/// no-op release would still have printed. Turn that flag on before theorising about Context lifetime.
/// </para>
/// </remarks>
public abstract partial class MLTestBase
{
    /// <summary>
    /// The number of live entries in SpawnJS's interop slot table.
    /// </summary>
    /// <remarks>
    /// A slot is released only when its .NET wrapper is disposed, so this counts exactly the handles the
    /// managed side is still holding - which is the thing the leak was made of.
    /// </remarks>
    private static int CountJSSlots(SpawnDev.SpawnJS.SpawnJSRuntime js)
    {
        using var table = js.Get<SpawnDev.SpawnJS.SpawnJSObjectReference>("SpawnJSInterop.spawnJSObjects");
        using var keys = js.Call<SpawnDev.SpawnJS.SpawnJSObjectReference,
                                 SpawnDev.SpawnJS.SpawnJSObjectReference>("Object.keys", table);
        return keys.Get<int>("length");
    }

    /// <summary>
    /// Creating and disposing the backend's real Context+Accelerator pair must not grow the slot table.
    /// </summary>
    [TestMethod(Timeout = 300000)]
    public async Task Context_Dispose_ReleasesBrowserDeviceHandles() => await RunPureTest(async () =>
    {
        var js = SpawnDev.SpawnJS.SpawnJSRuntime.Instance;
        if (js == null || !js.IsBrowser)
            throw new UnsupportedTestException("no JS interop on this lane - desktop devices own no handle");

        // The PRODUCTION path: whatever this backend does to enumerate devices and build an accelerator is
        // exactly what allocated the abandoned adapter. A hand-rolled bare Context would not touch it.
        async Task CycleAsync(int rounds)
        {
            for (int i = 0; i < rounds; i++)
            {
                var (context, accelerator) = await CreateAcceleratorAsync();
                accelerator.Dispose();
                context.Dispose();
            }
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();
            // ⚠️ A slot is released by the wrapper's FINALIZER, and that release crosses into JS. Counting
            // the instant WaitForPendingFinalizers returns catches releases still in flight and reads them
            // as a leak. Settle first, or the measurement invents growth that is not there.
            await Task.Delay(300);
        }

        // Warm: first-use caches are a one-time cost and must not be counted as growth.
        await CycleAsync(1);

        const int small = 2, large = 6;

        int start = CountJSSlots(js);
        await CycleAsync(small);
        int afterSmall = CountJSSlots(js);
        await CycleAsync(large);
        int afterLarge = CountJSSlots(js);

        int deltaSmall = afterSmall - start;
        int deltaLarge = afterLarge - afterSmall;
        double perContext = deltaLarge / (double)large;

        Console.WriteLine($"[SlotGrowth] {BackendName} start={start} +{small} rounds -> {afterSmall} "
                        + $"(delta {deltaSmall}); +{large} rounds -> {afterLarge} (delta {deltaLarge}, "
                        + $"{perContext:F2} per Context)");

        // ⚠️ One slot of slack absorbs an incidental hold; anything that SCALES with the round count is the
        // defect. With the bug this reads ~1.00 per Context and deltaLarge is 6.
        if (deltaLarge > 1)
            throw new Exception(
                $"{BackendName}: {large} Context create/dispose cycles added {deltaLarge} SpawnJS interop "
              + $"slots ({perContext:F2} per Context) after a full GC, while the preceding {small} cycles "
              + $"added {deltaSmall}. Growth that scales with the cycle count is an abandoned browser "
              + "handle - a Device holding a GPUAdapter or a WebGL context that Context.Dispose is not "
              + "releasing. Set ILGPU.Context.ContextDisposeTrace = true to see what Dispose actually did.");
    });
}
