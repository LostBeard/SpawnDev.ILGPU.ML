using System.Runtime.CompilerServices;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML;

/// <summary>
/// When SpawnDev.ILGPU.ML frees the compiler caches ILGPU keeps for kernel code (<c>accelerator.ReleaseCompilerCaches()</c>, SpawnDev.ILGPU 5.4.44+):
/// once per model, right after the model's first successful run (its warm-up), when nearly every kernel it will use has been compiled.
/// </summary>
/// <remarks>
/// WHY (measured 2026-11-22/23, MLTestBase.KernelLifetimeModelSwitchTests, SpawnDev.ILGPU 5.4.45): without the release, every distinct model
/// a browser page loads leaves 6.5-16 MiB of compiler state (IR, type information) on the .NET heap after it is disposed. With it, that drops
/// to ~0 on Wasm and ~4 MiB on WebGPU (the accelerator's shader cache, by design). Reloading a released model costs 0 compiles on WebGPU,
/// WebGL and Wasm (ILGPU's shader artifact cache), ~0.3-0.5 s on the first run; warm runs are unchanged.
/// <para>
/// WHEN, and why not more often: a kernel compiled for the first time AFTER a release re-imports the shared code (~5-15 ms for that first
/// kernel). Releasing once after warm-up keeps that to the rare kernel a model loads late; releasing per call or on a timer would make lazy
/// loads pay it again and again (the reason ILGPU never releases on its own).
/// </para>
/// <para>
/// DEFAULT: on for the browser backends (WebGPU, WebGL, Wasm), where memory is the constraint; off for CUDA, OpenCL and CPU, where
/// there is no artifact cache (a reload after a release recompiles every kernel, 1-3.5 s measured on CUDA) and memory is not the constraint.
/// Set <see cref="ReleaseAfterWarmup"/> to force it either way.
/// </para>
/// <para>
/// WHERE ML calls it: <see cref="InferenceSession"/> after its first run (every ONNX pipeline); <see cref="Pipelines.GgufGenerator"/> after
/// its first generate (it defers its session's release so the decode kernels are compiled first); the top-level pipelines of the native
/// ports. An application that assembles a model from components itself (text encoder, transformer, VAE objects) calls
/// <see cref="ModelWarmedUp(Accelerator)"/> once after the model's first complete run.
/// </para>
/// </remarks>
public static class CompilerCachePolicy
{
    /// <summary>
    /// null (default) = release after warm-up on WebGPU, WebGL and Wasm only. true / false = always / never, on every backend.
    /// </summary>
    public static bool? ReleaseAfterWarmup { get; set; }

    /// <summary>Whether a model warming up on <paramref name="accelerator"/> releases the compiler caches under the current setting.</summary>
    public static bool IsEnabledFor(Accelerator accelerator) =>
        ReleaseAfterWarmup ?? accelerator.AcceleratorType is AcceleratorType.WebGPU or AcceleratorType.WebGL or AcceleratorType.Wasm;

    static long _releases;

    /// <summary>How many releases this process has performed through this policy (diagnostic; the tests use it to prove the hooks fire).</summary>
    public static long ReleaseCount => Interlocked.Read(ref _releases);

    /// <summary>
    /// Call once when a model has completed its first full run on <paramref name="accelerator"/>. Releases the compiler caches if the
    /// policy is enabled for that backend. Thread-safe (ILGPU waits for compiles in flight on other threads). Returns whether it released.
    /// </summary>
    public static bool ModelWarmedUp(Accelerator accelerator)
    {
        if (!IsEnabledFor(accelerator)) return false;
        accelerator.ReleaseCompilerCaches();
        Interlocked.Increment(ref _releases);
        return true;
    }

    /// <summary>
    /// For library owners: runs <see cref="ModelWarmedUp(Accelerator)"/> the first time it is called with a given owner flag, then never again
    /// for that owner. Call it after a SUCCESSFUL run only, so a failed first attempt does not count as the warm-up.
    /// </summary>
    internal static void OnFirstRun(ref bool warmedUp, Accelerator accelerator)
    {
        if (warmedUp) return;
        warmedUp = true;
        ModelWarmedUp(accelerator);
    }
}
