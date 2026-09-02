using ILGPU;
using ILGPU.Runtime;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// A disposed Accelerator and Context must become COLLECTABLE. This is the gate on the leak that kills
/// long browser runs.
/// </summary>
/// <remarks>
/// <para>
/// ⚠️ WHAT THIS EXISTS TO CATCH. The browser lanes retained every Context and Accelerator ever created -
/// MEASURED by the census in <see cref="MLTestBase.RunTest"/> as <c>ctxAlive=683/683</c> on WebGPU and
/// <c>682/682</c> on WebGL, with a negative control that DID collect, so it is a real root and not GC
/// laziness. The managed heap climbed until the lane died on "Garbage collector could not allocate 16384u
/// bytes of memory for major heap section", which exits the .NET runtime and takes the page with it.
/// </para>
/// <para>
/// ⚠️ THE POINT OF SPLITTING IT IN TWO. The census can only say THAT something is rooted, never WHAT roots
/// it, and a full sweep is an hour per answer. These two tests bisect it in seconds:
/// </para>
/// <list type="bullet">
///   <item><description>
///     <see cref="Accelerator_IsCollectableAfterDispose_NoWork"/> creates and disposes a pair having run
///     NOTHING. A failure here means creation or disposal itself installs the root.
///   </description></item>
///   <item><description>
///     <see cref="Accelerator_IsCollectableAfterDispose_AfterAKernel"/> does the same but dispatches one
///     trivial kernel first. Passing the first and failing this one means the root is installed by RUNNING
///     work - a kernel cache, a param buffer, a captured plan - not by the accelerator's own lifecycle.
///   </description></item>
/// </list>
/// <para>
/// ⚠️ The negative control is not optional. A plain object allocated and referenced by nothing MUST be
/// collected by the same GC calls; if it is not, the WASM GC simply has not cleared weak references yet and
/// every conclusion drawn from the census is void. Without it "still alive" cannot be told from "not
/// collected yet".
/// </para>
/// <para>
/// ⚠️ These deliberately run on <see cref="MLTestBase.RunPureTest"/>, which creates no outer accelerator -
/// so the only pair in play is the one under test.
/// </para>
/// </remarks>
public abstract partial class MLTestBase
{
    /// <summary>Create a pair, dispose it, and report whether it survived a full collect.</summary>
    /// <returns>(acceleratorAlive, contextAlive, controlAlive).</returns>
    private async Task<(bool Accel, bool Ctx, bool Control)> ProbeCollectableAsync(Func<Accelerator, Task>? work)
    {
        WeakReference accelRef, ctxRef, controlRef;

        // Scoped so the strong locals are out of scope before the collect. Without this the JIT may keep
        // them alive to the end of the method and the probe reports a leak that is not there.
        {
            var (context, accelerator) = await CreateAcceleratorAsync();
            try
            {
                if (work != null) await work(accelerator);
                try { await accelerator.SynchronizeAsync(); } catch { }
            }
            finally
            {
                try { accelerator.Dispose(); } catch { }
                try { context.Dispose(); } catch { }
            }
            accelRef = new WeakReference(accelerator);
            ctxRef = new WeakReference(context);
            controlRef = new WeakReference(new object());
        }

        // Two passes: finalizers only QUEUE their objects' memory, so a single collect still reports them.
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();

        return (accelRef.IsAlive, ctxRef.IsAlive, controlRef.IsAlive);
    }

    private static void AssertCollectable(string what, (bool Accel, bool Ctx, bool Control) probe)
    {
        if (probe.Control)
            throw new UnsupportedTestException(
                "the negative control object survived a full collect, so this runtime's GC has not cleared "
              + "weak references yet - the probe cannot distinguish 'rooted' from 'not collected yet' and "
              + "would report a false leak. Skipping rather than asserting on a broken instrument.");

        if (probe.Accel || probe.Ctx)
            throw new Exception(
                $"after Dispose ({what}) the "
              + (probe.Accel && probe.Ctx ? "Accelerator AND Context are" : probe.Accel ? "Accelerator is" : "Context is")
              + " still reachable, while a control object allocated alongside them WAS collected. Something "
              + "holds a strong reference across Dispose. Every such pair retained is ~13 MiB of managed "
              + "heap on a browser lane, and the lane dies at ~650 MiB with \"Garbage collector could not "
              + "allocate 16384u bytes of memory for major heap section\". Suspect anything that outlives "
              + "the accelerator and can hold a delegate closing over it - an interop event subscribed with "
              + "`+=` and never `-=`, or a static cache keyed by accelerator.");
    }

    /// <summary>
    /// A bare ILGPU Context - no browser device registration, no accelerator - must be collectable.
    /// </summary>
    /// <remarks>
    /// ⚠️ THE NEXT BISECT STEP. The no-work test proves the root is installed by CREATION rather than by
    /// running work, and that desktop backends are clean. This narrows it further, because it uses the
    /// IDENTICAL code path on every backend: <c>MLContext.CreateContext()</c>, with no
    /// <c>builder.WebGPU()</c>, no device enumeration and no accelerator.
    ///
    /// If this LEAKS on the browser lanes, the root is in plain Context construction under WASM and nothing
    /// backend-specific is involved. If it COLLECTS, the root is in the browser device registration or the
    /// accelerator itself, and the search shrinks to those two.
    /// </remarks>
    [TestMethod(Timeout = 300000)]
    public async Task Accelerator_IsCollectableAfterDispose_BareContextOnly() => await RunPureTest(() =>
    {
        WeakReference ctxRef, controlRef;
        {
            var context = MLContext.CreateContext();
            context.Dispose();
            ctxRef = new WeakReference(context);
            controlRef = new WeakReference(new object());
        }
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();

        Console.WriteLine($"[AccelLifetime] {BackendName} bare-context: ctxAlive={ctxRef.IsAlive} "
                        + $"controlAlive={controlRef.IsAlive}");

        if (controlRef.IsAlive)
            throw new UnsupportedTestException(
                "the negative control survived a full collect - the probe cannot tell 'rooted' from 'not "
              + "collected yet' on this runtime, so asserting would be measuring the instrument.");
        if (ctxRef.IsAlive)
            throw new Exception(
                "a bare ILGPU Context - created with no browser device registration, no accelerator and no "
              + "work - is still reachable after Dispose while a control object collected. The root is in "
              + "Context construction itself, not in anything backend-specific.");
        return Task.CompletedTask;
    });

    /// <summary>
    /// The final split: a Context built WITHOUT <c>AllAccelerators()</c>.
    /// </summary>
    /// <remarks>
    /// ⚠️ <c>MLContext.Create()</c> is <c>Context.Create().AllAccelerators().EnableAlgorithms()</c>. On a
    /// browser, <c>AllAccelerators()</c> is the one step that touches JS - it probes for WebGPU, WebGL and
    /// Wasm devices. On desktop it registers CPU/CUDA/OpenCL and touches no interop, which is exactly the
    /// split in the measurements: bare Context LEAKS on all three browser lanes and COLLECTS on all three
    /// desktop lanes.
    ///
    /// So: if this passes while <see cref="Accelerator_IsCollectableAfterDispose_BareContextOnly"/> fails,
    /// the root is installed by device PROBING during <c>AllAccelerators()</c> - something created or
    /// subscribed while enumerating browser devices that outlives the Context. If it fails too, the root is
    /// in <c>Context</c> itself under WASM and device discovery is innocent.
    /// </remarks>
    [TestMethod(Timeout = 300000)]
    public async Task Accelerator_IsCollectableAfterDispose_ContextWithoutDeviceProbing() => await RunPureTest(() =>
    {
        WeakReference ctxRef, controlRef;
        {
            var context = Context.Create().EnableAlgorithms().ToContext();
            context.Dispose();
            ctxRef = new WeakReference(context);
            controlRef = new WeakReference(new object());
        }
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();

        Console.WriteLine($"[AccelLifetime] {BackendName} no-probe-context: ctxAlive={ctxRef.IsAlive} "
                        + $"controlAlive={controlRef.IsAlive}");

        if (controlRef.IsAlive)
            throw new UnsupportedTestException("negative control survived - instrument unreliable here.");
        if (ctxRef.IsAlive)
            throw new Exception(
                "a Context created WITHOUT AllAccelerators() is still reachable after Dispose. Device "
              + "probing is innocent - the root is in Context construction itself under WASM.");
        return Task.CompletedTask;
    });

    /// <summary>
    /// Is the probe measuring a root, or just a live LOCAL in an interpreter frame?
    /// </summary>
    /// <remarks>
    /// ⚠️ SUSPECT THE INSTRUMENT, AGAIN - and this time the flaw is mine. In every probe above the Context
    /// is stored in a LOCAL and <c>GC.Collect()</c> runs while that local is still in scope, whereas the
    /// controls are allocated INLINE into <c>new WeakReference(new object())</c> and never occupy a local at
    /// all. Under the WASM interpreter a frame's locals stay live for the whole method; a desktop JIT would
    /// have killed the slot at its last use. So "Context alive, control collected" can be produced by that
    /// asymmetry alone, with nothing rooted anywhere.
    ///
    /// This puts the control on EQUAL footing: same shape, stored in a local, collected in the same method.
    /// If the local control also survives, the earlier readings prove nothing and must be re-derived with
    /// the allocation moved into a separate non-inlined method whose frame is gone before the collect.
    /// </remarks>
    [TestMethod(Timeout = 300000)]
    public async Task Accelerator_ProbeControl_LocalVariableAlsoSurvives() => await RunPureTest(() =>
    {
        WeakReference inlineRef, localRef;
        {
            inlineRef = new WeakReference(new FatControl());
            var heldInALocal = new FatControl();          // the ONLY difference: it occupies a local slot
            localRef = new WeakReference(heldInALocal);
        }
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();

        Console.WriteLine($"[AccelLifetime] {BackendName} local-vs-inline: inlineAlive={inlineRef.IsAlive} "
                        + $"localAlive={localRef.IsAlive}");

        if (localRef.IsAlive && !inlineRef.IsAlive)
            throw new Exception(
                "an object referenced by NOTHING survived a full collect purely because it was stored in a "
              + "local, while an identical object allocated inline was reclaimed. Every ctxAlive reading "
              + "taken with the object in a local is therefore measuring the frame, not a root - including "
              + "the bare-Context probes. Re-derive them with the allocation in a separate non-inlined "
              + "method.");
        return Task.CompletedTask;
    });

    /// <summary>A control that is BIG, to test whether the probe is measuring size rather than roots.</summary>
    private sealed class FatControl
    {
        public byte[] Buffer = new byte[1024 * 1024];
        public object?[] Chain = new object?[2000];
        public FatControl()
        {
            for (int i = 0; i < Chain.Length; i++) Chain[i] = new int[16];
        }
    }

    /// <summary>
    /// Is the probe measuring a real root, or just the conservative WASM GC declining to free big graphs?
    /// </summary>
    /// <remarks>
    /// ⚠️ SUSPECT THE INSTRUMENT. A bare ILGPU Context - no device probing, no accelerator, no work - reads
    /// as rooted on all three BROWSER lanes and collectable on all three DESKTOP lanes, running IDENTICAL
    /// managed code. There is no code path that can do that, which points at the measurement rather than at
    /// a reference.
    ///
    /// Mono's WASM collector is conservative, and a conservative collector falsely retains a LARGE object
    /// graph far more readily than the 24-byte <c>new object()</c> used as the control everywhere else. So
    /// the existing control may simply be too small to be a fair comparison.
    ///
    /// This allocates a control of comparable weight (a 1 MiB array plus 2,000 small objects) and asks the
    /// same question. If the FAT control also survives while the small one collects, every "leak" the census
    /// reports is an artifact of object SIZE, and the real memory story is elsewhere. If the fat control
    /// collects and the Context does not, the Context is genuinely rooted.
    /// </remarks>
    [TestMethod(Timeout = 300000)]
    public async Task Accelerator_ProbeControl_FatObjectAlsoCollects() => await RunPureTest(() =>
    {
        WeakReference smallRef, fatRef;
        {
            smallRef = new WeakReference(new object());
            fatRef = new WeakReference(new FatControl());
        }
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();

        Console.WriteLine($"[AccelLifetime] {BackendName} control-sizes: smallAlive={smallRef.IsAlive} "
                        + $"fatAlive={fatRef.IsAlive}");

        if (fatRef.IsAlive && !smallRef.IsAlive)
            throw new Exception(
                "a LARGE object graph referenced by nothing survived a full collect while a small one was "
              + "reclaimed. The census cannot distinguish 'rooted' from 'too big for this collector to "
              + "prove unreachable' - so every ctxAlive/accelAlive figure measured with a small control is "
              + "suspect, and the retention conclusion drawn from them must be re-derived.");
        if (fatRef.IsAlive)
            throw new UnsupportedTestException("neither control collected - probe unusable on this runtime.");
        return Task.CompletedTask;
    });

    /// <summary>A pair that ran NOTHING must be collectable once disposed.</summary>
    [TestMethod(Timeout = 300000)]
    public async Task Accelerator_IsCollectableAfterDispose_NoWork() => await RunPureTest(async () =>
    {
        var probe = await ProbeCollectableAsync(null);
        Console.WriteLine($"[AccelLifetime] {BackendName} no-work: accelAlive={probe.Accel} "
                        + $"ctxAlive={probe.Ctx} controlAlive={probe.Control}");
        AssertCollectable("having run no work at all", probe);
    });

    /// <summary>And so must one that actually dispatched a kernel.</summary>
    /// <remarks>
    /// Separate from the no-work case on purpose: if this fails and that passes, the root is installed by
    /// running work rather than by the accelerator's own lifecycle, which is a completely different search.
    /// </remarks>
    [TestMethod(Timeout = 300000)]
    public async Task Accelerator_IsCollectableAfterDispose_AfterAKernel() => await RunPureTest(async () =>
    {
        var probe = await ProbeCollectableAsync(async accelerator =>
        {
            using var buf = accelerator.Allocate1D<float>(64);
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>>(
                (i, v) => v[i] = i * 2.0f);
            kernel((int)buf.Length, buf.View);
            await accelerator.SynchronizeAsync();
        });
        Console.WriteLine($"[AccelLifetime] {BackendName} after-kernel: accelAlive={probe.Accel} "
                        + $"ctxAlive={probe.Ctx} controlAlive={probe.Control}");
        AssertCollectable("after dispatching one kernel", probe);
    });
}
