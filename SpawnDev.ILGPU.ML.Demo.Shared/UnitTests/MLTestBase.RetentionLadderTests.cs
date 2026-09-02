using System.Runtime.CompilerServices;
using System.Threading;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// WHERE does the browser-only Context root attach? Diagnostic, not a gate.
/// </summary>
/// <remarks>
/// <para>
/// MEASURED 2026-09-02: every browser lane reports <c>ctxAlive=N/N</c> after Dispose and two full
/// collects, while a desktop probe collects <b>0/10</b> and a weight-matched 1 MiB control collects on
/// the browser too - so it is a REAL root, browser-only, and not conservative-GC behaviour.
/// </para>
/// <para>
/// ⚠️ Every static-holder theory is now DEAD by measurement, not by argument: a reflection census over
/// 353 static fields and delegates across 16 assemblies - SpawnJS, both ILGPU assemblies, ML, and Mono's
/// own <c>System.Runtime.InteropServices.JavaScript</c> - shows nothing growing per test. The only
/// growth was a bounded dispatch ring holding no object references.
/// </para>
/// <para>
/// So this asks WHERE the root attaches instead of WHAT holds it. Each rung builds strictly more than the
/// one below and disposes it; the first rung that is retained is the one that introduces the root, which
/// turns an unbounded search into a bounded one.
/// </para>
/// </remarks>
public abstract partial class MLTestBase
{
    /// <summary>⚠️ NoInlining: an inlined local can stay alive to the end of the caller and read as rooted.</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static WeakReference MakeContext(Action<Context.Builder> build)
    {
        var context = Context.Create(build);
        context.Dispose();
        return new WeakReference(context);
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static WeakReference MakeBareContext() => MakeContext(b => b.Default());

    /// <summary>
    /// Which PART of Context construction introduces the root?
    /// </summary>
    /// <remarks>
    /// A bare <c>Default()</c> Context is retained on all three browser lanes, so the root is inside
    /// construction. <c>Default()</c> enables the CPU accelerator and the default backends on top of the
    /// core (IRContext, TypeContext, ILFrontend, DebugInformationManager). An EMPTY builder constructs the
    /// core and nothing else, so the two together split the search in half:
    /// <list type="bullet">
    ///   <item>empty RETAINED  -> the root is in Context's core machinery.</item>
    ///   <item>empty COLLECTED -> the root comes with what <c>Default()</c> adds.</item>
    /// </list>
    /// ⚠️ ILFrontend runs WORKER THREADS, and a live thread's stack roots everything it references - the
    /// first thing to check if the core is implicated. Desktop collects cleanly (MEASURED 0/10), so
    /// whatever it is, it is something the browser runtime does not tear down.
    /// </remarks>
    [TestMethod(Timeout = 300000)]
    public async Task Diag_ContextConstructionSplit() => await RunPureTest(() =>
    {
        const int rounds = 6;
        var empty = new List<WeakReference>();
        var full = new List<WeakReference>();
        var ctl = new List<WeakReference>();

        for (int i = 0; i < rounds; i++)
        {
            empty.Add(MakeContext(_ => { }));
            full.Add(MakeContext(b => b.Default()));
            ctl.Add(MakeControl());
        }

        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();

        int emptyAlive = empty.Count(w => w.IsAlive);
        int fullAlive = full.Count(w => w.IsAlive);
        int ctlAlive = ctl.Count(w => w.IsAlive);

        Console.WriteLine($"[ML-LADDER] {BackendName} SPLIT empty={emptyAlive}/{rounds} "
                        + $"default={fullAlive}/{rounds} control={ctlAlive}/{rounds}");
        if (ctlAlive > 1)
            Console.WriteLine($"[ML-LADDER] {BackendName} ⚠️ control retained - not interpretable");
        else
            Console.WriteLine($"[ML-LADDER] {BackendName} verdict: "
                + (emptyAlive >= rounds ? "the root is in Context's CORE machinery (IRContext/TypeContext/ILFrontend)"
                   : emptyAlive <= 1 ? "the core is CLEAN - the root comes with what Default() adds"
                   : $"PARTIAL empty={emptyAlive}/{rounds} - investigate"));
        return Task.CompletedTask;
    });

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static WeakReference MakeControl() => new WeakReference(new HeapCensusFatControl());

    /// <summary>
    /// A control that is identical to the fat control EXCEPT that it has a finalizer.
    /// </summary>
    /// <remarks>
    /// ⚠️ THE DIFFERENCE THAT MIGHT EXPLAIN EVERYTHING. A single-threaded WASM runtime has no dedicated
    /// finalizer thread, so <c>GC.WaitForPendingFinalizers()</c> may never actually drain the queue. A
    /// finalizable object then sits in that queue, still reachable, and reads as ALIVE through a
    /// WeakReference forever - while its memory is equally un-reclaimed, which would produce real heap
    /// growth AND a census that says "rooted". <see cref="Context"/> derives from ILGPU's DisposeBase; the
    /// plain fat control has no finalizer, which is exactly why it collects. If THIS control is retained
    /// while the plain one collects, the "root" is finalization, not a reference - a completely different
    /// defect with a completely different fix.
    /// </remarks>
    private sealed class FinalizableFatControl
    {
        public byte[] Buffer = new byte[1024 * 1024];
        ~FinalizableFatControl() { }
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static WeakReference MakeFinalizableControl() => new WeakReference(new FinalizableFatControl());

    /// <summary>
    /// Is the retention a REFERENCE, or is it finalization never draining?
    /// </summary>
    [TestMethod(Timeout = 300000)]
    public async Task Diag_FinalizableControlRetention() => await RunPureTest(() =>
    {
        const int rounds = 6;
        var plain = new List<WeakReference>();
        var finalizable = new List<WeakReference>();
        var ctx = new List<WeakReference>();

        for (int i = 0; i < rounds; i++)
        {
            plain.Add(MakeControl());
            finalizable.Add(MakeFinalizableControl());
            ctx.Add(MakeContext(_ => { }));
        }

        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();

        int plainAlive = plain.Count(w => w.IsAlive);
        int finAlive = finalizable.Count(w => w.IsAlive);
        int ctxAlive = ctx.Count(w => w.IsAlive);

        Console.WriteLine($"[ML-LADDER] {BackendName} FINALIZER plain={plainAlive}/{rounds} "
                        + $"finalizable={finAlive}/{rounds} context={ctxAlive}/{rounds}");
        Console.WriteLine($"[ML-LADDER] {BackendName} verdict: "
            + (finAlive >= rounds && ctxAlive >= rounds
                ? "FINALIZATION, not a reference - a finalizable object is retained exactly like the Context"
               : finAlive <= 1 && ctxAlive >= rounds
                ? "a REAL REFERENCE - finalizable objects collect fine, the Context does not"
                : $"inconclusive (plain={plainAlive} fin={finAlive} ctx={ctxAlive})"));
        return Task.CompletedTask;
    });

    /// <summary>
    /// Environment facts that decide between the remaining theories.
    /// </summary>
    /// <remarks>
    /// ⚠️ Two candidates survive, and both are cheap to test directly rather than argue about.
    /// <list type="number">
    ///   <item><b>ILFrontend worker threads.</b> ILGPU spawns code-generation threads per Context unless
    ///   the platform refuses <c>Thread.Start</c>, in which case it falls back to sync mode. A LIVE thread's
    ///   stack roots everything it references. If threading is unsupported here, no threads exist and the
    ///   theory is dead.</item>
    ///   <item><b><c>Accelerator.Current</c>.</b> A <c>[ThreadStatic]</c> set by <c>Bind()</c> and never
    ///   cleared on disposal. It can only root ONE accelerator per thread, so on its own it cannot explain
    ///   6/6 - but a dangling one after Dispose is still a real defect worth knowing about.</item>
    /// </list>
    /// </remarks>
    [TestMethod(Timeout = 300000)]
    public async Task Diag_RuntimeFacts() => await RunPureTest(() =>
    {
        bool threadingSupported;
        try
        {
            var t = new Thread(() => { });
            t.Start();
            t.Join();
            threadingSupported = true;
        }
        catch (Exception ex) { threadingSupported = false; Console.WriteLine($"[ML-LADDER] thread start refused: {ex.GetType().Name}"); }

        var before = Accelerator.Current?.GetType().Name ?? "<null>";
        var ctx = Context.Create(_ => { });
        var during = Accelerator.Current?.GetType().Name ?? "<null>";
        ctx.Dispose();
        var after = Accelerator.Current?.GetType().Name ?? "<null>";

        Console.WriteLine($"[ML-LADDER] {BackendName} FACTS threading={threadingSupported} "
                        + $"procs={Environment.ProcessorCount} "
                        + $"Accelerator.Current before={before} during={during} afterDispose={after}");
        Console.WriteLine($"[ML-LADDER] {BackendName} verdict: "
            + (!threadingSupported
                ? "no threads on this lane - the ILFrontend worker-thread theory is DEAD"
                : "threads ARE created here - ILFrontend workers remain a live candidate")
            + (after != "<null>" ? "; ⚠️ Accelerator.Current DANGLES after Dispose" : "; Accelerator.Current is clear"));
        return Task.CompletedTask;
    });

    /// <summary>
    /// Is a BARE Context - no device probing, no accelerator, no kernel - already retained?
    /// </summary>
    /// <remarks>
    /// This is the load-bearing question. If a bare Context is retained, the root is in Context
    /// construction itself and nothing about accelerators, adapters or GPU interop is involved - which
    /// would rule out the entire family of theories pursued so far. If it is COLLECTED, the root attaches
    /// somewhere above it and the accelerator path is where to look.
    /// </remarks>
    [TestMethod(Timeout = 300000)]
    public async Task Diag_BareContextRetention() => await RunPureTest(() =>
    {
        const int rounds = 6;
        var ctxRefs = new List<WeakReference>();
        var ctlRefs = new List<WeakReference>();

        for (int i = 0; i < rounds; i++)
        {
            ctxRefs.Add(MakeBareContext());
            ctlRefs.Add(MakeControl());
        }

        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();

        int ctxAlive = ctxRefs.Count(w => w.IsAlive);
        int ctlAlive = ctlRefs.Count(w => w.IsAlive);

        Console.WriteLine($"[ML-LADDER] {BackendName} bare Context: ctxAlive={ctxAlive}/{rounds} "
                        + $"fatControlAlive={ctlAlive}/{rounds}");

        // ⚠️ Reports rather than asserts. The control is the interpretation guard: if the fat control is
        // ALSO retained the collector simply is not freeing large graphs here and the number above means
        // nothing, so say so instead of drawing a conclusion from it.
        if (ctlAlive > 1)
            Console.WriteLine($"[ML-LADDER] {BackendName} ⚠️ control retained {ctlAlive}/{rounds} - "
                            + "the measurement is not interpretable on this lane");
        else
            Console.WriteLine($"[ML-LADDER] {BackendName} verdict: bare Context is "
                            + (ctxAlive >= rounds ? "RETAINED - the root is in Context construction itself"
                               : ctxAlive <= 1 ? "COLLECTED - the root attaches ABOVE a bare Context"
                               : $"PARTIAL ({ctxAlive}/{rounds}) - investigate"));
        return Task.CompletedTask;
    });
}
