using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// Abstract base class for ML kernel tests.
/// Each backend (WebGPU, CPU, etc.) inherits and overrides CreateAcceleratorAsync().
/// Tests use [TestMethod] from SpawnDev.UnitTesting for Playwright discovery.
/// </summary>
public abstract partial class MLTestBase : IDisposable
{
    protected abstract Task<(Context context, Accelerator accelerator)> CreateAcceleratorAsync();
    protected abstract string BackendName { get; }

    /// <summary>
    /// HttpClient for loading models and test data via HTTP.
    /// Browser subclasses inject from DI. Desktop subclasses get it automatically
    /// from the TEST_SERVER_URL environment variable set by PlaywrightMultiTest.
    /// </summary>
    protected virtual System.Net.Http.HttpClient? GetHttpClient() => GetEnvHttpClient();

    private static System.Net.Http.HttpClient? _envHttpClient;
    private static bool _envHttpClientChecked;

    private static System.Net.Http.HttpClient? GetEnvHttpClient()
    {
        if (_envHttpClientChecked) return _envHttpClient;
        _envHttpClientChecked = true;
        var serverUrl = Environment.GetEnvironmentVariable("TEST_SERVER_URL");
        if (!string.IsNullOrEmpty(serverUrl))
        {
            var handler = new System.Net.Http.HttpClientHandler
            {
                ServerCertificateCustomValidationCallback = (_, _, _, _) => true
            };
            _envHttpClient = new System.Net.Http.HttpClient(handler) { BaseAddress = new Uri(serverUrl) };
        }
        return _envHttpClient;
    }

    /// <summary>
    /// The DI-registered <see cref="SpawnDev.WebTorrent.WebTorrentClient"/> the DEMO uses (OPFS-backed in the
    /// browser, so the zero-copy web-seed download path fires). Browser subclasses override to return the
    /// injected singleton; returns null elsewhere (desktop / lanes without it) so download-measurement tests
    /// skip rather than measure a different, non-demo client.
    /// </summary>
    protected virtual SpawnDev.WebTorrent.WebTorrentClient? GetWebTorrentClient() => null;

    /// <summary>The DI-registered OPFS file system the demo's WebTorrentClient persists to. Browser subclasses
    /// override to return the injected singleton; null elsewhere. Lets a download measurement clear stale
    /// cross-run OPFS state before a clean cold run.</summary>
    protected virtual SpawnDev.AsyncFileSystem.IAsyncFS? GetAsyncFS() => null;

    // The most recent invocation's pair, for ZOMBIE EVICTION only (see RunTest).
    private Accelerator? _prevAccelerator;
    private Context? _prevContext;

    /// <summary>
    /// Run a test that needs NO GPU - it creates no Context and no Accelerator.
    /// </summary>
    /// <remarks>
    /// <para>
    /// ⚠️ WHY THIS EXISTS, and it is not tidiness. <b>61 tests</b> in this suite are written
    /// <c>RunTest(_ =&gt; ...)</c> - the discard PROVES the body cannot touch the accelerator - and every one
    /// of them still built a full Context + Accelerator pair. Phonemizer, text normalisation,
    /// letter-to-sound, homographs, tokenizers, word decomposition, the ZipVoice reference trim: string and
    /// array work, no device involved.
    /// </para>
    /// <para>
    /// ⚠️ That is not free, because the browser backends RETAIN every Context and Accelerator ever created
    /// (MEASURED by the census in <see cref="RunTest"/>: <c>ctxAlive</c> tracks the test count 1:1, with the
    /// negative control collecting normally). At the ~13 MiB per test that census reports, 61 unnecessary
    /// pairs is on the order of <b>hundreds of MiB</b> of managed heap spent on tests that never dispatch a
    /// kernel - on the lane whose ceiling is ~630 MiB, where the run dies with "Garbage collector could not
    /// allocate 16384u bytes of memory for major heap section".
    /// </para>
    /// <para>
    /// ⚠️ Use this ONLY where the body genuinely needs no accelerator. It deliberately takes no
    /// <see cref="Accelerator"/>, so misuse cannot compile. The heap trend line is skipped too: a test that
    /// allocates no device pair contributes nothing to the curve, and printing a flat line per pure test
    /// would bury the tests that do.
    /// </para>
    /// </remarks>
    protected async Task RunPureTest(Func<Task> testBody)
    {
        // Still evicts a previous invocation's zombie pair, for the same reason RunTest does: a timed-out
        // predecessor must not keep a full accelerator alive alongside the next test.
        try { _prevAccelerator?.Dispose(); } catch { }
        try { _prevContext?.Dispose(); } catch { }
        _prevAccelerator = null;
        _prevContext = null;
        ResetStaticCaptureState();
        await testBody();
    }

    protected async Task RunTest(Func<Accelerator, Task> testBody,
        [System.Runtime.CompilerServices.CallerMemberName] string? testName = null)
    {
        // Each invocation OWNS its context + accelerator as LOCALS - never a shared
        // cache. The runner abandons a timed-out test and moves on, but the abandoned
        // (zombie) continuation still reaches this finally LATER; the previous
        // shared-field cache's cleanup disposed whatever the field held by then -
        // the NEXT test's LIVE accelerator/context. That was the 4.10.0 Wasm cascade:
        // ObjectDisposedException(WasmAccelerator) mid-RunKernelAsync and
        // ObjectDisposedException(ReaderWriterLockSlim) inside ILGPU
        // TypeInformationManager mid-compile, always right after a timeout, plus the
        // follow-on timeouts it knocked over. A zombie may now only dispose its own
        // pair. (Disposal per test also keeps a prior browser row's device state from
        // leaking into the next backend lane - that behavior is unchanged.)
        //
        // ZOMBIE EVICTION: dispose the PREVIOUS invocation's pair up front. A
        // timed-out-but-still-running zombie otherwise keeps a full accelerator (on
        // Wasm: a hardwareConcurrency worker pool) alive next to the new test's one -
        // pool pile-up oversubscribes the CPU and cascades the heavy tests into
        // timeouts. Eviction is deterministic (next test's START, before it creates
        // its own pair - never from the zombie's finally), so it cannot dispose a
        // live successor; the zombie's next dispatch hits the disposal guard and its
        // abandoned task faults harmlessly (its outcome was already recorded as the
        // timeout). Invocations on one instance are sequential by runner design.
        try { _prevAccelerator?.Dispose(); } catch { }
        try { _prevContext?.Dispose(); } catch { }
        _prevAccelerator = null;
        _prevContext = null;

        // STATIC CAPTURE-STATE EVICTION (memory-cascade guard): the GraphExecutor diagnostic
        // captures (CapturedOutputs/NodeInfo/NodeTimingsMs) and NormalizationKernels' captured GPU
        // buffers are opt-in STATICS. A capture-enabled test that TIMES OUT never reaches its
        // finally that nulls them, so they stay non-null — and then EVERY subsequent test's
        // GraphExecutor.Run appends its per-node tensors into the leaked dict unboundedly. In a long
        // sequential lane (the Wasm phase) that climbs until a heavy model (DistilGPT2) hits
        // OutOfMemory and the tests after it time out. Reset deterministically at each test's START
        // (a leaked-but-still-running zombie's later finally cannot resurrect it) so capture state
        // never crosses a test boundary. A test that legitimately wants capture sets it AFTER this.
        ResetStaticCaptureState();

        var (context, accelerator) = await CreateAcceleratorAsync();
        _prevAccelerator = accelerator;
        _prevContext = context;
        try
        {
            await testBody(accelerator);
        }
        finally
        {
            // Flush GPU work before disposal so pending dispatches complete.
            try { await accelerator.SynchronizeAsync(); } catch { }
            // Drop this accelerator's AssertCloseGpu kernel cache entry - the static
            // dictionary would otherwise grow one (kernels + dead accelerator key) per
            // test for the whole run.
            lock (_ewCache)
            {
                if (_ewCache.Remove(accelerator, out var ew))
                    try { (ew as IDisposable)?.Dispose(); } catch { }
            }
            try { accelerator.Dispose(); } catch { }
            try { context.Dispose(); } catch { }
            // Deregister from zombie eviction IF still ours - a zombie reaching this
            // line later must not clear the SUCCESSOR's registration.
            if (ReferenceEquals(_prevAccelerator, accelerator))
            {
                _prevAccelerator = null;
                _prevContext = null;
            }
            GC.Collect();
            GC.WaitForPendingFinalizers();
            // Second pass: the finalizers above only QUEUE their objects' memory for reclaim, so a single
            // collect reports them as still live and the trend line below would over-read by a whole test's
            // worth of finalizable interop wrappers.
            GC.Collect();

            // ── MANAGED-HEAP TREND (one line per test, always on) ──
            // The Wasm lane dies ~800 tests in with "Garbage collector could not allocate 16384u bytes of
            // memory for major heap section" - the MONO GC failing to grow the .NET managed heap. That heap
            // is invisible to `performance.memory.usedJSHeapSize` (V8 objects) and to a CDP JS-heap snapshot,
            // which is why the 2026-06-15 investigation - which chased exactly those - could not see it.
            // C# objects/byte[]/strings live HERE.
            //
            // ⚠️ This is the LIVE set after a full collect, deliberately, not a cumulative total: a monotonic
            // counter always climbs and can never evidence accumulation. See the 2026-06-14/15 day lost to
            // reading `TotalKernelsCompiled` (a next-id) as cache growth while the only real memory signal
            // was flat. If this number is flat, there is no managed leak, whatever else is rising.
            //
            // PMT drops ordinary browser console lines, so read it with PMT_CONSOLE_LOG=ML-HEAP.
            try
            {
                // ⚠️ LIVE-OBJECT census, not an event counter. Each test's Context+Accelerator are registered
                // as WEAK references AFTER they are disposed and collected; anything still reported alive is
                // being ROOTED by something. A climbing alive-count is direct evidence of what is retained,
                // where a "created so far" total would climb no matter what and prove nothing (2026-06-14/15).
                _ctxRefs.Add(new WeakReference(context));
                _accelRefs.Add(new WeakReference(accelerator));
                // ⚠️ NEGATIVE CONTROL. A plain object allocated here and referenced by nothing MUST be
                // collected. If ctlAlive climbs alongside ctxAlive then the census is measuring the WASM GC's
                // reluctance to clear weak references, not a leak - and every conclusion drawn from it is
                // void. Without this the instrument cannot tell "rooted" from "not collected yet".
                _controlRefs.Add(new WeakReference(new object()));
                // ⚠️ WEIGHT-MATCHED CONTROL. The 24-byte control above proves the WASM GC clears weak
                // references promptly, but it CANNOT prove a large graph would be collected: Mono's
                // collector is conservative, and a conservative collector falsely retains a big object
                // graph far more readily than a tiny one. A Context IS a big graph. Without this,
                // "ctxAlive=N/N while ctlAlive=1/N" is equally consistent with a real root and with
                // size-dependent GC behaviour - and those have completely different fixes.
                //   fatAlive ~ 1/N  => the collector DOES free large unreferenced graphs, so ctxAlive=N/N
                //                      is a REAL root and worth hunting.
                //   fatAlive ~ N/N  => the census cannot distinguish, and every "leak" it has reported
                //                      is uninterpretable until the instrument is replaced.
                _fatControlRefs.Add(new WeakReference(new HeapCensusFatControl()));
                int fatAlive = 0; foreach (var w in _fatControlRefs) if (w.IsAlive) fatAlive++;
                int ctxAlive = 0; foreach (var w in _ctxRefs) if (w.IsAlive) ctxAlive++;
                int accAlive = 0; foreach (var w in _accelRefs) if (w.IsAlive) accAlive++;
                int ctlAlive = 0; foreach (var w in _controlRefs) if (w.IsAlive) ctlAlive++;

                long live = GC.GetTotalMemory(forceFullCollection: false);
                var gcInfo = GC.GetGCMemoryInfo();
                Console.WriteLine($"[ML-HEAP] {BackendName} #{++_heapTraceIndex} {testName ?? "?"} "
                    + $"live={live / 1048576.0:F1}MiB committed={gcInfo.TotalCommittedBytes / 1048576.0:F1}MiB "
                    + $"heap={gcInfo.HeapSizeBytes / 1048576.0:F1}MiB gen2={GC.CollectionCount(2)} "
                    + $"ctxAlive={ctxAlive}/{_ctxRefs.Count} accelAlive={accAlive}/{_accelRefs.Count} "
                    + $"ctlAlive={ctlAlive}/{_controlRefs.Count} fatAlive={fatAlive}/{_fatControlRefs.Count}");
                Console.WriteLine($"[ML-REG] {BackendName} #{_heapTraceIndex} {InteropRegistrySizes()}");
            }
            catch { /* a diagnostic must never fail a test */ }
        }
    }

    /// <summary>Sequence number for the <c>[ML-HEAP]</c> trend line - the x-axis of the growth curve.</summary>
    private int _heapTraceIndex;

    // Weak references to every disposed Context/Accelerator, for the alive-census above. Weak, so holding
    // them cannot itself be the leak; a WeakReference is a few bytes against a ~0.9 MiB/test growth rate.
    private static readonly List<WeakReference> _ctxRefs = new();
    private static readonly List<WeakReference> _accelRefs = new();
    private static readonly List<WeakReference> _controlRefs = new();
    private static readonly List<WeakReference> _fatControlRefs = new();

    /// <summary>
    /// Sizes every static collection in the SpawnJS assembly, so a registry that GROWS PER TEST names
    /// itself.
    /// </summary>
    /// <remarks>
    /// ⚠️ The retention is browser-only across WebGPU, WebGL and Wasm running IDENTICAL managed code, and
    /// the one layer that exists only there is SpawnJS interop. SpawnJS keeps delegates in static
    /// registries (<c>Callback._callbacks</c>, <c>EventTarget.CallBackInfos</c>,
    /// <c>ActionExtensions._callbacks</c>, <c>FuncExtensions._callbacks</c>) and a delegate captures its
    /// target - so a subscription that is never released roots the Accelerator and, through it, the
    /// Context. That is a hypothesis; this measures it instead of arguing about it.
    ///
    /// Scanned by reflection rather than named explicitly: naming them means guessing the four that
    /// matter, and the point is to let the data pick. A count that tracks the test index IS the holder.
    /// Diagnostics must never fail a test, so every step is guarded.
    /// </remarks>
    /// <summary>
    /// Every static collection that could hold a reference, sized once per test, so a registry that GROWS
    /// PER TEST names itself.
    /// </summary>
    /// <remarks>
    /// ⚠️ Scanned by reflection rather than named explicitly - naming them means guessing which ones
    /// matter, and the point is to let the data pick. A count that tracks the test index IS the holder.
    ///
    /// MEASURED 2026-09-02: this cleanly EXONERATED SpawnJS. Across six WebGPU tests every SpawnJS static
    /// was flat (<c>Callback._callbacks=2</c> throughout), which killed the standing hypothesis that a
    /// static delegate registry was rooting the Accelerator. Scope now includes the ILGPU assemblies,
    /// because that hypothesis is dead and the holder is still unidentified.
    ///
    /// The member list is built ONCE - ILGPU is thousands of types and re-scanning per test would show up
    /// as a slowdown that looks like a leak in its own right. Diagnostics must never fail a test, so every
    /// step is guarded.
    /// </remarks>
    private static List<(string Label, Func<int> Count)>? _registryProbes;
    private static string _scannedAssemblies = "";

    private static List<(string Label, Func<int> Count)> BuildRegistryProbes()
    {
        var probes = new List<(string, Func<int>)>();
        var flags = System.Reflection.BindingFlags.Static
                  | System.Reflection.BindingFlags.Public
                  | System.Reflection.BindingFlags.NonPublic;

        // ⚠️ EVERY loaded assembly, not a hand-picked list. Naming assemblies is the same guess as naming
        // fields: SpawnJS was the obvious suspect and measured FLAT, and the retention is browser-only
        // (MEASURED 2026-09-02: a desktop probe collects 0/10 Contexts while every browser lane holds
        // N/N). The browser runtime is Mono with its own JS-interop tables in
        // System.Runtime.InteropServices.JavaScript, which root .NET objects handed to JS - and that is
        // exactly the kind of holder a curated list would miss.
        // ⚠️ BOUNDED. Scanning every loaded assembly HUNG the Wasm runtime outright - System.Private.CoreLib
        // alone is tens of thousands of members and the first test sat at "running" forever. Restricted to
        // the assemblies that can plausibly hold a reference to a Context: ours, plus Mono's JS-interop
        // assembly, which roots .NET objects handed to JS and is the reason this is browser-only.
        var wanted = new[]
        {
            "SpawnDev", "ILGPU", "System.Runtime.InteropServices.JavaScript",
        };
        var assemblies = new List<System.Reflection.Assembly>();
        try
        {
            foreach (var asm in AppDomain.CurrentDomain.GetAssemblies())
            {
                var name = asm.GetName().Name ?? "";
                if (wanted.Any(w => name.StartsWith(w, StringComparison.OrdinalIgnoreCase)))
                    assemblies.Add(asm);
            }
        }
        catch { }
        // ⚠️ Report WHAT WAS SCANNED. The scope was widened specifically to reach Mono's JS-interop
        // assembly; concluding "nothing roots it" while that assembly was silently absent would be a
        // false negative dressed as a result.
        _scannedAssemblies = string.Join(",", assemblies.Select(a => a.GetName().Name));

        foreach (var asm in assemblies)
        {
            Type[] types;
            try { types = asm.GetTypes(); }
            catch (System.Reflection.ReflectionTypeLoadException ex) { types = ex.Types.Where(t => t != null).ToArray()!; }
            catch { continue; }

            foreach (var t in types)
            {
                if (t == null || t.IsGenericTypeDefinition) continue;
                // ⚠️ FIELDS ONLY. Reading a static PROPERTY invokes its getter, which is arbitrary code -
                // it can block, throw, or do real work. An earlier version of this probe read properties
                // too and HUNG the Wasm runtime: the first test sat at "running" indefinitely. A field read
                // is inert and cannot have side effects.
                foreach (var f in t.GetFields(flags))
                {
                    if (f.FieldType.IsValueType || f.FieldType == typeof(string)) continue;
                    bool isDelegate = typeof(Delegate).IsAssignableFrom(f.FieldType);
                    bool isCollection = typeof(System.Collections.ICollection).IsAssignableFrom(f.FieldType)
                                     || f.FieldType == typeof(object);
                    if (!isDelegate && !isCollection) continue;
                    var label = $"{t.Name}.{f.Name}";
                    probes.Add((label, () =>
                    {
                        try
                        {
                            var v = f.GetValue(null);
                            // ⚠️ A MULTICAST DELEGATE IS NOT AN ICollection. A static event accumulates
                            // handlers in a delegate field and EACH HANDLER ROOTS ITS TARGET, so a
                            // collection-only probe is blind to exactly the shape that leaks here. Counting
                            // the invocation list is what makes a never-unsubscribed handler visible.
                            if (v is Delegate d) return d.GetInvocationList().Length;
                            if (v is System.Collections.ICollection c) return c.Count;
                            return -1;
                        }
                        catch { return -1; }
                    }));
                }
            }
        }
        return probes;
    }

    /// <summary>Only reports collections that are NON-EMPTY, so the line stays readable.</summary>
    private static string InteropRegistrySizes()
    {
        try
        {
            _registryProbes ??= BuildRegistryProbes();
            var parts = new List<string>();
            foreach (var (label, count) in _registryProbes)
            {
                int n;
                try { n = count(); } catch { continue; }
                if (n > 0) parts.Add($"{label}={n}");
            }
            parts.Sort();
            return $"asm=[{_scannedAssemblies}] " + string.Join(" ", parts);
        }
        catch (Exception ex) { return $"(registry scan failed: {ex.GetType().Name})"; }
    }


    /// <summary>
    /// A control of comparable WEIGHT to a Context, referenced by nothing, so the census can tell a real
    /// root from a conservative collector declining to free a large graph.
    /// </summary>
    private sealed class HeapCensusFatControl
    {
        public byte[] Buffer = new byte[1024 * 1024];
        public object?[] Chain = new object?[2000];
        public HeapCensusFatControl()
        {
            for (int i = 0; i < Chain.Length; i++) Chain[i] = new int[16];
        }
    }

    public virtual void Dispose()
    {
        try { _prevAccelerator?.Dispose(); } catch { }
        _prevAccelerator = null;
        try { _prevContext?.Dispose(); } catch { }
        _prevContext = null;
        ResetStaticCaptureState();
    }

    /// <summary>
    /// Null out the opt-in GraphExecutor / NormalizationKernels static diagnostic-capture state so
    /// it can never leak across a test boundary. A capture-enabled test that times out skips its own
    /// finally; without this, the leaked dict keeps accumulating every later test's per-node tensors
    /// (and the instance-norm capture keeps live GPU buffers), which is what turned the long Wasm
    /// lane into a DistilGPT2 OutOfMemory + cascade of follow-on timeouts. Disposes the captured
    /// GPU buffers before dropping them.
    /// </summary>
    private static void ResetStaticCaptureState()
    {
        try { Graph.GraphExecutor.CapturedOutputs = null; } catch { }
        try { Graph.GraphExecutor.CapturedNodeInfo = null; } catch { }
        try { Graph.GraphExecutor.CapturedNodeTimingsMs = null; } catch { }
        try
        {
            var inorm = Kernels.NormalizationKernels.CapturedInstanceNormPass1Outputs;
            if (inorm != null)
            {
                foreach (var e in inorm)
                {
                    try { e.means?.Dispose(); } catch { }
                    try { e.invStds?.Dispose(); } catch { }
                }
                Kernels.NormalizationKernels.CapturedInstanceNormPass1Outputs = null;
            }
        }
        catch { }
    }

    #region Helpers

    protected static float[] RandomFloats(int count, int seed = 42, float scale = 1f)
    {
        var rng = new Random(seed);
        var data = new float[count];
        for (int i = 0; i < count; i++)
            data[i] = (float)(rng.NextDouble() * 2 - 1) * scale;
        return data;
    }

    protected static float[] CpuMatMul(float[] A, float[] B, int M, int K, int N)
    {
        var C = new float[M * N];
        for (int r = 0; r < M; r++)
            for (int c = 0; c < N; c++)
            {
                float s = 0;
                for (int k = 0; k < K; k++)
                    s += A[r * K + k] * B[k * N + c];
                C[r * N + c] = s;
            }
        return C;
    }

    protected static float[] CpuLayerNorm(float[] input, float[] gamma, float[] beta, int rows, int C, float eps = 1e-6f)
    {
        var output = new float[rows * C];
        for (int r = 0; r < rows; r++)
        {
            float sum = 0;
            for (int i = 0; i < C; i++) sum += input[r * C + i];
            float mean = sum / C;
            float varSum = 0;
            for (int i = 0; i < C; i++) { float d = input[r * C + i] - mean; varSum += d * d; }
            float invStd = 1f / MathF.Sqrt(varSum / C + eps);
            for (int i = 0; i < C; i++)
                output[r * C + i] = gamma[i] * ((input[r * C + i] - mean) * invStd) + beta[i];
        }
        return output;
    }

    /// <summary>Abramowitz & Stegun erf approximation (max error ~1.5e-7). Matches ElementWiseKernels.GELUImpl.</summary>
    protected static float ErfApprox(float z)
    {
        float az = z < 0 ? -z : z;
        const float p = 0.3275911f;
        const float a1 = 0.254829592f, a2 = -0.284496736f, a3 = 1.421413741f, a4 = -1.453152027f, a5 = 1.061405429f;
        float t = 1f / (1f + p * az);
        float t2 = t * t, t3 = t2 * t, t4 = t3 * t, t5 = t4 * t;
        float erfAbs = 1f - (a1 * t + a2 * t2 + a3 * t3 + a4 * t4 + a5 * t5) * MathF.Exp(-az * az);
        return z < 0 ? -erfAbs : erfAbs;
    }

    protected static void AssertClose(float[] expected, float[] actual, float tolerance, string label = "")
    {
        if (expected.Length != actual.Length)
            throw new Exception($"{label}Length mismatch: expected={expected.Length}, actual={actual.Length}");
        float maxErr = 0;
        int worstIdx = -1;
        for (int i = 0; i < expected.Length; i++)
        {
            float err = MathF.Abs(expected[i] - actual[i]);
            if (err > maxErr) { maxErr = err; worstIdx = i; }
        }
        if (maxErr > tolerance)
            throw new Exception($"{label}Max error {maxErr:E3} at [{worstIdx}]: expected={expected[worstIdx]:F6}, actual={actual[worstIdx]:F6} (tol={tolerance:E1})");
    }

    /// <summary>
    /// GPU-side AssertClose — uploads expected to GPU, compares on GPU, reads back only 2 floats.
    /// Use instead of CopyToHostAsync + AssertClose to avoid large GPU→CPU transfers.
    /// </summary>
    // Cached ElementWiseKernels per accelerator — avoids creating 50+ instances across tests
    private static readonly Dictionary<Accelerator, ElementWiseKernels> _ewCache = new();
    private static ElementWiseKernels GetOrCreateEW(Accelerator accelerator)
    {
        lock (_ewCache)
        {
            if (!_ewCache.TryGetValue(accelerator, out var ew))
            {
                ew = new ElementWiseKernels(accelerator);
                _ewCache[accelerator] = ew;
            }
            return ew;
        }
    }

    protected static async Task AssertCloseGpu(Accelerator accelerator,
        ArrayView1D<float, Stride1D.Dense> actualGpu, float[] expected,
        float tolerance, string label = "")
    {
        int count = Math.Min((int)actualGpu.Length, expected.Length);
        using var expectedBuf = accelerator.Allocate1D(expected);
        var ew = GetOrCreateEW(accelerator);
        var (meanErr, maxErr) = await ew.CompareOnGpuAsync(
            actualGpu.SubView(0, count), expectedBuf.View.SubView(0, count), count);
        if (maxErr > tolerance)
            throw new Exception($"{label}GPU maxErr={maxErr:E3}, meanErr={meanErr:E3} (tol={tolerance:E1})");
    }

    #endregion
}
