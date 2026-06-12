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

    protected async Task RunTest(Func<Accelerator, Task> testBody)
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
        }
    }

    public virtual void Dispose()
    {
        try { _prevAccelerator?.Dispose(); } catch { }
        _prevAccelerator = null;
        try { _prevContext?.Dispose(); } catch { }
        _prevContext = null;
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
