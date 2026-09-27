using ILGPU;
using ILGPU.Runtime;
using Microsoft.AspNetCore.Components;
using SpawnDev.SpawnJS;
using SpawnDev.SpawnJS.JSObjects;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.ILGPU.WebGPU;
using System.Diagnostics;

namespace SpawnDev.ILGPU.ML.Demo.Pages;

public partial class BenchmarkPage : IDisposable
{
    [Inject] SpawnJSRuntime JS { get; set; } = default!;
    [Inject] HttpClient Http { get; set; } = default!;

    private Context? _context;

    /// <summary>
    /// Accelerators kept for the life of the page. Disposing a WebGPU accelerator destroys its
    /// GPUDevice; creating a second one from the same Context device often fails (adapter/device
    /// already spent), which is why a re-run used to mark WebGPU "(unavailable)".
    /// </summary>
    private readonly Dictionary<string, Accelerator> _accelerators = new(StringComparer.Ordinal);

    /// <summary>Stable group titles — MatMul used to bake GFLOPS into the name and split groups.</summary>
    private static readonly string[] TestOrder =
    [
        "MatMul 512x512",
        "Classification (SqueezeNet)",
        "Style Transfer (Mosaic)",
        "Super Resolution (ESPCN 3x)",
    ];

    protected override async Task OnAfterRenderAsync(bool firstRender)
    {
        if (firstRender)
        {
            try
            {
                using var navigator = JS.Get<Navigator>("navigator");
                _userAgent = navigator.UserAgent;
                using var gpu = JS.Get<GPU>("navigator.gpu");
                if (gpu != null)
                {
                    using var adapter = await gpu.RequestAdapter();
                    if (adapter != null)
                    {
                        var info = adapter.Info;
                        _gpuName = $"{info.Vendor} — {info.Architecture}";
                    }
                }
            }
            catch { }

            try
            {
                var builder = MLContext.Create();
                await builder.AllAcceleratorsAsync();
                _context = builder.ToContext();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[Benchmark] Context creation failed: {ex.Message}");
            }
            StateHasChanged();
        }
    }

    private async Task RunBenchmarks()
    {
        _results.Clear();
        _isRunning = true;
        _completedTests = 0;
        _copied = false;

        var backends = new List<string>();
        if (_benchWebGPU) backends.Add("WebGPU");
        if (_benchWebGL) backends.Add("WebGL");
        if (_benchWasm) backends.Add("Wasm");

        var tests = new List<string>();
        if (_runMatMul) tests.Add("MatMul 512x512");
        if (_runClassification) tests.Add("Classification (SqueezeNet)");
        if (_runStyleTransfer) tests.Add("Style Transfer (Mosaic)");
        if (_runSuperRes) tests.Add("Super Resolution (ESPCN 3x)");

        _totalTests = backends.Count * tests.Count;
        if (_totalTests == 0) { _isRunning = false; return; }

        StateHasChanged();
        await Task.Yield();

        foreach (var backendId in backends)
        {
            try
            {
                _currentTest = $"Creating {backendId} accelerator...";
                StateHasChanged();
                await Task.Yield();

                var accelerator = await GetOrCreateAcceleratorAsync(backendId);
                if (accelerator == null)
                {
                    foreach (var test in tests)
                    {
                        _results.Add(new BenchResult
                        {
                            TestName = test,
                            BackendName = backendId,
                            InferenceMs = -1,
                            Detail = "unavailable",
                        });
                        _completedTests++;
                    }
                    continue;
                }

                if (_runMatMul)
                    await RunMatMulBench(accelerator, backendId);

                if (_runClassification)
                    await RunClassificationBench(accelerator, backendId);

                if (_runStyleTransfer)
                    await RunStyleTransferBench(accelerator, backendId);

                if (_runSuperRes)
                    await RunSuperResBench(accelerator, backendId);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[Benchmark] {backendId} error: {ex}");
            }
            // Do NOT dispose the accelerator here — see _accelerators remarks.
        }

        _isRunning = false;
        _currentTest = "";
        StateHasChanged();
    }

    private async Task RunMatMulBench(Accelerator accelerator, string backendName)
    {
        const string testName = "MatMul 512x512";
        _currentTest = $"MatMul GFLOPS on {backendName}...";
        _progressPercent = (float)_completedTests / _totalTests * 100;
        StateHasChanged();
        await Task.Yield();

        try
        {
            var matMul = new SpawnDev.ILGPU.ML.MatMulKernel(accelerator);

            int M = 512, K = 512, N = 512;
            long flopsPerRun = 2L * M * K * N;

            using var a = accelerator.Allocate1D<float>(M * K);
            using var b = accelerator.Allocate1D<float>(K * N);
            using var c = accelerator.Allocate1D<float>(M * N);

            matMul.MatMul(a.View, b.View, c.View, M, K, N);
            await accelerator.SynchronizeAsync();

            int runs = 5;
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < runs; i++)
                matMul.MatMul(a.View, b.View, c.View, M, K, N);
            await accelerator.SynchronizeAsync();
            sw.Stop();

            double totalMs = sw.Elapsed.TotalMilliseconds;
            double gflops = (flopsPerRun * runs) / (totalMs / 1000.0) / 1e9;

            _results.Add(new BenchResult
            {
                TestName = testName,
                BackendName = backendName,
                InferenceMs = totalMs / runs,
                Detail = $"{gflops:F1} GFLOPS",
            });

            Console.WriteLine($"[Benchmark] MatMul/{backendName}: {totalMs / runs:F1}ms/run, {gflops:F1} GFLOPS");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[Benchmark] MatMul/{backendName} failed: {ex.Message}");
            _results.Add(new BenchResult
            {
                TestName = testName,
                BackendName = backendName,
                InferenceMs = -1,
                Detail = "error",
            });
        }

        _completedTests++;
        _progressPercent = (float)_completedTests / _totalTests * 100;
        StateHasChanged();
    }

    private async Task RunClassificationBench(Accelerator accelerator, string backendName)
    {
        const string testName = "Classification (SqueezeNet)";
        _currentTest = $"Classification on {backendName}...";
        _progressPercent = (float)_completedTests / _totalTests * 100;
        StateHasChanged();
        await Task.Yield();

        try
        {
            var sw = Stopwatch.StartNew();
            using var hub1 = new ModelHub(JS);
            var session = await InferenceSession.CreateFromHuggingFaceAsync(
                accelerator, hub1, ModelHub.KnownModels.SqueezeNet, "squeezenet1.1-7.onnx",
                http: Http);
            var loadMs = sw.Elapsed.TotalMilliseconds;

            var pipeline = new ClassificationPipeline(session, accelerator);
            int w = 224, h = 224;
            var pixels = CreateGradientImage(w, h);

            sw.Restart();
            var results = await pipeline.ClassifyAsync(pixels, w, h);
            sw.Stop();

            _results.Add(new BenchResult
            {
                TestName = testName,
                BackendName = backendName,
                InferenceMs = sw.Elapsed.TotalMilliseconds,
                ModelLoadMs = loadMs,
            });

            Console.WriteLine($"[Benchmark] Classification/{backendName}: {sw.Elapsed.TotalMilliseconds:F1}ms (load: {loadMs:F0}ms) — {results[0].Label}");

            pipeline.Dispose();
            session.Dispose();
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[Benchmark] Classification/{backendName} failed: {ex.Message}");
            _results.Add(new BenchResult
            {
                TestName = testName,
                BackendName = backendName,
                InferenceMs = -1,
                Detail = "error",
            });
        }

        _completedTests++;
        _progressPercent = (float)_completedTests / _totalTests * 100;
        StateHasChanged();
    }

    private async Task RunSuperResBench(Accelerator accelerator, string backendName)
    {
        const string testName = "Super Resolution (ESPCN 3x)";
        _currentTest = $"Super Resolution on {backendName}...";
        _progressPercent = (float)_completedTests / _totalTests * 100;
        StateHasChanged();
        await Task.Yield();

        try
        {
            var sw = Stopwatch.StartNew();
            using var hub2 = new ModelHub(JS);
            var session = await InferenceSession.CreateFromHuggingFaceAsync(
                accelerator, hub2, ModelHub.KnownModels.SuperResolution, "super-resolution-10.onnx",
                http: Http);
            var loadMs = sw.Elapsed.TotalMilliseconds;

            var pipeline = new SuperResolutionPipeline(session, accelerator);
            int w = 64, h = 64;
            var pixels = CreateGradientImage(w, h);

            sw.Restart();
            var result = await pipeline.UpscaleAsync(pixels, w, h);
            sw.Stop();

            _results.Add(new BenchResult
            {
                TestName = testName,
                BackendName = backendName,
                InferenceMs = sw.Elapsed.TotalMilliseconds,
                ModelLoadMs = loadMs,
            });

            Console.WriteLine($"[Benchmark] SuperRes/{backendName}: {sw.Elapsed.TotalMilliseconds:F1}ms (load: {loadMs:F0}ms) — {w}x{h} → {result.Width}x{result.Height}");

            pipeline.Dispose();
            session.Dispose();
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[Benchmark] SuperRes/{backendName} failed: {ex.Message}");
            _results.Add(new BenchResult
            {
                TestName = testName,
                BackendName = backendName,
                InferenceMs = -1,
                Detail = "error",
            });
        }

        _completedTests++;
        _progressPercent = (float)_completedTests / _totalTests * 100;
        StateHasChanged();
    }

    private async Task RunStyleTransferBench(Accelerator accelerator, string backendName)
    {
        const string testName = "Style Transfer (Mosaic)";
        _currentTest = $"Style Transfer on {backendName}...";
        _progressPercent = (float)_completedTests / _totalTests * 100;
        StateHasChanged();
        await Task.Yield();

        try
        {
            var sw = Stopwatch.StartNew();
            using var hub3 = new ModelHub(JS);
            var session = await InferenceSession.CreateFromHuggingFaceAsync(
                accelerator, hub3, ModelHub.KnownModels.StyleMosaic, "mosaic-9.onnx",
                http: Http);
            var loadMs = sw.Elapsed.TotalMilliseconds;

            var pipeline = new StyleTransferPipeline(session, accelerator);
            int w = 224, h = 224;
            var pixels = CreateGradientImage(w, h);

            sw.Restart();
            var result = await pipeline.TransferAsync(pixels, w, h);
            sw.Stop();

            _results.Add(new BenchResult
            {
                TestName = testName,
                BackendName = backendName,
                InferenceMs = sw.Elapsed.TotalMilliseconds,
                ModelLoadMs = loadMs,
            });

            Console.WriteLine($"[Benchmark] Style/{backendName}: {sw.Elapsed.TotalMilliseconds:F1}ms (load: {loadMs:F0}ms)");

            pipeline.Dispose();
            session.Dispose();
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[Benchmark] Style/{backendName} failed: {ex.Message}");
            _results.Add(new BenchResult
            {
                TestName = testName,
                BackendName = backendName,
                InferenceMs = -1,
                Detail = "error",
            });
        }

        _completedTests++;
        _progressPercent = (float)_completedTests / _totalTests * 100;
        StateHasChanged();
    }

    private async Task<Accelerator?> GetOrCreateAcceleratorAsync(string backendId)
    {
        if (_accelerators.TryGetValue(backendId, out var existing))
            return existing;

        var created = await CreateAcceleratorForBackendAsync(backendId);
        if (created != null)
            _accelerators[backendId] = created;
        return created;
    }

    private async Task<Accelerator?> CreateAcceleratorForBackendAsync(string backendId)
    {
        if (_context == null) return null;
        try
        {
            return backendId switch
            {
                "WebGPU" => await TryCreateAsync<WebGPUILGPUDevice>(),
                "WebGL" => await TryCreateAsync<SpawnDev.ILGPU.WebGL.WebGLILGPUDevice>(),
                "Wasm" => await TryCreateAsync<SpawnDev.ILGPU.Wasm.WasmILGPUDevice>(),
                _ => null
            };
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[Benchmark] CreateAccelerator {backendId} failed: {ex}");
            return null;
        }
    }

    private async Task<Accelerator?> TryCreateAsync<TDevice>() where TDevice : Device
    {
        var devices = _context!.GetDevices<TDevice>();
        return devices.Count > 0 ? await devices[0].CreateAcceleratorAsync(_context) : null;
    }

    /// <summary>Groups with a fixed test order so MatMul / Classification / … stay together.</summary>
    private IEnumerable<IGrouping<string, BenchResult>> GroupedResults()
    {
        var byName = _results.GroupBy(r => r.TestName).ToDictionary(g => g.Key, g => g);
        foreach (var name in TestOrder)
        {
            if (byName.TryGetValue(name, out var g))
                yield return g;
        }
        foreach (var g in byName.Values)
        {
            if (!TestOrder.Contains(g.Key))
                yield return g;
        }
    }

    private static List<BenchResult> Ranked(IEnumerable<BenchResult> group)
        => group.OrderBy(r => r.InferenceMs < 0 ? double.MaxValue : r.InferenceMs).ToList();

    private async Task CopyResults()
    {
        var lines = new List<string>();
        lines.Add("SpawnDev.ILGPU.ML — Backend Benchmark");
        lines.Add($"Date: {DateTime.UtcNow:yyyy-MM-dd HH:mm} UTC");
        if (_gpuName != null) lines.Add($"GPU: {_gpuName}");
        lines.Add("");

        foreach (var group in GroupedResults())
        {
            lines.Add($"  {group.Key}:");
            var sorted = Ranked(group);
            int rank = 0;
            foreach (var r in sorted)
            {
                rank++;
                if (r.InferenceMs < 0)
                {
                    lines.Add($"    — {r.BackendName}: {r.Detail ?? "failed"}");
                    continue;
                }
                var medal = rank switch { 1 => "1st", 2 => "2nd", 3 => "3rd", _ => $"{rank}th" };
                var extra = string.IsNullOrEmpty(r.Detail) ? "" : $" ({r.Detail})";
                var load = r.ModelLoadMs > 0 ? $", load {r.ModelLoadMs:F0}ms" : "";
                lines.Add($"    {medal} {r.BackendName}: {r.InferenceMs:F1}ms{extra}{load}");
            }
        }

        lines.Add("");
        lines.Add("Powered by SpawnDev.ILGPU.ML — 100% in-browser, no server");

        try
        {
            using var navigator = JS.Get<Navigator>("navigator");
            using var clipboard = navigator.Clipboard;
            await clipboard.WriteText(string.Join("\n", lines));
            _copied = true;
            StateHasChanged();
        }
        catch { }
    }

    private async Task CopyOneLiner()
    {
        var fastest = _results.Where(r => r.InferenceMs > 0).OrderBy(r => r.InferenceMs).FirstOrDefault();
        if (fastest == null) return;

        var text = $"SpawnDev.ILGPU.ML benchmark: {fastest.TestName} in {fastest.InferenceMs:F0}ms on {fastest.BackendName} — 100% in-browser, no cloud #WebGPU #dotnet #blazor";
        try
        {
            using var navigator = JS.Get<Navigator>("navigator");
            using var clipboard = navigator.Clipboard;
            await clipboard.WriteText(text);
            _copied = true;
            StateHasChanged();
        }
        catch { }
    }

    private void ClearResults()
    {
        _results.Clear();
        _completedTests = 0;
        _copied = false;
        StateHasChanged();
    }

    private static int[] CreateGradientImage(int w, int h)
    {
        var pixels = new int[w * h];
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
                pixels[y * w + x] = (int)(x * 255f / w) | ((int)(y * 255f / h) << 8) | (128 << 16) | (0xFF << 24);
        return pixels;
    }

    public void Dispose()
    {
        foreach (var acc in _accelerators.Values)
        {
            try { acc.Dispose(); } catch { }
        }
        _accelerators.Clear();
        _context?.Dispose();
        _context = null;
    }

    public class BenchResult
    {
        public string TestName { get; set; } = "";
        public string BackendName { get; set; } = "";
        public double InferenceMs { get; set; }
        public double ModelLoadMs { get; set; }
        /// <summary>Extra for the row (e.g. GFLOPS, or "unavailable" / "error").</summary>
        public string? Detail { get; set; }
    }
}
