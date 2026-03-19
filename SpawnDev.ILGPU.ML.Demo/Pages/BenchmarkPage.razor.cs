using ILGPU;
using ILGPU.Runtime;
using Microsoft.AspNetCore.Components;
using SpawnDev.BlazorJS;
using SpawnDev.BlazorJS.JSObjects;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.ILGPU.WebGPU;
using System.Diagnostics;

namespace SpawnDev.ILGPU.ML.Demo.Pages;

public partial class BenchmarkPage : IDisposable
{
    [Inject] BlazorJSRuntime JS { get; set; } = default!;
    [Inject] HttpClient Http { get; set; } = default!;

    private Context? _context;

    protected override async Task OnAfterRenderAsync(bool firstRender)
    {
        if (firstRender)
        {
            // Detect device info
            try
            {
                using var navigator = JS.Get<Navigator>("navigator");
                _userAgent = navigator.UserAgent;
                // Try to get GPU info from WebGPU adapter
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

            // Create context with all backends
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

        // Count total tests
        var backends = new List<string>();
        if (_benchWebGPU) backends.Add("WebGPU");
        if (_benchWebGL) backends.Add("WebGL");
        if (_benchWasm) backends.Add("Wasm");

        var tests = new List<string>();
        if (_runMatMul) tests.Add("MatMul");
        if (_runClassification) tests.Add("Classification");
        if (_runSuperRes) tests.Add("Super Resolution");
        if (_runStyleTransfer) tests.Add("Style Transfer");

        _totalTests = backends.Count * tests.Count;
        if (_totalTests == 0) { _isRunning = false; return; }

        StateHasChanged();
        await Task.Yield();

        foreach (var backendId in backends)
        {
            Accelerator? accelerator = null;
            try
            {
                _currentTest = $"Creating {backendId} accelerator...";
                StateHasChanged();
                await Task.Yield();

                accelerator = await CreateAcceleratorForBackendAsync(backendId);
                if (accelerator == null)
                {
                    // Skip this backend — mark all its tests as failed
                    foreach (var test in tests)
                    {
                        _results.Add(new BenchResult { TestName = test, BackendName = $"{backendId} (unavailable)", InferenceMs = -1 });
                        _completedTests++;
                    }
                    continue;
                }

                // Run each selected test on this backend
                if (_runMatMul && tests.Contains("MatMul"))
                    await RunMatMulBench(accelerator, backendId);

                if (_runClassification && tests.Contains("Classification"))
                    await RunClassificationBench(accelerator, backendId);

                if (_runSuperRes && tests.Contains("Super Resolution"))
                    await RunSuperResBench(accelerator, backendId);

                if (_runStyleTransfer && tests.Contains("Style Transfer"))
                    await RunStyleTransferBench(accelerator, backendId);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[Benchmark] {backendId} error: {ex.Message}");
            }
            finally
            {
                accelerator?.Dispose();
            }
        }

        _isRunning = false;
        _currentTest = "";
        StateHasChanged();
    }

    private async Task RunMatMulBench(Accelerator accelerator, string backendName)
    {
        _currentTest = $"MatMul GFLOPS on {backendName}...";
        _progressPercent = (float)_completedTests / _totalTests * 100;
        StateHasChanged();
        await Task.Yield();

        try
        {
            var matMul = new SpawnDev.ILGPU.ML.MatMulKernel(accelerator);

            // 512x512 matrix multiply — 2 * 512^3 = 268M FLOPs per run
            int M = 512, K = 512, N = 512;
            long flopsPerRun = 2L * M * K * N;

            using var a = accelerator.Allocate1D<float>(M * K);
            using var b = accelerator.Allocate1D<float>(K * N);
            using var c = accelerator.Allocate1D<float>(M * N);

            // Warmup run (triggers kernel compilation)
            matMul.MatMul(a.View, b.View, c.View, M, K, N);
            await accelerator.SynchronizeAsync();

            // Timed runs
            int runs = 5;
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < runs; i++)
            {
                matMul.MatMul(a.View, b.View, c.View, M, K, N);
            }
            await accelerator.SynchronizeAsync();
            sw.Stop();

            double totalMs = sw.Elapsed.TotalMilliseconds;
            double gflops = (flopsPerRun * runs) / (totalMs / 1000.0) / 1e9;

            _results.Add(new BenchResult
            {
                TestName = $"MatMul 512x512 ({gflops:F1} GFLOPS)",
                BackendName = backendName,
                InferenceMs = totalMs / runs,
            });

            Console.WriteLine($"[Benchmark] MatMul/{backendName}: {totalMs / runs:F1}ms/run, {gflops:F1} GFLOPS");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[Benchmark] MatMul/{backendName} failed: {ex.Message}");
            _results.Add(new BenchResult { TestName = "MatMul 512x512", BackendName = $"{backendName} (error)", InferenceMs = -1 });
        }

        _completedTests++;
        _progressPercent = (float)_completedTests / _totalTests * 100;
        StateHasChanged();
    }

    private async Task RunClassificationBench(Accelerator accelerator, string backendName)
    {
        _currentTest = $"Classification on {backendName}...";
        _progressPercent = (float)_completedTests / _totalTests * 100;
        StateHasChanged();
        await Task.Yield();

        try
        {
            var sw = Stopwatch.StartNew();
            var session = await InferenceSession.CreateAsync(accelerator, Http, "models/squeezenet");
            var loadMs = sw.Elapsed.TotalMilliseconds;

            var pipeline = new ClassificationPipeline(session, accelerator);

            // Create a simple test image (gradient)
            int w = 224, h = 224;
            var pixels = CreateGradientImage(w, h);

            sw.Restart();
            var results = await pipeline.ClassifyAsync(pixels, w, h);
            sw.Stop();

            _results.Add(new BenchResult
            {
                TestName = "Classification (SqueezeNet)",
                BackendName = backendName,
                InferenceMs = sw.Elapsed.TotalMilliseconds,
                ModelLoadMs = loadMs
            });

            Console.WriteLine($"[Benchmark] Classification/{backendName}: {sw.Elapsed.TotalMilliseconds:F1}ms (load: {loadMs:F0}ms) — {results[0].Label}");

            pipeline.Dispose();
            session.Dispose();
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[Benchmark] Classification/{backendName} failed: {ex.Message}");
            _results.Add(new BenchResult { TestName = "Classification (SqueezeNet)", BackendName = $"{backendName} (error)", InferenceMs = -1 });
        }

        _completedTests++;
        _progressPercent = (float)_completedTests / _totalTests * 100;
        StateHasChanged();
    }

    private async Task RunSuperResBench(Accelerator accelerator, string backendName)
    {
        _currentTest = $"Super Resolution on {backendName}...";
        _progressPercent = (float)_completedTests / _totalTests * 100;
        StateHasChanged();
        await Task.Yield();

        try
        {
            var sw = Stopwatch.StartNew();
            var session = await InferenceSession.CreateAsync(accelerator, Http, "models/super-resolution");
            var loadMs = sw.Elapsed.TotalMilliseconds;

            var pipeline = new SuperResolutionPipeline(session, accelerator);

            int w = 64, h = 64;
            var pixels = CreateGradientImage(w, h);

            sw.Restart();
            var result = await pipeline.UpscaleAsync(pixels, w, h);
            sw.Stop();

            _results.Add(new BenchResult
            {
                TestName = "Super Resolution (ESPCN 3x)",
                BackendName = backendName,
                InferenceMs = sw.Elapsed.TotalMilliseconds,
                ModelLoadMs = loadMs
            });

            Console.WriteLine($"[Benchmark] SuperRes/{backendName}: {sw.Elapsed.TotalMilliseconds:F1}ms (load: {loadMs:F0}ms) — {w}x{h} → {result.Width}x{result.Height}");

            pipeline.Dispose();
            session.Dispose();
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[Benchmark] SuperRes/{backendName} failed: {ex.Message}");
            _results.Add(new BenchResult { TestName = "Super Resolution (ESPCN 3x)", BackendName = $"{backendName} (error)", InferenceMs = -1 });
        }

        _completedTests++;
        _progressPercent = (float)_completedTests / _totalTests * 100;
        StateHasChanged();
    }

    private async Task RunStyleTransferBench(Accelerator accelerator, string backendName)
    {
        _currentTest = $"Style Transfer on {backendName}...";
        _progressPercent = (float)_completedTests / _totalTests * 100;
        StateHasChanged();
        await Task.Yield();

        try
        {
            var sw = Stopwatch.StartNew();
            var session = await InferenceSession.CreateAsync(accelerator, Http, "models/style-mosaic");
            var loadMs = sw.Elapsed.TotalMilliseconds;

            var pipeline = new StyleTransferPipeline(session, accelerator);

            int w = 224, h = 224;
            var pixels = CreateGradientImage(w, h);

            sw.Restart();
            var result = await pipeline.TransferAsync(pixels, w, h);
            sw.Stop();

            _results.Add(new BenchResult
            {
                TestName = "Style Transfer (Mosaic)",
                BackendName = backendName,
                InferenceMs = sw.Elapsed.TotalMilliseconds,
                ModelLoadMs = loadMs
            });

            Console.WriteLine($"[Benchmark] Style/{backendName}: {sw.Elapsed.TotalMilliseconds:F1}ms (load: {loadMs:F0}ms)");

            pipeline.Dispose();
            session.Dispose();
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[Benchmark] Style/{backendName} failed: {ex.Message}");
            _results.Add(new BenchResult { TestName = "Style Transfer (Mosaic)", BackendName = $"{backendName} (error)", InferenceMs = -1 });
        }

        _completedTests++;
        _progressPercent = (float)_completedTests / _totalTests * 100;
        StateHasChanged();
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
        catch { return null; }
    }

    private async Task<Accelerator?> TryCreateAsync<TDevice>() where TDevice : Device
    {
        var devices = _context!.GetDevices<TDevice>();
        return devices.Count > 0 ? await devices[0].CreateAcceleratorAsync(_context) : null;
    }

    private async Task CopyResults()
    {
        var lines = new List<string>();
        lines.Add("SpawnDev.ILGPU.ML — Backend Benchmark");
        lines.Add($"Date: {DateTime.UtcNow:yyyy-MM-dd HH:mm} UTC");
        if (_gpuName != null) lines.Add($"GPU: {_gpuName}");
        lines.Add("");

        foreach (var group in _results.Where(r => r.InferenceMs > 0).GroupBy(r => r.TestName))
        {
            lines.Add($"  {group.Key}:");
            var sorted = group.OrderBy(r => r.InferenceMs).ToList();
            int rank = 0;
            foreach (var r in sorted)
            {
                rank++;
                var medal = rank switch { 1 => "1st", 2 => "2nd", 3 => "3rd", _ => $"{rank}th" };
                lines.Add($"    {medal} {r.BackendName}: {r.InferenceMs:F1}ms (load: {r.ModelLoadMs:F0}ms)");
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
        _context?.Dispose();
    }

    public class BenchResult
    {
        public string TestName { get; set; } = "";
        public string BackendName { get; set; } = "";
        public double InferenceMs { get; set; }
        public double ModelLoadMs { get; set; }
    }
}
