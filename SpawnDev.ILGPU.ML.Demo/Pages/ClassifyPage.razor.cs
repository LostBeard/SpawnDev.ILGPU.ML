using ILGPU;
using ILGPU.Runtime;
using Microsoft.AspNetCore.Components;
using SpawnDev.BlazorJS;
using SpawnDev.BlazorJS.JSObjects;
using SpawnDev.ILGPU.ML.Demo.Services;
using SpawnDev.ILGPU.WebGPU;
using SpawnDev.ILGPU.WebGPU.Backend;

namespace SpawnDev.ILGPU.ML.Demo.Pages;

public partial class ClassifyPage : IDisposable
{
    [Inject] BlazorJSRuntime JS { get; set; } = default!;
    [Inject] HttpClient Http { get; set; } = default!;

    private ClassificationService? _classService;
    private Context? _context;
    private Accelerator? _accelerator;
    private byte[]? _imageBytes;
    private int[]? _rgbaPixels;
    private int _imageWidth;
    private int _imageHeight;

    protected override async Task OnAfterRenderAsync(bool firstRender)
    {
        if (firstRender)
        {
            await LoadBackendAndModelAsync();
        }
    }

    private async Task LoadBackendAndModelAsync()
    {
        try
        {
            _isModelLoading = true;
            _isModelLoaded = false;
            _error = null;
            _modelProgress = 5;
            StateHasChanged();

            // Create ILGPU context with all browser backends (first time only)
            if (_context == null)
            {
                var builder = MLContext.Create();
                await builder.AllAcceleratorsAsync();
                _context = builder.ToContext();
            }

            // Create accelerator for selected backend
            _modelProgress = 20;
            StateHasChanged();
            _accelerator = await CreateAcceleratorForBackendAsync(_selectedBackend);
            if (_accelerator == null)
            {
                _error = $"No {_selectedBackend} device available";
                _isModelLoading = false;
                StateHasChanged();
                return;
            }

            // Load SqueezeNet model
            _modelProgress = 50;
            StateHasChanged();
            _classService = new ClassificationService(Http);
            await _classService.LoadModelAsync("models/squeezenet", _accelerator);

            _isModelLoaded = true;
            _modelProgress = 100;
            Console.WriteLine($"[Classify] Model loaded on {_selectedBackend}: {_classService.ModelInfo}");
        }
        catch (Exception ex)
        {
            _error = $"Failed to load: {ex.Message}";
            Console.WriteLine($"[Classify] Error: {ex.Message}");
        }
        finally
        {
            _isModelLoading = false;
            StateHasChanged();
        }
    }

    private async Task<Accelerator?> CreateAcceleratorForBackendAsync(string backendId)
    {
        if (_context == null) return null;
        try
        {
            return backendId switch
            {
                "WebGPU" => await CreateWebGPUAsync(),
                "WebGL" => await CreateWebGLAsync(),
                "Wasm" => await CreateWasmAsync(),
                _ => null
            };
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[Classify] Backend {backendId} failed: {ex.Message}");
            return null;
        }
    }

    private async Task<Accelerator?> CreateWebGPUAsync()
    {
        var devices = _context!.GetDevices<WebGPUILGPUDevice>();
        return devices.Count > 0 ? await devices[0].CreateAcceleratorAsync(_context) : null;
    }

    private async Task<Accelerator?> CreateWebGLAsync()
    {
        var devices = _context!.GetDevices<SpawnDev.ILGPU.WebGL.WebGLILGPUDevice>();
        return devices.Count > 0 ? await devices[0].CreateAcceleratorAsync(_context) : null;
    }

    private async Task<Accelerator?> CreateWasmAsync()
    {
        var devices = _context!.GetDevices<SpawnDev.ILGPU.Wasm.WasmILGPUDevice>();
        return devices.Count > 0 ? await devices[0].CreateAcceleratorAsync(_context) : null;
    }

    private async Task HandleBackendChange(string backend)
    {
        if (backend == _selectedBackend && _isModelLoaded) return;
        _selectedBackend = backend;
        _predictions = null;

        // Dispose old resources (not the context — it holds all backend registrations)
        _classService?.Dispose();
        _classService = null;
        _accelerator?.Dispose();
        _accelerator = null;

        await LoadBackendAndModelAsync();
    }

    private async Task HandleImageLoaded(byte[] imageBytes)
    {
        _imageBytes = imageBytes;
        _predictions = null;
        _error = null;

        try
        {
            // Decode image file bytes to RGBA int[] via browser-native createImageBitmap
            using var blob = new Blob(
                new[] { imageBytes }, new BlobOptions { Type = "image/png" });
            using var window = JS.Get<Window>("window");
            using var bitmap = await window.CreateImageBitmap(blob);

            int w = (int)bitmap.Width;
            int h = (int)bitmap.Height;

            using var canvas = new HTMLCanvasElement();
            canvas.Width = w;
            canvas.Height = h;
            using var ctx = canvas.Get2DContext();
            ctx.DrawImage(bitmap, 0, 0, w, h);
            using var imageData = ctx.GetImageData(0, 0, w, h);
            using var data = imageData.Data;
            _rgbaPixels = data.Read<int>();
            _imageWidth = w;
            _imageHeight = h;

            Console.WriteLine($"[Classify] Decoded: {w}x{h}, {_rgbaPixels.Length} pixels");

            if (_isModelLoaded)
                await RunInference();
        }
        catch (Exception ex)
        {
            _error = $"Image decode failed: {ex.Message}";
            Console.WriteLine($"[Classify] Decode error: {ex.Message}");
        }
    }

    private async Task RunInference()
    {
        if (_classService == null || _rgbaPixels == null) return;

        _isRunning = true;
        _predictions = null;
        _error = null;
        StateHasChanged();
        await Task.Yield();

        try
        {
            var (results, ms) = await _classService.ClassifyAsync(
                _rgbaPixels, _imageWidth, _imageHeight);

            _inferenceMs = ms;
            _predictions = results.Select(r => new Components.ConfidenceBars.Prediction
            {
                Label = r.Label,
                Confidence = r.Confidence
            }).ToList();

            Console.WriteLine($"[Classify] {ms:F1}ms — Top: {results[0].Label} ({results[0].Confidence:P2})");
        }
        catch (Exception ex)
        {
            _error = $"Inference failed: {ex.Message}";
            Console.WriteLine($"[Classify] Error: {ex.Message}");
        }

        _isRunning = false;
        StateHasChanged();
    }

    private async Task CopyRaceResults()
    {
        if (_raceResults == null) return;
        var text = "SpawnDev.ILGPU.ML Backend Showdown (SqueezeNet)\n";
        foreach (var r in _raceResults.OrderBy(r => r.TimeMs))
            text += $"  {r.Backend}: {r.TimeMs:F1}ms\n";
        try
        {
            using var navigator = JS.Get<Navigator>("navigator");
            using var clipboard = navigator.Clipboard;
            await clipboard.WriteText(text);
        }
        catch { }
    }

    public void Dispose()
    {
        _classService?.Dispose();
        _accelerator?.Dispose();
        _context?.Dispose();
    }
}
