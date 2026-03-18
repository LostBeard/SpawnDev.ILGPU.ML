using ILGPU;
using ILGPU.Runtime;
using Microsoft.AspNetCore.Components;
using SpawnDev.BlazorJS;
using SpawnDev.ILGPU.ML.Demo.Services;
using SpawnDev.ILGPU.WebGPU;

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
            await LoadModel();
        }
    }

    private async Task LoadModel()
    {
        try
        {
            _isModelLoading = true;
            StateHasChanged();

            var builder = Context.Create();
            await builder.WebGPU();
            _context = builder.ToContext();
            var devices = _context.GetWebGPUDevices();
            if (devices.Count == 0)
            {
                Console.WriteLine("[Classify] No WebGPU devices found");
                _isModelLoading = false;
                StateHasChanged();
                return;
            }
            _accelerator = await devices[0].CreateAcceleratorAsync(_context);
            _selectedBackend = "WebGPU";

            _classService = new ClassificationService(Http);
            await _classService.LoadModelAsync("models/mobilenetv2", _accelerator);

            _isModelLoaded = true;
            _isModelLoading = false;
            Console.WriteLine("[Classify] MobileNetV2 loaded on WebGPU");
        }
        catch (Exception ex)
        {
            _isModelLoading = false;
            Console.WriteLine($"[Classify] Error: {ex.Message}");
        }
        StateHasChanged();
    }

    private async Task HandleImageLoaded(byte[] imageBytes)
    {
        _imageBytes = imageBytes;
        _predictions = null;

        // TODO: Decode image bytes to RGBA pixels using SpawnDev.BlazorJS
        // Pattern: Blob → createImageBitmap → OffscreenCanvas → getImageData → Uint8ClampedArray → int[]
        // TJ: please wire this with the correct BlazorJS API — I kept getting the types wrong.
        Console.WriteLine($"[Classify] Image received: {imageBytes.Length} bytes — needs decoding to RGBA");

        // For now, auto-run inference if we have pixels
        if (_rgbaPixels != null && _isModelLoaded)
            await RunInference();
    }

    private async Task RunInference()
    {
        if (_classService == null || _rgbaPixels == null) return;

        _isRunning = true;
        _predictions = null;
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

            Console.WriteLine($"[Classify] {ms:F1}ms — {results[0].Label} ({results[0].Confidence:P1})");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[Classify] Error: {ex.Message}\n{ex.StackTrace}");
        }

        _isRunning = false;
        StateHasChanged();
    }

    private async Task HandleBackendChange(string backend)
    {
        _selectedBackend = backend;
        // TODO: Dispose current accelerator, create new one for selected backend, reload model
    }

    private async Task CopyRaceResults()
    {
        if (_raceResults == null) return;
        var text = "SpawnDev.ILGPU.ML Backend Showdown (MobileNetV2)\n";
        foreach (var r in _raceResults.OrderBy(r => r.TimeMs))
            text += $"  {r.Backend}: {r.TimeMs:F1}ms\n";
        try
        {
            using var navigator = JS.Get<SpawnDev.BlazorJS.JSObjects.Navigator>("navigator");
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
