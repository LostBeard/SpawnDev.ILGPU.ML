using ILGPU;
using ILGPU.Runtime;
using Microsoft.AspNetCore.Components;
using SpawnDev.BlazorJS;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.ILGPU.WebGPU;
using System.Diagnostics;

namespace SpawnDev.ILGPU.ML.Demo.Pages;

public partial class StylePage : IDisposable
{
    [Inject] BlazorJSRuntime JS { get; set; } = default!;
    [Inject] HttpClient Http { get; set; } = default!;

    private InferenceSession? _session;
    private StyleTransferPipeline? _pipeline;
    private Context? _context;
    private Accelerator? _accelerator;
    private int[]? _rgbaPixels;
    private int _imageWidth, _imageHeight;

    private async Task LoadStyle(string styleName)
    {
        try
        {
            _isModelLoading = true;
            _selectedStyle = styleName;
            StateHasChanged();

            if (_accelerator == null)
            {
                var builder = MLContext.Create();
                await builder.WebGPU();
                _context = builder.ToContext();
                var devices = _context.GetWebGPUDevices();
                if (devices.Count == 0) return;
                _accelerator = await devices[0].CreateAcceleratorAsync(_context);
            }

            _session?.Dispose();
            _pipeline?.Dispose();

            _session = await InferenceSession.CreateAsync(_accelerator, Http, $"models/style-{styleName}");
            _pipeline = new StyleTransferPipeline(_session, _accelerator);

            _isModelLoaded = true;
            _isModelLoading = false;
            Console.WriteLine($"[Style] {styleName} loaded: {_session}");
        }
        catch (Exception ex)
        {
            _isModelLoading = false;
            Console.WriteLine($"[Style] Error: {ex.Message}");
        }
        StateHasChanged();
    }

    private async Task HandleImageLoaded(byte[] imageBytes)
    {
        try
        {
            using var blob = new SpawnDev.BlazorJS.JSObjects.Blob(
                new[] { imageBytes }, new SpawnDev.BlazorJS.JSObjects.BlobOptions { Type = "image/jpeg" });
            using var window = JS.Get<SpawnDev.BlazorJS.JSObjects.Window>("window");
            using var bitmap = await window.CreateImageBitmap(blob);
            int w = (int)bitmap.Width; int h = (int)bitmap.Height;
            using var canvas = new SpawnDev.BlazorJS.JSObjects.HTMLCanvasElement();
            canvas.Width = w; canvas.Height = h;
            using var ctx = canvas.Get2DContext();
            ctx.DrawImage(bitmap, 0, 0, w, h);
            using var imageData = ctx.GetImageData(0, 0, w, h);
            using var data = imageData.Data;
            _rgbaPixels = data.Read<int>();
            _imageWidth = w; _imageHeight = h;

            if (_isModelLoaded) await RunStyleTransfer();
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[Style] Decode error: {ex.Message}");
        }
    }

    private async Task RunStyleTransfer()
    {
        if (_pipeline == null || _rgbaPixels == null) return;
        _isRunning = true;
        StateHasChanged();
        await Task.Yield();

        try
        {
            var sw = Stopwatch.StartNew();
            var result = await _pipeline.TransferAsync(_rgbaPixels, _imageWidth, _imageHeight);
            sw.Stop();
            _inferenceMs = sw.Elapsed.TotalMilliseconds;

            _styledImageUrl = Services.ImageDisplayHelper.ToDataUrl(JS, result.RgbaPixels, result.Width, result.Height);
            Console.WriteLine($"[Style] {_inferenceMs:F0}ms, output {result.Width}x{result.Height}");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[Style] Error: {ex.Message}\n{ex.StackTrace}");
        }

        _isRunning = false;
        StateHasChanged();
    }

    private async Task HandleBackendChange(string backend) => _selectedBackend = backend;

    public void Dispose()
    {
        _pipeline?.Dispose();
        _session?.Dispose();
        _accelerator?.Dispose();
        _context?.Dispose();
    }
}
