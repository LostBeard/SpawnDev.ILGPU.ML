using ILGPU;
using ILGPU.Runtime;
using Microsoft.AspNetCore.Components;
using SpawnDev.BlazorJS;
using SpawnDev.BlazorJS.JSObjects;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.ILGPU.WebGPU;
using SpawnDev.ILGPU.WebGPU.Backend;
using System.Diagnostics;

namespace SpawnDev.ILGPU.ML.Demo.Pages;

public partial class SuperResPage : IDisposable
{
    [Inject] BlazorJSRuntime JS { get; set; } = default!;
    [Inject] HttpClient Http { get; set; } = default!;

    private InferenceSession? _session;
    private SuperResolutionPipeline? _pipeline;
    private Context? _context;
    private Accelerator? _accelerator;
    private int[]? _rgbaPixels;
    private int _imageWidth, _imageHeight;

    protected override async Task OnAfterRenderAsync(bool firstRender)
    {
        if (firstRender)
            await LoadBackendAndModelAsync();
    }

    private async Task LoadBackendAndModelAsync()
    {
        try
        {
            _isModelLoading = true;
            _isModelLoaded = false;
            StateHasChanged();

            // Create context with all browser backends (first time only)
            if (_context == null)
            {
                var builder = MLContext.Create();
                await builder.AllAcceleratorsAsync();
                _context = builder.ToContext();
            }

            // Create accelerator for selected backend
            _accelerator = await CreateAcceleratorForBackendAsync(_selectedBackend);
            if (_accelerator == null)
            {
                Console.WriteLine($"[SuperRes] No {_selectedBackend} device available");
                _isModelLoading = false;
                StateHasChanged();
                return;
            }

            _session = await InferenceSession.CreateFromFileAsync(_accelerator, Http, "models/super-resolution/model.onnx");
            _pipeline = new SuperResolutionPipeline(_session, _accelerator);

            _isModelLoaded = true;
            Console.WriteLine($"[SuperRes] Loaded on {_selectedBackend}: {_session}");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[SuperRes] Error: {ex.Message}");
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
                "WebGPU" => (await TryCreateAsync<WebGPUILGPUDevice>()),
                "WebGL" => (await TryCreateAsync<SpawnDev.ILGPU.WebGL.WebGLILGPUDevice>()),
                "Wasm" => (await TryCreateAsync<SpawnDev.ILGPU.Wasm.WasmILGPUDevice>()),
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

    private async Task HandleImageLoaded(byte[] imageBytes)
    {
        _enhancedImageUrl = null;
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

            if (_isModelLoaded) await RunSuperRes();
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[SuperRes] Decode error: {ex.Message}");
        }
    }

    private async Task RunSuperRes()
    {
        if (_pipeline == null || _rgbaPixels == null) return;
        _isRunning = true;
        StateHasChanged();
        await Task.Yield();

        try
        {
            var sw = Stopwatch.StartNew();
            var result = await _pipeline.UpscaleAsync(_rgbaPixels, _imageWidth, _imageHeight);
            sw.Stop();
            _inferenceMs = sw.Elapsed.TotalMilliseconds;
            _enhancedImageUrl = Services.ImageDisplayHelper.ToDataUrl(JS, result.RgbaPixels, result.Width, result.Height);
            Console.WriteLine($"[SuperRes] {_inferenceMs:F0}ms, {_imageWidth}x{_imageHeight} → {result.Width}x{result.Height}");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[SuperRes] Error: {ex.Message}\n{ex.StackTrace}");
        }

        _isRunning = false;
        StateHasChanged();
    }

    private async Task HandleBackendChange(string backend)
    {
        if (backend == _selectedBackend && _isModelLoaded) return;
        _selectedBackend = backend;
        _enhancedImageUrl = null;

        _pipeline?.Dispose();
        _session?.Dispose();
        _accelerator?.Dispose();
        _pipeline = null;
        _session = null;
        _accelerator = null;

        await LoadBackendAndModelAsync();
    }

    private void DownloadResult()
    {
        if (_enhancedImageUrl == null) return;
        try
        {
            using var document = JS.Get<SpawnDev.BlazorJS.JSObjects.Document>("document");
            using var link = document.CreateElement<SpawnDev.BlazorJS.JSObjects.HTMLAnchorElement>("a");
            link.Href = _enhancedImageUrl;
            link.Download = "enhanced-3x.png";
            using var body = document.Body!;
            body.AppendChild(link);
            link.Click();
            body.RemoveChild(link);
        }
        catch { }
    }

    private void ClearResult()
    {
        _enhancedImageUrl = null;
        _imageDataUrl = null;
        _rgbaPixels = null;
        StateHasChanged();
    }

    public void Dispose()
    {
        _pipeline?.Dispose();
        _session?.Dispose();
        _accelerator?.Dispose();
        _context?.Dispose();
    }
}
