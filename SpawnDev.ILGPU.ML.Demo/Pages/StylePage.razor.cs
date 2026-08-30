using ILGPU;
using ILGPU.Runtime;
using Microsoft.AspNetCore.Components;
using SpawnDev.SpawnJS;
using SpawnDev.SpawnJS.JSObjects;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.ILGPU.Rendering;
using SpawnDev.ILGPU.WebGPU;
using SpawnDev.ILGPU.WebGPU.Backend;
using System.Diagnostics;

namespace SpawnDev.ILGPU.ML.Demo.Pages;

public partial class StylePage : IDisposable
{
    [Inject] SpawnJSRuntime JS { get; set; } = default!;
    [Inject] SpawnDev.WebTorrent.WebTorrentClient Torrents { get; set; } = default!;
    [Inject] HttpClient Http { get; set; } = default!;

    private InferenceSession? _session;
    private StyleTransferPipeline? _pipeline;
    private Context? _context;
    private Accelerator? _accelerator;
    private int[]? _rgbaPixels;
    private int _imageWidth, _imageHeight;

    // GPU-direct canvas rendering for zero-copy output
    private ElementReference _outputCanvas;
    private ICanvasRenderer? _canvasRenderer;
    private bool _processingFrame;
    private MemoryBuffer2D<int, Stride2D.DenseX>? _lastFrameBuffer;

    // OPT-IN: no style model is downloaded/loaded on page entry. It loads on the first real
    // interaction (picking an image, selecting a style, or starting the webcam).

    private async Task LoadStyle(string styleName)
    {
        try
        {
            _isModelLoading = true;
            _selectedStyle = styleName;
            StateHasChanged();

            // Create context with all browser backends (first time only)
            if (_context == null)
            {
                var builder = MLContext.Create();
                await builder.AllAcceleratorsAsync();
                _context = builder.ToContext();
            }

            // Create accelerator if needed
            if (_accelerator == null)
            {
                _accelerator = await CreateAcceleratorForBackendAsync(_selectedBackend);
                if (_accelerator == null)
                {
                    Console.WriteLine($"[Style] No {_selectedBackend} device available");
                    _isModelLoading = false;
                    StateHasChanged();
                    return;
                }
            }

            _session?.Dispose();
            _pipeline?.Dispose();

            var modelName = styleName.ToLowerInvariant().Replace(" ", "-");
            var styleRepo = modelName switch
            {
                "mosaic" => ModelHub.KnownModels.StyleMosaic,
                "candy" => ModelHub.KnownModels.StyleCandy,
                "rain-princess" => ModelHub.KnownModels.StyleRainPrincess,
                "udnie" => ModelHub.KnownModels.StyleUdnie,
                "pointilism" => ModelHub.KnownModels.StylePointilism,
                _ => throw new ArgumentException($"Unknown style: {modelName}")
            };
            using var hub = new ModelHub(JS);
            // Weights are delivered as a LAZY-HASH torrent: streamable with random access, OPFS-cached
            // by piece, restored on reload with no re-download, and seeded to peers.
            _session = await InferenceSession.CreateFromHuggingFaceAsync(
                _accelerator, hub, styleRepo, $"{modelName}-9.onnx",
                webTorrent: Torrents, http: Http);
            _pipeline = new StyleTransferPipeline(_session, _accelerator);

            _isModelLoaded = true;
            Console.WriteLine($"[Style] {styleName} loaded on {_selectedBackend}: {_session}");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[Style] Error: {ex.Message}");
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
        try
        {
            using var blob = new SpawnDev.SpawnJS.JSObjects.Blob(
                new[] { imageBytes }, new SpawnDev.SpawnJS.JSObjects.BlobOptions { Type = "image/jpeg" });
            using var window = JS.Get<SpawnDev.SpawnJS.JSObjects.Window>("window");
            using var bitmap = await window.CreateImageBitmap(blob);
            int w = (int)bitmap.Width; int h = (int)bitmap.Height;
            using var canvas = new SpawnDev.SpawnJS.JSObjects.HTMLCanvasElement();
            canvas.Width = w; canvas.Height = h;
            using var ctx = canvas.Get2DContext();
            ctx.DrawImage(bitmap, 0, 0, w, h);
            using var imageData = ctx.GetImageData(0, 0, w, h);
            using var data = imageData.Data;
            _rgbaPixels = data.Read<int>();
            _imageWidth = w; _imageHeight = h;

            // Opt-in load: pull the style model on first use (user picked an image), not on page entry.
            if (!_isModelLoaded && !_isModelLoading)
                await LoadStyle(_selectedStyle);
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

    private async Task HandleBackendChange(string backend)
    {
        if (backend == _selectedBackend && _isModelLoaded) return;
        bool wasLoaded = _isModelLoaded;
        _selectedBackend = backend;
        _styledImageUrl = null;

        _pipeline?.Dispose();
        _session?.Dispose();
        _accelerator?.Dispose();
        _pipeline = null;
        _session = null;
        _accelerator = null;
        _isModelLoaded = false;

        // Opt-in: only reload the current style on the new backend if it was already loaded;
        // switching backend before first use must not trigger a download.
        if (wasLoaded)
            await LoadStyle(_selectedStyle);
        else
            StateHasChanged();
    }

    private void DownloadResult()
    {
        if (_styledImageUrl == null) return;
        try
        {
            using var document = JS.Get<SpawnDev.SpawnJS.JSObjects.Document>("document");
            using var link = document.CreateElement<SpawnDev.SpawnJS.JSObjects.HTMLAnchorElement>("a");
            link.Href = _styledImageUrl;
            link.Download = $"styled-{_selectedStyle.ToLowerInvariant().Replace(" ", "-")}.png";
            using var body = document.Body!;
            body.AppendChild(link);
            link.Click();
            body.RemoveChild(link);
        }
        catch { }
    }

    private void ClearResult()
    {
        _styledImageUrl = null;
        _imageDataUrl = null;
        _rgbaPixels = null;
        StateHasChanged();
    }

    private List<Components.ImageDropZone.SampleImage> _sampleImages = new()
    {
        new() { Label = "Cat", Url = "samples/cat.jpg" },
        new() { Label = "Landscape", Url = "samples/landscape.jpg" },
        new() { Label = "Street", Url = "samples/street.jpg" },
    };

    public void Dispose()
    {
        _lastFrameBuffer?.Dispose();
        _canvasRenderer?.Dispose();
        _pipeline?.Dispose();
        _session?.Dispose();
        _accelerator?.Dispose();
        _context?.Dispose();
    }
}
