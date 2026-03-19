using ILGPU;
using ILGPU.Runtime;
using Microsoft.AspNetCore.Components;
using SpawnDev.BlazorJS;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnDev.ILGPU.WebGPU;
using System.Diagnostics;

namespace SpawnDev.ILGPU.ML.Demo.Pages;

public partial class DepthPage : IDisposable
{
    [Inject] BlazorJSRuntime JS { get; set; } = default!;
    [Inject] HttpClient Http { get; set; } = default!;

    private InferenceSession? _session;
    private DepthEstimationPipeline? _pipeline;
    private Context? _context;
    private Accelerator? _accelerator;
    private int[]? _rgbaPixels;
    private int _imageWidth, _imageHeight;

    protected override async Task OnAfterRenderAsync(bool firstRender)
    {
        if (firstRender)
            await LoadModel();
    }

    private async Task LoadModel()
    {
        try
        {
            _isModelLoading = true;
            _statusMessage = "Loading Depth Anything V2 Small (95MB)...";
            StateHasChanged();

            var builder = MLContext.Create();
            await builder.WebGPU();
            _context = builder.ToContext();
            var devices = _context.GetWebGPUDevices();
            if (devices.Count == 0)
            {
                _statusMessage = "No WebGPU devices found";
                _isModelLoading = false;
                StateHasChanged();
                return;
            }
            _accelerator = await devices[0].CreateAcceleratorAsync(_context);

            // Try direct .onnx loading first, fall back to extracted format
            try
            {
                _session = await InferenceSession.CreateFromOnnxAsync(
                    _accelerator, Http, "models/depth-anything-v2-small/model.onnx");
            }
            catch
            {
                _session = await InferenceSession.CreateAsync(
                    _accelerator, Http, "models/depth-anything-v2-small");
            }

            _pipeline = new DepthEstimationPipeline(_session, _accelerator);

            _isModelLoaded = true;
            _isModelLoading = false;
            _statusMessage = null;
        }
        catch (Exception ex)
        {
            _isModelLoading = false;
            _statusMessage = $"Error loading model: {ex.Message}";
        }
        StateHasChanged();
    }

    private async Task HandleImageLoaded(byte[] imageBytes)
    {
        _depthImageUrl = null;
        _depthMap = null;
        _statusMessage = null;

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

            if (_isModelLoaded)
                await RunDepthEstimation();
        }
        catch (Exception ex)
        {
            _statusMessage = $"Image decode error: {ex.Message}";
            StateHasChanged();
        }
    }

    private async Task RunDepthEstimation()
    {
        if (_pipeline == null || _rgbaPixels == null) return;
        _isRunning = true;
        _statusMessage = "Running depth estimation...";
        StateHasChanged();
        await Task.Yield();

        try
        {
            var sw = Stopwatch.StartNew();
            var result = await _pipeline.EstimateAsync(_rgbaPixels, _imageWidth, _imageHeight);
            sw.Stop();
            _inferenceMs = sw.Elapsed.TotalMilliseconds;

            _depthMap = result.DepthMap;
            _depthWidth = result.Width;
            _depthHeight = result.Height;
            _minDepth = result.MinDepth;
            _maxDepth = result.MaxDepth;

            // Generate colorized depth image
            RecolorDepth();

            _statusMessage = null;
        }
        catch (Exception ex)
        {
            _statusMessage = $"Inference error: {ex.Message}";
        }

        _isRunning = false;
        StateHasChanged();
    }

    private void HandleBackendChange(string backend)
    {
        _selectedBackend = backend;
    }

    private void DownloadResult()
    {
        if (_depthImageUrl == null) return;
        try
        {
            using var document = JS.Get<SpawnDev.BlazorJS.JSObjects.Document>("document");
            using var link = document.CreateElement<SpawnDev.BlazorJS.JSObjects.HTMLAnchorElement>("a");
            link.Href = _depthImageUrl;
            link.Download = $"depth-{_colorPalette}.png";
            using var body = document.Body!;
            body.AppendChild(link);
            link.Click();
            body.RemoveChild(link);
        }
        catch { }
    }

    private void ClearResult()
    {
        _depthImageUrl = null;
        _imageDataUrl = null;
        _depthMap = null;
        _rgbaPixels = null;
        _statusMessage = null;
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
