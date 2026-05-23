using ILGPU;
using ILGPU.Runtime;
using Microsoft.AspNetCore.Components;
using SpawnDev.BlazorJS;
using SpawnDev.BlazorJS.JSObjects;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnDev.ILGPU.Rendering;
using SpawnDev.ILGPU.WebGPU;
using SpawnDev.ILGPU.WebGPU.Backend;
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

    // GPU-direct rendering. Raw depth stays on GPU so palette switches re-dispatch only
    // the colormap kernel — no re-inference, no CPU readback of depth values.
    private MemoryBuffer2D<int, Stride2D.DenseX>? _gpuDepthBuffer;
    private MemoryBuffer1D<float, Stride1D.Dense>? _gpuRawDepth;
    private float _gpuRawMinDepth, _gpuRawMaxDepth;

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
            _statusMessage = "Loading Depth Anything V2 Small (95MB)...";
            StateHasChanged();

            // Create context with all browser backends (first time only)
            if (_context == null)
            {
                var builder = MLContext.Create();
                await builder.AllAcceleratorsAsync();
                _context = builder.ToContext();
            }

            _accelerator = await CreateAcceleratorForBackendAsync(_selectedBackend);
            if (_accelerator == null)
            {
                _statusMessage = $"No {_selectedBackend} device available";
                _isModelLoading = false;
                StateHasChanged();
                return;
            }

            using var hub = new ModelHub(JS);
            _session = await InferenceSession.CreateFromHuggingFaceAsync(
                _accelerator, hub,
                ModelHub.KnownModels.DepthAnythingV2Small, ModelHub.KnownFiles.OnnxModel,
                inputShapes: new Dictionary<string, int[]>
                {
                    ["pixel_values"] = new[] { 1, 3, 518, 518 }
                });

            _pipeline = new DepthEstimationPipeline(_session, _accelerator);

            _isModelLoaded = true;
            _statusMessage = null;
        }
        catch (Exception ex)
        {
            _statusMessage = $"Error loading model: {ex.Message}";
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
        _depthImageUrl = null;
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
        if (_pipeline == null || _rgbaPixels == null || _accelerator == null) return;
        _isRunning = true;
        _statusMessage = "Running depth estimation...";
        StateHasChanged();
        await Task.Yield();

        try
        {
            var sw = Stopwatch.StartNew();

            // GPU-direct: inference returns the raw depth on the accelerator + min/max.
            // The colormap is a separate accelerator-side step; on palette change we
            // re-dispatch ApplyColormapGpuAsync against the cached raw depth — no
            // re-inference and no host readback of depth values.
            _gpuRawDepth?.Dispose();
            var (rawDepth, minD, maxD, w, h) = await _pipeline.EstimateGpuRawAsync(
                _rgbaPixels, _imageWidth, _imageHeight);
            _gpuRawDepth = rawDepth;
            _gpuRawMinDepth = minD;
            _gpuRawMaxDepth = maxD;
            _depthWidth = w;
            _depthHeight = h;

            sw.Stop();
            _inferenceMs = sw.Elapsed.TotalMilliseconds;

            await RecolorDepthGpuAsync();

            _statusMessage = null;
        }
        catch (Exception ex)
        {
            _statusMessage = $"Inference error: {ex.Message}";
        }

        _isRunning = false;
        StateHasChanged();
    }

    private async Task HandleBackendChange(string backend)
    {
        if (backend == _selectedBackend && _isModelLoaded) return;
        _selectedBackend = backend;
        _depthImageUrl = null;

        // Free cached GPU buffers tied to the old accelerator before tearing it down.
        _gpuDepthBuffer?.Dispose();
        _gpuDepthBuffer = null;
        _gpuRawDepth?.Dispose();
        _gpuRawDepth = null;

        _pipeline?.Dispose();
        _session?.Dispose();
        _accelerator?.Dispose();
        _pipeline = null;
        _session = null;
        _accelerator = null;

        await LoadBackendAndModelAsync();
    }

    /// <summary>
    /// Re-apply the colormap on the accelerator using the cached raw depth GPU buffer
    /// and current <see cref="_colorPalette"/>. No inference re-run, no host readback of
    /// depth values — only the final colored RGBA is read to CPU for the PNG data-URL
    /// display path. Called on every inference completion and every palette change.
    /// </summary>
    private async Task RecolorDepthGpuAsync()
    {
        if (_gpuRawDepth == null || _accelerator == null || _pipeline == null) return;

        int paletteId = SpawnDev.ILGPU.ML.Kernels.ImagePostprocessKernel.PaletteFromName(_colorPalette);
        _gpuDepthBuffer?.Dispose();
        _gpuDepthBuffer = await _pipeline.ApplyColormapGpuAsync(
            _gpuRawDepth.View, _depthWidth, _depthHeight,
            _gpuRawMinDepth, _gpuRawMaxDepth, paletteId);

        // Read the colored RGBA buffer to CPU for the BeforeAfterSlider <img> data URL.
        // This is the display path's required readback — it's just RGBA pixels, not the
        // depth values themselves, so the depth data stays GPU-side end-to-end.
        using var readBuf = _accelerator.Allocate1D<int>(_depthWidth * _depthHeight);
        readBuf.View.CopyFrom(_gpuDepthBuffer.View.BaseView);
        await _accelerator.SynchronizeAsync();
        var pixels = await readBuf.CopyToHostAsync<int>(0, _depthWidth * _depthHeight);
        _depthImageUrl = Services.ImageDisplayHelper.ToDataUrl(JS, pixels, _depthWidth, _depthHeight);
    }

    /// <summary>
    /// Razor dropdown handler — invoked via <c>@bind:after</c>. Fires the async GPU
    /// colormap re-application and refreshes the UI when done.
    /// </summary>
    private async Task OnPaletteChangedAsync()
    {
        await RecolorDepthGpuAsync();
        StateHasChanged();
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
        _rgbaPixels = null;
        _gpuDepthBuffer?.Dispose();
        _gpuDepthBuffer = null;
        _gpuRawDepth?.Dispose();
        _gpuRawDepth = null;
        _statusMessage = null;
        StateHasChanged();
    }

    public void Dispose()
    {
        _gpuDepthBuffer?.Dispose();
        _gpuRawDepth?.Dispose();
        _pipeline?.Dispose();
        _session?.Dispose();
        _accelerator?.Dispose();
        _context?.Dispose();
    }
}
