using ILGPU;
using ILGPU.Runtime;
using Microsoft.AspNetCore.Components;
using SpawnDev.SpawnJS;
using SpawnDev.SpawnJS.JSObjects;
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
    [Inject] SpawnJSRuntime JS { get; set; } = default!;
    [Inject] HttpClient Http { get; set; } = default!;

    private InferenceSession? _session;
    private DepthEstimationPipeline? _pipeline;
    private Context? _context;
    private Accelerator? _accelerator;
    private int[]? _rgbaPixels;
    private int _imageWidth, _imageHeight;

    // GPU-direct rendering. Raw depth stays on GPU so palette switches re-dispatch only
    // the colormap kernel; the colored GPU buffer renders straight to the slider's
    // <canvas> via ICanvasRenderer — no PNG encode, no base64 data URL, no Blob/URL
    // shuffling, no host readback of depth values.
    private MemoryBuffer2D<int, Stride2D.DenseX>? _gpuDepthBuffer;
    private MemoryBuffer1D<float, Stride1D.Dense>? _gpuRawDepth;
    private float _gpuRawMinDepth, _gpuRawMaxDepth;
    private ICanvasRenderer? _canvasRenderer;
    private bool _canvasReady;
    /// <summary>Slider's After-side canvas. Saved on canvas-ready so DownloadResult can
    /// pull a PNG blob URL from it on demand without holding a persistent data URL.</summary>
    private ElementReference? _afterCanvasRef;

    // OPT-IN: Depth Anything V2 Small (95MB) is NOT downloaded/loaded on page entry. It loads on
    // the first real interaction (picking an image) — see HandleImageLoaded.

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

            _pipeline = new DepthEstimationPipeline(_session, _accelerator);   // capture/replay ON by pipeline default

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
        // New image: drop any prior depth result so the slider unmounts; the canvas-ready
        // callback will re-fire when the new result mounts the slider again.
        _hasDepthResult = false;
        _canvasReady = false;
        _canvasRenderer?.Dispose();
        _canvasRenderer = null;
        _statusMessage = null;

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

            // Opt-in load: pull the model on first use (user picked an image), not on page entry.
            if (!_isModelLoaded && !_isModelLoading)
                await LoadBackendAndModelAsync();
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
            // re-inference, no host readback of depth values.
            _gpuRawDepth?.Dispose();
            var (rawDepth, minD, maxD, w, h) = await _pipeline.EstimateGpuRawAsync(
                _rgbaPixels, _imageWidth, _imageHeight);
            _gpuRawDepth = rawDepth;
            _gpuRawMinDepth = minD;
            _gpuRawMaxDepth = maxD;
            _depthWidth = w;
            _depthHeight = h;
            // Diagnostic: visual flat-blue investigation. Tells us if the raw depth
            // buffer is actually populated, what its value range is, and what the
            // canvas dimensions are when the slider mounts.
            Console.WriteLine($"[Depth] backend={_selectedBackend} src={_imageWidth}x{_imageHeight} -> depth={w}x{h} min={minD:F4} max={maxD:F4} range={(maxD-minD):F4}");

            sw.Stop();
            _inferenceMs = sw.Elapsed.TotalMilliseconds;

            // Setting _hasDepthResult mounts the BeforeAfterSlider; its canvas fires
            // OnAfterCanvasReady on first render, which attaches the ICanvasRenderer
            // and immediately presents the colored buffer. RecolorDepthGpuAsync
            // builds the colored buffer first so it's ready when the canvas attaches.
            await RecolorDepthGpuAsync(present: _canvasReady);
            _hasDepthResult = true;

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
        bool wasLoaded = _isModelLoaded;
        _selectedBackend = backend;
        _hasDepthResult = false;

        // Free cached GPU buffers + the canvas renderer tied to the old accelerator
        // before tearing it down. The canvas-ready callback will fire again when the
        // slider remounts with the new accelerator.
        _canvasRenderer?.Dispose();
        _canvasRenderer = null;
        _canvasReady = false;
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
        _isModelLoaded = false;

        // Opt-in: only re-load if the model was already loaded; switching backend before
        // first use must not trigger a download.
        if (wasLoaded)
            await LoadBackendAndModelAsync();
        else
            StateHasChanged();
    }

    /// <summary>
    /// Re-apply the colormap on the accelerator using the cached raw depth GPU buffer
    /// and the current <see cref="_colorPalette"/>. The colored GPU buffer is then
    /// presented straight to the slider's <c>&lt;canvas&gt;</c> via
    /// <see cref="ICanvasRenderer"/> (backend-optimized: WebGPU texture copy / WebGL
    /// blit / Wasm putImageData — no PNG encode, no data URL, no Blob).
    ///
    /// When <paramref name="present"/> is false, the colored buffer is produced but
    /// not presented — caller should present once the canvas is attached
    /// (initial inference case where the slider hasn't mounted yet).
    /// </summary>
    private async Task RecolorDepthGpuAsync(bool present = true)
    {
        if (_gpuRawDepth == null || _accelerator == null || _pipeline == null)
        {
            Console.WriteLine($"[Depth] RecolorDepthGpuAsync SKIPPED: rawDepth={_gpuRawDepth != null} accel={_accelerator != null} pipeline={_pipeline != null}");
            return;
        }

        int paletteId = SpawnDev.ILGPU.ML.Kernels.ImagePostprocessKernel.PaletteFromName(_colorPalette);
        _gpuDepthBuffer?.Dispose();
        _gpuDepthBuffer = await _pipeline.ApplyColormapGpuAsync(
            _gpuRawDepth.View, _depthWidth, _depthHeight,
            _gpuRawMinDepth, _gpuRawMaxDepth, paletteId);

        Console.WriteLine($"[Depth] Recolor palette={_colorPalette}(id={paletteId}) dim={_depthWidth}x{_depthHeight} present={present} renderer={(_canvasRenderer != null ? "ATTACHED" : "NULL")}");


        if (present && _canvasRenderer != null)
        {
            try
            {
                await _canvasRenderer.PresentAsync(_gpuDepthBuffer);
                Console.WriteLine("[Depth] PresentAsync completed");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[Depth] PresentAsync THREW: {ex.GetType().Name}: {ex.Message}");
                Console.WriteLine($"[Depth] StackTrace: {ex.StackTrace}");
            }
        }
    }

    /// <summary>
    /// Razor dropdown handler — invoked via <c>@bind:after</c>. Re-runs the GPU colormap
    /// and presents to the canvas (no inference, no CPU readback).
    /// </summary>
    private async Task OnPaletteChangedAsync()
    {
        await RecolorDepthGpuAsync();
        StateHasChanged();
    }

    /// <summary>
    /// Fires once when the BeforeAfterSlider's After-side canvas is rendered. Wraps the
    /// ElementReference as an HTMLCanvasElement, creates the backend-appropriate
    /// ICanvasRenderer, attaches it, and presents the colored buffer if inference has
    /// already produced one. From this point on every palette switch / new inference just
    /// dispatches one colormap kernel + one PresentAsync.
    /// </summary>
    private async Task OnAfterCanvasReady(ElementReference canvasRef)
    {
        Console.WriteLine($"[Depth] OnAfterCanvasReady FIRED accel={(_accelerator != null ? _accelerator.AcceleratorType.ToString() : "NULL")} bufferReady={_gpuDepthBuffer != null}");
        if (_accelerator == null) return;
        _canvasRenderer?.Dispose();
        _canvasRenderer = CanvasRendererFactory.Create(_accelerator);
        using var canvasEl = canvasRef.As<HTMLCanvasElement>();
        _canvasRenderer.AttachCanvas(canvasEl);
        _afterCanvasRef = canvasRef;
        _canvasReady = true;
        Console.WriteLine($"[Depth] Canvas attached, renderer={_canvasRenderer?.GetType().Name}");

        if (_gpuDepthBuffer != null)
        {
            try
            {
                Console.WriteLine($"[Depth] About to call PresentAsync on {_gpuDepthBuffer.GetType().Name} extent={_gpuDepthBuffer.Extent}");
                await _canvasRenderer.PresentAsync(_gpuDepthBuffer);
                Console.WriteLine("[Depth] Initial PresentAsync completed");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[Depth] PresentAsync THREW: {ex.GetType().Name}: {ex.Message}");
                Console.WriteLine($"[Depth] StackTrace: {ex.StackTrace}");
            }
        }
        else
        {
            Console.WriteLine("[Depth] _gpuDepthBuffer was NULL when canvas attached");
        }
    }

    private void DownloadResult()
    {
        // The depth result lives on the slider's canvas (rendered there by
        // ICanvasRenderer). Pull a PNG data URL from the canvas on demand for the
        // anchor download — only happens on user click, not the per-frame path.
        if (_afterCanvasRef == null) return;
        try
        {
            using var canvasEl = _afterCanvasRef?.As<HTMLCanvasElement>();
            var pngUrl = canvasEl.ToDataURL("image/png");
            using var document = JS.Get<SpawnDev.SpawnJS.JSObjects.Document>("document");
            using var link = document.CreateElement<SpawnDev.SpawnJS.JSObjects.HTMLAnchorElement>("a");
            link.Href = pngUrl;
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
        _hasDepthResult = false;
        _imageDataUrl = null;
        _rgbaPixels = null;
        _canvasRenderer?.Dispose();
        _canvasRenderer = null;
        _canvasReady = false;
        _afterCanvasRef = null;
        _gpuDepthBuffer?.Dispose();
        _gpuDepthBuffer = null;
        _gpuRawDepth?.Dispose();
        _gpuRawDepth = null;
        _statusMessage = null;
        StateHasChanged();
    }

    public void Dispose()
    {
        _canvasRenderer?.Dispose();
        _gpuDepthBuffer?.Dispose();
        _gpuRawDepth?.Dispose();
        _pipeline?.Dispose();
        _session?.Dispose();
        _accelerator?.Dispose();
        _context?.Dispose();
    }
}
