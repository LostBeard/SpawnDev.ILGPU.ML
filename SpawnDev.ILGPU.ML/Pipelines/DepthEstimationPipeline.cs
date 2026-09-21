using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnDev.ILGPU.ML.Tensors;
using TypedArray = SpawnDev.SpawnJS.JSObjects.TypedArray;

namespace SpawnDev.ILGPU.ML.Pipelines;

/// <summary>
/// Result from depth estimation — depth map as float array.
/// </summary>
public record DepthResult(float[] DepthMap, int Width, int Height, float MinDepth, float MaxDepth);

/// <summary>
/// High-level monocular depth estimation pipeline.
/// Wraps InferenceSession with image preprocessing and depth postprocessing.
///
/// Usage:
///   var pipeline = new DepthEstimationPipeline(session, accelerator);
///   var result = await pipeline.EstimateAsync(rgbaPixels, width, height);
///   // result.DepthMap is [Height × Width] normalized depth values
/// </summary>
public class DepthEstimationPipeline : IDisposable
{
    private readonly InferenceSession _session;
    private readonly Accelerator _accelerator;
    private readonly Kernels.ImagePreprocessKernel _preprocess;
    private readonly Kernels.ImagePostprocessKernel _postprocess;
    private readonly int _inputSize;

    /// <summary>
    /// Letterbox the input instead of stretching it to the model's square.
    ///
    /// On by default, because it is what Depth Anything expects and the alternative is wrong:
    /// a 3:4 photograph squashed into 518x518 enters the network 33% too wide, and the geometry
    /// it returns is distorted in a way that reads as bad depth. Settable so a caller that has
    /// calibrated against the old behaviour can keep it.
    /// </summary>
    public bool PreserveAspect { get; set; } = true;

    /// <summary>
    /// Opt-in graph capture for the VIDEO / repeat-inference path (default off; CUDA + WebGPU, no-op elsewhere).
    /// When set, <see cref="EstimateGpuRawAsync"/> captures the forward once at the first resolution it sees and
    /// REPLAYS it for every subsequent frame at that resolution. CUDA: a single cuGraphLaunch instead of the
    /// ~2524-node loop (~3x on DAv3-Small, bit-identical). WebGPU: a single interop crossing re-encodes the
    /// captured dispatch plan (<see cref="WebGPUGraphCapture"/>) - the 10.8s-warm-direct → ~65-75ms/frame
    /// production path (bit-identical, beats ORT-Web's 73ms warm). The first frame pays the capture cost
    /// (a few warm forwards); the resolution is re-captured if it changes.
    /// </summary>
    public bool EnableGraphCapture { get; set; } = true;   // ON by default: consumers forgetting the
    // flag is how /depth ran 6s-per-estimate direct forwards for weeks while the 66ms replay path
    // sat unused (Captain's rule: if it should always be on, it lives in the pipeline). Opt OUT for
    // one-shot workloads where the first-capture warmup (a few forwards) outweighs the replay win.
    // CUDA + WebGPU; no-op elsewhere. Bit-exactness gated (video gate + Captured_518_MatchesHost).
    private CudaGraphCapture? _capture;
    private WebGPUGraphCapture? _webGpuCapture;
    private MemoryBuffer1D<float, Stride1D.Dense>? _captureInputBuf;
    private int[]? _captureShape;

    /// <summary>Underlying session — diagnostics (weight presence) and advanced callers.</summary>
    public InferenceSession Session => _session;

    public DepthEstimationPipeline(InferenceSession session, Accelerator accelerator,
        int inputSize = 0)
    {
        _session = session;
        _accelerator = accelerator;
        _preprocess = new Kernels.ImagePreprocessKernel(accelerator);
        _postprocess = new Kernels.ImagePostprocessKernel(accelerator);
        // Derive input size from session's compiled input shapes if not specified.
        // Prevents mismatch between preprocessing resolution and compiled graph shapes,
        // which causes silent GPU memory corruption (OOB writes from Conv kernels).
        if (inputSize <= 0)
        {
            var firstShape = session.InputShapes.Values.FirstOrDefault();
            inputSize = firstShape != null && firstShape.Length >= 4 ? firstShape[^1] : 518;
        }
        _inputSize = inputSize;
    }

    /// <summary>
    /// The compiled model's input tensor shape for a CHW preprocessed frame. DAv2 = 4-D [1,3,H,W];
    /// DAv3 = 5-D [1,1,3,H,W] (the num_images dim). Same CHW element count, different rank — we MUST feed the
    /// rank the graph was compiled for, or the model reads the channel dim (3) as num_images and returns garbage.
    /// </summary>
    private int[] InputTensorShape()
        => _session.InputShapes.TryGetValue(_session.InputNames[0], out var s)
           && s.Length >= 4 && s.Aggregate(1, (a, b) => a * b) == 3 * _inputSize * _inputSize
            ? s : new[] { 1, 3, _inputSize, _inputSize };

    /// <summary>
    /// Create a depth pipeline from ONNX streams — zero-copy JS→GPU on browser (the model bytes never enter the
    /// .NET/WASM managed heap). Pass <paramref name="externalDataStream"/> for external-data models like DAv3
    /// (model.onnx structure + model.onnx_data weights). Single-file models: leave it null.
    /// </summary>
    public static async Task<DepthEstimationPipeline> CreateFromStreamsAsync(
        Accelerator accelerator, System.IO.Stream modelStream, System.IO.Stream? externalDataStream = null,
        Action<string, int>? onProgress = null, Dictionary<string, int[]>? inputShapes = null,
        int inputSize = 0, CancellationToken ct = default)
    {
        var session = await InferenceSession.CreateFromOnnxStreamAsync(accelerator, modelStream,
            onProgress: onProgress, inputShapes: inputShapes, externalDataStream: externalDataStream, ct: ct)
            .ConfigureAwait(false);
        return new DepthEstimationPipeline(session, accelerator, inputSize);
    }

    /// <summary>
    /// One-call factory: download (WebTorrent + OPFS cache via <paramref name="hubStream"/>) and zero-copy
    /// stream a depth model straight to the GPU, wrapped in a ready pipeline — the model bytes never touch the
    /// .NET/WASM heap. Mirrors Transformers.js <c>pipeline('depth-estimation', repoId)</c>. Handles
    /// external-data models (DAv3 = model.onnx + model.onnx_data) automatically.
    /// <code>
    ///   var pipe = await DepthEstimationPipeline.CreateFromHubAsync(acc, hubStream,
    ///                  "onnx-community/depth-anything-v3-small",
    ///                  inputShapes: new(){ ["pixel_values"] = new[]{1,1,3,518,518} });
    ///   var depth = await pipe.EstimateGpuAsync(rgba, w, h);   // zero-copy end to end
    /// </code>
    /// </summary>
    /// <remarks>
    /// ⚠️ BREAKING (2026-09-14): this took <c>HubModelStream</c>, which requires a <c>WebTorrentClient</c> -
    /// so this pipeline could not be used at all without WebTorrent. It now takes
    /// <see cref="Hub.IModelSource"/>. Pass <see cref="Hub.HubModelSource"/> for plain HTTP + OPFS (the
    /// default), or a <c>HubModelStream</c> for torrent delivery - it implements the same interface, so the
    /// torrent path is an opt-in upgrade and this call is otherwise unchanged.
    /// </remarks>
    public static async Task<DepthEstimationPipeline> CreateFromHubAsync(
        Accelerator accelerator, Hub.IModelSource source, string repoId,
        string modelFile = "onnx/model.onnx", string externalDataFile = "onnx/model.onnx_data",
        Action<string, int>? onProgress = null, Dictionary<string, int[]>? inputShapes = null,
        int inputSize = 0, CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(source);
        onProgress?.Invoke("open", 0);
        System.IO.Stream modelStream;
        System.IO.Stream? extData = null;
        if (!string.IsNullOrEmpty(externalDataFile))
        {
            // External-data model (DAv3): model.onnx is a SMALL structure file (weights live in model.onnx_data).
            // Fetch the structure whole (KBs) and STREAM only the big weights file — keeps the 100+ MB weights
            // off the managed heap. On the torrent source it also avoids a lazy-hash same-directory collision
            // that otherwise gave the model.onnx_data stream model.onnx's length (both live under onnx/).
            //
            // ⚠️ FAIL LOUD when the external-data file is requested but missing. Swallowing OpenAsync used to
            // leave extData=null, so large initializers (DAv3 pos-embed `/backbone/Transpose_output_0`) never
            // uploaded — Resize then threw "Tensor not found (producerOp=NONE elideBlocked=True)". Single-file
            // models must pass externalDataFile: "" (DAv2) so they take the else branch below.
            var modelBytes = await source.FetchBytesAsync(repoId, modelFile, ct).ConfigureAwait(false);
            modelStream = new System.IO.MemoryStream(modelBytes);
            try
            {
                extData = await source.OpenAsync(repoId, externalDataFile, ct).ConfigureAwait(false);
            }
            catch (Exception ex)
            {
                await modelStream.DisposeAsync().ConfigureAwait(false);
                throw new InvalidOperationException(
                    $"External-data model '{repoId}' requires '{externalDataFile}' but it could not be opened. " +
                    $"Pass externalDataFile: \"\" for single-file ONNX exports. Inner: {ex.Message}", ex);
            }
        }
        else
        {
            // Single-file model: stream model.onnx directly (weights embedded — keep them off the heap).
            modelStream = await source.OpenAsync(repoId, modelFile, ct).ConfigureAwait(false);
        }
        onProgress?.Invoke("open", 100);
        try
        {
            return await CreateFromStreamsAsync(accelerator, modelStream, extData,
                onProgress, inputShapes, inputSize, ct).ConfigureAwait(false);
        }
        finally
        {
            // Weights are on the GPU by now — the streams are done.
            await modelStream.DisposeAsync().ConfigureAwait(false);
            if (extData != null) await extData.DisposeAsync().ConfigureAwait(false);
        }
    }

    /// <summary>
    /// Estimate depth from an RGBA image (managed pixels — desktop / oracle).
    /// Returns a depth map normalized to [0, 1] (higher = closer).
    ///
    /// Output dimensions:
    ///   outputWidth = 0 &amp;&amp; outputHeight = 0 → match source (width, height) — default,
    ///       preserves aspect ratio so the depth map aligns 1:1 with the input.
    ///   outputWidth > 0 &amp;&amp; outputHeight = 0 → use outputWidth, derive height from source aspect.
    ///   outputWidth = 0 &amp;&amp; outputHeight > 0 → use outputHeight, derive width from source aspect.
    ///   outputWidth > 0 &amp;&amp; outputHeight > 0 → exact size (may not preserve aspect).
    /// Resize is done on the accelerator via bilinear interpolation — no CPU readback of the raw map.
    /// </summary>
    /// <remarks>
    /// ⚠️ IN A BROWSER, prefer <see cref="EstimateAsync(TypedArray, int, int, int, int)"/> or a
    /// GPU-resident view overload — this path pulls the frame onto the managed heap solely to upload it.
    /// </remarks>
    public async Task<DepthResult> EstimateAsync(int[] rgbaPixels, int width, int height,
        int outputWidth = 0, int outputHeight = 0)
    {
        using var rgbaBuf = RgbaUpload.FromManaged(_accelerator, rgbaPixels, width, height);
        return await EstimateAsync(rgbaBuf.View, width, height, outputWidth, outputHeight)
            .ConfigureAwait(false);
    }

    /// <summary>
    /// Browser path: RGBA still a JS typed array (e.g. <c>ImageData.Data</c>). Uploads via
    /// <see cref="MediaInterop.UploadToDevice{T}"/> — pixels never enter the .NET managed heap.
    /// </summary>
    public async Task<DepthResult> EstimateAsync(TypedArray rgbaPixels, int width, int height,
        int outputWidth = 0, int outputHeight = 0)
    {
        using var rgbaBuf = RgbaUpload.FromTypedArray(_accelerator, rgbaPixels, width, height);
        return await EstimateAsync(rgbaBuf.View, width, height, outputWidth, outputHeight)
            .ConfigureAwait(false);
    }

    /// <summary>
    /// GPU-resident packed RGBA — no upload. Use when the frame is already on the accelerator
    /// (e.g. SpawnScene <c>GpuImage.PackedRgba</c>).
    /// </summary>
    public Task<DepthResult> EstimateAsync(
        ArrayView1D<int, Stride1D.Dense> rgbaPixels, int width, int height,
        int outputWidth = 0, int outputHeight = 0)
        => EstimateHostCoreAsync(rgbaPixels, width, height, outputWidth, outputHeight);

    /// <summary>Same as the view overload; accepts an owned buffer.</summary>
    public Task<DepthResult> EstimateAsync(
        MemoryBuffer1D<int, Stride1D.Dense> rgbaPixels, int width, int height,
        int outputWidth = 0, int outputHeight = 0)
        => EstimateHostCoreAsync(rgbaPixels.View, width, height, outputWidth, outputHeight);

    private async Task<DepthResult> EstimateHostCoreAsync(
        ArrayView1D<int, Stride1D.Dense> rgbaPixels, int width, int height,
        int outputWidth, int outputHeight)
    {
        using var preprocessed = _accelerator.Allocate1D<float>(3 * _inputSize * _inputSize);
        _preprocess.Forward(rgbaPixels, preprocessed.View, width, height, _inputSize, _inputSize,
            preserveAspect: PreserveAspect);

        var inputTensor = new Tensor(preprocessed.View, InputTensorShape());
        var outputs = await _session.RunAsync(new Dictionary<string, Tensor>
        {
            [_session.InputNames[0]] = inputTensor
        }).ConfigureAwait(false);
        await _accelerator.SynchronizeAsync().ConfigureAwait(false);

        var output = outputs[_session.OutputNames[0]];
        int rawSize = output.ElementCount;
        int rawH = output.Shape.Length >= 3 ? output.Shape[^2] : _inputSize;
        int rawW = output.Shape.Length >= 3 ? output.Shape[^1] : _inputSize;
        if (InferenceSession.VerboseLogging) Console.WriteLine($"[Depth CPU] Output: shape=[{string.Join(",", output.Shape)}], elements={rawSize}");

        var (outW, outH) = ResolveOutputSize(width, height, rawW, rawH, outputWidth, outputHeight);
        int outSize = outW * outH;

        var post = _postprocess;
        using var resized = _accelerator.Allocate1D<float>(outSize);
        var (cx1, cy1, cw1, ch1) = ContentRect(width, height, rawW, rawH);
        if (cx1 != 0 || cy1 != 0 || cw1 != rawW || ch1 != rawH)
        {
            // Letterboxed input: only the content rect is real picture.
            post.ResizeBilinearFromRect(
                output.Data.SubView(0, rawSize), rawW, rawH, cx1, cy1, cw1, ch1,
                resized.View, outW, outH);
        }
        else
        {
            var srcView = new Tensors.TensorView<float>(output.Data.SubView(0, rawSize), new[] { rawH, rawW });
            var dstView = new Tensors.TensorView<float>(resized.View, new[] { outH, outW });
            post.ResizeBilinear(srcView, dstView);
        }
        await _accelerator.SynchronizeAsync().ConfigureAwait(false);

        var rawDepth = await resized.CopyToHostAsync<float>(0, outSize).ConfigureAwait(false);
        float min = rawDepth.Min();
        float max = rawDepth.Max();
        float range = max - min;
        var normalized = new float[outSize];
        if (range > 1e-6f)
        {
            for (int i = 0; i < outSize; i++)
                normalized[i] = (rawDepth[i] - min) / range;
        }

        return new DepthResult(normalized, outW, outH, min, max);
    }

    /// <summary>
    /// Resolve final output dimensions for a depth result from caller hints.
    ///   (0, 0) → (srcW, srcH) — match source, preserves aspect.
    ///   (w, 0) → (w, w * srcH / srcW) — preserve source aspect, fit width.
    ///   (0, h) → (h * srcW / srcH, h) — preserve source aspect, fit height.
    ///   (w, h) → (w, h) — exact, may distort.
    /// rawW/rawH are used as a fallback when srcW/srcH are non-positive.
    /// </summary>
    private static (int w, int h) ResolveOutputSize(int srcW, int srcH, int rawW, int rawH,
        int outW, int outH)
    {
        if (srcW <= 0 || srcH <= 0) { srcW = rawW; srcH = rawH; }
        if (outW <= 0 && outH <= 0) return (srcW, srcH);
        if (outW > 0 && outH <= 0)
        {
            int h = (int)MathF.Round(outW * (float)srcH / srcW);
            return (outW, Math.Max(1, h));
        }
        if (outH > 0 && outW <= 0)
        {
            int w = (int)MathF.Round(outH * (float)srcW / srcH);
            return (Math.Max(1, w), outH);
        }
        return (outW, outH);
    }

    /// <summary>
    /// Estimate depth and return a plasma colormap as a GPU MemoryBuffer2D for zero-copy
    /// presentation via ICanvasRenderer. The raw depth values stay on the accelerator;
    /// nothing about them leaves the GPU through this contract.
    ///
    /// Output dimensions follow the same convention as <see cref="EstimateAsync(int[], int, int, int, int)"/>.
    /// Caller owns the returned buffer and must dispose it.
    ///
    /// If you also need the raw depth buffer (to re-apply a different palette later or
    /// run additional postprocessing without re-running inference) use
    /// <see cref="EstimateGpuRawAsync(int[], int, int, int, int)"/> + <see cref="ApplyColormapGpuAsync"/> instead.
    /// </summary>
    /// <remarks>
    /// ⚠️ IN A BROWSER, prefer the <see cref="TypedArray"/> or GPU-view overloads.
    /// </remarks>
    public async Task<(MemoryBuffer2D<int, Stride2D.DenseX> Buffer, int Width, int Height)> EstimateGpuAsync(
        int[] rgbaPixels, int width, int height,
        int outputWidth = 0, int outputHeight = 0)
    {
        using var rgbaBuf = RgbaUpload.FromManaged(_accelerator, rgbaPixels, width, height);
        return await EstimateGpuAsync(rgbaBuf.View, width, height, outputWidth, outputHeight)
            .ConfigureAwait(false);
    }

    /// <summary>Browser path — JS typed array → GPU without a managed heap crossing.</summary>
    public async Task<(MemoryBuffer2D<int, Stride2D.DenseX> Buffer, int Width, int Height)> EstimateGpuAsync(
        TypedArray rgbaPixels, int width, int height,
        int outputWidth = 0, int outputHeight = 0)
    {
        using var rgbaBuf = RgbaUpload.FromTypedArray(_accelerator, rgbaPixels, width, height);
        return await EstimateGpuAsync(rgbaBuf.View, width, height, outputWidth, outputHeight)
            .ConfigureAwait(false);
    }

    /// <summary>GPU-resident packed RGBA — no upload.</summary>
    public async Task<(MemoryBuffer2D<int, Stride2D.DenseX> Buffer, int Width, int Height)> EstimateGpuAsync(
        ArrayView1D<int, Stride1D.Dense> rgbaPixels, int width, int height,
        int outputWidth = 0, int outputHeight = 0)
    {
        var (rawDepth, minD, maxD, outW, outH) = await EstimateGpuRawAsync(
            rgbaPixels, width, height, outputWidth, outputHeight).ConfigureAwait(false);
        try
        {
            var resultBuf = await ApplyColormapGpuAsync(rawDepth.View, outW, outH, minD, maxD,
                Kernels.ImagePostprocessKernel.PalettePlasma).ConfigureAwait(false);
            return (resultBuf, outW, outH);
        }
        finally
        {
            rawDepth.Dispose();
        }
    }

    /// <summary>Same as the view overload; accepts an owned buffer.</summary>
    public Task<(MemoryBuffer2D<int, Stride2D.DenseX> Buffer, int Width, int Height)> EstimateGpuAsync(
        MemoryBuffer1D<int, Stride1D.Dense> rgbaPixels, int width, int height,
        int outputWidth = 0, int outputHeight = 0)
        => EstimateGpuAsync(rgbaPixels.View, width, height, outputWidth, outputHeight);

    /// <summary>
    /// Run depth inference and return the raw normalized-range depth as a GPU buffer
    /// alongside its min/max scalars and dimensions. The buffer stays on the accelerator —
    /// callers can apply <see cref="ApplyColormapGpuAsync"/> as many times as they like
    /// (e.g. when a UI palette toggle changes) without re-running inference.
    ///
    /// Output dimensions follow the same convention as <see cref="EstimateAsync(int[], int, int, int, int)"/>.
    /// Caller owns the returned <see cref="MemoryBuffer1D{T, TStride}"/> and must dispose
    /// it when finished.
    /// </summary>
    /// <remarks>
    /// ⚠️ IN A BROWSER, prefer <see cref="EstimateGpuRawAsync(TypedArray, int, int, int, int)"/>
    /// or a GPU-resident view — this path copies through the managed heap solely to upload.
    /// </remarks>
    public async Task<(MemoryBuffer1D<float, Stride1D.Dense> RawDepth, float MinDepth, float MaxDepth, int Width, int Height)>
        EstimateGpuRawAsync(int[] rgbaPixels, int width, int height,
            int outputWidth = 0, int outputHeight = 0)
    {
        using var rgbaBuf = RgbaUpload.FromManaged(_accelerator, rgbaPixels, width, height);
        return await EstimateGpuRawAsync(rgbaBuf.View, width, height, outputWidth, outputHeight)
            .ConfigureAwait(false);
    }

    /// <summary>
    /// Browser path: RGBA as a JS typed array. Uploads via <see cref="MediaInterop.UploadToDevice{T}"/>
    /// — pixels never enter the .NET managed heap.
    /// </summary>
    public async Task<(MemoryBuffer1D<float, Stride1D.Dense> RawDepth, float MinDepth, float MaxDepth, int Width, int Height)>
        EstimateGpuRawAsync(TypedArray rgbaPixels, int width, int height,
            int outputWidth = 0, int outputHeight = 0)
    {
        using var rgbaBuf = RgbaUpload.FromTypedArray(_accelerator, rgbaPixels, width, height);
        return await EstimateGpuRawAsync(rgbaBuf.View, width, height, outputWidth, outputHeight)
            .ConfigureAwait(false);
    }

    /// <summary>
    /// GPU-resident packed RGBA — no upload. Prefer this when the frame is already on the
    /// accelerator (SpawnScene <c>GpuImage</c>); avoids the GPU→JS→.NET→GPU round-trip.
    /// </summary>
    public Task<(MemoryBuffer1D<float, Stride1D.Dense> RawDepth, float MinDepth, float MaxDepth, int Width, int Height)>
        EstimateGpuRawAsync(ArrayView1D<int, Stride1D.Dense> rgbaPixels, int width, int height,
            int outputWidth = 0, int outputHeight = 0)
        => EstimateGpuRawCoreAsync(rgbaPixels, width, height, outputWidth, outputHeight);

    /// <summary>Same as the view overload; accepts an owned buffer.</summary>
    public Task<(MemoryBuffer1D<float, Stride1D.Dense> RawDepth, float MinDepth, float MaxDepth, int Width, int Height)>
        EstimateGpuRawAsync(MemoryBuffer1D<int, Stride1D.Dense> rgbaPixels, int width, int height,
            int outputWidth = 0, int outputHeight = 0)
        => EstimateGpuRawCoreAsync(rgbaPixels.View, width, height, outputWidth, outputHeight);

    private async Task<(MemoryBuffer1D<float, Stride1D.Dense> RawDepth, float MinDepth, float MaxDepth, int Width, int Height)>
        EstimateGpuRawCoreAsync(ArrayView1D<int, Stride1D.Dense> rgbaPixels, int width, int height,
            int outputWidth, int outputHeight)
    {
        // Graph-capture path (opt-in; CUDA graphs / WebGPU dispatch plans): preprocess into a STABLE input
        // buffer the captured graph reads, then capture-once / replay-many. The per-frame preprocess dispatch
        // writes fresh data into that stable buffer and is queue-ordered before the replay's submit, so the
        // replay reads the new frame with NO extra copy. Falls back to a normal forward on other backends or
        // if capture is unavailable (TryCaptureAsync returns null). Non-capture path uses a transient input.
        bool useCapture = EnableGraphCapture
            && (_accelerator.AcceleratorType == AcceleratorType.Cuda
                || _accelerator.AcceleratorType == AcceleratorType.WebGPU);
        MemoryBuffer1D<float, Stride1D.Dense>? transientInput = null;
        ArrayView1D<float, Stride1D.Dense> preInput;
        if (useCapture)
        {
            _captureInputBuf ??= _accelerator.Allocate1D<float>(3 * _inputSize * _inputSize);
            preInput = _captureInputBuf.View;
        }
        else
        {
            transientInput = _accelerator.Allocate1D<float>(3 * _inputSize * _inputSize);
            preInput = transientInput.View;
        }
        _preprocess.Forward(rgbaPixels, preInput, width, height, _inputSize, _inputSize,
            preserveAspect: PreserveAspect);

        var inputTensor = new Tensor(preInput, InputTensorShape());
        var inputDict = new Dictionary<string, Tensor> { [_session.InputNames[0]] = inputTensor };

        Dictionary<string, Tensor> outputs;
        if (useCapture && _accelerator.AcceleratorType == AcceleratorType.Cuda)
        {
            var shape = InputTensorShape();
            if (_capture == null || _captureShape == null || !shape.AsSpan().SequenceEqual(_captureShape))
            {
                _capture?.Dispose();
                _capture = await CudaGraphCapture.TryCaptureAsync(_session, inputDict);   // first frame at this resolution
                _captureShape = shape;
            }
            outputs = _capture != null ? await _capture.ReplayAsync(inputDict) : await _session.RunAsync(inputDict);
        }
        else if (useCapture)   // WebGPU
        {
            var shape = InputTensorShape();
            // Same gate SessionGraphCapture now applies: a pool reclaim since recording leaves this plan's
            // bind groups pointing at disposed buckets. Drop and recapture rather than submit garbage
            // (or hit the ReplayAsync InvalidatedByReclaim throw).
            if (_webGpuCapture != null && _webGpuCapture.InvalidatedByReclaim)
            {
                _webGpuCapture.Dispose();
                _webGpuCapture = null;
                _captureShape = null;
            }
            if (_webGpuCapture == null || _captureShape == null || !shape.AsSpan().SequenceEqual(_captureShape))
            {
                _webGpuCapture?.Dispose();
                _webGpuCapture = await WebGPUGraphCapture.TryCaptureAsync(_session, inputDict);   // first frame at this resolution
                _captureShape = shape;
            }
            // inputDict wraps the SAME stable buffer the capture reads (fresh frame written by the
            // preprocess dispatch above), so ReplayAsync's same-buffer check skips the input copy.
            outputs = _webGpuCapture != null ? await _webGpuCapture.ReplayAsync(inputDict) : await _session.RunAsync(inputDict);
        }
        else
        {
            outputs = await _session.RunAsync(inputDict);
        }

        var output = outputs[_session.OutputNames[0]];
        int rawSize = output.ElementCount;
        int rawH = output.Shape.Length >= 3 ? output.Shape[^2] : _inputSize;
        int rawW = output.Shape.Length >= 3 ? output.Shape[^1] : _inputSize;

        if (InferenceSession.VerboseLogging) Console.WriteLine($"[Depth] Output: shape=[{string.Join(",", output.Shape)}], elements={rawSize}, dataLength={output.Data.Length}");

        var (outW, outH) = ResolveOutputSize(width, height, rawW, rawH, outputWidth, outputHeight);
        int outSize = outW * outH;

        // GPU bilinear resize from rawW×rawH → outW×outH. The caller-owned buffer is
        // returned untouched after this; the only readback is the 8-BYTE min/max scalar
        // pair from the GPU reduction (was: the full ~1MB resized map + host LINQ - the
        // dominant share of the video-path postprocess cost).
        int readRawSize = Math.Min(rawSize, (int)output.Data.Length);
        var rawDepth = _accelerator.Allocate1D<float>(outSize);
        var (cx2, cy2, cw2, ch2) = ContentRect(width, height, rawW, rawH);
        if (cx2 != 0 || cy2 != 0 || cw2 != rawW || ch2 != rawH)
        {
            // Letterboxed input: only the content rect is real picture.
            _postprocess.ResizeBilinearFromRect(
                output.Data.SubView(0, readRawSize), rawW, rawH, cx2, cy2, cw2, ch2,
                rawDepth.View, outW, outH);
        }
        else
        {
            // TensorView<float> carries shape inline — kernel reads dims from D0/D1.
            var srcView = new Tensors.TensorView<float>(output.Data.SubView(0, readRawSize), new[] { rawH, rawW });
            var dstView = new Tensors.TensorView<float>(rawDepth.View, new[] { outH, outW });
            _postprocess.ResizeBilinear(srcView, dstView);
        }
        var (minD, maxD) = await _postprocess.MinMaxAsync(rawDepth.View, outSize);

        if (InferenceSession.VerboseLogging)
        {
            // Diagnostic-only full readback (the production path above never touches the host).
            var resizedHost = await rawDepth.CopyToHostAsync<float>(0, outSize);
            Console.WriteLine($"[Depth] Values: min={minD:F4}, max={maxD:F4}, absMax={resizedHost.Max(v => MathF.Abs(v)):F4}, nonZero={resizedHost.Count(v => v != 0)}/{outSize}");
        }

        transientInput?.Dispose();   // the stable capture input buffer is a member (disposed in Dispose)
        return (rawDepth, minD, maxD, outW, outH);
    }

    /// <summary>
    /// Apply a colormap to a raw depth GPU buffer and return a fresh 2D RGBA buffer with
    /// the colored result. Inference is NOT re-run — this is just the postprocess step.
    /// Caller owns the returned buffer and must dispose it.
    ///
    /// Use <see cref="Kernels.ImagePostprocessKernel.PaletteFromName"/> to convert a UI
    /// palette name (plasma / viridis / inferno / grayscale) into the int palette index.
    /// </summary>
    public async Task<MemoryBuffer2D<int, Stride2D.DenseX>> ApplyColormapGpuAsync(
        ArrayView1D<float, Stride1D.Dense> rawDepth, int width, int height,
        float minDepth, float maxDepth, int palette)
    {
        var postprocess = _postprocess;
        var resultBuf = _accelerator.Allocate2DDenseX<int>(new Index2D(width, height));
        // Phase 2 TensorView<float> + TensorView<int> overload. Both tensors are
        // row-major [H, W]; the kernel reads count from depth.ElementCount.
        // Restored 2026-05-24 PM after SpawnDev.ILGPU 4.9.9-local.1 fixed the
        // scalar-slot drift that caused the original migration to flat-blue the
        // demo on WebGPU. See ILGPU commit d5154c6.
        var depthView = new Tensors.TensorView<float>(rawDepth, new[] { height, width });
        var rgbaView = new Tensors.TensorView<int>(resultBuf.View.BaseView, new[] { height, width });
        postprocess.DepthToColormapPalette(depthView, rgbaView, minDepth, maxDepth, palette);
        await _accelerator.SynchronizeAsync();
        return resultBuf;
    }

    /// <summary>
    /// Joint multi-view depth (DAv3-native): pack N RGBA frames into <c>[1,N,3,H,W]</c>, run one forward,
    /// return per-view GPU depth maps plus optional confidence / extrinsics / intrinsics by output name.
    /// Graph capture is skipped (N varies). Session shape-recompile handles N ≠ compile-time num_images.
    /// </summary>
    /// <param name="frames">Packed RGBA int buffers, one per view (same layout as monocular EstimateGpuRaw).</param>
    /// <param name="widths">Source width per view.</param>
    /// <param name="heights">Source height per view.</param>
    /// <param name="outputWidth">0 = match first frame width.</param>
    /// <param name="outputHeight">0 = match first frame height.</param>
    public async Task<MultiViewDepthGpuResult> EstimateMultiViewGpuAsync(
        IReadOnlyList<ArrayView1D<int, Stride1D.Dense>> frames,
        IReadOnlyList<int> widths, IReadOnlyList<int> heights,
        int outputWidth = 0, int outputHeight = 0)
    {
        int n = frames.Count;
        if (n < 1) throw new ArgumentException("At least one view is required.", nameof(frames));
        if (widths.Count != n || heights.Count != n)
            throw new ArgumentException("widths/heights length must match frames.");

        int chw = 3 * _inputSize * _inputSize;
        using var stacked = _accelerator.Allocate1D<float>(n * (long)chw);
        for (int i = 0; i < n; i++)
        {
            _preprocess.Forward(frames[i], stacked.View.SubView(i * chw, chw),
                widths[i], heights[i], _inputSize, _inputSize,
                preserveAspect: PreserveAspect);
        }

        var inputShape = new[] { 1, n, 3, _inputSize, _inputSize };
        var inputTensor = new Tensor(stacked.View, inputShape);
        var outputs = await _session.RunAsync(new Dictionary<string, Tensor>
        {
            [_session.InputNames[0]] = inputTensor
        }).ConfigureAwait(false);
        await _accelerator.SynchronizeAsync().ConfigureAwait(false);

        var depthTensor = FindOutput(outputs, "predicted_depth")
            ?? outputs[_session.OutputNames[0]];
        var views = await SplitDepthViewsAsync(depthTensor, n, widths[0], heights[0], outputWidth, outputHeight)
            .ConfigureAwait(false);

        IReadOnlyList<MemoryBuffer1D<float, Stride1D.Dense>>? confMaps = null;
        var confTensor = FindOutput(outputs, "confidence");
        if (confTensor != null)
        {
            confMaps = await SplitPlaneViewsAsync(confTensor, n, views[0].Width, views[0].Height)
                .ConfigureAwait(false);
        }

        float[][]? extrinsics = await TryReadPoseMatricesAsync(FindOutput(outputs, "extrinsics"), n, expectedElemsPerView: 12)
            .ConfigureAwait(false);
        float[][]? intrinsics = await TryReadPoseMatricesAsync(FindOutput(outputs, "intrinsics"), n, expectedElemsPerView: 9)
            .ConfigureAwait(false);

        if (InferenceSession.VerboseLogging)
            Console.WriteLine($"[Depth-MV] N={n} depthViews={views.Count} conf={(confMaps != null)} " +
                $"extrinsics={(extrinsics != null)} intrinsics={(intrinsics != null)}");

        return new MultiViewDepthGpuResult
        {
            Views = views,
            ConfidenceMaps = confMaps,
            Extrinsics = extrinsics,
            Intrinsics = intrinsics,
        };
    }

    /// <summary>Managed-RGBA convenience: uploads each frame then runs <see cref="EstimateMultiViewGpuAsync"/>.</summary>
    public async Task<MultiViewDepthGpuResult> EstimateMultiViewGpuAsync(
        IReadOnlyList<int[]> rgbaFrames, IReadOnlyList<int> widths, IReadOnlyList<int> heights,
        int outputWidth = 0, int outputHeight = 0)
    {
        int n = rgbaFrames.Count;
        var uploads = new MemoryBuffer1D<int, Stride1D.Dense>[n];
        var views = new ArrayView1D<int, Stride1D.Dense>[n];
        try
        {
            for (int i = 0; i < n; i++)
            {
                uploads[i] = RgbaUpload.FromManaged(_accelerator, rgbaFrames[i], widths[i], heights[i]);
                views[i] = uploads[i].View;
            }
            return await EstimateMultiViewGpuAsync(views, widths, heights, outputWidth, outputHeight)
                .ConfigureAwait(false);
        }
        finally
        {
            for (int i = 0; i < n; i++)
                uploads[i]?.Dispose();
        }
    }

    private static Tensor? FindOutput(Dictionary<string, Tensor> outputs, string nameHint)
    {
        foreach (var (name, t) in outputs)
        {
            if (name.Equals(nameHint, StringComparison.OrdinalIgnoreCase))
                return t;
            // Some exports prefix with "/" or a path segment.
            if (name.EndsWith("/" + nameHint, StringComparison.OrdinalIgnoreCase)
                || name.EndsWith(nameHint, StringComparison.OrdinalIgnoreCase))
                return t;
        }
        return null;
    }

    private async Task<List<(MemoryBuffer1D<float, Stride1D.Dense> RawDepth, float MinDepth, float MaxDepth, int Width, int Height)>>
        SplitDepthViewsAsync(Tensor depthTensor, int n, int srcW, int srcH, int outputWidth, int outputHeight)
    {
        // Common shapes: [1,N,H,W], [N,H,W], [1,N,1,H,W], or flat N*H*W.
        var shape = depthTensor.Shape;
        int rawH, rawW, viewElems;
        if (shape.Length >= 4 && shape[^3] == n)
        {
            rawH = shape[^2]; rawW = shape[^1];
            viewElems = rawH * rawW;
        }
        else if (shape.Length >= 3 && shape[0] == n)
        {
            rawH = shape[^2]; rawW = shape[^1];
            viewElems = rawH * rawW;
        }
        else if (shape.Length >= 4 && shape[1] == n)
        {
            rawH = shape[^2]; rawW = shape[^1];
            viewElems = rawH * rawW;
        }
        else
        {
            // Fallback: divide total elements evenly across N (square-ish spatial).
            int total = depthTensor.ElementCount;
            if (total % n != 0)
                throw new InvalidOperationException(
                    $"predicted_depth element count {total} is not divisible by num_images={n} (shape=[{string.Join(",", shape)}])");
            viewElems = total / n;
            rawH = rawW = (int)MathF.Round(MathF.Sqrt(viewElems));
            if (rawH * rawW != viewElems)
            {
                // Prefer H=W from compile input size when reshape is awkward.
                rawH = _inputSize; rawW = viewElems / Math.Max(1, rawH);
                if (rawH * rawW != viewElems)
                    throw new InvalidOperationException(
                        $"Cannot infer per-view HxW from predicted_depth shape=[{string.Join(",", shape)}] N={n}");
            }
        }

        var (outW, outH) = ResolveOutputSize(srcW, srcH, rawW, rawH, outputWidth, outputHeight);
        int outSize = outW * outH;
        var list = new List<(MemoryBuffer1D<float, Stride1D.Dense>, float, float, int, int)>(n);

        // Where the real picture sits inside a letterboxed output. The rect is computed in
        // MODEL space and scaled into the output's, because a model may predict at a different
        // resolution from its input.
        var (cropX, cropY, cropW, cropH) = ContentRect(srcW, srcH, rawW, rawH);

        // The resolution the model actually PREDICTS at, which is not always its input size -
        // a DPT-style head often emits at a fraction of it. Everything past this point is
        // upsampling, and soft edges in the final map are explained here or nowhere.
        Console.WriteLine(
            $"[Depth-MV] predicted {rawW}x{rawH} from a {_inputSize}x{_inputSize} input " +
            $"-> content {cropW}x{cropH} -> out {outW}x{outH} " +
            $"({(float)outW / Math.Max(1, cropW):F1}x upsample)");

        for (int i = 0; i < n; i++)
        {
            long offset = (long)i * viewElems;
            var srcSub = depthTensor.Data.SubView(offset, viewElems);
            var rawDepth = _accelerator.Allocate1D<float>(outSize);
            bool cropped = cropX != 0 || cropY != 0 || cropW != rawW || cropH != rawH;
            if (!cropped && outW == rawW && outH == rawH)
            {
                rawDepth.View.CopyFrom(srcSub.SubView(0, outSize));
            }
            else if (cropped)
            {
                _postprocess.ResizeBilinearFromRect(
                    srcSub, rawW, rawH, cropX, cropY, cropW, cropH,
                    rawDepth.View, outW, outH);
            }
            else
            {
                var srcView = new Tensors.TensorView<float>(srcSub, new[] { rawH, rawW });
                var dstView = new Tensors.TensorView<float>(rawDepth.View, new[] { outH, outW });
                _postprocess.ResizeBilinear(srcView, dstView);
            }
            var (minD, maxD) = await _postprocess.MinMaxAsync(rawDepth.View, outSize).ConfigureAwait(false);
            list.Add((rawDepth, minD, maxD, outW, outH));
        }
        return list;
    }

    /// <summary>
    /// The region of a model output that holds real picture, given a letterboxed input of
    /// <paramref name="srcW"/> x <paramref name="srcH"/>. The whole output when
    /// <see cref="PreserveAspect"/> is off.
    /// </summary>
    private (int X, int Y, int W, int H) ContentRect(int srcW, int srcH, int rawW, int rawH)
    {
        if (!PreserveAspect || srcW <= 0 || srcH <= 0) return (0, 0, rawW, rawH);

        var (cw, ch, px, py) = Kernels.ImagePreprocessKernel.Letterbox(
            srcW, srcH, _inputSize, _inputSize);
        float sx = (float)rawW / _inputSize;
        float sy = (float)rawH / _inputSize;

        int x = (int)MathF.Round(px * sx);
        int y = (int)MathF.Round(py * sy);
        int w = Math.Max(1, Math.Min(rawW - x, (int)MathF.Round(cw * sx)));
        int h = Math.Max(1, Math.Min(rawH - y, (int)MathF.Round(ch * sy)));
        return (x, y, w, h);
    }

    private async Task<List<MemoryBuffer1D<float, Stride1D.Dense>>> SplitPlaneViewsAsync(
        Tensor planeTensor, int n, int outW, int outH)
    {
        int total = planeTensor.ElementCount;
        if (total % n != 0)
            throw new InvalidOperationException(
                $"Plane tensor element count {total} not divisible by N={n}");
        int viewElems = total / n;
        int rawH = outH, rawW = outW;
        if (viewElems != outW * outH)
        {
            rawH = (int)MathF.Round(MathF.Sqrt(viewElems));
            rawW = viewElems / Math.Max(1, rawH);
        }
        var list = new List<MemoryBuffer1D<float, Stride1D.Dense>>(n);
        for (int i = 0; i < n; i++)
        {
            var srcSub = planeTensor.Data.SubView((long)i * viewElems, viewElems);
            var buf = _accelerator.Allocate1D<float>(outW * outH);
            if (rawW == outW && rawH == outH)
                buf.View.CopyFrom(srcSub.SubView(0, outW * outH));
            else
            {
                var srcView = new Tensors.TensorView<float>(srcSub, new[] { rawH, rawW });
                var dstView = new Tensors.TensorView<float>(buf.View, new[] { outH, outW });
                _postprocess.ResizeBilinear(srcView, dstView);
            }
            list.Add(buf);
        }
        await _accelerator.SynchronizeAsync().ConfigureAwait(false);
        return list;
    }

    /// <summary>
    /// Read a pose/matrix tensor to per-view float arrays. Returns null if missing, wrong size, or all-zero/NaN.
    /// </summary>
    private async Task<float[][]?> TryReadPoseMatricesAsync(Tensor? tensor, int n, int expectedElemsPerView)
    {
        if (tensor == null) return null;
        int total = tensor.ElementCount;
        if (total < n * expectedElemsPerView) return null;
        int perView = total / n;
        if (perView < expectedElemsPerView) return null;

        // Tiny readback (N*12 floats) — stage into a buffer so WebGPU/Wasm async path works.
        using var stage = _accelerator.Allocate1D<float>(total);
        stage.View.CopyFrom(tensor.Data.SubView(0, total));
        float[] host = await stage.CopyToHostAsync<float>(0, total).ConfigureAwait(false);

        var result = new float[n][];
        bool anyNonZero = false;
        for (int i = 0; i < n; i++)
        {
            var row = new float[expectedElemsPerView];
            int baseOff = i * perView;
            for (int k = 0; k < expectedElemsPerView; k++)
            {
                float v = host[baseOff + k];
                if (float.IsNaN(v) || float.IsInfinity(v)) return null;
                row[k] = v;
                if (MathF.Abs(v) > 1e-8f) anyNonZero = true;
            }
            result[i] = row;
        }
        return anyNonZero ? result : null;
    }

    public void Dispose()
    {
        _capture?.Dispose();
        _capture = null;
        _webGpuCapture?.Dispose();
        _webGpuCapture = null;
        _captureInputBuf?.Dispose();
        _captureInputBuf = null;
        _postprocess.Dispose();
    }
}
