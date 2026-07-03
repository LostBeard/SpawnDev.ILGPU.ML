using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Tensors;

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
    private readonly int _inputSize;

    /// <summary>
    /// Opt-in graph capture for the VIDEO / repeat-inference path (default off; CUDA + WebGPU, no-op elsewhere).
    /// When set, <see cref="EstimateGpuRawAsync"/> captures the forward once at the first resolution it sees and
    /// REPLAYS it for every subsequent frame at that resolution. CUDA: a single cuGraphLaunch instead of the
    /// ~2524-node loop (~3x on DAv3-Small, bit-identical). WebGPU: a single interop crossing re-encodes the
    /// captured dispatch plan (<see cref="WebGPUGraphCapture"/>) - the 10.8s-warm-direct → ~65-75ms/frame
    /// production path (bit-identical, beats ORT-Web's 73ms warm). The first frame pays the capture cost
    /// (a few warm forwards); the resolution is re-captured if it changes.
    /// </summary>
    public bool EnableGraphCapture { get; set; }
    private CudaGraphCapture? _capture;
    private WebGPUGraphCapture? _webGpuCapture;
    private MemoryBuffer1D<float, Stride1D.Dense>? _captureInputBuf;
    private int[]? _captureShape;

    public DepthEstimationPipeline(InferenceSession session, Accelerator accelerator,
        int inputSize = 0)
    {
        _session = session;
        _accelerator = accelerator;
        _preprocess = new Kernels.ImagePreprocessKernel(accelerator);
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
    public static async Task<DepthEstimationPipeline> CreateFromHubAsync(
        Accelerator accelerator, Hub.HubModelStream hubStream, string repoId,
        string modelFile = "onnx/model.onnx", string externalDataFile = "onnx/model.onnx_data",
        Action<string, int>? onProgress = null, Dictionary<string, int[]>? inputShapes = null,
        int inputSize = 0, CancellationToken ct = default)
    {
        onProgress?.Invoke("open", 0);
        System.IO.Stream modelStream;
        Hub.HubModelStream.HubModel? extData = null;
        if (!string.IsNullOrEmpty(externalDataFile))
        {
            // External-data model (DAv3): model.onnx is a SMALL structure file (weights live in model.onnx_data).
            // Fetch it over plain HTTP (KBs) and torrent-stream ONLY the big weights file — keeps the 100+ MB
            // weights zero-copy AND avoids a WebTorrent lazy-hash same-directory collision that otherwise gave the
            // model.onnx_data stream model.onnx's length (both live under onnx/). If model.onnx_data is absent
            // (a mislabeled single-file export), extData stays null and the structure file carries any weights.
            var modelBytes = await hubStream.FetchBytesAsync(repoId, modelFile, ct).ConfigureAwait(false);
            modelStream = new System.IO.MemoryStream(modelBytes);
            try { extData = await hubStream.OpenAsync(repoId, externalDataFile, ct: ct).ConfigureAwait(false); }
            catch { extData = null; }
        }
        else
        {
            // Single-file model: torrent-stream model.onnx directly (weights embedded — keep them off the heap).
            var modelTorrent = await hubStream.OpenAsync(repoId, modelFile, ct: ct).ConfigureAwait(false);
            modelStream = modelTorrent.Stream;
        }
        onProgress?.Invoke("open", 100);
        try
        {
            return await CreateFromStreamsAsync(accelerator, modelStream, extData?.Stream,
                onProgress, inputShapes, inputSize, ct).ConfigureAwait(false);
        }
        finally
        {
            // Weights are on the GPU by now — the streams are done.
            await modelStream.DisposeAsync().ConfigureAwait(false);
            if (extData != null) await extData.Stream.DisposeAsync().ConfigureAwait(false);
        }
    }

    /// <summary>
    /// Estimate depth from an RGBA image.
    /// Returns a depth map normalized to [0, 1] (higher = closer).
    ///
    /// Output dimensions:
    ///   outputWidth = 0 && outputHeight = 0 → match source (width, height) — default,
    ///       preserves aspect ratio so the depth map aligns 1:1 with the input.
    ///   outputWidth > 0 && outputHeight = 0 → use outputWidth, derive height from source aspect.
    ///   outputWidth = 0 && outputHeight > 0 → use outputHeight, derive width from source aspect.
    ///   outputWidth > 0 && outputHeight > 0 → exact size (may not preserve aspect).
    /// Resize is done on the accelerator via bilinear interpolation — no CPU readback of the raw map.
    /// </summary>
    public async Task<DepthResult> EstimateAsync(int[] rgbaPixels, int width, int height,
        int outputWidth = 0, int outputHeight = 0)
    {
        // Upload and preprocess
        using var rgbaBuf = _accelerator.Allocate1D(rgbaPixels);
        using var preprocessed = _accelerator.Allocate1D<float>(3 * _inputSize * _inputSize);
        _preprocess.Forward(rgbaBuf.View, preprocessed.View, width, height, _inputSize, _inputSize);

        // Create input tensor
        var inputTensor = new Tensor(preprocessed.View, InputTensorShape());

        // Run inference
        var outputs = await _session.RunAsync(new Dictionary<string, Tensor>
        {
            [_session.InputNames[0]] = inputTensor
        });
        await _accelerator.SynchronizeAsync();

        var output = outputs[_session.OutputNames[0]];
        int rawSize = output.ElementCount;
        int rawH = output.Shape.Length >= 3 ? output.Shape[^2] : _inputSize;
        int rawW = output.Shape.Length >= 3 ? output.Shape[^1] : _inputSize;
        if (InferenceSession.VerboseLogging) Console.WriteLine($"[Depth CPU] Output: shape=[{string.Join(",", output.Shape)}], elements={rawSize}");

        // Resolve output dimensions (default = source size, preserves source aspect).
        var (outW, outH) = ResolveOutputSize(width, height, rawW, rawH, outputWidth, outputHeight);
        int outSize = outW * outH;

        // GPU-side bilinear resize from raw model output (rawW × rawH) → (outW × outH).
        // TensorView<float> carries shape inline — no scalar W/H kernel params needed.
        var post = new Kernels.ImagePostprocessKernel(_accelerator);
        using var resized = _accelerator.Allocate1D<float>(outSize);
        var srcView = new Tensors.TensorView<float>(output.Data.SubView(0, rawSize), new[] { rawH, rawW });
        var dstView = new Tensors.TensorView<float>(resized.View, new[] { outH, outW });
        post.ResizeBilinear(srcView, dstView);
        await _accelerator.SynchronizeAsync();

        // Read resized depth to CPU for min/max + normalization.
        var rawDepth = await resized.CopyToHostAsync<float>(0, outSize);
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
    /// Output dimensions follow the same convention as <see cref="EstimateAsync"/>:
    ///   (0, 0) → match source (width, height), preserving aspect ratio.
    ///   (w, 0) / (0, h) → fit one axis, derive the other from source aspect.
    ///   (w, h) → exact.
    /// Resize is bilinear, executed on the accelerator before colormap.
    /// Caller owns the returned buffer and must dispose it.
    ///
    /// If you also need the raw depth buffer (to re-apply a different palette later or
    /// run additional postprocessing without re-running inference) use
    /// <see cref="EstimateGpuRawAsync"/> + <see cref="ApplyColormapGpuAsync"/> instead.
    /// </summary>
    public async Task<(MemoryBuffer2D<int, Stride2D.DenseX> Buffer, int Width, int Height)> EstimateGpuAsync(
        int[] rgbaPixels, int width, int height,
        int outputWidth = 0, int outputHeight = 0)
    {
        var (rawDepth, minD, maxD, outW, outH) = await EstimateGpuRawAsync(
            rgbaPixels, width, height, outputWidth, outputHeight);
        try
        {
            var resultBuf = await ApplyColormapGpuAsync(rawDepth.View, outW, outH, minD, maxD,
                Kernels.ImagePostprocessKernel.PalettePlasma);
            return (resultBuf, outW, outH);
        }
        finally
        {
            rawDepth.Dispose();
        }
    }

    /// <summary>
    /// Run depth inference and return the raw normalized-range depth as a GPU buffer
    /// alongside its min/max scalars and dimensions. The buffer stays on the accelerator —
    /// callers can apply <see cref="ApplyColormapGpuAsync"/> as many times as they like
    /// (e.g. when a UI palette toggle changes) without re-running inference.
    ///
    /// Output dimensions follow the same convention as <see cref="EstimateAsync"/>.
    /// Caller owns the returned <see cref="MemoryBuffer1D{T, TStride}"/> and must dispose
    /// it when finished.
    /// </summary>
    public async Task<(MemoryBuffer1D<float, Stride1D.Dense> RawDepth, float MinDepth, float MaxDepth, int Width, int Height)>
        EstimateGpuRawAsync(int[] rgbaPixels, int width, int height,
            int outputWidth = 0, int outputHeight = 0)
    {
        using var rgbaBuf = _accelerator.Allocate1D(rgbaPixels);

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
        _preprocess.Forward(rgbaBuf.View, preInput, width, height, _inputSize, _inputSize);

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

        var postprocess = new Kernels.ImagePostprocessKernel(_accelerator);

        // GPU bilinear resize from rawW×rawH → outW×outH. The caller-owned buffer is
        // returned untouched after this; the only readback is a small min/max for the
        // colormap normalization scalar pair (TODO: replace with a GPU reduction so the
        // raw path is entirely host-touch-free).
        int readRawSize = Math.Min(rawSize, (int)output.Data.Length);
        var rawDepth = _accelerator.Allocate1D<float>(outSize);
        // TensorView<float> carries shape inline — kernel reads dims from D0/D1.
        var srcView = new Tensors.TensorView<float>(output.Data.SubView(0, readRawSize), new[] { rawH, rawW });
        var dstView = new Tensors.TensorView<float>(rawDepth.View, new[] { outH, outW });
        postprocess.ResizeBilinear(srcView, dstView);
        await _accelerator.SynchronizeAsync();

        var resizedHost = await rawDepth.CopyToHostAsync<float>(0, outSize);
        float minD = resizedHost.Min();
        float maxD = resizedHost.Max();

        if (InferenceSession.VerboseLogging) Console.WriteLine($"[Depth] Values: min={minD:F4}, max={maxD:F4}, absMax={resizedHost.Max(v => MathF.Abs(v)):F4}, nonZero={resizedHost.Count(v => v != 0)}/{outSize}");

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
        var postprocess = new Kernels.ImagePostprocessKernel(_accelerator);
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

    public void Dispose()
    {
        _capture?.Dispose();
        _capture = null;
        _webGpuCapture?.Dispose();
        _webGpuCapture = null;
        _captureInputBuf?.Dispose();
        _captureInputBuf = null;
    }
}
