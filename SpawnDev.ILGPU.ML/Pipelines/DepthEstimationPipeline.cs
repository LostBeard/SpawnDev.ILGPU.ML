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

/// <summary>How <see cref="DepthEstimationPipeline"/> turns a picture into the model's input tensor.</summary>
public enum DepthResizeMode
{
    /// <summary>Fit inside the compiled square, centred, border replicated into the pad; the pad is
    /// cropped off the output. The historical default.</summary>
    Letterbox,
    /// <summary>Squash each axis to the compiled square independently (distorts aspect).</summary>
    Stretch,
    /// <summary>
    /// Depth Anything 3's own preprocessing: long side = <see cref="DepthEstimationPipeline.ProcessResolution"/>,
    /// aspect kept, NO padding, each side rounded to the nearest multiple of the ViT patch, OpenCV
    /// area / cubic resampling (<see cref="Kernels.ImagePreprocessKernel.ForwardNativeAspect"/>). The input
    /// tensor is non-square and follows the picture; the model must accept dynamic height/width (DAv3 does).
    /// MEASURED 2026-09-23 vs COLMAP ground truth on Truck: joint depth AbsRel 0.111 vs Letterbox's 0.150,
    /// camera-centre error 3.1% vs 9.6%, focal error 19% vs 44% - and fewer tokens (no pad patches).
    /// </summary>
    NativeAspect,
}

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
    public bool PreserveAspect
    {
        get => ResizeMode != DepthResizeMode.Stretch;
        set => ResizeMode = !value ? DepthResizeMode.Stretch
            : ResizeMode == DepthResizeMode.Stretch ? DepthResizeMode.Letterbox : ResizeMode;
    }

    /// <summary>
    /// How the picture becomes the input tensor. <see cref="DepthResizeMode.Letterbox"/> by default;
    /// <see cref="DepthResizeMode.NativeAspect"/> is the Depth Anything 3 reference preprocessing and
    /// MEASURED more accurate for DAv3 (see the enum).
    /// </summary>
    public DepthResizeMode ResizeMode { get; set; } = DepthResizeMode.Letterbox;

    /// <summary>
    /// <see cref="DepthResizeMode.NativeAspect"/> only: the long side, in pixels, before rounding to the
    /// patch grid. 0 = the compiled input's side (so a session bound at 518 keeps ~518's patch budget).
    /// The DA3 reference default is 504.
    /// </summary>
    public int ProcessResolution { get; set; }

    /// <summary><see cref="DepthResizeMode.NativeAspect"/> only: the ViT patch both sides are rounded to (14 for Depth Anything).</summary>
    public int PatchSize { get; set; } = 14;

    private int LongSide => ProcessResolution > 0 ? ProcessResolution : _inputSize;

    /// <summary>The model input (width, height) this pipeline builds for a <paramref name="srcW"/> x <paramref name="srcH"/> picture.</summary>
    public (int Width, int Height) ModelInputSize(int srcW, int srcH)
    {
        if (ResizeMode != DepthResizeMode.NativeAspect) return (_inputSize, _inputSize);
        var (_, _, w, h) = Kernels.ImagePreprocessKernel.NativeAspectSizes(srcW, srcH, LongSide, PatchSize);
        return (w, h);
    }

    // NativeAspect's stage-1 image (packed RGBA, device-resident). Grows; the old one is disposed only
    // after an awaited drain, because a queued dispatch may still read it (Wasm frees on dispose).
    private MemoryBuffer1D<int, Stride1D.Dense>? _nativeScratch;

    private async Task<ArrayView1D<int, Stride1D.Dense>> NativeScratchAsync(int srcW, int srcH)
    {
        int need = Kernels.ImagePreprocessKernel.NativeAspectScratchLength(srcW, srcH, LongSide, PatchSize);
        if (need == 0) return default;
        if (_nativeScratch == null || _nativeScratch.Length < need)
        {
            if (_nativeScratch != null)
            {
                await _accelerator.SynchronizeAsync().ConfigureAwait(false);
                _nativeScratch.Dispose();
            }
            _nativeScratch = _accelerator.Allocate1D<int>(need);
        }
        return _nativeScratch.View;
    }

    /// <summary>Picture -> one view's CHW input slice, per <see cref="ResizeMode"/>. Device only.</summary>
    private async Task PreprocessAsync(ArrayView1D<int, Stride1D.Dense> rgba, int srcW, int srcH,
        ArrayView1D<float, Stride1D.Dense> dst)
    {
        if (ResizeMode == DepthResizeMode.NativeAspect)
        {
            var scratch = await NativeScratchAsync(srcW, srcH).ConfigureAwait(false);
            _preprocess.ForwardNativeAspect(rgba, srcW, srcH, dst, scratch, LongSide, PatchSize);
        }
        else
        {
            _preprocess.Forward(rgba, dst, srcW, srcH, _inputSize, _inputSize,
                preserveAspect: ResizeMode == DepthResizeMode.Letterbox);
        }
    }

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
    /// <summary>
    /// One captured plan and the STABLE input buffer it reads, for one input shape. Single-view and joint
    /// multi-view keep separate slots so alternating between them does not re-record either plan.
    /// </summary>
    private sealed class CaptureSlot : IDisposable
    {
        public CudaGraphCapture? Cuda;
        public WebGPUGraphCapture? WebGpu;
        public int[]? Shape;
        public int[]? LastDirectShape;   // capture-on-repeat: the shape of the previous uncaptured call
        public MemoryBuffer1D<float, Stride1D.Dense>? Input;
        public void DropPlans() { Cuda?.Dispose(); Cuda = null; WebGpu?.Dispose(); WebGpu = null; Shape = null; }
        public void Dispose() { DropPlans(); Input?.Dispose(); Input = null; }
    }
    private readonly CaptureSlot _singleSlot = new();
    private readonly CaptureSlot _multiSlot = new();

    private bool UseCapture => EnableGraphCapture
        && (_accelerator.AcceleratorType == AcceleratorType.Cuda || _accelerator.AcceleratorType == AcceleratorType.WebGPU);

    /// <summary>The slot's stable input buffer, grown if needed (growing drops its plans: they read the old one).</summary>
    private async Task<ArrayView1D<float, Stride1D.Dense>> SlotInputAsync(CaptureSlot slot, int elems)
    {
        if (slot.Input != null && slot.Input.Length < elems)
        {
            // Bigger than any input before: the recorded plans read the old stable buffer, so they go with it.
            // Drain first - a replay may still be reading it.
            await _accelerator.SynchronizeAsync().ConfigureAwait(false);
            slot.DropPlans();
            slot.Input.Dispose(); slot.Input = null;
        }
        slot.Input ??= _accelerator.Allocate1D<float>(elems);
        return slot.Input.View.SubView(0, elems);
    }

    /// <summary>
    /// Forward through the slot's captured plan (capture on first use at a shape, replay after), or a plain
    /// forward when capture is unavailable. <c>Own</c> = the outputs are pool-rented and must be handed back
    /// with ReturnOutputs; a replay's outputs belong to the capture.
    /// </summary>
    private async Task<(Dictionary<string, Tensor> Outputs, bool Own)> RunWithSlotAsync(
        CaptureSlot slot, Dictionary<string, Tensor> inputDict, int[] shape, bool captureOnFirstUse = true)
    {
        // Recording costs ~3 forwards (two warm passes + the recorded one). With captureOnFirstUse off, a shape
        // runs direct the first time and is captured only when it comes back - a one-off shape never pays.
        bool planMatches = slot.Shape != null && shape.AsSpan().SequenceEqual(slot.Shape);
        if (!captureOnFirstUse && !planMatches
            && (slot.LastDirectShape == null || !shape.AsSpan().SequenceEqual(slot.LastDirectShape)))
        {
            slot.LastDirectShape = (int[])shape.Clone();
            return (await _session.RunAsync(inputDict), true);
        }
        slot.LastDirectShape = null;
        if (_accelerator.AcceleratorType == AcceleratorType.Cuda)
        {
            if (slot.Cuda == null || slot.Shape == null || !shape.AsSpan().SequenceEqual(slot.Shape))
            {
                slot.DropPlans();
                slot.Cuda = await CudaGraphCapture.TryCaptureAsync(_session, inputDict);   // first run at this shape
                slot.Shape = shape;
            }
            return slot.Cuda != null
                ? (await slot.Cuda.ReplayAsync(inputDict), false)
                : (await _session.RunAsync(inputDict), true);
        }
        // WebGPU. Same gate SessionGraphCapture applies: a pool reclaim since recording leaves this plan's bind
        // groups pointing at disposed buckets. Drop and recapture rather than submit garbage.
        if (slot.WebGpu != null && slot.WebGpu.InvalidatedByReclaim) slot.DropPlans();
        if (slot.WebGpu == null || slot.Shape == null || !shape.AsSpan().SequenceEqual(slot.Shape))
        {
            slot.DropPlans();
            // keepDrains: DAv3's capture recycles buffers on a plain forward's schedule instead of pinning every
            // intermediate - 16 GB -> 2.7 GB at 6 views @518 and 6.4 s -> 0.4 s replays on a 12 GB card. Gated
            // bit-exact vs onnxruntime by DA3_OrtParity_* / DA3_NativeAspect_Pipeline_MatchesOrt (replay path).
            slot.WebGpu = await WebGPUGraphCapture.TryCaptureAsync(_session, inputDict, keepDrains: true);
            slot.Shape = shape;
        }
        // inputDict wraps the SAME stable buffer the capture reads (fresh data written by the preprocess
        // dispatch), so ReplayAsync's same-buffer check skips the input copy.
        return slot.WebGpu != null
            ? (await slot.WebGpu.ReplayAsync(inputDict), false)
            : (await _session.RunAsync(inputDict), true);
    }

    /// <summary>Underlying session — diagnostics (weight presence) and advanced callers.</summary>
    public InferenceSession Session => _session;

    /// <summary>
    /// Give back the GPU memory a forward leaves parked for the next one (the session's free
    /// activation buckets, plus any graph-capture plan, whose bind groups would point at the
    /// disposed buckets). Weights stay; the next estimate re-allocates and, if capturing, re-records.
    /// Returns the bytes freed. Call it when the depth work is done for now and the GPU is needed for
    /// something else: on SpawnScene DrJohnson a finished 6-view DAv3 cascade held 3.9 GB here while
    /// the splat trainer tried to allocate its own 1.2 GB, and Chrome dropped the device at 7.4 GB.
    /// </summary>
    public long ReleaseWorkingMemory()
    {
        _singleSlot.DropPlans();
        _multiSlot.DropPlans();
        return _session.ReleaseWorkingMemory();
    }

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
    private int[] InputTensorShape(int h, int w)
    {
        bool fiveD = _session.InputShapes.TryGetValue(_session.InputNames[0], out var s) && s.Length == 5;
        return fiveD ? new[] { 1, 1, 3, h, w } : new[] { 1, 3, h, w };
    }

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
        var (inW, inH) = ModelInputSize(width, height);
        using var preprocessed = _accelerator.Allocate1D<float>(3 * inW * inH);
        await PreprocessAsync(rgbaPixels, width, height, preprocessed.View).ConfigureAwait(false);

        var inputTensor = new Tensor(preprocessed.View, InputTensorShape(inH, inW));
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
        _session.ReturnOutputs(outputs);   // resized holds everything we need from them

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
        bool useCapture = UseCapture;
        var (inW, inH) = ModelInputSize(width, height);
        int inElems = 3 * inW * inH;
        MemoryBuffer1D<float, Stride1D.Dense>? transientInput = null;
        ArrayView1D<float, Stride1D.Dense> preInput;
        if (useCapture)
            preInput = await SlotInputAsync(_singleSlot, inElems).ConfigureAwait(false);
        else
        {
            transientInput = _accelerator.Allocate1D<float>(inElems);
            preInput = transientInput.View;
        }
        await PreprocessAsync(rgbaPixels, width, height, preInput).ConfigureAwait(false);

        var inputTensor = new Tensor(preInput, InputTensorShape(inH, inW));
        var inputDict = new Dictionary<string, Tensor> { [_session.InputNames[0]] = inputTensor };

        // Outputs from a plain session run are pool-rented and ours to hand back once consumed; a
        // capture replay writes into the plan's fixed buffers, which the capture owns.
        Dictionary<string, Tensor> outputs;
        bool ownOutputs;
        if (useCapture)
            (outputs, ownOutputs) = await RunWithSlotAsync(_singleSlot, inputDict, inputTensor.Shape).ConfigureAwait(false);
        else
        {
            outputs = await _session.RunAsync(inputDict);
            ownOutputs = true;
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
        if (ownOutputs) _session.ReturnOutputs(outputs);   // consumed above (resize + min/max); leaked otherwise
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
    /// Captured per (N, H, W) like the single-view path (own slot). Session shape-recompile handles N ≠ compile-time num_images.
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

        // One joint forward takes ONE tensor shape. In NativeAspect every view's shape follows its own
        // picture, so views that land on different grids cannot share a pass. The DA3 reference
        // centre-crops them to the smallest; here that would hand back depth covering a different
        // region of each source than the caller's pixel grid, silently. Refuse instead: group views
        // by ModelInputSize (SpawnScene's PlanByShape already does).
        var (inW, inH) = ModelInputSize(widths[0], heights[0]);
        for (int i = 1; i < n; i++)
        {
            var (wi, hi) = ModelInputSize(widths[i], heights[i]);
            if (wi != inW || hi != inH)
                throw new ArgumentException(
                    $"view {i} ({widths[i]}x{heights[i]}) needs a {wi}x{hi} input but view 0 ({widths[0]}x{heights[0]}) " +
                    $"needs {inW}x{inH}; a joint pass takes one shape. Group views by ModelInputSize.", nameof(frames));
        }

        int chw = 3 * inW * inH;
        // The joint pass captures too, once per (N, H, W): SpawnScene's chunked passes repeat one shape, and a
        // direct forward pays ~2,500 per-node dispatch crossings - MEASURED 2026-09-23, 6 views @518 on WebGPU:
        // direct 3.4 s, replay 0.4 s (Transformers.js 0.6 s). Own slot, so single-view calls do not evict it.
        bool useCapture = UseCapture;
        MemoryBuffer1D<float, Stride1D.Dense>? transientStacked = null;
        ArrayView1D<float, Stride1D.Dense> stackedView;
        if (useCapture)
            stackedView = await SlotInputAsync(_multiSlot, n * chw).ConfigureAwait(false);
        else
        {
            transientStacked = _accelerator.Allocate1D<float>(n * (long)chw);
            stackedView = transientStacked.View;
        }
        for (int i = 0; i < n; i++)
            await PreprocessAsync(frames[i], widths[i], heights[i], stackedView.SubView(i * chw, chw)).ConfigureAwait(false);

        var inputShape = new[] { 1, n, 3, inH, inW };
        var inputTensor = new Tensor(stackedView, inputShape);
        var mvInputs = new Dictionary<string, Tensor> { [_session.InputNames[0]] = inputTensor };
        Dictionary<string, Tensor> outputs;
        bool ownOutputs;
        if (useCapture)
            (outputs, ownOutputs) = await RunWithSlotAsync(_multiSlot, mvInputs, inputShape, captureOnFirstUse: false).ConfigureAwait(false);
        else
        {
            outputs = await _session.RunAsync(mvInputs).ConfigureAwait(false);
            ownOutputs = true;
        }
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

        // Every output has been split into caller-owned per-view buffers or read to the host above.
        // Without this, each joint pass left its predicted_depth + confidence (two 16 MiB buckets at
        // N=6, 672x672) orphaned in the pool: +90 MB per pass on DrJohnson, MEASURED 2026-09-23.
        // (A replay's outputs belong to the capture and are not handed back.)
        if (ownOutputs) _session.ReturnOutputs(outputs);
        transientStacked?.Dispose();   // drained above (SynchronizeAsync + per-view readbacks)

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
            $"[Depth-MV] predicted {rawW}x{rawH} ({ResizeMode}) " +
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
    /// <see cref="ResizeMode"/> is not <see cref="DepthResizeMode.Letterbox"/>.
    /// </summary>
    private (int X, int Y, int W, int H) ContentRect(int srcW, int srcH, int rawW, int rawH)
    {
        // Only a letterbox has padding to crop: NativeAspect's whole output is picture.
        if (ResizeMode != DepthResizeMode.Letterbox || srcW <= 0 || srcH <= 0) return (0, 0, rawW, rawH);

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
        _singleSlot.Dispose();
        _multiSlot.Dispose();
        _nativeScratch?.Dispose();
        _nativeScratch = null;
        _postprocess.Dispose();
    }
}
