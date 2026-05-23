using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Tensors;

namespace SpawnDev.ILGPU.ML.Pipelines;

/// <summary>
/// Result from super-resolution — upscaled image as RGBA pixels.
/// </summary>
public record SuperResResult(int[] RgbaPixels, int Width, int Height, int UpscaleFactor);

/// <summary>
/// High-level super-resolution pipeline.
/// ESPCN model: input Y luminance channel [1, 1, H, W] → output [1, 1, H*3, W*3].
///
/// Usage:
///   var pipeline = new SuperResolutionPipeline(session, accelerator);
///   var result = await pipeline.UpscaleAsync(rgbaPixels, width, height);
/// </summary>
public class SuperResolutionPipeline : IDisposable
{
    private readonly InferenceSession _session;
    private readonly Accelerator _accelerator;
    private readonly int _upscaleFactor;

    private readonly int _modelH;
    private readonly int _modelW;
    private Kernels.ImagePreprocessKernel? _preprocess;

    public SuperResolutionPipeline(InferenceSession session, Accelerator accelerator,
        int upscaleFactor = 3)
    {
        _session = session;
        _accelerator = accelerator;
        _upscaleFactor = upscaleFactor;
        // Use model's declared input size (graph compiler uses static shapes)
        var inputShape = session.InputShapes.Values.FirstOrDefault() ?? new[] { 1, 1, 224, 224 };
        _modelH = inputShape.Length >= 4 ? inputShape[2] : (inputShape.Length >= 2 ? inputShape[^2] : 224);
        _modelW = inputShape.Length >= 4 ? inputShape[3] : (inputShape.Length >= 1 ? inputShape[^1] : 224);
        if (_modelH <= 0) _modelH = 224;
        if (_modelW <= 0) _modelW = 224;
    }

    /// <summary>
    /// Upscale an RGBA image using tile-based super-resolution.
    ///
    /// The model processes fixed-size Y patches (typically 224×224 for ESPCN exports).
    /// To honor the full source resolution and aspect, the pipeline tiles the source
    /// into overlapping <c>modelW × modelH</c> patches, runs each through the model
    /// independently, accumulates the super-resolved Y outputs into a single
    /// <c>(width * upscaleFactor) × (height * upscaleFactor)</c> destination plane,
    /// and finally combines that Y plane with source-derived Cb/Cr (bilinear up-
    /// sampled in the composite kernel) to produce a color RGBA result.
    ///
    /// Overlap in source pixels is held in <see cref="TileOverlap"/>. In overlap
    /// regions multiple tiles contribute to the same destination pixels; the
    /// per-pixel count buffer is used to average them, smoothing tile-boundary
    /// seams without requiring atomics (so the path works on WebGL too).
    ///
    /// Everything per-pixel runs on the accelerator. Only orchestration — tile
    /// indices and coordinates — runs on CPU. The only host readback is the final
    /// RGBA result.
    /// </summary>
    public async Task<SuperResResult> UpscaleAsync(int[] rgbaPixels, int width, int height)
    {
        int dstW = width * _upscaleFactor;
        int dstH = height * _upscaleFactor;

        using var rgbaBuf = _accelerator.Allocate1D(rgbaPixels);
        using var rgbaOutBuf = _accelerator.Allocate1D<int>(dstW * dstH);
        await RunTiledAsync(rgbaBuf.View, rgbaOutBuf.View, width, height, dstW, dstH);
        var result = await rgbaOutBuf.CopyToHostAsync<int>(0, dstW * dstH);
        return new SuperResResult(result, dstW, dstH, _upscaleFactor);
    }

    /// <summary>
    /// Upscale an RGBA image and return result as GPU MemoryBuffer2D for zero-copy
    /// presentation via ICanvasRenderer. Same tile-based algorithm as
    /// <see cref="UpscaleAsync"/>, but the final RGBA stays on the GPU.
    /// Caller owns the returned buffer and must dispose it.
    /// </summary>
    public async Task<(MemoryBuffer2D<int, Stride2D.DenseX> Buffer, int Width, int Height)> UpscaleGpuAsync(
        int[] rgbaPixels, int width, int height)
    {
        int dstW = width * _upscaleFactor;
        int dstH = height * _upscaleFactor;

        using var rgbaBuf = _accelerator.Allocate1D(rgbaPixels);
        var resultBuf = _accelerator.Allocate2DDenseX<int>(new Index2D(dstW, dstH));
        await RunTiledAsync(rgbaBuf.View, resultBuf.View.BaseView, width, height, dstW, dstH);
        return (resultBuf, dstW, dstH);
    }

    /// <summary>Source-pixel overlap between adjacent tiles. Larger = smoother boundaries
    /// at the cost of more tile inferences. 16 source pixels = 48 destination pixels at
    /// scale=3, which is enough to mask ESPCN's per-tile boundary differences.</summary>
    public int TileOverlap { get; set; } = 16;

    /// <summary>
    /// Tile-based super-resolution core. Y plane is built up by accumulating tile model
    /// outputs into a single destination Y buffer + per-pixel count buffer; counts are
    /// divided out at the end so overlap regions are mean-averaged. The composite kernel
    /// then combines that Y with source Cb/Cr to produce the final RGBA.
    /// </summary>
    private async Task RunTiledAsync(
        ArrayView1D<int, Stride1D.Dense> rgbaSrc,
        ArrayView1D<int, Stride1D.Dense> rgbaOut,
        int width, int height, int dstW, int dstH)
    {
        int modelW = _modelW, modelH = _modelH;
        int scale = _upscaleFactor;
        int overlap = TileOverlap;
        int strideX = Math.Max(1, modelW - overlap);
        int strideY = Math.Max(1, modelH - overlap);

        _preprocess ??= new Kernels.ImagePreprocessKernel(_accelerator);
        var postprocess = new Kernels.ImagePostprocessKernel(_accelerator);

        // Persistent per-call accelerator buffers. Re-using these across tiles avoids
        // alloc/free churn — ImagePreprocessKernel/InferenceSession internally bind
        // these views to the model graph.
        using var tileInBuf = _accelerator.Allocate1D<float>(modelW * modelH);
        using var dstY = _accelerator.Allocate1D<float>(dstW * dstH);
        using var dstCount = _accelerator.Allocate1D<int>(dstW * dstH);

        // Zero accumulators on GPU (no CPU pass).
        postprocess.ClearFloat(dstY.View);
        postprocess.ClearInt(dstCount.View);
        await _accelerator.SynchronizeAsync();

        // Tile positions: stride by (modelW - overlap), last tile anchored to edge so the
        // entire source is covered even when width / height aren't exact multiples.
        var tileXs = BuildTilePositions(width, modelW, strideX);
        var tileYs = BuildTilePositions(height, modelH, strideY);

        foreach (int tileY in tileYs)
        {
            foreach (int tileX in tileXs)
            {
                // Extract Y patch from source at this tile rect (clamps to edges if
                // source smaller than tile — happens for tiny inputs).
                _preprocess.ExtractYTile(rgbaSrc, tileInBuf.View, width, height,
                    tileX, tileY, modelW, modelH);

                var inputTensor = new Tensor(tileInBuf.View, new[] { 1, 1, modelH, modelW });
                var outputs = await _session.RunAsync(new Dictionary<string, Tensor>
                {
                    [_session.InputNames[0]] = inputTensor
                });
                await _accelerator.SynchronizeAsync();

                var output = outputs[_session.OutputNames[0]];
                int yH = output.Shape.Length >= 3 ? output.Shape[^2] : modelH * scale;
                int yW = output.Shape.Length >= 3 ? output.Shape[^1] : modelW * scale;

                // Accumulate this tile's super-res Y into the destination plane. Within
                // one kernel invocation each thread writes to a unique dst pixel, so no
                // atomics are required; sequential kernel invocations between tiles
                // serialize the writes safely.
                postprocess.AccumulateYTile(
                    output.Data.SubView(0, yH * yW),
                    dstY.View, dstCount.View,
                    yW, yH,
                    tileX * scale, tileY * scale,
                    dstW, dstH);
                await _accelerator.SynchronizeAsync();
            }
        }

        // Divide accumulator by per-pixel tile-contribution count → final Y plane.
        postprocess.NormalizeYAccumulator(dstY.View, dstCount.View);
        await _accelerator.SynchronizeAsync();

        // Combine super-res Y + source Cb/Cr → color RGBA at target dimensions.
        postprocess.SuperResCompositeYCbCr(
            rgbaSrc, dstY.View, rgbaOut,
            width, height, dstW, dstH, dstW, dstH);
        await _accelerator.SynchronizeAsync();
    }

    /// <summary>
    /// Compute tile starting positions along one dimension. Uses stride <c>step</c> with
    /// the last tile anchored to the edge so the full <c>length</c> is covered. When
    /// <c>length &lt;= tile</c>, returns a single tile at position 0 (with model-side
    /// clamping handling the partial coverage).
    /// </summary>
    private static int[] BuildTilePositions(int length, int tile, int step)
    {
        if (length <= tile) return new[] { 0 };
        var positions = new System.Collections.Generic.List<int>();
        int pos = 0;
        while (pos + tile < length)
        {
            positions.Add(pos);
            pos += step;
        }
        // Final tile anchored so its right edge meets the source edge exactly.
        int last = length - tile;
        if (positions.Count == 0 || positions[^1] != last) positions.Add(last);
        return positions.ToArray();
    }

    public void Dispose() { }
}
