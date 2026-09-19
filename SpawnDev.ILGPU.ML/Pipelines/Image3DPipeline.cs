using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnDev.ILGPU.ML.Tensors;
using TypedArray = SpawnDev.SpawnJS.JSObjects.TypedArray;

namespace SpawnDev.ILGPU.ML.Pipelines;

/// <summary>
/// Result from image-to-3D reconstruction.
/// </summary>
public record Image3DResult(
    float[] Vertices,
    int[] Indices,
    int VertexCount,
    int TriangleCount,
    double InferenceTimeMs);

/// <summary>
/// Image-to-3D reconstruction pipeline using TripoSR.
/// Single image → DINOv2 features → Triplane transformer → 3D volume → MarchingCubes → mesh.
///
/// Architecture (4 ONNX components):
///   1. image_tokenizer.onnx (343 MB): image → visual tokens
///   2. backbone.onnx (1.32 GB): tokens → triplane features
///   3. post_processor.onnx (660 KB): triplane → 3D volume
///   4. MarchingCubes (CPU): volume → triangle mesh
///
/// Usage:
///   var pipeline = new Image3DPipeline(tokenizer, backbone, postProcessor, accelerator);
///   var result = await pipeline.ReconstructAsync(rgbaPixels, width, height);
///   // Export: Formats.GLTFExporter.Export(result.Vertices, result.Indices, ...)
/// </summary>
public class Image3DPipeline : IDisposable
{
    private readonly InferenceSession _imageTokenizer;
    private readonly InferenceSession _backbone;
    private readonly InferenceSession _postProcessor;
    private readonly Accelerator _accelerator;
    private readonly Kernels.ImagePreprocessKernel _preprocess;

    /// <summary>Resolution of the 3D volume grid for MarchingCubes.</summary>
    public int VolumeResolution { get; set; } = 64;

    /// <summary>Isosurface threshold for MarchingCubes.</summary>
    public float IsoLevel { get; set; } = 0.0f;

    public Image3DPipeline(
        InferenceSession imageTokenizer,
        InferenceSession backbone,
        InferenceSession postProcessor,
        Accelerator accelerator)
    {
        _imageTokenizer = imageTokenizer;
        _backbone = backbone;
        _postProcessor = postProcessor;
        _accelerator = accelerator;
        _preprocess = new Kernels.ImagePreprocessKernel(accelerator);
    }

    /// <summary>
    /// Reconstruct a 3D mesh from a single RGBA image.
    /// Returns vertices and triangle indices for mesh export.
    /// </summary>
    /// <remarks>
    /// ⚠️ IN A BROWSER, prefer <see cref="ReconstructAsync(TypedArray, int, int)"/> or a
    /// GPU-resident view overload — this path pulls the frame onto the managed heap solely to upload it.
    /// </remarks>
    public async Task<Image3DResult> ReconstructAsync(int[] rgbaPixels, int width, int height)
    {
        using var rgbaBuf = RgbaUpload.FromManaged(_accelerator, rgbaPixels, width, height);
        return await ReconstructAsync(rgbaBuf.View, width, height).ConfigureAwait(false);
    }

    /// <summary>
    /// Browser path: RGBA still a JS typed array (e.g. <c>ImageData.Data</c>). Uploads via
    /// <see cref="MediaInterop.UploadToDevice{T}"/> — pixels never enter the .NET managed heap.
    /// </summary>
    public async Task<Image3DResult> ReconstructAsync(TypedArray rgbaPixels, int width, int height)
    {
        using var rgbaBuf = RgbaUpload.FromTypedArray(_accelerator, rgbaPixels, width, height);
        return await ReconstructAsync(rgbaBuf.View, width, height).ConfigureAwait(false);
    }

    /// <summary>GPU-resident packed RGBA — no upload.</summary>
    public Task<Image3DResult> ReconstructAsync(
        ArrayView1D<int, Stride1D.Dense> rgbaPixels, int width, int height)
        => ReconstructCoreAsync(rgbaPixels, width, height);

    /// <summary>Same as the view overload; accepts an owned buffer.</summary>
    public Task<Image3DResult> ReconstructAsync(
        MemoryBuffer1D<int, Stride1D.Dense> rgbaPixels, int width, int height)
        => ReconstructCoreAsync(rgbaPixels.View, width, height);

    private async Task<Image3DResult> ReconstructCoreAsync(
        ArrayView1D<int, Stride1D.Dense> rgbaPixels, int width, int height)
    {
        var sw = System.Diagnostics.Stopwatch.StartNew();

        // Step 1: Preprocess image to model input format
        int inputSize = 224; // TripoSR expects 224x224
        using var preprocessed = _accelerator.Allocate1D<float>(3 * inputSize * inputSize);
        _preprocess.Forward(rgbaPixels, preprocessed.View, width, height, inputSize, inputSize);

        var inputTensor = new Tensor(preprocessed.View, new[] { 1, 3, inputSize, inputSize });

        // Step 2: Image tokenizer → visual tokens
        var tokenizerOutputs = await _imageTokenizer.RunAsync(new Dictionary<string, Tensor>
        {
            [_imageTokenizer.InputNames[0]] = inputTensor
        }).ConfigureAwait(false);
        var visualTokens = tokenizerOutputs[_imageTokenizer.OutputNames[0]];

        // Step 3: Backbone → triplane features
        var backboneOutputs = await _backbone.RunAsync(new Dictionary<string, Tensor>
        {
            [_backbone.InputNames[0]] = visualTokens
        }).ConfigureAwait(false);
        var triplaneFeatures = backboneOutputs[_backbone.OutputNames[0]];

        // Step 4: Post-processor → 3D volume occupancy field
        var postOutputs = await _postProcessor.RunAsync(new Dictionary<string, Tensor>
        {
            [_postProcessor.InputNames[0]] = triplaneFeatures
        }).ConfigureAwait(false);
        var volume = postOutputs[_postProcessor.OutputNames[0]];

        // Step 5: Read volume to CPU for MarchingCubes
        int volumeSize = volume.ElementCount;
        using var readBuf = _accelerator.Allocate1D<float>(volumeSize);
        new ElementWiseKernels(_accelerator).Scale(volume.Data.SubView(0, volumeSize), readBuf.View, volumeSize, 1f);
        await _accelerator.SynchronizeAsync().ConfigureAwait(false);
        var volumeData = await readBuf.CopyToHostAsync<float>(0, volumeSize).ConfigureAwait(false);

        // Step 6: MarchingCubes → mesh
        int res = VolumeResolution;
        var mesh = Kernels.MarchingCubesKernel.ExtractSurface(volumeData, res, res, res, IsoLevel);

        sw.Stop();

        return new Image3DResult(
            mesh.Vertices, mesh.Indices,
            mesh.Vertices.Length / 3, mesh.Indices.Length / 3,
            sw.Elapsed.TotalMilliseconds);
    }

    public void Dispose()
    {
        _imageTokenizer?.Dispose();
        _backbone?.Dispose();
        _postProcessor?.Dispose();
    }
}
