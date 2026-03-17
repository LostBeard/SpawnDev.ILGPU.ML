using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.WebGPU;

namespace SpawnDev.ILGPU.ML;

/// <summary>
/// Complete DAv3 (Depth Anything V3 Small) inference pipeline.
/// Runs entirely on GPU via ILGPU — no ONNX Runtime dependency.
///
/// Pipeline:
///   1. Patch embedding: Conv2D 14×14 stride 14 → [384, 37, 37] → flatten → [1369, 384]
///   2. Position embedding: add learned [1, 384, 37, 37] → [1369, 384]
///   3. 12 Transformer blocks (attention + MLP + residual)
///   4. DPT head: 4 refinement stages → depth map
///
/// Usage:
///   var inference = new Dav3Inference(accelerator, weightLoader);
///   await inference.InitializeAsync();
///   var depth = inference.Run(preprocessedInput); // [3, 518, 518] NCHW
/// </summary>
public class Dav3Inference
{
    private const int C = 384;       // Embedding dimension
    private const int T = 1369;      // 37×37 patches
    private const int INPUT_SIZE = 518;
    private const int PATCH_SIZE = 14;
    private const int GRID_SIZE = 37; // 518 / 14 = 37

    private readonly WebGPUAccelerator _accelerator;
    private readonly WeightLoader _weights;

    // Kernels
    private readonly Conv2DKernel _conv2d;
    private readonly MatMulKernel _matMul;
    private readonly LayerNormKernel _layerNorm;
    private readonly SoftmaxKernel _softmax;
    private readonly ElementWiseKernels _elementWise;
    private readonly AttentionKernels _attention;
    private readonly TransformerBlock _transformer;

    // Pre-resolved block weights
    private TransformerBlock.BlockWeights[]? _blockWeights;
    private TransformerBlock.TempBuffers? _tmpBuffers;

    // Persistent buffers
    private MemoryBuffer1D<float, Stride1D.Dense>? _patchEmbedOut;
    private MemoryBuffer1D<float, Stride1D.Dense>? _tokenBuf1;
    private MemoryBuffer1D<float, Stride1D.Dense>? _tokenBuf2;

    // Saved block outputs for DPT head (blocks 4-11)
    private MemoryBuffer1D<float, Stride1D.Dense>[]? _blockOutputs;
    private DptHead? _dptHead;

    public bool IsInitialized => _blockWeights != null;

    public Dav3Inference(WebGPUAccelerator accelerator, WeightLoader weights)
    {
        _accelerator = accelerator;
        _weights = weights;

        _conv2d = new Conv2DKernel(accelerator);
        _matMul = new MatMulKernel(accelerator);
        _layerNorm = new LayerNormKernel(accelerator);
        _softmax = new SoftmaxKernel(accelerator);
        _elementWise = new ElementWiseKernels(accelerator);
        _attention = new AttentionKernels(accelerator);
        _transformer = new TransformerBlock(_matMul, _layerNorm, _softmax, _elementWise, _attention);
    }

    /// <summary>
    /// Initialize: resolve all 12 block weights and allocate persistent buffers.
    /// Call once after weights are loaded.
    /// </summary>
    public void Initialize()
    {
        if (!_weights.IsLoaded) throw new InvalidOperationException("Weights not loaded yet.");

        // Resolve weights for all 12 blocks
        _blockWeights = new TransformerBlock.BlockWeights[12];
        for (int i = 0; i < 12; i++)
            _blockWeights[i] = TransformerBlock.ResolveWeights(_weights, i);

        // Allocate persistent buffers
        _patchEmbedOut = _accelerator.Allocate1D<float>(C * GRID_SIZE * GRID_SIZE);
        _tokenBuf1 = _accelerator.Allocate1D<float>(T * C);
        _tokenBuf2 = _accelerator.Allocate1D<float>(T * C);
        _tmpBuffers = new TransformerBlock.TempBuffers(_accelerator, T);

        // Saved block outputs for DPT head (blocks 4-11)
        _blockOutputs = new MemoryBuffer1D<float, Stride1D.Dense>[12];
        for (int i = 4; i < 12; i++)
            _blockOutputs[i] = _accelerator.Allocate1D<float>(T * C);

        // DPT head — pre-allocates all intermediate buffers to avoid lifetime issues
        _dptHead = new DptHead(_accelerator, _conv2d, _elementWise, _layerNorm);
        _dptHead.Initialize();

        Console.WriteLine($"[Dav3] Initialized: 12 blocks, buffers allocated");
    }

    /// <summary>
    /// Run the full backbone: patch embed → position embed → 12 transformer blocks.
    /// Input: preprocessed NCHW float [3, 518, 518] on GPU.
    /// Returns: token features [T, C] = [1369, 384] on GPU.
    /// </summary>
    public ArrayView1D<float, Stride1D.Dense> RunBackbone(ArrayView1D<float, Stride1D.Dense> input)
    {
        if (_blockWeights == null || _tmpBuffers == null)
            throw new InvalidOperationException("Call Initialize() first.");

        // Step 1: Patch embedding — Conv2D 14×14 stride 14
        // Input: [3, 518, 518], Weight: [384, 3, 14, 14], Output: [384, 37, 37]
        var patchWeight = _weights.GetView("backbone.pretrained.patch_embed.proj.weight");
        var patchBias = _weights.GetView("backbone.pretrained.patch_embed.proj.bias");
        _conv2d.Forward(input, patchWeight, patchBias, _patchEmbedOut!.View,
            inC: 3, inH: INPUT_SIZE, inW: INPUT_SIZE,
            outC: C, kH: PATCH_SIZE, kW: PATCH_SIZE,
            stride: PATCH_SIZE, padding: 0);

        // Step 2: Reshape [384, 37, 37] → [1369, 384] (transpose: CHW → HW×C)
        // The patch embed output is [C, H, W]. We need [H*W, C] for the transformer.
        // Use TransposeLastTwo to go from [1, 384, 1369] → [1, 1369, 384]
        _elementWise.TransposeLastTwo(_patchEmbedOut!.View, _tokenBuf1!.View, 1, C, T);

        // Step 3: Add position embedding
        // DAv3 stores position embedding as [1, 384, 37, 37] transposed output
        var posEmbed = _weights.TryGetView("/backbone/Transpose_output_0");
        if (posEmbed != null)
        {
            // Position embedding is [1, 384, 37, 37] = [384, 1369] after flatten
            // Need to transpose [384, 1369] → [1369, 384] then add
            _elementWise.TransposeLastTwo(posEmbed.Value, _tokenBuf2!.View, 1, C, T);
            _elementWise.AddInPlace(_tokenBuf1!.View, _tokenBuf2!.View, T * C);
        }

        // Step 4: Run 12 transformer blocks, saving outputs 4-11 for DPT head
        // Zero block output buffers first so AddInPlace acts as a copy (supports repeated calls)
        for (int i = 4; i < 12; i++)
            _elementWise.ScaleInPlace(_blockOutputs![i].View, T * C, 0f);

        var currentIn = _tokenBuf1!.View;
        var currentOut = _tokenBuf2!.View;

        for (int b = 0; b < 12; b++)
        {
            _transformer.Forward(currentIn, currentOut, _blockWeights[b], T, _tmpBuffers);

            // Save block output for DPT head (blocks 4-11)
            if (b >= 4 && _blockOutputs != null && _blockOutputs[b] != null)
                _elementWise.AddInPlace(_blockOutputs![b].View, currentOut, T * C);

            (currentIn, currentOut) = (currentOut, currentIn);
        }

        _lastResultBuf = _tokenBuf1;
        return currentIn;
    }

    private MemoryBuffer1D<float, Stride1D.Dense>? _lastResultBuf;
    private MemoryBuffer1D<float, Stride1D.Dense>? _lastDepthBuf;

    /// <summary>
    /// Run full inference: backbone + DPT head → depth map [2, 37, 37].
    /// Channel 0 = depth, Channel 1 = confidence.
    /// Input: preprocessed NCHW float [3, 518, 518] on GPU.
    /// </summary>
    public MemoryBuffer1D<float, Stride1D.Dense> RunFull(ArrayView1D<float, Stride1D.Dense> input)
    {
        // Run backbone (saves block outputs 4-11)
        RunBackbone(input);

        // Run DPT head with saved block outputs
        var blockViews = new ArrayView1D<float, Stride1D.Dense>[12];
        for (int i = 4; i < 12; i++)
            blockViews[i] = _blockOutputs![i].View;

        _lastDepthBuf = _dptHead!.Forward(blockViews, _weights);

        return _lastDepthBuf;
    }

    /// <summary>Read back depth output to CPU. Call after RunFull. Returns [2, 37, 37] (depth + confidence).</summary>
    public async Task<float[]> ReadDepthOutputAsync()
    {
        if (_lastDepthBuf == null) throw new InvalidOperationException("Run RunFull first.");
        return await _lastDepthBuf.CopyToHostAsync<float>(0, _lastDepthBuf.Length);
    }

    /// <summary>Read back backbone output to CPU. Call after RunBackbone.</summary>
    public async Task<float[]> ReadBackboneOutputAsync(int offset, int count)
    {
        if (_lastResultBuf == null) throw new InvalidOperationException("Run RunBackbone first.");
        return await _lastResultBuf.CopyToHostAsync<float>(offset, count);
    }

    /// <summary>Read first N values from a saved block output (blocks 4-11) for diagnostics.</summary>
    public async Task<float[]> ReadBlockOutputAsync(int blockIndex, int count = 10)
    {
        if (_blockOutputs == null || _blockOutputs[blockIndex] == null)
            throw new InvalidOperationException($"blockOutputs[{blockIndex}] not available.");
        return await _blockOutputs[blockIndex].CopyToHostAsync<float>(0, count);
    }

    /// <summary>
    /// Diagnostic: run just the DPT head's first 3 stages for level 0 (concat→norm→transpose→proj)
    /// and return the first N values of the projection output. Reveals if DPT head has zero input.
    /// </summary>
    public async Task<float[]> DebugDptLevel0ProjAsync(int count = 10)
    {
        if (_blockOutputs == null) throw new InvalidOperationException("Call Initialize() and RunBackbone() first.");

        // Stage 1: ConcatLastDim [T,384] + [T,384] → [T,768]
        using var concatBuf = _accelerator.Allocate1D<float>(T * 768);
        _elementWise.ConcatLastDim(_blockOutputs[4].View, _blockOutputs[5].View, concatBuf.View, T, C);
        await _accelerator.SynchronizeAsync();
        var concatSample = await concatBuf.CopyToHostAsync<float>(0, count);
        float concatMax = concatSample.Max(v => MathF.Abs(v));
        Console.WriteLine($"[DptDbg] After concat: maxAbs={concatMax:E3}  [{string.Join(", ", concatSample.Select(v => v.ToString("E2")))}]");

        // Stage 2: LayerNorm [T, 768]
        using var normBuf = _accelerator.Allocate1D<float>(T * 768);
        var normGamma = _weights.GetView("head.norm.weight");
        var normBeta  = _weights.GetView("head.norm.bias");
        _layerNorm.Forward(concatBuf.View, normBuf.View, normGamma, normBeta, rows: T, C: 768);
        await _accelerator.SynchronizeAsync();
        var normSample = await normBuf.CopyToHostAsync<float>(0, count);
        float normMax = normSample.Max(v => MathF.Abs(v));
        Console.WriteLine($"[DptDbg] After layernorm: maxAbs={normMax:E3}  [{string.Join(", ", normSample.Select(v => v.ToString("E2")))}]");

        // Stage 3: TransposeLastTwo [T,768] → [768,T]
        using var transposeBuf = _accelerator.Allocate1D<float>(768 * T);
        _elementWise.TransposeLastTwo(normBuf.View, transposeBuf.View, 1, T, 768);
        await _accelerator.SynchronizeAsync();
        var transSample = await transposeBuf.CopyToHostAsync<float>(0, count);
        float transMax = transSample.Max(v => MathF.Abs(v));
        Console.WriteLine($"[DptDbg] After transpose: maxAbs={transMax:E3}");

        // Stage 4: Conv1×1 [768→48]
        using var projScratch = _accelerator.Allocate1D<float>(48 * GRID_SIZE * GRID_SIZE);
        var projW = _weights.GetView("head.projects.0.weight");
        var projB = _weights.GetView("head.projects.0.bias");
        _conv2d.Forward(transposeBuf.View, projW, projB, projScratch.View,
            768, GRID_SIZE, GRID_SIZE, 48, 1, 1);
        await _accelerator.SynchronizeAsync();
        var projSample = await projScratch.CopyToHostAsync<float>(0, count);
        float projMax = projSample.Max(v => MathF.Abs(v));
        Console.WriteLine($"[DptDbg] After proj[0] (768→48): maxAbs={projMax:E3}  [{string.Join(", ", projSample.Select(v => v.ToString("E2")))}]");

        return projSample;
    }

    /// <summary>Dispose persistent GPU buffers.</summary>
    public void Dispose()
    {
        _patchEmbedOut?.Dispose();
        _tokenBuf1?.Dispose();
        _tokenBuf2?.Dispose();
        _tmpBuffers?.Dispose();
        _dptHead?.Dispose();
        if (_blockOutputs != null)
            for (int i = 4; i < 12; i++) _blockOutputs[i]?.Dispose();
    }
}
