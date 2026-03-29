using ILGPU;
using ILGPU.Runtime;

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
    private const int T = 1369;      // 37×37 patches (spatial tokens only)
    private const int T_FULL = 1370; // T + 1 CLS token (what the backbone actually processes)
    private const int INPUT_SIZE = 518;
    private const int PATCH_SIZE = 14;
    private const int GRID_SIZE = 37; // 518 / 14 = 37

    private readonly Accelerator _accelerator;
    private readonly WeightLoader _weights;

    // Kernels
    private readonly Conv2DKernel _conv2d;
    private readonly ConvTranspose2DKernel _convTranspose;
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

    /// <param name="forceSimpleMatMul">When true, bypasses tiled MatMul even on WebGPU.
    /// Use for debugging — if depth improves, the tiled MatMul is the bug.</param>
    public Dav3Inference(Accelerator accelerator, WeightLoader weights, bool forceSimpleMatMul = false)
    {
        _accelerator = accelerator;
        _weights = weights;

        _conv2d = new Conv2DKernel(accelerator);
        _convTranspose = new ConvTranspose2DKernel(accelerator);
        _matMul = new MatMulKernel(accelerator, forceSimpleMatMul);
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

        // Allocate persistent buffers — backbone processes T_FULL tokens (CLS + patches)
        _patchEmbedOut = _accelerator.Allocate1D<float>(C * GRID_SIZE * GRID_SIZE);
        _tokenBuf1 = _accelerator.Allocate1D<float>(T_FULL * C);
        _tokenBuf2 = _accelerator.Allocate1D<float>(T_FULL * C);
        _tmpBuffers = new TransformerBlock.TempBuffers(_accelerator, T_FULL);

        // Saved block outputs for DPT head (blocks 4-11) — patches only, no CLS
        _blockOutputs = new MemoryBuffer1D<float, Stride1D.Dense>[12];
        for (int i = 4; i < 12; i++)
            _blockOutputs[i] = _accelerator.Allocate1D<float>(T * C);

        // DPT head — pre-allocates all intermediate buffers to avoid lifetime issues
        _dptHead = new DptHead(_accelerator, _conv2d, _convTranspose, _elementWise, _layerNorm);
        _dptHead.Initialize();

        if (InferenceSession.VerboseLogging) Console.WriteLine($"[Dav3] Initialized: 12 blocks, buffers allocated (build: 2026-03-18T10:00)");
    }

    /// <summary>
    /// Run the full backbone: patch embed → position embed → prepend CLS → 12 transformer blocks.
    /// Input: preprocessed NCHW float [3, 518, 518] on GPU.
    /// Returns: token features [T_FULL, C] = [1370, 384] on GPU (CLS at index 0, patches at 1-1369).
    /// </summary>
    public ArrayView1D<float, Stride1D.Dense> RunBackbone(ArrayView1D<float, Stride1D.Dense> input)
    {
        if (_blockWeights == null || _tmpBuffers == null)
            throw new InvalidOperationException("Call Initialize() first.");

        // Step 1: Patch embedding — Conv2D 14×14 stride 14
        var patchWeight = _weights.GetView("backbone.pretrained.patch_embed.proj.weight");
        var patchBias = _weights.GetView("backbone.pretrained.patch_embed.proj.bias");
        _conv2d.Forward(input, patchWeight, patchBias, _patchEmbedOut!.View,
            inC: 3, inH: INPUT_SIZE, inW: INPUT_SIZE,
            outC: C, kH: PATCH_SIZE, kW: PATCH_SIZE,
            stride: PATCH_SIZE, padding: 0);

        // Step 2: Reshape [384, 37, 37] → [1369, 384] (transpose: CHW → HW×C)
        // Write patches to _tokenBuf1 starting at offset C (leave room for CLS at position 0)
        _elementWise.TransposeLastTwo(_patchEmbedOut!.View, _tokenBuf1!.View.SubView(C, T * C), 1, C, T);

        // Step 3: Add position embedding to PATCHES only (positions 1-1369, not CLS)
        var posEmbed = _weights.TryGetView("/backbone/Transpose_output_0");
        if (posEmbed != null)
        {
            _elementWise.TransposeLastTwo(posEmbed.Value, _tokenBuf2!.View.SubView(0, T * C), 1, C, T);
            _elementWise.AddInPlace(_tokenBuf1!.View.SubView(C, T * C), _tokenBuf2!.View.SubView(0, T * C), T * C);
        }

        // Step 4: Prepend CLS token at position 0
        // ONNX graph adds cls_token TWICE: once as token value, once as CLS "position embedding"
        _elementWise.ScaleInPlace(_tokenBuf1!.View.SubView(0, C), C, 0f);
        var clsToken = _weights.GetView("backbone.pretrained.cls_token");
        _elementWise.AddInPlace(_tokenBuf1!.View.SubView(0, C), clsToken.SubView(0, C), C);
        _elementWise.AddInPlace(_tokenBuf1!.View.SubView(0, C), clsToken.SubView(0, C), C);
        // _tokenBuf1 now has [2*CLS, patch1+pos1, patch2+pos2, ...] = [1370, 384]

        // Step 5: Run 12 transformer blocks with T_FULL=1370 tokens
        // Zero block output buffers (patches only, T=1369)
        for (int i = 4; i < 12; i++)
            _elementWise.ScaleInPlace(_blockOutputs![i].View, T * C, 0f);

        var currentIn = _tokenBuf1!.View.SubView(0, T_FULL * C);
        var currentOut = _tokenBuf2!.View.SubView(0, T_FULL * C);

        for (int b = 0; b < 12; b++)
        {
            _transformer.Forward(currentIn, currentOut, _blockWeights[b], T_FULL, _tmpBuffers);

            // Save block output for DPT head (blocks 4-11) — patches only, skip CLS at index 0
            if (b >= 4 && _blockOutputs != null && _blockOutputs[b] != null)
                _elementWise.AddInPlace(_blockOutputs![b].View, currentOut.SubView(C, T * C), T * C);

            (currentIn, currentOut) = (currentOut, currentIn);
        }

        _lastResultBuf = _tokenBuf1;
        return currentIn;
    }

    /// <summary>
    /// Run backbone with per-block diagnostic output.
    /// Syncs GPU after each block and logs min/max/std/first values.
    /// SLOW — for debugging only. Compare output with ORT reference values.
    /// </summary>
    public async Task<ArrayView1D<float, Stride1D.Dense>> RunBackboneDiagnosticAsync(
        ArrayView1D<float, Stride1D.Dense> input)
    {
        if (_blockWeights == null || _tmpBuffers == null)
            throw new InvalidOperationException("Call Initialize() first.");

        // Steps 1-4: same as RunBackbone
        var patchWeight = _weights.GetView("backbone.pretrained.patch_embed.proj.weight");
        var patchBias = _weights.GetView("backbone.pretrained.patch_embed.proj.bias");
        _conv2d.Forward(input, patchWeight, patchBias, _patchEmbedOut!.View,
            inC: 3, inH: INPUT_SIZE, inW: INPUT_SIZE,
            outC: C, kH: PATCH_SIZE, kW: PATCH_SIZE,
            stride: PATCH_SIZE, padding: 0);
        _elementWise.TransposeLastTwo(_patchEmbedOut!.View, _tokenBuf1!.View.SubView(C, T * C), 1, C, T);
        var posEmbed = _weights.TryGetView("/backbone/Transpose_output_0");
        if (posEmbed != null)
        {
            _elementWise.TransposeLastTwo(posEmbed.Value, _tokenBuf2!.View.SubView(0, T * C), 1, C, T);
            _elementWise.AddInPlace(_tokenBuf1!.View.SubView(C, T * C), _tokenBuf2!.View.SubView(0, T * C), T * C);
        }
        _elementWise.ScaleInPlace(_tokenBuf1!.View.SubView(0, C), C, 0f);
        var clsToken = _weights.GetView("backbone.pretrained.cls_token");
        _elementWise.AddInPlace(_tokenBuf1!.View.SubView(0, C), clsToken.SubView(0, C), C);
        _elementWise.AddInPlace(_tokenBuf1!.View.SubView(0, C), clsToken.SubView(0, C), C);

        // Log input to block 0
        await _accelerator.SynchronizeAsync();
        await LogBufferStatsAsync("pre-backbone", _tokenBuf1!, 0, T_FULL * C);

        for (int i = 4; i < 12; i++)
            _elementWise.ScaleInPlace(_blockOutputs![i].View, T * C, 0f);

        var currentIn = _tokenBuf1!.View.SubView(0, T_FULL * C);
        var currentOut = _tokenBuf2!.View.SubView(0, T_FULL * C);
        // Track which buffer holds currentOut for readback
        var currentOutBuf = _tokenBuf2!;

        for (int b = 0; b < 12; b++)
        {
            if (b == 0)
            {
                // CPU cross-check for block 0: verify GPU LayerNorm1 + QKV MatMul against CPU
                await VerifyBlock0CpuAsync(currentIn);

                // Intra-block diagnostic for block 0
                await _transformer.ForwardDiagnosticAsync(
                    currentIn, currentOut, _blockWeights[b], T_FULL, _tmpBuffers,
                    _accelerator, b);
            }
            else
            {
                _transformer.Forward(currentIn, currentOut, _blockWeights[b], T_FULL, _tmpBuffers);
            }

            if (b >= 4 && _blockOutputs != null && _blockOutputs[b] != null)
                _elementWise.AddInPlace(_blockOutputs![b].View, currentOut.SubView(C, T * C), T * C);

            // Diagnostic: sync and log after each block
            await _accelerator.SynchronizeAsync();
            await LogBufferStatsAsync($"block[{b}]", currentOutBuf, 0, T_FULL * C);

            (currentIn, currentOut) = (currentOut, currentIn);
            currentOutBuf = (currentOutBuf == _tokenBuf1!) ? _tokenBuf2! : _tokenBuf1!;
        }

        _lastResultBuf = _tokenBuf1;
        return currentIn;
    }

    /// <summary>
    /// Read GPU ArrayView data to CPU via temp buffer.
    /// </summary>
    private async Task<float[]> ReadViewToCpuAsync(ArrayView1D<float, Stride1D.Dense> view, int count)
    {
        using var temp = _accelerator.Allocate1D<float>(count);
        _elementWise.ScaleInPlace(temp.View, count, 0f);
        _elementWise.AddInPlace(temp.View, view.SubView(0, count), count);
        await _accelerator.SynchronizeAsync();
        return await temp.CopyToHostAsync<float>(0, count);
    }

    /// <summary>
    /// CPU cross-check: verify GPU LayerNorm1 + QKV MatMul against CPU for block 0, row 0.
    /// Definitively detects WGSL code generation bugs.
    /// </summary>
    private async Task VerifyBlock0CpuAsync(ArrayView1D<float, Stride1D.Dense> input)
    {
        var w = _blockWeights![0];
        await _accelerator.SynchronizeAsync();

        // Read back row 0 of input (CLS token, 384 elements)
        var inputRow = await _tokenBuf1!.CopyToHostAsync<float>(0, C);
        // Read back norm1 weights and bias
        var norm1W = await ReadViewToCpuAsync(w.Norm1Weight, C);
        var norm1B = await ReadViewToCpuAsync(w.Norm1Bias, C);

        // CPU LayerNorm on row 0
        float sum = 0;
        for (int i = 0; i < C; i++) sum += inputRow[i];
        float mean = sum / C;
        float varSum = 0;
        for (int i = 0; i < C; i++) { float d = inputRow[i] - mean; varSum += d * d; }
        float invStd = 1f / MathF.Sqrt(varSum / C + 1e-6f);
        var cpuLn = new float[C];
        for (int i = 0; i < C; i++)
            cpuLn[i] = norm1W[i] * ((inputRow[i] - mean) * invStd) + norm1B[i];

        // GPU LN1 output for row 0
        _layerNorm.Forward(input, _tmpBuffers!.Norm, w.Norm1Weight, w.Norm1Bias, T_FULL, C);
        await _accelerator.SynchronizeAsync();
        var gpuLn = await _tmpBuffers.NormBuf.CopyToHostAsync<float>(0, C);

        // Compare
        float maxErr = 0;
        for (int i = 0; i < C; i++)
            maxErr = MathF.Max(maxErr, MathF.Abs(cpuLn[i] - gpuLn[i]));
        if (InferenceSession.VerboseLogging) Console.WriteLine($"[CPUCheck] LN1 row0: CPU first5=[{string.Join(", ", cpuLn.Take(5).Select(v => v.ToString("F4")))}]");
        if (InferenceSession.VerboseLogging) Console.WriteLine($"[CPUCheck] LN1 row0: GPU first5=[{string.Join(", ", gpuLn.Take(5).Select(v => v.ToString("F4")))}]");
        if (InferenceSession.VerboseLogging) Console.WriteLine($"[CPUCheck] LN1 row0: maxErr={maxErr:E3} ({(maxErr < 1e-3 ? "PASS" : "FAIL")})");

        // Now check QKV MatMul: compute dot product on CPU for first 5 output elements
        var qkvWeight = await ReadViewToCpuAsync(w.QkvWeight, C * 3 * C);
        var qkvBias = await ReadViewToCpuAsync(w.QkvBias, 3 * C);

        // CPU: QKV[0, n] = sum_k(LN1[0,k] * W[k,n]) + bias[n], W is [384, 1152] row-major
        var cpuQkv = new float[5];
        for (int n = 0; n < 5; n++)
        {
            float s = 0;
            for (int k = 0; k < C; k++)
                s += gpuLn[k] * qkvWeight[k * 3 * C + n];
            cpuQkv[n] = s + qkvBias[n];
        }

        // GPU QKV output row 0
        _matMul.MatMul(_tmpBuffers.Norm, w.QkvWeight, _tmpBuffers.Qkv, T_FULL, C, 3 * C);
        _elementWise.AddBias(_tmpBuffers.Qkv, w.QkvBias, T_FULL * 3 * C, 3 * C);
        await _accelerator.SynchronizeAsync();
        var gpuQkv = await _tmpBuffers.QkvBuf.CopyToHostAsync<float>(0, 5);

        if (InferenceSession.VerboseLogging) Console.WriteLine($"[CPUCheck] QKV row0: CPU first5=[{string.Join(", ", cpuQkv.Select(v => v.ToString("F4")))}]");
        if (InferenceSession.VerboseLogging) Console.WriteLine($"[CPUCheck] QKV row0: GPU first5=[{string.Join(", ", gpuQkv.Select(v => v.ToString("F4")))}]");
        float qkvErr = 0;
        for (int i = 0; i < 5; i++) qkvErr = MathF.Max(qkvErr, MathF.Abs(cpuQkv[i] - gpuQkv[i]));
        if (InferenceSession.VerboseLogging) Console.WriteLine($"[CPUCheck] QKV row0: maxErr={qkvErr:E3} ({(qkvErr < 0.05 ? "PASS" : "FAIL")})");
    }

    private async Task LogBufferStatsAsync(string label, MemoryBuffer1D<float, Stride1D.Dense> buffer, int offset, int count)
    {
        var data = await buffer.CopyToHostAsync<float>(offset, count);
        float min = float.MaxValue, max = float.MinValue;
        double sum = 0, sumSq = 0;
        for (int i = 0; i < data.Length; i++)
        {
            float v = data[i];
            if (v < min) min = v;
            if (v > max) max = v;
            sum += v;
            sumSq += (double)v * v;
        }
        double mean = sum / data.Length;
        double std = Math.Sqrt(sumSq / data.Length - mean * mean);
        var first5 = string.Join(", ", data.Take(5).Select(v => v.ToString("F4")));
        if (InferenceSession.VerboseLogging) Console.WriteLine($"[Dav3Diag] {label}: min={min:F4} max={max:F4} std={std:F4} first5=[{first5}]");
    }

    private MemoryBuffer1D<float, Stride1D.Dense>? _lastResultBuf;
    private MemoryBuffer1D<float, Stride1D.Dense>? _lastDepthBuf;

    /// <summary>
    /// Run full inference: backbone + DPT head → depth map [2, 518, 518].
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

    /// <summary>
    /// Diagnostic version: runs backbone with per-block logging, then DPT head.
    /// SLOW — syncs GPU after every block. For debugging backbone divergence.
    /// </summary>
    public async Task<MemoryBuffer1D<float, Stride1D.Dense>> RunFullDiagnosticAsync(
        ArrayView1D<float, Stride1D.Dense> input)
    {
        await RunBackboneDiagnosticAsync(input);

        var blockViews = new ArrayView1D<float, Stride1D.Dense>[12];
        for (int i = 4; i < 12; i++)
            blockViews[i] = _blockOutputs![i].View;

        _lastDepthBuf = _dptHead!.Forward(blockViews, _weights);

        return _lastDepthBuf;
    }

    /// <summary>Read back depth output to CPU. Call after RunFull. Returns [2, 296, 296] (depth + confidence).</summary>
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

    /// <summary>Get layer4_rn buffer view for diagnostics. Returns null if not available.</summary>
    public ArrayView1D<float, Stride1D.Dense>? DebugGetLayerRn3() => _dptHead?.DebugGetLayerRnView(3);

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
        if (InferenceSession.VerboseLogging) Console.WriteLine($"[DptDbg] After concat: maxAbs={concatMax:E3}  [{string.Join(", ", concatSample.Select(v => v.ToString("E2")))}]");

        // Stage 2: LayerNorm [T, 768]
        using var normBuf = _accelerator.Allocate1D<float>(T * 768);
        var normGamma = _weights.GetView("head.norm.weight");
        var normBeta  = _weights.GetView("head.norm.bias");
        _layerNorm.Forward(concatBuf.View, normBuf.View, normGamma, normBeta, rows: T, C: 768);
        await _accelerator.SynchronizeAsync();
        var normSample = await normBuf.CopyToHostAsync<float>(0, count);
        float normMax = normSample.Max(v => MathF.Abs(v));
        if (InferenceSession.VerboseLogging) Console.WriteLine($"[DptDbg] After layernorm: maxAbs={normMax:E3}  [{string.Join(", ", normSample.Select(v => v.ToString("E2")))}]");

        // Stage 3: TransposeLastTwo [T,768] → [768,T]
        using var transposeBuf = _accelerator.Allocate1D<float>(768 * T);
        _elementWise.TransposeLastTwo(normBuf.View, transposeBuf.View, 1, T, 768);
        await _accelerator.SynchronizeAsync();
        var transSample = await transposeBuf.CopyToHostAsync<float>(0, count);
        float transMax = transSample.Max(v => MathF.Abs(v));
        if (InferenceSession.VerboseLogging) Console.WriteLine($"[DptDbg] After transpose: maxAbs={transMax:E3}");

        // Stage 4: Conv1×1 [768→48]
        using var projScratch = _accelerator.Allocate1D<float>(48 * GRID_SIZE * GRID_SIZE);
        var projW = _weights.GetView("head.projects.0.weight");
        var projB = _weights.GetView("head.projects.0.bias");
        _conv2d.Forward(transposeBuf.View, projW, projB, projScratch.View,
            768, GRID_SIZE, GRID_SIZE, 48, 1, 1);
        await _accelerator.SynchronizeAsync();
        var projSample = await projScratch.CopyToHostAsync<float>(0, count);
        float projMax = projSample.Max(v => MathF.Abs(v));
        if (InferenceSession.VerboseLogging) Console.WriteLine($"[DptDbg] After proj[0] (768→48): maxAbs={projMax:E3}  [{string.Join(", ", projSample.Select(v => v.ToString("E2")))}]");

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
