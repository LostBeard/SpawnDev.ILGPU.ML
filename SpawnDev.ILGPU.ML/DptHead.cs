using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.WebGPU;

namespace SpawnDev.ILGPU.ML;

/// <summary>
/// DPT (Dense Prediction Transformer) head for DAv3.
/// Converts backbone token features into a dense depth map.
///
/// All intermediate GPU buffers are pre-allocated once (Initialize) and reused
/// across Forward calls. This avoids destroying buffers before GPU commands
/// that reference them have executed (ILGPU batches commands until SynchronizeAsync).
/// </summary>
public class DptHead : IDisposable
{
    private const int C = 384;
    private const int C_CONCAT = 768; // two 384-channel blocks concatenated
    private const int T = 37 * 37;   // 1369 tokens
    private const int GRID = 37;      // 37×37 patch grid

    private readonly WebGPUAccelerator _accelerator;
    private readonly Conv2DKernel _conv2d;
    private readonly ElementWiseKernels _elementWise;
    private readonly LayerNormKernel _layerNorm;

    // Pre-allocated intermediate buffers (alive until Dispose)
    private MemoryBuffer1D<float, Stride1D.Dense>? _concatBuf;
    private MemoryBuffer1D<float, Stride1D.Dense>? _normBuf;
    private MemoryBuffer1D<float, Stride1D.Dense>? _transposeBuf;
    private MemoryBuffer1D<float, Stride1D.Dense>? _projScratch;
    private MemoryBuffer1D<float, Stride1D.Dense>[]? _layerRnOutputs;
    private MemoryBuffer1D<float, Stride1D.Dense>? _zeroBias64;
    private MemoryBuffer1D<float, Stride1D.Dense>? _refineBuf;
    private MemoryBuffer1D<float, Stride1D.Dense>? _refineTemp;
    private MemoryBuffer1D<float, Stride1D.Dense>? _oc1Out;
    private MemoryBuffer1D<float, Stride1D.Dense>? _oc2Out;
    private MemoryBuffer1D<float, Stride1D.Dense>? _depthOut;

    public DptHead(WebGPUAccelerator accelerator, Conv2DKernel conv2d,
        ElementWiseKernels elementWise, LayerNormKernel layerNorm)
    {
        _accelerator = accelerator;
        _conv2d = conv2d;
        _elementWise = elementWise;
        _layerNorm = layerNorm;
    }

    /// <summary>
    /// Allocate all intermediate GPU buffers once. Call before Forward.
    /// </summary>
    public void Initialize()
    {
        int H = GRID, W = GRID;
        int refineC = 64;

        _concatBuf    = _accelerator.Allocate1D<float>(T * C_CONCAT);
        _normBuf      = _accelerator.Allocate1D<float>(T * C_CONCAT);
        _transposeBuf = _accelerator.Allocate1D<float>(C_CONCAT * H * W);
        _projScratch  = _accelerator.Allocate1D<float>(384 * H * W);  // max projChannels * H * W

        _zeroBias64   = _accelerator.Allocate1D<float>(64); // zero-initialized, no bias for layer_rn

        _layerRnOutputs = new MemoryBuffer1D<float, Stride1D.Dense>[4];
        for (int i = 0; i < 4; i++)
            _layerRnOutputs[i] = _accelerator.Allocate1D<float>(refineC * H * W);

        _refineBuf  = _accelerator.Allocate1D<float>(refineC * H * W);
        _refineTemp = _accelerator.Allocate1D<float>(refineC * H * W);

        _oc1Out   = _accelerator.Allocate1D<float>(32 * H * W);
        _oc2Out   = _accelerator.Allocate1D<float>(32 * H * W);
        _depthOut = _accelerator.Allocate1D<float>(2 * H * W);
    }

    /// <summary>
    /// Run a ResConfUnit: two Conv3×3 + ReLU layers with residual connection.
    /// input → ReLU → Conv3×3 → ReLU → Conv3×3 → + input
    /// </summary>
    private void ResConfUnit(
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<float, Stride1D.Dense> temp,
        ArrayView1D<float, Stride1D.Dense> conv1Weight, ArrayView1D<float, Stride1D.Dense> conv1Bias,
        ArrayView1D<float, Stride1D.Dense> conv2Weight, ArrayView1D<float, Stride1D.Dense> conv2Bias,
        int channels, int H, int W)
    {
        int totalElements = channels * H * W;

        // ReLU(input) → temp
        _elementWise.ReLU(input, temp, totalElements);

        // Conv3×3(temp) → output
        _conv2d.Forward(temp, conv1Weight, conv1Bias, output,
            channels, H, W, channels, 3, 3, stride: 1, padding: 1);

        // ReLU(output) → temp
        _elementWise.ReLU(output, temp, totalElements);

        // Conv3×3(temp) → output
        _conv2d.Forward(temp, conv2Weight, conv2Bias, output,
            channels, H, W, channels, 3, 3, stride: 1, padding: 1);

        // Residual: output += input
        _elementWise.AddInPlace(output, input, totalElements);
    }

    /// <summary>
    /// Run the DPT head. Returns the persistent depth output buffer [2, 37, 37].
    /// Caller must call SynchronizeAsync before reading results.
    /// Buffer is valid until the next Forward call or Dispose.
    /// Channel 0 = depth, Channel 1 = confidence.
    /// </summary>
    public MemoryBuffer1D<float, Stride1D.Dense> Forward(
        ArrayView1D<float, Stride1D.Dense>[] blockOutputs, // [12] block outputs, each [T*C]
        WeightLoader weights)
    {
        if (_concatBuf == null) throw new InvalidOperationException("Call Initialize() first.");

        int H = GRID, W = GRID;

        // Block pair starts: level 0=blocks[4,5], 1=[6,7], 2=[8,9], 3=[10,11]
        int[] pairStarts = { 4, 6, 8, 10 };
        // After projects: [48, 96, 192, 384] channels
        int[] projChannels = { 48, 96, 192, 384 };

        var normGamma = weights.GetView("head.norm.weight"); // [768]
        var normBeta  = weights.GetView("head.norm.bias");   // [768]

        // Step 1-3: Per-level: concat → norm → transpose → project → layer_rn
        for (int level = 0; level < 4; level++)
        {
            int bA = pairStarts[level];
            int bB = pairStarts[level] + 1;

            // Concat [T, 384] + [T, 384] → [T, 768]
            _elementWise.ConcatLastDim(blockOutputs[bA], blockOutputs[bB],
                _concatBuf.View, T, C);

            // LayerNorm over 768 channels (head.norm is shared across all levels)
            _layerNorm.Forward(_concatBuf.View, _normBuf!.View, normGamma, normBeta,
                rows: T, C: C_CONCAT);

            // Transpose [T, 768] → [768, H, W]
            _elementWise.TransposeLastTwo(_normBuf.View, _transposeBuf!.View, 1, T, C_CONCAT);

            // Conv1×1 [768 → projChannels[level]]
            int projC = projChannels[level];
            var projW = weights.GetView($"head.projects.{level}.weight");
            var projB = weights.GetView($"head.projects.{level}.bias");
            _conv2d.Forward(_transposeBuf.View, projW, projB, _projScratch!.View,
                C_CONCAT, H, W, projC, 1, 1);

            // Conv3×3 layer_rn [projC → 64] (no bias — zeroBias64 provides 64 valid zeros)
            var lrnW = weights.GetView($"head.scratch.layer{level + 1}_rn.weight");
            _conv2d.Forward(_projScratch.View, lrnW, _zeroBias64!.View,
                _layerRnOutputs![level].View, projC, H, W, 64, 3, 3, stride: 1, padding: 1);
        }

        // Step 4: RefineNet stages (bottom-up, MVP: all at 37×37, no upsampling)
        const int refineC = 64;
        int refineElements = refineC * H * W;

        // RefineNet4: level 3 → ResConfUnit2 → out_conv
        ResConfUnit(_layerRnOutputs![3].View, _refineBuf!.View, _refineTemp!.View,
            weights.GetView("head.scratch.refinenet4.resConfUnit2.conv1.weight"),
            weights.GetView("head.scratch.refinenet4.resConfUnit2.conv1.bias"),
            weights.GetView("head.scratch.refinenet4.resConfUnit2.conv2.weight"),
            weights.GetView("head.scratch.refinenet4.resConfUnit2.conv2.bias"),
            refineC, H, W);
        _conv2d.Forward(_refineBuf.View,
            weights.GetView("head.scratch.refinenet4.out_conv.weight"),
            weights.GetView("head.scratch.refinenet4.out_conv.bias"),
            _refineTemp.View, refineC, H, W, refineC, 1, 1);

        // RefineNets 3→1: fuse with previous output + two ResConfUnits + out_conv
        for (int level = 2; level >= 0; level--)
        {
            int refIdx = level + 1; // refinenet{3,2,1}

            // Add previous refinement output into current level features
            _elementWise.AddInPlace(_layerRnOutputs[level].View, _refineTemp.View, refineElements);

            // ResConfUnit1
            ResConfUnit(_layerRnOutputs[level].View, _refineBuf.View, _refineTemp.View,
                weights.GetView($"head.scratch.refinenet{refIdx}.resConfUnit1.conv1.weight"),
                weights.GetView($"head.scratch.refinenet{refIdx}.resConfUnit1.conv1.bias"),
                weights.GetView($"head.scratch.refinenet{refIdx}.resConfUnit1.conv2.weight"),
                weights.GetView($"head.scratch.refinenet{refIdx}.resConfUnit1.conv2.bias"),
                refineC, H, W);

            // ResConfUnit2
            ResConfUnit(_refineBuf.View, _layerRnOutputs[level].View, _refineTemp.View,
                weights.GetView($"head.scratch.refinenet{refIdx}.resConfUnit2.conv1.weight"),
                weights.GetView($"head.scratch.refinenet{refIdx}.resConfUnit2.conv1.bias"),
                weights.GetView($"head.scratch.refinenet{refIdx}.resConfUnit2.conv2.weight"),
                weights.GetView($"head.scratch.refinenet{refIdx}.resConfUnit2.conv2.bias"),
                refineC, H, W);

            // out_conv 1×1
            _conv2d.Forward(_layerRnOutputs[level].View,
                weights.GetView($"head.scratch.refinenet{refIdx}.out_conv.weight"),
                weights.GetView($"head.scratch.refinenet{refIdx}.out_conv.bias"),
                _refineTemp.View, refineC, H, W, refineC, 1, 1);
        }

        // Step 5: Output convolutions
        // output_conv1: [64→32, 3×3] + ReLU
        _conv2d.Forward(_refineTemp.View,
            weights.GetView("head.scratch.output_conv1.weight"),
            weights.GetView("head.scratch.output_conv1.bias"),
            _oc1Out!.View, 64, H, W, 32, 3, 3, stride: 1, padding: 1);
        _elementWise.ReLUInPlace(_oc1Out.View, 32 * H * W);

        // output_conv2.0: [32→32, 3×3] + ReLU
        _conv2d.Forward(_oc1Out.View,
            weights.GetView("head.scratch.output_conv2.0.weight"),
            weights.GetView("head.scratch.output_conv2.0.bias"),
            _oc2Out!.View, 32, H, W, 32, 3, 3, stride: 1, padding: 1);
        _elementWise.ReLUInPlace(_oc2Out.View, 32 * H * W);

        // output_conv2.2: [32→2, 1×1] (depth + confidence)
        _conv2d.Forward(_oc2Out.View,
            weights.GetView("head.scratch.output_conv2.2.weight"),
            weights.GetView("head.scratch.output_conv2.2.bias"),
            _depthOut!.View, 32, H, W, 2, 1, 1);

        // [2, 37, 37] — ch0=depth, ch1=confidence
        return _depthOut;
    }

    public void Dispose()
    {
        _concatBuf?.Dispose();
        _normBuf?.Dispose();
        _transposeBuf?.Dispose();
        _projScratch?.Dispose();
        _zeroBias64?.Dispose();
        if (_layerRnOutputs != null)
            foreach (var b in _layerRnOutputs) b?.Dispose();
        _refineBuf?.Dispose();
        _refineTemp?.Dispose();
        _oc1Out?.Dispose();
        _oc2Out?.Dispose();
        _depthOut?.Dispose();
    }
}
