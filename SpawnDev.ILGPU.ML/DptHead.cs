using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.WebGPU;

namespace SpawnDev.ILGPU.ML;

/// <summary>
/// DPT (Dense Prediction Transformer) head for DAv3.
/// Converts backbone token features into a dense depth map.
///
/// Architecture:
///   1. Concatenate adjacent block pairs: [4+5, 6+7, 8+9, 10+11] → 768 channels each
///   2. LayerNorm (768) each level
///   3. Reshape [768, H, W] → Conv1×1 projects → [48/96/192/384, H, W]
///   4. Resize layers (ConvTranspose or bilinear) → multi-scale features
///   5. Layer RN: Conv3×3 → [64, H, W] per level
///   6. RefineNet stages (bottom-up): residual Conv3×3 + ReLU + upsample
///   7. Output convolutions → [1, H, W] depth map
///
/// For MVP: simplified implementation using bilinear resize instead of ConvTranspose.
/// </summary>
public class DptHead
{
    private const int C = 384;
    private const int C_CONCAT = 768; // two 384-channel blocks concatenated
    private const int T = 37 * 37;   // 1369 tokens
    private const int GRID = 37;      // 37×37 patch grid

    private readonly WebGPUAccelerator _accelerator;
    private readonly Conv2DKernel _conv2d;
    private readonly ElementWiseKernels _elementWise;
    private readonly LayerNormKernel _layerNorm;

    public DptHead(WebGPUAccelerator accelerator, Conv2DKernel conv2d,
        ElementWiseKernels elementWise, LayerNormKernel layerNorm)
    {
        _accelerator = accelerator;
        _conv2d = conv2d;
        _elementWise = elementWise;
        _layerNorm = layerNorm;
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
    /// Run the DPT head. Takes block pairs from backbone, concatenates them to 768 channels,
    /// applies LayerNorm, projects, layer_rn convolutions, and RefineNet fusion.
    ///
    /// Architecture (simplified MVP: no multi-scale resize, all at 37×37):
    ///   For each level (0-3):
    ///     Concat(blocks[2*level+4], blocks[2*level+5]) → [T, 768]
    ///     → head.norm LayerNorm (768)
    ///     → Transpose → [768, H, W]
    ///     → head.projects.{level} Conv1×1 → [projChannels[level], H, W]
    ///     → head.scratch.layer{level+1}_rn Conv3×3 → [64, H, W]
    ///   RefineNet4 → RefineNet3 → RefineNet2 → RefineNet1 (bottom-up)
    ///   → output_conv1/2 → [2, H, W] (depth + confidence)
    ///
    /// Returns: [2, 37, 37] depth map on GPU. Channel 0 = depth, Channel 1 = confidence.
    /// </summary>
    public MemoryBuffer1D<float, Stride1D.Dense> Forward(
        ArrayView1D<float, Stride1D.Dense>[] blockOutputs, // [12] block outputs, each [T*C]
        WeightLoader weights)
    {
        int H = GRID, W = GRID;

        // Block pair starts: level 0=blocks[4,5], 1=[6,7], 2=[8,9], 3=[10,11]
        int[] pairStarts = { 4, 6, 8, 10 };
        // After projects: [48, 96, 192, 384] channels
        int[] projChannels = { 48, 96, 192, 384 };

        // Scratch buffers — sized for 768-channel concat
        var concatBuf = _accelerator.Allocate1D<float>(T * C_CONCAT);      // [T, 768]
        var normBuf   = _accelerator.Allocate1D<float>(T * C_CONCAT);      // [T, 768]
        var transposeBuf = _accelerator.Allocate1D<float>(C_CONCAT * H * W); // [768, H, W]
        var projScratch  = _accelerator.Allocate1D<float>(384 * H * W);    // max projChannels*H*W

        // Shared head.norm weights (applied after concat at each level)
        var normGamma = weights.GetView("head.norm.weight"); // [768]
        var normBeta  = weights.GetView("head.norm.bias");   // [768]
        // layer_rn has no bias — allocate 64 zeros (valid WebGPU buffer; 0-size buffers are invalid)
        using var zeroBias64 = _accelerator.Allocate1D<float>(64);

        var layerRnOutputs = new MemoryBuffer1D<float, Stride1D.Dense>[4];

        try
        {
            // Step 1-3: Per-level: concat → norm → transpose → project → layer_rn
            for (int level = 0; level < 4; level++)
            {
                int bA = pairStarts[level];
                int bB = pairStarts[level] + 1;

                // Concat [T, 384] + [T, 384] → [T, 768]
                _elementWise.ConcatLastDim(blockOutputs[bA], blockOutputs[bB],
                    concatBuf.View, T, C);

                // LayerNorm over 768 channels (head.norm is shared across all levels)
                _layerNorm.Forward(concatBuf.View, normBuf.View, normGamma, normBeta,
                    rows: T, C: C_CONCAT);

                // Transpose [T, 768] → [768, H, W]
                _elementWise.TransposeLastTwo(normBuf.View, transposeBuf.View, 1, T, C_CONCAT);

                // Conv1×1 [768 → projChannels[level]]
                int projC = projChannels[level];
                var projW = weights.GetView($"head.projects.{level}.weight");
                var projB = weights.GetView($"head.projects.{level}.bias");
                // projScratch holds at most 384*H*W which covers all levels
                _conv2d.Forward(transposeBuf.View, projW, projB, projScratch.View,
                    C_CONCAT, H, W, projC, 1, 1);

                // Conv3×3 layer_rn [projC → 64] (no bias — zeroBias64 provides 64 valid zeros)
                var lrnW = weights.GetView($"head.scratch.layer{level + 1}_rn.weight");
                layerRnOutputs[level] = _accelerator.Allocate1D<float>(64 * H * W);
                _conv2d.Forward(projScratch.View, lrnW, zeroBias64.View,
                    layerRnOutputs[level].View, projC, H, W, 64, 3, 3, stride: 1, padding: 1);
            }

            // Step 4: RefineNet stages (bottom-up, MVP: all at 37×37, no upsampling)
            const int refineC = 64;
            int refineElements = refineC * H * W;
            var refineBuf  = _accelerator.Allocate1D<float>(refineElements);
            var refineTemp = _accelerator.Allocate1D<float>(refineElements);

            // RefineNet4: level 3 → ResConfUnit2 → out_conv
            ResConfUnit(layerRnOutputs[3].View, refineBuf.View, refineTemp.View,
                weights.GetView("head.scratch.refinenet4.resConfUnit2.conv1.weight"),
                weights.GetView("head.scratch.refinenet4.resConfUnit2.conv1.bias"),
                weights.GetView("head.scratch.refinenet4.resConfUnit2.conv2.weight"),
                weights.GetView("head.scratch.refinenet4.resConfUnit2.conv2.bias"),
                refineC, H, W);
            _conv2d.Forward(refineBuf.View,
                weights.GetView("head.scratch.refinenet4.out_conv.weight"),
                weights.GetView("head.scratch.refinenet4.out_conv.bias"),
                refineTemp.View, refineC, H, W, refineC, 1, 1);

            // RefineNets 3→1: fuse with previous output + two ResConfUnits + out_conv
            for (int level = 2; level >= 0; level--)
            {
                int refIdx = level + 1; // refinenet{3,2,1}

                // Add previous refinement output (refineTemp) into current level features
                _elementWise.AddInPlace(layerRnOutputs[level].View, refineTemp.View, refineElements);

                // ResConfUnit1
                ResConfUnit(layerRnOutputs[level].View, refineBuf.View, refineTemp.View,
                    weights.GetView($"head.scratch.refinenet{refIdx}.resConfUnit1.conv1.weight"),
                    weights.GetView($"head.scratch.refinenet{refIdx}.resConfUnit1.conv1.bias"),
                    weights.GetView($"head.scratch.refinenet{refIdx}.resConfUnit1.conv2.weight"),
                    weights.GetView($"head.scratch.refinenet{refIdx}.resConfUnit1.conv2.bias"),
                    refineC, H, W);

                // ResConfUnit2
                ResConfUnit(refineBuf.View, layerRnOutputs[level].View, refineTemp.View,
                    weights.GetView($"head.scratch.refinenet{refIdx}.resConfUnit2.conv1.weight"),
                    weights.GetView($"head.scratch.refinenet{refIdx}.resConfUnit2.conv1.bias"),
                    weights.GetView($"head.scratch.refinenet{refIdx}.resConfUnit2.conv2.weight"),
                    weights.GetView($"head.scratch.refinenet{refIdx}.resConfUnit2.conv2.bias"),
                    refineC, H, W);

                // out_conv 1×1
                _conv2d.Forward(layerRnOutputs[level].View,
                    weights.GetView($"head.scratch.refinenet{refIdx}.out_conv.weight"),
                    weights.GetView($"head.scratch.refinenet{refIdx}.out_conv.bias"),
                    refineTemp.View, refineC, H, W, refineC, 1, 1);
            }

            // Step 5: Output convolutions
            // output_conv1: [64→32, 3×3] + ReLU
            var oc1Out = _accelerator.Allocate1D<float>(32 * H * W);
            _conv2d.Forward(refineTemp.View,
                weights.GetView("head.scratch.output_conv1.weight"),
                weights.GetView("head.scratch.output_conv1.bias"),
                oc1Out.View, 64, H, W, 32, 3, 3, stride: 1, padding: 1);
            _elementWise.ReLUInPlace(oc1Out.View, 32 * H * W);

            // output_conv2.0: [32→32, 3×3] + ReLU
            var oc2Out = _accelerator.Allocate1D<float>(32 * H * W);
            _conv2d.Forward(oc1Out.View,
                weights.GetView("head.scratch.output_conv2.0.weight"),
                weights.GetView("head.scratch.output_conv2.0.bias"),
                oc2Out.View, 32, H, W, 32, 3, 3, stride: 1, padding: 1);
            _elementWise.ReLUInPlace(oc2Out.View, 32 * H * W);

            // output_conv2.2: [32→2, 1×1] (depth + confidence)
            var depthOut = _accelerator.Allocate1D<float>(2 * H * W);
            _conv2d.Forward(oc2Out.View,
                weights.GetView("head.scratch.output_conv2.2.weight"),
                weights.GetView("head.scratch.output_conv2.2.bias"),
                depthOut.View, 32, H, W, 2, 1, 1);

            refineBuf.Dispose();
            refineTemp.Dispose();
            oc1Out.Dispose();
            oc2Out.Dispose();
            for (int i = 0; i < 4; i++) layerRnOutputs[i].Dispose();

            // [2, 37, 37] — ch0=depth, ch1=confidence
            return depthOut;
        }
        finally
        {
            concatBuf.Dispose();
            normBuf.Dispose();
            transposeBuf.Dispose();
            projScratch.Dispose();
        }
    }
}
