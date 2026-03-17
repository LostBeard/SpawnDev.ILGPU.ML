using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.WebGPU;

namespace SpawnDev.ILGPU.ML;

/// <summary>
/// DPT (Dense Prediction Transformer) head for DAv3 — multi-scale version.
///
/// Weight architecture (from weight manifest):
///   resize_layers.0: ConvTranspose [48,48,4,4]  stride=4 → 37→148
///   resize_layers.1: ConvTranspose [96,96,2,2]  stride=2 → 37→74
///   resize_layers.2: Identity (no weights)      → 37
///   resize_layers.3: Conv2D [384,384,3,3] s=2   → 37→19
///
/// RefineNet fusion (bottom-up): 19→37→74→148
///   Between each stage: bilinear 2× upsample of the previous output before add.
///
/// output_conv2.1 = interpolate 2×: 148→296
/// Final output: [2, 296, 296]
///
/// All intermediate buffers pre-allocated once in Initialize() — never allocate/dispose
/// inside Forward (ILGPU batches GPU commands until SynchronizeAsync).
/// </summary>
public class DptHead : IDisposable
{
    private const int C = 384;
    private const int C_CONCAT = 768;
    private const int T = 37 * 37;
    private const int GRID = 37;

    // Spatial sizes per level after resize_layers
    private const int H3 = 19;   // level 3: Conv2D stride=2 → 37→19
    private const int H2 = 37;   // level 2: identity
    private const int H1 = 74;   // level 1: ConvTranspose stride=2 → 37→74
    private const int H0 = 148;  // level 0: ConvTranspose stride=4 → 37→148

    private readonly WebGPUAccelerator _accelerator;
    private readonly Conv2DKernel _conv2d;
    private readonly ConvTranspose2DKernel _convTranspose;
    private readonly ElementWiseKernels _elementWise;
    private readonly LayerNormKernel _layerNorm;

    // Shared per-level processing buffers (reused across levels, sized to max)
    private MemoryBuffer1D<float, Stride1D.Dense>? _concatBuf;    // [T, 768]
    private MemoryBuffer1D<float, Stride1D.Dense>? _normBuf;      // [T, 768]
    private MemoryBuffer1D<float, Stride1D.Dense>? _transposeBuf; // [768, GRID, GRID]
    private MemoryBuffer1D<float, Stride1D.Dense>? _projScratch;  // [384, GRID, GRID] max proj

    // Per-level layer_rn outputs at their native spatial size
    // _layerRnOutputs[0]: [64, H0, H0] = [64,148,148]
    // _layerRnOutputs[1]: [64, H1, H1] = [64,74,74]
    // _layerRnOutputs[2]: [64, H2, H2] = [64,37,37]
    // _layerRnOutputs[3]: [64, H3, H3] = [64,19,19]
    private MemoryBuffer1D<float, Stride1D.Dense>[]? _layerRnOutputs;

    // Per-level resize outputs (after resize_layers), before layer_rn
    // Sized to match each level's spatial resolution
    private MemoryBuffer1D<float, Stride1D.Dense>[]? _resizeOutputs;

    private MemoryBuffer1D<float, Stride1D.Dense>? _zeroBias64;

    // RefineNet working buffers — sized to current level (max = H0)
    private MemoryBuffer1D<float, Stride1D.Dense>? _refineBuf;   // [64, H0, H0]
    private MemoryBuffer1D<float, Stride1D.Dense>? _refineTemp;  // [64, H0, H0]
    private MemoryBuffer1D<float, Stride1D.Dense>? _upsampleBuf; // [64, H_next, H_next]

    // Output conv buffers
    private MemoryBuffer1D<float, Stride1D.Dense>? _oc1Out;  // [32, H0, H0]
    private MemoryBuffer1D<float, Stride1D.Dense>? _oc2Out;  // [32, H0, H0]
    private MemoryBuffer1D<float, Stride1D.Dense>? _oc21Out; // [32, 296, 296]  after 2× bilinear
    private MemoryBuffer1D<float, Stride1D.Dense>? _depthOut; // [2, 296, 296]

    // Output spatial size
    public const int OutputH = 296;
    public const int OutputW = 296;

    public DptHead(WebGPUAccelerator accelerator, Conv2DKernel conv2d,
        ConvTranspose2DKernel convTranspose,
        ElementWiseKernels elementWise, LayerNormKernel layerNorm)
    {
        _accelerator = accelerator;
        _conv2d = conv2d;
        _convTranspose = convTranspose;
        _elementWise = elementWise;
        _layerNorm = layerNorm;
    }

    /// <summary>Allocate all intermediate GPU buffers once. Call before Forward.</summary>
    public void Initialize()
    {
        _concatBuf    = _accelerator.Allocate1D<float>(T * C_CONCAT);
        _normBuf      = _accelerator.Allocate1D<float>(T * C_CONCAT);
        _transposeBuf = _accelerator.Allocate1D<float>(C_CONCAT * GRID * GRID);
        _projScratch  = _accelerator.Allocate1D<float>(384 * GRID * GRID); // max projC * GRID*GRID

        _zeroBias64   = _accelerator.Allocate1D<float>(64); // zero-initialized

        // resize outputs per level (proj output before layer_rn, at native spatial size)
        int[] projC = { 48, 96, 192, 384 };
        int[] spatialH = { H0, H1, H2, H3 };

        _resizeOutputs = new MemoryBuffer1D<float, Stride1D.Dense>[4];
        _layerRnOutputs = new MemoryBuffer1D<float, Stride1D.Dense>[4];
        for (int i = 0; i < 4; i++)
        {
            int h = spatialH[i];
            _resizeOutputs[i] = _accelerator.Allocate1D<float>(projC[i] * h * h);
            _layerRnOutputs[i] = _accelerator.Allocate1D<float>(64 * h * h);
        }

        // RefineNet working buffers sized to H0 (largest level)
        _refineBuf   = _accelerator.Allocate1D<float>(64 * H0 * H0);
        _refineTemp  = _accelerator.Allocate1D<float>(64 * H0 * H0);
        _upsampleBuf = _accelerator.Allocate1D<float>(64 * H0 * H0); // upsample scratch

        // Output conv buffers at H0=148
        _oc1Out  = _accelerator.Allocate1D<float>(32 * H0 * H0);
        _oc2Out  = _accelerator.Allocate1D<float>(32 * H0 * H0);

        // output_conv2.1 bilinear 2×: 148→296
        _oc21Out  = _accelerator.Allocate1D<float>(32 * OutputH * OutputW);
        _depthOut = _accelerator.Allocate1D<float>(2 * OutputH * OutputW);
    }

    /// <summary>
    /// Run a ResConfUnit: input → ReLU → Conv3×3 → ReLU → Conv3×3 → + input.
    /// output and temp must be separate buffers (and from input).
    /// </summary>
    private void ResConfUnit(
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<float, Stride1D.Dense> temp,
        ArrayView1D<float, Stride1D.Dense> conv1Weight, ArrayView1D<float, Stride1D.Dense> conv1Bias,
        ArrayView1D<float, Stride1D.Dense> conv2Weight, ArrayView1D<float, Stride1D.Dense> conv2Bias,
        int channels, int H, int W)
    {
        int n = channels * H * W;
        _elementWise.ReLU(input, temp, n);
        _conv2d.Forward(temp, conv1Weight, conv1Bias, output, channels, H, W, channels, 3, 3, 1, 1);
        _elementWise.ReLU(output, temp, n);
        _conv2d.Forward(temp, conv2Weight, conv2Bias, output, channels, H, W, channels, 3, 3, 1, 1);
        _elementWise.AddInPlace(output, input, n);
    }

    /// <summary>
    /// Run the multi-scale DPT head.
    /// Returns persistent [2, 296, 296] depth output buffer.
    /// Valid until next Forward call or Dispose.
    /// </summary>
    public MemoryBuffer1D<float, Stride1D.Dense> Forward(
        ArrayView1D<float, Stride1D.Dense>[] blockOutputs,
        WeightLoader weights)
    {
        if (_concatBuf == null) throw new InvalidOperationException("Call Initialize() first.");

        int[] pairStarts = { 4, 6, 8, 10 };
        int[] projC      = { 48, 96, 192, 384 };
        int[] spatialH   = { H0, H1, H2, H3 };

        var normGamma = weights.GetView("head.norm.weight");
        var normBeta  = weights.GetView("head.norm.bias");

        // ── Step 1: Per-level encode → resize → layer_rn ──────────────────────
        for (int level = 0; level < 4; level++)
        {
            int bA = pairStarts[level];
            int bB = bA + 1;
            int pc = projC[level];
            int h  = spatialH[level];

            // Concat [T, 384] + [T, 384] → [T, 768]
            _elementWise.ConcatLastDim(blockOutputs[bA], blockOutputs[bB],
                _concatBuf!.View, T, C);

            // LayerNorm [T, 768]
            _layerNorm.Forward(_concatBuf.View, _normBuf!.View, normGamma, normBeta,
                rows: T, C: C_CONCAT);

            // Transpose [T, 768] → [768, GRID, GRID]
            _elementWise.TransposeLastTwo(_normBuf.View, _transposeBuf!.View, 1, T, C_CONCAT);

            // Conv1×1 [768 → projC] — output still at GRID×GRID in _projScratch
            var projW = weights.GetView($"head.projects.{level}.weight");
            var projB = weights.GetView($"head.projects.{level}.bias");
            _conv2d.Forward(_transposeBuf.View, projW, projB, _projScratch!.View,
                C_CONCAT, GRID, GRID, pc, 1, 1);

            // Resize to target spatial resolution
            if (level == 0)
            {
                // ConvTranspose [48,48,4,4] stride=4 → 37→148
                var rtW = weights.GetView("head.resize_layers.0.weight");
                var rtB = weights.GetView("head.resize_layers.0.bias");
                _convTranspose.Forward(_projScratch.View, rtW, rtB, _resizeOutputs![0].View,
                    pc, GRID, GRID, pc, 4, 4, stride: 4, padding: 0);
            }
            else if (level == 1)
            {
                // ConvTranspose [96,96,2,2] stride=2 → 37→74
                var rtW = weights.GetView("head.resize_layers.1.weight");
                var rtB = weights.GetView("head.resize_layers.1.bias");
                _convTranspose.Forward(_projScratch.View, rtW, rtB, _resizeOutputs![1].View,
                    pc, GRID, GRID, pc, 2, 2, stride: 2, padding: 0);
            }
            else if (level == 2)
            {
                // Identity — just copy projScratch → resizeOutputs[2]
                // Use AddInPlace with zeroed dest: scale to 0, then add
                _elementWise.ScaleInPlace(_resizeOutputs![2].View, pc * GRID * GRID, 0f);
                _elementWise.AddInPlace(_resizeOutputs[2].View, _projScratch.View, pc * GRID * GRID);
            }
            else // level == 3
            {
                // Conv2D [384,384,3,3] stride=2 pad=1 → 37→19
                var rtW = weights.GetView("head.resize_layers.3.weight");
                var rtB = weights.GetView("head.resize_layers.3.bias");
                _conv2d.Forward(_projScratch.View, rtW, rtB, _resizeOutputs![3].View,
                    pc, GRID, GRID, pc, 3, 3, stride: 2, padding: 1);
            }

            // layer_rn: Conv3×3 [projC → 64] no bias
            var lrnW = weights.GetView($"head.scratch.layer{level + 1}_rn.weight");
            _conv2d.Forward(_resizeOutputs![level].View, lrnW, _zeroBias64!.View,
                _layerRnOutputs![level].View, pc, h, h, 64, 3, 3, stride: 1, padding: 1);
        }

        // ── Step 2: RefineNet fusion (bottom-up: level3→2→1→0) ────────────────
        // Each stage: fuse(skip+upsamp_prev) → RCU1 → RCU2 → out_conv
        const int refineC = 64;

        // RefineNet4: level 3 (H3=19) — no previous, just ResConfUnit2 + out_conv
        ResConfUnit(_layerRnOutputs![3].View, _refineBuf!.View, _refineTemp!.View,
            weights.GetView("head.scratch.refinenet4.resConfUnit2.conv1.weight"),
            weights.GetView("head.scratch.refinenet4.resConfUnit2.conv1.bias"),
            weights.GetView("head.scratch.refinenet4.resConfUnit2.conv2.weight"),
            weights.GetView("head.scratch.refinenet4.resConfUnit2.conv2.bias"),
            refineC, H3, H3);
        _conv2d.Forward(_refineBuf.View,
            weights.GetView("head.scratch.refinenet4.out_conv.weight"),
            weights.GetView("head.scratch.refinenet4.out_conv.bias"),
            _refineTemp.View, refineC, H3, H3, refineC, 1, 1);
        // _refineTemp now holds refinenet4 output [64, 19, 19]

        // RefineNet3: level 2 (H2=37). Upsample 19→37 then fuse.
        _elementWise.BilinearUpsample(_refineTemp.View, _upsampleBuf!.View,
            refineC, H3, H3, H2, H2);
        _elementWise.AddInPlace(_layerRnOutputs[2].View, _upsampleBuf.View, refineC * H2 * H2);

        ResConfUnit(_layerRnOutputs[2].View, _refineBuf.View, _refineTemp.View,
            weights.GetView("head.scratch.refinenet3.resConfUnit1.conv1.weight"),
            weights.GetView("head.scratch.refinenet3.resConfUnit1.conv1.bias"),
            weights.GetView("head.scratch.refinenet3.resConfUnit1.conv2.weight"),
            weights.GetView("head.scratch.refinenet3.resConfUnit1.conv2.bias"),
            refineC, H2, H2);
        ResConfUnit(_refineBuf.View, _layerRnOutputs[2].View, _refineTemp.View,
            weights.GetView("head.scratch.refinenet3.resConfUnit2.conv1.weight"),
            weights.GetView("head.scratch.refinenet3.resConfUnit2.conv1.bias"),
            weights.GetView("head.scratch.refinenet3.resConfUnit2.conv2.weight"),
            weights.GetView("head.scratch.refinenet3.resConfUnit2.conv2.bias"),
            refineC, H2, H2);
        _conv2d.Forward(_layerRnOutputs[2].View,
            weights.GetView("head.scratch.refinenet3.out_conv.weight"),
            weights.GetView("head.scratch.refinenet3.out_conv.bias"),
            _refineTemp.View, refineC, H2, H2, refineC, 1, 1);
        // _refineTemp now holds refinenet3 output [64, 37, 37]

        // RefineNet2: level 1 (H1=74). Upsample 37→74 then fuse.
        _elementWise.BilinearUpsample(_refineTemp.View, _upsampleBuf.View,
            refineC, H2, H2, H1, H1);
        _elementWise.AddInPlace(_layerRnOutputs[1].View, _upsampleBuf.View, refineC * H1 * H1);

        ResConfUnit(_layerRnOutputs[1].View, _refineBuf.View, _refineTemp.View,
            weights.GetView("head.scratch.refinenet2.resConfUnit1.conv1.weight"),
            weights.GetView("head.scratch.refinenet2.resConfUnit1.conv1.bias"),
            weights.GetView("head.scratch.refinenet2.resConfUnit1.conv2.weight"),
            weights.GetView("head.scratch.refinenet2.resConfUnit1.conv2.bias"),
            refineC, H1, H1);
        ResConfUnit(_refineBuf.View, _layerRnOutputs[1].View, _refineTemp.View,
            weights.GetView("head.scratch.refinenet2.resConfUnit2.conv1.weight"),
            weights.GetView("head.scratch.refinenet2.resConfUnit2.conv1.bias"),
            weights.GetView("head.scratch.refinenet2.resConfUnit2.conv2.weight"),
            weights.GetView("head.scratch.refinenet2.resConfUnit2.conv2.bias"),
            refineC, H1, H1);
        _conv2d.Forward(_layerRnOutputs[1].View,
            weights.GetView("head.scratch.refinenet2.out_conv.weight"),
            weights.GetView("head.scratch.refinenet2.out_conv.bias"),
            _refineTemp.View, refineC, H1, H1, refineC, 1, 1);
        // _refineTemp now holds refinenet2 output [64, 74, 74]

        // RefineNet1: level 0 (H0=148). Upsample 74→148 then fuse.
        _elementWise.BilinearUpsample(_refineTemp.View, _upsampleBuf.View,
            refineC, H1, H1, H0, H0);
        _elementWise.AddInPlace(_layerRnOutputs[0].View, _upsampleBuf.View, refineC * H0 * H0);

        ResConfUnit(_layerRnOutputs[0].View, _refineBuf.View, _refineTemp.View,
            weights.GetView("head.scratch.refinenet1.resConfUnit1.conv1.weight"),
            weights.GetView("head.scratch.refinenet1.resConfUnit1.conv1.bias"),
            weights.GetView("head.scratch.refinenet1.resConfUnit1.conv2.weight"),
            weights.GetView("head.scratch.refinenet1.resConfUnit1.conv2.bias"),
            refineC, H0, H0);
        ResConfUnit(_refineBuf.View, _layerRnOutputs[0].View, _refineTemp.View,
            weights.GetView("head.scratch.refinenet1.resConfUnit2.conv1.weight"),
            weights.GetView("head.scratch.refinenet1.resConfUnit2.conv1.bias"),
            weights.GetView("head.scratch.refinenet1.resConfUnit2.conv2.weight"),
            weights.GetView("head.scratch.refinenet1.resConfUnit2.conv2.bias"),
            refineC, H0, H0);
        _conv2d.Forward(_layerRnOutputs[0].View,
            weights.GetView("head.scratch.refinenet1.out_conv.weight"),
            weights.GetView("head.scratch.refinenet1.out_conv.bias"),
            _refineTemp.View, refineC, H0, H0, refineC, 1, 1);
        // _refineTemp now holds refinenet1 output [64, 148, 148]

        // ── Step 3: Output convolutions ───────────────────────────────────────
        // output_conv1: [64→32, 3×3] + ReLU  (at 148×148)
        _conv2d.Forward(_refineTemp.View,
            weights.GetView("head.scratch.output_conv1.weight"),
            weights.GetView("head.scratch.output_conv1.bias"),
            _oc1Out!.View, refineC, H0, H0, 32, 3, 3, 1, 1);
        _elementWise.ReLUInPlace(_oc1Out.View, 32 * H0 * H0);

        // output_conv2.0: [32→32, 3×3] + ReLU  (at 148×148)
        _conv2d.Forward(_oc1Out.View,
            weights.GetView("head.scratch.output_conv2.0.weight"),
            weights.GetView("head.scratch.output_conv2.0.bias"),
            _oc2Out!.View, 32, H0, H0, 32, 3, 3, 1, 1);
        _elementWise.ReLUInPlace(_oc2Out.View, 32 * H0 * H0);

        // output_conv2.1: bilinear 2×  148→296
        _elementWise.BilinearUpsample(_oc2Out.View, _oc21Out!.View,
            32, H0, H0, OutputH, OutputW);

        // output_conv2.2: [32→2, 1×1]  (at 296×296)
        _conv2d.Forward(_oc21Out.View,
            weights.GetView("head.scratch.output_conv2.2.weight"),
            weights.GetView("head.scratch.output_conv2.2.bias"),
            _depthOut!.View, 32, OutputH, OutputW, 2, 1, 1);

        return _depthOut;
    }

    public void Dispose()
    {
        _concatBuf?.Dispose();
        _normBuf?.Dispose();
        _transposeBuf?.Dispose();
        _projScratch?.Dispose();
        _zeroBias64?.Dispose();
        if (_resizeOutputs != null) foreach (var b in _resizeOutputs) b?.Dispose();
        if (_layerRnOutputs != null) foreach (var b in _layerRnOutputs) b?.Dispose();
        _refineBuf?.Dispose();
        _refineTemp?.Dispose();
        _upsampleBuf?.Dispose();
        _oc1Out?.Dispose();
        _oc2Out?.Dispose();
        _oc21Out?.Dispose();
        _depthOut?.Dispose();
    }
}
