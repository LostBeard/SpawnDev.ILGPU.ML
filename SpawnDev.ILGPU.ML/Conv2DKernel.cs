using ILGPU;
using ILGPU.Runtime;
using System.Numerics;

namespace SpawnDev.ILGPU.ML;

/// <summary>
/// 2D Convolution kernel for neural network inference.
/// Supports arbitrary kernel sizes (1×1, 3×3, 14×14), stride, padding, and dilation.
/// Group=1 (standard) and group=inC (depthwise via dedicated entry points).
///
/// Layout: NCHW (input [N,C,H,W], weight [outC,inC,kH,kW]) and NHWC (input [N,H,W,C],
/// weight [outC,kH,kW,inC] — TFLite-native).
///
/// Parameters are captured as scalars per the SpawnDev.ILGPU.ML CLAUDE.md guidance
/// (Lambda Kernels). No shared params buffer = no params-buffer race under async
/// dispatch on Wasm.
/// </summary>
public class Conv2DKernel : IDisposable
{
    private readonly Accelerator _accelerator;

    // params: inC, inH, inW, outC, kHW(packed), stride, padTL(packed), outH, outW, dilHW(packed) - see Pack16
    private Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        int, int, int, int, int, int, int, int, int, int>?
        _conv2dKernel;

    // Native low-precision weights: identical to _conv2dKernel but the WEIGHT (2nd view) is a low-p type T
    // (ILGPU.Half / BFloat16 / Float8E*). One compiled kernel per concrete T, cached; lazily loaded on first
    // use of that type. object-typed because each delegate is T-specific.
    private readonly Dictionary<Type, object> _conv2dLowPWeightKernels = new();

    // params: C, inH, inW, kHW(packed), stride, padTL(packed), outH, outW, dilHW(packed) - see Pack16
    private Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        int, int, int, int, int, int, int, int, int>?
        _depthwiseKernel;

    private Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        int, int, int, int, int, int, int, int, int, int>?
        _conv2dNHWCKernel;

    private Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        int, int, int, int, int, int, int, int, int>?
        _depthwiseNHWCKernel;

    public Conv2DKernel(Accelerator accelerator) => _accelerator = accelerator;

    private static long _convCallCount;

    /// <summary>
    /// Two non-negative values in one kernel int (hi in bits 16-30, lo in bits 0-15). The kernels take their scalars
    /// this way to stay within ILGPU's typed-loader argument limit. Only SMALL quantities are packed - kernel size,
    /// begin pads, dilations; the output dims each get a full int. They used to share one (outH &lt;&lt; 16 | outW): at
    /// outH = 32768 the sign bit flipped, the kernel decoded a negative height and indexed out of bounds (RaCo-ALIKED's
    /// 1x1 descriptor conv over [2,128,32768,1], 2026-10-01: an ILGPU CPU "X index out of bounds" abort, silent
    /// garbage on the GPU backends). A value that does not fit throws here instead of wrapping.
    /// </summary>
    private static int Pack16(int hi, int lo, string what)
    {
        if ((uint)hi > 0x7FFF || (uint)lo > 0xFFFF)
            throw new NotSupportedException($"Conv2D {what} ({hi}, {lo}) is outside the packed kernel-parameter range (0..32767, 0..65535)");
        return (hi << 16) | lo;
    }

    // ── Implicit-GEMM tiled conv (NCHW) ──
    // The naive one-thread-per-output kernel has ZERO data reuse: every MAC does two global loads, and a
    // 3x3 conv re-reads each input pixel 9x from DRAM - measured 420-960 GFLOPS on the 4070 vs the
    // register-blocked GEMM's 4.3-5.7 TFLOPS at DAv3 shapes. Convolution IS a GEMM:
    //   C[outC, outH*outW] = W[outC, inC*kH*kW] x im2col(input)[inC*kH*kW, outH*outW]
    // and the weight is ALREADY row-major [outC, K] - so this kernel reuses RegisterBlockedMatMul's exact
    // 64x64-tile / 4x4-register structure, with the B-tile stage doing the im2col ADDRESSING on the fly
    // (no materialized im2col buffer, no extra memory). Shared-memory staging supplies the data reuse the
    // naive kernel forfeits: 64 outC rows share each input patch, 64 output pixels share each weight row.
    // Gated to backends with a 256-thread group + shared memory (WebGL/CPU keep the naive kernel).
    private const int RbBlock = 16;              // 16x16 = 256 threads (WebGPU max workgroup)
    private const int RbReg = 4;                 // 4x4 outputs per thread
    private const int RbTile = RbBlock * RbReg;  // 64x64 output tile

    private Action<KernelConfig,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        int, int, int, int, int, int, int, int, int, int>?
        _implicitGemmKernel;

    private static void Conv2DImplicitGemmImpl(
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> weight,
        ArrayView1D<float, Stride1D.Dense> bias,
        ArrayView1D<float, Stride1D.Dense> output,
        int inC, int inH, int inW, int outC, int kHW,
        int strideDilHW, int padTL, int outH, int outW, int numTilesN)
    {
        // strideDilHW = (strideH << 24) | (strideW << 16) | (dilationH << 8) | dilationW - LoadStreamKernel's typed
        // loaders cap at 14 params, so the small scalars share ints (validated on the host - see Pack16).
        int strideH = strideDilHW >> 24, strideW = (strideDilHW >> 16) & 0xFF;
        int kH = kHW >> 16, kW = kHW & 0xFFFF;
        int padTop = padTL >> 16, padLeft = padTL & 0xFFFF;
        int dilationH = (strideDilHW >> 8) & 0xFF, dilationW = strideDilHW & 0xFF;

        int K = inC * kH * kW;
        int N = outH * outW;

        // Batch from Grid.IdxY (DAv3 multi-view); batch=1 dispatches Y=1 so both bases are 0.
        int batch = Grid.IdxY;
        int inBatchBase = batch * inC * inH * inW;
        int outBatchBase = batch * outC * N;

        var aTile = SharedMemory.Allocate<float>(RbTile * RbBlock); // weights: 64 outC x 16 k
        var bTile = SharedMemory.Allocate<float>(RbBlock * RbTile); // patches: 16 k x 64 n

        int tileIdx = Grid.IdxX;
        int tileRow = tileIdx / numTilesN;  // outC tile
        int tileCol = tileIdx % numTilesN;  // output-pixel tile
        int localIdx = Group.IdxX;
        int threadRow = localIdx / RbBlock;
        int threadCol = localIdx % RbBlock;

        float c00 = 0, c01 = 0, c02 = 0, c03 = 0;
        float c10 = 0, c11 = 0, c12 = 0, c13 = 0;
        float c20 = 0, c21 = 0, c22 = 0, c23 = 0;
        float c30 = 0, c31 = 0, c32 = 0, c33 = 0;

        int numKTiles = (K + RbBlock - 1) / RbBlock;
        for (int t = 0; t < numKTiles; t++)
        {
            // A tile: plain weight loads (row-major [outC, K] - coalesced along k).
            for (int r = 0; r < RbReg; r++)
            {
                int oc = tileRow * RbTile + threadRow * RbReg + r;
                int kk = t * RbBlock + threadCol;
                int sIdx = (threadRow * RbReg + r) * RbBlock + threadCol;
                aTile[sIdx] = (oc < outC && kk < K) ? weight[oc * K + kk] : 0f;
            }
            // B tile: implicit im2col - decode (k -> ic,ky,kx) and (n -> oy,ox), bounds-checked padded read.
            for (int r = 0; r < RbReg; r++)
            {
                int kk = t * RbBlock + threadRow;
                int n = tileCol * RbTile + threadCol * RbReg + r;
                int sIdx = threadRow * RbTile + threadCol * RbReg + r;
                int ic = kk / (kH * kW);
                int rem = kk - ic * (kH * kW);
                int ky = rem / kW;
                int kx = rem - ky * kW;
                int oy = n / outW;
                int ox = n - oy * outW;
                int iy = oy * strideH + ky * dilationH - padTop;
                int ix = ox * strideW + kx * dilationW - padLeft;
                bTile[sIdx] = (kk < K && n < N && iy >= 0 && iy < inH && ix >= 0 && ix < inW)
                    ? input[inBatchBase + (ic * inH + iy) * inW + ix] : 0f;
            }

            Group.Barrier();

            for (int k = 0; k < RbBlock; k++)
            {
                float a0 = aTile[(threadRow * RbReg + 0) * RbBlock + k];
                float a1 = aTile[(threadRow * RbReg + 1) * RbBlock + k];
                float a2 = aTile[(threadRow * RbReg + 2) * RbBlock + k];
                float a3 = aTile[(threadRow * RbReg + 3) * RbBlock + k];
                float b0 = bTile[k * RbTile + threadCol * RbReg + 0];
                float b1 = bTile[k * RbTile + threadCol * RbReg + 1];
                float b2 = bTile[k * RbTile + threadCol * RbReg + 2];
                float b3 = bTile[k * RbTile + threadCol * RbReg + 3];
                c00 += a0 * b0; c01 += a0 * b1; c02 += a0 * b2; c03 += a0 * b3;
                c10 += a1 * b0; c11 += a1 * b1; c12 += a1 * b2; c13 += a1 * b3;
                c20 += a2 * b0; c21 += a2 * b1; c22 += a2 * b2; c23 += a2 * b3;
                c30 += a3 * b0; c31 += a3 * b1; c32 += a3 * b2; c33 += a3 * b3;
            }

            Group.Barrier();
        }

        // Epilogue: C[oc, n] = acc + bias[oc] (bias always read, matching the naive kernel's contract).
        int baseOc = tileRow * RbTile + threadRow * RbReg;
        int baseN = tileCol * RbTile + threadCol * RbReg;
        if (baseOc + 0 < outC)
        {
            float bb = bias[baseOc + 0];
            if (baseN + 0 < N) output[outBatchBase + (baseOc + 0) * N + baseN + 0] = c00 + bb;
            if (baseN + 1 < N) output[outBatchBase + (baseOc + 0) * N + baseN + 1] = c01 + bb;
            if (baseN + 2 < N) output[outBatchBase + (baseOc + 0) * N + baseN + 2] = c02 + bb;
            if (baseN + 3 < N) output[outBatchBase + (baseOc + 0) * N + baseN + 3] = c03 + bb;
        }
        if (baseOc + 1 < outC)
        {
            float bb = bias[baseOc + 1];
            if (baseN + 0 < N) output[outBatchBase + (baseOc + 1) * N + baseN + 0] = c10 + bb;
            if (baseN + 1 < N) output[outBatchBase + (baseOc + 1) * N + baseN + 1] = c11 + bb;
            if (baseN + 2 < N) output[outBatchBase + (baseOc + 1) * N + baseN + 2] = c12 + bb;
            if (baseN + 3 < N) output[outBatchBase + (baseOc + 1) * N + baseN + 3] = c13 + bb;
        }
        if (baseOc + 2 < outC)
        {
            float bb = bias[baseOc + 2];
            if (baseN + 0 < N) output[outBatchBase + (baseOc + 2) * N + baseN + 0] = c20 + bb;
            if (baseN + 1 < N) output[outBatchBase + (baseOc + 2) * N + baseN + 1] = c21 + bb;
            if (baseN + 2 < N) output[outBatchBase + (baseOc + 2) * N + baseN + 2] = c22 + bb;
            if (baseN + 3 < N) output[outBatchBase + (baseOc + 2) * N + baseN + 3] = c23 + bb;
        }
        if (baseOc + 3 < outC)
        {
            float bb = bias[baseOc + 3];
            if (baseN + 0 < N) output[outBatchBase + (baseOc + 3) * N + baseN + 0] = c30 + bb;
            if (baseN + 1 < N) output[outBatchBase + (baseOc + 3) * N + baseN + 1] = c31 + bb;
            if (baseN + 2 < N) output[outBatchBase + (baseOc + 3) * N + baseN + 2] = c32 + bb;
            if (baseN + 3 < N) output[outBatchBase + (baseOc + 3) * N + baseN + 3] = c33 + bb;
        }
    }

    /// <summary>
    /// Conv2D NCHW: one thread per output element. inC, inH, inW, outC, kH, kW,
    /// stride, padding, dilationH, dilationW are captured as scalar parameters.
    /// </summary>
    private static void Conv2DImpl(
        Index1D idx,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> weight,
        ArrayView1D<float, Stride1D.Dense> bias,
        ArrayView1D<float, Stride1D.Dense> output,
        int inC, int inH, int inW, int outC, int kHW,
        int strideHW, int padTL, int outH, int outW, int dilHW)
    {
        // Kernel size, BEGIN pads and dilations are passed PACKED (Pack16) to stay within ILGPU's 15-arg kernel
        // limit; outH/outW are passed whole. Recomputing dims here from a single symmetric pad silently truncated
        // stride-2 SAME convs (192->95 instead of 96), shearing every downstream feature map.
        int kH = kHW >> 16, kW = kHW & 0xFFFF;
        int padTop = padTL >> 16, padLeft = padTL & 0xFFFF;
        int dilationH = dilHW >> 16, dilationW = dilHW & 0xFFFF;
        int strideH = strideHW >> 16, strideW = strideHW & 0xFFFF;
        // BATCH-aware decode: idx spans (batch * outC * outH * outW). Decode the batch index and offset the input
        // read by its stride. Single-view models launch batch=1 so b==0 and inBatchBase==0 — byte-identical to
        // the old kernel; only batch>1 (DAv3 multi-view: pixel_values=[1,N,3,H,W] → Conv over N views) changes.
        // Without this the kernel only ever wrote batch 0 and left every later view as uninitialized garbage.
        int perBatchOut = outC * outH * outW;
        int b = idx / perBatchOut;
        int r = idx - b * perBatchOut;
        int ox = r % outW;
        int rem = r / outW;
        int oy = rem % outH;
        int oc = rem / outH;
        int inBatchBase = b * inC * inH * inW;

        // f32 accumulation (the ML-standard for conv): the rounding error over the inC*kH*kW MACs is ~1e-5
        // relative — imperceptible in an 8-bit image and within the MAC-scaled conv-test tolerance. f64
        // accumulation here was over-cautious "ultimate quality" and is far slower on every GPU backend
        // (consumer cards run f64 at ~1/64 of f32; WebGPU/WebGL EMULATE f64 via Dekker — the conv-heavy UNet/VAE
        // paid that on every MAC). Always read bias — no branch (ANGLE optimizer workaround).
        float sum = bias[oc];

        for (int ic = 0; ic < inC; ic++)
        {
            int icBase = inBatchBase + ic * inH * inW;
            int wcBase = oc * inC * kH * kW + ic * kH * kW;
            for (int ky = 0; ky < kH; ky++)
            {
                int iy = oy * strideH + ky * dilationH - padTop;
                if (iy < 0 || iy >= inH) continue;

                for (int kx = 0; kx < kW; kx++)
                {
                    int ix = ox * strideW + kx * dilationW - padLeft;
                    if (ix < 0 || ix >= inW) continue;

                    sum += input[icBase + iy * inW + ix] * weight[wcBase + ky * kW + kx];
                }
            }
        }

        output[idx] = sum;
    }

    /// <summary>
    /// Conv2D NCHW with NATIVE low-precision WEIGHTS (<typeparamref name="T"/> = ILGPU.Half / BFloat16 /
    /// Float8E*) — identical math to Conv2DImpl, but each filter weight is read NATIVELY and converted to
    /// float in-register (PrecisionConvert) for the f32 accumulation. The weight stays native in
    /// GPU memory (no f32 temp buffer); input/bias/output stay fp32, no accuracy loss. The UNet is mostly
    /// Conv, so this is the bulk of the low-p memory win for SD-Turbo.
    /// </summary>
    private static void Conv2DLowPWeightImpl<T>(
        Index1D idx,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<T, Stride1D.Dense> weight,
        ArrayView1D<float, Stride1D.Dense> bias,
        ArrayView1D<float, Stride1D.Dense> output,
        int inC, int inH, int inW, int outC, int kHW,
        int strideHW, int padTL, int outH, int outW, int dilHW)
        where T : unmanaged, INumber<T>
    {
        int kH = kHW >> 16, kW = kHW & 0xFFFF;
        int padTop = padTL >> 16, padLeft = padTL & 0xFFFF;
        int dilationH = dilHW >> 16, dilationW = dilHW & 0xFFFF;
        int strideH = strideHW >> 16, strideW = strideHW & 0xFFFF;
        // BATCH-aware, exactly as Conv2DImpl: this kernel used to compute image 0 only.
        int perBatchOut = outC * outH * outW;
        int b = idx / perBatchOut;
        int r = idx - b * perBatchOut;
        int ox = r % outW;
        int rem = r / outW;
        int oy = rem % outH;
        int oc = rem / outH;
        int inBatchBase = b * inC * inH * inW;

        float sum = bias[oc];

        for (int ic = 0; ic < inC; ic++)
        {
            int icBase = inBatchBase + ic * inH * inW;
            int wcBase = oc * inC * kH * kW + ic * kH * kW;
            for (int ky = 0; ky < kH; ky++)
            {
                int iy = oy * strideH + ky * dilationH - padTop;
                if (iy < 0 || iy >= inH) continue;

                for (int kx = 0; kx < kW; kx++)
                {
                    int ix = ox * strideW + kx * dilationW - padLeft;
                    if (ix < 0 || ix >= inW) continue;

                    sum += input[icBase + iy * inW + ix] * PrecisionConvert.ConvertToSingle(weight[wcBase + ky * kW + kx]);
                }
            }
        }

        output[idx] = sum;
    }

    /// <summary>
    /// Run Conv2D NCHW. Input: [inC, inH, inW]. Output: [outC, outH, outW].
    /// Weight: [outC, inC, kH, kW]. Bias: [outC] or empty.
    /// </summary>
    public void Forward(
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> weight,
        ArrayView1D<float, Stride1D.Dense> bias,
        ArrayView1D<float, Stride1D.Dense> output,
        int inC, int inH, int inW,
        int outC, int kH, int kW,
        int stride = 1, int padding = 0,
        int dilationH = 1, int dilationW = 1)
        => ForwardPadded(input, weight, bias, output, inC, inH, inW, outC, kH, kW,
            stride, padding, padding, padding, padding, dilationH, dilationW);

    /// <summary>
    /// Conv2D NCHW with explicit asymmetric ONNX pads [padTop, padLeft, padBottom, padRight].
    /// Output dims are computed from the FULL (begin+end) pads — never from a single symmetric
    /// value — so stride-2 SAME convs (ONNX pads like [0,0,1,1]) produce the correct grid
    /// instead of a 1-short, sheared one.
    /// </summary>
    public void ForwardPadded(
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> weight,
        ArrayView1D<float, Stride1D.Dense> bias,
        ArrayView1D<float, Stride1D.Dense> output,
        int inC, int inH, int inW,
        int outC, int kH, int kW,
        int stride, int padTop, int padLeft, int padBottom, int padRight,
        int dilationH = 1, int dilationW = 1, int batch = 1, int strideW = 0)
    {
        EnsureLoaded();
        if (strideW <= 0) strideW = stride;   // `stride` is the H stride; W defaults to it (square)

        int effKH = dilationH * (kH - 1) + 1;
        int effKW = dilationW * (kW - 1) + 1;
        int outH = (inH + padTop + padBottom - effKH) / stride + 1;
        int outW = (inW + padLeft + padRight - effKW) / strideW + 1;
        if (outH <= 0 || outW <= 0)
            throw new InvalidOperationException(
                $"Conv2D output dimensions are invalid: outH={outH}, outW={outW} " +
                $"(inH={inH}, inW={inW}, kH={kH}, kW={kW}, stride={stride}, pads=[{padTop},{padLeft},{padBottom},{padRight}], dilation={dilationH}x{dilationW}). " +
                $"This usually means SAME padding was not applied correctly.");
        if (batch < 1) batch = 1;
        // Total threads span ALL batches: the kernel decodes the batch index from idx and strides the input.
        // (DAv3 multi-view Conv sees batch = num_images; single-view models pass batch=1 = the old behavior.)
        int totalOutputElements = batch * outC * outH * outW;
        _convCallCount++;
        if (output.Length < totalOutputElements)
            throw new InvalidOperationException(
                $"Conv2D NCHW output buffer too small: output.Length={output.Length} but kernel will write {totalOutputElements} elements " +
                $"(batch={batch} outH={outH} outW={outW} outC={outC}, inC={inC} inH={inH} inW={inW} kH={kH} kW={kW} stride={stride} pads=[{padTop},{padLeft},{padBottom},{padRight}] dilation={dilationH}x{dilationW}). " +
                $"Upstream shape inference allocated wrong size.");

        // Implicit-GEMM tiled route: backends with a 256-thread group + shared memory (not WebGL, not CPU's
        // 16/axis groups), and enough work that a 64x64-tiled launch beats the naive kernel's zero-setup
        // (tiny convs keep the naive path - a 1-tile launch has no reuse to win). K/N thresholds are the
        // tile geometry, not tuning magic.
        int gemmK = inC * kH * kW;
        int gemmN = outH * outW;
        if (_accelerator.MaxNumThreadsPerGroup >= RbBlock * RbBlock
            && _accelerator.AcceleratorType != global::ILGPU.Runtime.AcceleratorType.WebGL
            && gemmK >= RbBlock && gemmN >= RbTile && outC >= RbReg)
        {
            _implicitGemmKernel ??= _accelerator.LoadStreamKernel<
                ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
                int, int, int, int, int, int, int, int, int, int>(Conv2DImplicitGemmImpl);
            int numTilesM = (outC + RbTile - 1) / RbTile;
            int numTilesN = (gemmN + RbTile - 1) / RbTile;
            var cfg = new KernelConfig(
                new Index2D(numTilesM * numTilesN, batch),
                new Index2D(RbBlock * RbBlock, 1));
            if ((uint)dilationH > 0xFF || (uint)dilationW > 0xFF || (uint)stride > 0x7F || (uint)strideW > 0xFF)
                throw new NotSupportedException($"Conv2D stride {stride}x{strideW} / dilation {dilationH}x{dilationW} exceeds the implicit-GEMM packed range (stride H 0..127, W 0..255, dilation 0..255)");
            _implicitGemmKernel(cfg, input, weight, bias, output,
                inC, inH, inW, outC, Pack16(kH, kW, "kernel size"),
                (stride << 24) | (strideW << 16) | (dilationH << 8) | dilationW,
                Pack16(padTop, padLeft, "begin pads"), outH, outW, numTilesN);
            return;
        }

        try
        {
            _conv2dKernel!(totalOutputElements, input, weight, bias, output,
                inC, inH, inW, outC, Pack16(kH, kW, "kernel size"), Pack16(stride, strideW, "stride"), Pack16(padTop, padLeft, "begin pads"),
                outH, outW, Pack16(dilationH, dilationW, "dilation"));
        }
        catch (global::ILGPU.Runtime.OpenCL.CLException clEx)
        {
            throw new InvalidOperationException(
                $"[Conv2DKernel.ForwardPadded call #{_convCallCount} {_accelerator.AcceleratorType}] "
                + $"OpenCL {clEx.Error} (CLError) at "
                + $"input=[{inC},{inH},{inW}] outC={outC} k={kH}x{kW} stride={stride} pads=[{padTop},{padLeft},{padBottom},{padRight}] "
                + $"totalOutput={totalOutputElements}", clEx);
        }
    }

    /// <summary>fp16-weight Conv2D NCHW (asymmetric ONNX pads). T=Half wrapper over
    /// <see cref="ForwardPaddedLowPWeight{T}"/> (callers unchanged).</summary>
    public void ForwardPaddedHalfWeight(
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<global::ILGPU.Half, Stride1D.Dense> weight,
        ArrayView1D<float, Stride1D.Dense> bias,
        ArrayView1D<float, Stride1D.Dense> output,
        int inC, int inH, int inW,
        int outC, int kH, int kW,
        int stride, int padTop, int padLeft, int padBottom, int padRight,
        int dilationH = 1, int dilationW = 1)
        => ForwardPaddedLowPWeight(input, weight, bias, output, inC, inH, inW, outC, kH, kW,
            stride, padTop, padLeft, padBottom, padRight, dilationH, dilationW);

    /// <summary>Conv2D NCHW (asymmetric ONNX pads) with NATIVE low-precision weights (<typeparamref name="T"/>
    /// = ILGPU.Half / BFloat16 / Float8E*). Identical to <see cref="ForwardPadded"/> but the weight stays
    /// native in GPU memory (no f32 temp); each weight is converted to float in-register via PrecisionConvert,
    /// fp32/fp64 accumulate.</summary>
    public void ForwardPaddedLowPWeight<T>(
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<T, Stride1D.Dense> weight,
        ArrayView1D<float, Stride1D.Dense> bias,
        ArrayView1D<float, Stride1D.Dense> output,
        int inC, int inH, int inW,
        int outC, int kH, int kW,
        int stride, int padTop, int padLeft, int padBottom, int padRight,
        int dilationH = 1, int dilationW = 1, int batch = 1, int strideW = 0)
        where T : unmanaged, INumber<T>
    {
        var kernel = GetConv2DLowPWeightKernel<T>();
        if (strideW <= 0) strideW = stride;
        int effKH = dilationH * (kH - 1) + 1;
        int effKW = dilationW * (kW - 1) + 1;
        int outH = (inH + padTop + padBottom - effKH) / stride + 1;
        int outW = (inW + padLeft + padRight - effKW) / strideW + 1;
        if (outH <= 0 || outW <= 0)
            throw new InvalidOperationException(
                $"Conv2D(low-p) output dims invalid: outH={outH}, outW={outW} (inH={inH}, inW={inW}, kH={kH}, kW={kW}, " +
                $"stride={stride}, pads=[{padTop},{padLeft},{padBottom},{padRight}], dilation={dilationH}x{dilationW}).");
        if (batch < 1) batch = 1;
        int totalOutputElements = batch * outC * outH * outW;
        _convCallCount++;
        if (output.Length < totalOutputElements)
            throw new InvalidOperationException(
                $"Conv2D(low-p) NCHW output buffer too small: output.Length={output.Length} < {totalOutputElements} elements.");
        kernel(totalOutputElements, input, weight, bias, output,
            inC, inH, inW, outC, Pack16(kH, kW, "kernel size"), Pack16(stride, strideW, "stride"), Pack16(padTop, padLeft, "begin pads"),
            outH, outW, Pack16(dilationH, dilationW, "dilation"));
    }

    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<T, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        int, int, int, int, int, int, int, int, int, int> GetConv2DLowPWeightKernel<T>()
        where T : unmanaged, INumber<T>
    {
        if (!_conv2dLowPWeightKernels.TryGetValue(typeof(T), out var k))
            _conv2dLowPWeightKernels[typeof(T)] = k = _accelerator.LoadAutoGroupedStreamKernel<Index1D,
                ArrayView1D<float, Stride1D.Dense>, ArrayView1D<T, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
                int, int, int, int, int, int, int, int, int, int>(Conv2DLowPWeightImpl<T>);
        return (Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<T, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            int, int, int, int, int, int, int, int, int, int>)k;
    }

    /// <summary>fp16-weight Conv2D NCHW (symmetric padding). See <see cref="Forward"/>.</summary>
    public void ForwardHalfWeight(
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<global::ILGPU.Half, Stride1D.Dense> weight,
        ArrayView1D<float, Stride1D.Dense> bias,
        ArrayView1D<float, Stride1D.Dense> output,
        int inC, int inH, int inW,
        int outC, int kH, int kW,
        int stride = 1, int padding = 0,
        int dilationH = 1, int dilationW = 1)
        => ForwardPaddedHalfWeight(input, weight, bias, output, inC, inH, inW, outC, kH, kW,
            stride, padding, padding, padding, padding, dilationH, dilationW);

    /// <summary>
    /// Depthwise Conv2D NCHW: each input channel convolved independently.
    /// Weight: [C, 1, kH, kW]. Bias: [C].
    /// </summary>
    private static void DepthwiseConv2DImpl(
        Index1D idx,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> weight,
        ArrayView1D<float, Stride1D.Dense> bias,
        ArrayView1D<float, Stride1D.Dense> output,
        int C, int inH, int inW, int kHW,
        int strideHW, int padTL, int outH, int outW, int dilHW)
    {
        // Kernel size + begin pads + dilations packed (Pack16). BATCH-aware: idx spans batch*C*outH*outW (this
        // kernel used to compute image 0 only).
        int kH = kHW >> 16, kW = kHW & 0xFFFF;
        int padTop = padTL >> 16, padLeft = padTL & 0xFFFF;
        int dilationH = dilHW >> 16, dilationW = dilHW & 0xFFFF;
        int strideH = strideHW >> 16, strideW = strideHW & 0xFFFF;
        int ox = idx % outW;
        int rem = idx / outW;
        int oy = rem % outH;
        int bc = rem / outH;          // b * C + c
        int c = bc % C;

        float sum = bias[c];

        int inBase = bc * inH * inW;
        int wBase = c * kH * kW;
        for (int ky = 0; ky < kH; ky++)
        {
            int iy = oy * strideH + ky * dilationH - padTop;
            if (iy < 0 || iy >= inH) continue;

            for (int kx = 0; kx < kW; kx++)
            {
                int ix = ox * strideW + kx * dilationW - padLeft;
                if (ix < 0 || ix >= inW) continue;

                sum += input[inBase + iy * inW + ix] * weight[wBase + ky * kW + kx];
            }
        }

        output[idx] = sum;
    }

    public void ForwardDepthwise(
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> weight,
        ArrayView1D<float, Stride1D.Dense> bias,
        ArrayView1D<float, Stride1D.Dense> output,
        int C, int inH, int inW,
        int kH, int kW,
        int stride = 1, int padding = 0,
        int dilationH = 1, int dilationW = 1)
        => ForwardDepthwisePadded(input, weight, bias, output, C, inH, inW, kH, kW,
            stride, padding, padding, padding, padding, dilationH, dilationW);

    /// <summary>Depthwise Conv2D NCHW with explicit asymmetric ONNX pads [top,left,bottom,right].</summary>
    public void ForwardDepthwisePadded(
        ArrayView1D<float, Stride1D.Dense> input, ArrayView1D<float, Stride1D.Dense> weight,
        ArrayView1D<float, Stride1D.Dense> bias, ArrayView1D<float, Stride1D.Dense> output,
        int C, int inH, int inW, int kH, int kW,
        int stride, int padTop, int padLeft, int padBottom, int padRight,
        int dilationH = 1, int dilationW = 1, int batch = 1, int strideW = 0)
    {
        EnsureLoaded();
        if (batch < 1) batch = 1;
        if (strideW <= 0) strideW = stride;   // `stride` is the H stride; W defaults to it (square)
        int effKH = dilationH * (kH - 1) + 1;
        int effKW = dilationW * (kW - 1) + 1;
        int outH = (inH + padTop + padBottom - effKH) / stride + 1;
        int outW = (inW + padLeft + padRight - effKW) / strideW + 1;
        if (outH <= 0 || outW <= 0)
            throw new InvalidOperationException(
                $"DepthwiseConv2D output dimensions are invalid: outH={outH}, outW={outW} " +
                $"(C={C}, inH={inH}, inW={inW}, kH={kH}, kW={kW}, stride={stride}, pads=[{padTop},{padLeft},{padBottom},{padRight}], dilation={dilationH}x{dilationW}). " +
                $"This usually means SAME padding was not applied correctly.");
        long needed = (long)batch * C * outH * outW;
        if (output.Length < needed)
            throw new InvalidOperationException(
                $"DepthwiseConv2D NCHW output buffer too small: output.Length={output.Length} but kernel will write {needed} elements " +
                $"(C={C} outH={outH} outW={outW}, inH={inH} inW={inW} kH={kH} kW={kW} stride={stride} pads=[{padTop},{padLeft},{padBottom},{padRight}] dilation={dilationH}x{dilationW}). " +
                $"Upstream shape inference allocated wrong size.");

        _depthwiseKernel!((int)needed, input, weight, bias, output,
            C, inH, inW, Pack16(kH, kW, "kernel size"), Pack16(stride, strideW, "stride"), Pack16(padTop, padLeft, "begin pads"),
            outH, outW, Pack16(dilationH, dilationW, "dilation"));
    }

    // ═══ NHWC Variants (TFLite native layout) ═══

    /// <summary>
    /// Conv2D NHWC: input [N,H,W,inC], weight [outC,kH,kW,inC], output [N,outH,outW,outC].
    /// </summary>
    private static void Conv2DNHWCImpl(
        Index1D idx,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> weight,
        ArrayView1D<float, Stride1D.Dense> bias,
        ArrayView1D<float, Stride1D.Dense> output,
        int inC, int inH, int inW, int outC, int kHW,
        int strideHW, int padTL, int outH, int outW, int dilHW)
    {
        // NHWC output: [b, oy, ox, oc] indexing. Kernel size + begin pads + dilations packed (Pack16). BATCH-aware
        // (this kernel used to compute image 0 only).
        int kH = kHW >> 16, kW = kHW & 0xFFFF;
        int padTop = padTL >> 16, padLeft = padTL & 0xFFFF;
        int dilationH = dilHW >> 16, dilationW = dilHW & 0xFFFF;
        int strideH = strideHW >> 16, strideW = strideHW & 0xFFFF;
        int oc = idx % outC;
        int rem = idx / outC;
        int ox = rem % outW;
        int rem3 = rem / outW;
        int oy = rem3 % outH;
        int b = rem3 / outH;
        int inBatchBase = b * inH * inW * inC;

        float sum = bias[oc];

        int kernelSize = inC * kH * kW;
        for (int k = 0; k < kernelSize; k++)
        {
            int ic = k / (kH * kW);
            int rem2 = k % (kH * kW);
            int ky = rem2 / kW;
            int kx = rem2 % kW;

            int iy = oy * strideH + ky * dilationH - padTop;
            if (iy < 0 || iy >= inH) continue;
            int ix = ox * strideW + kx * dilationW - padLeft;
            if (ix < 0 || ix >= inW) continue;

            int inIdx = inBatchBase + (iy * inW + ix) * inC + ic;
            int wIdx = ((oc * kH + ky) * kW + kx) * inC + ic;
            sum += input[inIdx] * weight[wIdx];
        }

        output[idx] = sum;
    }

    public void ForwardNHWC(
        ArrayView1D<float, Stride1D.Dense> input, ArrayView1D<float, Stride1D.Dense> weight,
        ArrayView1D<float, Stride1D.Dense> bias, ArrayView1D<float, Stride1D.Dense> output,
        int inC, int inH, int inW, int outC, int kH, int kW, int stride = 1, int padding = 0,
        int dilationH = 1, int dilationW = 1)
        => ForwardNHWCPadded(input, weight, bias, output, inC, inH, inW, outC, kH, kW,
            stride, padding, padding, padding, padding, dilationH, dilationW);

    /// <summary>Conv2D NHWC with explicit asymmetric ONNX pads [top,left,bottom,right].</summary>
    public void ForwardNHWCPadded(
        ArrayView1D<float, Stride1D.Dense> input, ArrayView1D<float, Stride1D.Dense> weight,
        ArrayView1D<float, Stride1D.Dense> bias, ArrayView1D<float, Stride1D.Dense> output,
        int inC, int inH, int inW, int outC, int kH, int kW,
        int stride, int padTop, int padLeft, int padBottom, int padRight,
        int dilationH = 1, int dilationW = 1, int batch = 1, int strideW = 0)
    {
        EnsureLoaded();
        if (batch < 1) batch = 1;
        if (strideW <= 0) strideW = stride;   // `stride` is the H stride; W defaults to it (square)
        int effKH = dilationH * (kH - 1) + 1;
        int effKW = dilationW * (kW - 1) + 1;
        int outH = (inH + padTop + padBottom - effKH) / stride + 1;
        int outW = (inW + padLeft + padRight - effKW) / strideW + 1;
        if (outH <= 0 || outW <= 0)
            throw new InvalidOperationException(
                $"Conv2D NHWC output dimensions are invalid: outH={outH}, outW={outW} " +
                $"(inC={inC}, inH={inH}, inW={inW}, outC={outC}, kH={kH}, kW={kW}, stride={stride}, pads=[{padTop},{padLeft},{padBottom},{padRight}], dilation={dilationH}x{dilationW}). " +
                $"This usually means SAME padding was not applied correctly.");
        long needed = (long)batch * outH * outW * outC;
        if (output.Length < needed)
            throw new InvalidOperationException(
                $"Conv2D NHWC output buffer too small: output.Length={output.Length} but kernel will write {needed} elements " +
                $"(outH={outH} outW={outW} outC={outC}, inC={inC} inH={inH} inW={inW} kH={kH} kW={kW} stride={stride} pads=[{padTop},{padLeft},{padBottom},{padRight}] dilation={dilationH}x{dilationW}). " +
                $"Upstream shape inference allocated wrong size.");
        _conv2dNHWCKernel!((int)needed, input, weight, bias, output,
            inC, inH, inW, outC, Pack16(kH, kW, "kernel size"), Pack16(stride, strideW, "stride"), Pack16(padTop, padLeft, "begin pads"),
            outH, outW, Pack16(dilationH, dilationW, "dilation"));
    }

    /// <summary>
    /// Depthwise Conv2D NHWC: input [N,H,W,C], weight [1,kH,kW,C], output [N,outH,outW,C].
    /// </summary>
    private static void DepthwiseConv2DNHWCImpl(
        Index1D idx,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> weight,
        ArrayView1D<float, Stride1D.Dense> bias,
        ArrayView1D<float, Stride1D.Dense> output,
        int C, int inH, int inW, int kHW,
        int strideHW, int padTL, int outH, int outW, int dilHW)
    {
        // Kernel size + begin pads + dilations packed (Pack16). BATCH-aware: [b, oy, ox, c] (this kernel used to
        // compute image 0 only).
        int kH = kHW >> 16, kW = kHW & 0xFFFF;
        int padTop = padTL >> 16, padLeft = padTL & 0xFFFF;
        int dilationH = dilHW >> 16, dilationW = dilHW & 0xFFFF;
        int strideH = strideHW >> 16, strideW = strideHW & 0xFFFF;
        int c = idx % C;
        int rem = idx / C;
        int ox = rem % outW;
        int rem3 = rem / outW;
        int oy = rem3 % outH;
        int b = rem3 / outH;
        int inBatchBase = b * inH * inW * C;

        float sum = bias[c];

        int kernelSize = kH * kW;
        for (int k = 0; k < kernelSize; k++)
        {
            int ky = k / kW;
            int kx = k % kW;
            int iy = oy * strideH + ky * dilationH - padTop;
            if (iy < 0 || iy >= inH) continue;
            int ix = ox * strideW + kx * dilationW - padLeft;
            if (ix < 0 || ix >= inW) continue;

            int inIdx = inBatchBase + (iy * inW + ix) * C + c;
            int wIdx = (ky * kW + kx) * C + c;
            sum += input[inIdx] * weight[wIdx];
        }

        output[idx] = sum;
    }

    public void ForwardDepthwiseNHWC(
        ArrayView1D<float, Stride1D.Dense> input, ArrayView1D<float, Stride1D.Dense> weight,
        ArrayView1D<float, Stride1D.Dense> bias, ArrayView1D<float, Stride1D.Dense> output,
        int C, int inH, int inW, int kH, int kW, int stride = 1, int padding = 0,
        int dilationH = 1, int dilationW = 1)
        => ForwardDepthwiseNHWCPadded(input, weight, bias, output, C, inH, inW, kH, kW,
            stride, padding, padding, padding, padding, dilationH, dilationW);

    /// <summary>Depthwise Conv2D NHWC with explicit asymmetric ONNX pads [top,left,bottom,right].</summary>
    public void ForwardDepthwiseNHWCPadded(
        ArrayView1D<float, Stride1D.Dense> input, ArrayView1D<float, Stride1D.Dense> weight,
        ArrayView1D<float, Stride1D.Dense> bias, ArrayView1D<float, Stride1D.Dense> output,
        int C, int inH, int inW, int kH, int kW,
        int stride, int padTop, int padLeft, int padBottom, int padRight,
        int dilationH = 1, int dilationW = 1, int batch = 1, int strideW = 0)
    {
        EnsureLoaded();
        if (batch < 1) batch = 1;
        if (strideW <= 0) strideW = stride;   // `stride` is the H stride; W defaults to it (square)
        int effKH = dilationH * (kH - 1) + 1;
        int effKW = dilationW * (kW - 1) + 1;
        int outH = (inH + padTop + padBottom - effKH) / stride + 1;
        int outW = (inW + padLeft + padRight - effKW) / strideW + 1;
        if (outH <= 0 || outW <= 0)
            throw new InvalidOperationException(
                $"DepthwiseConv2D NHWC output dimensions are invalid: outH={outH}, outW={outW} " +
                $"(C={C}, inH={inH}, inW={inW}, kH={kH}, kW={kW}, stride={stride}, pads=[{padTop},{padLeft},{padBottom},{padRight}], dilation={dilationH}x{dilationW}). " +
                $"This usually means SAME padding was not applied correctly.");
        long needed = (long)batch * outH * outW * C;
        if (output.Length < needed)
            throw new InvalidOperationException(
                $"DepthwiseConv2D NHWC output buffer too small: output.Length={output.Length} but kernel will write {needed} elements " +
                $"(outH={outH} outW={outW} C={C}, inH={inH} inW={inW} kH={kH} kW={kW} stride={stride} pads=[{padTop},{padLeft},{padBottom},{padRight}] dilation={dilationH}x{dilationW}). " +
                $"Upstream shape inference allocated wrong size.");
        _depthwiseNHWCKernel!((int)needed, input, weight, bias, output,
            C, inH, inW, Pack16(kH, kW, "kernel size"), Pack16(stride, strideW, "stride"), Pack16(padTop, padLeft, "begin pads"),
            outH, outW, Pack16(dilationH, dilationW, "dilation"));
    }

    private void EnsureLoaded()
    {
        _conv2dKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            int, int, int, int, int, int, int, int, int, int>(Conv2DImpl);
        // Low-p-weight conv kernels are lazy per concrete T (see ForwardPaddedLowPWeight / GetConv2DLowPWeightKernel).
        _depthwiseKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            int, int, int, int, int, int, int, int, int>(DepthwiseConv2DImpl);
        _conv2dNHWCKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            int, int, int, int, int, int, int, int, int, int>(Conv2DNHWCImpl);
        _depthwiseNHWCKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            int, int, int, int, int, int, int, int, int>(DepthwiseConv2DNHWCImpl);
    }

    public void Dispose() { /* no buffers owned */ }
}
