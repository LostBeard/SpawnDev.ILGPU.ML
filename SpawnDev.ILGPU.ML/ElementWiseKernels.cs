using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML;

/// <summary>Binary operations for GPU broadcast kernels.</summary>
public enum BroadcastOp { Add, Sub, Mul, Div, Pow }

/// <summary>
/// Element-wise neural network operations: GELU, ReLU, Add, Mul, AddBias.
/// All use auto-grouped 1D kernels (no shared memory needed).
///
/// Future home: SpawnDev.ILGPU.ML
/// </summary>
public class ElementWiseKernels : IDisposable
{
    private readonly Accelerator _accelerator;

    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>? _geluKernel;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>? _reluKernel;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>? _addKernel;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>? _mulKernel;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int>? _addBiasKernel;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int>? _broadcastMulKernel;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, float>? _scaleKernel;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int>? _transposeLastTwoKernel;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>>? _geluInPlaceKernel;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>>? _reluInPlaceKernel;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, float>? _scaleInPlaceKernel;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, float>? _fillKernel;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, DelegateSpecialization<Func<float, float, float>>>? _broadcastBinaryKernel;
    // Kept alive until next BroadcastBinaryOpND call to avoid synchronous Synchronize()
    // which deadlocks on WebGPU/WebGL/Wasm backends. By the next call, the GPU has
    // finished reading the previous strides buffer.
    private MemoryBuffer1D<int, Stride1D.Dense>? _lastStridesBuf;
    private MemoryBuffer1D<int, Stride1D.Dense>? _broadcastStridesBuf;
    private readonly List<MemoryBuffer1D<int, Stride1D.Dense>> _oldStridesBufs = new();
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>? _addInPlaceKernel;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int>? _concatLastDimKernel;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int, int, int, int>? _bilinearUpsampleKernel;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int, int, int, int>? _bilinearUpsampleACKernel;

    public ElementWiseKernels(Accelerator accelerator) => _accelerator = accelerator;

    // ─────────────────────────────────────────────────────────────
    //  Kernel implementations
    // ─────────────────────────────────────────────────────────────

    /// <summary>
    /// GELU activation using erf approximation (matches PyTorch nn.GELU default):
    /// GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
    /// </summary>
    private static void GELUImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output)
    {
        float x = input[idx];
        if (x > 10f) { output[idx] = x; return; }
        if (x < -10f) { output[idx] = 0f; return; }
        const float INV_SQRT2 = 0.7071067811865475f;
        float z = x * INV_SQRT2;
        float az = z < 0f ? -z : z;
        const float p = 0.3275911f;
        const float a1 = 0.254829592f;
        const float a2 = -0.284496736f;
        const float a3 = 1.421413741f;
        const float a4 = -1.453152027f;
        const float a5 = 1.061405429f;
        float t = 1f / (1f + p * az);
        float t2 = t * t;
        float t3 = t2 * t;
        float t4 = t3 * t;
        float t5 = t4 * t;
        float erfAbs = 1f - (a1 * t + a2 * t2 + a3 * t3 + a4 * t4 + a5 * t5) * MathF.Exp(-az * az);
        float erf = z < 0f ? -erfAbs : erfAbs;
        output[idx] = 0.5f * x * (1f + erf);
    }

    /// <summary>ReLU: y = max(0, x)</summary>
    private static void ReLUImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output)
    {
        float x = input[idx];
        output[idx] = x > 0f ? x : 0f;
    }

    /// <summary>In-place ReLU: data[i] = max(0, data[i]). Single binding.</summary>
    private static void ReLUInPlaceImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> data)
    {
        float x = data[idx];
        if (x < 0f) data[idx] = 0f;
    }

    /// <summary>Element-wise add: out = a + b</summary>
    private static void AddImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> a,
        ArrayView1D<float, Stride1D.Dense> b,
        ArrayView1D<float, Stride1D.Dense> output)
    {
        output[idx] = a[idx] + b[idx];
    }

    /// <summary>Element-wise multiply: out = a * b</summary>
    private static void MulImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> a,
        ArrayView1D<float, Stride1D.Dense> b,
        ArrayView1D<float, Stride1D.Dense> output)
    {
        output[idx] = a[idx] * b[idx];
    }

    /// <summary>
    /// Add bias to each row: output[r*C + c] = input[r*C + c] + bias[c].
    /// Used after MatMul for linear layers.
    /// </summary>
    private static void AddBiasImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> data,  // in-place [rows * C]
        ArrayView1D<float, Stride1D.Dense> bias,   // [C]
        int C)
    {
        data[idx] += bias[idx % C];
    }

    /// <summary>
    /// Broadcast multiply: out[i] = a[i] * b[i % C].
    /// Used for LayerScale: gamma[C] broadcast across T rows.
    /// </summary>
    private static void BroadcastMulImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> scale,  // [C] — broadcast
        ArrayView1D<float, Stride1D.Dense> output,
        int C)
    {
        output[idx] = input[idx] * scale[idx % C];
    }

    /// <summary>Scale all elements: out[i] = input[i] * scalar.</summary>
    private static void ScaleImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output,
        float scalar)
    {
        output[idx] = input[idx] * scalar;
    }

    /// <summary>
    /// In-place GELU using erf approximation (matches PyTorch nn.GELU default / ONNX Erf subgraph):
    /// GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
    /// erf approximated via Abramowitz & Stegun (max error 1.5e-7).
    /// </summary>
    private static void GELUInPlaceImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> data)
    {
        float x = data[idx];
        // For large |x|, GELU(x) ≈ x (positive) or 0 (negative)
        if (x > 10f) { return; }
        if (x < -10f) { data[idx] = 0f; return; }
        // erf(x/√2) via Abramowitz & Stegun 5-term approximation
        const float INV_SQRT2 = 0.7071067811865475f;
        float z = x * INV_SQRT2;
        float az = z < 0f ? -z : z; // |z|
        const float p = 0.3275911f;
        const float a1 = 0.254829592f;
        const float a2 = -0.284496736f;
        const float a3 = 1.421413741f;
        const float a4 = -1.453152027f;
        const float a5 = 1.061405429f;
        float t = 1f / (1f + p * az);
        float t2 = t * t;
        float t3 = t2 * t;
        float t4 = t3 * t;
        float t5 = t4 * t;
        float erfAbs = 1f - (a1 * t + a2 * t2 + a3 * t3 + a4 * t4 + a5 * t5) * MathF.Exp(-az * az);
        float erf = z < 0f ? -erfAbs : erfAbs;
        data[idx] = 0.5f * x * (1f + erf);
    }

    // ═══════════════════════════════════════════════════════════
    //  Broadcast Binary Kernel (N-D stride-based, DelegateSpecialization)
    //  Supports arbitrary broadcast shapes up to 5D.
    //  Strides encode how each output index maps to input indices.
    //  Operation selected at dispatch time via DelegateSpecialization.
    // ═══════════════════════════════════════════════════════════

    static float BroadcastAddOp(float a, float b) => a + b;
    static float BroadcastSubOp(float a, float b) => a - b;
    static float BroadcastMulOp(float a, float b) => a * b;
    static float BroadcastDivOp(float a, float b) => b != 0f ? a / b : 0f;
    static float BroadcastPowOp(float a, float b) => MathF.Pow(a, b);

    /// <summary>
    /// Unified broadcast binary kernel: output[i] = op(a[mapA(i)], b[mapB(i)]).
    /// The operation is selected at dispatch time via DelegateSpecialization —
    /// Add, Sub, Mul, or Div are inlined into the kernel at compile time.
    /// </summary>
    private static void BroadcastBinaryKernel(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> a,
        ArrayView1D<float, Stride1D.Dense> b,
        ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<int, Stride1D.Dense> strides,
        DelegateSpecialization<Func<float, float, float>> op)
    {
        // strides layout: [rank, aStrides[0..rank], bStrides[0..rank], outStrides[0..rank]]
        int rank = strides[0];
        int aIdx = 0, bIdx = 0, remaining = idx;
        for (int d = 0; d < rank; d++)
        {
            int outStride = strides[1 + 2 * rank + d];
            int coord = outStride > 0 ? remaining / outStride : 0;
            remaining = outStride > 0 ? remaining % outStride : remaining;
            aIdx += coord * strides[1 + d];
            bIdx += coord * strides[1 + rank + d];
        }
        output[idx] = op.Value(a[aIdx], b[bIdx]);
    }

    /// <summary>Fill: data[i] = value. Sets every element to a constant.</summary>
    private static void FillImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> data,
        float value)
    {
        data[idx] = value;
    }

    /// <summary>In-place scale: data[i] *= scalar. Single binding, no aliasing.</summary>
    private static void ScaleInPlaceImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> data,
        float scalar)
    {
        data[idx] *= scalar;
    }

    /// <summary>In-place add: data[i] += other[i]. Two separate bindings.</summary>
    private static void AddInPlaceImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> data,
        ArrayView1D<float, Stride1D.Dense> other)
    {
        data[idx] += other[idx];
    }

    /// <summary>
    /// Transpose last two dims: [batch, rows, cols] → [batch, cols, rows].
    /// One thread per element.
    /// </summary>
    private static void TransposeLastTwoImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output,
        int rows, int cols)
    {
        int batchStride = rows * cols;
        int b = idx / batchStride;
        int rem = idx % batchStride;
        int r = rem / cols;
        int c = rem % cols;
        // output[b, c, r] = input[b, r, c]
        output[b * batchStride + c * rows + r] = input[idx];
    }

    /// <summary>
    /// Concatenate two [T, C] tensors along the last dim → [T, 2C].
    /// Inputs flat [T*C], output flat [T*2C].
    /// output[t, :C] = a[t, :]; output[t, C:] = b[t, :]
    /// </summary>
    private static void ConcatLastDimImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> a,
        ArrayView1D<float, Stride1D.Dense> b,
        ArrayView1D<float, Stride1D.Dense> output,
        int T, int C)
    {
        int C2 = C * 2;
        int t = idx / C2;
        int c = idx % C2;
        output[idx] = (c < C) ? a[t * C + c] : b[t * C + (c - C)];
    }

    /// <summary>
    /// Bilinear upsample: [C, inH, inW] → [C, outH, outW]. Align-corners=false.
    /// </summary>
    private static void BilinearUpsampleImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output,
        int C, int inH, int inW, int outH, int outW)
    {
        int ox = idx % outW;
        int rem = idx / outW;
        int oy = rem % outH;
        int c = rem / outH;

        float fy = ((oy + 0.5f) * inH / outH) - 0.5f;
        float fx = ((ox + 0.5f) * inW / outW) - 0.5f;

        float floorY = MathF.Floor(fy); float floorX = MathF.Floor(fx);
        int y0 = (int)floorY; int y1 = y0 + 1;
        int x0 = (int)floorX; int x1 = x0 + 1;
        float ty = fy - floorY; float tx = fx - floorX;
        if (y0 < 0) y0 = 0; if (y1 >= inH) y1 = inH - 1;
        if (x0 < 0) x0 = 0; if (x1 >= inW) x1 = inW - 1;

        int b = c * inH * inW;
        float v00 = input[b + y0 * inW + x0]; float v01 = input[b + y0 * inW + x1];
        float v10 = input[b + y1 * inW + x0]; float v11 = input[b + y1 * inW + x1];
        output[idx] = v00 * (1f - ty) * (1f - tx) + v01 * (1f - ty) * tx
                    + v10 * ty * (1f - tx) + v11 * ty * tx;
    }

    /// <summary>
    /// Bilinear upsample: [C, inH, inW] → [C, outH, outW]. Align-corners=true.
    /// Matches ONNX Resize with coordinate_transformation_mode=align_corners.
    /// </summary>
    private static void BilinearUpsampleACImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output,
        int C, int inH, int inW, int outH, int outW)
    {
        int ox = idx % outW;
        int rem = idx / outW;
        int oy = rem % outH;
        int c = rem / outH;

        float fy = (outH > 1) ? (float)oy * (inH - 1) / (outH - 1) : 0f;
        float fx = (outW > 1) ? (float)ox * (inW - 1) / (outW - 1) : 0f;

        float floorY = MathF.Floor(fy); float floorX = MathF.Floor(fx);
        int y0 = (int)floorY; int y1 = y0 + 1;
        int x0 = (int)floorX; int x1 = x0 + 1;
        float ty = fy - floorY; float tx = fx - floorX;
        if (y1 >= inH) y1 = inH - 1;
        if (x1 >= inW) x1 = inW - 1;

        int b = c * inH * inW;
        float v00 = input[b + y0 * inW + x0]; float v01 = input[b + y0 * inW + x1];
        float v10 = input[b + y1 * inW + x0]; float v11 = input[b + y1 * inW + x1];
        output[idx] = v00 * (1f - ty) * (1f - tx) + v01 * (1f - ty) * tx
                    + v10 * ty * (1f - tx) + v11 * ty * tx;
    }

    // ─────────────────────────────────────────────────────────────
    //  Public API
    // ─────────────────────────────────────────────────────────────

    public void GELU(ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output, int count)
    {
        EnsureLoaded();
        _geluKernel!(count, input, output);
    }

    public void ReLU(ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output, int count)
    {
        EnsureLoaded();
        _reluKernel!(count, input, output);
    }

    /// <summary>Element-wise subtract: output = a - b</summary>
    public void Sub(ArrayView1D<float, Stride1D.Dense> a,
        ArrayView1D<float, Stride1D.Dense> b,
        ArrayView1D<float, Stride1D.Dense> output, int count)
    {
        var kernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>>(
            (Index1D idx, ArrayView1D<float, Stride1D.Dense> x, ArrayView1D<float, Stride1D.Dense> y,
             ArrayView1D<float, Stride1D.Dense> o) => { o[idx] = x[idx] - y[idx]; });
        kernel(count, a, b, output);
    }

    /// <summary>LeakyReLU: output = x >= 0 ? x : alpha * x</summary>
    public void LeakyReLU(ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output, int count, float alpha)
    {
        var kernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, float>(
            (Index1D idx, ArrayView1D<float, Stride1D.Dense> inp, ArrayView1D<float, Stride1D.Dense> outp, float a) =>
            {
                float x = inp[idx];
                outp[idx] = x >= 0f ? x : a * x;
            });
        kernel(count, input, output, alpha);
    }

    public void Add(ArrayView1D<float, Stride1D.Dense> a,
        ArrayView1D<float, Stride1D.Dense> b,
        ArrayView1D<float, Stride1D.Dense> output, int count)
    {
        EnsureLoaded();
        _addKernel!(count, a, b, output);
    }

    public void Mul(ArrayView1D<float, Stride1D.Dense> a,
        ArrayView1D<float, Stride1D.Dense> b,
        ArrayView1D<float, Stride1D.Dense> output, int count)
    {
        EnsureLoaded();
        _mulKernel!(count, a, b, output);
    }

    /// <summary>In-place bias addition: data[i] += bias[i % C].</summary>
    public void AddBias(ArrayView1D<float, Stride1D.Dense> data,
        ArrayView1D<float, Stride1D.Dense> bias, int totalElements, int C)
    {
        EnsureLoaded();
        _addBiasKernel!(totalElements, data, bias, C);
    }

    /// <summary>Broadcast multiply: out[i] = a[i] * scale[i % C]. For LayerScale.</summary>
    public void BroadcastMul(ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> scale,
        ArrayView1D<float, Stride1D.Dense> output, int totalElements, int C)
    {
        EnsureLoaded();
        _broadcastMulKernel!(totalElements, input, scale, output, C);
    }

    /// <summary>Scale all elements by a constant: out[i] = in[i] * scalar.</summary>
    public void Scale(ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output, int count, float scalar)
    {
        EnsureLoaded();
        _scaleKernel!(count, input, output, scalar);
    }

    /// <summary>Transpose last two dims: [batch, rows, cols] → [batch, cols, rows].</summary>
    public void TransposeLastTwo(ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output, int batch, int rows, int cols)
    {
        EnsureLoaded();
        _transposeLastTwoKernel!(batch * rows * cols, input, output, rows, cols);
    }

    /// <summary>In-place GELU. No aliasing (single buffer binding).</summary>
    public void GELUInPlace(ArrayView1D<float, Stride1D.Dense> data, int count)
    {
        EnsureLoaded();
        _geluInPlaceKernel!(count, data);
    }

    /// <summary>In-place ReLU: data[i] = max(0, data[i]). Single buffer binding.</summary>
    public void ReLUInPlace(ArrayView1D<float, Stride1D.Dense> data, int count)
    {
        EnsureLoaded();
        _reluInPlaceKernel!(count, data);
    }

    /// <summary>
    /// Concatenate two [T, C] tensors along last dim → [T, 2C].
    /// Inputs flat [T*C], output flat [T*2C]. output must be a separate buffer.
    /// </summary>
    public void ConcatLastDim(
        ArrayView1D<float, Stride1D.Dense> a,
        ArrayView1D<float, Stride1D.Dense> b,
        ArrayView1D<float, Stride1D.Dense> output,
        int T, int C)
    {
        EnsureLoaded();
        _concatLastDimKernel!(T * 2 * C, a, b, output, T, C);
    }

    /// <summary>
    /// General N-D broadcast binary operation on GPU.
    /// Handles arbitrary shape combinations: [N,T,C] op [N,T,1], [B,C,H,W] op [1,C,1,1], etc.
    /// Uses stride-based index mapping — same algorithm as CPU BroadcastHelper but runs on GPU.
    /// </summary>
    public void BroadcastBinaryOpND(
        ArrayView1D<float, Stride1D.Dense> a,
        ArrayView1D<float, Stride1D.Dense> b,
        ArrayView1D<float, Stride1D.Dense> output,
        int[] aShape, int[] bShape, int[] outShape,
        BroadcastOp op)
    {
        EnsureLoaded();
        int rank = outShape.Length;
        int outCount = 1;
        for (int i = 0; i < rank; i++) outCount *= outShape[i];

        // Compute broadcast strides
        var aStrides = Operators.BroadcastHelper.ComputeStrides(aShape, outShape);
        var bStrides = Operators.BroadcastHelper.ComputeStrides(bShape, outShape);
        var outStrides = Operators.BroadcastHelper.ComputeStrides(outShape, outShape);

        // Pack strides: [rank, aStrides[0..rank], bStrides[0..rank], outStrides[0..rank]]
        // CRITICAL: Allocate a new buffer per call — WebGPU dispatch is async,
        // and reusing a shared buffer causes race conditions when multiple
        // BroadcastBinaryOpND calls are queued (e.g., decomposed LayerNorm:
        // Sub, Pow, Div, Mul all dispatch in sequence without sync).
        int paramsSize = 1 + 3 * rank;
        // Accumulate stride buffers — disposal happens in Dispose().
        // On WebGPU/WebGL/Wasm, inline disposal causes ObjectDisposedException because
        // the buffer may still be referenced by pending dispatches in the command encoder.
        if (_lastStridesBuf != null) _oldStridesBufs.Add(_lastStridesBuf);
        _lastStridesBuf = _accelerator.Allocate1D<int>(paramsSize);
        var paramsData = new int[paramsSize];
        paramsData[0] = rank;
        for (int i = 0; i < rank; i++) paramsData[1 + i] = aStrides[i];
        for (int i = 0; i < rank; i++) paramsData[1 + rank + i] = bStrides[i];
        for (int i = 0; i < rank; i++) paramsData[1 + 2 * rank + i] = outStrides[i];
        _lastStridesBuf.View.SubView(0, paramsSize).CopyFromCPU(paramsData);

        var opSpec = op switch
        {
            BroadcastOp.Add => new DelegateSpecialization<Func<float, float, float>>(BroadcastAddOp),
            BroadcastOp.Sub => new DelegateSpecialization<Func<float, float, float>>(BroadcastSubOp),
            BroadcastOp.Mul => new DelegateSpecialization<Func<float, float, float>>(BroadcastMulOp),
            BroadcastOp.Div => new DelegateSpecialization<Func<float, float, float>>(BroadcastDivOp),
            BroadcastOp.Pow => new DelegateSpecialization<Func<float, float, float>>(BroadcastPowOp),
            _ => throw new ArgumentException($"Unsupported broadcast op: {op}")
        };
        _broadcastBinaryKernel!(outCount, a, b, output, _lastStridesBuf.View, opSpec);
    }

    /// <summary>Fill every element with a constant value. Handles -Infinity, Infinity, NaN.</summary>
    public void Fill(ArrayView1D<float, Stride1D.Dense> data, int count, float value)
    {
        EnsureLoaded();
        _fillKernel!(count, data, value);
    }

    /// <summary>In-place scale: data[i] *= scalar. No aliasing (single buffer binding).</summary>
    public void ScaleInPlace(ArrayView1D<float, Stride1D.Dense> data, int count, float scalar)
    {
        EnsureLoaded();
        _scaleInPlaceKernel!(count, data, scalar);
    }

    /// <summary>
    /// Bilinear upsample [C, inH, inW] → [C, outH, outW]. Align-corners=false.
    /// </summary>
    public void BilinearUpsample(
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output,
        int C, int inH, int inW, int outH, int outW)
    {
        EnsureLoaded();
        _bilinearUpsampleKernel!(C * outH * outW, input, output, C, inH, inW, outH, outW);
    }

    /// <summary>
    /// Nearest-neighbor upsample for 4D NCHW tensors. Pure GPU — no CPU readback.
    /// Each output element copies from the nearest input element based on scale ratios.
    /// params: [inC, inH, inW, outH, outW]
    /// </summary>
    private static void NearestUpsample4DImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<int, Stride1D.Dense> p)
    {
        int inC = p[0]; int inH = p[1]; int inW = p[2];
        int outH = p[3]; int outW = p[4];

        int outHW = outH * outW;
        int c = idx / outHW;
        int rem = idx % outHW;
        int oy = rem / outW;
        int ox = rem % outW;

        // Nearest-neighbor mapping (floor of scaled coordinate)
        int iy = oy * inH / outH;
        int ix = ox * inW / outW;
        if (iy >= inH) iy = inH - 1;
        if (ix >= inW) ix = inW - 1;

        output[idx] = input[c * inH * inW + iy * inW + ix];
    }

    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<int, Stride1D.Dense>>? _nearestUpsampleKernel;
    private MemoryBuffer1D<int, Stride1D.Dense>? _nearestParamsBuf;

    /// <summary>
    /// Nearest-neighbor upsample: maps each output element to the nearest input element.
    /// Handles 4D NCHW tensors. Pure GPU kernel — no CPU readback, works on all backends.
    /// </summary>
    public void NearestUpsample(
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output,
        int[] inputShape, int[] outputShape)
    {
        EnsureLoaded();
        _nearestUpsampleKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>>(NearestUpsample4DImpl);

        // For 4D NCHW: C = product of all dims except last 2
        int rank = inputShape.Length;
        int inH = rank >= 2 ? inputShape[rank - 2] : 1;
        int inW = rank >= 1 ? inputShape[rank - 1] : 1;
        int outH = rank >= 2 ? outputShape[rank - 2] : 1;
        int outW = rank >= 1 ? outputShape[rank - 1] : 1;
        int inC = 1;
        for (int i = 0; i < rank - 2; i++) inC *= inputShape[i];

        int totalOut = inC * outH * outW;

        _nearestParamsBuf ??= _accelerator.Allocate1D<int>(5);
        _nearestParamsBuf.CopyFromCPU(new int[] { inC, inH, inW, outH, outW });

        _nearestUpsampleKernel(totalOut, input, output, _nearestParamsBuf.View);
    }

    /// <summary>
    /// Bilinear upsample [C, inH, inW] → [C, outH, outW]. Align-corners=true.
    /// Matches ONNX Resize with coordinate_transformation_mode=align_corners.
    /// </summary>
    public void BilinearUpsampleAlignCorners(
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output,
        int C, int inH, int inW, int outH, int outW)
    {
        EnsureLoaded();
        _bilinearUpsampleACKernel!(C * outH * outW, input, output, C, inH, inW, outH, outW);
    }

    /// <summary>In-place add: data[i] += other[i]. Two separate buffers required.</summary>
    public void AddInPlace(ArrayView1D<float, Stride1D.Dense> data,
        ArrayView1D<float, Stride1D.Dense> other, int count)
    {
        EnsureLoaded();
        _addInPlaceKernel!(count, data, other);
    }

    // ─────────────────────────────────────────────────────────────
    //  Additional element-wise ops (Sqrt, Exp, Div, Pow, Abs, Neg, Reciprocal, Erf)
    // ─────────────────────────────────────────────────────────────

    private static void SqrtImpl(Index1D idx, ArrayView1D<float, Stride1D.Dense> input, ArrayView1D<float, Stride1D.Dense> output)
    { output[idx] = MathF.Sqrt(input[idx]); }

    private static void SinImpl(Index1D idx, ArrayView1D<float, Stride1D.Dense> input, ArrayView1D<float, Stride1D.Dense> output)
    { output[idx] = MathF.Sin(input[idx]); }

    private static void CosImpl(Index1D idx, ArrayView1D<float, Stride1D.Dense> input, ArrayView1D<float, Stride1D.Dense> output)
    { output[idx] = MathF.Cos(input[idx]); }

    private static void TanImpl(Index1D idx, ArrayView1D<float, Stride1D.Dense> input, ArrayView1D<float, Stride1D.Dense> output)
    { output[idx] = MathF.Tan(input[idx]); }

    /// <summary>ArgMax along axis: each thread handles one output element (outer × inner).</summary>
    private static void ArgMaxImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output,
        int outerSize, int axisSize, int innerSize)
    {
        int outIdx = idx;
        if (outIdx >= outerSize * innerSize) return;
        int o = outIdx / innerSize;
        int inn = outIdx % innerSize;

        float maxVal = float.NegativeInfinity;
        int maxIdx = 0;
        for (int a = 0; a < axisSize; a++)
        {
            float val = input[(o * axisSize + a) * innerSize + inn];
            if (val > maxVal) { maxVal = val; maxIdx = a; }
        }
        output[outIdx] = maxIdx;
    }

    private static void ExpImpl(Index1D idx, ArrayView1D<float, Stride1D.Dense> input, ArrayView1D<float, Stride1D.Dense> output)
    {
        float x = input[idx];
        output[idx] = x > 80f ? float.PositiveInfinity : (x < -80f ? 0f : MathF.Exp(x));
    }

    private static void DivImpl(Index1D idx, ArrayView1D<float, Stride1D.Dense> a, ArrayView1D<float, Stride1D.Dense> b, ArrayView1D<float, Stride1D.Dense> output)
    { output[idx] = a[idx] / b[idx]; }

    private static void PowImpl(Index1D idx, ArrayView1D<float, Stride1D.Dense> a, ArrayView1D<float, Stride1D.Dense> b, ArrayView1D<float, Stride1D.Dense> output)
    { output[idx] = MathF.Pow(a[idx], b[idx]); }

    private static void AbsImpl(Index1D idx, ArrayView1D<float, Stride1D.Dense> input, ArrayView1D<float, Stride1D.Dense> output)
    { float x = input[idx]; output[idx] = x < 0f ? -x : x; }

    private static void NegImpl(Index1D idx, ArrayView1D<float, Stride1D.Dense> input, ArrayView1D<float, Stride1D.Dense> output)
    { output[idx] = -input[idx]; }

    private static void ReciprocalImpl(Index1D idx, ArrayView1D<float, Stride1D.Dense> input, ArrayView1D<float, Stride1D.Dense> output)
    { output[idx] = 1f / input[idx]; }

    /// <summary>Erf approximation (Abramowitz & Stegun 5-term, max error 1.5e-7).</summary>
    private static void ErfImpl(Index1D idx, ArrayView1D<float, Stride1D.Dense> input, ArrayView1D<float, Stride1D.Dense> output)
    {
        float x = input[idx];
        float ax = x < 0f ? -x : x;
        const float p = 0.3275911f;
        const float a1 = 0.254829592f, a2 = -0.284496736f, a3 = 1.421413741f;
        const float a4 = -1.453152027f, a5 = 1.061405429f;
        float t = 1f / (1f + p * ax);
        float t2 = t * t; float t3 = t2 * t; float t4 = t3 * t; float t5 = t4 * t;
        float erfAbs = 1f - (a1 * t + a2 * t2 + a3 * t3 + a4 * t4 + a5 * t5) * MathF.Exp(-ax * ax);
        output[idx] = x < 0f ? -erfAbs : erfAbs;
    }

    private static void FloorImpl(Index1D idx, ArrayView1D<float, Stride1D.Dense> input, ArrayView1D<float, Stride1D.Dense> output)
    { output[idx] = MathF.Floor(input[idx]); }

    private static void CeilImpl(Index1D idx, ArrayView1D<float, Stride1D.Dense> input, ArrayView1D<float, Stride1D.Dense> output)
    { output[idx] = MathF.Ceiling(input[idx]); }

    private static void LogImpl(Index1D idx, ArrayView1D<float, Stride1D.Dense> input, ArrayView1D<float, Stride1D.Dense> output)
    { output[idx] = MathF.Log(input[idx]); }

    private static void RoundImpl(Index1D idx, ArrayView1D<float, Stride1D.Dense> input, ArrayView1D<float, Stride1D.Dense> output)
    {
        // ONNX Round = round-half-to-even (banker's rounding).
        // MathF.Round doesn't map to ILGPU intrinsic on all backends.
        // Implement using only Floor + arithmetic (works on all backends):
        float x = input[idx];
        float rounded = MathF.Floor(x + 0.5f);
        // If exactly halfway (x+0.5 is integer), round to even:
        // check if rounded is odd via Floor(r/2)*2 != r
        float xp = x + 0.5f;
        if (xp == rounded && MathF.Floor(rounded * 0.5f) * 2f != rounded)
            rounded -= 1f;
        output[idx] = rounded;
    }

    /// <summary>Truncate: round toward zero (C-style cast from float to int).</summary>
    private static void TruncateImpl(Index1D idx, ArrayView1D<float, Stride1D.Dense> input, ArrayView1D<float, Stride1D.Dense> output)
    { output[idx] = MathF.Truncate(input[idx]); }

    private static void MinImpl(Index1D idx, ArrayView1D<float, Stride1D.Dense> a, ArrayView1D<float, Stride1D.Dense> b, ArrayView1D<float, Stride1D.Dense> output)
    { output[idx] = a[idx] < b[idx] ? a[idx] : b[idx]; }

    private static void MaxImpl(Index1D idx, ArrayView1D<float, Stride1D.Dense> a, ArrayView1D<float, Stride1D.Dense> b, ArrayView1D<float, Stride1D.Dense> output)
    { output[idx] = a[idx] > b[idx] ? a[idx] : b[idx]; }

    private static void EqualImpl(Index1D idx, ArrayView1D<float, Stride1D.Dense> a, ArrayView1D<float, Stride1D.Dense> b, ArrayView1D<float, Stride1D.Dense> output)
    { output[idx] = a[idx] == b[idx] ? 1f : 0f; }

    private static void GreaterImpl(Index1D idx, ArrayView1D<float, Stride1D.Dense> a, ArrayView1D<float, Stride1D.Dense> b, ArrayView1D<float, Stride1D.Dense> output)
    { output[idx] = a[idx] > b[idx] ? 1f : 0f; }

    private static void LessImpl(Index1D idx, ArrayView1D<float, Stride1D.Dense> a, ArrayView1D<float, Stride1D.Dense> b, ArrayView1D<float, Stride1D.Dense> output)
    { output[idx] = a[idx] < b[idx] ? 1f : 0f; }

    private static void ClipImpl(Index1D idx, ArrayView1D<float, Stride1D.Dense> input, ArrayView1D<float, Stride1D.Dense> output, float minVal, float maxVal)
    {
        float x = input[idx];
        if (x < minVal) x = minVal;
        if (x > maxVal) x = maxVal;
        output[idx] = x;
    }

    private static void ThresholdedReluImpl(Index1D idx, ArrayView1D<float, Stride1D.Dense> input, ArrayView1D<float, Stride1D.Dense> output, float alpha)
    { output[idx] = input[idx] > alpha ? input[idx] : 0f; }

    private static void IsNaNImpl(Index1D idx, ArrayView1D<float, Stride1D.Dense> input, ArrayView1D<float, Stride1D.Dense> output)
    { float x = input[idx]; output[idx] = (x != x) ? 1f : 0f; }

    private static void TriluImpl(Index1D idx, ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output, ArrayView1D<int, Stride1D.Dense> paramsArr)
    {
        // params: [rows, cols, k, upper, batchStride]
        int rows = paramsArr[0]; int cols = paramsArr[1]; int k = paramsArr[2];
        int upper = paramsArr[3]; int batchStride = paramsArr[4];
        int inBatch = idx / batchStride;
        int posInBatch = idx - inBatch * batchStride;
        int r = posInBatch / cols;
        int c = posInBatch - r * cols;
        bool keep = upper != 0 ? (c >= r + k) : (c <= r + k);
        output[idx] = keep ? input[idx] : 0f;
    }

    private static void LRNImpl(Index1D idx, ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output, ArrayView1D<int, Stride1D.Dense> paramsArr,
        ArrayView1D<float, Stride1D.Dense> fparams)
    {
        // int params: [C, spatial, halfSize, size]
        // float params: [alpha, beta, bias]
        int C = paramsArr[0]; int spatial = paramsArr[1]; int halfSize = paramsArr[2]; int size = paramsArr[3];
        float alpha = fparams[0]; float beta = fparams[1]; float bias = fparams[2];
        // idx = n * C * spatial + c * spatial + s
        int n_cs = idx;
        int s = n_cs % spatial; n_cs /= spatial;
        int c = n_cs % C; int n = n_cs / C;
        int cStart = c - halfSize; if (cStart < 0) cStart = 0;
        int cEnd = c + halfSize; if (cEnd >= C) cEnd = C - 1;
        float sqSum = 0f;
        int nOff = n * C * spatial;
        for (int cp = cStart; cp <= cEnd; cp++)
        {
            float v = input[nOff + cp * spatial + s];
            sqSum += v * v;
        }
        float denom = MathF.Pow(bias + alpha / size * sqSum, beta);
        output[idx] = input[idx] / denom;
    }

    private static void LpNormImpl(Index1D idx, ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output, ArrayView1D<int, Stride1D.Dense> paramsArr)
    {
        // params: [axisSize, inner, p]
        // idx ranges over outer * inner
        int axisSize = paramsArr[0]; int inner = paramsArr[1]; int p = paramsArr[2];
        int o = idx / inner;
        int inn = idx - o * inner;
        // Compute Lp norm along axis for this (outer, inner) position
        float norm = 0f;
        for (int a = 0; a < axisSize; a++)
        {
            int srcIdx = (o * axisSize + a) * inner + inn;
            float v = input[srcIdx]; if (v < 0f) v = -v;
            norm += p == 1 ? v : v * v;
        }
        if (norm < 1e-10f) norm = 1e-10f;
        if (p != 1) norm = MathF.Sqrt(norm);
        // Write normalized values
        for (int a = 0; a < axisSize; a++)
        {
            int srcIdx = (o * axisSize + a) * inner + inn;
            output[srcIdx] = input[srcIdx] / norm;
        }
    }

    private static void GlobalLpPoolImpl(Index1D idx, ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output, ArrayView1D<int, Stride1D.Dense> paramsArr)
    {
        // params: [C, spatial, p]
        // idx = n * C + c (one thread per channel per batch)
        int C = paramsArr[0]; int spatial = paramsArr[1]; int p = paramsArr[2];
        int n = idx / C; int c = idx - n * C;
        float sum = 0f;
        int off = (n * C + c) * spatial;
        for (int s = 0; s < spatial; s++)
        {
            float v = input[off + s]; if (v < 0f) v = -v;
            sum += p == 1 ? v : v * v;
        }
        output[idx] = p == 1 ? sum : MathF.Sqrt(sum);
    }

    private static void AffineGridImpl(Index1D idx, ArrayView1D<float, Stride1D.Dense> theta,
        ArrayView1D<float, Stride1D.Dense> output, ArrayView1D<int, Stride1D.Dense> paramsArr)
    {
        // params: [H, W, alignCorners]
        int H = paramsArr[0]; int W = paramsArr[1]; int alignCorners = paramsArr[2];
        // idx = n * H * W + pixel_idx
        int hw = H * W;
        int n = idx / hw;
        int pixelIdx = idx - n * hw;
        int iy = pixelIdx / W;
        int ix = pixelIdx - iy * W;
        // Compute normalized coordinates
        float nx, ny;
        if (alignCorners != 0)
        {
            nx = W > 1 ? 2f * ix / (W - 1) - 1f : 0f;
            ny = H > 1 ? 2f * iy / (H - 1) - 1f : 0f;
        }
        else
        {
            nx = (2f * ix + 1f) / W - 1f;
            ny = (2f * iy + 1f) / H - 1f;
        }
        // theta[n] is [2,3]: [[a,b,c],[d,e,f]]
        int tOff = n * 6;
        float ox = theta[tOff + 0] * nx + theta[tOff + 1] * ny + theta[tOff + 2];
        float oy = theta[tOff + 3] * nx + theta[tOff + 4] * ny + theta[tOff + 5];
        int outOff = (n * H * W + pixelIdx) * 2;
        output[outOff] = ox;
        output[outOff + 1] = oy;
    }

    private static void LpPoolImpl(Index1D idx, ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output, ArrayView1D<int, Stride1D.Dense> paramsArr)
    {
        // params: [C, H, W, outH, outW, kH, kW, sH, sW, pH, pW, p]
        int C = paramsArr[0]; int H = paramsArr[1]; int W = paramsArr[2];
        int outH = paramsArr[3]; int outW = paramsArr[4];
        int kH = paramsArr[5]; int kW = paramsArr[6];
        int sH = paramsArr[7]; int sW = paramsArr[8];
        int pH = paramsArr[9]; int pW = paramsArr[10]; int p = paramsArr[11];
        // idx = n * C * outH * outW + c * outH * outW + oh * outW + ow
        int tmp = idx;
        int ow = tmp % outW; tmp /= outW;
        int oh = tmp % outH; tmp /= outH;
        int c = tmp % C; int n = tmp / C;
        float sum = 0f;
        int hStart = oh * sH - pH; int wStart = ow * sW - pW;
        for (int kh = 0; kh < kH; kh++)
        {
            int ih = hStart + kh;
            if (ih < 0 || ih >= H) continue;
            for (int kw = 0; kw < kW; kw++)
            {
                int iw = wStart + kw;
                if (iw < 0 || iw >= W) continue;
                float v = input[((n * C + c) * H + ih) * W + iw];
                if (v < 0f) v = -v;
                sum += p == 2 ? v * v : (p == 1 ? v : MathF.Pow(v, p));
            }
        }
        output[idx] = p == 2 ? MathF.Sqrt(sum) : (p == 1 ? sum : MathF.Pow(sum, 1f / p));
    }

    private static void GridSampleImpl(Index1D idx, ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> grid, ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<int, Stride1D.Dense> paramsArr)
    {
        // params: [N, C, Hin, Win, Hout, Wout, alignCorners]
        int C = paramsArr[1]; int Hin = paramsArr[2]; int Win = paramsArr[3];
        int Hout = paramsArr[4]; int Wout = paramsArr[5]; int alignCorners = paramsArr[6];
        // idx = n * C * Hout * Wout + c * Hout * Wout + h * Wout + w
        int tmp = idx;
        int w = tmp % Wout; tmp /= Wout;
        int h = tmp % Hout; tmp /= Hout;
        int c = tmp % C; int n = tmp / C;
        // Read grid coords for this pixel
        int gridIdx = ((n * Hout + h) * Wout + w) * 2;
        float gx = grid[gridIdx]; float gy = grid[gridIdx + 1];
        // Denormalize
        float ix, iy;
        if (alignCorners != 0)
        {
            ix = (gx + 1f) * 0.5f * (Win - 1);
            iy = (gy + 1f) * 0.5f * (Hin - 1);
        }
        else
        {
            ix = ((gx + 1f) * Win - 1f) * 0.5f;
            iy = ((gy + 1f) * Hin - 1f) * 0.5f;
        }
        // Bilinear interpolation
        int x0 = (int)MathF.Floor(ix); int y0 = (int)MathF.Floor(iy);
        int x1 = x0 + 1; int y1 = y0 + 1;
        float tx = ix - x0; float ty = iy - y0;
        int chOff = (n * C + c) * Hin;
        float v00 = 0f, v01 = 0f, v10 = 0f, v11 = 0f;
        if (x0 >= 0 && x0 < Win && y0 >= 0 && y0 < Hin) v00 = input[(chOff + y0) * Win + x0];
        if (x1 >= 0 && x1 < Win && y0 >= 0 && y0 < Hin) v01 = input[(chOff + y0) * Win + x1];
        if (x0 >= 0 && x0 < Win && y1 >= 0 && y1 < Hin) v10 = input[(chOff + y1) * Win + x0];
        if (x1 >= 0 && x1 < Win && y1 >= 0 && y1 < Hin) v11 = input[(chOff + y1) * Win + x1];
        output[idx] = v00 * (1f - tx) * (1f - ty) + v01 * tx * (1f - ty)
            + v10 * (1f - tx) * ty + v11 * tx * ty;
    }

    private static void RoiAlignImpl(Index1D idx, ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> rois, ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<int, Stride1D.Dense> paramsArr, ArrayView1D<float, Stride1D.Dense> fparams)
    {
        // int params: [C, Hin, Win, outH, outW, samplingRatio]
        // float params: [spatialScale]
        // rois: [numRois, 4] (x1, y1, x2, y2) or [numRois, 5] (batchIdx, x1, y1, x2, y2)
        int C = paramsArr[0]; int Hin = paramsArr[1]; int Win = paramsArr[2];
        int outH = paramsArr[3]; int outW = paramsArr[4]; int samplingRatio = paramsArr[5];
        float spatialScale = fparams[0];
        // idx = r * C * outH * outW + c * outH * outW + oh * outW + ow
        int tmp = idx;
        int ow = tmp % outW; tmp /= outW;
        int oh = tmp % outH; tmp /= outH;
        int c = tmp % C; int r = tmp / C;
        // Get ROI bounds
        int roiOff = r * 4; // assume 4-element ROIs with batch_indices separate
        float x1 = rois[roiOff] * spatialScale;
        float y1 = rois[roiOff + 1] * spatialScale;
        float x2 = rois[roiOff + 2] * spatialScale;
        float y2 = rois[roiOff + 3] * spatialScale;
        float roiW = x2 - x1; float roiH = y2 - y1;
        if (roiW < 1f) roiW = 1f; if (roiH < 1f) roiH = 1f;
        float binH = roiH / outH; float binW = roiW / outW;
        int sH = samplingRatio > 0 ? samplingRatio : (int)MathF.Ceiling(binH);
        int sW = samplingRatio > 0 ? samplingRatio : (int)MathF.Ceiling(binW);
        if (sH < 1) sH = 1; if (sW < 1) sW = 1;
        float sum = 0f;
        int chOff = c * Hin; // batch index 0 for simplicity
        for (int sh = 0; sh < sH; sh++)
        {
            float fy = y1 + binH * (oh + (sh + 0.5f) / sH);
            for (int sw = 0; sw < sW; sw++)
            {
                float fx = x1 + binW * (ow + (sw + 0.5f) / sW);
                // Bilinear
                int x0i = (int)MathF.Floor(fx); int y0i = (int)MathF.Floor(fy);
                float tx = fx - x0i; float ty = fy - y0i;
                float v00 = 0f, v01 = 0f, v10 = 0f, v11 = 0f;
                if (x0i >= 0 && x0i < Win && y0i >= 0 && y0i < Hin) v00 = input[(chOff + y0i) * Win + x0i];
                if (x0i + 1 < Win && y0i >= 0 && y0i < Hin) v01 = input[(chOff + y0i) * Win + x0i + 1];
                if (x0i >= 0 && x0i < Win && y0i + 1 < Hin) v10 = input[(chOff + y0i + 1) * Win + x0i];
                if (x0i + 1 < Win && y0i + 1 < Hin) v11 = input[(chOff + y0i + 1) * Win + x0i + 1];
                sum += v00 * (1f - tx) * (1f - ty) + v01 * tx * (1f - ty)
                    + v10 * (1f - tx) * ty + v11 * tx * ty;
            }
        }
        output[idx] = sum / (sH * sW);
    }

    private static void ReverseSequenceImpl(Index1D idx, ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output, ArrayView1D<float, Stride1D.Dense> seqLens,
        ArrayView1D<int, Stride1D.Dense> paramsArr)
    {
        // params: [batchAxis, timeAxis, batchSize, timeSize, innerSize]
        int batchAxis = paramsArr[0]; int timeAxis = paramsArr[1];
        int batchSize = paramsArr[2]; int timeSize = paramsArr[3]; int innerSize = paramsArr[4];
        // Decode idx → (batch, time, inner) assuming shape [batch, time, inner] or [time, batch, inner]
        int inn = idx % innerSize;
        int tmp = idx / innerSize;
        int timeIdx, batchIdx;
        if (batchAxis == 0) { timeIdx = tmp % timeSize; batchIdx = tmp / timeSize; }
        else { batchIdx = tmp % batchSize; timeIdx = tmp / batchSize; }
        int seqLen = (int)seqLens[batchIdx];
        int srcTime = (timeIdx < seqLen) ? (seqLen - 1 - timeIdx) : timeIdx;
        int srcIdx;
        if (batchAxis == 0) srcIdx = (batchIdx * timeSize + srcTime) * innerSize + inn;
        else srcIdx = (srcTime * batchSize + batchIdx) * innerSize + inn;
        output[idx] = input[srcIdx];
    }

    private static void DFTImpl(Index1D idx, ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output, ArrayView1D<int, Stride1D.Dense> paramsArr)
    {
        // params: [N, dftLength, outputN, isComplex, inverse]
        int N = paramsArr[0]; int dftLength = paramsArr[1]; int outputN = paramsArr[2];
        int isComplex = paramsArr[3]; int inverse = paramsArr[4];
        // idx = b * outputN + k
        int k = idx % outputN;
        int b = idx / outputN;
        float sign = inverse != 0 ? 1f : -1f;
        float scale = inverse != 0 ? 1f / dftLength : 1f;
        float sumReal = 0f, sumImag = 0f;
        int limit = N < dftLength ? N : dftLength;
        for (int n = 0; n < limit; n++)
        {
            float angle = sign * 2f * MathF.PI * k * n / dftLength;
            float cosA = MathF.Cos(angle);
            float sinA = MathF.Sin(angle);
            float xReal, xImag;
            if (isComplex != 0)
            {
                xReal = input[(b * N + n) * 2];
                xImag = input[(b * N + n) * 2 + 1];
            }
            else
            {
                xReal = input[b * N + n];
                xImag = 0f;
            }
            sumReal += xReal * cosA - xImag * sinA;
            sumImag += xReal * sinA + xImag * cosA;
        }
        int outIdx = (b * outputN + k) * 2;
        output[outIdx] = sumReal * scale;
        output[outIdx + 1] = sumImag * scale;
    }

    private static void Col2ImImpl(Index1D idx, ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output, ArrayView1D<int, Stride1D.Dense> paramsArr)
    {
        // Each thread handles one scatter source position
        // params: [C, L, kH, kW, outH, outW, sH, sW, pH, pW, blocksW, colDim]
        int C = paramsArr[0]; int L = paramsArr[1];
        int kH = paramsArr[2]; int kW = paramsArr[3];
        int outH = paramsArr[4]; int outW = paramsArr[5];
        int sH = paramsArr[6]; int sW = paramsArr[7];
        int pH = paramsArr[8]; int pW = paramsArr[9];
        int blocksW = paramsArr[10]; int colDim = paramsArr[11];
        // idx = n * colDim * L + colIdx * L + l
        int tmp = idx;
        int l = tmp % L; tmp /= L;
        int colIdx = tmp % colDim; int n = tmp / colDim;
        int bh = l / blocksW; int bw = l - bh * blocksW;
        int c = colIdx / (kH * kW);
        int rem = colIdx - c * kH * kW;
        int kh = rem / kW; int kw = rem - kh * kW;
        int oh = bh * sH + kh - pH;
        int ow = bw * sW + kw - pW;
        if (oh >= 0 && oh < outH && ow >= 0 && ow < outW)
        {
            float val = input[(n * colDim + colIdx) * L + l];
            // Atomic add not available in ILGPU for float — use non-atomic accumulation
            // This is safe when each (oh, ow) position is written by at most one thread,
            // which holds for non-overlapping kernels (stride >= kernel).
            // For overlapping kernels, results may have race conditions — acceptable for now.
            output[((n * C + c) * outH + oh) * outW + ow] += val;
        }
    }

    private static void MaxRoiPoolImpl(Index1D idx, ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> rois, ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<int, Stride1D.Dense> paramsArr, ArrayView1D<float, Stride1D.Dense> fparams)
    {
        // params: [C, H, W, outH, outW]
        // fparams: [spatialScale]
        int C = paramsArr[0]; int H = paramsArr[1]; int W = paramsArr[2];
        int outH = paramsArr[3]; int outW = paramsArr[4];
        float spatialScale = fparams[0];
        int tmp = idx;
        int ow = tmp % outW; tmp /= outW;
        int oh = tmp % outH; tmp /= outH;
        int c = tmp % C; int r = tmp / C;
        int roiOff = r * 4;
        float x1 = rois[roiOff] * spatialScale;
        float y1 = rois[roiOff + 1] * spatialScale;
        float x2 = rois[roiOff + 2] * spatialScale;
        float y2 = rois[roiOff + 3] * spatialScale;
        float roiW = x2 - x1; if (roiW < 1f) roiW = 1f;
        float roiH = y2 - y1; if (roiH < 1f) roiH = 1f;
        float binH = roiH / outH; float binW = roiW / outW;
        int hStart = (int)MathF.Floor(y1 + binH * oh);
        int hEnd = (int)MathF.Ceiling(y1 + binH * (oh + 1));
        int wStart = (int)MathF.Floor(x1 + binW * ow);
        int wEnd = (int)MathF.Ceiling(x1 + binW * (ow + 1));
        if (hStart < 0) hStart = 0; if (hEnd > H) hEnd = H;
        if (wStart < 0) wStart = 0; if (wEnd > W) wEnd = W;
        float maxVal = -1e10f;
        int chOff = c * H;
        for (int ih = hStart; ih < hEnd; ih++)
            for (int iw = wStart; iw < wEnd; iw++)
            {
                float v = input[(chOff + ih) * W + iw];
                if (v > maxVal) maxVal = v;
            }
        output[idx] = maxVal;
    }

    private static void MaxUnpoolImpl(Index1D idx, ArrayView1D<float, Stride1D.Dense> vals,
        ArrayView1D<float, Stride1D.Dense> indices, ArrayView1D<float, Stride1D.Dense> output,
        int outSize)
    {
        int targetIdx = (int)indices[idx];
        if (targetIdx >= 0 && targetIdx < outSize)
            output[targetIdx] = vals[idx];
    }

    private static void HardmaxImpl(Index1D idx, ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output, int axisSize)
    {
        // idx = outer batch index, each batch has axisSize elements
        int offset = idx * axisSize;
        float maxVal = input[offset];
        int maxIdx = 0;
        for (int i = 1; i < axisSize; i++)
        {
            float v = input[offset + i];
            if (v > maxVal) { maxVal = v; maxIdx = i; }
        }
        for (int i = 0; i < axisSize; i++)
            output[offset + i] = (i == maxIdx) ? 1f : 0f;
    }

    private static void DynamicQuantizeImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> input, ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<float, Stride1D.Dense> scaleOut, ArrayView1D<float, Stride1D.Dense> zpOut,
        ArrayView1D<float, Stride1D.Dense> maxBuf, ArrayView1D<float, Stride1D.Dense> minBuf)
    {
        // Compute scale and zero_point from pre-reduced max/min (read from GPU buffers)
        float xMax = maxBuf[0]; if (xMax < 0f) xMax = 0f;
        float xMin = minBuf[0]; if (xMin > 0f) xMin = 0f;
        float yScale = (xMax - xMin) / 255f;
        if (yScale == 0f) yScale = 1f;
        float yZeroPoint = MathF.Floor(-xMin / yScale + 0.5f);
        if (yZeroPoint < 0f) yZeroPoint = 0f;
        if (yZeroPoint > 255f) yZeroPoint = 255f;
        // First thread writes scale and zero_point
        if (idx == 0)
        {
            scaleOut[0] = yScale;
            zpOut[0] = yZeroPoint;
        }
        // Quantize: y = clamp(round(x / scale) + zero_point, 0, 255)
        float val = MathF.Floor(input[idx] / yScale + 0.5f) + yZeroPoint;
        if (val < 0f) val = 0f;
        if (val > 255f) val = 255f;
        output[idx] = val;
    }

    /// <summary>Where: output[i] = condition[i] != 0 ? x[i] : y[i].</summary>
    private static void WhereImpl(Index1D idx, ArrayView1D<float, Stride1D.Dense> cond,
        ArrayView1D<float, Stride1D.Dense> x, ArrayView1D<float, Stride1D.Dense> y,
        ArrayView1D<float, Stride1D.Dense> output)
    { output[idx] = cond[idx] != 0f ? x[idx] : y[idx]; }

    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>? _sqrtKernel;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>? _sinKernel;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>? _cosKernel;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>? _tanKernel;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int, int>? _argMaxKernel;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>? _expKernel;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>? _divKernel;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>? _powKernel;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>? _absKernel;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>? _negKernel;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>? _reciprocalKernel;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>? _erfKernel;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>? _whereKernel;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>? _floorKernel;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>? _ceilKernel;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>? _logKernel;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>? _roundKernel;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>? _minKernel;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>? _maxKernel;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, float, float>? _clipKernel;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, float>? _thresholdedReluKernel;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>? _dynamicQuantizeKernel;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>? _isNaNKernel;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<int, Stride1D.Dense>>? _triluKernel;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<int, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>? _lrnKernel;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int>? _hardmaxKernel;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<int, Stride1D.Dense>>? _lpNormKernel;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<int, Stride1D.Dense>>? _globalLpPoolKernel;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<int, Stride1D.Dense>>? _affineGridKernel;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<int, Stride1D.Dense>>? _lpPoolKernel;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>>? _gridSampleKernel;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>>? _roiAlignKernel;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>>? _reverseSequenceKernel;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, int>? _maxUnpoolKernel;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<int, Stride1D.Dense>>? _dftKernel;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<int, Stride1D.Dense>>? _col2ImKernel;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>>? _maxRoiPoolKernel;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>? _equalKernel;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>? _greaterKernel;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>? _lessKernel;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>? _truncateKernel;

    public void Sqrt(ArrayView1D<float, Stride1D.Dense> input, ArrayView1D<float, Stride1D.Dense> output, int count)
    { EnsureLoaded2(); _sqrtKernel!(count, input, output); }
    public void Sin(ArrayView1D<float, Stride1D.Dense> input, ArrayView1D<float, Stride1D.Dense> output, int count)
    { EnsureLoaded2(); _sinKernel!(count, input, output); }
    public void Cos(ArrayView1D<float, Stride1D.Dense> input, ArrayView1D<float, Stride1D.Dense> output, int count)
    { EnsureLoaded2(); _cosKernel!(count, input, output); }
    public void Tan(ArrayView1D<float, Stride1D.Dense> input, ArrayView1D<float, Stride1D.Dense> output, int count)
    { EnsureLoaded2(); _tanKernel!(count, input, output); }
    public void ArgMax(ArrayView1D<float, Stride1D.Dense> input, ArrayView1D<float, Stride1D.Dense> output, int outerSize, int axisSize, int innerSize)
    { EnsureLoaded2(); _argMaxKernel!(outerSize * innerSize, input, output, outerSize, axisSize, innerSize); }
    public void Exp(ArrayView1D<float, Stride1D.Dense> input, ArrayView1D<float, Stride1D.Dense> output, int count)
    { EnsureLoaded2(); _expKernel!(count, input, output); }
    public void Div(ArrayView1D<float, Stride1D.Dense> a, ArrayView1D<float, Stride1D.Dense> b, ArrayView1D<float, Stride1D.Dense> output, int count)
    { EnsureLoaded2(); _divKernel!(count, a, b, output); }
    public void Pow(ArrayView1D<float, Stride1D.Dense> a, ArrayView1D<float, Stride1D.Dense> b, ArrayView1D<float, Stride1D.Dense> output, int count)
    { EnsureLoaded2(); _powKernel!(count, a, b, output); }
    public void Abs(ArrayView1D<float, Stride1D.Dense> input, ArrayView1D<float, Stride1D.Dense> output, int count)
    { EnsureLoaded2(); _absKernel!(count, input, output); }
    public void Neg(ArrayView1D<float, Stride1D.Dense> input, ArrayView1D<float, Stride1D.Dense> output, int count)
    { EnsureLoaded2(); _negKernel!(count, input, output); }
    public void Reciprocal(ArrayView1D<float, Stride1D.Dense> input, ArrayView1D<float, Stride1D.Dense> output, int count)
    { EnsureLoaded2(); _reciprocalKernel!(count, input, output); }
    public void Erf(ArrayView1D<float, Stride1D.Dense> input, ArrayView1D<float, Stride1D.Dense> output, int count)
    { EnsureLoaded2(); _erfKernel!(count, input, output); }
    public void Where(ArrayView1D<float, Stride1D.Dense> cond, ArrayView1D<float, Stride1D.Dense> x,
        ArrayView1D<float, Stride1D.Dense> y, ArrayView1D<float, Stride1D.Dense> output, int count)
    { EnsureLoaded2(); _whereKernel!(count, cond, x, y, output); }

    public void Floor(ArrayView1D<float, Stride1D.Dense> input, ArrayView1D<float, Stride1D.Dense> output, int count)
    { EnsureLoaded2(); _floorKernel!(count, input, output); }
    public void Ceil(ArrayView1D<float, Stride1D.Dense> input, ArrayView1D<float, Stride1D.Dense> output, int count)
    { EnsureLoaded2(); _ceilKernel!(count, input, output); }
    public void Log(ArrayView1D<float, Stride1D.Dense> input, ArrayView1D<float, Stride1D.Dense> output, int count)
    { EnsureLoaded2(); _logKernel!(count, input, output); }
    public void Round(ArrayView1D<float, Stride1D.Dense> input, ArrayView1D<float, Stride1D.Dense> output, int count)
    { EnsureLoaded2(); _roundKernel!(count, input, output); }
    public void Min(ArrayView1D<float, Stride1D.Dense> a, ArrayView1D<float, Stride1D.Dense> b, ArrayView1D<float, Stride1D.Dense> output, int count)
    { EnsureLoaded2(); _minKernel!(count, a, b, output); }
    public void Max(ArrayView1D<float, Stride1D.Dense> a, ArrayView1D<float, Stride1D.Dense> b, ArrayView1D<float, Stride1D.Dense> output, int count)
    { EnsureLoaded2(); _maxKernel!(count, a, b, output); }
    public void Clip(ArrayView1D<float, Stride1D.Dense> input, ArrayView1D<float, Stride1D.Dense> output, int count, float minVal, float maxVal)
    { EnsureLoaded2(); _clipKernel!(count, input, output, minVal, maxVal); }
    public void ThresholdedRelu(ArrayView1D<float, Stride1D.Dense> input, ArrayView1D<float, Stride1D.Dense> output, int count, float alpha)
    { EnsureLoaded2(); _thresholdedReluKernel!(count, input, output, alpha); }
    public void DynamicQuantize(ArrayView1D<float, Stride1D.Dense> input, ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<float, Stride1D.Dense> scaleOut, ArrayView1D<float, Stride1D.Dense> zpOut,
        ArrayView1D<float, Stride1D.Dense> maxBuf, ArrayView1D<float, Stride1D.Dense> minBuf, int count)
    { EnsureLoaded2(); _dynamicQuantizeKernel!(count, input, output, scaleOut, zpOut, maxBuf, minBuf); }
    public void IsNaN(ArrayView1D<float, Stride1D.Dense> input, ArrayView1D<float, Stride1D.Dense> output, int count)
    { EnsureLoaded2(); _isNaNKernel!(count, input, output); }
    public void Trilu(ArrayView1D<float, Stride1D.Dense> input, ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<int, Stride1D.Dense> paramsBuf, int count)
    { EnsureLoaded2(); _triluKernel!(count, input, output, paramsBuf); }
    public void LRN(ArrayView1D<float, Stride1D.Dense> input, ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<int, Stride1D.Dense> paramsBuf, ArrayView1D<float, Stride1D.Dense> fparamsBuf, int count)
    { EnsureLoaded2(); _lrnKernel!(count, input, output, paramsBuf, fparamsBuf); }
    public void Hardmax(ArrayView1D<float, Stride1D.Dense> input, ArrayView1D<float, Stride1D.Dense> output,
        int outerSize, int axisSize)
    { EnsureLoaded2(); _hardmaxKernel!(outerSize, input, output, axisSize); }
    public void LpNorm(ArrayView1D<float, Stride1D.Dense> input, ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<int, Stride1D.Dense> paramsBuf, int outerTimesInner)
    { EnsureLoaded2(); _lpNormKernel!(outerTimesInner, input, output, paramsBuf); }
    public void GlobalLpPool(ArrayView1D<float, Stride1D.Dense> input, ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<int, Stride1D.Dense> paramsBuf, int nTimesC)
    { EnsureLoaded2(); _globalLpPoolKernel!(nTimesC, input, output, paramsBuf); }
    public void AffineGrid(ArrayView1D<float, Stride1D.Dense> theta, ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<int, Stride1D.Dense> paramsBuf, int totalPixels)
    { EnsureLoaded2(); _affineGridKernel!(totalPixels, theta, output, paramsBuf); }
    public void LpPool(ArrayView1D<float, Stride1D.Dense> input, ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<int, Stride1D.Dense> paramsBuf, int totalOutput)
    { EnsureLoaded2(); _lpPoolKernel!(totalOutput, input, output, paramsBuf); }
    public void GridSample(ArrayView1D<float, Stride1D.Dense> input, ArrayView1D<float, Stride1D.Dense> grid,
        ArrayView1D<float, Stride1D.Dense> output, ArrayView1D<int, Stride1D.Dense> paramsBuf, int totalOutput)
    { EnsureLoaded2(); _gridSampleKernel!(totalOutput, input, grid, output, paramsBuf); }
    public void RoiAlign(ArrayView1D<float, Stride1D.Dense> input, ArrayView1D<float, Stride1D.Dense> rois,
        ArrayView1D<float, Stride1D.Dense> output, ArrayView1D<int, Stride1D.Dense> paramsBuf,
        ArrayView1D<float, Stride1D.Dense> fparamsBuf, int totalOutput)
    { EnsureLoaded2(); _roiAlignKernel!(totalOutput, input, rois, output, paramsBuf, fparamsBuf); }
    public void ReverseSequence(ArrayView1D<float, Stride1D.Dense> input, ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<float, Stride1D.Dense> seqLens, ArrayView1D<int, Stride1D.Dense> paramsBuf, int totalElements)
    { EnsureLoaded2(); _reverseSequenceKernel!(totalElements, input, output, seqLens, paramsBuf); }
    public void MaxUnpool(ArrayView1D<float, Stride1D.Dense> vals, ArrayView1D<float, Stride1D.Dense> indices,
        ArrayView1D<float, Stride1D.Dense> output, int inputCount, int outSize)
    { EnsureLoaded2(); _maxUnpoolKernel!(inputCount, vals, indices, output, outSize); }
    public void DFT(ArrayView1D<float, Stride1D.Dense> input, ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<int, Stride1D.Dense> paramsBuf, int totalOutputBins)
    { EnsureLoaded2(); _dftKernel!(totalOutputBins, input, output, paramsBuf); }
    public void Col2Im(ArrayView1D<float, Stride1D.Dense> input, ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<int, Stride1D.Dense> paramsBuf, int totalScatterOps)
    { EnsureLoaded2(); _col2ImKernel!(totalScatterOps, input, output, paramsBuf); }
    public void MaxRoiPool(ArrayView1D<float, Stride1D.Dense> input, ArrayView1D<float, Stride1D.Dense> rois,
        ArrayView1D<float, Stride1D.Dense> output, ArrayView1D<int, Stride1D.Dense> paramsBuf,
        ArrayView1D<float, Stride1D.Dense> fparamsBuf, int totalOutput)
    { EnsureLoaded2(); _maxRoiPoolKernel!(totalOutput, input, rois, output, paramsBuf, fparamsBuf); }
    public void Equal(ArrayView1D<float, Stride1D.Dense> a, ArrayView1D<float, Stride1D.Dense> b, ArrayView1D<float, Stride1D.Dense> output, int count)
    { EnsureLoaded2(); _equalKernel!(count, a, b, output); }
    public void Greater(ArrayView1D<float, Stride1D.Dense> a, ArrayView1D<float, Stride1D.Dense> b, ArrayView1D<float, Stride1D.Dense> output, int count)
    { EnsureLoaded2(); _greaterKernel!(count, a, b, output); }
    public void Less(ArrayView1D<float, Stride1D.Dense> a, ArrayView1D<float, Stride1D.Dense> b, ArrayView1D<float, Stride1D.Dense> output, int count)
    { EnsureLoaded2(); _lessKernel!(count, a, b, output); }

    /// <summary>Truncate: round toward zero. output[i] = MathF.Truncate(input[i]).</summary>
    public void Truncate(ArrayView1D<float, Stride1D.Dense> input, ArrayView1D<float, Stride1D.Dense> output, int count)
    { EnsureLoaded2(); _truncateKernel!(count, input, output); }

    private void EnsureLoaded2()
    {
        var a = _accelerator;
        _sqrtKernel ??= a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(SqrtImpl);
        _sinKernel ??= a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(SinImpl);
        _cosKernel ??= a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(CosImpl);
        _tanKernel ??= a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(TanImpl);
        _argMaxKernel ??= a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int, int>(ArgMaxImpl);
        _expKernel ??= a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(ExpImpl);
        _divKernel ??= a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(DivImpl);
        _powKernel ??= a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(PowImpl);
        _absKernel ??= a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(AbsImpl);
        _negKernel ??= a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(NegImpl);
        _reciprocalKernel ??= a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(ReciprocalImpl);
        _erfKernel ??= a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(ErfImpl);
        _whereKernel ??= a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(WhereImpl);
        _floorKernel ??= a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(FloorImpl);
        _ceilKernel ??= a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(CeilImpl);
        _logKernel ??= a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(LogImpl);
        _roundKernel ??= a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(RoundImpl);
        _minKernel ??= a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(MinImpl);
        _maxKernel ??= a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(MaxImpl);
        _clipKernel ??= a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, float, float>(ClipImpl);
        _thresholdedReluKernel ??= a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, float>(ThresholdedReluImpl);
        _dynamicQuantizeKernel ??= a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(DynamicQuantizeImpl);
        _isNaNKernel ??= a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(IsNaNImpl);
        _triluKernel ??= a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>>(TriluImpl);
        _lrnKernel ??= a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(LRNImpl);
        _hardmaxKernel ??= a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int>(HardmaxImpl);
        _lpNormKernel ??= a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>>(LpNormImpl);
        _globalLpPoolKernel ??= a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>>(GlobalLpPoolImpl);
        _affineGridKernel ??= a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>>(AffineGridImpl);
        _lpPoolKernel ??= a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>>(LpPoolImpl);
        _gridSampleKernel ??= a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>>(GridSampleImpl);
        _roiAlignKernel ??= a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>>(RoiAlignImpl);
        _reverseSequenceKernel ??= a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>>(ReverseSequenceImpl);
        _maxUnpoolKernel ??= a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, int>(MaxUnpoolImpl);
        _dftKernel ??= a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>>(DFTImpl);
        _col2ImKernel ??= a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>>(Col2ImImpl);
        _maxRoiPoolKernel ??= a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>>(MaxRoiPoolImpl);
        _equalKernel ??= a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(EqualImpl);
        _greaterKernel ??= a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(GreaterImpl);
        _lessKernel ??= a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(LessImpl);
        _truncateKernel ??= a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(TruncateImpl);
    }

    private void EnsureLoaded()
    {
        var accelerator = _accelerator;
        _geluKernel ??= accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(GELUImpl);
        _reluKernel ??= accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(ReLUImpl);
        _addKernel ??= accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>>(AddImpl);
        _mulKernel ??= accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>>(MulImpl);
        _addBiasKernel ??= accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            int>(AddBiasImpl);
        _broadcastMulKernel ??= accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, int>(BroadcastMulImpl);
        _scaleKernel ??= accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            float>(ScaleImpl);
        _transposeLastTwoKernel ??= accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            int, int>(TransposeLastTwoImpl);
        _geluInPlaceKernel ??= accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>>(GELUInPlaceImpl);
        _reluInPlaceKernel ??= accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>>(ReLUInPlaceImpl);
        _scaleInPlaceKernel ??= accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, float>(ScaleInPlaceImpl);
        _fillKernel ??= accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, float>(FillImpl);
        _broadcastBinaryKernel ??= accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>,
            DelegateSpecialization<Func<float, float, float>>>(BroadcastBinaryKernel);
        _addInPlaceKernel ??= accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(AddInPlaceImpl);
        _concatLastDimKernel ??= accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, int, int>(ConcatLastDimImpl);
        _bilinearUpsampleKernel ??= accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            int, int, int, int, int>(BilinearUpsampleImpl);
        _bilinearUpsampleACKernel ??= accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            int, int, int, int, int>(BilinearUpsampleACImpl);
    }

    // ─────────────────────────────────────────────────────────────
    //  Generic unary op via DelegateSpecialization (trig, activation, etc.)
    // ─────────────────────────────────────────────────────────────

    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        DelegateSpecialization<Func<float, float>>>? _unaryOpKernel;

    private static void UnaryOpKernel(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output,
        DelegateSpecialization<Func<float, float>> op)
    {
        output[idx] = op.Value(input[idx]);
    }

    private void EnsureUnaryLoaded()
    {
        _unaryOpKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            DelegateSpecialization<Func<float, float>>>(UnaryOpKernel);
    }

    /// <summary>Apply a generic unary function element-wise via DelegateSpecialization.</summary>
    public void UnaryOp(ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output, int count,
        DelegateSpecialization<Func<float, float>> op)
    {
        EnsureUnaryLoaded();
        _unaryOpKernel!(count, input, output, op);
    }

    // Trig functions
    private static float AcosOp(float x) => MathF.Acos(x);
    // Acosh/Asinh/Atanh use mathematical identities — MathF.Acosh etc. cause PTX JIT failure on CUDA
    private static float AcoshOp(float x) => MathF.Log(x + MathF.Sqrt(x * x - 1f));
    private static float AsinOp(float x) => MathF.Asin(x);
    private static float AsinhOp(float x) => MathF.Log(x + MathF.Sqrt(x * x + 1f));
    private static float AtanOp(float x) => MathF.Atan(x);
    private static float AtanhOp(float x) => 0.5f * MathF.Log((1f + x) / (1f - x));
    private static float CoshOp(float x) => MathF.Cosh(x);
    private static float SinhOp(float x) => MathF.Sinh(x);

    // Activations
    private static float EluOp(float x) => x >= 0f ? x : MathF.Exp(x) - 1f;
    private static float CeluOp(float x) => MathF.Max(0f, x) + MathF.Min(0f, MathF.Exp(x) - 1f);
    private static float SeluOp(float x) => x > 0f ? 1.0507f * x : 1.0507f * 1.67326f * (MathF.Exp(x) - 1f);
    private static float SoftplusOp(float x) => MathF.Log(1f + MathF.Exp(x));
    private static float SoftsignOp(float x) => x / (1f + MathF.Abs(x));
    private static float MishOp(float x) => x * MathF.Tanh(MathF.Log(1f + MathF.Exp(x)));
    private static float ThresholdedReluOp(float x) => x > 1f ? x : 0f; // default alpha=1
    internal static float ShrinkOp(float x) => x > 0.5f ? x - 0.5f : x < -0.5f ? x + 0.5f : 0f; // default bias=0, lambd=0.5
    // float.IsInfinity generates invalid GLSL on WebGL — use comparison instead
    private static float IsInfOp(float x) => (x == float.PositiveInfinity || x == float.NegativeInfinity) ? 1f : 0f;

    public void Acos(ArrayView1D<float, Stride1D.Dense> i, ArrayView1D<float, Stride1D.Dense> o, int n)
        => UnaryOp(i, o, n, new DelegateSpecialization<Func<float, float>>(AcosOp));
    public void Acosh(ArrayView1D<float, Stride1D.Dense> i, ArrayView1D<float, Stride1D.Dense> o, int n)
        => UnaryOp(i, o, n, new DelegateSpecialization<Func<float, float>>(AcoshOp));
    public void Asin(ArrayView1D<float, Stride1D.Dense> i, ArrayView1D<float, Stride1D.Dense> o, int n)
        => UnaryOp(i, o, n, new DelegateSpecialization<Func<float, float>>(AsinOp));
    public void Asinh(ArrayView1D<float, Stride1D.Dense> i, ArrayView1D<float, Stride1D.Dense> o, int n)
        => UnaryOp(i, o, n, new DelegateSpecialization<Func<float, float>>(AsinhOp));
    public void Atan(ArrayView1D<float, Stride1D.Dense> i, ArrayView1D<float, Stride1D.Dense> o, int n)
        => UnaryOp(i, o, n, new DelegateSpecialization<Func<float, float>>(AtanOp));
    public void Atanh(ArrayView1D<float, Stride1D.Dense> i, ArrayView1D<float, Stride1D.Dense> o, int n)
        => UnaryOp(i, o, n, new DelegateSpecialization<Func<float, float>>(AtanhOp));
    public void Cosh(ArrayView1D<float, Stride1D.Dense> i, ArrayView1D<float, Stride1D.Dense> o, int n)
        => UnaryOp(i, o, n, new DelegateSpecialization<Func<float, float>>(CoshOp));
    public void Sinh(ArrayView1D<float, Stride1D.Dense> i, ArrayView1D<float, Stride1D.Dense> o, int n)
        => UnaryOp(i, o, n, new DelegateSpecialization<Func<float, float>>(SinhOp));
    public void Elu(ArrayView1D<float, Stride1D.Dense> i, ArrayView1D<float, Stride1D.Dense> o, int n)
        => UnaryOp(i, o, n, new DelegateSpecialization<Func<float, float>>(EluOp));
    public void Celu(ArrayView1D<float, Stride1D.Dense> i, ArrayView1D<float, Stride1D.Dense> o, int n)
        => UnaryOp(i, o, n, new DelegateSpecialization<Func<float, float>>(CeluOp));
    public void Selu(ArrayView1D<float, Stride1D.Dense> i, ArrayView1D<float, Stride1D.Dense> o, int n)
        => UnaryOp(i, o, n, new DelegateSpecialization<Func<float, float>>(SeluOp));
    public void Softplus(ArrayView1D<float, Stride1D.Dense> i, ArrayView1D<float, Stride1D.Dense> o, int n)
        => UnaryOp(i, o, n, new DelegateSpecialization<Func<float, float>>(SoftplusOp));
    public void Softsign(ArrayView1D<float, Stride1D.Dense> i, ArrayView1D<float, Stride1D.Dense> o, int n)
        => UnaryOp(i, o, n, new DelegateSpecialization<Func<float, float>>(SoftsignOp));
    public void Mish(ArrayView1D<float, Stride1D.Dense> i, ArrayView1D<float, Stride1D.Dense> o, int n)
        => UnaryOp(i, o, n, new DelegateSpecialization<Func<float, float>>(MishOp));
    public void IsInf(ArrayView1D<float, Stride1D.Dense> i, ArrayView1D<float, Stride1D.Dense> o, int n)
        => UnaryOp(i, o, n, new DelegateSpecialization<Func<float, float>>(IsInfOp));

    /// <summary>Validate GELU against CPU reference.</summary>
    public async Task ValidateGELUAsync(int count = 1000)
    {
        
        EnsureLoaded();
        var accelerator = _accelerator;

        var rng = new Random(42);
        var input = new float[count];
        for (int i = 0; i < count; i++) input[i] = (float)(rng.NextDouble() * 6 - 3);

        // CPU reference (erf-based GELU matching GPU implementation)
        var cpuOut = new float[count];
        for (int i = 0; i < count; i++)
        {
            float x = input[i];
            // erf-based GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
            double erfVal = Erf(x / Math.Sqrt(2.0));
            cpuOut[i] = (float)(0.5 * x * (1.0 + erfVal));
        }

        // Abramowitz & Stegun erf approximation (same as GPU kernel)
        static double Erf(double x)
        {
            double ax = Math.Abs(x);
            const double p = 0.3275911;
            const double a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741;
            const double a4 = -1.453152027, a5 = 1.061405429;
            double t = 1.0 / (1.0 + p * ax);
            double erfAbs = 1.0 - (a1 * t + a2 * t * t + a3 * t * t * t + a4 * t * t * t * t + a5 * t * t * t * t * t) * Math.Exp(-ax * ax);
            return x < 0 ? -erfAbs : erfAbs;
        }

        using var inputBuf = accelerator.Allocate1D(input);
        using var outputBuf = accelerator.Allocate1D<float>(count);
        GELU(inputBuf.View, outputBuf.View, count);
        await accelerator.SynchronizeAsync();
        var gpuOut = await outputBuf.CopyToHostAsync<float>(0, count);

        float maxErr = 0f;
        for (int i = 0; i < count; i++)
            maxErr = MathF.Max(maxErr, MathF.Abs(cpuOut[i] - gpuOut[i]));

        if (InferenceSession.VerboseLogging) Console.WriteLine($"[GELU] Validate {count} elements: maxErr={maxErr:E3}");
    }

    // ─────────────────────────────────────────────────────────────
    //  GPU-side verification (no large CPU readbacks)
    // ─────────────────────────────────────────────────────────────

    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>>? _compareReduceKernel;
    private MemoryBuffer1D<float, Stride1D.Dense>? _compareResultBuf;

    /// <summary>
    /// GPU kernel: compute |actual[i] - expected[i]|, atomically accumulate sum and max
    /// into results[0] (sum) and results[1] (max). Results buffer must be zeroed first.
    /// </summary>
    private static void CompareReduceImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> actual,
        ArrayView1D<float, Stride1D.Dense> expected,
        ArrayView1D<float, Stride1D.Dense> results)
    {
        float diff = actual[idx] - expected[idx];
        float absDiff = diff < 0f ? -diff : diff;
        Atomic.Add(ref results[0], absDiff);
        Atomic.Max(ref results[1], absDiff);
    }

    /// <summary>
    /// Compare two GPU buffers and return (meanError, maxError).
    /// Entire comparison runs on GPU — only 2 floats read back to CPU.
    /// </summary>
    public async Task<(float meanError, float maxError)> CompareOnGpuAsync(
        ArrayView1D<float, Stride1D.Dense> actual,
        ArrayView1D<float, Stride1D.Dense> expected,
        int count)
    {
        _compareReduceKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>>(CompareReduceImpl);

        // Allocate or reuse 2-element results buffer [sum, max], zero it
        _compareResultBuf ??= _accelerator.Allocate1D<float>(2);
        _compareResultBuf.CopyFromCPU(new float[] { 0f, 0f });

        _compareReduceKernel(count, actual, expected, _compareResultBuf.View);
        await _accelerator.SynchronizeAsync();

        var results = await _compareResultBuf.CopyToHostAsync<float>(0, 2);
        return (results[0] / count, results[1]);
    }

    public void Dispose()
    {
        _lastStridesBuf?.Dispose();
        _broadcastStridesBuf?.Dispose();
        foreach (var buf in _oldStridesBufs) buf.Dispose();
        _oldStridesBufs.Clear();
        _compareResultBuf?.Dispose();
        _nearestParamsBuf?.Dispose();
    }
}
