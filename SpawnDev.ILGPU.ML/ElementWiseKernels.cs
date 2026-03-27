using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML;

/// <summary>Binary operations for GPU broadcast kernels.</summary>
public enum BroadcastOp { Add, Sub, Mul, Div }

/// <summary>
/// Element-wise neural network operations: GELU, ReLU, Add, Mul, AddBias.
/// All use auto-grouped 1D kernels (no shared memory needed).
///
/// Future home: SpawnDev.ILGPU.ML
/// </summary>
public class ElementWiseKernels
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
    private MemoryBuffer1D<int, Stride1D.Dense>? _broadcastStridesBuf;
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
        using var stridesBuf = _accelerator.Allocate1D<int>(paramsSize);
        var paramsData = new int[paramsSize];
        paramsData[0] = rank;
        for (int i = 0; i < rank; i++) paramsData[1 + i] = aStrides[i];
        for (int i = 0; i < rank; i++) paramsData[1 + rank + i] = bStrides[i];
        for (int i = 0; i < rank; i++) paramsData[1 + 2 * rank + i] = outStrides[i];
        stridesBuf.View.SubView(0, paramsSize).CopyFromCPU(paramsData);

        var opSpec = op switch
        {
            BroadcastOp.Add => new DelegateSpecialization<Func<float, float, float>>(BroadcastAddOp),
            BroadcastOp.Sub => new DelegateSpecialization<Func<float, float, float>>(BroadcastSubOp),
            BroadcastOp.Mul => new DelegateSpecialization<Func<float, float, float>>(BroadcastMulOp),
            BroadcastOp.Div => new DelegateSpecialization<Func<float, float, float>>(BroadcastDivOp),
            _ => throw new ArgumentException($"Unsupported broadcast op: {op}")
        };
        _broadcastBinaryKernel!(outCount, a, b, output, stridesBuf.View, opSpec);
        // Synchronize to ensure the kernel reads strides before the buffer is disposed
        _accelerator.Synchronize();
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
    { output[idx] = MathF.Round(input[idx]); }

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

    /// <summary>Where: output[i] = condition[i] != 0 ? x[i] : y[i].</summary>
    private static void WhereImpl(Index1D idx, ArrayView1D<float, Stride1D.Dense> cond,
        ArrayView1D<float, Stride1D.Dense> x, ArrayView1D<float, Stride1D.Dense> y,
        ArrayView1D<float, Stride1D.Dense> output)
    { output[idx] = cond[idx] != 0f ? x[idx] : y[idx]; }

    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>? _sqrtKernel;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>? _sinKernel;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>? _cosKernel;
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

        Console.WriteLine($"[GELU] Validate {count} elements: maxErr={maxErr:E3}");
    }
}
