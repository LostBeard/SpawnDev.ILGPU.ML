using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.WebGPU;

namespace SpawnDev.ILGPU.ML;

/// <summary>
/// Element-wise neural network operations: GELU, ReLU, Add, Mul, AddBias.
/// All use auto-grouped 1D kernels (no shared memory needed).
///
/// Future home: SpawnDev.ILGPU.ML
/// </summary>
public class ElementWiseKernels
{
    private readonly WebGPUAccelerator _accelerator;

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
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>? _addInPlaceKernel;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int>? _concatLastDimKernel;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int, int, int, int>? _bilinearUpsampleKernel;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int, int, int, int>? _bilinearUpsampleACKernel;

    public ElementWiseKernels(WebGPUAccelerator accelerator) => _accelerator = accelerator;

    // ─────────────────────────────────────────────────────────────
    //  Kernel implementations
    // ─────────────────────────────────────────────────────────────

    /// <summary>
    /// GELU activation (fast approximation):
    /// y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    /// </summary>
    private static void GELUImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output)
    {
        float x = input[idx];
        if (x > 10f) { output[idx] = x; return; }
        if (x < -10f) { output[idx] = 0f; return; }
        const float sqrt2pi = 0.7978845608f;
        float inner = sqrt2pi * (x + 0.044715f * x * x * x);
        float clamped = inner * 2f;
        if (clamped > 80f) clamped = 80f;
        if (clamped < -80f) clamped = -80f;
        float exp2x = MathF.Exp(clamped);
        float tanh = (exp2x - 1f) / (exp2x + 1f);
        output[idx] = 0.5f * x * (1f + tanh);
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

    /// <summary>In-place GELU: data[i] = gelu(data[i]). Single binding. Clamped to prevent exp overflow.</summary>
    private static void GELUInPlaceImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> data)
    {
        float x = data[idx];
        // For large |x|, GELU(x) ≈ x (positive) or 0 (negative)
        if (x > 10f) { return; } // data[idx] already = x ≈ GELU(x)
        if (x < -10f) { data[idx] = 0f; return; }
        const float sqrt2pi = 0.7978845608f;
        float inner = sqrt2pi * (x + 0.044715f * x * x * x);
        float clamped = inner * 2f;
        if (clamped > 80f) clamped = 80f; // prevent exp overflow
        if (clamped < -80f) clamped = -80f;
        float exp2x = MathF.Exp(clamped);
        float tanh = (exp2x - 1f) / (exp2x + 1f);
        data[idx] = 0.5f * x * (1f + tanh);
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

        int y0 = (int)fy; int y1 = y0 + 1;
        int x0 = (int)fx; int x1 = x0 + 1;
        float ty = fy - y0; float tx = fx - x0;
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

        int y0 = (int)fy; int y1 = y0 + 1;
        int x0 = (int)fx; int x1 = x0 + 1;
        float ty = fy - y0; float tx = fx - x0;
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

        // CPU reference (same fast GELU approximation)
        var cpuOut = new float[count];
        for (int i = 0; i < count; i++)
        {
            float x = input[i];
            float inner = 0.7978845608f * (x + 0.044715f * x * x * x);
            float exp2x = MathF.Exp(2f * inner);
            float tanh = (exp2x - 1f) / (exp2x + 1f);
            cpuOut[i] = 0.5f * x * (1f + tanh);
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
