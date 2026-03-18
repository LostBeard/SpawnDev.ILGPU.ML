using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML;

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

    // ─────────────────────────────────────────────────────────────
    //  Additional element-wise ops (Sqrt, Exp, Div, Pow, Abs, Neg, Reciprocal, Erf)
    // ─────────────────────────────────────────────────────────────

    private static void SqrtImpl(Index1D idx, ArrayView1D<float, Stride1D.Dense> input, ArrayView1D<float, Stride1D.Dense> output)
    { output[idx] = MathF.Sqrt(input[idx]); }

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

    private static void MinImpl(Index1D idx, ArrayView1D<float, Stride1D.Dense> a, ArrayView1D<float, Stride1D.Dense> b, ArrayView1D<float, Stride1D.Dense> output)
    { output[idx] = a[idx] < b[idx] ? a[idx] : b[idx]; }

    private static void MaxImpl(Index1D idx, ArrayView1D<float, Stride1D.Dense> a, ArrayView1D<float, Stride1D.Dense> b, ArrayView1D<float, Stride1D.Dense> output)
    { output[idx] = a[idx] > b[idx] ? a[idx] : b[idx]; }

    /// <summary>Where: output[i] = condition[i] != 0 ? x[i] : y[i].</summary>
    private static void WhereImpl(Index1D idx, ArrayView1D<float, Stride1D.Dense> cond,
        ArrayView1D<float, Stride1D.Dense> x, ArrayView1D<float, Stride1D.Dense> y,
        ArrayView1D<float, Stride1D.Dense> output)
    { output[idx] = cond[idx] != 0f ? x[idx] : y[idx]; }

    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>? _sqrtKernel;
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

    public void Sqrt(ArrayView1D<float, Stride1D.Dense> input, ArrayView1D<float, Stride1D.Dense> output, int count)
    { EnsureLoaded2(); _sqrtKernel!(count, input, output); }
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

    private void EnsureLoaded2()
    {
        var a = _accelerator;
        _sqrtKernel ??= a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(SqrtImpl);
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
