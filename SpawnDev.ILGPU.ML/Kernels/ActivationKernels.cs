using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Kernels;

/// <summary>
/// GPU activation function kernels. All use auto-grouped 1D dispatch.
/// In-place variants avoid extra buffer allocation; out-of-place variants write
/// to a separate output (preferred over CopyFrom + InPlace — one dispatch).
/// </summary>
public class ActivationKernels
{
    private readonly Accelerator _accelerator;

    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>>? _sigmoidInPlace;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>>? _tanhInPlace;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>>? _siluInPlace;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>>? _leakyReluInPlace;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>>? _clipInPlace;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>>? _hardSigmoidInPlace;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, float, float>? _hardSigmoidParamsInPlace;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>>? _hardSwishInPlace;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>>? _swiGLU;

    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>? _sigmoid;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>? _tanh;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>? _silu;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>? _hardSigmoid;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, float, float>? _hardSigmoidParams;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>? _hardSwish;

    public ActivationKernels(Accelerator accelerator) => _accelerator = accelerator;

    // ── Kernel implementations (in-place) ──

    private static void SigmoidInPlaceImpl(Index1D idx, ArrayView1D<float, Stride1D.Dense> data)
    {
        float x = data[idx];
        // Overflow-free logistic: e = exp(-|x|) is in (0, 1], so nothing overflows and no clamp is needed. The old
        // clamp sent x < -80 to 0 although sigmoid(-85) = 1.2e-37 is a normal float (Log(Sigmoid) -> -inf).
        float e = MathF.Exp(-MathF.Abs(x));
        float s = 1f / (1f + e);
        data[idx] = x >= 0f ? s : e * s;
    }

    private static void TanhInPlaceImpl(Index1D idx, ArrayView1D<float, Stride1D.Dense> data)
    {
        float x = data[idx];
        if (x > 40f) { data[idx] = 1f; return; }
        if (x < -40f) { data[idx] = -1f; return; }
        float e2x = MathF.Exp(2f * x);
        data[idx] = (e2x - 1f) / (e2x + 1f);
    }

    /// <summary>SiLU (Swish): x * sigmoid(x)</summary>
    private static void SiLUInPlaceImpl(Index1D idx, ArrayView1D<float, Stride1D.Dense> data)
    {
        float x = data[idx];
        // Overflow-free logistic (see SigmoidInPlaceImpl): no clamps.
        float e = MathF.Exp(-MathF.Abs(x));
        float s = 1f / (1f + e);
        data[idx] = x * (x >= 0f ? s : e * s);
    }

    /// <summary>Fused SwiGLU: out = (gate · sigmoid(gate)) · up — the SiLU-gated MLP activation in ONE pass,
    /// replacing the graph's Sigmoid + Mul + Mul (3 dispatches → 1; biggest on WebGPU dispatch overhead).
    /// BIT-IDENTICAL: the sigmoid clamps (&gt;80→1, &lt;-80→0) match <see cref="SigmoidInPlaceImpl"/> and the
    /// multiply order (gate·sig)·up matches the two-Mul graph. gate/up/out are distinct buffers (no aliasing).</summary>
    private static void SwiGLUImpl(Index1D idx, ArrayView1D<float, Stride1D.Dense> gate,
        ArrayView1D<float, Stride1D.Dense> up, ArrayView1D<float, Stride1D.Dense> outp)
    {
        float g = gate[idx];
        float sig = g > 80f ? 1f : (g < -80f ? 0f : 1f / (1f + MathF.Exp(-g)));
        outp[idx] = (g * sig) * up[idx];
    }

    /// <summary>LeakyReLU: max(alpha*x, x). Alpha passed via int buffer (as float bits).</summary>
    private static void LeakyReluInPlaceImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> data,
        ArrayView1D<int, Stride1D.Dense> p)
    {
        float alpha = Interop.FloatAsInt(p[0]); // reinterpret int bits as float
        float x = data[idx];
        data[idx] = x >= 0f ? x : alpha * x;
    }

    /// <summary>Clip: clamp(x, min, max). Min/max passed via int buffer (as float bits).</summary>
    private static void ClipInPlaceImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> data,
        ArrayView1D<int, Stride1D.Dense> p)
    {
        float minVal = Interop.FloatAsInt(p[0]);
        float maxVal = Interop.FloatAsInt(p[1]);
        float x = data[idx];
        if (x < minVal) data[idx] = minVal;
        else if (x > maxVal) data[idx] = maxVal;
    }

    /// <summary>HardSigmoid: max(0, min(1, alpha*x + beta)). Uses alpha=1/6, beta=0.5 (HardSwish default).</summary>
    private static void HardSigmoidInPlaceImpl(Index1D idx, ArrayView1D<float, Stride1D.Dense> data)
    {
        float x = data[idx];
        float y = x / 6f + 0.5f;
        if (y < 0f) y = 0f;
        if (y > 1f) y = 1f;
        data[idx] = y;
    }

    /// <summary>HardSigmoid with configurable alpha/beta: max(0, min(1, alpha*x + beta)).</summary>
    private static void HardSigmoidParamsInPlaceImpl(Index1D idx, ArrayView1D<float, Stride1D.Dense> data, float alpha, float beta)
    {
        float x = data[idx];
        float y = alpha * x + beta;
        if (y < 0f) y = 0f;
        if (y > 1f) y = 1f;
        data[idx] = y;
    }

    /// <summary>HardSwish: x * HardSigmoid(x)</summary>
    private static void HardSwishInPlaceImpl(Index1D idx, ArrayView1D<float, Stride1D.Dense> data)
    {
        float x = data[idx];
        float y = x / 6f + 0.5f;
        if (y < 0f) y = 0f;
        if (y > 1f) y = 1f;
        data[idx] = x * y;
    }

    // ── Kernel implementations (out-of-place) — same math as InPlace, separate src/dst ──

    private static void SigmoidImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> input, ArrayView1D<float, Stride1D.Dense> output)
    {
        float x = input[idx];
        // Overflow-free logistic: e = exp(-|x|) is in (0, 1], so nothing overflows and no clamp is needed. The old
        // clamp sent x < -80 to 0 although sigmoid(-85) = 1.2e-37 is a normal float (Log(Sigmoid) -> -inf).
        float e = MathF.Exp(-MathF.Abs(x));
        float s = 1f / (1f + e);
        output[idx] = x >= 0f ? s : e * s;
    }

    private static void TanhImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> input, ArrayView1D<float, Stride1D.Dense> output)
    {
        float x = input[idx];
        if (x > 40f) { output[idx] = 1f; return; }
        if (x < -40f) { output[idx] = -1f; return; }
        float e2x = MathF.Exp(2f * x);
        output[idx] = (e2x - 1f) / (e2x + 1f);
    }

    private static void SiLUImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> input, ArrayView1D<float, Stride1D.Dense> output)
    {
        float x = input[idx];
        // Overflow-free logistic (see SigmoidInPlaceImpl): no clamps.
        float e = MathF.Exp(-MathF.Abs(x));
        float s = 1f / (1f + e);
        output[idx] = x * (x >= 0f ? s : e * s);
    }

    private static void HardSigmoidImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> input, ArrayView1D<float, Stride1D.Dense> output)
    {
        float x = input[idx];
        float y = x / 6f + 0.5f;
        if (y < 0f) y = 0f;
        if (y > 1f) y = 1f;
        output[idx] = y;
    }

    private static void HardSigmoidParamsImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> input, ArrayView1D<float, Stride1D.Dense> output, float alpha, float beta)
    {
        float x = input[idx];
        float y = alpha * x + beta;
        if (y < 0f) y = 0f;
        if (y > 1f) y = 1f;
        output[idx] = y;
    }

    private static void HardSwishImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> input, ArrayView1D<float, Stride1D.Dense> output)
    {
        float x = input[idx];
        float y = x / 6f + 0.5f;
        if (y < 0f) y = 0f;
        if (y > 1f) y = 1f;
        output[idx] = x * y;
    }

    // ── Public API ──

    public void SigmoidInPlace(ArrayView1D<float, Stride1D.Dense> data, int count)
    { SigmoidInPlaceLazy(count, data); }

    public void TanhInPlace(ArrayView1D<float, Stride1D.Dense> data, int count)
    { TanhInPlaceLazy(count, data); }

    public void SiLUInPlace(ArrayView1D<float, Stride1D.Dense> data, int count)
    { SiluInPlace(count, data); }

    /// <summary>Fused SwiGLU: out = (gate · sigmoid(gate)) · up. Replaces Sigmoid + Mul + Mul (3 dispatches → 1).</summary>
    public void SwiGLU(ArrayView1D<float, Stride1D.Dense> gate, ArrayView1D<float, Stride1D.Dense> up,
        ArrayView1D<float, Stride1D.Dense> outp, int count)
    { SwiGLULazy(count, gate, up, outp); }

    public void HardSigmoidInPlace(ArrayView1D<float, Stride1D.Dense> data, int count)
    { HardSigmoidInPlaceLazy(count, data); }

    /// <summary>HardSigmoid with configurable alpha/beta: max(0, min(1, alpha*x + beta)).</summary>
    public void HardSigmoidInPlace(ArrayView1D<float, Stride1D.Dense> data, int count, float alpha, float beta)
    { HardSigmoidParamsInPlace(count, data, alpha, beta); }

    public void HardSwishInPlace(ArrayView1D<float, Stride1D.Dense> data, int count)
    { HardSwishInPlaceLazy(count, data); }

    /// <summary>Out-of-place Sigmoid: output[i] = sigmoid(input[i]). Prefer over CopyFrom + InPlace.</summary>
    public void Sigmoid(ArrayView1D<float, Stride1D.Dense> input, ArrayView1D<float, Stride1D.Dense> output, int count)
    { SigmoidLazy(count, input, output); }

    /// <summary>Out-of-place Tanh.</summary>
    public void Tanh(ArrayView1D<float, Stride1D.Dense> input, ArrayView1D<float, Stride1D.Dense> output, int count)
    { TanhLazy(count, input, output); }

    /// <summary>Out-of-place SiLU (Swish): output[i] = input[i] * sigmoid(input[i]).</summary>
    public void SiLU(ArrayView1D<float, Stride1D.Dense> input, ArrayView1D<float, Stride1D.Dense> output, int count)
    { Silu(count, input, output); }

    /// <summary>Out-of-place HardSigmoid with HardSwish defaults (alpha=1/6, beta=0.5).</summary>
    public void HardSigmoid(ArrayView1D<float, Stride1D.Dense> input, ArrayView1D<float, Stride1D.Dense> output, int count)
    { HardSigmoidLazy(count, input, output); }

    /// <summary>Out-of-place HardSigmoid with configurable alpha/beta.</summary>
    public void HardSigmoid(ArrayView1D<float, Stride1D.Dense> input, ArrayView1D<float, Stride1D.Dense> output, int count, float alpha, float beta)
    { HardSigmoidParams(count, input, output, alpha, beta); }

    /// <summary>Out-of-place HardSwish: output[i] = input[i] * HardSigmoid(input[i]).</summary>
    public void HardSwish(ArrayView1D<float, Stride1D.Dense> input, ArrayView1D<float, Stride1D.Dense> output, int count)
    { HardSwishLazy(count, input, output); }

    // Each kernel loads on its first use. They were all loaded on the first call of any one: MobileNetV2's first run compiled about 30 kernels it never runs
    // (every element-wise kernel for one ReLU, every normalization for one BatchNorm; ColdStart_MobileNetV2_ShaderCompile_SerialVsParallel, 2026-11-25).
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>> SigmoidInPlaceLazy => _sigmoidInPlace ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>>(SigmoidInPlaceImpl);
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>> TanhInPlaceLazy => _tanhInPlace ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>>(TanhInPlaceImpl);
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>> SiluInPlace => _siluInPlace ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>>(SiLUInPlaceImpl);
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>> HardSigmoidInPlaceLazy => _hardSigmoidInPlace ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>>(HardSigmoidInPlaceImpl);
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, float, float> HardSigmoidParamsInPlace => _hardSigmoidParamsInPlace ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, float, float>(HardSigmoidParamsInPlaceImpl);
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>> HardSwishInPlaceLazy => _hardSwishInPlace ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>>(HardSwishInPlaceImpl);
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>> SwiGLULazy => _swiGLU ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(SwiGLUImpl);
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>> SigmoidLazy => _sigmoid ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(SigmoidImpl);
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>> TanhLazy => _tanh ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(TanhImpl);
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>> Silu => _silu ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(SiLUImpl);
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>> HardSigmoidLazy => _hardSigmoid ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(HardSigmoidImpl);
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, float, float> HardSigmoidParams => _hardSigmoidParams ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, float, float>(HardSigmoidParamsImpl);
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>> HardSwishLazy => _hardSwish ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(HardSwishImpl);
}
