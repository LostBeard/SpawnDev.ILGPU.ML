using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Kernels;

/// <summary>
/// GPU activation function kernels. All use auto-grouped 1D dispatch.
/// In-place variants avoid extra buffer allocation.
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

    public ActivationKernels(Accelerator accelerator) => _accelerator = accelerator;

    // ── Kernel implementations ──

    private static void SigmoidInPlaceImpl(Index1D idx, ArrayView1D<float, Stride1D.Dense> data)
    {
        float x = data[idx];
        if (x > 80f) { data[idx] = 1f; return; }
        if (x < -80f) { data[idx] = 0f; return; }
        data[idx] = 1f / (1f + MathF.Exp(-x));
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
        if (x > 80f) { return; } // sigmoid(x) ≈ 1, so SiLU(x) ≈ x
        if (x < -80f) { data[idx] = 0f; return; }
        float sig = 1f / (1f + MathF.Exp(-x));
        data[idx] = x * sig;
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

    // ── Public API ──

    public void SigmoidInPlace(ArrayView1D<float, Stride1D.Dense> data, int count)
    { EnsureLoaded(); _sigmoidInPlace!(count, data); }

    public void TanhInPlace(ArrayView1D<float, Stride1D.Dense> data, int count)
    { EnsureLoaded(); _tanhInPlace!(count, data); }

    public void SiLUInPlace(ArrayView1D<float, Stride1D.Dense> data, int count)
    { EnsureLoaded(); _siluInPlace!(count, data); }

    /// <summary>Fused SwiGLU: out = (gate · sigmoid(gate)) · up. Replaces Sigmoid + Mul + Mul (3 dispatches → 1).</summary>
    public void SwiGLU(ArrayView1D<float, Stride1D.Dense> gate, ArrayView1D<float, Stride1D.Dense> up,
        ArrayView1D<float, Stride1D.Dense> outp, int count)
    { EnsureLoaded(); _swiGLU!(count, gate, up, outp); }

    public void HardSigmoidInPlace(ArrayView1D<float, Stride1D.Dense> data, int count)
    { EnsureLoaded(); _hardSigmoidInPlace!(count, data); }

    /// <summary>HardSigmoid with configurable alpha/beta: max(0, min(1, alpha*x + beta)).</summary>
    public void HardSigmoidInPlace(ArrayView1D<float, Stride1D.Dense> data, int count, float alpha, float beta)
    { EnsureLoaded(); _hardSigmoidParamsInPlace!(count, data, alpha, beta); }

    public void HardSwishInPlace(ArrayView1D<float, Stride1D.Dense> data, int count)
    { EnsureLoaded(); _hardSwishInPlace!(count, data); }

    private void EnsureLoaded()
    {
        var a = _accelerator;
        _sigmoidInPlace ??= a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>>(SigmoidInPlaceImpl);
        _tanhInPlace ??= a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>>(TanhInPlaceImpl);
        _siluInPlace ??= a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>>(SiLUInPlaceImpl);
        _hardSigmoidInPlace ??= a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>>(HardSigmoidInPlaceImpl);
        _hardSigmoidParamsInPlace ??= a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, float, float>(HardSigmoidParamsInPlaceImpl);
        _hardSwishInPlace ??= a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>>(HardSwishInPlaceImpl);
        _swiGLU ??= a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(SwiGLUImpl);
    }
}
