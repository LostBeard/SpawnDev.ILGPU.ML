using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Kernels;

/// <summary>
/// Spatial Memory Unit (SMU) GPU kernel for AsyncMDE.
/// Combines a fast-path depth estimate with a cached slow-path memory
/// using learned per-pixel trust factors.
///
/// O = T * M + (1-T) * F
/// where T = trust factor (sigmoid), M = memory (slow path cache), F = fast path output
///
/// Enables real-time depth estimation by decoupling the fast inference path
/// from the slow but accurate depth model.
/// </summary>
public class SpatialMemoryKernel
{
    private readonly Accelerator _accelerator;

    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>>? _combineKernel;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, float>? _emaKernel;

    public SpatialMemoryKernel(Accelerator accelerator) => _accelerator = accelerator;

    /// <summary>
    /// Convex combination: output = trust * memory + (1 - trust) * fastPath.
    /// All arrays are [H, W] spatial maps.
    /// </summary>
    public void Combine(
        ArrayView1D<float, Stride1D.Dense> trust,
        ArrayView1D<float, Stride1D.Dense> memory,
        ArrayView1D<float, Stride1D.Dense> fastPath,
        ArrayView1D<float, Stride1D.Dense> output,
        int count)
    {
        _combineKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(CombineImpl);
        _combineKernel(count, trust, memory, fastPath, output);
    }

    private static void CombineImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> trust,
        ArrayView1D<float, Stride1D.Dense> memory,
        ArrayView1D<float, Stride1D.Dense> fastPath,
        ArrayView1D<float, Stride1D.Dense> output)
    {
        float t = trust[idx];
        output[idx] = t * memory[idx] + (1f - t) * fastPath[idx];
    }

    /// <summary>
    /// Exponential moving average update for memory cache.
    /// memory = beta * memory + (1 - beta) * newValue
    /// </summary>
    public void EMAUpdate(
        ArrayView1D<float, Stride1D.Dense> memory,
        ArrayView1D<float, Stride1D.Dense> newValue,
        int count, float beta = 0.9f)
    {
        _emaKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            float>(EMAImpl);
        _emaKernel(count, memory, newValue, beta);
    }

    private static void EMAImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> memory,
        ArrayView1D<float, Stride1D.Dense> newValue,
        float beta)
    {
        memory[idx] = beta * memory[idx] + (1f - beta) * newValue[idx];
    }
}
