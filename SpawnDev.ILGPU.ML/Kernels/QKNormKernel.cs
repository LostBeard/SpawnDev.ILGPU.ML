using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Kernels;

/// <summary>
/// QK Normalization GPU kernel — L2-normalizes Q and K per head before attention.
/// Used by Depth Anything V3 and newer vision transformers.
/// Prevents attention logits from growing unbounded with sequence length.
///
/// For each head: Q_norm = Q / ||Q|| and K_norm = K / ||K||
/// </summary>
public class QKNormKernel
{
    private readonly Accelerator _accelerator;

    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, int, int>? _normKernel;

    public QKNormKernel(Accelerator accelerator) => _accelerator = accelerator;

    /// <summary>
    /// L2-normalize each vector (row) in-place.
    /// data [numVectors, dim] — each row normalized to unit length.
    /// </summary>
    public void NormalizeRows(
        ArrayView1D<float, Stride1D.Dense> data,
        ArrayView1D<float, Stride1D.Dense> output,
        int numVectors, int dim)
    {
        // One thread per output element — gather-only, WebGL TF compatible.
        _normKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            int, int>(NormalizeRowImpl);
        _normKernel(numVectors * dim, data, output, numVectors, dim);
    }

    /// <summary>
    /// One thread per output element. Each thread recomputes the L2 norm of its row
    /// (D reads) and produces a single scalar output. Replaces a scatter pattern where
    /// one thread per vector wrote D outputs — silently dropped by WebGL Transform
    /// Feedback. Same arithmetic intensity, more parallelism.
    /// </summary>
    private static void NormalizeRowImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> data,
        ArrayView1D<float, Stride1D.Dense> output,
        int N, int D)
    {
        int vecIdx = idx / D;
        int offset = vecIdx * D;
        float sumSq = 0f;
        for (int i = 0; i < D; i++)
            sumSq += data[offset + i] * data[offset + i];
        float invNorm = 1f / MathF.Sqrt(sumSq + 1e-12f);
        output[idx] = data[idx] * invNorm;
    }
}
