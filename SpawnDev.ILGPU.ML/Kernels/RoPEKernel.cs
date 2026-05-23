using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Kernels;

/// <summary>
/// Rotary Position Embedding (RoPE) GPU kernel.
/// Applies rotation-based position encoding to query and key tensors.
/// Used by Depth Anything V3, LLaMA, Mistral, and modern transformers.
///
/// RoPE encodes position by rotating pairs of dimensions:
///   x'[2i]   = x[2i]   * cos(θ) - x[2i+1] * sin(θ)
///   x'[2i+1] = x[2i]   * sin(θ) + x[2i+1] * cos(θ)
/// where θ = position / base^(2i/d), base=10000 by default.
///
/// Key property: dot(RoPE(q,pos_q), RoPE(k,pos_k)) depends only on (pos_q - pos_k),
/// giving relative position awareness without explicit position embeddings.
/// </summary>
public class RoPEKernel
{
    private readonly Accelerator _accelerator;
    private readonly float _base;

    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, int, int, int, float>? _ropeKernel;

    public RoPEKernel(Accelerator accelerator, float ropeBase = 10000f)
    {
        _accelerator = accelerator;
        _base = ropeBase;
    }

    /// <summary>
    /// Apply RoPE to a batch of vectors.
    /// input [numPositions, headDim] → output [numPositions, headDim]
    /// Positions are assumed sequential: 0, 1, 2, ..., numPositions-1.
    /// </summary>
    public void Apply(
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output,
        int numPositions, int headDim, int startPosition = 0)
    {
        // One thread per scalar output (gather, not scatter — WebGL TF compatible).
        // idx = pos * headDim + k. Doubles the thread count vs the prior pair-per-thread
        // launch, but each thread now writes exactly one position (its own idx).
        _ropeKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            int, int, int, float>(RoPEImpl);
        _ropeKernel(numPositions * headDim, input, output,
            numPositions, headDim, startPosition, _base);
    }

    /// <summary>
    /// Apply RoPE in-place.
    /// </summary>
    public void ApplyInPlace(
        ArrayView1D<float, Stride1D.Dense> data,
        int numPositions, int headDim, int startPosition = 0)
    {
        Apply(data, data, numPositions, headDim, startPosition);
    }

    private static void RoPEImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output,
        int numPos, int D, int startPos, float ropeBase)
    {
        // idx = (pos * D + k)
        int halfD = D / 2;
        int k = idx % D;
        int pos = idx / D + startPos;
        // Both halves share the same dimIdx in [0, halfD); first-half outputs cos-sin,
        // second-half outputs sin+cos. The two inputs needed are always at k_low and k_low+halfD.
        bool secondHalf = k >= halfD;
        int dimIdx = secondHalf ? k - halfD : k;

        // Frequency: θ = pos / base^(2*dimIdx/D)
        float freqExp = 2f * dimIdx / (float)D;
        float invFreq = 1f / MathF.Pow(ropeBase, freqExp);
        float theta = pos * invFreq;
        float cosTheta = MathF.Cos(theta);
        float sinTheta = MathF.Sin(theta);

        int rowStart = (idx / D) * D;
        float x0 = input[rowStart + dimIdx];          // first half
        float x1 = input[rowStart + dimIdx + halfD];  // second half
        output[idx] = secondHalf ? (x0 * sinTheta + x1 * cosTheta)
                                 : (x0 * cosTheta - x1 * sinTheta);
    }
}
