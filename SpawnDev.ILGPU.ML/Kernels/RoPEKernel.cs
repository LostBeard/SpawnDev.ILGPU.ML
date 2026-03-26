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
        int halfDim = headDim / 2;
        _ropeKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            int, int, int, float>(RoPEImpl);
        _ropeKernel(numPositions * halfDim, input, output,
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
        int halfD = D / 2;
        int pos = idx / halfD + startPos;
        int dimPair = idx % halfD;

        // Frequency: θ = pos / base^(2*dimPair/D)
        float freqExp = 2f * dimPair / (float)D;
        float invFreq = 1f / MathF.Pow(ropeBase, freqExp);
        float theta = pos * invFreq;

        float cosTheta = MathF.Cos(theta);
        float sinTheta = MathF.Sin(theta);

        int baseIdx = (idx / halfD) * D; // row start
        int i = dimPair;

        float x0 = input[baseIdx + 2 * i];
        float x1 = input[baseIdx + 2 * i + 1];

        output[baseIdx + 2 * i]     = x0 * cosTheta - x1 * sinTheta;
        output[baseIdx + 2 * i + 1] = x0 * sinTheta + x1 * cosTheta;
    }
}
