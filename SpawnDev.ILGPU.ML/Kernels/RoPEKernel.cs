using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Kernels;

/// <summary>
/// Rotary Position Embedding (RoPE) GPU kernel.
/// Applies rotation-based position encoding to query and key tensors.
/// Used by LLaMA, Mistral, gemma, GPT-J/NeoX-family and modern transformers.
///
/// Two pairing STYLES exist in the wild and produce different layouts:
/// - NeoX / split-half (LLaMA, Mistral, gemma; the default here):
///     pair = (x[i], x[i + rotaryDim/2]) for i in [0, rotaryDim/2)
/// - GPT-J / interleaved:
///     pair = (x[2i], x[2i+1]) for i in [0, rotaryDim/2)
/// Both rotate each pair by θ_i = position / base^(2i / rotaryDim):
///     x0' = x0·cos θ − x1·sin θ ; x1' = x0·sin θ + x1·cos θ
///
/// PER-CALL parameters (gemma4 needs them per LAYER - 5:1 SWA/global interleave uses
/// base 10000 on local layers and 1000000 on global layers; the graph wiring selects
/// and passes them, this kernel just honors the arguments):
/// - ropeBase: the θ base for this call (ctor value is only the default).
/// - rotaryDim: rotate only the first rotaryDim dims of each head; pass-through the
///   rest (partial rotary, e.g. GPT-NeoX 25%). Must be even; rotaryDim == headDim
///   is the usual full rotation.
/// - interleaved: pairing style (false = NeoX split-half, matching the previous
///   behavior of this kernel; its earlier doc described the interleaved form while
///   implementing split-half - the flag makes both real and the docs honest).
///
/// Key property: dot(RoPE(q,pos_q), RoPE(k,pos_k)) depends only on (pos_q − pos_k),
/// giving relative position awareness without explicit position embeddings.
/// </summary>
public class RoPEKernel : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly float _base;

    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, int, int, int, float, int, int, int,
        ArrayView1D<float, Stride1D.Dense>, int>? _ropeKernel;

    // Shared 1-element {1f} buffer bound as freq_factors when none are provided; the
    // kernel reads index i * hasFactors, so the absent case always reads this 1.0
    // (branch-free - no kernel variant, no in-kernel branching).
    private MemoryBuffer1D<float, Stride1D.Dense>? _onesBuf;

    public RoPEKernel(Accelerator accelerator, float ropeBase = 10000f)
    {
        _accelerator = accelerator;
        _base = ropeBase;
    }

    /// <summary>
    /// Apply RoPE with the constructor base, full head dim, NeoX style - the original
    /// API, behavior unchanged.
    /// input [numPositions, headDim] → output [numPositions, headDim];
    /// positions are startPosition, startPosition+1, ...
    /// </summary>
    public void Apply(
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output,
        int numPositions, int headDim, int startPosition = 0) =>
        Apply(input, output, numPositions, headDim, startPosition,
            _base, headDim, interleaved: false);

    /// <summary>
    /// Apply RoPE with per-call base / partial rotary / pairing style (see class doc).
    /// </summary>
    public void Apply(
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output,
        int numPositions, int headDim, int startPosition,
        float ropeBase, int rotaryDim, bool interleaved) =>
        Apply(input, output, numPositions, headDim, startPosition,
            ropeBase, rotaryDim, interleaved, rowsPerPosition: 1);

    /// <summary>
    /// Apply RoPE where each SEQUENCE position spans several consecutive [headDim] rows -
    /// the multi-head pre-transpose layout [seq, heads, headDim] passes
    /// rowsPerPosition = heads, so all of a position's heads rotate by that position's
    /// angle (sequence position = rowIndex / rowsPerPosition + startPosition).
    /// numPositions counts ROWS here (seq * heads), matching the input's row count.
    /// </summary>
    public void Apply(
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output,
        int numPositions, int headDim, int startPosition,
        float ropeBase, int rotaryDim, bool interleaved, int rowsPerPosition) =>
        Apply(input, output, numPositions, headDim, startPosition,
            ropeBase, rotaryDim, interleaved, rowsPerPosition, freqFactors: null);

    /// <summary>
    /// Apply RoPE with optional per-pair FREQUENCY FACTORS (NTK / proportional rope):
    /// <paramref name="freqFactors"/> has rotaryDim/2 entries and the pair angle becomes
    /// θ_i = position · base^(−2i/rotaryDim) / freqFactors[i] - ggml semantics
    /// (theta_base is DIVIDED by the factor; verified verbatim against llama.cpp's rope
    /// kernels, `freq_factors[i0/2]`, 2026-06-11). gemma4 global layers carry
    /// rope_freqs.weight; sliding layers pass null (= all-ones behavior).
    /// </summary>
    public void Apply(
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output,
        int numPositions, int headDim, int startPosition,
        float ropeBase, int rotaryDim, bool interleaved, int rowsPerPosition,
        ArrayView1D<float, Stride1D.Dense>? freqFactors)
    {
        if (rotaryDim <= 0 || rotaryDim > headDim || (rotaryDim & 1) != 0)
            throw new ArgumentOutOfRangeException(nameof(rotaryDim),
                $"rotaryDim must be even and in (0, headDim]; got {rotaryDim} for headDim {headDim}");
        if (rowsPerPosition <= 0)
            throw new ArgumentOutOfRangeException(nameof(rowsPerPosition));
        if (freqFactors.HasValue && freqFactors.Value.Length < rotaryDim / 2)
            throw new ArgumentOutOfRangeException(nameof(freqFactors),
                $"freqFactors must have rotaryDim/2 = {rotaryDim / 2} entries; got {freqFactors.Value.Length}");

        if (_onesBuf == null)
        {
            _onesBuf = _accelerator.Allocate1D(new[] { 1f });
        }
        var ffView = freqFactors ?? _onesBuf.View;
        int hasFactors = freqFactors.HasValue ? 1 : 0;

        // One thread per scalar output (gather, not scatter — WebGL TF compatible).
        _ropeKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            int, int, int, float, int, int, int,
            ArrayView1D<float, Stride1D.Dense>, int>(RoPEImpl);
        _ropeKernel(numPositions * headDim, input, output,
            numPositions, headDim, startPosition, ropeBase, rotaryDim,
            interleaved ? 1 : 0, rowsPerPosition, ffView, hasFactors);
    }

    public void Dispose()
    {
        _onesBuf?.Dispose();
        _onesBuf = null;
    }

    /// <summary>Apply RoPE in-place (original API).</summary>
    public void ApplyInPlace(
        ArrayView1D<float, Stride1D.Dense> data,
        int numPositions, int headDim, int startPosition = 0) =>
        Apply(data, data, numPositions, headDim, startPosition);

    /// <summary>Apply RoPE in-place with per-call parameters.</summary>
    public void ApplyInPlace(
        ArrayView1D<float, Stride1D.Dense> data,
        int numPositions, int headDim, int startPosition,
        float ropeBase, int rotaryDim, bool interleaved) =>
        Apply(data, data, numPositions, headDim, startPosition, ropeBase, rotaryDim, interleaved);

    private static void RoPEImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output,
        int numPos, int D, int startPos, float ropeBase, int rotDim, int interleaved,
        int rowsPerPos, ArrayView1D<float, Stride1D.Dense> freqFactors, int hasFactors)
    {
        // idx = (row * D + k). Branch shape: the pass-through/rotate split and the
        // style selects below are TOP-LEVEL (not inside any loop) - safe for the
        // WebGL emitter (cf. the loop-body branch explosion documented on
        // FusedDequantMatMul; this kernel has no loops at all).
        int k = idx % D;
        int pos = (idx / D) / rowsPerPos + startPos;
        int rowStart = (idx / D) * D;

        if (k >= rotDim)
        {
            // Partial rotary: dims beyond rotaryDim pass through unchanged.
            output[idx] = input[idx];
            return;
        }

        int half = rotDim / 2;

        // Pair geometry per style:
        // NeoX (interleaved=0): lane i = k % half; x0 at i, x1 at i + half; k >= half is the second element.
        // GPT-J (interleaved=1): lane i = k / 2;   x0 at 2i, x1 at 2i+1;   odd k is the second element.
        int laneNeoX = k >= half ? k - half : k;
        int laneJ = k >> 1;
        int i = interleaved == 1 ? laneJ : laneNeoX;
        int x0Idx = interleaved == 1 ? 2 * laneJ : laneNeoX;
        int x1Idx = interleaved == 1 ? 2 * laneJ + 1 : laneNeoX + half;
        bool second = interleaved == 1 ? (k & 1) == 1 : k >= half;

        // θ = pos / (base^(2i / rotaryDim) · freqFactor_i) - ggml DIVIDES theta by the
        // per-pair factor (NTK / proportional rope). Absent factors: hasFactors = 0
        // makes every lane read freqFactors[0] = 1.0 from the shared ones-buffer
        // (index arithmetic, no branch).
        float freqExp = 2f * i / (float)rotDim;
        float invFreq = 1f / MathF.Pow(ropeBase, freqExp);
        float ff = freqFactors[i * hasFactors];
        float theta = pos * invFreq / ff;
        float cosTheta = MathF.Cos(theta);
        float sinTheta = MathF.Sin(theta);

        float x0 = input[rowStart + x0Idx];
        float x1 = input[rowStart + x1Idx];
        output[idx] = second ? (x0 * sinTheta + x1 * cosTheta)
                             : (x0 * cosTheta - x1 * sinTheta);
    }
}
