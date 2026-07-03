using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Kernels;

/// <summary>
/// Parallel GPU argmax over a single large vector (the LLM logits row, vocab ~262K) that reads back ONLY the
/// index — replacing the per-token full-vocab GPU→host readback + CPU argmax in greedy decode. Greedy
/// next-token selection doesn't need ~1 MB of logits on the host every token; it needs one int.
///
/// Strategy (NO shared memory / barriers, so it runs on every backend including WebGL): P threads each scan a
/// strided slice of the input and emit a partial (maxValue, firstMaxIndex); the host combines the P partials
/// (P ~ 1024 — a trivial loop) instead of scanning the whole vocab. Per token this drops the transfer from
/// ~vocab*4 bytes to ~P*8 bytes and the host scan from vocab to P (the latter matters most in WASM, where a
/// 262K single-threaded argmax per token is real work). The partial buffers are allocated once and reused —
/// no per-token GPU allocation.
///
/// Tie-break = LOWEST index, matching <see cref="Preprocessing.TextGenerationSampler.Greedy"/>'s first-max-wins,
/// so greedy tokens are identical to the CPU path. Indices travel as float (exact for vocab &lt; 2^24), the same
/// representation the existing <c>ElementWiseKernels.ArgMax</c> uses.
/// </summary>
public sealed class GpuArgMax : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly int _numPartials;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        int, int>? _kernel;
    // Interleaved [val0, idx0, val1, idx1, ...] — ONE output buffer (no WebGPU same-buffer aliasing) read back
    // in ONE GPU→host round-trip (browser readback cost is latency-bound, so one round-trip beats two).
    private MemoryBuffer1D<float, Stride1D.Dense>? _partials;

    public GpuArgMax(Accelerator accelerator, int numPartials = 1024)
    {
        _accelerator = accelerator;
        _numPartials = Math.Max(1, numPartials);
    }

    // Each of P threads scans input[p], input[p+P], input[p+2P], ... keeping the FIRST (lowest-index) max in
    // its stride; emits that partial as an interleaved (value, index) pair. No cross-thread communication →
    // no shared memory (runs on WebGL too).
    private static void PartialArgMaxImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> outPairs,
        int n, int numPartials)
    {
        int p = idx;
        if (p >= numPartials) return;
        float best = float.NegativeInfinity;
        int bestIdx = p < n ? p : 0;
        for (int i = p; i < n; i += numPartials)
        {
            float v = input[i];
            if (v > best) { best = v; bestIdx = i; }
        }
        outPairs[2 * p] = best;
        outPairs[2 * p + 1] = bestIdx;
    }

    /// <summary>Index of the maximum element of <paramref name="logits"/>[0..n), lowest index on ties (matches
    /// CPU greedy). Reads back only the P partial pairs (one round-trip), never the whole vector.</summary>
    public async Task<int> ArgMaxAsync(ArrayView1D<float, Stride1D.Dense> logits, int n)
    {
        if (n <= 0) return 0;
        int P = Math.Min(_numPartials, n);
        _kernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int>(PartialArgMaxImpl);
        _partials ??= _accelerator.Allocate1D<float>(2 * _numPartials);

        _kernel(P, logits, _partials.View, n, P);
        await _accelerator.SynchronizeAsync();
        var host = await _partials.CopyToHostAsync<float>(0, 2 * P);

        float best = float.NegativeInfinity;
        int bestIdx = 0;
        for (int p = 0; p < P; p++)
        {
            float v = host[2 * p];
            int pi = (int)host[2 * p + 1];
            if (v > best || (v == best && pi < bestIdx)) { best = v; bestIdx = pi; }
        }
        return bestIdx;
    }

    /// <summary>
    /// Dispatch ONLY the partial-argmax kernel (no sync, no readback) - used by the decode
    /// capture/replay driver to record the argmax INTO the captured dispatch plan, so a replayed
    /// token needs no separate argmax dispatch. Pair with <see cref="ReadPartialsAsync"/>.
    /// </summary>
    public void DispatchPartials(ArrayView1D<float, Stride1D.Dense> logits, int n)
    {
        if (n <= 0) return;
        int P = Math.Min(_numPartials, n);
        _kernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int>(PartialArgMaxImpl);
        _partials ??= _accelerator.Allocate1D<float>(2 * _numPartials);
        _kernel(P, logits, _partials.View, n, P);
    }

    /// <summary>
    /// Read the partial pairs and reduce to the argmax index WITHOUT a separate SynchronizeAsync -
    /// the readback's own mapAsync fence waits for all previously queued work (the replayed plan +
    /// the in-plan partial kernel), making it the decode loop's ONLY per-token GPU round-trip.
    /// </summary>
    public async Task<int> ReadPartialsAsync(int n)
    {
        int P = Math.Min(_numPartials, n);
        var host = await _partials!.CopyToHostAsync<float>(0, 2 * P);
        float best = float.NegativeInfinity;
        int bestIdx = 0;
        for (int p = 0; p < P; p++)
        {
            float v = host[2 * p];
            int pi = (int)host[2 * p + 1];
            if (v > best || (v == best && pi < bestIdx)) { best = v; bestIdx = pi; }
        }
        return bestIdx;
    }

    public void Dispose()
    {
        _partials?.Dispose();
        _partials = null;
    }
}
