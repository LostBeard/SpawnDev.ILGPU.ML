using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Kernels;

using FV = ArrayView1D<float, Stride1D.Dense>;
using IV = ArrayView1D<int, Stride1D.Dense>;

/// <summary>
/// One token from a row of logits on the device, as <see cref="Preprocessing.TextGenerationSampler"/> draws it on the host (greedy, top-k, top-p, the
/// repetition penalty first), so a generation step reads back one int instead of the vocabulary: the penalty over the given window (each distinct token
/// once), the row's exact top-<see cref="TopKKernels.K"/> and its softmax mass at the temperature (<see cref="TopKKernels"/>), then the draw with the
/// caller's uniform (the host keeps its <see cref="Random"/>, so a seeded generation stays reproducible).
/// </summary>
/// <remarks>
/// Same candidates as the host (the tokens within 20 x T of the best), same nucleus (best first until the cumulative mass reaches p), same draw (the first
/// token whose running mass reaches u). Ties go to the lower id (the host's Array.Sort leaves them in no particular order). When the candidates cannot
/// answer - a nucleus wider than <see cref="TopKKernels.K"/>, top-k with k above it, more tokens at the floor than the gather holds -
/// <see cref="SampleAsync"/> returns -1 and the caller samples the row on the host. Not on WebGL (the gather takes atomic slots).
/// </remarks>
public sealed class DeviceSampler : IDisposable
{
    private readonly Accelerator _acc;
    private int _vocab, _nch, _S;
    private const int Chunk = 256;
    // the staged step: [T, topP, u, topK, penalty, window length, 0, 0, the window as floats...] (ids are exact as floats below 2^24)
    private const int Header = 8;
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _params;
    private MemoryBuffer1D<float, Stride1D.Dense>? _maxV, _selV, _cand;
    private MemoryBuffer1D<int, Stride1D.Dense>? _maxI, _selI, _selN;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _out;
    private readonly Action<Index1D, FV, FV> _penalize;
    private readonly Action<Index1D, FV, FV, IV, int, int, int> _chunkMax;
    private readonly Action<Index1D, FV, IV, FV, IV, int, int> _floor;
    private readonly Action<Index1D, FV, FV, IV, FV, IV, int, int, int, float, int> _select;
    private readonly Action<Index1D, FV, IV, IV, FV, int> _final;
    private readonly Action<Index1D, FV, FV, IV, int, int> _draw;
    private bool _staged;
    private Strategy _stagedStrategy;
    private float _stagedT;
    private int _stagedWindow;

    /// <summary>True when <paramref name="accelerator"/> can run it (not WebGL: the candidate gather takes atomic slots).</summary>
    public static bool Supports(Accelerator accelerator) => accelerator.AcceleratorType != AcceleratorType.WebGL;

    /// <summary>Most tokens of a penalty window.</summary>
    public const int MaxWindow = 1024;

    public DeviceSampler(Accelerator accelerator)
    {
        if (!Supports(accelerator)) throw new NotSupportedException("DeviceSampler: not on WebGL (the candidate gather takes atomic slots)");
        _acc = accelerator;
        _params = _acc.Allocate1D<float>(Header + MaxWindow);
        _out = _acc.Allocate1D<int>(64);
        _penalize = _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV>(PenalizeImpl);
        _chunkMax = _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, IV, int, int, int>(TopKKernels.ChunkMaxImpl);
        _floor = _acc.LoadAutoGroupedStreamKernel<Index1D, FV, IV, FV, IV, int, int>(TopKKernels.FloorImpl);
        _select = _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, IV, FV, IV, int, int, int, float, int>(TopKKernels.SelectImpl);
        _final = _acc.LoadAutoGroupedStreamKernel<Index1D, FV, IV, IV, FV, int>(TopKKernels.FinalImpl);
        _draw = _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, IV, int, int>(DrawImpl);
    }

    private void EnsureVocab(int vocab)
    {
        if (vocab == _vocab) return;
        if (vocab >= 1 << 24) throw new NotSupportedException($"vocabulary {vocab}: candidate ids travel as floats (exact below 2^24)");
        _maxV?.Dispose(); _maxI?.Dispose(); _selV?.Dispose(); _selI?.Dispose(); _selN?.Dispose(); _cand?.Dispose();
        _vocab = vocab; _nch = (vocab + Chunk - 1) / Chunk; _S = 2 * TopKKernels.K + 1 + _nch;
        _maxV = _acc.Allocate1D<float>(_nch); _maxI = _acc.Allocate1D<int>(_nch);
        _selV = _acc.Allocate1D<float>(TopKKernels.SelectCap); _selI = _acc.Allocate1D<int>(TopKKernels.SelectCap); _selN = _acc.Allocate1D<int>(64);
        _cand = _acc.Allocate1D<float>(_S);
    }

    /// <summary>The strategies, as <see cref="Preprocessing.GenerationConfig.Strategy"/> names them.</summary>
    public enum Strategy { Greedy = 0, TopK = 1, TopP = 2 }

    /// <summary>
    /// The next draw's settings in ONE upload: <paramref name="window"/> (the recent tokens, any order, the last <see cref="MaxWindow"/> used) penalised by
    /// <paramref name="penalty"/> when it is not 1, and <paramref name="u"/> in [0, 1). Stage BEFORE the step's forward pass is queued: on CUDA and OpenCL a
    /// host-to-device copy waits for the work queued before it, so a copy made after the forward costs the step a second round trip (measured: the device
    /// draw ran slower than reading the whole row back until the settings moved here).
    /// </summary>
    public void Stage(Strategy strategy, float temperature, float topP, int topK, double u, ReadOnlySpan<int> window, float penalty)
    {
        if (!(temperature > 0f)) strategy = Strategy.Greedy;              // a sampled strategy at temperature 0 is the argmax (the host's divides by zero)
        int n = penalty != 1f ? Math.Min(window.Length, MaxWindow) : 0;
        var staged = new float[Header + n];
        staged[0] = strategy == Strategy.Greedy ? 0f : temperature; staged[1] = topP; staged[2] = (float)u; staged[3] = topK; staged[4] = penalty; staged[5] = n;
        var w = window[(window.Length - n)..];
        for (int j = 0; j < n; j++) staged[Header + j] = w[j];
        _params.View.SubView(0, staged.Length).CopyFromCPU(staged);
        _staged = true; _stagedStrategy = strategy; _stagedT = staged[0]; _stagedWindow = n;
    }

    /// <summary>
    /// One token from <paramref name="logits"/> (<paramref name="vocab"/> floats on the device; the penalty is applied to them in place) with the settings
    /// <see cref="Stage"/> uploaded. Returns the token, or -1 when the candidates cannot answer (sample the row on the host instead; the penalty is already in
    /// it). No upload; one read of one int.
    /// </summary>
    /// <remarks>
    /// MEASURED 2026-11-01 (WebGPU, Qwen2.5-0.5B, 256 tokens): a device-drawn top-p + penalty step decodes at 73.9 tok/s against 74.9 for plain greedy (the
    /// capture's folded argmax) and 62.2 for the host draw. Recording these dispatches into a WebGPU plan and replaying it measured no gain (74.5 without,
    /// 73.9-74.6 with), so they are dispatched directly.
    /// </remarks>
    public async Task<int> SampleAsync(FV logits, int vocab)
    {
        if (!_staged) throw new InvalidOperationException("DeviceSampler.SampleAsync: Stage the draw first");
        _staged = false;
        if (logits.Length < vocab) throw new ArgumentException($"{vocab} logits expected", nameof(logits));
        EnsureVocab(vocab);
        var row = logits.SubView(0, vocab);
        if (_stagedWindow > 0) _penalize(_stagedWindow, row, _params.View);
        _chunkMax(_nch, row, _maxV!.View, _maxI!.View, _vocab, _nch, Chunk);
        _floor(TopKKernels.FloorThreads(1, _nch), _maxV.View, _maxI.View, _cand!.View, _selN!.View, _nch, _S);
        _select(_nch, row, _cand.View, _selN.View, _selV!.View, _selI!.View, _vocab, _nch, _S, _stagedT, Chunk);
        _final(TopKKernels.FinalThreads(1), _selV.View, _selI.View, _selN.View, _cand.View, _S);
        _draw(1, _cand.View, _params.View, _out.View, _nch, (int)_stagedStrategy);
        // the read is ordered after the kernels on every backend (no SynchronizeAsync: a second round trip)
        var r = await _out.View.SubView(0, 1).CopyToHostAsync().ConfigureAwait(false);
        return r[0];
    }

    /// <summary><see cref="Stage"/> then <see cref="SampleAsync(FV, int)"/>.</summary>
    public Task<int> SampleAsync(FV logits, int vocab, Strategy strategy, float temperature, float topP, int topK, double u, ReadOnlySpan<int> window, float penalty)
    {
        Stage(strategy, temperature, topP, topK, u, window, penalty);
        return SampleAsync(logits, vocab);
    }

    // the repetition penalty: thread j < prm[5] (the window length) takes the window's token j (prm[Header + j]); a token also at an earlier slot is left to that slot's thread (each
    // distinct token once); a positive logit is divided by the penalty (prm[4]), the rest multiplied (TextGenerationSampler.ApplyRepetitionPenalty)
    private static void PenalizeImpl(Index1D i, FV logits, FV prm)
    {
        int j = i, n = (int)prm[5];
        if (j >= n) return;
        int tok = (int)prm[Header + j];
        bool dup = false;
        for (int k = 0; k < j; k++)
        {
            bool same = (int)prm[Header + k] == tok;
            if (same) { dup = true; break; }
        }
        bool inRange = (tok >= 0) & (tok < logits.Length);
        bool apply = (!dup) & inRange;
        if (!apply) return;
        float pen = prm[4], v = logits[tok];
        logits[tok] = v > 0f ? v / pen : v * pen;
    }

    // the draw from the candidates (cand: K values best first, K ids, the count at the floor, the chunk masses at T): prm = [T, topP, u, topK];
    // out[0] = the token, or -1 when the candidates cannot answer
    private static void DrawImpl(Index1D i, FV cand, FV prm, IV outv, int nch, int strategy)
    {
        const int K = TopKKernels.K;
        float T = prm[0], topP = prm[1], u = prm[2];
        int topK = (int)prm[3];
        bool overflow = cand[2 * K] > TopKKernels.SelectCap;
        int tok = (int)cand[K];                                   // the argmax (exact even on overflow: the best chunk's first maximum)
        if (strategy == 0) { outv[0] = tok; return; }
        if (overflow) { outv[0] = -1; return; }
        float max = cand[0], floorT = max - 20f * T;
        if (strategy == 1)
        {
            // top-k: the k best of the tokens within 20 T of the best, softmax over them, the draw
            if (topK > K) { outv[0] = -1; return; }
            int m = 0; float sum = 0f;
            for (int j = 0; j < topK; j++)
            {
                bool inWindow = cand[j] >= floorT;
                if (!inWindow) break;
                sum += MathF.Exp((cand[j] - max) / T); m = j + 1;
            }
            float x = u * sum, acc = 0f;
            int pick = (int)cand[K + m - 1];
            for (int j = 0; j < m; j++)
            {
                acc += MathF.Exp((cand[j] - max) / T);
                if (x <= acc) { pick = (int)cand[K + j]; break; }
            }
            outv[0] = pick; return;
        }
        // top-p: the nucleus (best first until the mass reaches p of the whole window's), renormalised, the draw
        float total = 0f;
        for (int ch = 0; ch < nch; ch++) total += cand[2 * K + 1 + ch];
        float target = topP * total, keep = 0f;
        int mp = 0;
        for (int j = 0; j < K; j++)
        {
            bool go = (cand[j] >= floorT) & (keep < target);
            if (!go) break;
            keep += MathF.Exp((cand[j] - max) / T); mp = j + 1;
        }
        if (keep < target) { outv[0] = -1; return; }          // the nucleus reaches past the candidates
        float y = u * keep, run = 0f;
        int pk = (int)cand[K + mp - 1];
        for (int j = 0; j < mp; j++)
        {
            run += MathF.Exp((cand[j] - max) / T);
            if (y <= run) { pk = (int)cand[K + j]; break; }
        }
        outv[0] = pk;
    }

    public void Dispose()
    {
        _maxV?.Dispose(); _maxI?.Dispose(); _selV?.Dispose(); _selI?.Dispose(); _selN?.Dispose(); _cand?.Dispose(); _params.Dispose(); _out.Dispose();
    }
}
