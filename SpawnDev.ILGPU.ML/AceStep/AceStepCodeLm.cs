using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Flux2;
using SpawnDev.ILGPU.ML.Hub;

namespace SpawnDev.ILGPU.ML.AceStep;

/// <summary>
/// ACE-Step 1.5's 5 Hz audio-code LM (ComfyUI's qwen_1.7b_ace15: Qwen3 1.7B, tied embeddings) sampling codes as TextEncodeAceStepAudio1.5 does with generate_audio_codes on
/// (measured on SpawnGen's dumps; Plans/ace-step-1.5-2026-11-16.md): a positive and a negative prompt run as two sequences (<see cref="Qwen3KvDecoder"/>, one key / value cache
/// each), every step reads the logits of the 64000 code tokens (token 151669 + code: all of the probability mass at every measured step), combines them as
/// neg + cfg (pos - neg), applies the temperature and nucleus (top-p) sampling, and feeds the chosen token to both sequences.
/// <para>ComfyUI left-pads the negative prompt with &lt;|endoftext|&gt; to the positive's length; pass it unpadded here (leading pads stripped): its positions start at 0
/// (measured: AceStep_CodeLm_KvCache_MatchesComfyUi checks the negative row too). The draws use this class's own seeded generator, not torch's, so codes match ComfyUI's only
/// in distribution; the model itself is gated teacher-forced.</para>
/// </summary>
public sealed class AceStepCodeLm : IDisposable
{
    /// <summary>The token of audio code 0; codes run to <see cref="CodeCount"/> - 1.</summary>
    public const int CodeToken0 = 151669, CodeCount = 64000, EndOfText = 151643;

    private readonly Accelerator _acc;
    private readonly Flux2Weights _w;
    private readonly Qwen3KvDecoder _pos, _neg;

    /// <summary>The positive sequence's decoder (tests read its state).</summary>
    public Qwen3KvDecoder Positive => _pos;
    /// <summary>The negative sequence's decoder.</summary>
    public Qwen3KvDecoder Negative => _neg;

    private AceStepCodeLm(Accelerator acc, Flux2Weights w, int layers, int maxPositions, int maxPrompt)
    {
        _acc = acc; _w = w;
        _pos = new Qwen3KvDecoder(acc, w, layers, maxPositions, maxPrompt); _neg = new Qwen3KvDecoder(acc, w, layers, maxPositions, maxPrompt);
        _pos.SetLogitRange(CodeToken0, CodeCount); _neg.SetLogitRange(CodeToken0, CodeCount);
    }

    /// <summary>Load the LM (ComfyUI's qwen_1.7b_ace15, 28 layers) with caches for <paramref name="maxPrompt"/> prompt tokens plus <paramref name="maxCodes"/> codes.</summary>
    public static async Task<AceStepCodeLm> LoadAsync(Accelerator accelerator, IModelSource source, string repo, string file, int maxPrompt, int maxCodes, int layers = 28, CancellationToken ct = default)
    {
        var w = await Flux2Weights.LoadAsync(accelerator, source, repo, file, layers, finalNorm: true, ct);
        try { return new AceStepCodeLm(accelerator, w, layers, maxPrompt + maxCodes, maxPrompt); }
        catch { w.Dispose(); throw; }
    }

    /// <summary>The negative prompt as this class takes it: ComfyUI's left-padded ids with the leading &lt;|endoftext|&gt; removed.</summary>
    public static int[] StripLeftPadding(int[] ids) => ids.SkipWhile(t => t == EndOfText).ToArray();

    /// <summary>
    /// Sample <paramref name="count"/> codes (5 per second of audio) for the positive / negative prompt ids. <paramref name="cfg"/>, <paramref name="temperature"/> and
    /// <paramref name="topP"/>, <paramref name="topK"/> and <paramref name="minP"/> are ComfyUI's node inputs (its defaults 2, 0.85, 0.9, 0, 0; 0 turns top-k / min-p off);
    /// <paramref name="onCode"/> sees each code as it is chosen.
    /// </summary>
    public async Task<int[]> GenerateCodesAsync(int[] positivePrompt, int[] negativePrompt, int count, int seed, float cfg = 2f, float temperature = 0.85f, float topP = 0.9f,
        Action<int, int>? onCode = null, CancellationToken ct = default, int topK = 0, float minP = 0f)
    {
        var rng = new Random(seed);
        var codes = new int[count];
        _pos.Prefill(positivePrompt); _neg.Prefill(negativePrompt);
        var logits = new double[CodeCount];
        for (int i = 0; i < count; i++)
        {
            ct.ThrowIfCancellationRequested();
            var lp = await _pos.Logits.CopyToHostAsync(); var ln = await _neg.Logits.CopyToHostAsync();
            for (int c = 0; c < CodeCount; c++) logits[c] = ln[c] + cfg * ((double)lp[c] - ln[c]);
            int code = Sample(logits, temperature, topP, rng, topK, minP);
            codes[i] = code; onCode?.Invoke(i, code);
            if (i + 1 < count) { _pos.Step(CodeToken0 + code); _neg.Step(CodeToken0 + code); }
        }
        return codes;
    }

    /// <summary>
    /// Temperature, then (Hugging Face's warper order) top-k: the <paramref name="topK"/> most likely codes (0 = all); top-p: the smallest set of the most likely whose probability
    /// reaches <paramref name="topP"/>; min-p: drop codes below <paramref name="minP"/> times the most likely one's probability (0 = off); renormalised, then one draw. The order
    /// follows Hugging Face generate's; ComfyUI's draws come from torch's generator, so the sampler is checked for its distribution, not against ComfyUI's codes.
    /// </summary>
    public static int Sample(double[] logits, float temperature, float topP, Random rng, int topK = 0, float minP = 0f)
    {
        int n = logits.Length;
        if (temperature <= 0f) { int best = 0; for (int i = 1; i < n; i++) if (logits[i] > logits[best]) best = i; return best; }
        double max = logits.Max();
        var p = new double[n]; double sum = 0;
        for (int i = 0; i < n; i++) { p[i] = Math.Exp((logits[i] - max) / temperature); sum += p[i]; }
        var order = Enumerable.Range(0, n).ToArray();
        Array.Sort(order, (a, b) => p[b].CompareTo(p[a]));
        int limit = topK > 0 ? Math.Min(topK, n) : n;
        double sumK = 0; for (int i = 0; i < limit; i++) sumK += p[order[i]];
        int keep = 0; double cum = 0;
        while (keep < limit) { cum += p[order[keep]] / sumK; keep++; if (topP < 1f && cum >= topP) break; }
        if (minP > 0f)
        {
            double floor = minP * p[order[0]];                                                   // relative to the most likely code (min-p is scale free: before or after renormalising)
            while (keep > 1 && p[order[keep - 1]] < floor) keep--;
        }
        double kept = 0; for (int i = 0; i < keep; i++) kept += p[order[i]];
        double r = rng.NextDouble() * kept;
        for (int i = 0; i < keep; i++) { r -= p[order[i]]; if (r <= 0) return order[i]; }
        return order[keep - 1];
    }

    public void Dispose()
    {
        _pos.Dispose(); _neg.Dispose(); _w.Dispose();
    }
}
