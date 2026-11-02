using SpawnDev.ILGPU.ML.Preprocessing;
using System.Text;

namespace SpawnDev.ILGPU.ML.Pipelines;

/// <summary>
/// The text side of a generation, shared by the decode engines (<see cref="GgufGenerator"/>, <see cref="GgufBatchTextEngine"/>): fitting a prompt and its answer
/// into the cache, the UTF-8-safe streaming detokenizer, stop strings (the earliest match cuts the text there) and the held-back tail (text that could still
/// be the start of a stop string is not streamed until it cannot).
/// </summary>
internal static class GenerationText
{
    /// <summary>
    /// Fit <paramref name="promptIds"/> plus <paramref name="maxNew"/> generated tokens into a cache of <paramref name="maxSeqLen"/>: an over-long prompt keeps its
    /// most recent tokens (where the question is; agentic clients send prompts far larger than a small model's context) with room reserved for the answer, and
    /// the answer's budget shrinks to what is left. <c>Truncated</c>: the prompt lost tokens, so its positions no longer match a cached prefix.
    /// </summary>
    public static (int[] Ids, int MaxNew, bool Truncated) FitPrompt(int[] promptIds, int maxNew, int maxSeqLen)
    {
        bool truncated = false;
        if (promptIds.Length + maxNew > maxSeqLen - 1)
        {
            int reserve = Math.Clamp(maxNew, 64, Math.Max(64, maxSeqLen / 8));
            int keep = maxSeqLen - 1 - reserve;
            if (keep >= 1 && promptIds.Length > keep)
            {
                promptIds = promptIds[^keep..];
                truncated = true;
            }
            maxNew = Math.Min(maxNew, maxSeqLen - 1 - promptIds.Length);
            if (maxNew < 1) maxNew = 1;
        }
        return (promptIds, maxNew, truncated);
    }

    /// <summary>Index of the earliest stop-string match at or after <paramref name="from"/>, or -1.</summary>
    public static int EarliestStopMatch(StringBuilder sb, int from, IReadOnlyList<string>? stops)
    {
        if (stops == null || stops.Count == 0) return -1;
        string s = sb.ToString();
        int best = -1;
        foreach (var stopStr in stops)
        {
            if (string.IsNullOrEmpty(stopStr)) continue;
            // Search a little before `from` so a stop string straddling the last emit boundary is still found.
            int searchFrom = Math.Max(0, from - stopStr.Length + 1);
            int idx = s.IndexOf(stopStr, searchFrom, StringComparison.Ordinal);
            if (idx >= 0 && (best < 0 || idx < best)) best = idx;
        }
        return best;
    }

    /// <summary>
    /// Length of the suffix of <paramref name="sb"/> that is a (possibly full) prefix of some stop string —
    /// the tail that must be held back from streaming until we know it isn't the start of a stop match.
    /// </summary>
    public static int PartialStopSuffix(StringBuilder sb, int maxStopLen, IReadOnlyList<string>? stops)
    {
        if (stops == null || stops.Count == 0 || maxStopLen == 0) return 0;
        string s = sb.ToString();
        int maxCheck = Math.Min(maxStopLen, s.Length);
        for (int len = maxCheck; len > 0; len--)
        {
            var tail = s.AsSpan(s.Length - len);
            foreach (var stopStr in stops)
            {
                if (string.IsNullOrEmpty(stopStr) || stopStr.Length < len) continue;
                if (stopStr.AsSpan(0, len).SequenceEqual(tail)) return len; // tail is a prefix of this stop
            }
        }
        return 0;
    }
}

/// <summary>
/// One answer's text as its tokens arrive: detokenized UTF-8-safely, cut at the earliest stop string, and streamed to a delta callback without the tail that
/// could still begin a stop string. <see cref="PushAsync"/> per generated token (false: a stop string matched, the text ends there), then
/// <see cref="FinishAsync"/> when the generation ends any other way.
/// </summary>
internal sealed class TextStream
{
    private readonly SentencePieceStreamingDecoder _detok;
    private readonly IReadOnlyList<string>? _stops;
    private readonly int _maxStopLen;
    private readonly Func<string, Task>? _onDelta;
    private readonly StringBuilder _full = new();
    private int _emitted;

    public TextStream(SentencePieceTokenizer tokenizer, IReadOnlyList<string>? stopStrings, Func<string, Task>? onDelta)
    {
        _detok = tokenizer.CreateStreamingDecoder();
        _stops = stopStrings;
        _onDelta = onDelta;
        if (stopStrings != null) foreach (var s in stopStrings) if (s.Length > _maxStopLen) _maxStopLen = s.Length;
    }

    /// <summary>The text so far (after a stop string: up to it).</summary>
    public string Text => _full.ToString();

    /// <summary>True once a stop string ended the text.</summary>
    public bool StoppedOnString { get; private set; }

    /// <summary>Add a generated token. False: a stop string matched; the text was cut at its start and streamed up to there.</summary>
    public async ValueTask<bool> PushAsync(int token)
    {
        _full.Append(_detok.Push(token));
        int cut = GenerationText.EarliestStopMatch(_full, _emitted, _stops);
        if (cut >= 0)
        {
            if (cut > _emitted && _onDelta != null) await _onDelta(_full.ToString(_emitted, cut - _emitted));
            _full.Length = cut;
            _emitted = cut;
            StoppedOnString = true;
            return false;
        }
        // stream everything that cannot still be the start of a stop string (hold back the tail)
        int safe = _full.Length - GenerationText.PartialStopSuffix(_full, _maxStopLen, _stops);
        if (safe > _emitted && _onDelta != null)
        {
            await _onDelta(_full.ToString(_emitted, safe - _emitted));
            _emitted = safe;
        }
        return true;
    }

    /// <summary>The generation ended without a stop string: flush the detokenizer and the held-back tail.</summary>
    public async Task FinishAsync()
    {
        if (StoppedOnString) return;
        _full.Append(_detok.Finish());
        if (_full.Length > _emitted && _onDelta != null)
            await _onDelta(_full.ToString(_emitted, _full.Length - _emitted));
        _emitted = _full.Length;
    }
}
