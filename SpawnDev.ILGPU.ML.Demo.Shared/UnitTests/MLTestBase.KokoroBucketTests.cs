using System.Diagnostics;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// Does PADDING the token sequence to a bucket change what Kokoro says?
/// </summary>
/// <remarks>
/// 🔴 THIS DECIDES WHETHER STREAMING TTS CAN BE FASTER THAN REALTIME. A recorded dispatch plan is bound to
/// ONE sequence length. Replaying one is 0.34x realtime; running a new length uncaptured is ~1.8x, i.e.
/// slower than speech, which is exactly the pauses-between-chunks failure. In streaming every chunk is a
/// different length, so the repeat case - the only fast one - is the case a conversation never hits.
///
/// Unless lengths are made to repeat. If padding a short sequence up to a bucket boundary produces the
/// SAME audio for the real phonemes, then a handful of buckets covers every chunk, consecutive chunks
/// reuse one plan, and streaming runs at replay speed.
///
/// ⚠️ THE STYLE ROW MUST STILL COME FROM THE REAL PHONEME COUNT. KokoroVoicePack.StyleFor indexes a table
/// BY token count, so choosing it by the padded count makes a short sentence adopt a longer utterance's
/// timbre - audible as the voice changing mid-reply, and not something a length assertion would catch.
/// That is what SpeakTokensAsync(styleTokenCount:) is for.
///
/// This test answers one question and asserts nothing about speed: is the padded audio the same audio?
/// Read it with <c>PMT_CONSOLE_LOG=KokoroBucket</c>.
/// </remarks>
public abstract partial class MLTestBase
{
    [TestMethod(Timeout = 900000, Category = "HeavyModel,WasmHeavy")]
    public async Task Kokoro_PaddingToABucket_SaysTheSameThing() => await RunTest(async accelerator =>
    {
        RequireShippableTtsBackend(accelerator);
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");

        byte[] modelBytes, voiceBytes;
        try
        {
            modelBytes = await http.GetByteArrayAsync($"{KokoroHubBase}/onnx/model.onnx");
            voiceBytes = await http.GetByteArrayAsync($"{KokoroHubBase}/voices/af_heart.bin");
        }
        catch (HttpRequestException ex)
        {
            throw new UnsupportedTestException($"the hub did not serve Kokoro: {ex.Message}");
        }

        var pack = KokoroVoicePack.FromBytes("af_heart", voiceBytes);
        using var pipeline = KokoroPipeline.Create(accelerator, modelBytes);

        var real = KokoroReferenceTokens;                    // 35 ids, ending in the pad token
        var baseline = await pipeline.SpeakTokensAsync(real, pack);

        // Pad to the next multiple of 16 with the PAD token (id 0) - the same id the sequence already
        // begins and ends with - and keep the style row for the REAL count.
        const int bucket = 16;
        int padded = ((real.Length + bucket - 1) / bucket) * bucket;
        if (padded == real.Length) padded += bucket;
        var paddedTokens = new long[padded];
        Array.Copy(real, paddedTokens, real.Length);         // remainder stays 0 = pad
        var bucketed = await pipeline.SpeakTokensAsync(paddedTokens, pack, styleTokenCount: real.Length);

        Console.WriteLine($"[KokoroBucket] {BackendName}: {real.Length} ids -> {baseline.Samples.Length} samples "
                        + $"({baseline.Seconds:F2}s); padded to {padded} -> {bucketed.Samples.Length} samples "
                        + $"({bucketed.Seconds:F2}s)");

        if (bucketed.Samples.Length == 0)
            throw new Exception($"{BackendName}: the padded sequence produced no audio at all");

        // Compare the overlap. Padding may append silence for the extra pad tokens - that is fine and
        // trimmable - but the REAL phonemes must render identically, or bucketing changes what is said.
        int n = Math.Min(baseline.Samples.Length, bucketed.Samples.Length);
        double num = 0, da = 0, db = 0, worst = 0;
        for (int i = 0; i < n; i++)
        {
            double a = baseline.Samples[i], b = bucketed.Samples[i];
            num += a * b; da += a * a; db += b * b;
            var d = Math.Abs(a - b); if (d > worst) worst = d;
        }
        double corr = (da > 0 && db > 0) ? num / Math.Sqrt(da * db) : 0;

        // How much of the padded output is extra, and is that extra silence?
        double tailPeak = 0;
        for (int i = n; i < bucketed.Samples.Length; i++)
            tailPeak = Math.Max(tailPeak, Math.Abs(bucketed.Samples[i]));

        Console.WriteLine($"[KokoroBucket] {BackendName}: overlap {n} samples, correlation {corr:F4}, "
                        + $"worst |delta| {worst:E2}, extra {bucketed.Samples.Length - n} samples "
                        + $"with peak {tailPeak:F4}");

        // ⚠️ REPORTED, NOT GATED, on purpose. The question here is empirical - "can we bucket?" - and the
        // answer is the correlation, not a pass/fail. Asserting a threshold before knowing the answer
        // would just encode a guess. The one thing that IS a failure is producing nothing.
        if (corr < 0.99)
            Console.WriteLine($"[KokoroBucket] {BackendName}: VERDICT - padding CHANGES the audio "
                            + $"(corr {corr:F4}). Shape bucketing is NOT viable this way; a captured plan "
                            + "per length, or a graph change, is required instead.");
        else
            Console.WriteLine($"[KokoroBucket] {BackendName}: VERDICT - padding preserves the audio "
                            + $"(corr {corr:F4}, tail peak {tailPeak:F4}). Bucketing IS viable: pad to a "
                            + "bucket, keep the real style row, trim the tail.");
    });
}
