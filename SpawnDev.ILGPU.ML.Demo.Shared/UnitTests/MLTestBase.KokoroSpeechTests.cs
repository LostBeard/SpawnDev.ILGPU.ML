using System.Diagnostics;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// Kokoro-82M end to end, against an onnxruntime reference waveform.
/// </summary>
/// <remarks>
/// <para>
/// 🔴 THIS IS THE GATE THE PORT DID NOT HAVE. Nine separate defects were found by running this model and
/// every one of them produced a full, correctly-shaped, non-throwing buffer: a ConvTranspose bias rented
/// from the pool and never zeroed (0.333 added to all 54,620 samples), a fused Sigmoid that was declared,
/// mapped and then silently not applied (every phoneme duration clamped to 1), an STFT hop that shape
/// inference could not see (5 frames instead of 121), an LSTM whose runtime shape re-inference was skipped
/// because an OPTIONAL input was absent, an epsilon deleted from every normalisation by strength reduction
/// reading an int-truncated constant, and an atan2 written longhand that returns the wrong sign of pi at an
/// exactly-zero imaginary part. Not one of them threw. The ONLY thing that separates "this model runs" from
/// "this model is right" is comparing the samples to a reference.
/// </para>
/// <para>
/// ⚠️ EACH ASSERTION BELOW CORRESPONDS TO A DEFECT THAT ACTUALLY SHIPPED, and they are not
/// interchangeable - the length catches the duration and shape bugs, the peak catches the un-zeroed bias
/// (which gave peak 2226) and a silent output, correlation catches the epsilon (0.45) and the atan2
/// (0.95), and the best-fit scale catches noise in the log-magnitude, which inflates loudness through
/// Exp without ever decorrelating the signal.
/// </para>
/// <para>
/// ⚠️ <b>HeavyModel</b>: fetches ~326 MB of model through the hub, so it is excluded from every sweep by
/// default and runs when asked for:
/// <c>PMT_EXCLUDE_CATEGORIES= PMT_FILTER=Pipeline_Kokoro dotnet test PlaywrightMultiTest/...</c>
/// </para>
/// <para>
/// ⭐ PROVENANCE OF THE FIXTURE, so it can be regenerated rather than trusted:
/// <c>node tools/kokoro-oracle.mjs &lt;model.onnx&gt; &lt;af_heart.bin&gt; &lt;tokens-csv&gt;
/// SpawnDev.ILGPU.ML.Demo/wwwroot/test-refs/kokoro-af_heart-paris.f32</c> - onnxruntime, fp32 export,
/// the token ids below. ⚠️ The <b>fp32</b> export specifically: the published fp16 one correlates only
/// 0.918 against it, a bigger deviation than this engine's entire remaining error.
/// </para>
/// </remarks>
public abstract partial class MLTestBase
{
    private const string KokoroHubBase =
        "https://hub.spawndev.com:44365/hf/onnx-community/Kokoro-82M-v1.0-ONNX";

    /// <summary>The line the reference waveform was rendered from, and its exact phoneme tokens.</summary>
    /// <remarks>
    /// ⚠️ The TOKENS are the fixture, not the text. Feeding the reference engine the text would compare two
    /// FRONT ENDS as well as two inference engines, so a phonemizer change would read as an engine
    /// regression. The KokoroFrontEnd tests cover the text-to-token half
    /// separately; this covers the token-to-audio half.
    /// </remarks>
    private const string KokoroReferenceLine = "The capital of France is Paris.";

    private static readonly long[] KokoroReferenceTokens =
    {
        0, 81, 83, 16, 53, 156, 72, 58, 83, 125, 83, 54, 16, 138, 64, 16, 48, 123, 156, 72, 56, 61, 16,
        102, 68, 16, 58, 156, 86, 123, 102, 61, 16, 4, 0,
    };

    /// <summary>
    /// Synthesise the reference line and compare every sample to onnxruntime's.
    /// </summary>
    [TestMethod(Timeout = 900000, Category = "HeavyModel,WasmHeavy")]
    public async Task Pipeline_Kokoro_MatchesOnnxRuntimeWaveform() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");

        // The reference is served from the demo's wwwroot; the model comes through the hub, which is the
        // ONLY route to a Hugging Face artefact in this project.
        float[] reference;
        try
        {
            var refBytes = await http.GetByteArrayAsync("test-refs/kokoro-af_heart-paris.f32");
            reference = new float[refBytes.Length / 4];
            Buffer.BlockCopy(refBytes, 0, reference, 0, refBytes.Length);
        }
        catch (HttpRequestException ex)
        {
            // NOT UnsupportedTestException. The reference is COMMITTED next to the demo, so it not being
            // served means the harness is wrong - and a gate that answers "unsupported" is green.
            throw new Exception("test-refs/kokoro-af_heart-paris.f32 is committed but was not served: "
                + ex.Message);
        }

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

        // ⚠️ TWICE WHERE THAT IS AFFORDABLE, because the first run of this graph is not the cost of the
        // graph. Every kernel compiles on its FIRST execution, and on a browser backend that compile is
        // most of a cold run - MEASURED on WebGPU (real RTX 4070, isFallbackAdapter=False): 8.1 s cold,
        // 2.1 s warm. Reporting only the cold figure describes the first reply of a session as though it
        // were every reply, which is the difference between "slower than realtime" and "faster than
        // realtime" for this model.
        //
        // 🔴 But the SECOND pass is a measurement, and correctness is the gate. The CPU backend takes
        // ~312 s per pass, so running it twice cost 624 s against PMT's 600 s outer cap and the row was
        // KILLED - a backend that was verifying correctness perfectly well reported as a failure, for a
        // timing number nobody would ship anyway. So the warm pass runs only where the cold one showed it
        // is affordable, and its absence is stated rather than papered over.
        var cold = Stopwatch.StartNew();
        var audio = await pipeline.SpeakTokensAsync(KokoroReferenceTokens, pack);
        cold.Stop();

        const int warmBudgetMs = 60_000;
        Stopwatch? sw = null;
        if (cold.ElapsedMilliseconds <= warmBudgetMs)
        {
            sw = Stopwatch.StartNew();
            audio = await pipeline.SpeakTokensAsync(KokoroReferenceTokens, pack);
            sw.Stop();
        }

        // ── Length. A wrong duration, a wrong STFT hop and a stale LSTM sequence length all land here,
        //    and all of them produce audio rather than an error. EXACT, not approximate: the reference
        //    and this engine are computing the same deterministic frame count from the same tokens.
        if (audio.Samples.Length != reference.Length)
            throw new Exception($"length {audio.Samples.Length} != reference {reference.Length} on "
                + $"{BackendName} - a duration or frame-count defect, not a numeric one");

        var peak = 0f;
        foreach (var s in audio.Samples) { var a = MathF.Abs(s); if (a > peak) peak = a; }
        if (peak < 0.05f)
            throw new Exception($"peak {peak:F4} on {BackendName} - the graph ran and produced silence");
        if (peak > 1.5f)
            throw new Exception($"peak {peak:F4} on {BackendName} - a stale buffer or a missing "
                + "activation; speech does not leave the interval this model's output lives in");

        double rr = 0, oo = 0, ro = 0;
        for (var i = 0; i < reference.Length; i++)
        {
            double r = reference[i], o = audio.Samples[i];
            rr += r * r; oo += o * o; ro += r * o;
        }
        var correlation = ro / Math.Sqrt(rr * oo);
        // The least-squares gain of ours over the reference. Noise entering the LOG magnitude survives Exp
        // as a systematic loudness gain without decorrelating anything, so correlation alone cannot see it
        // - the epsilon defect left correlation at 0.95 and this at 1.3.
        var scale = ro / rr;

        // 0.98 is set by the model, not by taste. Its vocoder takes atan2 of an STFT whose quiet bins are
        // near zero, so the phase there is decided by rounding: perturbing onnxruntime's OWN spectrogram by
        // one part in 1e6 moves its waveform to correlation 0.974 against itself. 0.98 sits above every
        // measured defect (0.45, 0.95) and below that noise floor.
        if (correlation < 0.98)
            throw new Exception($"correlation {correlation:F4} against onnxruntime on {BackendName} "
                + $"(scale {scale:F3}) - the samples are wrong, not merely rounded");
        if (scale < 0.85 || scale > 1.2)
            throw new Exception($"best-fit gain {scale:F3} on {BackendName} (correlation {correlation:F4}) "
                + "- correlated but at the wrong level");

        var warmth = sw == null
            ? $"warm not measured (cold exceeded the {warmBudgetMs / 1000}s budget for a second pass)"
            : $"warm {sw.ElapsedMilliseconds} ms = RTF {sw.Elapsed.TotalSeconds / audio.Seconds:F2}x";
        Console.WriteLine($"[Kokoro] \"{KokoroReferenceLine}\" -> {audio.Samples.Length} samples "
            + $"({audio.Seconds:F2}s), {warmth} (cold {cold.ElapsedMilliseconds} ms = RTF "
            + $"{cold.Elapsed.TotalSeconds / audio.Seconds:F2}x), peak {peak:F3}, correlation "
            + $"{correlation:F4}, gain {scale:F3} on {BackendName}");
    });
}
