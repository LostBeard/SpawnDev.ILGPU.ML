using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.ILGPU.ML.Preprocessing;

namespace ZipVoiceHarness;

/// <summary>
/// Runs each ZipVoice graph on BOTH engines with byte-identical inputs and reports how far apart they
/// land.
/// </summary>
/// <remarks>
/// This is the test that makes the port checkable. The orchestration is shared, and the inputs to each
/// stage are pinned - the same tokens, the same reference features, the same noise - so the only thing
/// that varies is which engine executed the graph. A difference here is ours and nothing else's.
/// <para>
/// Comparing stage by stage rather than only comparing the final audio matters because the flow-matching
/// loop feeds its own output back in four times: a small encoder error and a large decoder error produce
/// the same unusable waveform at the end, and only a per-stage diff tells them apart.
/// </para>
/// </remarks>
public static class Compare
{
    public static async Task<int> RunAsync(string modelDir, ZipVoiceFixture fixture, float[] reference, int referenceRate)
    {
        var config = new ZipVoiceConfig();
        if (Environment.GetEnvironmentVariable("VERBOSE") == "1") InferenceSession.VerboseLogging = true;

        var builder = MLContext.Create();
        await builder.AllAcceleratorsAsync();
        var context = builder.ToContext();
        var accelerator = await context.CreatePreferredAcceleratorAsync();
        if (accelerator == null) { Console.WriteLine("no accelerator available"); return 3; }
        Console.WriteLine($"device   : {accelerator.AcceleratorType} {accelerator.Name}");

        var encoderPath = Path.Combine(modelDir, "text_encoder.onnx");
        var decoderPath = Path.Combine(modelDir, "fm_decoder.onnx");
        var vocoderPath = Path.Combine(modelDir, "vocos_24khz.onnx");

        using var ort = new OrtZipVoiceGraphs(modelDir);

        // The reference features are computed once and handed to both engines, so a difference in the mel
        // can never be mistaken for a difference in the graphs.
        var promptFeatures = ZipVoiceFeatures.ComputePromptFeatures(reference, referenceRate, config, out int promptFrames);
        Console.WriteLine($"prompt   : {promptFrames} mel frames");

        // ---- encoder ------------------------------------------------------------------------------
        var ortEncoding = await ort.RunEncoderAsync(fixture.Tokens, fixture.PromptTokens, promptFrames, config.Speed);
        Console.WriteLine($"encoder  : ORT {ortEncoding.NumFrames} frames x {ortEncoding.FeatDim}");

        var encoderSession = InferenceSession.CreateFromFile(accelerator, File.ReadAllBytes(encoderPath));
        var vocoderSession = InferenceSession.CreateFromFile(accelerator, File.ReadAllBytes(vocoderPath));
        var skipDecoder = Environment.GetEnvironmentVariable("ZIPVOICE_SKIP_DECODER") == "1";
        var decoderSession = skipDecoder
            ? vocoderSession   // placeholder; never run below when skipping
            : InferenceSession.CreateFromFile(accelerator, File.ReadAllBytes(decoderPath));

        using var ours = new IlgpuZipVoiceGraphs(encoderSession, decoderSession, vocoderSession, accelerator);

        int failures = 0;

        // Each stage is reported independently. A stage that throws must not hide the verdict on the
        // others - knowing that two of three graphs are already correct is what tells you how big the
        // remaining problem is.
        var ourEncoding = default(ZipVoiceEncoding);
        bool encoderOk = false;
        try
        {
            ourEncoding = await ours.RunEncoderAsync(fixture.Tokens, fixture.PromptTokens, promptFrames, config.Speed);
            if (ourEncoding.NumFrames != ortEncoding.NumFrames || ourEncoding.FeatDim != ortEncoding.FeatDim)
            {
                Console.WriteLine($"encoder  : SHAPE MISMATCH - ours {ourEncoding.NumFrames}x{ourEncoding.FeatDim}, " +
                                  $"ORT {ortEncoding.NumFrames}x{ortEncoding.FeatDim}");
                failures++;
            }
            else
            {
                failures += Report("encoder", ortEncoding.TextCondition, ourEncoding.TextCondition, 1e-3f);
                encoderOk = true;
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"encoder  : THREW - {Flatten(ex)}");
            failures++;
        }

        // ---- decoder, one step --------------------------------------------------------------------
        // One step is enough to grade the graph, and four would only compound the same difference.
        if (!skipDecoder)
        {
          try
          {
            int count = ortEncoding.NumFrames * ortEncoding.FeatDim;
            var x = ZipVoicePipeline.GaussianNoise(count, seed: 1234);
            var speechCondition = new float[count];
            Array.Copy(promptFeatures, speechCondition, Math.Min(promptFeatures.Length, promptFrames * ortEncoding.FeatDim));
            float t = ZipVoiceFeatures.Timesteps(config)[0];

            var ortVelocity = await ort.RunDecoderAsync(
                t, x, ortEncoding.TextCondition, speechCondition, config.GuidanceScale,
                ortEncoding.NumFrames, ortEncoding.FeatDim);
            var ourVelocity = await ours.RunDecoderAsync(
                t, x, ortEncoding.TextCondition, speechCondition, config.GuidanceScale,
                ortEncoding.NumFrames, ortEncoding.FeatDim);
            failures += Report("decoder", ortVelocity, ourVelocity, 1e-2f);
          }
          catch (Exception ex)
          {
            Console.WriteLine($"decoder  : THREW - {Flatten(ex)}");
            failures++;
          }
        }
        else
        {
            Console.WriteLine("decoder  : skipped (ZIPVOICE_SKIP_DECODER=1)");
        }

        // ---- vocoder --------------------------------------------------------------------------------
        // Driven from the REFERENCE clip's own mel rather than from generated output, so this stage is
        // graded on an input that is known-good regardless of how the stages before it did.
        int melFrames = promptFrames;
        var mel = new float[config.NumMels * melFrames];
        float invFeatScale = 1f / config.FeatScale;
        for (int f = 0; f < melFrames; f++)
            for (int c = 0; c < config.NumMels; c++)
                mel[c * melFrames + f] = promptFeatures[f * config.NumMels + c] * invFeatScale;

        try
        {
            var ortSpectrum = await ort.RunVocoderAsync(mel, config.NumMels, melFrames);
            var ourSpectrum = await ours.RunVocoderAsync(mel, config.NumMels, melFrames);
            failures += Report("vocoder mag", ortSpectrum.Magnitude, ourSpectrum.Magnitude, 1e-2f);
            failures += Report("vocoder cos", ortSpectrum.Cos, ourSpectrum.Cos, 1e-2f);
            failures += Report("vocoder sin", ortSpectrum.Sin, ourSpectrum.Sin, 1e-2f);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"vocoder  : THREW - {Flatten(ex)}");
            failures++;
        }

        Console.WriteLine($"summary  : encoder {(encoderOk ? "matches" : "NOT matching")}");

        Console.WriteLine(failures == 0 ? "RESULT   : PASS" : $"RESULT   : FAIL ({failures} stage(s) out of tolerance)");
        return failures == 0 ? 0 : 1;
    }

    /// <summary>
    /// The outer message names the failing NODE, the inner one says what went wrong inside it - so both
    /// are kept. Flattening to just the innermost loses which operator to go and look at.
    /// </summary>
    private static string Flatten(Exception ex)
    {
        var outer = ex.Message.ReplaceLineEndings(" ");
        var inner = ex;
        while (inner.InnerException != null) inner = inner.InnerException;
        var text = ReferenceEquals(inner, ex) ? outer : outer + " || " + inner.Message.ReplaceLineEndings(" ");
        return text.Length > 500 ? text[..500] : text;
    }

    /// <summary>
    /// Compare two engines' output for one stage and print the worst and typical disagreement.
    /// </summary>
    /// <remarks>
    /// Both the maximum and the mean are printed on purpose: a single bad element and a uniformly shifted
    /// tensor are very different defects, and the mean alone hides the first while the maximum alone
    /// hides the second. The relative measure is what the tolerance is applied to, so a stage whose
    /// values are naturally large is not judged more harshly than one whose values are small.
    /// </remarks>
    private static int Report(string stage, float[] expected, float[] actual, float tolerance)
    {
        if (expected.Length != actual.Length)
        {
            Console.WriteLine($"{stage,-12}: LENGTH MISMATCH - ORT {expected.Length}, ours {actual.Length}");
            return 1;
        }

        double maxAbs = 0, sumAbs = 0, maxMagnitude = 0;
        int worst = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            double diff = Math.Abs((double)expected[i] - actual[i]);
            sumAbs += diff;
            if (diff > maxAbs) { maxAbs = diff; worst = i; }
            maxMagnitude = Math.Max(maxMagnitude, Math.Abs((double)expected[i]));
        }

        double meanAbs = sumAbs / expected.Length;
        double relative = maxMagnitude > 0 ? maxAbs / maxMagnitude : maxAbs;
        bool pass = relative <= tolerance;

        Console.WriteLine($"{stage,-12}: max |d| {maxAbs:E3} ({relative:P3} of peak {maxMagnitude:F3}), " +
                          $"mean |d| {meanAbs:E3}, worst at {worst} " +
                          $"(ORT {expected[worst]:F5} vs ours {actual[worst]:F5})  {(pass ? "OK" : "OUT OF TOLERANCE")}");
        return pass ? 0 : 1;
    }
}
