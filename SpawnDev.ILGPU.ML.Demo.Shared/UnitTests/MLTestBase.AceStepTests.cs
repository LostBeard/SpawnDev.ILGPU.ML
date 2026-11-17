using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.AceStep;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// ACE-Step 1.5 (ComfyUI's audio_ace_step_1_5_split template) against SpawnGen's ComfyUI dumps (float32 CPU; _DevComms 2026-11-17-0010; Plans/ace-step-1.5-2026-11-16.md).
/// Files: ACE_DIR (default F:/spawndev-acestep): comfy/split_files/{vae, text_encoders, diffusion_models}/..., comfyui-dumps/ace_*.
/// </summary>
public abstract partial class MLTestBase
{
    private static string Ace_Dir() => Environment.GetEnvironmentVariable("ACE_DIR") ?? "F:/spawndev-acestep";
    private static string Ace_Dumps() => Path.Combine(Ace_Dir(), "comfyui-dumps");

    private static string Ace_Require(Accelerator accelerator, string relative)
    {
        if (accelerator.AcceleratorType is not (AcceleratorType.Cuda or AcceleratorType.OpenCL)) throw new UnsupportedTestException("reads the files from disk: a desktop GPU");
        var p = Path.Combine(Ace_Dir(), "comfy", "split_files", relative);
        if (!File.Exists(p)) throw new UnsupportedTestException($"{p} not found (Comfy-Org/ACE-Step-1.5)");
        return p;
    }

    /// <summary>
    /// The whole codes-off graph (<see cref="AceStepGenerator"/>: caption + lyrics -> condition tokens -> 2 Euler steps of the DiT, sigmas 1, 0.75, 0 -> VAE) from ComfyUI's
    /// own ids, starting noise (call 0's input) and silence latent (its refer_audio), against ComfyUI's final latent [64, 250] and audio [2, 480000]. ComfyUI's call output is the
    /// denoised x - sigma v and its step plain Euler (measured on the dumps: 2.6e-8 / 2.7e-8).
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task AceStep_Graph_CodesOff_MatchesComfyUi() => await RunTest(accelerator => Ace_GraphAsync(accelerator, "ace_g0", false, 6e-4, 1.5e-3));   // MEASURED 2026-11-17 CUDA: latent 1.8e-4, audio 3.9e-4

    /// <summary>
    /// The whole codes-on graph: the same with ComfyUI's 50 audio codes (the code LM's choice, from the codes-on run), whose decoded hints replace the silence latent as the DiT's
    /// source latents (<see cref="AceStepCodeDecoder"/>), against ComfyUI's final latent and audio.
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task AceStep_Graph_CodesOn_MatchesComfyUi() => await RunTest(accelerator => Ace_GraphAsync(accelerator, "ace_g1", true, 1e-3, 2.5e-3));       // MEASURED 2026-11-17 CUDA: latent 3.4e-4, audio 8.7e-4

    private async Task<string> Ace_GraphAsync(Accelerator accelerator, string run, bool codesOn, double latentBound, double audioBound)
    {
        var dit = Ace_Require(accelerator, Path.Combine("diffusion_models", "acestep_v1.5_turbo.safetensors"));
        var te = Ace_Require(accelerator, Path.Combine("text_encoders", "qwen_0.6b_ace15.safetensors"));
        var vae = Ace_Require(accelerator, Path.Combine("vae", "ace_1.5_vae.safetensors"));
        var d = Ace_Dumps();
        int[] Ids(int call)
        {
            using var doc = System.Text.Json.JsonDocument.Parse(File.ReadAllText(Path.Combine(d, $"{run}_ids.json")));
            var e = doc.RootElement.GetProperty("calls")[call].GetProperty("ids");
            static IEnumerable<int> Flat(System.Text.Json.JsonElement x) => x.ValueKind == System.Text.Json.JsonValueKind.Array ? x.EnumerateArray().SelectMany(Flat) : new[] { x.GetInt32() };
            if (e.ValueKind == System.Text.Json.JsonValueKind.String) { using var inner = System.Text.Json.JsonDocument.Parse(e.GetString()!); return Flat(inner.RootElement).ToArray(); }
            return Flat(e).ToArray();
        }
        const int frames = 250;
        var noise = F2_Raw(Path.Combine(d, $"{run}_call_0_input_f32_1x64x250.raw"), 64L * frames);
        var referCm = F2_Raw(Path.Combine(d, $"{run}_call_0_c_refer_audio_f32_1x64x250.raw"), 64L * frames);
        var silence = new float[frames * 64];
        for (int c = 0; c < 64; c++) for (int t = 0; t < frames; t++) silence[t * 64 + c] = referCm[c * frames + t];
        var expectedLatent = F2_Raw(Path.Combine(d, $"{run}_out_samples_f32_1x64x250.raw"), 64L * frames);
        var expectedAudio = F2_Raw(Path.Combine(d, $"{run}_audio_waveform_f32_1x2x480000.raw"), 2L * 480000);
        var src = new LocalFolderModelSource(Path.Combine(Ace_Dir(), "comfy", "split_files"));
        using var gen = await AceStepGenerator.LoadAsync(accelerator, src, "", Path.Combine("diffusion_models", Path.GetFileName(dit)), Path.Combine("text_encoders", Path.GetFileName(te)),
            Path.Combine("vae", Path.GetFileName(vae)), frames, 256, 512);
        var sw = System.Diagnostics.Stopwatch.StartNew();
        int[]? codes = null;
        if (codesOn)
        {
            var codeBytes = File.ReadAllBytes(Path.Combine(d, $"{run}_taps_diffusion_model_0_kw-audio_codes0_i64_1x50.raw"));
            codes = Enumerable.Range(0, codeBytes.Length / 8).Select(i => (int)BitConverter.ToInt64(codeBytes, i * 8)).ToArray();
        }
        var latent = await gen.GenerateLatentAsync(Ids(0), Ids(1), noise, silence, frames, new[] { 1f, 0.75f, 0f }, codes);
        double tl = sw.Elapsed.TotalSeconds;
        var audio = await gen.DecodeAsync(latent, frames);
        double el = F2_RelRms(latent, expectedLatent), ea = F2_RelRms(audio, expectedAudio);
        string report = $"latent [64, 250] {el:E2}, audio [2, 480000] {ea:E2} (2 steps {tl:F1} s, decode {sw.Elapsed.TotalSeconds - tl:F1} s)";
        Console.WriteLine($"[AceStep] codes-{(codesOn ? "on" : "off")} graph vs ComfyUI: " + report);
        Ur_Expect(el < latentBound && ea < audioBound, report);
        return report;
    }

    /// <summary>
    /// ComfyUI's ACE-Step 1.5 text conditioning with the audio codes off (TextEncodeAceStepAudio1.5, SpawnGen's te dump): the tags' 127 ids (ComfyUI's template
    /// included, ids as dumped) through qwen_0.6b_ace15 (Qwen3, 28 layers: <see cref="Flux2.Flux2TextEncoder"/>) give cond [1, 127, 1024]; the lyrics' 350 ids give
    /// conditioning_lyrics [1, 350, 1024] = the token embedding table alone (no layers). Both readings of the last layer are reported (with / without the final
    /// norm); the one ComfyUI uses must match.
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task AceStep_TextConditioning_MatchesComfyUi() => await RunTest(async accelerator =>
    {
        var tePath = Ace_Require(accelerator, Path.Combine("text_encoders", "qwen_0.6b_ace15.safetensors"));
        var d = Ace_Dumps();
        using var idsDoc = System.Text.Json.JsonDocument.Parse(File.ReadAllText(Path.Combine(d, "ace_te_codes0_ids.json")));
        int[] Ids(int call)
        {
            var e = idsDoc.RootElement.GetProperty("calls")[call].GetProperty("ids");
            if (e.ValueKind == System.Text.Json.JsonValueKind.String) using (var inner = System.Text.Json.JsonDocument.Parse(e.GetString()!)) return Flat(inner.RootElement).ToArray();
            return Flat(e).ToArray();
            static IEnumerable<int> Flat(System.Text.Json.JsonElement x) => x.ValueKind == System.Text.Json.JsonValueKind.Array ? x.EnumerateArray().SelectMany(Flat) : new[] { x.GetInt32() };
        }
        int[] tags = Ids(0), lyrics = Ids(1);
        Ur_Expect(tags.Length == 127 && lyrics.Length == 350, $"ids {tags.Length} / {lyrics.Length}");
        var cond = F2_Raw(Path.Combine(d, "ace_te_codes0_cond_0_cond_f32_1x127x1024.raw"), 127L * 1024);
        var lyr = F2_Raw(Path.Combine(d, "ace_te_codes0_cond_0_conditioning_lyrics_f32_1x350x1024.raw"), 350L * 1024);
        var report = new List<string>();
        var src = new LocalFolderModelSource(Path.GetDirectoryName(tePath)!);
        using var w = await Flux2.Flux2Weights.LoadAsync(accelerator, src, "", Path.GetFileName(tePath), 28, finalNorm: true);
        // ComfyUI's caption pass: the prompt is real up to its first <|endoftext|> (151643, Qwen's pad token); every key from there on is padding (causal + padding mask), while
        // those rows still give outputs. MEASURED: keys masked from the first pad 2.2e-6 (numpy float64 rebuild, _tuvok-handoff/ace_qwen06_variants.py); pads masked only where
        // they stand 9.0e-2; no mask (plain causal) 1.0e-1.
        int real = Array.IndexOf(tags, 151643) is int first and >= 0 ? first : tags.Length;
        double best = 0, control = 0;
        using (var te = new Flux2.Flux2TextEncoder(accelerator, w, new[] { 28 }, finalNorm: true, maxTokens: 127) { PreciseActivations = true })
        {
            foreach (int r in new[] { real, tags.Length })                                    // ComfyUI's reading, then the control: every key valid must miss
            {
                var ours = await te.Encode(tags, r).SubView(0, 127L * 1024).CopyToHostAsync();
                double e = F2_RelRms(ours, cond);
                string Row(int i) => $"row {i} {F2_RelRms(ours[(i * 1024)..((i + 1) * 1024)], cond[(i * 1024)..((i + 1) * 1024)]):E1}";
                report.Add($"tags -> layer 28 + final norm, keys valid below {r}: {e:E2} ({Row(0)}, {Row(real - 1)}, {Row(Math.Min(real, 126))}, {Row(126)})");
                if (r == real) best = e; else control = e;
            }
        }
        // the lyrics: the embedding table rows (bf16 in the file, exact in fp32)
        var table = w.Vector("embed_tokens.weight");                                            // the bf16 table as stored (no decode scratch)
        var rows = await table.SubView(0, (long)(lyrics.Max() + 1) * 1024).CopyToHostAsync();
        var emb = lyrics.SelectMany(t => Enumerable.Range(0, 1024).Select(j => (float)rows[(long)t * 1024 + j])).ToArray();
        double el = F2_RelRms(emb, lyr);
        report.Add($"lyrics = embed_tokens[ids]: {el:E2}");
        foreach (var line in report) Console.WriteLine("[AceStep] text conditioning vs ComfyUI: " + line);
        Ur_Expect(best < 1e-4 && control > 100 * best && el < 1e-6, string.Join("; ", report));  // MEASURED 2026-11-17 CUDA: 2.1e-5 (control 0.104), lyrics 7.7e-9
        return string.Join("; ", report);
    });

    /// <summary>
    /// The condition encoder (<see cref="AceStepConditionEncoder"/>, acestep_v1.5_turbo's encoder.*) on ComfyUI's own inputs of the codes-off graph's first call (SpawnGen's
    /// encoder taps: caption states [127, 1024], lyric embeddings [350, 1024], reference latent [250, 64]) against its output [478, 2048] = [lyrics | timbre | text].
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task AceStep_ConditionEncoder_MatchesComfyUiTap() => await RunTest(async accelerator =>
    {
        var path = Ace_Require(accelerator, Path.Combine("diffusion_models", "acestep_v1.5_turbo.safetensors"));
        var d = Ace_Dumps();
        string T(string rest) => Path.Combine(d, "ace_g0_taps_diffusion_model_encoder_0_" + rest + ".raw");
        var text = F2_Raw(T("in0_f32_1x127x1024"), 127L * 1024);
        var lyrics = F2_Raw(T("in1_f32_1x350x1024"), 350L * 1024);
        var refer = F2_Raw(T("in2_f32_1x250x64"), 250L * 64);
        var expected = F2_Raw(T("out0_f32_1x478x2048"), 478L * 2048);
        using var enc = await AceStepConditionEncoder.LoadAsync(accelerator, new LocalFolderModelSource(Path.GetDirectoryName(path)!), "", Path.GetFileName(path), 350);
        using var dT = accelerator.Allocate1D(text); using var dL = accelerator.Allocate1D(lyrics); using var dR = accelerator.Allocate1D(refer);
        using var outp = accelerator.Allocate1D<float>(478L * 2048);
        var lines = new List<string>(); double best = double.MaxValue, control = 0;
        Ur_Expect(enc.SlidingWindow == -1, "the default must be ComfyUI's full attention");
        foreach (int window in new[] { -1, AceStepConditionEncoder.Window })                  // ComfyUI's full attention (default), then the control: ACE-Step's own window must miss
        {
            enc.SlidingWindow = window;
            enc.Encode(dT.View, 127, dL.View, 350, dR.View, 250, outp.View);
            var ours = await outp.View.CopyToHostAsync();
            double all = F2_RelRms(ours, expected);
            if (window < 0) best = all; else control = all;
            double Part(int r0, int rows) => F2_RelRms(ours[(r0 * 2048)..((r0 + rows) * 2048)], expected[(r0 * 2048)..((r0 + rows) * 2048)]);
            lines.Add($"window {(window < 0 ? "none" : window.ToString())}: [478, 2048] {all:E2} (lyrics {Part(0, 350):E2}, timbre {Part(350, 1):E2}, text {Part(351, 127):E2})");
        }
        foreach (var line in lines) Console.WriteLine("[AceStep] condition encoder vs ComfyUI: " + line);
        string report = string.Join("; ", lines);
        Ur_Expect(best < 5e-5 && control > 100 * best, report);                               // MEASURED 2026-11-17 CUDA: full attention 6.3e-6 (lyrics 7.5e-5), window 128: 3.1e-2
        return report;
    });

    /// <summary>
    /// The DiT (<see cref="AceStepDiT"/>, acestep_v1.5_turbo's decoder.*) on ComfyUI's own inputs of the codes-off graph's first model call (SpawnGen's decoder taps:
    /// hidden_states [250, 64], timestep, timestep_r, encoder_hidden_states [478, 2048], context_latents [250, 128]) against the decoder's output [250, 64].
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task AceStep_DiT_MatchesComfyUiDecoderTap() => await RunTest(async accelerator =>
    {
        var path = Ace_Require(accelerator, Path.Combine("diffusion_models", "acestep_v1.5_turbo.safetensors"));
        var d = Ace_Dumps();
        string T(string rest) => Path.Combine(d, "ace_g0_taps_diffusion_model_decoder_0_" + rest + ".raw");
        var x = F2_Raw(T("kw-hidden_states0_f32_1x250x64"), 250L * 64);
        var ctx = F2_Raw(T("kw-context_latents0_f32_1x250x128"), 250L * 128);
        var enc = F2_Raw(T("kw-encoder_hidden_states0_f32_1x478x2048"), 478L * 2048);
        float t = F2_Raw(T("kw-timestep0_f32_1"), 1)[0], tr = F2_Raw(T("kw-timestep_r0_f32_1"), 1)[0];
        var expected = F2_Raw(T("out0_f32_1x250x64"), 250L * 64);
        using var dit = await AceStepDiT.LoadAsync(accelerator, new LocalFolderModelSource(Path.GetDirectoryName(path)!), "", Path.GetFileName(path), 250, 478);
        using var dX = accelerator.Allocate1D(x); using var dC = accelerator.Allocate1D(ctx); using var dE = accelerator.Allocate1D(enc);
        using var outp = accelerator.Allocate1D<float>(250L * 64);
        var sw = System.Diagnostics.Stopwatch.StartNew();
        dit.Forward(dX.View, dC.View, dE.View, 250, 478, t, tr, outp.View);
        var ours = await outp.View.CopyToHostAsync();
        double e = F2_RelRms(ours, expected);
        string report = $"t {t:G6}, r {tr:G6}: velocity [250, 64] {e:E2} ({sw.Elapsed.TotalSeconds:F1} s)";
        Console.WriteLine("[AceStep] DiT vs ComfyUI decoder tap: " + report);
        Ur_Expect(e < 5e-4, report);                                                          // MEASURED 2026-11-17 CUDA: 1.50e-4 (t = r = 1; 125 patch tokens, inside the window)
        return report;
    });

    /// <summary>
    /// The audio codes to the DiT's hints (<see cref="AceStepCodeDecoder"/>: FSQ, project_out, the detokenizer) on ComfyUI's own 50 codes of the codes-on graph (SpawnGen's
    /// taps) against its detokenizer output [250, 64].
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task AceStep_CodeDecoder_MatchesComfyUiTap() => await RunTest(async accelerator =>
    {
        var path = Ace_Require(accelerator, Path.Combine("diffusion_models", "acestep_v1.5_turbo.safetensors"));
        var d = Ace_Dumps();
        var codeBytes = File.ReadAllBytes(Path.Combine(d, "ace_g1_taps_diffusion_model_0_kw-audio_codes0_i64_1x50.raw"));
        var codes = Enumerable.Range(0, 50).Select(i => (int)BitConverter.ToInt64(codeBytes, i * 8)).ToArray();
        var expected = F2_Raw(Path.Combine(d, "ace_g1_taps_diffusion_model_detokenizer_0_out0_f32_1x250x64.raw"), 250L * 64);
        using var dec = await AceStepCodeDecoder.LoadAsync(accelerator, new LocalFolderModelSource(Path.GetDirectoryName(path)!), "", Path.GetFileName(path), 50);
        using var outp = accelerator.Allocate1D<float>(250L * 64);
        dec.Decode(codes, outp.View);
        double e = F2_RelRms(await outp.View.CopyToHostAsync(), expected);
        string report = $"50 codes -> hints [250, 64] {e:E2}";
        Console.WriteLine("[AceStep] code decoder vs ComfyUI: " + report);
        Ur_Expect(e < 1e-4, report);                                                          // MEASURED 2026-11-17 CUDA: 1.19e-5
        return report;
    });

    /// <summary>
    /// The 5 Hz audio-code LM (qwen_1.7b_ace15, Qwen3 1.7B, 28 layers) teacher-forced on ComfyUI's own run (SpawnGen's telogits2 dump): the positive prompt [478] then the
    /// first 11 code tokens ComfyUI sampled, one causal pass; our final-norm rows against ComfyUI's model.norm taps (call 0: the prompt pass, rows 0 .. 477; call k: the step
    /// that took code k - 1, our row 477 + k). The logits are these rows times the tied embedding table, so equal rows are equal logits.
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task AceStep_CodeLm_TeacherForced_MatchesComfyUi() => await RunTest(async accelerator =>
    {
        var lmPath = Ace_Require(accelerator, Path.Combine("text_encoders", "qwen_1.7b_ace15.safetensors"));
        var d = Ace_Dumps();
        using var idsDoc = System.Text.Json.JsonDocument.Parse(File.ReadAllText(Path.Combine(d, "ace_tl_codes1_norm_ids.json")));
        var calls = idsDoc.RootElement.GetProperty("calls");
        int[] Ids(int call)
        {
            var e = calls[call].GetProperty("ids");
            static IEnumerable<int> Flat(System.Text.Json.JsonElement x) => x.ValueKind == System.Text.Json.JsonValueKind.Array ? x.EnumerateArray().SelectMany(Flat) : new[] { x.GetInt32() };
            if (e.ValueKind == System.Text.Json.JsonValueKind.String) { using var inner = System.Text.Json.JsonDocument.Parse(e.GetString()!); return Flat(inner.RootElement).ToArray(); }
            return Flat(e).ToArray();
        }
        var prompt = Ids(2);                                                                     // the positive LM prompt (call 3 is the CFG negative, left-padded)
        const int Steps = 11, W = 2048;
        var steps = Enumerable.Range(4, Steps).Select(c => Ids(c)[0]).ToArray();                 // the token fed at each step (both rows get the same)
        var seq = prompt.Concat(steps).ToArray();
        string Tap(int call, string shape) => Path.Combine(d, $"ace_tl_codes1_norm_qwen3_2b_transformer_model_norm_{call}_out0_f32_{shape}.raw");
        var src = new LocalFolderModelSource(Path.GetDirectoryName(lmPath)!);
        using var w = await Flux2.Flux2Weights.LoadAsync(accelerator, src, "", Path.GetFileName(lmPath), 28, finalNorm: true);
        using var lm = new Flux2.Flux2TextEncoder(accelerator, w, new[] { 28 }, finalNorm: true, maxTokens: seq.Length) { PreciseActivations = true };
        var ours = await lm.Encode(seq, seq.Length).SubView(0, (long)seq.Length * W).CopyToHostAsync();
        var report = new List<string>(); double worst = 0;
        var prompt0 = F2_Raw(Tap(0, "2x478x2048"), 2L * 478 * W)[..(478 * W)];                // the positive row of the prompt pass
        double ep = F2_RelRms(ours[..(478 * W)], prompt0); worst = Math.Max(worst, ep);
        report.Add($"prompt [478] {ep:E2} (last row {F2_RelRms(ours[(477 * W)..(478 * W)], prompt0[(477 * W)..]):E2})");
        var stepErr = new List<string>();
        for (int k = 1; k <= Steps; k++)
        {
            var tap = F2_Raw(Tap(k, "2x1x2048"), 2L * W)[..W];
            double e = F2_RelRms(ours[((477 + k) * W)..((478 + k) * W)], tap); worst = Math.Max(worst, e);
            stepErr.Add($"{e:E1}");
        }
        report.Add($"steps 1..{Steps}: {string.Join(" ", stepErr)}");
        foreach (var line in report) Console.WriteLine("[AceStep] code LM teacher-forced vs ComfyUI: " + line);
        Ur_Expect(worst < 1e-4, string.Join("; ", report));                                   // MEASURED 2026-11-17 CUDA: prompt 2.4e-5, steps 2.5e-6 .. 5.9e-6
        return string.Join("; ", report);
    });

    /// <summary>
    /// The Oobleck VAE decoder (<see cref="OobleckDecoder"/>, ace_1.5_vae) against ComfyUI's VAEDecodeAudio of the codes-off graph's final latent [1, 64, 250]:
    /// stereo [2, 480000] at 48 kHz (1920 samples per latent frame).
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task AceStep_VaeDecode_MatchesComfyUi() => await RunTest(async accelerator =>
    {
        var path = Ace_Require(accelerator, Path.Combine("vae", "ace_1.5_vae.safetensors"));
        var d = Ace_Dumps();
        var latent = F2_Raw(Path.Combine(d, "ace_g0_out_samples_f32_1x64x250.raw"), 64L * 250);
        var expected = F2_Raw(Path.Combine(d, "ace_g0_audio_waveform_f32_1x2x480000.raw"), 2L * 480000);
        using var vae = await OobleckDecoder.LoadAsync(accelerator, new LocalFolderModelSource(Path.GetDirectoryName(path)!), "", Path.GetFileName(path));
        Ur_Expect(vae.HopLength == 1920 && vae.LatentChannels == 64 && vae.AudioChannels == 2, $"layout hop {vae.HopLength}, latent {vae.LatentChannels}, audio {vae.AudioChannels}");
        using var dLat = accelerator.Allocate1D(latent);
        var sw = System.Diagnostics.Stopwatch.StartNew();
        using var audio = await vae.DecodeAsync(dLat.View, 250);
        var ours = await audio.View.CopyToHostAsync();
        double e = F2_RelRms(ours, expected);
        float peakOurs = ours.Max(MathF.Abs), peakTheirs = expected.Max(MathF.Abs);
        string report = $"audio [2, 480000] {e:E2} (peak ours {peakOurs:F4}, ComfyUI {peakTheirs:F4}; {sw.Elapsed.TotalSeconds:F1} s)";
        Console.WriteLine("[AceStep] VAE decode vs ComfyUI: " + report);
        Ur_Expect(e < 2e-5, report);                                                          // MEASURED 2026-11-16 CUDA: 3.1e-6 (peaks equal, 0.6337)
        return report;
    });
}
