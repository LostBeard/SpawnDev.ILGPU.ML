using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Net.Http;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// The whole hands-free loop, in the browser, on our engine: audio in -> VAD -> Whisper -> ZipVoice -> audio out.
/// </summary>
/// <remarks>
/// <para>
/// Every stage of this already had a passing test. That is exactly why this file exists: "the VAD works",
/// "Whisper works" and "ZipVoice works" are three claims, and "you can talk to it and it talks back" is a
/// fourth that none of them establishes. The stages have to hand real data to each other - a VAD segment
/// has to be something Whisper can transcribe, and Whisper's text has to be something the tokenizer can
/// speak - and only running them in series proves that.
/// </para>
/// <para>
/// ⚠️ The reference voice is the SAME clip the loop is listening to. That is not a shortcut, it is the
/// product: the reply is spoken in the voice of whoever just talked, which is what a voice-cloning TTS in
/// a conversational loop is for.
/// </para>
/// <para>
/// ⚠️ <b>Fixture audio, not a live microphone.</b> A real mic cannot run in a headless gate, and a test
/// that needs someone to speak into it is a test that never runs. The fixture enters through exactly the
/// API the microphone feeds - <c>AcceptWaveformAsync</c> on a 16 kHz float stream, in chunks - so the code
/// under test cannot tell the difference. What this does NOT cover is mic capture itself, which has its
/// own gate.
/// </para>
/// <para>
/// ⚠️ <b>HeavyModel + WasmHeavy</b>: this pulls whisper-tiny plus ZipVoice's three graphs, so it runs when
/// asked for rather than taxing the release gate, and never on the Wasm lane's interpreted-IL budget.
/// </para>
/// </remarks>
public abstract partial class MLTestBase
{
    [TestMethod(Timeout = 2700000, Category = "HeavyModel,WasmHeavy")]
    public async Task HandsFree_SpeechToSpeech_InTheBrowser() => await RunTest(async accelerator =>
    {
        var assets = GetHttpClient();
        if (assets == null) throw new UnsupportedTestException("HttpClient not available");

        if (accelerator.AcceleratorType == AcceleratorType.CPU
            && Environment.GetEnvironmentVariable("ZIPVOICE_ALLOW_CPU") != "1")
            throw new UnsupportedTestException(
                "the ILGPU CPU backend exceeds PMT's 600s outer console cap for ZipVoice's decoder alone; "
              + "the stages are covered individually on the CPU lane");

        var total = Stopwatch.StartNew();

        // ── what the microphone would have produced ─────────────────────────────────────────────────
        var wavBytes = await assets.GetByteArrayAsync("test-audio/librivox-public-domain.wav");
        var heard = WavDecoder.DecodeWavFile(wavBytes)
            ?? throw new Exception("could not decode test-audio/librivox-public-domain.wav");
        double heardSeconds = heard.Length / 16000.0;

        // ── models ──────────────────────────────────────────────────────────────────────────────────
        using var http = CreateHuggingFaceHttpClient();
        var hf = new HuggingFaceClient(http);
        var sw = Stopwatch.StartNew();

        var vadBytes = await assets.GetByteArrayAsync("references/vad/silero_vad.onnx");

        var whisperRepo = ModelHub.KnownModels.WhisperTiny;
        var asrEncBytes = await InferenceSession.DownloadBytesChunkedAsync(
            http, HuggingFaceClient.GetDownloadUrl(whisperRepo, "onnx/encoder_model.onnx"));
        var asrDecBytes = await InferenceSession.DownloadBytesChunkedAsync(
            http, HuggingFaceClient.GetDownloadUrl(whisperRepo, "onnx/decoder_model.onnx"));
        var tokenizerJson = Encoding.UTF8.GetString(await hf.DownloadFileAsync(whisperRepo, "tokenizer.json"));

        var ttsEncBytes = await InferenceSession.DownloadBytesChunkedAsync(
            http, HuggingFaceClient.GetDownloadUrl(ZipVoiceRepo, "zipvoice_distill/text_encoder_int8.onnx"));
        var ttsDecBytes = await InferenceSession.DownloadBytesChunkedAsync(
            http, HuggingFaceClient.GetDownloadUrl(ZipVoiceRepo, "zipvoice_distill/fm_decoder_int8.onnx"));
        var tokensBytes = await InferenceSession.DownloadBytesChunkedAsync(
            http, HuggingFaceClient.GetDownloadUrl(ZipVoiceRepo, "zipvoice_distill/tokens.txt"));
        await WarmArchiveAsync(http, VocoderArchive);
        var vocoderBytes = await InferenceSession.DownloadBytesChunkedAsync(
            http, ArchiveMemberUrl(VocoderArchive, VocoderMember));
        if (vocoderBytes.Length != 54_157_409)
            throw new Exception($"the vocoder is {vocoderBytes.Length} bytes, expected 54,157,409 - a "
                              + "different file here does not fail loudly, it renders noise");
        double loadMs = sw.Elapsed.TotalMilliseconds;
        Console.WriteLine($"[HandsFree] models fetched in {loadMs / 1000:F1}s");

        // ── 1. LISTEN: find the turn, the way a live loop would ─────────────────────────────────────
        sw.Restart();
        using var vad = SileroVad.Create(accelerator, vadBytes);
        using var detector = new VoiceActivityDetector(vad, new VadOptions());
        var turns = new List<SpeechSegment>();
        detector.OnSegment += turns.Add;

        // Fed in 512-sample frames - the same granularity a mic callback delivers, so the endpointer sees
        // the stream arriving rather than one whole clip it could never have in real time.
        const int frame = 512;
        for (int off = 0; off < heard.Length; off += frame)
        {
            int n = Math.Min(frame, heard.Length - off);
            await detector.AcceptWaveformAsync(heard.AsSpan(off, n).ToArray());
        }
        await detector.FlushAsync();     // the speaker stopped talking
        double vadMs = sw.Elapsed.TotalMilliseconds;

        if (turns.Count == 0)
            throw new Exception("the VAD found no speech in a clip that is almost entirely speech - the "
                              + "loop would never start listening");
        var turn = turns.OrderByDescending(t => t.Samples.Length).First();
        Console.WriteLine($"[HandsFree] heard a {turn.DurationSeconds:F2}s turn "
                        + $"(of {heardSeconds:F2}s audio) in {vadMs:F0} ms");

        // ── 2. UNDERSTAND ───────────────────────────────────────────────────────────────────────────
        sw.Restart();
        using var asrEnc = InferenceSession.CreateFromFile(accelerator, asrEncBytes);
        using var asrDec = InferenceSession.CreateFromFile(accelerator, asrDecBytes);
        using var asr = new SpeechRecognitionPipeline(asrEnc, asrDec, accelerator);
        asr.LoadTokenizer(tokenizerJson);
        var heardText = (await asr.TranscribeAsync(turn.Samples, 16000)).Text ?? "";
        double asrMs = sw.Elapsed.TotalMilliseconds;

        if (string.IsNullOrWhiteSpace(heardText))
            throw new Exception("the transcript is empty - the loop has nothing to answer");
        Console.WriteLine($"[HandsFree] understood \"{heardText.Trim()}\" in {asrMs:F0} ms");

        // ⚠️ Assert on CONTENT, not just non-emptiness. A recogniser handed a mis-sliced segment returns
        // confident, fluent, wrong words - and every downstream stage would happily speak them. whisper-tiny
        // mangles the proper noun "LibriVox" on every backend (measured), so the check is on the words it
        // does get right, which are the ones that prove the SEGMENT was correct.
        var norm = Regex.Replace(heardText.ToLowerInvariant(), @"[^a-z0-9 ]", " ");
        norm = Regex.Replace(norm, @"\s+", " ").Trim();
        foreach (var word in new[] { "recordings", "public", "domain" })
            if (!norm.Contains(word))
                throw new Exception($"the transcript is missing \"{word}\": \"{norm}\". The VAD segment "
                                  + "handed to the recogniser was probably cut wrong - each stage passes "
                                  + "its own test in isolation, which is what this one exists to get past.");

        // ── 3. REPLY: spoken back in the voice that was just heard ──────────────────────────────────
        var reply = $"You said: {heardText.Trim()}";
        sw.Restart();
        var tokenizer = ZipVoiceTokenizer.CreateFromTokens(Encoding.UTF8.GetString(tokensBytes));
        using var ttsGraphs = IlgpuZipVoiceGraphs.Create(accelerator, ttsEncBytes, ttsDecBytes, vocoderBytes);
        using var tts = new ZipVoicePipeline(ttsGraphs);
        var spoken = await tts.SpeakAsync(reply, heardText, turn.Samples, 16000, tokenizer);
        double ttsMs = sw.Elapsed.TotalMilliseconds;

        var audio = spoken.Audio;
        if (audio == null || audio.Length == 0)
            throw new Exception("the loop produced no reply audio");

        float peak = 0f;
        double energy = 0;
        foreach (var v in audio) { peak = MathF.Max(peak, MathF.Abs(v)); energy += (double)v * v; }
        double rms = Math.Sqrt(energy / audio.Length);
        double spokenSeconds = audio.Length / (double)spoken.SampleRate;

        // Silence is the failure mode to guard hardest: zeros have the right length and the right type.
        if (peak < 0.01f || rms < 0.005)
            throw new Exception($"the reply is effectively SILENCE (peak {peak:F5}, rms {rms:F5}) - a loop "
                              + "that answers with zeros passes every check except this one");
        if (spokenSeconds < 0.5 || spokenSeconds > 30.0)
            throw new Exception($"the reply is {spokenSeconds:F2}s for {reply.Length} characters");

        // ── the number that decides whether this is usable ──────────────────────────────────────────
        double loopMs = vadMs + asrMs + ttsMs;
        double turnSec = turn.DurationSeconds;
        Console.WriteLine($"[Benchmark] HandsFree [{accelerator.AcceleratorType}]: "
                        + $"vad {vadMs:F0} + asr {asrMs:F0} + tts {ttsMs:F0} = {loopMs:F0} ms "
                        + $"to answer a {turnSec:F2}s turn with {spokenSeconds:F2}s of speech "
                        + $"({loopMs / 1000 / turnSec:F2}x the turn length; under 1.0 keeps up with a talker) "
                        + $"| peak {peak:F3} rms {rms:F4} | models {loadMs / 1000:F1}s | wall "
                        + $"{total.Elapsed.TotalSeconds:F1}s");
        Console.WriteLine($"[HandsFree] replied \"{reply}\" as {spokenSeconds:F2}s of audio in {ttsMs:F0} ms");
    });
}
