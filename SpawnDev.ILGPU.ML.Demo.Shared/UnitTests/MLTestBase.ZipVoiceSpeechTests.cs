using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.Pipelines;
using System;
using System.Diagnostics;
using System.Linq;
using System.Net.Http;
using System.Text;
using System.Threading.Tasks;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// Does ZipVoice actually SPEAK in a browser?
///
/// <para>
/// It has spoken on our engine since 2026-08-30 - but only ever in the desktop harness. NOTHING in the
/// browser-capable suite constructed <c>ZipVoicePipeline</c> or <c>IlgpuZipVoiceGraphs</c> before this
/// file, so "ZipVoice works" and "ZipVoice works where the demo runs" were different claims and only the
/// first had evidence. That gap is the same shape as the one that hid a microphone capture which had never
/// been implemented, and the one that let a reversed slice write nothing on every backend.
/// </para>
///
/// <para>
/// ⚠️ Auditing the path first found exactly ONE filesystem dependency in it -
/// <c>ZipVoiceTokenizer.CreateDefault</c> read <c>tokens.txt</c> off disk. The pipeline, the graphs and the
/// feature extraction are all filesystem-free, and <c>IZipVoiceGraphs</c>' own remarks say it is async
/// precisely so it is not "quietly restricted to the desktop" - which, at that last step, it was.
/// <c>CreateFromTokens</c> closes it.
/// </para>
///
/// <para>
/// ⚠️ <b>HeavyModel</b>: this fetches ~185 MB (fm_decoder_int8 124.7 MB, mel_spec_24khz 54.2 MB,
/// text_encoder_int8 5.6 MB). HeavyModel is excluded from every sweep by default, so this runs when asked
/// for and does not tax the release gate.
/// </para>
/// </summary>
public abstract partial class MLTestBase
{
    private const string ZipVoiceRepo = "k2-fsa/ZipVoice";

    /// <summary>
    /// ZipVoice's vocoder - mel spectrogram back to a waveform - reached through the hub's source proxy.
    /// </summary>
    /// <remarks>
    /// <para>
    /// ⚠️ <b>It is NOT <c>wetdog/vocos-mel-24khz-onnx</c>.</b> That repo looks exactly right and is the
    /// wrong direction: it holds the mel EXTRACTOR (audio -> mel), not the vocoder (mel -> audio). The two
    /// files sit 431 bytes apart in size, which is how the wrong one passed for the right one - a decoy that
    /// survives every check except actually listening to the output.
    /// </para>
    /// <para>
    /// ⚠️ The vocoder is published ONLY inside a sherpa-onnx release tarball, and it must be the <b>fp32</b>
    /// package: the int8 package ships an encoder, a decoder and tokens.txt and <b>no vocoder at all</b>
    /// (verified by listing it - 3 members, none of them a vocoder). A <c>.tar.bz2</c> cannot be seeked
    /// into, so the whole archive must be fetched and decompressed by somebody; the hub does it once, on the
    /// machine with the disk, instead of every visitor doing it in a browser tab.
    /// </para>
    /// </remarks>
    private const string VocoderArchive =
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/sherpa-onnx-zipvoice-distill-zh-en-emilia.tar.bz2";
    private const string VocoderMember = "sherpa-onnx-zipvoice-distill-zh-en-emilia/vocos_24khz.onnx";
    private const string SourceProxyBaseUrl = "https://hub.spawndev.com:44365";

    /// <summary>The hub URL that serves one file out of a remote archive as if it were a plain file.</summary>
    private static string ArchiveMemberUrl(string archiveUrl, string member)
        => $"{SourceProxyBaseUrl}/src?url={Uri.EscapeDataString(archiveUrl)}"
         + $"&member={Uri.EscapeDataString(member)}";

    /// <summary>
    /// Ask the hub to cache an archive, and wait until it has, without holding a request open.
    /// </summary>
    /// <remarks>
    /// ⚠️ Do NOT simply request the member and wait. First contact with this archive means the hub fetches
    /// 634 MB and bzip2-decompresses it, which takes minutes - and the gateway in front of the hub answers
    /// <b>504 at 25 seconds</b> while the hub is still working perfectly. A test written the obvious way
    /// fails with a timeout that looks like a broken server and is not one. <c>/src/warm</c> returns 202
    /// immediately and 200 once cached, so waiting happens between requests instead of inside one.
    /// </remarks>
    private static async Task WarmArchiveAsync(HttpClient http, string archiveUrl)
    {
        var warmUrl = $"{SourceProxyBaseUrl}/src/warm?url={Uri.EscapeDataString(archiveUrl)}";
        var sw = Stopwatch.StartNew();
        while (sw.Elapsed < TimeSpan.FromMinutes(25))
        {
            var res = await http.GetAsync(warmUrl);
            if (res.StatusCode == System.Net.HttpStatusCode.NotFound)
                throw new UnsupportedTestException(
                    "the deployed hub has no /src/warm yet - redeploy the hub to arm this test");
            if (res.IsSuccessStatusCode && res.StatusCode != System.Net.HttpStatusCode.Accepted)
                return;                                     // 200: cached and ready
            if (res.StatusCode != System.Net.HttpStatusCode.Accepted)
                throw new Exception($"the hub could not cache the archive: {(int)res.StatusCode} "
                                  + await res.Content.ReadAsStringAsync());
            Console.WriteLine($"[ZipVoice] hub still caching the vocoder archive "
                            + $"({sw.Elapsed.TotalSeconds:F0}s): {await res.Content.ReadAsStringAsync()}");
            await Task.Delay(TimeSpan.FromSeconds(10));
        }
        throw new Exception("the hub did not finish caching the vocoder archive within 25 minutes");
    }

    /// <summary>The librivox fixture's transcript, as WE know it independently of any recogniser.</summary>
    /// <remarks>
    /// ⚠️ The reference transcript must be EXACT. `SpeakAsync`'s own remarks are explicit that anything
    /// present in the audio and missing here bleeds into the start of the generated line - so a sloppy
    /// transcript degrades the clone for a reason nobody would be able to see in the output.
    /// </remarks>
    private const string LibrivoxTranscript = "All LibriVox recordings are in the public domain.";

    // "WasmHeavy" as well as "HeavyModel": this drives ~185 MB of models through a full TTS pipeline, which
    // is exactly the shape the Wasm lane's interpreted-IL budget exists to keep out - it PASSES on every
    // other backend and would only ever be a timeout there. Run it deliberately with
    // PMT_EXCLUDE_CATEGORIES_WASM= plus a name filter.
    [TestMethod(Timeout = 1800000, Category = "HeavyModel,WasmHeavy")]
    public async Task Pipeline_ZipVoice_SpeaksInTheBrowser() => await RunTest(async accelerator =>
    {
        var assets = GetHttpClient();
        if (assets == null) throw new UnsupportedTestException("HttpClient not available");

        // ── the reference voice: real speech with a transcript we know ───────────────────────────────
        var wavBytes = await assets.GetByteArrayAsync("test-audio/librivox-public-domain.wav");
        var reference = WavDecoder.DecodeWavFile(wavBytes)
            ?? throw new Exception("could not decode test-audio/librivox-public-domain.wav");

        // ── the three graphs, int8, straight from HuggingFace ────────────────────────────────────────
        using var http = CreateHuggingFaceHttpClient();
        var sw = Stopwatch.StartNew();
        var encoderBytes = await InferenceSession.DownloadBytesChunkedAsync(
            http, HuggingFaceClient.GetDownloadUrl(ZipVoiceRepo, "zipvoice_distill/text_encoder_int8.onnx"));
        var decoderBytes = await InferenceSession.DownloadBytesChunkedAsync(
            http, HuggingFaceClient.GetDownloadUrl(ZipVoiceRepo, "zipvoice_distill/fm_decoder_int8.onnx"));
        // Through the hub rather than from HuggingFace: this file exists nowhere as a standalone download.
        // Warmed first so the member request itself is always a warm, fast one - see WarmArchiveAsync.
        await WarmArchiveAsync(http, VocoderArchive);
        var vocoderBytes = await InferenceSession.DownloadBytesChunkedAsync(
            http, ArchiveMemberUrl(VocoderArchive, VocoderMember));
        // The vocoder is 54,157,409 bytes. Checked explicitly because the failure this guards against is a
        // PLAUSIBLE wrong file rather than a missing one, and a wrong vocoder does not throw - it renders
        // noise, which every remaining assertion in this test would happily accept as speech.
        if (vocoderBytes.Length != 54_157_409)
            throw new Exception($"the vocoder is {vocoderBytes.Length} bytes, expected 54,157,409. A "
                              + "different file here does not fail loudly, it just makes noise.");
        var tokensBytes = await InferenceSession.DownloadBytesChunkedAsync(
            http, HuggingFaceClient.GetDownloadUrl(ZipVoiceRepo, "zipvoice_distill/tokens.txt"));
        double fetchMs = sw.Elapsed.TotalMilliseconds;

        Console.WriteLine($"[Benchmark] ZipVoice [{accelerator.AcceleratorType}] fetched "
                        + $"{(encoderBytes.Length + decoderBytes.Length + vocoderBytes.Length) / 1048576.0:F1} MB "
                        + $"in {fetchMs / 1000:F1}s");

        // The symbol table comes from CONTENT, not a path - the whole point of this test.
        var tokenizer = ZipVoiceTokenizer.CreateFromTokens(Encoding.UTF8.GetString(tokensBytes));

        using var graphs = IlgpuZipVoiceGraphs.Create(accelerator, encoderBytes, decoderBytes, vocoderBytes);
        using var pipeline = new ZipVoicePipeline(graphs);

        // ── speak ────────────────────────────────────────────────────────────────────────────────────
        const string line = "Paint the sockets in the wall dull green.";
        sw.Restart();
        var result = await pipeline.SpeakAsync(line, LibrivoxTranscript, reference, 16000, tokenizer);
        double speakMs = sw.Elapsed.TotalMilliseconds;

        var audio = result.Audio;
        if (audio == null || audio.Length == 0)
            throw new Exception("ZipVoice produced NO audio - the pipeline ran and returned nothing.");

        // ── assertions that a broken render cannot satisfy ───────────────────────────────────────────
        // Silence is the failure mode to guard hardest: a pipeline that returns zeros has the right
        // length and the right type, and only an amplitude check can tell it from speech.
        float peak = 0f;
        double energy = 0;
        foreach (var v in audio) { peak = MathF.Max(peak, MathF.Abs(v)); energy += (double)v * v; }
        double rms = Math.Sqrt(energy / audio.Length);
        double seconds = audio.Length / (double)result.SampleRate;

        if (peak < 0.01f)
            throw new Exception($"ZipVoice returned effectively SILENCE (peak {peak:F5} over {audio.Length} "
                              + "samples). A pipeline that produces zeros looks identical to one that works "
                              + "at every check except amplitude.");
        if (rms < 0.005)
            throw new Exception($"ZipVoice output RMS {rms:F5} is too low to be speech (peak {peak:F4}).");
        // A ~40-character line is not going to be under half a second or over twenty.
        if (seconds < 0.5 || seconds > 20.0)
            throw new Exception($"ZipVoice produced {seconds:F2}s for a {line.Length}-character line - the "
                              + "duration prediction inside the encoder decides this, so a wild value means "
                              + "the encoder ran wrong rather than the vocoder.");

        Console.WriteLine($"[Benchmark] ZipVoice [{accelerator.AcceleratorType}]: spoke {seconds:F2}s of audio "
                        + $"at {result.SampleRate} Hz in {speakMs / 1000:F2}s "
                        + $"({speakMs / 1000 / seconds:F2}x realtime, lower is better) "
                        + $"| peak {peak:F3} rms {rms:F4}");

        // The SPLIT, not just the total. Where the time goes decides what is worth optimising, and the
        // pipeline already measures it - printing only the total throws that away and invites the next
        // person to guess. The decoder is the stage to watch: it runs Config.NumSteps Euler steps, and the
        // integration between them happens on the HOST, so each step is a GPU round trip on the largest
        // model in the pipeline (fm_decoder_int8, 124.7 MB). That is the same shape that cost the Silero
        // VAD 22.8x before its readbacks were removed.
        Console.WriteLine($"[Benchmark] ZipVoice [{accelerator.AcceleratorType}] stage split: "
                        + $"encoder {result.EncoderMs:F0} ms, decoder {result.DecoderMs:F0} ms, "
                        + $"vocoder {result.VocoderMs:F0} ms, total {result.TotalMs:F0} ms "
                        + $"| decoder is {100 * result.DecoderMs / Math.Max(1, result.TotalMs):F0}% of it");
    });
}
