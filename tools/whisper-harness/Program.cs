// End-to-end Whisper speech-to-text gate, on the desktop, against a known speech recording.
//
//   dotnet run --project tools/whisper-harness -c Release
//   dotnet run --project tools/whisper-harness -c Release -- <file.wav> [modelDir]
//
// WHY THIS EXISTS: the browser was a terrible place to debug this. Every attempt cost an extension reload,
// a page reload, a WebRTC link, and someone standing near a robot talking - and the audio level varied
// between runs, so a bad transcript could not be told apart from a quiet room. A fixed recording with known
// content removes every one of those variables and turns a multi-minute human-in-the-loop cycle into a
// command. It also runs on CUDA here rather than a contended WebGPU, so it is minutes faster.
//
// The recording is Harvard sentences (phonetically balanced) from the Open Speech Repository, 8 kHz 16-bit
// PCM - it deliberately does NOT match Whisper's 16 kHz, so the resample path is exercised too.
using System.Diagnostics;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.ILGPU.ML.Preprocessing;

SpawnDev.ILGPU.ML.Operators.SliceOperator.CaptureResolvedParams = new Dictionary<string, string>();
if (Environment.GetEnvironmentVariable("VERBOSE") == "1") SpawnDev.ILGPU.ML.InferenceSession.VerboseLogging = true;
var wavPath = args.Length > 0 ? args[0] : @"C:\Users\TJ\Downloads\OSR_us_000_0030_8k.wav";
var modelDir = args.Length > 1
    ? args[1]
    : @"D:\users\tj\Projects\SpawnDev.ILGPU.ML\SpawnDev.ILGPU.ML\SpawnDev.ILGPU.ML.Demo\wwwroot\models\whisper-tiny";

if (!File.Exists(wavPath)) { Console.Error.WriteLine($"no wav at {wavPath}"); return 2; }
if (!Directory.Exists(modelDir)) { Console.Error.WriteLine($"no model dir at {modelDir}"); return 2; }

// ---- Read the WAV ourselves so the harness reports what it actually fed the model -------------------
var (samples, sampleRate, channels) = ReadWav(File.ReadAllBytes(wavPath));
Console.WriteLine($"wav      : {Path.GetFileName(wavPath)}");
Console.WriteLine($"format   : {sampleRate} Hz, {channels} ch, {samples.Length} samples "
                + $"({samples.Length / (double)sampleRate:F1}s)");

double sumSq = 0; float peak = 0;
foreach (var s in samples) { sumSq += (double)s * s; peak = MathF.Max(peak, MathF.Abs(s)); }
var rms = Math.Sqrt(sumSq / samples.Length);
Console.WriteLine($"level    : {20 * Math.Log10(rms):F1} dBFS rms, peak {peak:F3}");
if (rms < 0.001) Console.WriteLine("WARNING  : this recording is essentially silent - a blank transcript would mean nothing.");

// Whisper's window is 30s; anything past that is ignored by the pipeline, so say so rather than
// silently transcribing the first third of a file and calling it a pass.
var windowSeconds = AudioPreprocessor.WhisperMaxSamples / (double)AudioPreprocessor.WhisperSampleRate;
var seconds = samples.Length / (double)sampleRate;
if (seconds > windowSeconds)
    Console.WriteLine($"NOTE     : only the first {windowSeconds:F0}s fit Whisper's window; the rest is ignored.");

// ---- What the encoder is actually being handed ------------------------------------------------------
// Whisper's normalisation puts log-mel in roughly [-1, 1] with a mean somewhere below zero. Values far
// outside that, or a constant, mean the encoder is being fed something it has never seen - and that failure
// is silent: the model just declines to transcribe rather than erroring.
{
    var probe = samples;
    if (sampleRate != AudioPreprocessor.WhisperSampleRate)
        probe = AudioPreprocessor.Resample(probe, sampleRate, AudioPreprocessor.WhisperSampleRate);
    var mel = AudioPreprocessor.ComputeLogMelSpectrogram(probe);
    float mn = float.MaxValue, mx = float.MinValue; double mean = 0;
    foreach (var v in mel) { mn = MathF.Min(mn, v); mx = MathF.Max(mx, v); mean += v; }
    mean /= mel.Length;
    Console.WriteLine($"mel      : {mel.Length / AudioPreprocessor.WhisperMelBins} frames x {AudioPreprocessor.WhisperMelBins}, "
                    + $"min {mn:F3} max {mx:F3} mean {mean:F3}");
    if (mx - mn < 0.01f) Console.WriteLine("WARNING  : mel is essentially constant.");
    // Export for the ORT A/B: same mel, their decoder. Isolates "our mel is wrong" from "our decode loops".
    var melBytes = new byte[mel.Length * 4];
    Buffer.BlockCopy(mel, 0, melBytes, 0, melBytes.Length);
    File.WriteAllBytes(Path.Combine(Path.GetTempPath(), "our_mel.bin"), melBytes);
}

// ---- Load the model ---------------------------------------------------------------------------------
var sw = Stopwatch.StartNew();
var builder = MLContext.Create();
await builder.AllAcceleratorsAsync();
var context = builder.ToContext();
// Not disposed: ILGPU throws an NRE tearing down accelerator child objects at exit, and the process is
// about to end anyway - a crash dump after a successful run would just be noise.
var accelerator = await context.CreatePreferredAcceleratorAsync();
if (accelerator == null) { Console.Error.WriteLine("no accelerator"); return 3; }
Console.WriteLine($"device   : {accelerator.AcceleratorType} {accelerator.Name}");

// Identify the exact bytes. This harness is the fast reference other backends get compared against, and a
// reference running a DIFFERENT export of the same architecture is worse than no reference: whisper-tiny's
// demo export and its onnx-community export decompose LayerNorm differently and apply the attention scale
// at different points, which made a GraphOptimizer bug look for hours like a WebGPU-only defect. Print the
// hash so "same model" is checkable rather than assumed.
static string Sha8(byte[] b) => Convert.ToHexString(System.Security.Cryptography.SHA256.HashData(b))[..16];
var encBytes = File.ReadAllBytes(Path.Combine(modelDir, "encoder_model.onnx"));
var decBytes = File.ReadAllBytes(Path.Combine(modelDir, "decoder_model.onnx"));
Console.WriteLine($"model    : {modelDir}");
Console.WriteLine($"           encoder {encBytes.Length} bytes sha256:{Sha8(encBytes)}");
Console.WriteLine($"           decoder {decBytes.Length} bytes sha256:{Sha8(decBytes)}");
var encoder = InferenceSession.CreateFromFile(accelerator, encBytes);
var decoder = InferenceSession.CreateFromFile(accelerator, decBytes);
// Optional with-past decoder. Present => O(n) decode; absent => the old quadratic full-recompute path, so
// the same harness measures both and a transcript diff between them is a real regression signal.
var withPastPath = Path.Combine(modelDir, "decoder_with_past_model.onnx");
InferenceSession? decoderWithPast = null;
if (File.Exists(withPastPath) && Environment.GetEnvironmentVariable("WHISPER_NO_KVCACHE") != "1")
{
    var wpBytes = File.ReadAllBytes(withPastPath);
    Console.WriteLine($"           with_past {wpBytes.Length} bytes sha256:{Sha8(wpBytes)}");
    decoderWithPast = InferenceSession.CreateFromFile(accelerator, wpBytes);
}
Console.WriteLine($"inputs   : enc[{string.Join(",", encoder.InputNames)}] dec[{string.Join(",", decoder.InputNames)}]");

var pipeline = new SpeechRecognitionPipeline(encoder, decoder, accelerator, decoderWithPast);
Console.WriteLine("kvcache  : " + (pipeline.UsesKVCache ? "ON (decoder_with_past)" : "off - quadratic full-recompute"));
var firstTokens = new List<int>();
pipeline.OnTokenGenerated += (step, tok) => { if (step < 12) firstTokens.Add(tok); };
pipeline.LoadTokenizer(File.ReadAllText(Path.Combine(modelDir, "tokenizer.json")));
Console.WriteLine($"loaded   : {sw.ElapsedMilliseconds}ms");

// ---- Encoder oracle: is the ENCODER right, or the decoder? ------------------------------------------
// The repo ships a fixed pair from ONNX Runtime - tone_mel.bin in, tone_encoder_output.bin out. That mel
// was generated with the old HTK/uncentred preprocessing, so it is NOT what Whisper wants, but that does
// not matter here: it is still a recorded input/output pair for the encoder, so feeding the same input and
// comparing tells us whether OUR encoder computes what ONNX Runtime computed. It separates "encoder is
// broken" from "decoder is broken", which no amount of staring at an empty transcript can.
var refDir = Path.Combine(modelDir, "..", "..", "references", "whisper-tiny");
var refMelPath = Path.GetFullPath(Path.Combine(refDir, "tone_mel.bin"));
var refEncPath = Path.GetFullPath(Path.Combine(refDir, "tone_encoder_output.bin"));
if (File.Exists(refMelPath) && File.Exists(refEncPath))
{
    var refMel = ReadFloats(refMelPath);
    var refEnc = ReadFloats(refEncPath);
    using var melBuf = accelerator.Allocate1D<float>(refMel.Length);
    melBuf.View.BaseView.CopyFromCPU(refMel);
    var melTensor = new SpawnDev.ILGPU.ML.Tensors.Tensor(melBuf.View, new[] { 1, 80, 3000 });
    var outs = await encoder.RunAsync(new Dictionary<string, SpawnDev.ILGPU.ML.Tensors.Tensor>
    {
        [encoder.InputNames[0]] = melTensor,
    });
    var got = outs[encoder.OutputNames[0]];
    var hostGot = new float[got.Data.Length];
    got.Data.BaseView.CopyToCPU(hostGot);

    int n = Math.Min(hostGot.Length, refEnc.Length);
    double worst = 0, scale = 0; int worstAt = -1;
    for (int i = 0; i < n; i++)
    {
        var d = Math.Abs(hostGot[i] - refEnc[i]);
        if (d > worst) { worst = d; worstAt = i; }
        scale = Math.Max(scale, Math.Abs(refEnc[i]));
    }
    Console.WriteLine($"encoder  : ours {hostGot.Length} vs ref {refEnc.Length} floats; "
                    + $"max abs diff {worst:E3} at [{worstAt}], ref scale {scale:F3} "
                    + $"-> {(worst < scale * 0.02 ? "MATCHES ONNX Runtime" : "DIFFERS from ONNX Runtime")}");
}
else Console.WriteLine($"encoder  : (no reference pair at {refDir})");

// ---- Decoder probe: what does step 0 actually look like? --------------------------------------------
// The encoder is verified against ONNX Runtime above, the mel is in range, and the prompt tokens are the
// ones from tokenizer.json - yet the first sampled token is end-of-text. So run ONE decoder step by hand
// and look at the logits: the shape says whether the position offset is even addressing the right row, and
// the top few tokens say whether EOT wins by a mile (input encoded wrongly) or by a hair (something subtle).
{
    var promptTokens = new[] { 50258, 50259, 50359, 50363 };   // SOT, <|en|>, transcribe, notimestamps
    var probeMel = samples;
    if (sampleRate != AudioPreprocessor.WhisperSampleRate)
        probeMel = AudioPreprocessor.Resample(probeMel, sampleRate, AudioPreprocessor.WhisperSampleRate);
    var melArr = AudioPreprocessor.ComputeLogMelSpectrogram(probeMel);
    using var mb = accelerator.Allocate1D<float>(melArr.Length);
    mb.View.BaseView.CopyFromCPU(melArr);
    var encOut = await encoder.RunAsync(new Dictionary<string, SpawnDev.ILGPU.ML.Tensors.Tensor>
    {
        [encoder.InputNames[0]] = new SpawnDev.ILGPU.ML.Tensors.Tensor(mb.View, new[] { 1, 80, 3000 }),
    });
    var hidden = encOut[encoder.OutputNames[0]];

    var idsF = promptTokens.Select(t => (float)t).ToArray();
    using var ib = accelerator.Allocate1D<float>(idsF.Length);
    ib.View.BaseView.CopyFromCPU(idsF);
    var decOut = await decoder.RunAsync(new Dictionary<string, SpawnDev.ILGPU.ML.Tensors.Tensor>
    {
        [decoder.InputNames[0]] = new SpawnDev.ILGPU.ML.Tensors.Tensor(ib.View, new[] { 1, promptTokens.Length }),
        [decoder.InputNames[1]] = hidden,
    });
    var logits = decOut[decoder.OutputNames[0]];
    Console.WriteLine($"decoder  : out '{decoder.OutputNames[0]}' shape [{string.Join(",", logits.Shape)}], {logits.Data.Length} floats");

    var host = new float[logits.Data.Length];
    logits.Data.BaseView.CopyToCPU(host);
    int vocab = logits.Shape[^1];
    int rows = host.Length / vocab;
    for (int row = 0; row < rows; row++)
    {
        var top = Enumerable.Range(0, vocab)
            .Select(i => (Token: i, Logit: host[row * vocab + i]))
            .OrderByDescending(x => x.Logit).Take(5).ToArray();
        Console.WriteLine($"  row {row} (after token {promptTokens[Math.Min(row, promptTokens.Length - 1)]}): "
            + string.Join("  ", top.Select(t => $"{t.Token}={t.Logit:F2}")));
    }
}

// ---- Prompt-length sweep: is only position 0 correct? -----------------------------------------------
// Row 0 predicts <|en|> at logit 26 - the decoder clearly reads the encoder states. Rows 1..3 collapse to
// ~1-2, so end-of-text wins by default. Feeding the prompt one token at a time and looking at the LAST row
// each time separates "later positions are broken" from "this particular prompt is unlucky". Whisper should
// answer: [SOT]->50259 <|en|>, [SOT,en]->50359 <|transcribe|>, [+transcribe]->50363 <|notimestamps|>,
// [+notimestamps]-> a text token.
{
    var full = new[] { 50258, 50259, 50359, 50363 };
    var probe2 = samples;
    if (sampleRate != AudioPreprocessor.WhisperSampleRate)
        probe2 = AudioPreprocessor.Resample(probe2, sampleRate, AudioPreprocessor.WhisperSampleRate);
    var mel2 = AudioPreprocessor.ComputeLogMelSpectrogram(probe2);
    using var mb2 = accelerator.Allocate1D<float>(mel2.Length);
    mb2.View.BaseView.CopyFromCPU(mel2);
    var enc2 = await encoder.RunAsync(new Dictionary<string, SpawnDev.ILGPU.ML.Tensors.Tensor>
    {
        [encoder.InputNames[0]] = new SpawnDev.ILGPU.ML.Tensors.Tensor(mb2.View, new[] { 1, 80, 3000 }),
    });
    var hidden2 = enc2[encoder.OutputNames[0]];

    float[]? row0Reference = null; double causalDrift = 0;
    Console.WriteLine("sweep    : last-row prediction for each prompt length");
    for (int len = 1; len <= full.Length; len++)
    {
        var idsF = full.Take(len).Select(t => (float)t).ToArray();
        using var ib2 = accelerator.Allocate1D<float>(idsF.Length);
        ib2.View.BaseView.CopyFromCPU(idsF);
        var d2 = await decoder.RunAsync(new Dictionary<string, SpawnDev.ILGPU.ML.Tensors.Tensor>
        {
            [decoder.InputNames[0]] = new SpawnDev.ILGPU.ML.Tensors.Tensor(ib2.View, new[] { 1, len }),
            [decoder.InputNames[1]] = hidden2,
        });
        var lg = d2[decoder.OutputNames[0]];
        var h = new float[lg.Data.Length];
        lg.Data.BaseView.CopyToCPU(h);
        int vocab = lg.Shape[^1];
        int last = (len - 1) * vocab;
        var top = Enumerable.Range(0, vocab)
            .Select(i => (Token: i, Logit: h[last + i]))
            .OrderByDescending(x => x.Logit).Take(3).ToArray();
        Console.WriteLine($"  [{string.Join(",", full.Take(len))}] -> "
            + string.Join("  ", top.Select(t => $"{t.Token}={t.Logit:F2}")));

        // CAUSALITY: position 0 attends only to itself, so its logits must not change when tokens are
        // APPENDED after it. If they do, the decoder's self-attention is letting each position see the
        // future - which is exactly the failure shape here (position 0 confident, every later position
        // collapsing to noise so end-of-text wins by default, and an empty transcript).
        if (len == 1) row0Reference = h.Take(vocab).ToArray();
        else if (row0Reference != null)
        {
            double drift = 0;
            for (int i = 0; i < vocab; i++) drift = Math.Max(drift, Math.Abs(h[i] - row0Reference[i]));
            causalDrift = Math.Max(causalDrift, drift);
        }
    }
    Console.WriteLine(causalDrift <= 0.01
        ? $"causality: position 0 stable across lengths (max drift {causalDrift:E2}) - mask looks correct"
        : $"causality: FAIL - position 0 logits move by {causalDrift:F2} when later tokens are appended; "
          + "the decoder self-attention is NOT causally masked");
}

// ---- Transcribe -------------------------------------------------------------------------------------
sw.Restart();
var result = await pipeline.TranscribeAsync(samples, sampleRate);
sw.Stop();

Console.WriteLine();
Console.WriteLine($"tokens   : {(firstTokens.Count == 0 ? "(none generated)" : string.Join(", ", firstTokens))}");
Console.WriteLine();
Console.WriteLine($"TRANSCRIPT ({sw.ElapsedMilliseconds}ms):");
Console.WriteLine($"  \"{result.Text}\"");
Console.WriteLine();

// ---- Slice params: wrong PARAMS or wrong KERNEL? ----------------------------------------------------
// The decoder's causal mask is a Slice of a big precomputed triangular constant, and it came out with only
// row 0 populated. SliceOperator.CaptureResolvedParams records the RESOLVED starts/ends/axes/steps plus
// which resolution path produced them - which separates "we asked for the wrong region" from "we asked
// correctly and the kernel wrote the wrong thing".
foreach (var kv in (SpawnDev.ILGPU.ML.Operators.SliceOperator.CaptureResolvedParams ?? new())
             .Where(k => k.Key.Contains("mask", StringComparison.OrdinalIgnoreCase)
                      || k.Key.Contains("self_attn", StringComparison.OrdinalIgnoreCase))
             .Take(4))
    Console.WriteLine($"[slice] {kv.Key}\n        {kv.Value}");

// ---- Judge ------------------------------------------------------------------------------------------
// The bar is real words, not "it returned something". Every failure so far produced either nothing or one
// token repeated, so those are called out by name rather than lumped into a generic fail.
var text = (result.Text ?? "").Trim();
var words = text.Split(new[] { ' ', '\n', '\r', '\t' }, StringSplitOptions.RemoveEmptyEntries);
var distinct = words.Distinct(StringComparer.OrdinalIgnoreCase).Count();
var letters = text.Count(char.IsLetter);

if (text.Length == 0) { Console.WriteLine("FAIL: empty transcript."); return 1; }
if (distinct <= 2 && words.Length > 5)
{
    Console.WriteLine($"FAIL: degenerate output - {words.Length} tokens but only {distinct} distinct.");
    return 1;
}
if (letters < text.Length / 4)
{
    Console.WriteLine($"FAIL: transcript is mostly non-letters ({letters} letters of {text.Length} chars).");
    return 1;
}
Console.WriteLine($"PASS: {words.Length} words, {distinct} distinct, {letters} letters.");
return 0;

// -----------------------------------------------------------------------------------------------------
// Minimal RIFF/WAVE reader: 16-bit PCM, any rate, downmixed to mono. Written here rather than reused so
// the harness reports the file's REAL rate and channel count instead of assuming them.
static (float[] Samples, int SampleRate, int Channels) ReadWav(byte[] data)
{
    if (data.Length < 44) throw new InvalidDataException("too short to be a WAV");
    if (data[0] != 'R' || data[1] != 'I' || data[2] != 'F' || data[3] != 'F') throw new InvalidDataException("not RIFF");
    if (data[8] != 'W' || data[9] != 'A' || data[10] != 'V' || data[11] != 'E') throw new InvalidDataException("not WAVE");

    int pos = 12, channels = 1, bits = 16, rate = 16000;
    while (pos + 8 <= data.Length)
    {
        var id = System.Text.Encoding.ASCII.GetString(data, pos, 4);
        int size = BitConverter.ToInt32(data, pos + 4);
        int body = pos + 8;
        if (id == "fmt ")
        {
            channels = BitConverter.ToInt16(data, body + 2);
            rate = BitConverter.ToInt32(data, body + 4);
            bits = BitConverter.ToInt16(data, body + 14);
        }
        else if (id == "data")
        {
            if (bits != 16) throw new NotSupportedException($"{bits}-bit WAV not supported (16-bit PCM only)");
            int count = Math.Min(size, data.Length - body) / 2;
            int frames = count / Math.Max(1, channels);
            var outp = new float[frames];
            for (int f = 0; f < frames; f++)
            {
                int sum = 0;
                for (int c = 0; c < channels; c++)
                    sum += BitConverter.ToInt16(data, body + (f * channels + c) * 2);
                outp[f] = sum / (float)channels / 32768f;
            }
            return (outp, rate, channels);
        }
        pos = body + size + (size & 1);   // chunks are word-aligned
    }
    throw new InvalidDataException("no data chunk");
}

static float[] ReadFloats(string path)
{
    var bytes = File.ReadAllBytes(path);
    var outp = new float[bytes.Length / 4];
    Buffer.BlockCopy(bytes, 0, outp, 0, outp.Length * 4);
    return outp;
}

