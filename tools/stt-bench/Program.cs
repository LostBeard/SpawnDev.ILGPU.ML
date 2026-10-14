// Speech-to-text benchmark on OUR engine: word error rate and latency per model, on a folder of utterances
// (<id>.wav 16 kHz mono + <id>.txt reference). Leaderboards disagree with each other; this measures the model on the
// audio we actually care about, with the pipeline we actually ship.
//
//   dotnet run --project tools/stt-bench -c Release -- <corpusDir> parakeet <modelDir> [--half|--int8] [--limit N]
//   dotnet run --project tools/stt-bench -c Release -- <corpusDir> whisper  <modelDir> [--limit N]
//
// WER = (substitutions + deletions + insertions) / reference words over the whole set, after lower-casing and
// dropping everything except letters, digits and apostrophes. Latency is the engine's own time for one utterance
// (audio already in memory) after two untimed warm-up utterances; RTF = latency / audio duration.
using System.Diagnostics;
using System.Text;
using System.Text.RegularExpressions;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.Pipelines;

var positional = args.Where(a => !a.StartsWith("--")).ToArray();
if (positional.Length < 3) { Console.Error.WriteLine("usage: <corpusDir> parakeet|whisper <modelDir> [--half] [--int8] [--limit N]"); return 2; }
var corpus = positional[0];
var engine = positional[1];
var modelDir = positional[2];
var half = args.Contains("--half");
var int8 = args.Contains("--int8");
var limitAt = Array.IndexOf(args, "--limit");
var limit = limitAt >= 0 ? int.Parse(args[limitAt + 1]) : int.MaxValue;

var everyAt = Array.IndexOf(args, "--every");
var every = everyAt >= 0 ? int.Parse(args[everyAt + 1]) : 1;   // every Nth utterance: a spread of speakers from a big set
var items = Directory.GetFiles(corpus, "*.wav").OrderBy(f => f, StringComparer.Ordinal)
    .Where((w, i) => i % every == 0)
    .Select(w => (Wav: w, Ref: File.ReadAllText(Path.ChangeExtension(w, ".txt")).Trim())).Take(limit).ToList();
Console.WriteLine($"corpus   : {corpus} ({items.Count} utterances)");

var builder = MLContext.Create();
await builder.AllAcceleratorsAsync();
var context = builder.ToContext();
var accelerator = await context.CreatePreferredAcceleratorAsync() ?? throw new Exception("no accelerator");
Console.WriteLine($"device   : {accelerator.AcceleratorType} {accelerator.Name}");

async Task<InferenceSession> Load(string path, bool lowPrecision = false)
{
    var data = path + ".data";
    var graph = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 20, FileOptions.Asynchronous | FileOptions.RandomAccess);
    var external = File.Exists(data) ? new FileStream(data, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 20, FileOptions.Asynchronous | FileOptions.RandomAccess) : null;
    return await InferenceSession.CreateFromStreamAsync(accelerator, graph, externalDataStream: external,
        weightStorage: lowPrecision ? WeightStorage.Half : WeightStorage.Source);
}

Func<float[], Task<(string Text, double Ms)>> transcribe;
var loadClock = Stopwatch.StartNew();
if (engine == "parakeet")
{
    var pre = await Load(Path.Combine(modelDir, "nemo128.onnx"));
    var enc = await Load(Path.Combine(modelDir, int8 ? "encoder-model.int8.onnx" : "encoder-model.onnx"), half);
    var dec = await Load(Path.Combine(modelDir, "decoder_joint-model.onnx"));
    var pipeline = new ParakeetTdtPipeline(pre, enc, dec, accelerator, ParakeetTdtPipeline.ParseVocabulary(File.ReadAllText(Path.Combine(modelDir, "vocab.txt"))));
    transcribe = async s => { var r = await pipeline.TranscribeAsync(s); return (r.Text, r.InferenceTimeMs); };
}
else if (engine == "whisper")
{
    var enc = await Load(Path.Combine(modelDir, "encoder_model.onnx"));
    var dec = await Load(Path.Combine(modelDir, "decoder_model.onnx"));
    var withPastPath = Path.Combine(modelDir, "decoder_with_past_model.onnx");
    var past = File.Exists(withPastPath) ? await Load(withPastPath) : null;
    var pipeline = new SpeechRecognitionPipeline(enc, dec, accelerator, past);
    pipeline.LoadTokenizer(File.ReadAllText(Path.Combine(modelDir, "tokenizer.json")));
    transcribe = async s => { var r = await pipeline.TranscribeAsync(s, 16000); return (r.Text, r.InferenceTimeMs); };
}
else { Console.Error.WriteLine($"unknown engine '{engine}'"); return 2; }
Console.WriteLine($"loaded   : {engine} {Path.GetFileName(modelDir.TrimEnd('/', '\\'))}{(half ? " (fp16 weights)" : "")}{(int8 ? " (int8)" : "")} in {loadClock.Elapsed.TotalSeconds:F1} s");

static string[] Words(string s) =>
    Regex.Replace(s.ToLowerInvariant().Replace('-', ' '), @"[^a-z0-9' ]", " ").Split(' ', StringSplitOptions.RemoveEmptyEntries);

static int EditDistance(string[] a, string[] b)
{
    var d = new int[a.Length + 1, b.Length + 1];
    for (int i = 0; i <= a.Length; i++) d[i, 0] = i;
    for (int j = 0; j <= b.Length; j++) d[0, j] = j;
    for (int i = 1; i <= a.Length; i++)
        for (int j = 1; j <= b.Length; j++)
            d[i, j] = Math.Min(Math.Min(d[i - 1, j] + 1, d[i, j - 1] + 1), d[i - 1, j - 1] + (a[i - 1] == b[j - 1] ? 0 : 1));
    return d[a.Length, b.Length];
}

static float[] ReadWav(string path)
{
    var b = File.ReadAllBytes(path);
    return WavDecoder.DecodeWavFile(b) ?? throw new InvalidDataException(path);
}

// warm-up: kernels compile on first execution
for (int i = 0; i < Math.Min(2, items.Count); i++) await transcribe(ReadWav(items[i].Wav));

int refWords = 0, errors = 0;
var latencies = new List<double>(); var rtfs = new List<double>();
var rows = new List<(string Id, double Wer, string Ref, string Got)>();
foreach (var (wav, reference) in items)
{
    var samples = ReadWav(wav);
    var (text, ms) = await transcribe(samples);
    var r = Words(reference); var g = Words(text);
    var e = EditDistance(r, g);
    refWords += r.Length; errors += e;
    latencies.Add(ms); rtfs.Add(ms / (samples.Length / 16000.0 * 1000.0));
    rows.Add((Path.GetFileNameWithoutExtension(wav), e / (double)Math.Max(1, r.Length), reference, text));
}
latencies.Sort(); rtfs.Sort();
double Pct(List<double> v, double p) => v[Math.Min(v.Count - 1, (int)(v.Count * p))];
Console.WriteLine($"WER      : {100.0 * errors / refWords:F2}%  ({errors} errors / {refWords} reference words, {items.Count} utterances)");
Console.WriteLine($"latency  : median {Pct(latencies, 0.5):F0} ms, p95 {Pct(latencies, 0.95):F0} ms, max {latencies[^1]:F0} ms");
Console.WriteLine($"RTF      : median {Pct(rtfs, 0.5):F3}, p95 {Pct(rtfs, 0.95):F3}  (1.0 = real time)");
Console.WriteLine("worst utterances:");
foreach (var w in rows.OrderByDescending(r => r.Wer).Take(5))
    Console.WriteLine($"  {w.Wer:P0} {w.Id}\n    ref: {w.Ref}\n    got: {w.Got}");
return 0;
