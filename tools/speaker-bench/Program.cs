// Speaker identification benchmark on OUR engine: how well does SpeakerEmbeddingPipeline tell voices apart, and is
// it the same model onnxruntime runs?
//
//   node tools/speaker-oracle.mjs <ReDimNet2B6.onnx> <corpusDir> <ort.json> [maxSeconds]     # reference, once
//   dotnet run --project tools/speaker-bench -c Release -- <ReDimNet2B6.onnx> <corpusDir> [ort.json] [--seconds S] [--enroll K]
//
// Corpus: <speaker>_<utterance>.wav (16 kHz mono) - tools write LibriSpeech that way. With --seconds S every clip is
// cut to its first S seconds first (conversational turns are short; the model card says under 2 s is unreliable).
//
// Reported:
//  * parity     - cosine(ours, onnxruntime's) for every clip, when ort.json is given (must be ~1.0000)
//  * closed-set - enrol each speaker from K clips (centroid), identify the remaining clips among all speakers
//  * EER        - verification over every same-speaker / different-speaker clip pair
//  * open-set   - hold a quarter of the speakers OUT of enrolment: at the threshold giving 1% false accepts of
//                 strangers, how many of the enrolled speakers' own clips are still accepted, and how many strangers get in
//  * speed      - ms per embedding
using System.Diagnostics;
using System.Text.Json;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.Pipelines;

var positional = args.Where(a => !a.StartsWith("--") && !(Array.IndexOf(args, a) > 0 && args[Array.IndexOf(args, a) - 1].StartsWith("--"))).ToArray();
if (positional.Length < 2) { Console.Error.WriteLine("usage: <ReDimNet2B6.onnx> <corpusDir> [ort.json] [--seconds S] [--enroll K]"); return 2; }
var modelPath = positional[0];
var corpus = positional[1];
var ortPath = positional.Length > 2 ? positional[2] : null;
double Opt(string name, double fallback) { var i = Array.IndexOf(args, name); return i >= 0 ? double.Parse(args[i + 1]) : fallback; }
var seconds = Opt("--seconds", 0);
var enrollCount = (int)Opt("--enroll", 3);

var builder = MLContext.Create();
await builder.AllAcceleratorsAsync();
var context = builder.ToContext();
var accelerator = await context.CreatePreferredAcceleratorAsync() ?? throw new Exception("no accelerator");
Console.WriteLine($"device   : {accelerator.AcceleratorType} {accelerator.Name}");

var session = await InferenceSession.CreateFromStreamAsync(accelerator,
    new FileStream(modelPath, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 20, FileOptions.Asynchronous | FileOptions.RandomAccess));
var pipeline = new SpeakerEmbeddingPipeline(session, accelerator);

var files = Directory.GetFiles(corpus, "*.wav").OrderBy(f => f, StringComparer.Ordinal).ToList();
// --profile: ONE warm embedding with a device sync after every node, so each op's time includes its own GPU work
// (without PerOpSync a node's time is dispatch only, and the work surfaces at the next sync point - it names the
// wrong op). Prints time per op type and the slowest nodes.
if (args.Contains("--profile"))
{
    var clip = WavDecoder.DecodeWavFile(File.ReadAllBytes(files[0]))!;
    await pipeline.EmbedAsync(clip); await pipeline.EmbedAsync(clip);   // warm: kernels compiled, pools primed
    SpawnDev.ILGPU.ML.Graph.GraphExecutor.PerOpSync = true;
    SpawnDev.ILGPU.ML.Graph.GraphExecutor.CapturedNodeTimingsMs = new Dictionary<string, double>();
    var t0 = Stopwatch.StartNew();
    await pipeline.EmbedAsync(clip);
    var total = t0.Elapsed.TotalMilliseconds;
    var timings = SpawnDev.ILGPU.ML.Graph.GraphExecutor.CapturedNodeTimingsMs!;
    SpawnDev.ILGPU.ML.Graph.GraphExecutor.PerOpSync = false;
    Console.WriteLine($"profile  : {timings.Count} nodes, {total:F0} ms total with a sync after every node (sum of node times {timings.Values.Sum():F0} ms)");
    foreach (var g in timings.GroupBy(kv => kv.Key.Split('_')[1]).Select(g => (Op: g.Key, Ms: g.Sum(x => x.Value), N: g.Count())).OrderByDescending(x => x.Ms).Take(12))
        Console.WriteLine($"  {g.Ms,8:F1} ms  {100 * g.Ms / timings.Values.Sum(),5:F1}%  {g.N,4} x {g.Op}");
    Console.WriteLine("slowest nodes:");
    foreach (var kv in timings.OrderByDescending(kv => kv.Value).Take(12))
        Console.WriteLine($"  {kv.Value,8:F1} ms  {kv.Key}");
    return 0;
}

var ids = new List<string>(); var speakers = new List<string>(); var embeddings = new List<float[]>(); var ms = new List<double>();
// warm-up (kernels compile on first execution)
{ var w = WavDecoder.DecodeWavFile(File.ReadAllBytes(files[0]))!; await pipeline.EmbedAsync(w); await pipeline.EmbedAsync(w); }
foreach (var f in files)
{
    var pcm = WavDecoder.DecodeWavFile(File.ReadAllBytes(f)) ?? throw new InvalidDataException(f);
    if (seconds > 0 && pcm.Length > seconds * 16000) pcm = pcm[..(int)(seconds * 16000)];
    var sw = Stopwatch.StartNew();
    var e = await pipeline.EmbedAsync(pcm);
    ms.Add(sw.Elapsed.TotalMilliseconds);
    var id = Path.GetFileNameWithoutExtension(f);
    ids.Add(id); speakers.Add(id.Split('_')[0]); embeddings.Add(e);
}
Console.WriteLine($"embedded : {files.Count} clips{(seconds > 0 ? $" (first {seconds:0.#} s of each)" : "")}, {speakers.Distinct().Count()} speakers");
ms.Sort();
Console.WriteLine($"speed    : median {ms[ms.Count / 2]:F1} ms, p95 {ms[(int)(ms.Count * 0.95)]:F1} ms per embedding");

if (ortPath != null)
{
    var ort = JsonSerializer.Deserialize<Dictionary<string, float[]>>(File.ReadAllText(ortPath))!;
    double worst = 1; string worstId = "";
    for (int i = 0; i < ids.Count; i++)
    {
        var c = SpeakerEmbeddingPipeline.Cosine(embeddings[i], ort[ids[i]]);
        if (c < worst) { worst = c; worstId = ids[i]; }
    }
    Console.WriteLine($"parity   : lowest cosine vs onnxruntime {worst:F6} ({worstId})  {(worst > 0.999 ? "ok" : "DIVERGES")}");
}

// Group by speaker
var bySpeaker = speakers.Select((s, i) => (s, i)).GroupBy(t => t.s).ToDictionary(g => g.Key, g => g.Select(t => t.i).ToList());
float[] Centroid(IEnumerable<int> idx)
{
    var sum = new float[embeddings[0].Length];
    foreach (var i in idx) for (int d = 0; d < sum.Length; d++) sum[d] += embeddings[i][d];
    var n = MathF.Sqrt(sum.Sum(v => v * v));
    for (int d = 0; d < sum.Length; d++) sum[d] /= n;
    return sum;
}

// closed set
{
    var enrolled = bySpeaker.Where(kv => kv.Value.Count > enrollCount).ToDictionary(kv => kv.Key, kv => Centroid(kv.Value.Take(enrollCount)));
    int right = 0, trials = 0;
    foreach (var (spk, idx) in bySpeaker)
        if (enrolled.ContainsKey(spk))
            foreach (var i in idx.Skip(enrollCount))
            {
                trials++;
                var best = enrolled.OrderByDescending(kv => SpeakerEmbeddingPipeline.Cosine(embeddings[i], kv.Value)).First().Key;
                if (best == spk) right++;
            }
    Console.WriteLine($"closed   : {right}/{trials} identified = {100.0 * right / trials:F1}% ({enrolled.Count} speakers, {enrollCount} clips each to enrol)");
}

// EER over all pairs
{
    var genuine = new List<float>(); var impostor = new List<float>();
    for (int a = 0; a < embeddings.Count; a++)
        for (int b = a + 1; b < embeddings.Count; b++)
            (speakers[a] == speakers[b] ? genuine : impostor).Add(SpeakerEmbeddingPipeline.Cosine(embeddings[a], embeddings[b]));
    var g = genuine.OrderBy(x => x).ToArray(); var im = impostor.OrderBy(x => x).ToArray();
    // first index with value >= thr (lower bound)
    static int Below(float[] sorted, float thr) { int lo = 0, hi = sorted.Length; while (lo < hi) { int mid = (lo + hi) / 2; if (sorted[mid] < thr) lo = mid + 1; else hi = mid; } return lo; }
    double eer = 1, eerThr = 0, bestGap = double.MaxValue;
    foreach (var thr in g.Concat(im).Distinct())
    {
        double frr = Below(g, thr) / (double)g.Length;                 // genuine pairs rejected
        double far = (im.Length - Below(im, thr)) / (double)im.Length; // impostor pairs accepted
        var gap = Math.Abs(frr - far);
        if (gap < bestGap) { bestGap = gap; eer = (frr + far) / 2; eerThr = thr; }
    }
    Console.WriteLine($"EER      : {100 * eer:F2}% at cosine {eerThr:F3}  (genuine pairs {g.Length}, impostor pairs {im.Length}; genuine median {g[g.Length / 2]:F3}, impostor median {im[im.Length / 2]:F3}, impostor p99 {im[(int)(im.Length * 0.99)]:F3})");
}

// open set: a quarter of the speakers are strangers
{
    var spks = bySpeaker.Keys.OrderBy(s => s, StringComparer.Ordinal).ToList();
    var strangers = spks.Where((s, i) => i % 4 == 3).ToHashSet();
    var known = bySpeaker.Where(kv => !strangers.Contains(kv.Key) && kv.Value.Count > enrollCount).ToDictionary(kv => kv.Key, kv => Centroid(kv.Value.Take(enrollCount)));
    var strangerScores = strangers.SelectMany(s => bySpeaker[s]).Select(i => known.Max(kv => SpeakerEmbeddingPipeline.Cosine(embeddings[i], kv.Value))).OrderBy(x => x).ToList();
    var thr = strangerScores[(int)(strangerScores.Count * 0.99)];
    int accepted = 0, total = 0, wrongName = 0;
    foreach (var (spk, c) in known)
        foreach (var i in bySpeaker[spk].Skip(enrollCount))
        {
            total++;
            var ranked = known.OrderByDescending(kv => SpeakerEmbeddingPipeline.Cosine(embeddings[i], kv.Value)).First();
            var score = SpeakerEmbeddingPipeline.Cosine(embeddings[i], ranked.Value);
            if (score >= thr) { if (ranked.Key == spk) accepted++; else wrongName++; }
        }
    Console.WriteLine($"open set : {strangers.Count} strangers ({strangerScores.Count} clips), {known.Count} enrolled; threshold {thr:F3} lets 1% of strangers in -> " +
        $"{100.0 * accepted / total:F1}% of enrolled speakers' clips recognised correctly, {100.0 * wrongName / total:F1}% named as the wrong person");
}
return 0;

