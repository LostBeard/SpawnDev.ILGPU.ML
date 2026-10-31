#:project ../../SpawnDev.ILGPU.ML/SpawnDev.ILGPU.ML.csproj
#:property JsonSerializerIsReflectionEnabledByDefault=true
#:property PublishAot=false
// Self-consistency on our engine: the problems of pilot_search.py's run, one greedy answer and N samples each (GgufBatchDecoder: the N decoded together), the vote by
// SelfConsistency. Writes one JSON line per problem to OUT (a rerun skips the problems already there); score_sc.py checks the answers with math_verify.
//
//   dotnet run sc_eval.cs -c Release -- ROOT GGUF OUT [N_SAMPLES=8] [MAX_NEW=1024]
//   (ROOT holds MATH-500/test.jsonl and pilot_results.jsonl, the pilot's problems in its order)
using System.Text.Json;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.ILGPU.ML.Preprocessing;

string root = args[0], gguf = args[1], outPath = args[2];
int n = args.Length > 3 ? int.Parse(args[3]) : 8, maxNew = args.Length > 4 ? int.Parse(args[4]) : 1024;
var problems = File.ReadLines(Path.Combine(root, "MATH-500", "test.jsonl")).Select(l => JsonDocument.Parse(l).RootElement)
    .ToDictionary(e => e.GetProperty("unique_id").GetString()!, e => e);
var order = File.ReadLines(Path.Combine(root, "pilot_results.jsonl")).Select(l => JsonDocument.Parse(l).RootElement.GetProperty("unique_id").GetString()!).ToList();
var done = File.Exists(outPath) ? File.ReadLines(outPath).Select(l => JsonDocument.Parse(l).RootElement.GetProperty("unique_id").GetString()!).ToHashSet() : new HashSet<string>();

using var context = MLContext.Create().ToContext();
using var acc = context.GetCudaDevices()[0].CreateAccelerator(context);
using var dec = await GgufBatchDecoder.CreateAsync(acc, File.OpenRead(gguf), maxSeq: 1536, maxBatch: n);
var tok = SentencePieceTokenizer.FromGGUF(dec.Model)!;
string Text(int[] t) { var d = tok.CreateStreamingDecoder(); var sb = new System.Text.StringBuilder(); foreach (var x in t) sb.Append(d.Push(x)); sb.Append(d.Finish()); return sb.ToString(); }
Console.WriteLine($"{acc.Name}: {order.Count} problems, {n} samples, {done.Count} done");

var t0 = System.Diagnostics.Stopwatch.StartNew();
using var fo = new StreamWriter(outPath, append: true);
for (int k = 0; k < order.Count; k++)
{
    if (done.Contains(order[k])) continue;
    var pr = problems[order[k]];
    var msgs = new List<(string, string)> { ("system", "Please reason step by step, and put your final answer within \\boxed{}."), ("user", pr.GetProperty("problem").GetString()!) };
    var (ids, stops) = ChatTemplates.BuildChatPrompt(dec.Model, tok, msgs, thinking: false);
    var stopSet = stops.ToHashSet(); if (tok.EosId >= 0) stopSet.Add(tok.EosId);
    int room = Math.Min(maxNew, dec.MaxSeq - ids.Length - 1);
    var sw = System.Diagnostics.Stopwatch.StartNew();
    var greedy = Text((await dec.GenerateAsync(ids, 1, room, 0f, 1f, 0, stopSet))[0]);
    double tg = sw.Elapsed.TotalSeconds; sw.Restart();
    var sampled = await dec.GenerateAsync(ids, n, room, 0.7f, 0.8f, 1000 * k, stopSet);
    double ts = sw.Elapsed.TotalSeconds;
    var texts = sampled.Select(Text).ToList();
    var vote = SelfConsistency.Vote(texts);
    var rec = new
    {
        unique_id = order[k], level = pr.GetProperty("level").GetInt32(), gold = pr.GetProperty("answer").GetString(),
        greedy, samples = texts, vote = vote.Count > 0 ? vote[0].Answer : null, votes = vote.Count > 0 ? vote[0].Votes : 0,
        vote_member = vote.Count > 0 ? vote[0].Members[0] : -1, groups = vote.Count,
        greedy_seconds = tg, sample_seconds = ts, sample_tokens = sampled.Sum(s => s.Length), row_reads = dec.RowReads,
    };
    fo.WriteLine(JsonSerializer.Serialize(rec)); fo.Flush();
    Console.WriteLine($"{k + 1}/{order.Count} level {rec.level}: vote {rec.vote} ({rec.votes}/{n}, {rec.groups} groups), gold {rec.gold}; greedy {tg:F0} s, {n} samples {rec.sample_tokens} tokens {ts:F0} s ({rec.sample_tokens / ts:F0} tok/s) [{t0.Elapsed.TotalMinutes:F1} min]");
}
