#:project ../../SpawnDev.ILGPU.ML/SpawnDev.ILGPU.ML.csproj
#:property PublishAot=false
// AssistZ's probe table, reproduced on OUR engine (2026-11-03): every clip of a labelled set (AssistZ Lab/label_set_*.jsonl: run, split, clip start,
// truth) is embedded with ClapAudioClassifier.ClassifyWithEmbeddingAsync (zero-shot scores + embedding, one tower run), then a ClapProbe (two
// classes: other / gunfire) is scored (1) leave one TRAIN run out and (2) fitted on all train runs, tested on the test split, against the prompts
// ("gunshot" / "machine gun fire" on top = gunfire, AssistZ's IsGunfire).
//   dotnet run tools/clap-probe-eval/clap_probe_eval.cs -c Release -- <label_set.jsonl> <Lab dir> [cuda|opencl|cpu] [repo] [--noshot <run> ...]
// --noshot: a recording with no shots at all; every event of its replay_emb_s0.jsonl (clip starts) is truth "other" (false-alarm test).
using System.Text.Json;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.Pipelines;

string setPath = args[0], lab = args[1], backend = args.Length > 2 ? args[2] : "cuda";
string repo = args.Length > 3 && !args[3].StartsWith("--") ? args[3] : ClapAudioClassifier.DefaultRepo;
var noShotRuns = new List<string>();
for (int i = 0; i < args.Length - 1; i++) if (args[i] == "--noshot") noShotRuns.Add(args[i + 1]);
string[] names = { "gunshot", "machine gun fire", "explosion", "footsteps", "zombie groaning", "speech", "car engine", "helicopter", "wolf howling",
    "dog barking", "birds chirping", "rain", "thunder", "wind", "door closing", "metal clanking", "music" };
var labels = names.Select(n => "This is a sound of " + n).ToArray();

var setRows = File.ReadLines(setPath).Where(l => l.Trim().Length > 0).Select(l => JsonDocument.Parse(l).RootElement).Select(e => new Row(
    e.GetProperty("run").GetString()!, e.GetProperty("split").GetString()!, e.GetProperty("clip").GetInt64(), e.GetProperty("t").GetDouble(),
    e.GetProperty("truth").GetString()!)).ToList();
var rows = setRows.Concat(noShotRuns.SelectMany(run => File.ReadLines(Path.Combine(lab, "runs", run, "replay_emb_s0.jsonl")).Where(l => l.Trim().Length > 0)
    .Select(l => JsonDocument.Parse(l).RootElement)
    .Select(e => new Row(run, "noshot", e.GetProperty("clip").GetInt64(), e.GetProperty("sample").GetInt64() / 48000.0, "other")))).ToList();
Console.WriteLine($"{rows.Count} clips, {rows.Select(r => r.Run).Distinct().Count()} runs; backend {backend}, {repo}");

var builder = MLContext.Create();
await builder.AllAcceleratorsAsync();
using var ctx = builder.ToContext();
var device = ctx.Devices.FirstOrDefault(d => string.Equals(d.AcceleratorType.ToString(), backend, StringComparison.OrdinalIgnoreCase))
    ?? throw new Exception($"no {backend} device (have {string.Join(", ", ctx.Devices.Select(d => d.AcceleratorType))})");
var accel = device.CreateAccelerator(ctx);
using var _accel = accel;
using var http = new HttpClient { Timeout = Timeout.InfiniteTimeSpan };
using var clap = await ClapAudioClassifier.CreateFromHubAsync(accel, new SpawnDev.ILGPU.ML.Hub.HttpClientModelSource(http), repo);

// embed every clip (mono 0.5 (L + R) from <Lab>/runs/<run>/audio.npy, 48000 samples from `clip`, as AssistZ cuts it)
var audio = new Dictionary<string, float[]>();
var sw = System.Diagnostics.Stopwatch.StartNew();
foreach (var r in rows)
{
    if (!audio.TryGetValue(r.Run, out var stereo)) audio[r.Run] = stereo = ReadNpyFloat32(Path.Combine(lab, "runs", r.Run, "audio.npy"));
    var mono = new float[48000];
    for (int i = 0; i < mono.Length; i++) { long k = (r.Clip + i) * 2; mono[i] = k + 1 < stereo.Length ? 0.5f * (stereo[k] + stereo[k + 1]) : 0f; }
    var res = await clap.ClassifyWithEmbeddingAsync(mono, labels);
    int top = Array.IndexOf(res.Scores, res.Scores.MaxBy(s => s.Probability));
    r.Embedding = res.Embedding;
    r.PromptGunfire = names[top] is "gunshot" or "machine gun fire";
}
Console.WriteLine($"embedded {rows.Count} clips in {sw.Elapsed.TotalSeconds:F1} s ({clap.AudioTowerRuns} audio-tower runs)");

static bool IsGun(Row r) => r.Truth == "gunfire";
ClapProbe Fit(IEnumerable<Row> train)
{
    var t = train.Where(r => r.Truth is "gunfire" or "other").ToList();
    return ClapProbe.FitLogistic(t.Select(r => r.Embedding!).ToList(), t.Select(r => r.Truth).ToList(), classes: new[] { "other", "gunfire" }, checkpoint: repo);
}
string Rate(IEnumerable<Row> g, Func<Row, bool> pred) { var l = g.ToList(); return $"{l.Count(pred)}/{l.Count}"; }

// (1) leave one train run out
var trainRuns = rows.Where(r => r.Split == "train").Select(r => r.Run).Distinct().ToList();
var pooled = new List<(Row R, bool Probe)>();
foreach (var held in trainRuns)
{
    var probe = Fit(rows.Where(r => r.Split == "train" && r.Run != held));
    foreach (var r in rows.Where(r => r.Run == held)) pooled.Add((r, probe.PredictLabel(r.Embedding!) == "gunfire"));
}
var g = pooled.Where(p => IsGun(p.R)).ToList(); var o = pooled.Where(p => p.R.Truth == "other").ToList();
Console.WriteLine($"leave one run out ({trainRuns.Count} runs): gunfire named gunfire  probe {g.Count(p => p.Probe)}/{g.Count}  prompts {g.Count(p => p.R.PromptGunfire)}/{g.Count};"
    + $"  other named gunfire  probe {o.Count(p => p.Probe)}/{o.Count}  prompts {o.Count(p => p.R.PromptGunfire)}/{o.Count}");

// (2) fitted on every train run, tested on the test split
var full = Fit(rows.Where(r => r.Split == "train"));
foreach (var run in rows.Where(r => r.Split == "test").Select(r => r.Run).Distinct())
{
    var test = rows.Where(r => r.Run == run && r.Split == "test").ToList();
    foreach (var truth in test.Select(r => r.Truth).Distinct())
    {
        var grp = test.Where(r => r.Truth == truth).ToList();
        Console.WriteLine($"test {run} truth {truth,-9}: named gunfire  probe {Rate(grp, r => full.PredictLabel(r.Embedding!) == "gunfire")}  prompts {Rate(grp, r => r.PromptGunfire)}");
    }
    var burst = test.Where(r => IsGun(r) && r.T > 118 && r.T < 128).ToList();
    if (burst.Count > 0)
        Console.WriteLine($"test {run} AUG burst 118-128 s: probe {Rate(burst, r => full.PredictLabel(r.Embedding!) == "gunfire")}  prompts {Rate(burst, r => r.PromptGunfire)}");
}
foreach (var run in noShotRuns)
{
    var grp = rows.Where(r => r.Run == run && r.Split == "noshot").ToList();
    Console.WriteLine($"no-shot {run}: called gunfire  probe {Rate(grp, r => full.PredictLabel(r.Embedding!) == "gunfire")}  prompts {Rate(grp, r => r.PromptGunfire)}");
}
File.WriteAllText(Path.Combine(Path.GetTempPath(), "clap_probe_assistz.json"), full.ToJson());

static float[] ReadNpyFloat32(string path)
{
    var bytes = File.ReadAllBytes(path);
    if (bytes[0] != 0x93 || bytes[1] != (byte)'N') throw new Exception($"{path}: not an .npy");
    int major = bytes[6], hlen = major == 1 ? BitConverter.ToUInt16(bytes, 8) : BitConverter.ToInt32(bytes, 8), start = (major == 1 ? 10 : 12) + hlen;
    var header = System.Text.Encoding.ASCII.GetString(bytes, major == 1 ? 10 : 12, hlen);
    if (!header.Contains("'<f4'") || header.Contains("'fortran_order': True")) throw new Exception($"{path}: expected little-endian float32, C order ({header.Trim()})");
    var data = new float[(bytes.Length - start) / 4];
    Buffer.BlockCopy(bytes, start, data, 0, data.Length * 4);
    return data;
}

sealed record Row(string Run, string Split, long Clip, double T, string Truth)
{
    public float[]? Embedding { get; set; }
    public bool PromptGunfire { get; set; }
}
