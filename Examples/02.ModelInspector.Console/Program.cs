// ─────────────────────────────────────────────────────────────────────────────────────────────────
//  SpawnDev.ILGPU.ML — Example 02: Model Inspector (console)
//
//  Drop a model PATH or URL and see its structure — architecture, operators, tensors, quantization,
//  and engine compatibility — WITHOUT downloading the whole model. For header-front formats (GGUF,
//  SafeTensors) only the metadata header is streamed, so a multi-GB model inspects from a few KB.
//
//    dotnet run -- path/to/model.onnx
//    dotnet run -- https://host/model.gguf
//    dotnet run -- gemma4:12b            # an Ollama model, by name (resolves manifest -> blob)
//    dotnet run                          # interactive: prompts for a path/URL
//    dotnet run -- --ci                  # offline self-check, exit 0 on success
//
//  Self-contained on purpose: everything is in this one file. The only dependency is
//  SpawnDev.ILGPU.ML for the inspector; no GPU/accelerator is created (inspection is pure parsing).
// ─────────────────────────────────────────────────────────────────────────────────────────────────

using System.Net;
using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using SpawnDev.ILGPU.ML.Onnx;

if (args.Contains("--ci"))
    return await RunSelfCheck();

var target = args.FirstOrDefault(a => !a.StartsWith("--"));
if (string.IsNullOrWhiteSpace(target))
{
    if (Console.IsInputRedirected) { Usage(); return 2; }          // no TTY, no arg -> can't prompt
    Console.Write("Model path or URL (file / http(s) / Ollama name:tag): ");
    target = Console.ReadLine();
    if (string.IsNullOrWhiteSpace(target)) { Usage(); return 2; }
}

try
{
    return await Inspect(target.Trim());
}
catch (Exception ex)
{
    Console.Error.WriteLine($"Inspection failed: {ex.Message}");
    return 1;
}

// ── inspect a single target and print the report ────────────────────────────────────────────────
async Task<int> Inspect(string source)
{
    using var http = new HttpClient { Timeout = TimeSpan.FromSeconds(60) };
    http.DefaultRequestHeaders.UserAgent.ParseAdd("SpawnDev.ILGPU.ML-inspector/1.0");

    await using var stream = await OpenAsync(source, http);

    // ONE stream pass: structure + (for ONNX) operator compatibility, header-only.
    var (info, compat) = await ModelInspectorHelper.InspectWithCompatibilityAsync(stream);
    Report(source, info, compat);
    return 0;
}

// ── resolve a target to a readable stream (file / URL / Ollama ref), reading only the header ──────
async Task<Stream> OpenAsync(string source, HttpClient http)
{
    if (File.Exists(source))
        return File.OpenRead(source);

    var url = source;
    if (!source.StartsWith("http", StringComparison.OrdinalIgnoreCase) && LooksLikeOllamaRef(source))
        url = await ResolveOllamaBlobUrl(source, http);

    if (!url.StartsWith("http", StringComparison.OrdinalIgnoreCase))
        throw new FileNotFoundException($"Not a local file, URL, or Ollama ref: {source}");

    var req = new HttpRequestMessage(HttpMethod.Get, url);
    req.Headers.Range = new RangeHeaderValue(0, 32 * 1024 * 1024 - 1); // header-only; bound the transfer
    var resp = await http.SendAsync(req, HttpCompletionOption.ResponseHeadersRead);
    if (!resp.IsSuccessStatusCode && resp.StatusCode != HttpStatusCode.PartialContent)
        throw new HttpRequestException($"HTTP {(int)resp.StatusCode} {resp.ReasonPhrase} for {url}");
    return await resp.Content.ReadAsStreamAsync();
}

// ── Ollama "name:tag" / "ns/name:tag" -> the GGUF blob URL (the manifest -> blob acquisition flow) ─
async Task<string> ResolveOllamaBlobUrl(string reference, HttpClient http)
{
    int colon = reference.LastIndexOf(':');
    string name = reference[..colon];
    string tag = reference[(colon + 1)..];
    if (!name.Contains('/')) name = "library/" + name;             // official models live under library/

    string manifestUrl = $"https://registry.ollama.ai/v2/{name}/manifests/{tag}";
    Console.WriteLine($"Resolving Ollama manifest: {manifestUrl}");
    var json = await http.GetStringAsync(manifestUrl);
    using var doc = JsonDocument.Parse(json);
    foreach (var layer in doc.RootElement.GetProperty("layers").EnumerateArray())
    {
        var mt = layer.GetProperty("mediaType").GetString() ?? "";
        if (mt.Contains("model"))                                   // the GGUF weights layer
        {
            var digest = layer.GetProperty("digest").GetString();
            long size = layer.TryGetProperty("size", out var s) ? s.GetInt64() : 0;
            Console.WriteLine($"Model layer: {digest} ({size / 1024.0 / 1024.0:F0} MB) — streaming header only");
            return $"https://registry.ollama.ai/v2/{name}/blobs/{digest}";
        }
    }
    throw new InvalidOperationException("No model layer found in the Ollama manifest.");
}

static bool LooksLikeOllamaRef(string s)
    => !s.Contains("://") && !s.Contains('\\')
       && Regex.IsMatch(s, @"^[\w.-]+(/[\w.-]+)?:[\w.-]+$");

// ── pretty-print the inspection result ───────────────────────────────────────────────────────────
static void Report(string source, InspectionResult r, CompatibilityResult c)
{
    static void H(string s) { Console.WriteLine(); Console.WriteLine(s); Console.WriteLine(new string('-', s.Length)); }

    Console.WriteLine();
    Console.WriteLine($"=== {r.GraphName} ===");
    Console.WriteLine($"source     : {source}");
    Console.WriteLine($"producer   : {r.ProducerName} {r.ProducerVersion}".TrimEnd());
    Console.WriteLine($"file size  : {r.FileSizeMB}");
    Console.WriteLine($"params     : {r.TotalParametersFormatted}   weights: {r.TotalWeightMB}");
    Console.WriteLine($"nodes      : {r.NodeCount}   initializers/tensors: {r.InitializerCount}");

    if (r.Inputs.Length > 0) { H("Inputs"); foreach (var t in r.Inputs) Console.WriteLine($"  {t.Name} {t.ShapeStr} {t.DataType}"); }
    if (r.Outputs.Length > 0) { H("Outputs"); foreach (var t in r.Outputs) Console.WriteLine($"  {t.Name} {t.ShapeStr} {t.DataType}"); }

    if (r.Operators.Length > 0)
    {
        H($"Operators / tensor-types (top 15 of {r.Operators.Length})");
        foreach (var o in r.Operators.Take(15)) Console.WriteLine($"  {o.Count,6}  {o.OpType}");
    }

    // GGUF-only: the distinct tensor templates (blk.N collapsed) — surfaces the small norms/scales.
    if (r.TensorTemplates.Length > 0)
    {
        H($"Tensor templates ({r.TensorTemplates.Length})");
        foreach (var t in r.TensorTemplates)
            Console.WriteLine($"  {t.Name,-34} {t.ShapeStr,-18} {t.DataType,-6} x{t.Count}");
    }

    if (r.LargestWeights.Length > 0)
    {
        H("Largest weights (top 10)");
        foreach (var w in r.LargestWeights.Take(10))
            Console.WriteLine($"  {w.SizeFormatted,10}  {w.DataType,-8} {w.ShapeStr,-22} {w.Name}");
    }

    // GGUF-only: the full metadata KV map (arrays summarized) — arch, rope, soft-cap, sliding-window, etc.
    if (r.Metadata.Length > 0)
    {
        H($"Metadata ({r.Metadata.Length} keys)");
        foreach (var m in r.Metadata) Console.WriteLine($"  {m.Key,-44} {m.Value}");
    }

    H("Engine compatibility");
    Console.WriteLine($"  {c.Summary}");
}

static void Usage() => Console.Error.WriteLine(
    """
    Usage: ModelInspector.Console <model-path-or-url>
      <path>          a local .onnx / .gguf / .safetensors / ... file
      <http(s) url>   a model URL (header streamed, no full download)
      <name:tag>      an Ollama model, e.g. gemma4:12b (resolves manifest -> blob)
      --ci            run an offline self-check (exit 0 on success)
    """);

// ── offline self-check: inspect a synthetic GGUF and assert the metadata + tensor templates surface ─
async Task<int> RunSelfCheck()
{
    Console.WriteLine("[--ci] inspector self-check on a synthetic GGUF (offline)...");
    var info = await ModelInspectorHelper.InspectAsync(new MemoryStream(BuildSyntheticGguf()));

    var problems = new List<string>();
    if (info.Metadata.Length != 3) problems.Add($"metadata count {info.Metadata.Length} != 3");
    var hck = info.Metadata.FirstOrDefault(m => m.Key.EndsWith("head_count_kv"));
    if (hck is null || !hck.Value.Contains("i32") || !hck.Value.Contains("8, 8, 1")) problems.Add($"head_count_kv: '{hck?.Value}'");
    var swp = info.Metadata.FirstOrDefault(m => m.Key.EndsWith("sliding_window_pattern"));
    if (swp is null || !swp.Value.Contains("bool") || !swp.Value.Contains("True")) problems.Add($"sliding_window_pattern: '{swp?.Value}'");
    var norm = info.TensorTemplates.FirstOrDefault(t => t.Name == "blk.*.attn_q_norm.weight");
    if (norm is null || norm.Count != 2) problems.Add($"blk.* norm template: count={norm?.Count}");
    if (!info.TensorTemplates.Any(t => t.Name == "token_embd.weight")) problems.Add("token_embd template missing");

    Console.WriteLine($"  metadata keys: {info.Metadata.Length}, tensor templates: {info.TensorTemplates.Length}");
    if (problems.Count > 0)
    {
        Console.Error.WriteLine("SELF-CHECK FAILED:\n  " + string.Join("\n  ", problems));
        return 1;
    }
    Console.WriteLine("SELF-CHECK PASS — inspector surfaces the GGUF metadata KV map + tensor templates.");
    return 0;
}

// Minimal valid GGUF v3 with two metadata arrays (i32 + bool) and two per-layer + one model tensor.
static byte[] BuildSyntheticGguf()
{
    using var ms = new MemoryStream();
    using var bw = new BinaryWriter(ms);
    void Str(string s) { var b = Encoding.UTF8.GetBytes(s); bw.Write((ulong)b.Length); bw.Write(b); }
    void TensorInfo(string n, ulong dim, ulong off) { Str(n); bw.Write((uint)1); bw.Write(dim); bw.Write((uint)0 /*F32*/); bw.Write(off); }

    bw.Write((byte)'G'); bw.Write((byte)'G'); bw.Write((byte)'U'); bw.Write((byte)'F');
    bw.Write((uint)3);          // version
    bw.Write((ulong)3);         // tensor count
    bw.Write((ulong)3);         // metadata KV count

    Str("general.architecture"); bw.Write((uint)8 /*string*/); Str("exampletest");
    Str("exampletest.attention.head_count_kv");
    bw.Write((uint)9 /*array*/); bw.Write((uint)5 /*i32*/); bw.Write((ulong)3); bw.Write(8); bw.Write(8); bw.Write(1);
    Str("exampletest.attention.sliding_window_pattern");
    bw.Write((uint)9 /*array*/); bw.Write((uint)7 /*bool*/); bw.Write((ulong)3); bw.Write((byte)1); bw.Write((byte)1); bw.Write((byte)0);

    TensorInfo("blk.0.attn_q_norm.weight", 4, 0);
    TensorInfo("blk.1.attn_q_norm.weight", 4, 16);
    TensorInfo("token_embd.weight", 8, 32);

    while (ms.Position % 32 != 0) bw.Write((byte)0);   // align to data section
    for (int i = 0; i < 16; i++) bw.Write(0.0f);       // 64-byte data section
    bw.Flush();
    return ms.ToArray();
}
