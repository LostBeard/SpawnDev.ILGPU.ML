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
using SpawnDev.ILGPU.ML.GGUF;

if (args.Contains("--ci"))
    return await RunSelfCheck();

// --parsebench <file.gguf> : split the header-parse stage into read / scan / materialise.
//
// 🔴 WHY IT EXISTS. The 09-15 rewrite took GGUF header parse from 2.2s to 0.6s by reading the header
// region once instead of per field. What it left behind is a grow-and-retry loop: the first read is
// 4 MiB, a Qwen3-class header is ~5.7 MiB, so the FIRST attempt materialises ~70% of a 300K-string
// vocab, throws all of it away on overrun, doubles the buffer and materialises the whole thing again.
// That was written down as "the vocab is parsed ~1.7 times" and never measured. This measures it.
//
// Runs off a MemoryStream as well as the file, so the number is parse cost and not disk weather.
if (args.Contains("--parsebench"))
    return await ParseBench(args.FirstOrDefault(a => !a.StartsWith("--")));

// --tensors=<prefix,prefix,...> : dump RAW per-tensor dims (no template collapse) for tensors whose
// name starts with any prefix. Surfaces per-layer shape VARIANCE the collapsed template view can hide
// (e.g. a frontier arch where global vs sliding layers carry different head_dim / KV-head counts).
var tensorsArg = args.FirstOrDefault(a => a.StartsWith("--tensors", StringComparison.Ordinal));
if (tensorsArg is not null)
{
    var eq = tensorsArg.IndexOf('=');
    var prefixes = (eq >= 0 ? tensorsArg[(eq + 1)..] : "")
        .Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
    var tgt = args.FirstOrDefault(a => !a.StartsWith("--"));
    if (string.IsNullOrWhiteSpace(tgt)) { Usage(); return 2; }
    return await DumpRawTensors(tgt.Trim(), prefixes);
}

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

// ── raw per-tensor dump (no template collapse) for the given name-prefixes ────────────────────────
async Task<int> DumpRawTensors(string source, string[] prefixes)
{
    using var http = new HttpClient { Timeout = TimeSpan.FromSeconds(60) };
    http.DefaultRequestHeaders.UserAgent.ParseAdd("SpawnDev.ILGPU.ML-inspector/1.0");
    await using var stream = await OpenAsync(source, http);
    var model = await GGUFParser.ParseHeaderAsync(stream);

    Console.WriteLine($"\n=== raw tensors ({source}) — prefixes: {string.Join(", ", prefixes)} ===");
    bool Match(string n) => prefixes.Length == 0 || prefixes.Any(p => n.StartsWith(p, StringComparison.Ordinal));
    foreach (var t in model.Tensors.Where(t => Match(t.Name)).OrderBy(t => t.Name, StringComparer.Ordinal))
        Console.WriteLine($"  {t.Name,-36} [{string.Join(", ", t.Dimensions)}]  {t.Type}");
    return 0;
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

// Splits the header stage so each piece has ONE owner, per the lesson that a stage LABEL is not a cause.
async Task<int> ParseBench(string? path)
{
    if (string.IsNullOrWhiteSpace(path) || !File.Exists(path))
    {
        Console.Error.WriteLine("Usage: ModelInspector.Console --parsebench <file.gguf>");
        return 2;
    }

    // 1. What IS the header? Parse it once off the file so we learn its true length.
    var sw = System.Diagnostics.Stopwatch.StartNew();
    GGUFModel probe;
    await using (var fs0 = File.OpenRead(path))
        probe = await GGUFParser.ParseHeaderAsync(fs0);
    var fileMs = sw.Elapsed.TotalMilliseconds;

    int headerBytes = (int)probe.DataStartOffset;
    var fi = new FileInfo(path);
    Console.WriteLine($"file          : {fi.Name}  {fi.Length:N0} B");
    Console.WriteLine($"header        : {headerBytes:N0} B  ({headerBytes / 1048576.0:F2} MiB)"
                      + $"  metadata keys {probe.Metadata.Count}, tensors {probe.Tensors.Length}");
    var vocab = probe.Metadata.TryGetValue("tokenizer.ggml.tokens", out var tv) && tv is string[] ta ? ta.Length : 0;
    var merges = probe.Metadata.TryGetValue("tokenizer.ggml.merges", out var mv) && mv is string[] ma ? ma.Length : 0;
    Console.WriteLine($"vocab strings : tokens {vocab:N0}, merges {merges:N0}");

    // 2. The header bytes, in memory. Everything below runs off THIS, so no measurement includes disk.
    var head = new byte[headerBytes];
    await using (var fs1 = File.OpenRead(path))
    {
        int got = 0;
        while (got < head.Length)
        {
            int n = await fs1.ReadAsync(head.AsMemory(got, head.Length - got));
            if (n == 0) break;
            got += n;
        }
    }

    // 3. ONE materialising parse of a buffer that already holds the whole header. This is the FLOOR:
    //    the work that must happen no matter how the bytes arrive.
    //    Warm once - a first pass pays JIT and dictionary growth that the steady state does not.
    _ = GGUFParser.Parse(head);
    _ = GGUFParser.ParseHeaderAsync(new MemoryStream(head, false)).AsTask().GetAwaiter().GetResult();

    // ⚠️ Interleaved, not one batch then the other. Run to run this machine moves a parse by 20%+, so
    // two batches measured back to back can differ by more than the change being tested. Alternating
    // puts both paths in the same weather, and MIN is what to read: the fastest observed run is the one
    // with the least interference in it.
    // ⚠️ --only isolates ONE path per process. Interleaving them in one process looked fair and was not:
    // the stream path allocates multi-MiB buffers, so it moves the GC schedule under the floor path and
    // the SAME Parse(byte[]) call measured 8.9 ms in one configuration and 14.3 ms in another. A shared
    // managed heap is shared state, and a benchmark that shares it is measuring both arms at once.
    var only = args.FirstOrDefault(a => a.StartsWith("--only=", StringComparison.Ordinal))?["--only=".Length..];
    bool doFloor = only is null or "floor", doStream = only is null or "stream";

    const int Iters = 15;
    var floor = new List<double>(Iters);
    var stream = new List<double>(Iters);
    for (int i = 0; i < Iters; i++)
    {
        if (doFloor) floor.Add(Time(() => GGUFParser.Parse(head)));
        if (doStream) stream.Add(Time(() =>
        {
            using var ms = new MemoryStream(head, writable: false);
            return GGUFParser.ParseHeaderAsync(ms).AsTask().GetAwaiter().GetResult();
        }));
    }
    if (floor.Count == 0) floor.Add(double.NaN);
    if (stream.Count == 0) stream.Add(double.NaN);
    while (floor.Count < Iters) floor.Add(double.NaN);
    while (stream.Count < Iters) stream.Add(double.NaN);
    floor.Sort(); stream.Sort();
    double floorMs = floor[0], streamMs = stream[0];
    double floorMed = floor[Iters / 2], streamMed = stream[Iters / 2];

    // 5. The stage split, straight from the parser's own counters - so the phases SUM to the total
    //    instead of being attributed by argument.
    using (var ms2 = new MemoryStream(head, writable: false))
        await GGUFParser.ParseHeaderAsync(ms2);
    Console.WriteLine();
    Console.WriteLine($"scans .......... {GGUFParser.LastHeaderScanCount} x, {GGUFParser.LastHeaderScanMs,7:F1} ms total");
    Console.WriteLine($"materialise .... {GGUFParser.LastHeaderParseMs,7:F1} ms");
    Console.WriteLine($"stream read .... {GGUFParser.LastHeaderReadMs,7:F1} ms  ({GGUFParser.LastHeaderBytesRead:N0} B)");
    Console.WriteLine($"buffer mgmt .... {GGUFParser.LastHeaderBufferMs,7:F1} ms  (Array.Resize on grow)");

    Console.WriteLine();
    Console.WriteLine($"                                     min      median   ({Iters} interleaved runs)");
    Console.WriteLine($"parse once, in memory .......... {floorMs,8:F1} {floorMed,9:F1} ms   <- the floor");
    Console.WriteLine($"ParseHeaderAsync over memory ... {streamMs,8:F1} {streamMed,9:F1} ms   <- what we actually do");
    Console.WriteLine($"ParseHeaderAsync over the file . {fileMs,8:F1}           ms   (single cold run, includes I/O)");
    Console.WriteLine($"RATIO (stream / floor) ......... {streamMs / floorMs,8:F2} {streamMed / floorMed,9:F2}x");
    Console.WriteLine();
    Console.WriteLine(streamMs / floorMs > 1.25
        ? $"=> the header is materialised about {streamMs / floorMs:F2} times. The overshoot is thrown-away work."
        : "=> the header is materialised about once; there is no re-parse left to remove.");
    return 0;

    static double Time(Func<GGUFModel> f)
    {
        var w = System.Diagnostics.Stopwatch.StartNew();
        var m = f();
        w.Stop();
        // A benchmark that never looks at the result can be optimised into measuring nothing.
        if (m.Tensors.Length == 0) throw new Exception("parse returned no tensors");
        return w.Elapsed.TotalMilliseconds;
    }
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
