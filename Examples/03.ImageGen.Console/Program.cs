// ─────────────────────────────────────────────────────────────────────────────────────────────────
//  SpawnDev.ILGPU.ML — Example 03: Image Generation (console)
//
//  Everything the browser demo's /generate page does, on native ILGPU GPU kernels (CUDA / OpenCL), no ONNX Runtime:
//    FLUX 3 Image Klein 5.6B (16 steps) with the Qwen3 4B caption writer, FLUX.2 klein 4B (4 steps), or SD-Turbo (one step);
//    up to 10 reference images to edit or combine (FLUX); 1 / 4 / 8 / any number of pictures per prompt (consecutive seeds);
//    sizes 512 / 768 / 1024 (FLUX) or 256 / 512 (SD-Turbo); SD-Turbo's steps, guidance and negative prompt; 4-bit or 8-bit FLUX weights.
//  With NO prompt argument it drops into a continuous session: the model loads ONCE, then you type a prompt, get pictures, and keep going until /exit.
//
//    dotnet run                                          # interactive, SD-Turbo (the default model)
//    dotnet run -- a photo of a cat                      # one-shot with SD-Turbo
//    dotnet run -- "a red fox in snow" --model flux2 --size 768 --pictures 4 --seed 7
//    dotnet run -- "a lighthouse in a storm" --model flux3                         # the caption writer is on: Qwen3 expands the prompt first
//    dotnet run -- "give the knight a red plume" --model flux2 --ref knight.png    # edit a picture (FLUX takes up to 10 --ref)
//    dotnet run -- --ci                                  # fixed prompt + seed; asserts every picture is real (non-flat)
//
//  Options: --model flux3|flux2|sd  --seed N  --pictures N  --size N  --out PATH  --ref PATH (repeatable)  --no-caption  --bits 4|8
//           --steps N --guidance F --negative TEXT (SD-Turbo)  --cache DIR (where models are kept)  --repeat N  --prompts "a|b|c"
//  In-session commands (/help): /model /size /pictures /ref /clearrefs /caption /negative /steps /guidance /seed /out /exit
//
//  Output is PNG (or BMP when --out ends in .bmp). More than one picture: the seed is added to the file name.
//  The accelerator is created here (the app owns it); models stream from the SpawnDev models host on first run (cached after, resumable).
//  NOTE: first run downloads GBs of weights (FLUX 3 5.7 GB + 2.4 GB for the caption writer, FLUX.2 4.6 GB, SD-Turbo 2.5 GB).
//  FLUX 3 weights are under the FLUX Kommunity License (non-commercial use).
// ─────────────────────────────────────────────────────────────────────────────────────────────────

using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using ILGPU.Runtime.OpenCL;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.Assets;
using SpawnDev.ILGPU.ML.Flux2;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.ILGPU.ML.Tensors;

// ── args: prompt words (non-flag) + options ───────────────────────────────────────────────────────
var o = new Options();
var promptWords = new List<string>();
for (int i = 0; i < args.Length; i++)
{
    switch (args[i])
    {
        case "--ci": o.Ci = true; break;
        case "--seed" when i + 1 < args.Length: o.Seed = int.Parse(args[++i]); break;
        case "--out" when i + 1 < args.Length: o.Out = args[++i]; break;
        case "--repeat" when i + 1 < args.Length: o.Repeat = Math.Max(1, int.Parse(args[++i])); break;
        case "--prompts" when i + 1 < args.Length:
            o.PromptSeq = args[++i].Split('|', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).ToList(); break;
        case "--model" when i + 1 < args.Length: o.Engine = ParseEngine(args[++i]); break;
        case "--pictures" when i + 1 < args.Length: o.Pictures = Math.Max(1, int.Parse(args[++i])); break;
        case "--size" when i + 1 < args.Length: o.Size = int.Parse(args[++i]); break;
        case "--ref" when i + 1 < args.Length: o.Refs.Add(args[++i]); break;
        case "--no-caption": o.Caption = false; break;
        case "--bits" when i + 1 < args.Length: o.Bits = int.Parse(args[++i]) == 8 ? 8 : 4; break;
        case "--steps" when i + 1 < args.Length: o.Steps = Math.Max(1, int.Parse(args[++i])); break;
        case "--guidance" when i + 1 < args.Length: o.Guidance = float.Parse(args[++i], System.Globalization.CultureInfo.InvariantCulture); break;
        case "--negative" when i + 1 < args.Length: o.Negative = args[++i]; break;
        case "--cache" when i + 1 < args.Length: o.Cache = args[++i]; break;
        default: if (!args[i].StartsWith("--")) promptWords.Add(args[i]); break;
    }
}

string prompt = string.Join(' ', promptWords);
if (o.PromptSeq is { Count: > 0 } && string.IsNullOrWhiteSpace(prompt))
    prompt = o.PromptSeq[0];                                      // satisfy the has-prompt guard; Run iterates the full sequence
if (o.Ci)
{
    o.Seed ??= 42;                                                // deterministic pictures for the self-check
    if (string.IsNullOrWhiteSpace(prompt)) prompt = "a photo of a cat";
}

// No prompt + a real console (not piped) + not CI → continuous interactive session. A prompt arg or --ci
// is a one-shot. No prompt + piped/redirected input is the usage error (nothing to generate, can't ask).
bool interactive = string.IsNullOrWhiteSpace(prompt) && !o.Ci && !Console.IsInputRedirected;
if (string.IsNullOrWhiteSpace(prompt) && !o.Ci && !interactive)
{
    Console.Error.WriteLine("Usage: ImageGen.Console <prompt words> [--model flux3|flux2|sd] [--seed N] [--pictures N] [--size N] [--ref file]... [--out file.png] [--ci]   (no prompt = interactive session)");
    return 2;
}

try
{
    return await Run(prompt, o, interactive);
}
catch (Exception ex)
{
    Console.Error.WriteLine($"Generation failed: {ex.Message}");
    return 1;
}

async Task<int> Run(string firstPrompt, Options o, bool interactive)
{
    // The application owns the accelerator (library code never disposes it). Prefer CUDA (the verified path),
    // then OpenCL, then whatever ILGPU prefers.
    using var context = MLContext.Create().ToContext();
    var cuda = context.GetCudaDevices();
    var opencl = context.GetCLDevices();
    Device device = cuda.Count > 0 ? (Device)cuda[0]
                  : opencl.Count > 0 ? (Device)opencl[0]
                  : context.GetPreferredDevice(preferCPU: false);
    using var accelerator = device.CreateAccelerator(context);
    if (Environment.GetEnvironmentVariable("ML_VERBOSE") == "1") InferenceSession.VerboseLogging = true;
    if (Environment.GetEnvironmentVariable("WL_TRACE") == "1") InferenceSession.TraceWeightLoad = true; // per-tensor upload attribution
    Console.WriteLine($"Accelerator: {accelerator.Name} ({accelerator.AcceleratorType})");

    // Model acquisition: the SpawnDev models host streams the weights over plain HTTP (Hugging Face is the fallback), cached on disk
    // after the first run. HttpClientModelSource + FileModelStore are the desktop pair (HubModelSource + OPFS is the browser one), and
    // both satisfy IModelSource so the pipelines are the same. An interrupted download RESUMES on the next run.
    using var http = new HttpClient { Timeout = Timeout.InfiniteTimeSpan };
    var store = o.Cache != null ? new FileModelStore(o.Cache) : FileModelStore.Default();
    var source = new HttpClientModelSource(http, store);
    Console.WriteLine($"Model cache: {store.RootDirectory}");

    using var gen = new ImageGenerator(accelerator, source, (repo, file, progress) => source.OpenAsync(repo, file, progress)) { FluxBits = o.Bits };
    int lastLoadPct = -1;
    string lastStatus = "";
    gen.Status = s => { if (s.Contains("downloading") || s.Contains("writing") || s.Contains(":upload") || s == lastStatus) return; lastStatus = s; Console.WriteLine($"  [{s}]"); };
    gen.LoadProgress = p => { int q = (int)p / 10 * 10; if (q != lastLoadPct) { lastLoadPct = q; Console.WriteLine($"  [load {q}%]"); } };
    gen.StepProgress = (step, total) => { if (step > 0) Console.Write($"\r  step {step}/{total}   "); if (step == total) Console.WriteLine(); };
    int captionShown = 0;
    gen.CaptionText = text => { if (text.Length - captionShown >= 400 || text.Length < captionShown) { captionShown = text.Length; Console.WriteLine($"  [caption {text.Length} chars...]"); } };

    // Loaded ONCE and reused for every generation below — the load is the expensive part; an interactive session must never pay it per image.
    Console.WriteLine($"Loading {EngineName(o.Engine)} (first run downloads the weights)...");
    await gen.LoadAsync(o.Engine);

    if (o.Engine == ImageEngine.SdTurbo)
    {
        var pipe = gen.SdPipeline!;
        // Introspection for the tiled-decode weight walk: dump the VAE up-block node sequence (OpType +
        // inputs→outputs) so we can see the real conv/γ/β tensor names. VAE_INTROSPECT=1 dumps + exits.
        if (Environment.GetEnvironmentVariable("VAE_INTROSPECT") == "1")
        {
            var vae = pipe.VaeDecoder!;
            Console.WriteLine($"[INTROSPECT] VAE nodes={vae.NodeCount}, weights={vae.WeightCount}");
            for (int i = 0; i < vae.NodeCount; i++)
            {
                var (op, ins, outs) = vae.GetNode(i);
                string outName = outs.Length > 0 ? outs[0] : "";
                Console.WriteLine($"  [{i,3}] {op,-22} in=[{string.Join(", ", ins)}] -> {outName}");
            }
            Console.WriteLine("[INTROSPECT] done"); return 0;
        }

        // TE_INTROSPECT=1: dump the CLIP text_encoder node sequence (the Range keystone chain). Focus the walk
        // on the Range op + its input cone so we can see exactly what feeds /text_model/Cast_output_0.
        if (Environment.GetEnvironmentVariable("TE_INTROSPECT") == "1")
        {
            var te = pipe.TextEncoder!;
            Console.WriteLine($"[TE-INTROSPECT] text_encoder nodes={te.NodeCount}, weights={te.WeightCount}");
            for (int i = 0; i < te.NodeCount; i++)
            {
                var (op, ins, outs) = te.GetNode(i);
                string outName = outs.Length > 0 ? outs[0] : "";
                // Print the shape/index-math ops (skip the bulk transformer MatMul/Add/LayerNorm body)
                if (op is "Range" or "Shape" or "Gather" or "Cast" or "Slice" or "Unsqueeze" or "Squeeze"
                        or "Concat" or "ConstantOfShape" or "Reshape" or "Expand" or "Mul" or "Add" or "Sub"
                        or "Div" or "Where" or "Equal" or "Constant" or "Sub" or "Min" or "Max")
                    Console.WriteLine($"  [{i,3}] {op,-16} in=[{string.Join(", ", ins)}] -> {outName}");
            }
            Console.WriteLine("[TE-INTROSPECT] done"); return 0;
        }

        // SDTURBO_ELIDE=1: force dispatch-elide + shape-subgraph-fold ON (the keystone repro). This is the
        // WebGPU normal-gen config that throws at the CLIP text_model Range op; reproduces on CUDA too since
        // ShapeInterpElideDispatch forces the CPU shape interpreter on all backends. SDTURBO_ELIDE_DIAG=1 also
        // logs every shape op the interpreter collapses to EMPTY (first non-empty-input line = the root op).
        if (Environment.GetEnvironmentVariable("SDTURBO_ELIDE") == "1")
        {
            SpawnDev.ILGPU.ML.Graph.GraphExecutor.ShapeInterpElideDispatch = true;
            SpawnDev.ILGPU.ML.Graph.GraphCompiler.ShapeSubgraphFoldEnabled = true;
            Console.WriteLine("[SDTURBO_ELIDE] dispatch-elide + shape-subgraph-fold ENABLED");
        }
        else if (Environment.GetEnvironmentVariable("SDTURBO_ELIDE") == "0")
        {
            SpawnDev.ILGPU.ML.Graph.GraphExecutor.ShapeInterpElideDispatch = false;
            SpawnDev.ILGPU.ML.Graph.GraphCompiler.ShapeSubgraphFoldEnabled = false;
            Console.WriteLine("[SDTURBO_ELIDE] dispatch-elide + shape-subgraph-fold DISABLED");
        }
        if (Environment.GetEnvironmentVariable("SDTURBO_ELIDE_DIAG") == "1")
        {
            SpawnDev.ILGPU.ML.Graph.GraphExecutor.LogEmptyShapeInterp = true;
            Console.WriteLine("[SDTURBO_ELIDE_DIAG] empty-shape-interp logging ENABLED");
        }
        BufferPool.TrackPeaks = true;
        if (Environment.GetEnvironmentVariable("PEAK_COMPOSITION") == "1")
            BufferPool.TrackLivePeakComposition = true;
        if (Environment.GetEnvironmentVariable("SDTURBO_LEAKTRACE") == "1")
        {
            BufferPool.LogLargeRunAllocs = true;   // log large NEW allocs during a run (per-gen not-Returned leak hunt)
            BufferPool.LargeRunAllocThresholdBytes = 8L * 1024 * 1024; // 8 MiB
            SpawnDev.ILGPU.ML.Graph.GraphExecutor.LogLeakedBuffers = true; // run-end: name leaked buffers + stuck refcount
        }
    }

    if (!interactive)
    {
        // A batch of generations in ONE resident process. --prompts "a|b|c" runs each distinct prompt;
        // --repeat N runs the same prompt N times. Generation 1 is cold; the warm shape-readback cache
        // finalizes from two probe runs (ideally different prompts) and is active from generation 3 on —
        // that warm path is what matters for perf (the model stays resident, users generate many images).
        // Distinct out names per iteration so they can be diffed (a bit-identical repeat proves the warm
        // cache preserved output). NODETIME (SDTURBO_NODE_TIMING=1) prints the per-gen readback split.
        var seq = o.PromptSeq is { Count: > 0 } ? o.PromptSeq : Enumerable.Repeat(firstPrompt, o.Repeat).ToList();
        if (seq.Count > 1)
        {
            int rc = 0;
            for (int r = 0; r < seq.Count; r++)
            {
                string? op = o.Out == null ? null
                    : Path.Combine(Path.GetDirectoryName(o.Out) is { Length: > 0 } d ? d : ".",
                        $"{Path.GetFileNameWithoutExtension(o.Out)}_{r}{Path.GetExtension(o.Out)}");
                string tag = r == 0 ? "COLD" : r == 1 ? "PROBE2" : "WARM";
                Console.WriteLine($"\n=== generation {r + 1}/{seq.Count} ({tag}): \"{seq[r]}\" ===");
                rc = await GenerateOne(gen, o, seq[r], o.Seed, op);
            }
            return rc;
        }
        return await GenerateOne(gen, o, firstPrompt, o.Seed, o.Out);
    }

    // ── Continuous session: load once (done), then prompt → pictures → repeat until /exit ──
    var rng = new Random();
    int? fixedSeed = o.Seed;        // null = a fresh random seed per prompt (printed, so it's reproducible)
    string? nextOut = null;         // one-shot override from /out for the very next picture
    PrintHelp(o);
    while (true)
    {
        Console.Write("\nprompt> ");
        string? line = Console.ReadLine();
        if (line == null) break;    // EOF (Ctrl+Z / piped input ended)
        line = line.Trim();
        if (line.Length == 0) continue;

        if (line is "/exit" or "/quit") break;
        if (line is "/help" or "/?") { PrintHelp(o); continue; }
        if (line.StartsWith('/'))
        {
            try { nextOut = HandleCommand(line, o, ref fixedSeed, nextOut, gen); }
            catch (Exception ex) { Console.WriteLine($"[{ex.Message}]"); }
            continue;
        }

        int useSeed = fixedSeed ?? rng.Next();
        try { await GenerateOne(gen, o, line, useSeed, string.IsNullOrEmpty(nextOut) ? null : nextOut); }
        catch (Exception ex) { Console.Error.WriteLine($"[generation error: {ex.GetType().Name}: {ex.Message}]"); }
        nextOut = null; // the /out override applies to one prompt only
    }
    Console.WriteLine("bye 🖖");
    return 0;
}

// The in-session commands. Returns the (possibly new) next-out path.
string? HandleCommand(string line, Options o, ref int? fixedSeed, string? nextOut, ImageGenerator gen)
{
    var parts = line.Split(' ', 2, StringSplitOptions.TrimEntries);
    string cmd = parts[0].ToLowerInvariant(), arg = parts.Length > 1 ? parts[1] : "";
    switch (cmd)
    {
        case "/seed":
            if (arg.Equals("random", StringComparison.OrdinalIgnoreCase)) { fixedSeed = null; Console.WriteLine("[seed: random per prompt]"); }
            else if (int.TryParse(arg, out var s)) { fixedSeed = s; Console.WriteLine($"[seed fixed to {s}]"); }
            else Console.WriteLine("[usage: /seed <N> | /seed random]");
            break;
        case "/out":
            nextOut = arg;
            Console.WriteLine(nextOut.Length > 0 ? $"[next picture(s) -> {nextOut}]" : "[next picture(s) -> auto name]");
            break;
        case "/model":
        {
            var engine = ParseEngine(arg);
            o.Engine = engine;
            if (engine == ImageEngine.SdTurbo) o.Refs.Clear();
            if (o.Size is not (256 or 512) && engine == ImageEngine.SdTurbo) o.Size = 0;
            Console.WriteLine($"[model: {EngineName(engine)}; it loads on the next prompt]");
            break;
        }
        case "/size": o.Size = int.Parse(arg); Console.WriteLine($"[size {o.Size}]"); break;
        case "/pictures": o.Pictures = Math.Max(1, int.Parse(arg)); Console.WriteLine($"[{o.Pictures} picture(s) per prompt, consecutive seeds]"); break;
        case "/ref":
            if (o.Engine == ImageEngine.SdTurbo) { Console.WriteLine("[SD-Turbo takes no reference images: /model flux2 or flux3 first]"); break; }
            if (o.Refs.Count >= 10) { Console.WriteLine("[at most 10 reference images]"); break; }
            LoadReference(arg); o.Refs.Add(arg); Console.WriteLine($"[{o.Refs.Count} reference image(s): the prompt now says what to do with them]");
            break;
        case "/clearrefs": o.Refs.Clear(); Console.WriteLine("[reference images cleared]"); break;
        case "/caption": o.Caption = !arg.Equals("off", StringComparison.OrdinalIgnoreCase); Console.WriteLine($"[FLUX 3 caption writer {(o.Caption ? "on" : "off")}]"); break;
        case "/negative": o.Negative = arg; Console.WriteLine($"[negative prompt: {(arg.Length > 0 ? arg : "none")}]"); break;
        case "/steps": o.Steps = Math.Max(1, int.Parse(arg)); Console.WriteLine($"[SD-Turbo steps {o.Steps}]"); break;
        case "/guidance": o.Guidance = float.Parse(arg, System.Globalization.CultureInfo.InvariantCulture); Console.WriteLine($"[SD-Turbo guidance {o.Guidance}]"); break;
        default: Console.WriteLine($"[unknown command {cmd} — /help for the list]"); break;
    }
    return nextOut;
}

// One prompt: Pictures pictures, reusing the already-loaded pipeline. Returns the process exit code (only meaningful in --ci).
async Task<int> GenerateOne(ImageGenerator gen, Options o, string prompt, int? seed, string? outPath)
{
    int useSeed = seed ?? Environment.TickCount & 0x7FFFFFFF;
    var refs = o.Engine != ImageEngine.SdTurbo && o.Refs.Count > 0 ? o.Refs.Select(LoadReference).ToList() : null;
    int size = o.Size > 0 ? o.Size : 512;
    if (o.Engine == ImageEngine.SdTurbo && size is not (256 or 512)) throw new ArgumentException("SD-Turbo draws 256 or 512 pixels square");
    bool caption = o.Engine == ImageEngine.Flux3 && refs == null && o.Caption;
    bool sd = o.Engine == ImageEngine.SdTurbo;
    if (sd) BufferPool.ResetPeaks();

    Console.WriteLine($"Generating {o.Pictures} picture(s) with {EngineName(o.Engine)}: \"{prompt}\" (seed {useSeed}, {size} px{(refs != null ? $", {refs.Count} reference image(s)" : "")}{(caption ? ", caption writer on" : "")})");
    bool perOp = sd && Environment.GetEnvironmentVariable("SDTURBO_PEROP") == "1";
    if (perOp)
    {
        SpawnDev.ILGPU.ML.Graph.GraphExecutor.CapturedNodeTimingsMs = new();
        SpawnDev.ILGPU.ML.Graph.GraphExecutor.PerOpSync = true; // true per-op GPU attribution
    }
    var clock = System.Diagnostics.Stopwatch.StartNew();
    bool allReal = true;
    string? firstCaption = null;
    var pictures = await gen.GenerateAsync(new ImageGenerationRequest
    {
        Prompt = prompt, Engine = o.Engine, Seed = useSeed, Pictures = o.Pictures, Size = size, References = refs, WriteCaption = caption,
        NegativePrompt = o.Negative, SdSteps = o.Steps, SdGuidance = o.Guidance,
    }, picture =>
    {
        Console.WriteLine($"{picture.Width}x{picture.Height}, seed {picture.Seed}, {clock.Elapsed.TotalMilliseconds:F0} ms since start");
        firstCaption ??= picture.Caption;
        string path = PicturePath(o, prompt, picture.Seed, outPath, o.Pictures);
        WritePicture(path, picture.Rgba, picture.Width, picture.Height);
        Console.WriteLine($"Saved {Path.GetFullPath(path)}");
        if (o.Ci) allReal &= Verify(picture);
        return Task.CompletedTask;
    });
    if (firstCaption != null) Console.WriteLine($"Caption used: {firstCaption}");
    if (gen.LastTimings.Count > 0) Console.WriteLine("last picture: " + string.Join(" · ", gen.LastTimings.Select(t => $"{t.Stage} {t.Seconds:F1} s")));

    if (perOp && SpawnDev.ILGPU.ML.Graph.GraphExecutor.CapturedNodeTimingsMs is { } _pt && _pt.Count > 0)
    {
        // Aggregate by TRUE op-type ("NNN_OpType_/path" -> "OpType") with node COUNT + avg, so it diffs
        // cleanly against the WebGPU per-op run (find where WGSL codegen/exec is worst vs PTX). Plus top nodes.
        static string OpType(string n) { var p = n.Split('_'); return p.Length > 1 ? p[1] : n; }
        var byType = new Dictionary<string, double>();
        var cntType = new Dictionary<string, int>();
        foreach (var kv in _pt)
        {
            var key = OpType(kv.Key);
            byType[key] = byType.TryGetValue(key, out var v) ? v + kv.Value : kv.Value;
            cntType[key] = cntType.TryGetValue(key, out var c) ? c + 1 : 1;
        }
        double tot = _pt.Values.Sum();
        Console.WriteLine($"[PEROP] total captured = {tot:F0}ms across {_pt.Count} nodes. By op-type (ms / count / avg):");
        foreach (var kv in byType.OrderByDescending(k => k.Value).Take(20))
            Console.WriteLine($"    {kv.Value,8:F1}ms  {100 * kv.Value / tot,5:F1}%  {cntType[kv.Key],5}n  {kv.Value / cntType[kv.Key],7:F2}avg  {kv.Key}");
        Console.WriteLine("[PEROP] top 15 individual nodes:");
        foreach (var kv in _pt.OrderByDescending(k => k.Value).Take(15))
            Console.WriteLine($"    {kv.Value,8:F1}ms  {kv.Key}");
        SpawnDev.ILGPU.ML.Graph.GraphExecutor.CapturedNodeTimingsMs = null;
        SpawnDev.ILGPU.ML.Graph.GraphExecutor.PerOpSync = false;
    }
    if (sd)
    {
        Console.WriteLine($"[POOL] peak TOTAL allocated = {BufferPool.PeakTotalBytes / 1048576.0:F0} MiB" +
                          $" | peak LIVE (working set) = {BufferPool.PeakLiveBytes / 1048576.0:F0} MiB");
        var snap = BufferPool.PeakLiveSnapshot;
        if (snap != null)
        {
            Console.WriteLine($"[POOL] LIVE-peak composition: {snap.Count} buffers, " +
                $"{snap.Count(s => s.isHalf)} fp16 / {snap.Count(s => !s.isHalf)} fp32; top 15 by bytes:");
            foreach (var s in snap.OrderByDescending(s => s.bytes).Take(15))
                Console.WriteLine($"    {s.bytes / 1048576.0,7:F2} MiB  {(s.isHalf ? "f16" : "f32")}  {s.name}");
        }
    }
    return o.Ci && !allReal ? 1 : 0;
}

// --ci: a real picture is NOT all-black and NOT a flat constant (broken diffusion -> zeros/flat).
static bool Verify(GeneratedImage picture)
{
    int px = picture.Width * picture.Height;
    long nonZero = 0; double sum = 0, sumSq = 0;
    for (int i = 0; i < px; i++)
    {
        byte r = picture.Rgba[i * 4], g = picture.Rgba[i * 4 + 1], b = picture.Rgba[i * 4 + 2];
        if (r != 0 || g != 0 || b != 0) nonZero++;
        double lum = r + g + b; sum += lum; sumSq += lum * lum;
    }
    double mean = sum / px, std = Math.Sqrt(Math.Max(0, sumSq / px - mean * mean));
    bool ok = nonZero >= px / 100 && std >= 5.0;
    Console.WriteLine($"[--ci] seed {picture.Seed}: nonZero={nonZero}/{px} lumStd={std:F1} -> {(ok ? "PASS" : "FAIL")}");
    return ok;
}

// A reference picture as RGBA, shrunk so its longer side is at most 768 pixels (the browser demo does the same before it encodes it).
static Flux2ReferenceImage LoadReference(string path)
{
    using var img = Image.Load<Rgba32>(path);
    int longer = Math.Max(img.Width, img.Height);
    if (longer > 768) { double scale = 768.0 / longer; img.Mutate(x => x.Resize(Math.Max(16, (int)Math.Round(img.Width * scale)), Math.Max(16, (int)Math.Round(img.Height * scale)))); }
    var rgba = new byte[img.Width * img.Height * 4];
    img.CopyPixelDataTo(rgba);
    return new Flux2ReferenceImage(img.Width, img.Height, rgba);
}

static ImageEngine ParseEngine(string s) => s.Trim().ToLowerInvariant() switch
{
    "flux3" or "flux-3" or "flux3klein" => ImageEngine.Flux3,
    "flux2" or "flux.2" or "flux" or "klein" => ImageEngine.Flux2,
    "sd" or "sdturbo" or "sd-turbo" or "turbo" => ImageEngine.SdTurbo,
    _ => throw new ArgumentException($"unknown model '{s}': flux3, flux2 or sd"),
};

static string EngineName(ImageEngine e) => e switch { ImageEngine.Flux3 => "FLUX 3 Image Klein", ImageEngine.Flux2 => "FLUX.2 klein", _ => "SD-Turbo" };

static string EngineTag(ImageEngine e) => e switch { ImageEngine.Flux3 => "flux3", ImageEngine.Flux2 => "flux2", _ => "sdturbo" };

// Where a picture goes: --out / /out, else an auto name with the prompt + seed (the seed also goes into an --out name when there are several pictures).
static string PicturePath(Options o, string prompt, int seed, string? outPath, int pictures)
{
    if (outPath == null) return $"{EngineTag(o.Engine)}_{Sanitize(prompt)}_{seed}.png";
    if (pictures <= 1) return outPath;
    return Path.Combine(Path.GetDirectoryName(outPath) is { Length: > 0 } d ? d : ".", $"{Path.GetFileNameWithoutExtension(outPath)}_{seed}{Path.GetExtension(outPath)}");
}

static void WritePicture(string path, byte[] rgba, int width, int height)
{
    if (path.EndsWith(".bmp", StringComparison.OrdinalIgnoreCase)) WriteBmp(path, rgba, width, height);
    else File.WriteAllBytes(path, AssetGenerator.EncodePng(rgba, width, height));
}

static void PrintHelp(Options o) => Console.WriteLine(
    $"\n{EngineName(o.Engine)} image session. Type a prompt and press Enter to generate.\n" +
    "  /model flux3|flux2|sd  switch model (loads on the next prompt)     /size <N>      512 | 768 | 1024 (FLUX), 256 | 512 (SD)\n" +
    "  /pictures <N>          pictures per prompt (consecutive seeds)      /seed <N>      fix the seed (/seed random = random per prompt)\n" +
    "  /ref <file>            add a reference image (FLUX, up to 10)       /clearrefs     remove the reference images\n" +
    "  /caption on|off        FLUX 3: Qwen3 writes the full caption        /negative <t>  SD-Turbo negative prompt\n" +
    "  /steps <N>             SD-Turbo steps                               /guidance <F>  SD-Turbo guidance\n" +
    "  /out <path>            write the next prompt's picture(s) to <path>\n" +
    "  /help   /exit");

static string Sanitize(string s)
{
    var clean = new string(s.Select(c => char.IsLetterOrDigit(c) ? char.ToLowerInvariant(c) : '_').ToArray());
    return clean.Length > 40 ? clean[..40] : clean;
}

// 24-bit BMP (bottom-up, BGR) from RGBA pixels. Minimal + self-contained; opens in any image viewer.
static void WriteBmp(string path, byte[] rgba, int width, int height)
{
    int dataSize = width * 3 * height;
    var bmp = new byte[54 + dataSize];
    bmp[0] = (byte)'B'; bmp[1] = (byte)'M';
    BitConverter.GetBytes(54 + dataSize).CopyTo(bmp, 2);
    BitConverter.GetBytes(54).CopyTo(bmp, 10);            // pixel-data offset
    BitConverter.GetBytes(40).CopyTo(bmp, 14);            // DIB header size
    BitConverter.GetBytes(width).CopyTo(bmp, 18);
    BitConverter.GetBytes(height).CopyTo(bmp, 22);
    BitConverter.GetBytes((short)1).CopyTo(bmp, 26);      // planes
    BitConverter.GetBytes((short)24).CopyTo(bmp, 28);     // bpp
    BitConverter.GetBytes(dataSize).CopyTo(bmp, 34);
    int off = 54;
    for (int y = height - 1; y >= 0; y--)                 // BMP rows are bottom-up
        for (int x = 0; x < width; x++)
        {
            int p = (y * width + x) * 4;
            bmp[off++] = rgba[p + 2]; bmp[off++] = rgba[p + 1]; bmp[off++] = rgba[p + 0]; // BGR
        }
    File.WriteAllBytes(path, bmp);
}

sealed class Options
{
    public ImageEngine Engine = ImageEngine.SdTurbo;       // the original example's model; --model flux2|flux3 for the FLUX pair
    public bool Ci;
    public int? Seed;
    public string? Out, Cache, Negative = "";
    public int Repeat = 1, Pictures = 1, Size = 0, Bits = 4, Steps = 1;
    public float Guidance;
    public bool Caption = true;                            // FLUX 3 without references: the caption writer (a short prompt gives a generic picture)
    public List<string>? PromptSeq;
    public List<string> Refs = new();
}
