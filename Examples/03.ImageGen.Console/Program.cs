// ─────────────────────────────────────────────────────────────────────────────────────────────────
//  SpawnDev.ILGPU.ML — Example 03: Image Generation (console)
//
//  Text -> image with SD-Turbo (single-step diffusion) running on native ILGPU GPU kernels — no ONNX
//  Runtime. Like the gemma4 chat example (05), with NO prompt argument it drops into a continuous
//  session: the ~2.5 GB SD-Turbo pipeline loads ONCE, then you type a prompt, get an image, and keep
//  going until /exit — the model never reloads between images.
//
//    dotnet run                                  # interactive: load once, then prompt -> image -> repeat
//    dotnet run -- a photo of a cat              # one-shot: generate a single image and exit
//    dotnet run -- "a watercolor fox" --seed 7 --out fox.bmp
//    dotnet run -- --ci                          # fixed prompt + seed; asserts a real (non-flat) image
//
//  In-session commands:
//    /seed <N>        fix the seed for the next images (reproducible); /seed random goes back to random
//    /out <path>      write the next image to <path> (else an auto name from the prompt + seed)
//    /help            show commands
//    /exit  (/quit)   leave
//
//  Self-contained: the accelerator is created here (the app owns it), the SD-Turbo weights stream from
//  the SpawnDev hub on first run (cached after), and the RGBA result is written with the small inline
//  BMP encoder below. The only dependency is SpawnDev.ILGPU.ML (which brings the pipeline + hub).
//
//  NOTE: first run downloads ~2.5 GB of SD-Turbo weights and runs GPU diffusion — give it a moment.
// ─────────────────────────────────────────────────────────────────────────────────────────────────

using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using ILGPU.Runtime.OpenCL;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.ILGPU.ML.Tensors;

// ── args: prompt words (non-flag) + optional --seed N / --out PATH / --ci ─────────────────────────
bool ci = false;
int? seed = null;
string? outPath = null;
var promptWords = new List<string>();
for (int i = 0; i < args.Length; i++)
{
    switch (args[i])
    {
        case "--ci": ci = true; break;
        case "--seed" when i + 1 < args.Length: seed = int.Parse(args[++i]); break;
        case "--out" when i + 1 < args.Length: outPath = args[++i]; break;
        default: if (!args[i].StartsWith("--")) promptWords.Add(args[i]); break;
    }
}

string prompt = string.Join(' ', promptWords);
if (ci)
{
    seed ??= 42;                                                  // deterministic image for the self-check
    if (string.IsNullOrWhiteSpace(prompt)) prompt = "a photo of a cat";
}

// No prompt + a real console (not piped) + not CI → continuous interactive session. A prompt arg or --ci
// is a one-shot. No prompt + piped/redirected input is the usage error (nothing to generate, can't ask).
bool interactive = string.IsNullOrWhiteSpace(prompt) && !ci && !Console.IsInputRedirected;
if (string.IsNullOrWhiteSpace(prompt) && !ci && !interactive)
{
    Console.Error.WriteLine("Usage: ImageGen.Console <prompt words> [--seed N] [--out file.bmp] [--ci]   (no prompt = interactive session)");
    return 2;
}

try
{
    return await Run(prompt, seed, outPath, ci, interactive);
}
catch (Exception ex)
{
    Console.Error.WriteLine($"Generation failed: {ex.Message}");
    return 1;
}

async Task<int> Run(string firstPrompt, int? seed, string? outPath, bool ci, bool interactive)
{
    // The application owns the accelerator (library code never disposes it). Prefer CUDA (SD-Turbo's
    // verified path), then OpenCL, then whatever ILGPU prefers.
    using var context = MLContext.Create().ToContext();
    var cuda = context.GetCudaDevices();
    var opencl = context.GetCLDevices();
    Device device = cuda.Count > 0 ? (Device)cuda[0]
                  : opencl.Count > 0 ? (Device)opencl[0]
                  : context.GetPreferredDevice(preferCPU: false);
    using var accelerator = device.CreateAccelerator(context);
    if (Environment.GetEnvironmentVariable("ML_VERBOSE") == "1") InferenceSession.VerboseLogging = true;
    Console.WriteLine($"Accelerator: {accelerator.Name} ({accelerator.AcceleratorType})");

    // Model acquisition: the SpawnDev hub streams SD-Turbo's ONNX weights (cached after first run).
    using var http = new HttpClient();
    await using var webTorrent = new SpawnDev.WebTorrent.WebTorrentClient();
    var hub = new HubModelStream(webTorrent, http);

    Console.WriteLine("Loading SD-Turbo (first run downloads ~2.5 GB)...");
    // Loaded ONCE and reused for every generation below — the load is the expensive part; an interactive
    // session must never pay it per image.
    var pipe = await ImageGenerationPipeline.CreateAsync(accelerator, hub, ModelHub.KnownModels.SDTurbo,
        onProgress: (stage, pct) => Console.WriteLine($"  [load] {stage} {pct}%"));
    using (pipe)
    {
        if (!pipe.IsReady) throw new Exception("SD-Turbo pipeline not ready (a sub-model failed to load).");

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

        pipe.NumInferenceSteps = 1;  // SD-Turbo is single-step
        pipe.GuidanceScale = 0f;     // SD-Turbo uses no classifier-free guidance
        BufferPool.TrackPeaks = true;
        if (Environment.GetEnvironmentVariable("PEAK_COMPOSITION") == "1")
            BufferPool.TrackLivePeakComposition = true;

        if (!interactive)
            return await GenerateOne(pipe, firstPrompt, seed, outPath, ci);

        // ── Continuous session: load once (done), then prompt → image → repeat until /exit ──
        var rng = new Random();
        int? fixedSeed = seed;          // null = a fresh random seed per image (printed, so it's reproducible)
        string? nextOut = null;         // one-shot override from /out for the very next image
        PrintHelp();
        while (true)
        {
            Console.Write("\nprompt> ");
            string? line = Console.ReadLine();
            if (line == null) break;    // EOF (Ctrl+Z / piped input ended)
            line = line.Trim();
            if (line.Length == 0) continue;

            if (line is "/exit" or "/quit") break;
            if (line is "/help" or "/?") { PrintHelp(); continue; }
            if (line.StartsWith("/seed ", StringComparison.OrdinalIgnoreCase))
            {
                var arg = line[6..].Trim();
                if (arg.Equals("random", StringComparison.OrdinalIgnoreCase)) { fixedSeed = null; Console.WriteLine("[seed: random per image]"); }
                else if (int.TryParse(arg, out var s)) { fixedSeed = s; Console.WriteLine($"[seed fixed to {s}]"); }
                else Console.WriteLine("[usage: /seed <N> | /seed random]");
                continue;
            }
            if (line.StartsWith("/out ", StringComparison.OrdinalIgnoreCase))
            {
                nextOut = line[5..].Trim();
                Console.WriteLine(nextOut.Length > 0 ? $"[next image -> {nextOut}]" : "[next image -> auto name]");
                continue;
            }
            if (line.StartsWith("/")) { Console.WriteLine($"[unknown command {line.Split(' ')[0]} — /help for the list]"); continue; }

            int useSeed = fixedSeed ?? rng.Next();
            try { await GenerateOne(pipe, line, useSeed, string.IsNullOrEmpty(nextOut) ? null : nextOut, ci: false); }
            catch (Exception ex) { Console.Error.WriteLine($"[generation error: {ex.GetType().Name}: {ex.Message}]"); }
            nextOut = null; // the /out override applies to one image only
        }
        Console.WriteLine("bye 🖖");
        return 0;
    }
}

// One image, reusing the already-loaded pipeline. Returns the process exit code (only meaningful in --ci).
async Task<int> GenerateOne(ImageGenerationPipeline pipe, string prompt, int? seed, string? outPath, bool ci)
{
    if (seed.HasValue) pipe.Seed = seed.Value;
    BufferPool.ResetPeaks();

    Console.WriteLine($"Generating: \"{prompt}\"" + (seed.HasValue ? $" (seed {seed})" : ""));
    var result = await pipe.RunAsync(new ImageGenerationInput { Prompt = prompt });
    Console.WriteLine($"{result.Width}x{result.Height} in {result.InferenceTimeMs:F0}ms ({result.NumSteps} step)");
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

    // Auto name includes the seed so repeated prompts in a session don't clobber each other.
    outPath ??= seed.HasValue ? $"sdturbo_{Sanitize(prompt)}_{seed}.bmp" : $"sdturbo_{Sanitize(prompt)}.bmp";
    WriteBmp(outPath, result.ImageRGBA, result.Width, result.Height);
    Console.WriteLine($"Saved {Path.GetFullPath(outPath)}");

    if (ci)
    {
        // A real generation is NOT all-black and NOT a flat constant (broken diffusion -> zeros/flat).
        int px = result.Width * result.Height;
        long nonZero = 0; double sum = 0, sumSq = 0;
        for (int i = 0; i < px; i++)
        {
            byte r = result.ImageRGBA[i * 4], g = result.ImageRGBA[i * 4 + 1], b = result.ImageRGBA[i * 4 + 2];
            if (r != 0 || g != 0 || b != 0) nonZero++;
            double lum = r + g + b; sum += lum; sumSq += lum * lum;
        }
        double mean = sum / px, std = Math.Sqrt(Math.Max(0, sumSq / px - mean * mean));
        bool ok = nonZero >= px / 100 && std >= 5.0;
        Console.WriteLine($"[--ci] nonZero={nonZero}/{px} lumStd={std:F1} -> {(ok ? "PASS" : "FAIL")}");
        return ok ? 0 : 1;
    }
    return 0;
}

static void PrintHelp() => Console.WriteLine(
    "\nSD-Turbo image session. Type a prompt and press Enter to generate.\n" +
    "  /seed <N>     fix the seed for the next images (/seed random = random per image)\n" +
    "  /out <path>   write the next image to <path>\n" +
    "  /help         show this\n" +
    "  /exit         quit");

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
