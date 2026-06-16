// ─────────────────────────────────────────────────────────────────────────────────────────────────
//  SpawnDev.ILGPU.ML — Example 03: Image Generation (console)
//
//  Text -> image with SD-Turbo (single-step diffusion) running on native ILGPU GPU kernels — no ONNX
//  Runtime. Give a prompt as an argument, or get prompted at the console; the result is written to a
//  .bmp you can open.
//
//    dotnet run -- a photo of a cat
//    dotnet run -- "a watercolor fox" --seed 7 --out fox.bmp
//    dotnet run                                  # interactive: prompts for the text
//    dotnet run -- --ci                          # fixed prompt + seed; asserts a real (non-flat) image
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
if (string.IsNullOrWhiteSpace(prompt))
{
    if (ci) prompt = "a photo of a cat";
    else if (!Console.IsInputRedirected) { Console.Write("Prompt: "); prompt = (Console.ReadLine() ?? "").Trim(); }
}
if (string.IsNullOrWhiteSpace(prompt))
{
    Console.Error.WriteLine("Usage: ImageGen.Console <prompt words> [--seed N] [--out file.bmp] [--ci]");
    return 2;
}
if (ci) seed ??= 42; // deterministic image for the self-check

try
{
    return await Generate(prompt, seed, outPath, ci);
}
catch (Exception ex)
{
    Console.Error.WriteLine($"Generation failed: {ex.Message}");
    return 1;
}

async Task<int> Generate(string prompt, int? seed, string? outPath, bool ci)
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
    var pipe = await ImageGenerationPipeline.CreateAsync(accelerator, hub, ModelHub.KnownModels.SDTurbo,
        onProgress: (stage, pct) => Console.WriteLine($"  [load] {stage} {pct}%"));
    using (pipe)
    {
        if (!pipe.IsReady) throw new Exception("SD-Turbo pipeline not ready (a sub-model failed to load).");
        pipe.NumInferenceSteps = 1;  // SD-Turbo is single-step
        pipe.GuidanceScale = 0f;     // SD-Turbo uses no classifier-free guidance
        if (seed.HasValue) pipe.Seed = seed.Value;

        SpawnDev.ILGPU.ML.Tensors.BufferPool.TrackPeaks = true;
        SpawnDev.ILGPU.ML.Tensors.BufferPool.ResetPeaks();

        Console.WriteLine($"Generating: \"{prompt}\"" + (seed.HasValue ? $" (seed {seed})" : ""));
        var result = await pipe.RunAsync(new ImageGenerationInput { Prompt = prompt });
        Console.WriteLine($"{result.Width}x{result.Height} in {result.InferenceTimeMs:F0}ms ({result.NumSteps} step)");
        Console.WriteLine($"[POOL] peak TOTAL allocated = {SpawnDev.ILGPU.ML.Tensors.BufferPool.PeakTotalBytes / 1048576.0:F0} MiB" +
                          $" | peak LIVE (working set) = {SpawnDev.ILGPU.ML.Tensors.BufferPool.PeakLiveBytes / 1048576.0:F0} MiB");

        outPath ??= $"sdturbo_{Sanitize(prompt)}.bmp";
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
    }
    return 0;
}

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
