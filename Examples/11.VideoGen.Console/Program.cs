// ----------------------------------------------------------------------------------------------------------------------------------
//  SpawnDev.ILGPU.ML - Example 11: Video Generation (console)
//
//  Wan 2.2 TI2V-5B on native ILGPU GPU kernels (CUDA / OpenCL), no ONNX Runtime and no Python: text to video, or image to video with a first frame.
//  The stages run one after another so only one big model is on the device at a time (umT5 text encoder streamed layer by layer, the 8-bit or 4-bit DiT,
//  the VAE), which is what lets a 12 GB card run a 5B-parameter video model.
//
//    dotnet run -- "A red fox walking through fresh snow at golden hour" --size 640x352 --frames 17 --steps 30 --seed 7
//    dotnet run -- "the knight turns and raises a sword" --image knight.png --size 640x352 --frames 33        # image to video: the picture is frame 0
//    dotnet run -- --ci                                                                                       # small fixed run, asserts the video is real
//
//  Options: --size WxH (multiples of 32; the model's own sizes are 1280x704 and 704x1280)  --frames N (4n+1: 17, 33, 49 ... 121 = 5 s at 24 fps)
//           --steps N (50 is the reference; 20-30 is usually fine)  --seed N  --guidance F (5)  --shift F (5)  --negative TEXT  --image PATH
//           --turbo (the step-distilled Wan2.2-TI2V-5B-Turbo: 4 steps, no guidance, few-step sampler: 720p in about a minute; the best default)
//           --ltx (LTX-Video 2B 0.9.8 distilled instead of Wan: text to video or --image to video, 7 steps, no guidance, 8n+1 frames, native 704x480; --fps sets the position-embedding frame rate;
//                  --upscale runs the model's two-pass recipe: 2/3 size, the x2 latent upscaler, 3 steps at twice that size, so 704x480 comes out at 896x576)
//           --dit q8|q4|bf16|turbo_q8 (5.6 GB / 3.1 GB / 10 GB / 5.6 GB)  --sampler unipc|fewstep  --text q8|bf16 (6.4 GB / 11.4 GB; default q8, the hosted one: bf16 only with --models)  --models DIR (a local folder of the exports; else they
//           download from the SpawnDev models host into the cache)  --cache DIR  --out DIR  --fps N  --no-frames (skip the PNG frames)
//
//  Output: <out>/video.webp (animated), <out>/frame_0000.png ...  A bigger size or more frames costs memory: on a 12 GB card 640x352 with 17-49 frames is comfortable.
//  The model gives good and bad clips by seed at small sizes: try another --seed before suspecting anything else.
// ----------------------------------------------------------------------------------------------------------------------------------

using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using ILGPU.Runtime.OpenCL;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Formats.Webp;
using SixLabors.ImageSharp.PixelFormats;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.Ltx;
using SpawnDev.ILGPU.ML.Wan;

var o = new Options();
var words = new List<string>();
for (int i = 0; i < args.Length; i++)
{
    switch (args[i])
    {
        case "--ci": o.Ci = true; break;
        case "--size" when i + 1 < args.Length: { var p = args[++i].Split('x', 'X'); o.Width = int.Parse(p[0]); o.Height = int.Parse(p[1]); break; }
        case "--frames" when i + 1 < args.Length: o.Frames = int.Parse(args[++i]); break;
        case "--steps" when i + 1 < args.Length: o.Steps = int.Parse(args[++i]); break;
        case "--seed" when i + 1 < args.Length: o.Seed = long.Parse(args[++i]); break;
        case "--guidance" when i + 1 < args.Length: o.Guidance = double.Parse(args[++i], System.Globalization.CultureInfo.InvariantCulture); break;
        case "--shift" when i + 1 < args.Length: o.Shift = double.Parse(args[++i], System.Globalization.CultureInfo.InvariantCulture); break;
        case "--negative" when i + 1 < args.Length: o.Negative = args[++i]; break;
        case "--image" when i + 1 < args.Length: o.Image = args[++i]; break;
        case "--dit" when i + 1 < args.Length: o.Dit = args[++i]; break;
        case "--text" when i + 1 < args.Length: o.Text = args[++i]; break;
        case "--models" when i + 1 < args.Length: o.Models = args[++i]; break;
        case "--cache" when i + 1 < args.Length: o.Cache = args[++i]; break;
        case "--out" when i + 1 < args.Length: o.Out = args[++i]; break;
        case "--fps" when i + 1 < args.Length: o.Fps = int.Parse(args[++i]); break;
        case "--no-frames": o.NoFrames = true; break;
        case "--sampler" when i + 1 < args.Length: o.FewStep = args[++i].Equals("fewstep", StringComparison.OrdinalIgnoreCase); break;
        case "--turbo": o.Turbo = true; break;
        case "--ltx": o.Ltx = true; break;
        case "--upscale": o.Upscale = true; break;
        default: if (!args[i].StartsWith("--")) words.Add(args[i]); break;
    }
}
string prompt = string.Join(' ', words);
if (o.Turbo)                                                    // Wan2.2-TI2V-5B-Turbo: 4 steps, no guidance, the few-step sampler (each can still be overridden by a later option)
{
    o.Dit = "turbo_q8"; o.FewStep = true; o.Guidance = 1.0;
    if (!args.Contains("--steps")) o.Steps = 4;
}
if (o.Ltx)                                                      // LTX-Video 2B 0.9.8 distilled: 7 steps, no guidance, 8n + 1 frames; its own native size
{
    if (!args.Contains("--size")) { o.Width = 704; o.Height = 480; }
    if (!args.Contains("--frames")) o.Frames = 65;
}
if (o.Ci)
{
    o.Width = 320; o.Height = 192; o.Frames = o.Ltx ? 9 : 9; o.Steps = 8; o.Seed = 1;
    if (string.IsNullOrWhiteSpace(prompt)) prompt = "A red fox walking through fresh snow at golden hour, the camera follows it slowly.";
}
if (string.IsNullOrWhiteSpace(prompt)) { Console.Error.WriteLine("Usage: dotnet run -- \"a prompt\" [--size 640x352] [--frames 17] [--steps 30] [--seed N] [--image first.png] [--models DIR]   (see the header of Program.cs)"); return 2; }

try { return await Run(prompt, o); }
catch (Exception ex) { Console.Error.WriteLine($"Generation failed: {ex.Message}"); return 1; }

async Task<int> Run(string prompt, Options o)
{
    // The application owns the accelerator (library code never disposes it). Prefer CUDA, then OpenCL.
    using var context = MLContext.Create().ToContext();
    var cuda = context.GetCudaDevices();
    var opencl = context.GetCLDevices();
    Device device = cuda.Count > 0 ? (Device)cuda[0] : opencl.Count > 0 ? (Device)opencl[0] : context.GetPreferredDevice(preferCPU: false);
    using var accelerator = device.CreateAccelerator(context);
    Console.WriteLine($"Accelerator: {accelerator.Name} ({accelerator.AcceleratorType})");

    using var http = new HttpClient { Timeout = Timeout.InfiniteTimeSpan };
    if (o.Ltx) return await RunLtx(accelerator, http, prompt, o);
    string ditFile = Wan22Paths.DitFile(o.Dit), textFile = Wan22Paths.TextEncoderFile(o.Text);
    Wan22Pipeline pipe;
    if (o.Models != null)
    {
        Console.WriteLine($"Models: {o.Models}");
        pipe = new Wan22Pipeline(accelerator, Wan22Paths.InFolder(o.Models, ditFile, textFile));
    }
    else
    {
        // the desktop pair: HttpClientModelSource downloads (and resumes) into a FileModelStore; the pipeline reads the cached files by offset
        var store = o.Cache != null ? new FileModelStore(o.Cache) : FileModelStore.Default();
        Console.WriteLine($"Model cache: {store.RootDirectory} (first run downloads the DiT, the VAE and the text encoder from the models host)");
        pipe = new Wan22Pipeline(accelerator, new HttpClientModelSource(http, store), Wan22Pipeline.DefaultRepo, Wan22Paths.InRepo(ditFile, textFile));
    }

    var request = new WanRequest
    {
        Prompt = prompt, NegativePrompt = o.Negative, Width = o.Width, Height = o.Height, Frames = o.Frames, Steps = o.Steps, GuidanceScale = o.Guidance, Shift = o.Shift, Seed = o.Seed, Sampler = o.FewStep ? WanSampler.FewStep : WanSampler.UniPC,
    };
    if (o.Image != null)
    {
        using var img = Image.Load<Rgb24>(o.Image);
        var rgb = new byte[img.Width * img.Height * 3];
        img.CopyPixelDataTo(rgb);
        request = new WanRequest
        {
            Prompt = prompt, NegativePrompt = o.Negative, Width = o.Width, Height = o.Height, Frames = o.Frames, Steps = o.Steps, GuidanceScale = o.Guidance, Shift = o.Shift, Seed = o.Seed, Sampler = o.FewStep ? WanSampler.FewStep : WanSampler.UniPC,
            ImageRgb = rgb, ImageWidth = img.Width, ImageHeight = img.Height,
        };
    }

    var sw = System.Diagnostics.Stopwatch.StartNew();
    var video = await pipe.GenerateAsync(request, ConsoleProgress(sw));
    Console.WriteLine();
    Console.WriteLine($"{video.Frames} frames {video.Width}x{video.Height} in {sw.Elapsed.TotalSeconds:F0} s");
    return Save(video, o);
}

IProgress<WanProgress> ConsoleProgress(System.Diagnostics.Stopwatch sw)
{
    string lastStage = "";
    return new Progress<WanProgress>(p =>
    {
        if (p.Stage != lastStage) { if (lastStage != "") Console.WriteLine(); lastStage = p.Stage; Console.Write($"[{sw.Elapsed.TotalSeconds,6:F0} s] {p.Stage}: "); }
        Console.Write($"\r[{sw.Elapsed.TotalSeconds,6:F0} s] {p.Stage}: {p.Message}   ");
    });
}

// LTX-Video 2B 0.9.8 distilled (text to video, or image to video with --image): T5-v1.1-XXL streamed, the VAE encoder for the picture, the 2B DiT with the 7-step distilled sampler,
// the timestep-conditioned VAE decode
async Task<int> RunLtx(Accelerator accelerator, HttpClient http, string prompt, Options o)
{
    string ditFile = LtxPaths.DitFile(args.Contains("--dit") ? o.Dit : "bf16"), textFile = LtxPaths.TextEncoderFile(o.Text);       // LTX: --dit bf16|q8 (3.6 / 2.0 GB), --text bf16|q8 (8.9 / 5.0 GB)
    LtxPipeline pipe;
    if (o.Models != null)
    {
        Console.WriteLine($"Models: {o.Models} ({ditFile}, {textFile})");
        pipe = new LtxPipeline(accelerator, LtxPaths.InFolder(o.Models, ditFile, textFile));
    }
    else
    {
        var store = o.Cache != null ? new FileModelStore(o.Cache) : FileModelStore.Default();
        Console.WriteLine($"Model cache: {store.RootDirectory} (first run downloads the DiT, the VAE and the text encoder from the models host)");
        pipe = new LtxPipeline(accelerator, new HttpClientModelSource(http, store), LtxPipeline.DefaultRepo, LtxPaths.InRepo(ditFile, textFile));
    }
    LtxCondition[]? conditions = null;
    if (o.Image != null)
    {
        using var img = Image.Load<Rgb24>(o.Image);
        var rgb = new byte[img.Width * img.Height * 3];
        img.CopyPixelDataTo(rgb);
        conditions = new[] { LtxCondition.Image(rgb, img.Width, img.Height) };          // the picture is frame 0 (scaled to cover the size, centre-cropped)
    }
    var request = new LtxRequest { Prompt = prompt, Width = o.Width, Height = o.Height, Frames = o.Frames, Seed = o.Seed, FrameRate = o.Fps, Conditions = conditions, Upscale = o.Upscale };
    var sw = System.Diagnostics.Stopwatch.StartNew();
    var video = await pipe.GenerateAsync(request, ConsoleProgress(sw));
    Console.WriteLine();
    Console.WriteLine($"{video.Frames} frames {video.Width}x{video.Height} in {sw.Elapsed.TotalSeconds:F0} s");
    return Save(video, o);
}

int Save(WanVideo video, Options o)
{
    Directory.CreateDirectory(o.Out);
    Image<Rgba32>? anim = null;
    for (int f = 0; f < video.Frames; f++)
    {
        var frame = Image.LoadPixelData<Rgba32>(video.GetFrameRgba(f), video.Width, video.Height);
        if (!o.NoFrames) frame.SaveAsPng(Path.Combine(o.Out, $"frame_{f:D4}.png"));
        if (anim == null) anim = frame; else { anim.Frames.AddFrame(frame.Frames.RootFrame); frame.Dispose(); }
        anim.Frames[f].Metadata.GetWebpMetadata().FrameDelay = (uint)Math.Max(1, 1000 / o.Fps);
    }
    anim.Metadata.GetWebpMetadata().RepeatCount = 0;
    string webp = Path.Combine(o.Out, "video.webp");
    anim!.SaveAsWebp(webp, new WebpEncoder { FileFormat = WebpFileFormatType.Lossy, Quality = 90 });
    anim?.Dispose();
    Console.WriteLine($"Wrote {webp}" + (o.NoFrames ? "" : $" and {video.Frames} PNG frames in {o.Out}"));

    if (o.Ci)
    {
        double sum = 0, sum2 = 0; foreach (var v in video.Data) { if (!float.IsFinite(v)) { Console.Error.WriteLine("CI FAIL: a pixel is not finite"); return 1; } sum += v; sum2 += (double)v * v; }
        double mean = sum / video.Data.Length, std = Math.Sqrt(sum2 / video.Data.Length - mean * mean);
        Console.WriteLine($"CI: pixel std {std:F3}");
        if (!(std > 0.02)) { Console.Error.WriteLine("CI FAIL: the video is flat"); return 1; }
    }
    return 0;
}

sealed class Options
{
    public bool Ci, NoFrames, FewStep, Turbo, Ltx, Upscale;
    public int Width = 640, Height = 352, Frames = 17, Steps = 30, Fps = 24;
    public long Seed = 42;
    public double Guidance = 5.0, Shift = 5.0;
    public string? Negative, Image, Models, Cache;
    public string Dit = "q8", Text = "q8", Out = "video_out";      // text q8: the models host serves the 8-bit text encoders only (bf16 needs --models with the 16-bit export)
}
