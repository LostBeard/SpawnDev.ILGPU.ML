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
//           --dit q8|q4|bf16 (5.6 GB / 3.1 GB / 10 GB)  --text q8|bf16 (6.4 GB / 11.4 GB)  --models DIR (a local folder of the exports; else they
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
        default: if (!args[i].StartsWith("--")) words.Add(args[i]); break;
    }
}
string prompt = string.Join(' ', words);
if (o.Ci)
{
    o.Width = 320; o.Height = 192; o.Frames = 9; o.Steps = 8; o.Seed = 1;
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

    string ditFile = Wan22Paths.DitFile(o.Dit), textFile = Wan22Paths.TextEncoderFile(o.Text);
    using var http = new HttpClient { Timeout = Timeout.InfiniteTimeSpan };
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
        Prompt = prompt, NegativePrompt = o.Negative, Width = o.Width, Height = o.Height, Frames = o.Frames, Steps = o.Steps, GuidanceScale = o.Guidance, Shift = o.Shift, Seed = o.Seed,
    };
    if (o.Image != null)
    {
        using var img = Image.Load<Rgb24>(o.Image);
        var rgb = new byte[img.Width * img.Height * 3];
        img.CopyPixelDataTo(rgb);
        request = new WanRequest
        {
            Prompt = prompt, NegativePrompt = o.Negative, Width = o.Width, Height = o.Height, Frames = o.Frames, Steps = o.Steps, GuidanceScale = o.Guidance, Shift = o.Shift, Seed = o.Seed,
            ImageRgb = rgb, ImageWidth = img.Width, ImageHeight = img.Height,
        };
    }

    var sw = System.Diagnostics.Stopwatch.StartNew();
    string lastStage = "";
    var progress = new Progress<WanProgress>(p =>
    {
        if (p.Stage != lastStage) { if (lastStage != "") Console.WriteLine(); lastStage = p.Stage; Console.Write($"[{sw.Elapsed.TotalSeconds,6:F0} s] {p.Stage}: "); }
        Console.Write($"\r[{sw.Elapsed.TotalSeconds,6:F0} s] {p.Stage}: {p.Message}   ");
    });
    var video = await pipe.GenerateAsync(request, progress);
    Console.WriteLine();
    Console.WriteLine($"{video.Frames} frames {video.Width}x{video.Height} in {sw.Elapsed.TotalSeconds:F0} s");

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
    public bool Ci, NoFrames;
    public int Width = 640, Height = 352, Frames = 17, Steps = 30, Fps = 24;
    public long Seed = 42;
    public double Guidance = 5.0, Shift = 5.0;
    public string? Negative, Image, Models, Cache;
    public string Dit = "q8", Text = "bf16", Out = "video_out";
}
