// ─────────────────────────────────────────────────────────────────
//  SpawnDev.ILGPU.ML — Example 07: Asset generation (console)
//
//  A prompt becomes a textured, rigged, animated 3D character, entirely on your own GPU (CUDA or OpenCL here, WebGPU in the browser demo):
//
//     FLUX.2 klein draws it  ->  RMBG-1.4 cuts it out  ->  TRELLIS.2 builds the mesh + PBR textures  ->  SkinTokens rigs it  ->  NVIDIA Kimodo moves animate it
//
//    dotnet run -- "a knight in plate armor, full body, standing"
//    dotnet run -- "a wooden treasure chest" --no-rig                       # props do not need a skeleton
//    dotnet run -- "a friendly robot" --pictures 4 --pick                   # draw 4, look at them, choose one
//    dotnet run -- "a pirate" --out pirate --seed 11 --moves walk,dance,wave --arm-clearance 10
//    dotnet run -- "a dragon" --image-model sd                              # SD-Turbo instead of FLUX.2 (smaller, faster, older look)
//    dotnet run -- --ci                                                     # short fixed run that asserts files were written
//
//  Options:
//    --out <name>              output name prefix / folder (default: from the prompt)
//    --seed <N>                first seed (default 1); --pictures <N> draws N seeds (default 2) and keeps the one that fits its frame best
//    --pick                    show the drawn pictures (PNG files) and ask which to build
//    --size <px>               picture side for FLUX.2, a multiple of 16 (default 512)
//    --image-model flux2|sd    default flux2             --quality 4|8   FLUX.2 weights: 4-bit (default, 2.6 GB on the GPU) or 8-bit (near lossless, 4.4 GB)
//    --faces <N> --texture <N> mesh triangle budget (default 300000) and texture size (default 2048)
//    --bridge-hops <N>         rig cleanup: drop skin influences from bones N+ hops from a vertex's own and cut triangles joining parts further apart (default 4; 0 = off)
//    --no-rig                  stop after the 3D model          --no-animate  stop after the rig
//    --moves a,b,c             clips to bake in (default: all 22; names are the files in motions/)
//    --arm-clearance <deg>     swing the upper arms away from the body for bulky characters (default 8)
//    --cache <dir>             where models are cached (default: your user profile; ~25 GB in all: put it on a big drive)
//
//  First run downloads the models (about 4.6 GB FLUX.2 + 11 GB TRELLIS.2 + 1.2 GB SkinTokens + small ones) and resumes if interrupted; every later run reads the cache.
//  Needs a GPU with about 8 GB free (the models load one at a time). Outputs: <name>.png (the chosen picture), <name>.glb, <name>_rigged.glb, <name>_animated.glb.
//
//  Self-contained on purpose: the accelerator is created here (the app owns it); everything else is SpawnDev.ILGPU.ML's AssetGenerator.
// ─────────────────────────────────────────────────────────────────

using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using ILGPU.Runtime.OpenCL;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.Animation;
using SpawnDev.ILGPU.ML.Assets;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.Trellis2;

bool ci = false, pick = false, rig = true, animate = true;
int seed = 1, pictures = 2, size = 512, faces = 300_000, texture = 2048, quality = 4, bridgeHops = 4;
double clearance = 8;
string? outName = null, cacheDir = null, movesArg = null; var model = ImageModel.Flux2Klein;
var words = new List<string>();
for (int i = 0; i < args.Length; i++)
{
    switch (args[i])
    {
        case "--ci": ci = true; break;
        case "--pick": pick = true; break;
        case "--no-rig": rig = false; break;
        case "--bridge-hops" when i + 1 < args.Length: bridgeHops = int.Parse(args[++i]); break;
        case "--no-animate": animate = false; break;
        case "--seed" when i + 1 < args.Length: seed = int.Parse(args[++i]); break;
        case "--pictures" when i + 1 < args.Length: pictures = Math.Max(1, int.Parse(args[++i])); break;
        case "--quality" when i + 1 < args.Length: quality = int.Parse(args[++i]) == 8 ? 8 : 4; break;
        case "--size" when i + 1 < args.Length: size = int.Parse(args[++i]); break;
        case "--faces" when i + 1 < args.Length: faces = int.Parse(args[++i]); break;
        case "--texture" when i + 1 < args.Length: texture = int.Parse(args[++i]); break;
        case "--arm-clearance" when i + 1 < args.Length: clearance = double.Parse(args[++i], System.Globalization.CultureInfo.InvariantCulture); break;
        case "--out" when i + 1 < args.Length: outName = args[++i]; break;
        case "--cache" when i + 1 < args.Length: cacheDir = args[++i]; break;
        case "--moves" when i + 1 < args.Length: movesArg = args[++i]; break;
        case "--image-model" when i + 1 < args.Length: model = args[++i].ToLowerInvariant() is "sd" or "sdturbo" ? ImageModel.SdTurbo : ImageModel.Flux2Klein; break;
        default: if (!args[i].StartsWith("--")) words.Add(args[i]); break;
    }
}
string prompt = string.Join(' ', words);
if (ci) { if (prompt == "") prompt = "a wooden treasure chest with iron bands"; pictures = 1; rig = false; texture = 1024; faces = 50_000; }
if (prompt == "") { Console.Error.WriteLine("Usage: AssetGen.Console \"<prompt>\" [--out name] [--pictures N] [--pick] [--no-rig] [--moves a,b] ...   (see the header of Program.cs)"); return 2; }
outName ??= new string(prompt.ToLowerInvariant().Select(c => char.IsLetterOrDigit(c) ? c : '_').ToArray()).Trim('_');
if (outName.Length > 40) outName = outName[..40];

try { return await Run(); }
catch (Exception ex) { Console.Error.WriteLine($"Generation failed: {ex}"); return 1; }

async Task<int> Run()
{
    using var context = MLContext.Create().ToContext();
    var cuda = context.GetCudaDevices(); var opencl = context.GetCLDevices();
    Device device = cuda.Count > 0 ? (Device)cuda[0] : opencl.Count > 0 ? (Device)opencl[0] : context.GetPreferredDevice(preferCPU: false);
    using var accelerator = device.CreateAccelerator(context);
    Console.WriteLine($"Accelerator: {accelerator.Name} ({accelerator.AcceleratorType})");

    using var http = new HttpClient { Timeout = Timeout.InfiniteTimeSpan };
    var store = cacheDir != null ? new FileModelStore(cacheDir) : FileModelStore.Default();
    var source = new HttpClientModelSource(http, store);
    Console.WriteLine($"Model cache: {store.RootDirectory}");

    var (loader, ids) = AssetGenerator.MotionsFromDirectory(Path.Combine(AppContext.BaseDirectory, "motions"));
    var moves = movesArg != null ? movesArg.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).ToList() : ids.ToList();
    foreach (var m in moves) if (!ids.Contains(m)) { Console.Error.WriteLine($"unknown move '{m}'; available: {string.Join(", ", ids)}"); return 2; }

    using var gen = new AssetGenerator(accelerator, source, loader) { ImageModelBits = quality, BridgeHops = bridgeHops };
    var total = System.Diagnostics.Stopwatch.StartNew();
    gen.StageDone = (stage, s) => Console.WriteLine($"  {stage,-44} {s,7:F1} s   (total {total.Elapsed.TotalSeconds:F0} s)");
    gen.Status = s => { if (s.Contains('%')) Console.Write($"\r  {s,-70}"); };

    Console.WriteLine($"Prompt: \"{prompt}\"  ({model}, {pictures} picture{(pictures == 1 ? "" : "s")} from seed {seed})");
    Func<IReadOnlyList<AssetPicture>, Task<int>>? choose = null;
    if (pick)
        choose = async drawn =>
        {
            for (int i = 0; i < drawn.Count; i++) { var f = $"{outName}_candidate{i}.png"; await File.WriteAllBytesAsync(f, AssetGenerator.EncodePng(drawn[i].Rgba, drawn[i].Width, drawn[i].Height)); Console.WriteLine($"  [{i}] {f}  (seed {drawn[i].Seed})"); }
            if (Console.IsInputRedirected) return 0;
            Console.Write("Build which picture? [0] ");
            return int.TryParse(Console.ReadLine(), out int n) ? n : 0;
        };

    var asset = await gen.GenerateAsync(prompt, seed, pictures, model, size, new Trellis2Options { Seed = seed, MaxFaces = faces, TextureSize = texture },
        rig, animate ? moves : null, clearance, choose);

    await File.WriteAllBytesAsync(outName + ".png", AssetGenerator.EncodePng(asset.Picture.Rgba, asset.Picture.Width, asset.Picture.Height));
    await File.WriteAllBytesAsync(outName + ".glb", asset.Glb);
    Console.WriteLine($"\nWrote {outName}.png (seed {asset.Picture.Seed}) and {outName}.glb ({asset.Glb.Length / 1048576.0:F1} MB)");
    if (asset.RiggedGlb != null) { await File.WriteAllBytesAsync(outName + "_rigged.glb", asset.RiggedGlb); Console.WriteLine($"Wrote {outName}_rigged.glb ({asset.RiggedGlb.Length / 1048576.0:F1} MB)"); }
    if (asset.AnimatedGlb != null) { await File.WriteAllBytesAsync(outName + "_animated.glb", asset.AnimatedGlb); Console.WriteLine($"Wrote {outName}_animated.glb ({asset.AnimatedGlb.Length / 1048576.0:F1} MB, {asset.Moves.Count} moves)"); }
    Console.WriteLine($"Done in {total.Elapsed.TotalSeconds:F0} s. View a .glb at https://gltf-viewer.donmccurdy.com/ or in any glTF viewer (Blender, Windows 3D Viewer).");

    if (ci && (!File.Exists(outName + ".glb") || new FileInfo(outName + ".glb").Length < 100_000)) { Console.Error.WriteLine("CI check failed: the GLB is missing or tiny"); return 1; }
    return 0;
}
