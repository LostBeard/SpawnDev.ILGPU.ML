// ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
//  SpawnDev.ILGPU.ML — Example 09: a scene from a sentence (desktop, CUDA / OpenCL)
//
//    dotnet run -- "a blacksmith's workshop" --out workshop
//    dotnet run -- "a small farm yard with a barn" --llm gemma4:12b --objects 5 --faces 60000
//    dotnet run -- --layout workshop_layout.json --out workshop     # skip the LLM: build a layout you wrote or edited
//
//  1. A local LLM (an Ollama model name such as gemma4:12b, or a .gguf path) writes a LAYOUT as JSON: the objects of the scene, how tall each is, where each one stands.
//  2. The LLM is freed. Every distinct object is generated once: FLUX.2 klein picture -> RMBG-1.4 cut-out -> TRELLIS.2 3D (the same chain as Example 07).
//  3. GltfSceneComposer puts them in ONE glTF scene (a ground quad, one node per instance, repeated objects stored once, each scaled to its height and stood on the ground).
//
//    --llm <name|file.gguf>   default qwen3:14b       --objects N   about how many different kinds of object (default 6)
//    --pictures N             pictures drawn per object, the cleanest cut-out is built (default 2)
//    --faces N / --texture N  per object triangle budget (default 80000) / texture size (default 1024)
//    --plan-only              write <name>_layout.json and stop (look at, edit, then rerun with --layout)
//    --seed N   --out name    --cache dir (model cache)   --layout file.json (use this layout, no LLM)
//
//  Outputs: <name>.glb (the scene), <name>_layout.json (the layout, editable), <name>_<object>.glb (each object). Each object takes about 1.5 minutes on an RTX 4070.
//  Needs a GPU with about 10 GB free for the LLM stage (it is freed before the 3D stages). The accelerator is created here (the app owns it).
// ─────────────────────────────────────────────────────────────────────────────────────────────────────────────

using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using ILGPU.Runtime.OpenCL;
using OllamaServer.Console;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.Assets;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.ILGPU.ML.Trellis2;

bool planOnly = false;
int seed = 1, objects = 6, pictures = 2, faces = 80_000, texture = 1024;
string? outName = null, cacheDir = null, layoutFile = null, llm = "qwen3:14b";
var words = new List<string>();
for (int i = 0; i < args.Length; i++)
{
    switch (args[i])
    {
        case "--seed" when i + 1 < args.Length: seed = int.Parse(args[++i]); break;
        case "--objects" when i + 1 < args.Length: objects = Math.Clamp(int.Parse(args[++i]), 1, 12); break;
        case "--pictures" when i + 1 < args.Length: pictures = Math.Max(1, int.Parse(args[++i])); break;
        case "--faces" when i + 1 < args.Length: faces = int.Parse(args[++i]); break;
        case "--texture" when i + 1 < args.Length: texture = int.Parse(args[++i]); break;
        case "--out" when i + 1 < args.Length: outName = args[++i]; break;
        case "--cache" when i + 1 < args.Length: cacheDir = args[++i]; break;
        case "--llm" when i + 1 < args.Length: llm = args[++i]; break;
        case "--layout" when i + 1 < args.Length: layoutFile = args[++i]; break;
        case "--plan-only": planOnly = true; break;
        default: if (!args[i].StartsWith("--")) words.Add(args[i]); break;
    }
}
string description = string.Join(' ', words);
if (description == "" && layoutFile == null) { Console.Error.WriteLine("Usage: SceneGen.Console \"<scene description>\" [--out name] [--llm gemma4:12b] [--objects N] ...   (see the header of Program.cs)"); return 2; }
outName ??= layoutFile != null ? Path.GetFileNameWithoutExtension(layoutFile).Replace("_layout", "") : new string(description.ToLowerInvariant().Select(c => char.IsLetterOrDigit(c) ? c : '_').ToArray()).Trim('_');
if (outName.Length > 40) outName = outName[..40];

// the LLM fast paths are off by default in the library (12 vs 52 tokens/s measured on this card): the app turns them on, as Example 06 does
SpawnDev.ILGPU.ML.Kernels.FusedAttentionKernel.EnableGroupedAttention = true;
SpawnDev.ILGPU.ML.GGUF.GGUFGraphBuilder.EnableLastPositionLogits = true;
SpawnDev.ILGPU.ML.Kernels.FusedDequantMatMul.EnableWarpGemv = true;
SpawnDev.ILGPU.ML.Kernels.FusedDequantMatMul.EnableDp4aGemv = true;

try { return await Run(); }
catch (Exception ex) { Console.Error.WriteLine($"Scene generation failed: {ex}"); return 1; }

async Task<int> Run()
{
    using var context = MLContext.Create().ToContext();
    var cuda = context.GetCudaDevices(); var opencl = context.GetCLDevices();
    Device device = cuda.Count > 0 ? (Device)cuda[0] : opencl.Count > 0 ? (Device)opencl[0] : context.GetPreferredDevice(preferCPU: false);
    using var accelerator = device.CreateAccelerator(context);
    Console.WriteLine($"Accelerator: {accelerator.Name} ({accelerator.AcceleratorType})");
    var total = System.Diagnostics.Stopwatch.StartNew();

    SceneLayout layout;
    if (layoutFile != null)
    {
        layout = SceneLayout.Parse(await File.ReadAllTextAsync(layoutFile));
        Console.WriteLine($"Layout from {layoutFile}: {layout.Objects.Count} kinds of object");
    }
    else
    {
        string? gguf = File.Exists(llm) ? llm : new OllamaModelStore().Resolve(llm!)?.GgufPath;
        if (gguf == null || !File.Exists(gguf)) { Console.Error.WriteLine($"No such model: {llm} (an Ollama name that is in its cache, or a .gguf file)"); return 2; }
        Console.WriteLine($"Planning the scene with {llm} ...");
        using (var pipe = await GgufTextGenerationPipeline.CreateFromFileAsync(accelerator, gguf, maxSeqLen: 4096))
            layout = await SceneLayoutPlanner.FromPipeline(pipe).PlanAsync(description, objects);
        accelerator.Synchronize();                                       // the LLM's memory is free before the 3D models load
        Console.WriteLine($"  {layout.Objects.Count} kinds of object, {layout.Objects.Sum(o => o.Instances.Count)} placements ({total.Elapsed.TotalSeconds:F0} s)");
    }
    await File.WriteAllTextAsync(outName + "_layout.json", layout.ToJson());
    foreach (var o in layout.Objects) Console.WriteLine($"  - {o.Name} x{o.Instances.Count}, {o.Height:F2} m: {o.Prompt}");

    if (planOnly) { Console.WriteLine($"Wrote {outName}_layout.json"); return 0; }

    using var http = new HttpClient { Timeout = Timeout.InfiniteTimeSpan };
    var store = cacheDir != null ? new FileModelStore(cacheDir) : FileModelStore.Default();
    var source = new HttpClientModelSource(http, store);
    Console.WriteLine($"Model cache: {store.RootDirectory}");
    using var assets = new AssetGenerator(accelerator, source);
    assets.StageDone = (stage, s) => { if (s > 5) Console.WriteLine($"    {stage,-40} {s,6:F1} s"); };
    var scene = new SceneGenerator(assets) { Progress = p => Console.WriteLine($"[{total.Elapsed.TotalSeconds:F0} s] {p}") };
    var result = await scene.BuildAsync(layout, seed, pictures, ImageModel.Flux2Klein, 512, new Trellis2Options { MaxFaces = faces, TextureSize = texture },
        onAsset: async (name, glb) => await File.WriteAllBytesAsync($"{outName}_{new string(name.ToLowerInvariant().Select(c => char.IsLetterOrDigit(c) ? c : '_').ToArray())}.glb", glb));
    await File.WriteAllBytesAsync(outName + ".glb", result.Glb);
    Console.WriteLine($"\nWrote {outName}.glb ({result.Glb.Length / 1048576.0:F1} MB, {layout.Objects.Count} kinds of object, {layout.Objects.Sum(o => o.Instances.Count)} placements) in {total.Elapsed.TotalMinutes:F1} min.");
    return 0;
}
