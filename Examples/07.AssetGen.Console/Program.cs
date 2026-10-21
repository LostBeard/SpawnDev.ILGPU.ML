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
//    dotnet run -- "a dragon" --image-model flux3                           # FLUX 3 Image Klein 5.6B, with Qwen3 writing the full caption first (best pictures, slower)
//    dotnet run -- --image crown.png --no-rig                               # image to 3D: your own picture of one whole object (a transparent background is used as is, any other is cut out)
//    dotnet run -- --ci                                                     # short fixed run that asserts files were written
//
//  Options:
//    --out <name>              output name prefix / folder (default: from the prompt)
//    --seed <N>                first seed (default 1); --pictures <N> draws N seeds (default 2) and keeps the one that fits its frame best
//    --pick                    show the drawn pictures (PNG files) and ask which to build; answer "m" to draw N more (the browser demo's "Draw 4 more")
//    --image <file>            image to 3D: build this picture instead of drawing one (no prompt needed); any format ImageSharp reads
//    --no-caption              FLUX 3: do not let Qwen3 4B write the full scene caption (FLUX 3 sent a short prompt as typed draws a generic picture)
//    --size <px>               picture side for FLUX.2 and FLUX 3, a multiple of 16: 512, 768 or 1024 (default 512)
//    --edit-picture p.png|"a prompt" --edit-region "z>0.1"   after the model is built, re-texture that REGION (generator frame: x right, y depth, z up; the model fills [-0.5,0.5]^3; also box:x0,y0,z0,x1,y1,z1 and sphere:cx,cy,cz,r)
//                              (--edit-shape: change the SHAPE there too, geometry can be added or removed) from another picture (RGBA with a cut-out alpha, or any picture: it is cut out), keeping the rest; writes <name>_edited.glb. No rig / animation in this mode.
//    --edit-instruction "give the knight a red plume"   like --edit-picture, but the edit picture is made by FLUX.2 klein EDITING the picture the model was built from (reference-image mode); it keeps what the instruction does not mention
//    --image-model flux3|flux2|sd   default flux2        --quality 4|8   FLUX weights: 4-bit (default, 2.6 GB on the GPU) or 8-bit (near lossless, 4.4 GB)
//    --no-smooth-normals       write the reference's plain vertex normals (default: bilateral-filtered face normals, which take the voxel-scale facets off glossy surfaces and keep hard edges)
//    --no-remesh               skip the narrow-band dual-contouring remesh (the reference's to_glb(remesh=False) branch): simplify the decoder's own mesh instead
//    --faces <N> --texture <N> mesh triangle budget (default 300000) and texture size (default 2048)
//    --resolution 512|1024|1024c|1536c   TRELLIS.2 pipeline (the reference's types): 512 (default), 1024, or the cascades 1024c (the reference's own default) and 1536c (the finest the released model does); finer detail, slower, more GPU memory
//    --bridge-hops <N>         rig cleanup: drop skin influences from bones N+ hops from a vertex's own and cut triangles joining parts further apart (default 4; 0 = off)
//    --no-rig                  stop after the 3D model          --no-animate  stop after the rig
//    --moves a,b,c             clips to bake in (default: all 22; names are the files in motions/)
//    --arm-clearance <deg>     swing the upper arms away from the body for bulky characters (default 8)
//    --cache <dir>             where models are cached (default: your user profile; ~25 GB in all: put it on a big drive)
//
//  First run downloads the models (about 4.6 GB FLUX.2 + 11 GB TRELLIS.2 + 1.2 GB SkinTokens + small ones; FLUX 3 is 5.7 GB + 2.4 GB for the caption writer) and resumes if interrupted; every later run reads the cache.
//  FLUX 3 weights are under the FLUX Kommunity License (non-commercial use).
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
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

bool ci = false, pick = false, rig = true, animate = true, caption = true;
int seed = 1, pictures = 2, size = 512, faces = 300_000, texture = 2048, quality = 4, bridgeHops = 4, resolution = 512; bool cascade = false, smoothNormals = true, remesh = true;
double clearance = 8;
string? outName = null, cacheDir = null, movesArg = null, editPicture = null, editRegion = null, editInstruction = null, imagePath = null; bool editShape = false; var model = ImageModel.Flux2Klein;
var words = new List<string>();
for (int i = 0; i < args.Length; i++)
{
    switch (args[i])
    {
        case "--ci": ci = true; break;
        case "--pick": pick = true; break;
        case "--no-rig": rig = false; break;
        case "--no-smooth-normals": smoothNormals = false; break;
        case "--no-remesh": remesh = false; break;
        case "--bridge-hops" when i + 1 < args.Length: bridgeHops = int.Parse(args[++i]); break;
        case "--no-animate": animate = false; break;
        case "--seed" when i + 1 < args.Length: seed = int.Parse(args[++i]); break;
        case "--pictures" when i + 1 < args.Length: pictures = Math.Max(1, int.Parse(args[++i])); break;
        case "--quality" when i + 1 < args.Length: quality = int.Parse(args[++i]) == 8 ? 8 : 4; break;
        case "--resolution" when i + 1 < args.Length:
            {   // 512 | 1024 | 1024c | 1536c (the reference's pipeline types; "c" = the cascade)
                var r = args[++i].Trim().ToLowerInvariant(); cascade = r.EndsWith('c'); resolution = int.Parse(cascade ? r[..^1] : r); break;
            }
        case "--size" when i + 1 < args.Length: size = int.Parse(args[++i]); break;
        case "--faces" when i + 1 < args.Length: faces = int.Parse(args[++i]); break;
        case "--texture" when i + 1 < args.Length: texture = int.Parse(args[++i]); break;
        case "--arm-clearance" when i + 1 < args.Length: clearance = double.Parse(args[++i], System.Globalization.CultureInfo.InvariantCulture); break;
        case "--out" when i + 1 < args.Length: outName = args[++i]; break;
        case "--cache" when i + 1 < args.Length: cacheDir = args[++i]; break;
        case "--moves" when i + 1 < args.Length: movesArg = args[++i]; break;
        case "--edit-picture" when i + 1 < args.Length: editPicture = args[++i]; break;
        case "--edit-region" when i + 1 < args.Length: editRegion = args[++i]; break;
        case "--edit-shape": editShape = true; break;
        case "--edit-instruction" when i + 1 < args.Length: editInstruction = args[++i]; break;
        case "--image-model" when i + 1 < args.Length: model = args[++i].ToLowerInvariant() switch { "sd" or "sdturbo" => ImageModel.SdTurbo, "flux3" or "flux-3" => ImageModel.Flux3Klein, _ => ImageModel.Flux2Klein }; break;
        case "--image" when i + 1 < args.Length: imagePath = args[++i]; break;
        case "--no-caption": caption = false; break;
        default: if (!args[i].StartsWith("--")) words.Add(args[i]); break;
    }
}
string prompt = string.Join(' ', words);
if (ci) { if (prompt == "") prompt = "a wooden treasure chest with iron bands"; pictures = 1; rig = false; texture = 1024; faces = 50_000; }
if (prompt == "" && imagePath == null) { Console.Error.WriteLine("Usage: AssetGen.Console \"<prompt>\" | --image <file> [--out name] [--pictures N] [--pick] [--no-rig] [--moves a,b] ...   (see the header of Program.cs)"); return 2; }
if (imagePath != null && !File.Exists(imagePath)) { Console.Error.WriteLine($"--image: no such file '{imagePath}'"); return 2; }
if (outName == null)
{   // a name made from the prompt (or the picture's file name) is shortened; one given with --out is used as given
    outName = imagePath != null ? Path.GetFileNameWithoutExtension(imagePath) : new string(prompt.ToLowerInvariant().Select(c => char.IsLetterOrDigit(c) ? c : '_').ToArray()).Trim('_');
    if (outName.Length > 40) outName = outName[..40];
}

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

    using var gen = new AssetGenerator(accelerator, source, loader)
    {
        ImageModelBits = quality, BridgeHops = bridgeHops, WriteCaption = caption,
        OpenCaptionModel = (repo, file, progress) => source.OpenAsync(repo, file, progress),
        KeepImageModelResident = pick,                         // "draw more" reuses the transformer; the 3D stage releases it
    };
    var total = System.Diagnostics.Stopwatch.StartNew();
    gen.StageDone = (stage, s) => Console.WriteLine($"  {stage,-44} {s,7:F1} s   (total {total.Elapsed.TotalSeconds:F0} s)");
    gen.Status = s => { if (s.Contains('%')) Console.Write($"\r  {s,-70}"); };

    if (imagePath != null) Console.WriteLine($"Image to 3D from {imagePath}");
    else Console.WriteLine($"Prompt: \"{prompt}\"  ({model}, {pictures} picture{(pictures == 1 ? "" : "s")} from seed {seed})");

    if (editPicture != null || editInstruction != null)
    {
        var region = AssetGenerator.RegionFromSpec(editRegion ?? throw new ArgumentException("--edit-picture needs --edit-region"));
        var drawn = await gen.DrawAsync(prompt, Enumerable.Range(seed, Math.Max(1, pictures)).ToArray(), model, size);
        gen.ReleaseImageModel();
        var (cut, picked) = (Array.Empty<int>(), drawn[0]); double best = double.MaxValue;
        foreach (var p in drawn) { var (px, touch) = await gen.CutOutAsync(p); if (touch < best) { best = touch; picked = p; cut = px; } }
        using var built = await gen.BuildEditableModelAsync(cut, picked.Width, picked.Height, new Trellis2Options { Seed = picked.Seed, MaxFaces = faces, TextureSize = texture });
        await File.WriteAllBytesAsync(outName + ".png", AssetGenerator.EncodePng(picked.Rgba, picked.Width, picked.Height));
        await File.WriteAllBytesAsync(outName + ".glb", built.Glb);
        Console.WriteLine($"\nWrote {outName}.glb ({built.Glb.Length / 1048576.0:F1} MB, {built.Tokens} tokens)");
        int ew, eh; byte[] rgba;
        if (editInstruction != null)
        {   // FLUX.2 klein edits the picture the model was built from: the instruction says what to change, the rest stays
            var ep = await gen.EditPictureAsync(picked, editInstruction, seed + 500);
            gen.ReleaseImageModel();
            (ew, eh, rgba) = (ep.Width, ep.Height, ep.Rgba);
            await File.WriteAllBytesAsync(outName + "_editpicture.png", AssetGenerator.EncodePng(rgba, ew, eh));
        }
        else if (File.Exists(editPicture))
        {
            var png = await File.ReadAllBytesAsync(editPicture);
            (ew, eh) = SpawnDev.ILGPU.ML.Preprocessing.PngDecoder.GetDimensions(png);
            rgba = SpawnDev.ILGPU.ML.Preprocessing.PngDecoder.DecodePixels(png) ?? throw new InvalidDataException("the edit picture must be an 8-bit, non-interlaced PNG");
        }
        else
        {   // not a file: it is a prompt, the edit picture is drawn from it with the same image model
            var ep = (await gen.DrawAsync(editPicture, new[] { seed + 500 }, model, size))[0];
            gen.ReleaseImageModel();
            (ew, eh, rgba) = (ep.Width, ep.Height, ep.Rgba);
            await File.WriteAllBytesAsync(outName + "_editpicture.png", AssetGenerator.EncodePng(rgba, ew, eh));
        }
        var epix = new int[ew * eh]; Buffer.BlockCopy(rgba, 0, epix, 0, epix.Length * 4);
        bool hasAlpha = epix.Any(v => ((v >> 24) & 255) < 255);
        int[] ecut = epix;
        if (!hasAlpha) ecut = (await gen.CutOutAsync(new AssetPicture(0, ew, eh, rgba))).Pixels;
        using var edited = editShape
            ? await gen.EditShapeRegionAsync(built, ecut, ew, eh, region, new Trellis2Options { Seed = picked.Seed, MaxFaces = faces, TextureSize = texture }, seed)
            : await gen.EditRegionAsync(built, ecut, ew, eh, region, new Trellis2Options { Seed = picked.Seed, MaxFaces = faces, TextureSize = texture }, seed);
        await File.WriteAllBytesAsync(outName + "_edited.glb", edited.Glb);
        Console.WriteLine($"Wrote {outName}_edited.glb ({edited.Glb.Length / 1048576.0:F1} MB). Done in {total.Elapsed.TotalSeconds:F0} s.");
        return 0;
    }

    var trellisOptions = new Trellis2Options { Seed = seed, MaxFaces = faces, TextureSize = texture, Resolution = resolution, Cascade = cascade, SmoothNormals = smoothNormals, Remesh = remesh };
    GeneratedAsset asset;
    if (imagePath != null)
    {   // image to 3D: a transparent picture is used as is, any other is cut out first
        var photo = LoadPicture(imagePath, seed);
        asset = await gen.GenerateFromPictureAsync(photo, trellisOptions, rig, animate ? moves : null, clearance);
    }
    else if (pick)
    {   // draw, look at the pictures, choose one or ask for more
        var all = new List<AssetPicture>(); int next = seed; int index = 0;
        while (true)
        {
            var batch = await gen.DrawAsync(prompt, Enumerable.Range(next, pictures).ToArray(), model, size);
            next += pictures;
            foreach (var p in batch)
            {
                var f = $"{outName}_candidate{all.Count}.png"; await File.WriteAllBytesAsync(f, AssetGenerator.EncodePng(p.Rgba, p.Width, p.Height));
                Console.WriteLine($"  [{all.Count}] {f}  (seed {p.Seed})"); all.Add(p);
            }
            if (gen.UsedCaption != null) Console.WriteLine($"  caption used: {gen.UsedCaption}");
            if (Console.IsInputRedirected) break;
            Console.Write($"Build which picture? [0], or m to draw {pictures} more: ");
            var answer = Console.ReadLine()?.Trim();
            if (string.Equals(answer, "m", StringComparison.OrdinalIgnoreCase)) continue;
            index = int.TryParse(answer, out int n) ? Math.Clamp(n, 0, all.Count - 1) : 0;
            break;
        }
        gen.ReleaseImageModel();
        asset = await gen.GenerateFromPictureAsync(all[index], trellisOptions, rig, animate ? moves : null, clearance);
    }
    else
        asset = await gen.GenerateAsync(prompt, seed, pictures, model, size, trellisOptions, rig, animate ? moves : null, clearance);
    if (gen.UsedCaption != null && !pick) Console.WriteLine($"Caption used: {gen.UsedCaption}");

    await File.WriteAllBytesAsync(outName + ".png", AssetGenerator.EncodePng(asset.Picture.Rgba, asset.Picture.Width, asset.Picture.Height));
    await File.WriteAllBytesAsync(outName + ".glb", asset.Glb);
    // what the 3D stage actually saw: the picture with its background removed (the alpha is the cut-out). A bad cut-out is a bad model, so it is always written.
    var cutBytes = new byte[asset.CutOut.Length * 4]; Buffer.BlockCopy(asset.CutOut, 0, cutBytes, 0, cutBytes.Length);
    await File.WriteAllBytesAsync(outName + "_cutout.png", AssetGenerator.EncodePng(cutBytes, asset.Picture.Width, asset.Picture.Height));
    Console.WriteLine($"\nWrote {outName}.png (seed {asset.Picture.Seed}), {outName}_cutout.png and {outName}.glb ({asset.Glb.Length / 1048576.0:F1} MB)");
    if (asset.RiggedGlb != null) { await File.WriteAllBytesAsync(outName + "_rigged.glb", asset.RiggedGlb); Console.WriteLine($"Wrote {outName}_rigged.glb ({asset.RiggedGlb.Length / 1048576.0:F1} MB)"); }
    if (asset.AnimatedGlb != null) { await File.WriteAllBytesAsync(outName + "_animated.glb", asset.AnimatedGlb); Console.WriteLine($"Wrote {outName}_animated.glb ({asset.AnimatedGlb.Length / 1048576.0:F1} MB, {asset.Moves.Count} moves)"); }
    Console.WriteLine($"Done in {total.Elapsed.TotalSeconds:F0} s. View a .glb at https://gltf-viewer.donmccurdy.com/ or in any glTF viewer (Blender, Windows 3D Viewer).");

    if (ci && (!File.Exists(outName + ".glb") || new FileInfo(outName + ".glb").Length < 100_000)) { Console.Error.WriteLine("CI check failed: the GLB is missing or tiny"); return 1; }
    return 0;
}

// A picture file as RGBA (any format ImageSharp reads; a transparent background is kept: the 3D stage then uses it as the cut-out).
static AssetPicture LoadPicture(string path, int seed)
{
    using var img = Image.Load<Rgba32>(path);
    var rgba = new byte[img.Width * img.Height * 4];
    img.CopyPixelDataTo(rgba);
    return new AssetPicture(seed, img.Width, img.Height, rgba);
}
