using System.IO.Compression;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Animation;
using SpawnDev.ILGPU.ML.Flux2;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.ILGPU.ML.SkinTokens;
using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnDev.ILGPU.ML.Trellis2;
using SpawnDev.ILGPU.ML.UniRig;

namespace SpawnDev.ILGPU.ML.Assets;

/// <summary>Which model draws the picture.</summary>
public enum ImageModel
{
    /// <summary>FLUX.2 [klein] 4B, 4 steps (Black Forest Labs, Apache-2.0), the 4-bit export: sharper, better anatomy.</summary>
    Flux2Klein,
    /// <summary>SD-Turbo, one step (Stability AI): faster, smaller, a 2023 look.</summary>
    SdTurbo,
}

/// <summary>A picture drawn for a prompt.</summary>
public sealed record AssetPicture(int Seed, int Width, int Height, byte[] Rgba);

/// <summary>
/// A generated 3D model that keeps its latents, so a region of it can be re-textured (<see cref="AssetGenerator.EditRegionAsync"/>). Holds a few GB of device memory until disposed
/// (the mesh and per-voxel attributes); the GLB is already built.
/// </summary>
public sealed class EditableModel : IDisposable
{
    public byte[] Glb { get; }
    internal Trellis2Pipeline.Output Output { get; }
    internal EditableModel(byte[] glb, Trellis2Pipeline.Output output) { Glb = glb; Output = output; }
    /// <summary>Tokens of the structured latent (the unit a region is made of).</summary>
    public int Tokens => Output.Edit?.Count ?? 0;
    public void Dispose() => Output.Dispose();
}

/// <summary>What <see cref="AssetGenerator.GenerateAsync"/> made.</summary>
public sealed record GeneratedAsset(AssetPicture Picture, int[] CutOut, byte[] Glb, byte[]? RiggedGlb, byte[]? AnimatedGlb, IReadOnlyList<string> Moves, float? PromptMatch = null);

/// <summary>
/// The whole text-to-character chain on the native GPU backends (CUDA, OpenCL) or WebGPU, the same stages as the browser demo, as one object an app can drive step by step or in one call:
/// <b>draw</b> (FLUX.2 klein or SD-Turbo) -> <b>cut out</b> (RMBG-1.4) -> <b>3D</b> (TRELLIS.2, textured PBR GLB) -> <b>rig</b> (SkinTokens: skeleton and learned skin weights) -> <b>animate</b>
/// (NVIDIA Kimodo clips retargeted onto the skeleton, glTF animations). Every model streams from <paramref name="source"/> on first use (an <see cref="HttpClientModelSource"/> over a
/// <see cref="FileModelStore"/> on a desktop) and is loaded one stage at a time, so a 12 GB card is enough.
/// </summary>
public sealed class AssetGenerator : IDisposable
{
    public const string FluxRepo = "LostBeard/FLUX.2-klein-4B-q4-ILGPU", SkinTokensRepo = "LostBeard/SkinTokens-bf16";
    /// <summary>Appended to the prompt: one whole object, centred, plain white background, so the cut-out and the 3D stage get a clean subject.</summary>
    /// <summary>The triangle budget the 3D stage simplifies to by default. 300k: at 150k the edge collapse starts to eat the thin shells of armour and cloth (dark flecks and cracks; measured on a knight, 497k faces before simplification), at 300k they are mostly kept.</summary>
    public const int DefaultMaxFaces = 300_000;
    public const string PromptSuffix = ", a single object, centered, the whole object in frame with an empty margin around it, isolated on a plain white background, soft studio lighting, high detail";

    private readonly Accelerator _acc;
    private readonly IModelSource _source;
    private readonly Func<string, Task<byte[]>>? _motions;
    private Flux2Pipeline? _flux;
    private ImageGenerationPipeline? _sd;
    private BackgroundRemovalPipeline? _rmbg;
    private Trellis2ImageTo3D? _trellis;

    /// <summary>Called as each stage finishes: (name, seconds).</summary>
    public Action<string, double>? StageDone { get; set; }
    /// <summary>Called with what is happening now ("drawing", "cutting out", ...), and download / load progress lines.</summary>
    public Action<string>? Status { get; set; }
    /// <summary>Keep the image model's transformer on the device after a draw, so drawing more of the same prompt is quick (about 8 s a picture in a browser, ~2 s on a desktop GPU).
    /// <see cref="ReleaseImageModel"/> (called automatically before the 3D stage) frees it.</summary>
    public bool KeepImageModelResident { get; set; }
    /// <summary>FLUX.2 weight precision: 4 (default; 4.4 GB download, 2.6 GB on the GPU at a time) or 8 (near lossless; 7.9 GB, 4.4 GB at a time). Set before the first draw.</summary>
    public int ImageModelBits { get; set; } = 4;
    /// <summary>
    /// Check the drawn pictures against the prompt with CLIP (<see cref="PromptMatchScorer"/>, 600 MB downloaded once) when <see cref="GenerateAsync"/> draws several: the picture that best matches the words
    /// and fits its frame is built, and when even the best one is below <see cref="MinPromptMatch"/> another batch is drawn (<see cref="LowMatchRetries"/> times) before the minutes-long 3D stage is spent on it. Default on.
    /// </summary>
    public bool ScorePictures { get; set; } = true;
    /// <summary>The CLIP similarity below which a picture is taken for "not of the thing asked for". Matched pairs scored 0.24-0.31 and other objects 0.13-0.19 (knight pictures, the cat and portrait samples); 0.21 sits between.</summary>
    public float MinPromptMatch { get; set; } = 0.21f;
    /// <summary>How many more batches of pictures are drawn when the best of a batch matches the prompt worse than <see cref="MinPromptMatch"/>.</summary>
    public int LowMatchRetries { get; set; } = 1;
    private PromptMatchScorer? _scorer;

    /// <summary>The rig cleanup (<see cref="SkinTokensRigger.BridgeHops"/>): 4 by default, 0 keeps the learned weights and the mesh untouched.</summary>
    public int BridgeHops { get; set; } = 4;

    /// <param name="motionClip">Loads a Kimodo clip by id (the packed .npz bytes); null disables <see cref="AnimateAsync"/>. See <see cref="MotionsFromDirectory"/>.</param>
    public AssetGenerator(Accelerator accelerator, IModelSource source, Func<string, Task<byte[]>>? motionClip = null)
    {
        _acc = accelerator; _source = source; _motions = motionClip;
    }

    /// <summary>A motion loader over a folder of <c>*.bin</c> clips (the demo's wwwroot/motions), and the ids it holds.</summary>
    public static (Func<string, Task<byte[]>> Loader, IReadOnlyList<string> Ids) MotionsFromDirectory(string directory)
    {
        var ids = Directory.Exists(directory) ? Directory.GetFiles(directory, "*.bin").Select(f => Path.GetFileNameWithoutExtension(f)).OrderBy(x => x, StringComparer.Ordinal).ToList() : new List<string>();
        return (id => File.ReadAllBytesAsync(Path.Combine(directory, id + ".bin")), ids);
    }

    private async Task<T> Timed<T>(string stage, Func<Task<T>> work)
    {
        Status?.Invoke(stage);
        var sw = System.Diagnostics.Stopwatch.StartNew();
        var r = await work();
        StageDone?.Invoke(stage, sw.Elapsed.TotalSeconds);
        return r;
    }

    // ───────────────────────────── 1. draw ─────────────────────────────

    /// <summary>Draw one picture per seed. <paramref name="size"/> (FLUX.2 only) is the side in pixels, a multiple of 16: 512 is the verified default; larger ones are drawn tile by tile in the VAE so memory stays flat.</summary>
    public async Task<List<AssetPicture>> DrawAsync(string prompt, int[] seeds, ImageModel model = ImageModel.Flux2Klein, int size = 512, Func<AssetPicture, Task>? onEach = null, bool addSuffix = true)
    {
        var full = addSuffix ? prompt + PromptSuffix : prompt;
        var pictures = new List<AssetPicture>();
        if (model == ImageModel.SdTurbo)
        {
            _sd ??= await Timed("load SD-Turbo", () => ImageGenerationPipeline.CreateAsync(_acc, _source, ModelHub.KnownModels.SDTurbo, (s, p) => Status?.Invoke($"SD-Turbo {s} {p}%")));
            foreach (var seed in seeds)
            {
                var img = await Timed($"draw seed {seed}", () => _sd.RunAsync(new ImageGenerationInput { Prompt = full, Seed = seed }));
                var pic = new AssetPicture(seed, img.Width, img.Height, img.ImageRGBA);
                pictures.Add(pic); if (onEach != null) await onEach(pic);
            }
            return pictures;
        }
        _flux ??= await Timed("load FLUX.2 klein", () => Flux2Pipeline.CreateAsync(_acc, _source, FluxRepo, 512, (s, p) => Status?.Invoke($"FLUX.2 klein {s} {p}%"), default, ImageModelBits));
        _flux.KeepTransformerResident = KeepImageModelResident;
        var sw = System.Diagnostics.Stopwatch.StartNew();
        await _flux.RunManyAsync(new Flux2Request { Prompt = full, Size = size, Steps = 4, Progress = (s, i, n) => Status?.Invoke($"FLUX.2 klein: {s}") }, seeds, async (i, img) =>
        {
            var pic = new AssetPicture(seeds[i], img.Width, img.Height, img.Rgba);
            pictures.Add(pic);
            StageDone?.Invoke($"draw seed {seeds[i]}", sw.Elapsed.TotalSeconds); sw.Restart();
            if (onEach != null) await onEach(pic);
        });
        return pictures;
    }

    /// <summary>Free the image model's resident transformer (<see cref="KeepImageModelResident"/>) so the 3D stage has the GPU.</summary>
    public void ReleaseImageModel() => _flux?.ReleaseResident();

    // ───────────────────────────── 2. cut out ─────────────────────────────

    /// <summary>The picture with its background removed (RMBG-1.4): RGBA pixels as ints whose alpha is the foreground mask, and the share of the frame's border the subject touches (0 = a whole object with margin,
    /// high = cropped by the frame, which makes a truncated 3D model).</summary>
    public async Task<(int[] Pixels, double BorderTouch)> CutOutAsync(AssetPicture picture)
    {
        _rmbg ??= await Timed("load RMBG-1.4", async () =>
        {
            await using var s = await _source.OpenAsync(ModelHub.KnownModels.RMBG14, ModelHub.KnownFiles.OnnxModel);
            return new BackgroundRemovalPipeline(await InferenceSession.CreateFromStreamAsync(_acc, s), _acc);
        });
        var px = new int[picture.Width * picture.Height];
        Buffer.BlockCopy(picture.Rgba, 0, px, 0, px.Length * 4);
        for (int i = 0; i < px.Length; i++) px[i] |= unchecked((int)0xFF000000);
        var cut = await Timed($"cut out seed {picture.Seed}", async () => (await _rmbg.RemoveBackgroundAsync(px, picture.Width, picture.Height)).ResultPixels);
        _rmbg.ReleaseWorkingMemory();                                  // its 1024 px feature maps would otherwise stay parked while the next model runs
        return (cut, BorderTouch(cut, picture.Width, picture.Height));
    }

    /// <summary>Share of the outer 2-pixel ring of the frame that the cut-out subject covers.</summary>
    public static double BorderTouch(int[] px, int w, int h)
    {
        int ring = 0, hit = 0;
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                if (x >= 2 && y >= 2 && x < w - 2 && y < h - 2) continue;
                ring++;
                if (((px[y * w + x] >> 24) & 255) > 127) hit++;
            }
        return ring == 0 ? 0 : (double)hit / ring;
    }

    // ───────────────────────────── 3. 3D ─────────────────────────────

    /// <summary>TRELLIS.2: the cut-out RGBA picture to a textured PBR GLB. The image model is released first (the 3D stages want the memory).</summary>
    public async Task<byte[]> BuildModelAsync(int[] cutOutPixels, int width, int height, Trellis2Options? options = null)
    {
        ReleaseImageModel();
        _trellis ??= new Trellis2ImageTo3D(_acc, Trellis2Weights.FromModelSource(_source)) { StageDone = (n, s) => StageDone?.Invoke(n, s) };
        Status?.Invoke("building the 3D model (TRELLIS.2)");
        return await _trellis.GenerateGlbAsync(cutOutPixels, width, height, options);
    }

    /// <summary><see cref="BuildModelAsync"/>, keeping the latents so a region can be edited later with <see cref="EditRegionAsync"/>. The caller disposes the model.</summary>
    public async Task<EditableModel> BuildEditableModelAsync(int[] cutOutPixels, int width, int height, Trellis2Options? options = null)
    {
        options ??= new Trellis2Options();
        ReleaseImageModel();
        _trellis ??= new Trellis2ImageTo3D(_acc, Trellis2Weights.FromModelSource(_source)) { StageDone = (n, s) => StageDone?.Invoke(n, s) };
        Status?.Invoke("building the 3D model (TRELLIS.2)");
        using var pixels = RgbaUpload.FromManaged(_acc, cutOutPixels, width, height);
        var (output, pipe) = await _trellis.GenerateAsync(pixels.View, width, height, options);
        pipe.RepairMesh = options.RepairMesh; pipe.Remesh = options.Remesh;
        var glb = await pipe.ToGlbAsync(output, options.TextureSize, maxFaces: options.MaxFaces);
        return new EditableModel(glb, output);
    }

    /// <summary>
    /// Re-texture a region of <paramref name="source"/> from a new cut-out picture: tokens whose centre (world coordinates, the model lives in [-0.5, 0.5]^3,
    /// generator frame: x right, y depth, z up; the GLB writer turns z up into y up) is inside <paramref name="inRegion"/> are regenerated under the new picture, everything else keeps its texture; the geometry does not change.
    /// Returns a new editable model (so edits can be chained); <paramref name="source"/> is left alone.
    /// </summary>
    public async Task<EditableModel> EditRegionAsync(EditableModel source, int[] cutOutPixels, int width, int height, Func<float, float, float, bool> inRegion, Trellis2Options? options = null, int seed = 1)
    {
        options ??= new Trellis2Options();
        ReleaseImageModel();
        _trellis ??= new Trellis2ImageTo3D(_acc, Trellis2Weights.FromModelSource(_source)) { StageDone = (n, s) => StageDone?.Invoke(n, s) };
        Status?.Invoke("editing a region (TRELLIS.2 texture flow)");
        using var pixels = RgbaUpload.FromManaged(_acc, cutOutPixels, width, height);
        var (output, pipe) = await _trellis.EditRegionAsync(source.Output, pixels.View, width, height, inRegion, seed);
        pipe.RepairMesh = options.RepairMesh; pipe.Remesh = options.Remesh;
        var glb = await pipe.ToGlbAsync(output, options.TextureSize, maxFaces: options.MaxFaces);
        return new EditableModel(glb, output);
    }

    /// <summary>
    /// Edit the SHAPE inside a region of <paramref name="source"/> from a new cut-out picture: geometry can be added or removed inside the region (the structure, shape and texture are re-sampled there
    /// under the new picture), everything outside keeps its shape and texture. Same region language as <see cref="EditRegionAsync"/>. Returns a new editable model; <paramref name="source"/> is left alone.
    /// </summary>
    public async Task<EditableModel> EditShapeRegionAsync(EditableModel source, int[] cutOutPixels, int width, int height, Func<float, float, float, bool> inRegion, Trellis2Options? options = null, int seed = 1)
    {
        options ??= new Trellis2Options();
        ReleaseImageModel();
        _trellis ??= new Trellis2ImageTo3D(_acc, Trellis2Weights.FromModelSource(_source)) { StageDone = (n, s) => StageDone?.Invoke(n, s) };
        Status?.Invoke("editing the shape of a region (TRELLIS.2 structure, shape and texture flows)");
        using var pixels = RgbaUpload.FromManaged(_acc, cutOutPixels, width, height);
        var (output, pipe) = await _trellis.EditShapeRegionAsync(source.Output, pixels.View, width, height, inRegion, seed);
        pipe.RepairMesh = options.RepairMesh; pipe.Remesh = options.Remesh;
        var glb = await pipe.ToGlbAsync(output, options.TextureSize, maxFaces: options.MaxFaces);
        return new EditableModel(glb, output);
    }

    /// <summary>
    /// A region from text, in the generator's frame (x right, y depth, z up; the model fills [-0.5, 0.5]^3): <c>z&gt;0.1</c>, <c>x&lt;-0.05</c> (one axis against a bound), <c>box:x0,y0,z0,x1,y1,z1</c>,
    /// <c>sphere:cx,cy,cz,r</c>. Several joined with <c>;</c> are a union.
    /// </summary>
    public static Func<float, float, float, bool> RegionFromSpec(string spec)
    {
        var parts = new List<Func<float, float, float, bool>>();
        var inv = System.Globalization.CultureInfo.InvariantCulture;
        foreach (var raw in spec.Split(';', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
        {
            var p = raw.ToLowerInvariant();
            if (p.StartsWith("box:"))
            {
                var v = p[4..].Split(',').Select(t => float.Parse(t, inv)).ToArray();
                if (v.Length != 6) throw new FormatException("box:x0,y0,z0,x1,y1,z1");
                parts.Add((x, y, z) => x >= Math.Min(v[0], v[3]) && x <= Math.Max(v[0], v[3]) && y >= Math.Min(v[1], v[4]) && y <= Math.Max(v[1], v[4]) && z >= Math.Min(v[2], v[5]) && z <= Math.Max(v[2], v[5]));
            }
            else if (p.StartsWith("sphere:"))
            {
                var v = p[7..].Split(',').Select(t => float.Parse(t, inv)).ToArray();
                if (v.Length != 4) throw new FormatException("sphere:cx,cy,cz,r");
                parts.Add((x, y, z) => (x - v[0]) * (x - v[0]) + (y - v[1]) * (y - v[1]) + (z - v[2]) * (z - v[2]) <= v[3] * v[3]);
            }
            else if (p.Length > 2 && "xyz".Contains(p[0]) && (p[1] == '>' || p[1] == '<'))
            {
                int axis = p[0] - 'x'; bool greater = p[1] == '>'; float bound = float.Parse(p[2..], inv);
                parts.Add((x, y, z) => { float c = axis == 0 ? x : axis == 1 ? y : z; return greater ? c > bound : c < bound; });
            }
            else throw new FormatException($"unknown region '{raw}' (use z>0.1, box:..., sphere:...)");
        }
        if (parts.Count == 0) throw new FormatException("an empty region");
        return (x, y, z) => { foreach (var f in parts) if (f(x, y, z)) return true; return false; };
    }

    // ───────────────────────────── 4. rig ─────────────────────────────

    /// <summary>SkinTokens: a skeleton and learned skin weights for a humanoid GLB, one pass. The weights (1.15 GB) are loaded for the call and freed after it.</summary>
    public async Task<byte[]> RigAsync(byte[] glb)
    {
        ReleaseImageModel();
        using var weights = await Timed("load SkinTokens", () => SkinTokensWeights.LoadAsync(_acc, _source, SkinTokensRepo));
        using var rigger = new SkinTokensRigger(_acc, weights) { BridgeHops = BridgeHops, StageDone = (n, s) => StageDone?.Invoke("rig: " + n, s) };
        var rig = await Timed("rig", () => rigger.RigGlbAsync(glb));
        if (rigger.LastCleanup.Vertices > 0 || rigger.LastCleanup.Triangles > 0) StageDone?.Invoke($"rig cleanup: {rigger.LastCleanup.Vertices} vertices lost far influences, {rigger.LastCleanup.Triangles} bridge triangles cut", 0);
        return rig.Glb;
    }

    // ───────────────────────────── 5. animate ─────────────────────────────

    /// <summary>Retarget Kimodo clips onto a rigged humanoid and write them as glTF animations. <paramref name="armClearanceDegrees"/> swings the upper arms away from the body (bulky characters).
    /// Throws <see cref="NotAHumanoidException"/> when the skeleton is not a standing humanoid.</summary>
    public async Task<byte[]> AnimateAsync(byte[] riggedGlb, IEnumerable<string> moveIds, double armClearanceDegrees = 8)
    {
        if (_motions == null) throw new InvalidOperationException("no motion clips: pass a motion loader to the constructor (MotionsFromDirectory)");
        var sk = RiggedGlb.ReadSkeleton(riggedGlb);
        var heads = Enumerable.Range(0, sk.Count).Select(i => new Vec3(sk.Joints[i * 3], sk.Joints[i * 3 + 1], sk.Joints[i * 3 + 2])).ToArray();
        var retargeter = new HumanoidRetargeter(sk.Parents, heads);
        var clips = new List<RetargetedClip>();
        foreach (var id in moveIds) clips.Add(retargeter.Retarget(MotionClip.FromNpz(await _motions(id), id), 1.0, armClearanceDegrees));
        Status?.Invoke("animate");
        return AnimatedGlb.Add(riggedGlb, clips);
    }

    // ───────────────────────────── the whole chain ─────────────────────────────

    /// <summary>Everything from a prompt: draw <paramref name="pictures"/> seeds, keep the one whose cut-out fits its frame best, build, rig and animate it.</summary>
    /// <param name="choose">Pick one of the drawn pictures yourself (return its index) instead of the best fit; null = automatic.</param>
    public async Task<GeneratedAsset> GenerateAsync(string prompt, int seed = 1, int pictures = 2, ImageModel model = ImageModel.Flux2Klein, int size = 512,
        Trellis2Options? trellis = null, bool rig = true, IEnumerable<string>? moves = null, double armClearanceDegrees = 8, Func<IReadOnlyList<AssetPicture>, Task<int>>? choose = null)
    {
        var drawn = await DrawAsync(prompt, Enumerable.Range(seed, Math.Max(1, pictures)).ToArray(), model, size);
        AssetPicture picked; int[] cut; float? match = null;
        if (choose != null)
        {
            picked = drawn[Math.Clamp(await choose(drawn), 0, drawn.Count - 1)];
            ReleaseImageModel();
            cut = (await CutOutAsync(picked)).Pixels;
        }
        else
        {
            ReleaseImageModel();
            (picked, cut, match) = await ChooseBestAsync(prompt, drawn);
            for (int retry = 1; retry <= LowMatchRetries && match != null && match < MinPromptMatch; retry++)
            {
                Status?.Invoke($"no picture matches \"{prompt}\" (best {match:F2} < {MinPromptMatch:F2}): drawing {drawn.Count} more");
                var more = await DrawAsync(prompt, Enumerable.Range(seed + 1000 * retry, Math.Max(1, pictures)).ToArray(), model, size);
                ReleaseImageModel();
                var (p2, c2, m2) = await ChooseBestAsync(prompt, more);
                if (m2 > match) { picked = p2; cut = c2; match = m2; }
            }
        }
        var glb = await BuildModelAsync(cut, picked.Width, picked.Height, trellis ?? new Trellis2Options { Seed = picked.Seed, MaxFaces = DefaultMaxFaces });
        byte[]? rigged = null, animated = null; var used = new List<string>();
        if (rig)
        {
            rigged = await RigAsync(glb);
            if (_motions != null)
            {
                used.AddRange(moves ?? Array.Empty<string>());
                if (used.Count > 0) animated = await AnimateAsync(rigged, used, armClearanceDegrees);
            }
        }
        return new GeneratedAsset(picked, cut, glb, rigged, animated, used, match);
    }

    /// <summary>
    /// The best of several drawn pictures: cut each out, and rank by how well it matches <paramref name="prompt"/> (CLIP, when <see cref="ScorePictures"/> and there is a choice) minus 1.5 times the
    /// share of the frame's border its subject touches (a cropped subject makes a truncated model); without scoring, the picture that fits its frame best, as before.
    /// </summary>
    private async Task<(AssetPicture Picture, int[] CutOut, float? Match)> ChooseBestAsync(string prompt, IReadOnlyList<AssetPicture> drawn)
    {
        var cuts = new List<(int[] Px, double Touch)>();
        foreach (var p in drawn) cuts.Add(await CutOutAsync(p));
        bool score = ScorePictures && drawn.Count > 1;
        float[]? sims = null;
        if (score)
        {
            _scorer ??= await Timed("load CLIP", () => PromptMatchScorer.CreateAsync(_acc, _source));
            sims = new float[drawn.Count];
            for (int i = 0; i < drawn.Count; i++) sims[i] = await _scorer.ScoreAsync(drawn[i], prompt);
        }
        int best = 0; double bestRank = double.MinValue;
        for (int i = 0; i < drawn.Count; i++)
        {
            double rank = (sims != null ? sims[i] : 0) - (sims != null ? 1.5 : 1.0) * cuts[i].Touch;
            if (rank > bestRank) { bestRank = rank; best = i; }
        }
        if (sims != null) StageDone?.Invoke($"picture match to the prompt: {string.Join(", ", sims.Select(x => x.ToString("F2")))}", 0);
        return (drawn[best], cuts[best].Px, sims?[best]);
    }

    // ───────────────────────────── helpers ─────────────────────────────

    /// <summary>A PNG file (RGBA, 8 bit) from RGBA bytes.</summary>
    public static byte[] EncodePng(byte[] rgba, int width, int height)
    {
        using var ms = new MemoryStream();
        ms.Write(new byte[] { 0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A });
        void Chunk(string type, byte[] data)
        {
            var len = BitConverter.GetBytes((uint)data.Length); if (BitConverter.IsLittleEndian) Array.Reverse(len);
            ms.Write(len);
            var td = new byte[4 + data.Length]; System.Text.Encoding.ASCII.GetBytes(type).CopyTo(td, 0); data.CopyTo(td, 4);
            ms.Write(td);
            var crc = BitConverter.GetBytes(Crc32(td)); if (BitConverter.IsLittleEndian) Array.Reverse(crc);
            ms.Write(crc);
        }
        var ihdr = new byte[13];
        BitConverter.GetBytes((uint)width).Reverse().ToArray().CopyTo(ihdr, 0); BitConverter.GetBytes((uint)height).Reverse().ToArray().CopyTo(ihdr, 4);
        ihdr[8] = 8; ihdr[9] = 6;                                     // 8 bit, colour type 6 = RGBA
        Chunk("IHDR", ihdr);
        using (var raw = new MemoryStream())
        {
            using (var z = new ZLibStream(raw, CompressionLevel.Fastest, true))
            {
                var row = new byte[1 + width * 4];
                for (int y = 0; y < height; y++) { row[0] = 0; Buffer.BlockCopy(rgba, y * width * 4, row, 1, width * 4); z.Write(row); }
            }
            Chunk("IDAT", raw.ToArray());
        }
        Chunk("IEND", Array.Empty<byte>());
        return ms.ToArray();
    }

    private static readonly uint[] CrcTable = Enumerable.Range(0, 256).Select(n => { uint c = (uint)n; for (int k = 0; k < 8; k++) c = (c & 1) != 0 ? 0xEDB88320u ^ (c >> 1) : c >> 1; return c; }).ToArray();
    private static uint Crc32(byte[] d) { uint c = 0xFFFFFFFFu; foreach (var b in d) c = CrcTable[(c ^ b) & 0xFF] ^ (c >> 8); return c ^ 0xFFFFFFFFu; }

    public void Dispose()
    {
        _flux?.Dispose(); _sd?.Dispose(); _rmbg?.Dispose(); _scorer?.Dispose();
        _flux = null; _sd = null; _rmbg = null; _scorer = null; _trellis = null;
    }
}
