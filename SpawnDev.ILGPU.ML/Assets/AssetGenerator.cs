using System.IO.Compression;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Animation;
using SpawnDev.ILGPU.ML.Flux2;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.ILGPU.ML.SkinTokens;
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

/// <summary>What <see cref="AssetGenerator.GenerateAsync"/> made.</summary>
public sealed record GeneratedAsset(AssetPicture Picture, int[] CutOut, byte[] Glb, byte[]? RiggedGlb, byte[]? AnimatedGlb, IReadOnlyList<string> Moves);

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
        AssetPicture picked; int[] cut;
        if (choose != null)
        {
            picked = drawn[Math.Clamp(await choose(drawn), 0, drawn.Count - 1)];
            ReleaseImageModel();
            cut = (await CutOutAsync(picked)).Pixels;
        }
        else
        {
            ReleaseImageModel();
            picked = drawn[0]; cut = Array.Empty<int>(); double best = double.MaxValue;
            foreach (var p in drawn)
            {
                var (px, touch) = await CutOutAsync(p);
                if (touch < best) { best = touch; picked = p; cut = px; }
            }
        }
        var glb = await BuildModelAsync(cut, picked.Width, picked.Height, trellis ?? new Trellis2Options { Seed = picked.Seed });
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
        return new GeneratedAsset(picked, cut, glb, rigged, animated, used);
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
        _flux?.Dispose(); _sd?.Dispose(); _rmbg?.Dispose();
        _flux = null; _sd = null; _rmbg = null; _trellis = null;
    }
}
