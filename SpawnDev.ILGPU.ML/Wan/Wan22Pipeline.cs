using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnDev.ILGPU.ML.Trellis2;

namespace SpawnDev.ILGPU.ML.Wan;

/// <summary>The Wan 2.2 TI2V-5B files (the exports of tools/wan-harness/reference: DiT bf16 / 8-bit / 4-bit, VAE fp16, umT5 bf16, the tokenizer), as names the pipeline model source resolves: paths on disk for a local folder, file names within the repo for the hub.</summary>
public sealed class Wan22Paths
{
    /// <summary><c>wan22_ti2v_5b_dit_bf16.safetensors</c> (10 GB), <c>..._q8</c> (5.6 GB, lossless in practice) or <c>..._q4</c> (3.1 GB).</summary>
    public required string Dit { get; init; }
    /// <summary><c>wan22_vae_f16.safetensors</c> (1.4 GB).</summary>
    public required string Vae { get; init; }
    /// <summary><c>umt5_xxl_enc_bf16.safetensors</c> (11.4 GB, streamed layer by layer).</summary>
    public required string TextEncoder { get; init; }
    /// <summary>The umT5 <c>tokenizer.json</c>.</summary>
    public required string Tokenizer { get; init; }

    /// <summary>The file names the exporters write and a model repo holds (relative, forward slashes).</summary>
    public static Wan22Paths InRepo(string dit = "wan22_ti2v_5b_dit_q8.safetensors") => new()
    {
        Dit = dit, Vae = "wan22_vae_f16.safetensors", TextEncoder = "umt5_xxl_enc_bf16.safetensors", Tokenizer = "google/umt5-xxl/tokenizer.json",
    };

    /// <summary>The conventional layout of one folder on disk: the file names the exporters write.</summary>
    public static Wan22Paths InFolder(string folder, string dit = "wan22_ti2v_5b_dit_q8.safetensors") => new()
    {
        Dit = Path.Combine(folder, dit), Vae = Path.Combine(folder, "wan22_vae_f16.safetensors"), TextEncoder = Path.Combine(folder, "umt5_xxl_enc_bf16.safetensors"),
        Tokenizer = Path.Combine(folder, "google", "umt5-xxl", "tokenizer.json"),
    };
}

/// <summary>What to generate.</summary>
public sealed class WanRequest
{
    public required string Prompt { get; init; }
    /// <summary>What to steer away from; null = the reference's default negative prompt.</summary>
    public string? NegativePrompt { get; init; }
    /// <summary>Output size, multiples of 32 (the DiT works on 1x2x2 patches of a 16x downsampled latent). The model's own sizes are 1280x704 and 704x1280.</summary>
    public int Width { get; init; } = 1280;
    public int Height { get; init; } = 704;
    /// <summary>Frame count, 4n + 1 (121 = five seconds at 24 fps).</summary>
    public int Frames { get; init; } = 121;
    public int Steps { get; init; } = 50;
    public double GuidanceScale { get; init; } = 5.0;
    public double Shift { get; init; } = 5.0;
    public long Seed { get; init; } = 42;
    /// <summary>Starting noise [48, F/4 + 1, H/16, W/16] to use instead of one drawn from <see cref="Seed"/> (to reproduce another implementation's run exactly).</summary>
    public float[]? InitialNoise { get; init; }
    /// <summary>Log the time of every VAE decoder stage to the console (<c>[Wan] vae</c> lines).</summary>
    public bool Trace { get; init; }
    /// <summary>Optional first frame (image-to-video): RGB bytes, 3 per pixel, row-major; resized and centre-cropped to the output size.</summary>
    public byte[]? ImageRgb { get; init; }
    public int ImageWidth { get; init; }
    public int ImageHeight { get; init; }
}

/// <summary>The result: frames as floats in [-1, 1], channel-first [3, Frames, Height, Width] (the VAE's own layout).</summary>
public sealed class WanVideo
{
    public required float[] Data { get; init; }
    public required int Width { get; init; }
    public required int Height { get; init; }
    public required int Frames { get; init; }
    /// <summary>The denoised latent the video was decoded from, [48, F/4 + 1, H/16, W/16] (small; for tests and for re-decoding).</summary>
    public float[]? Latent { get; init; }

    /// <summary>One frame as RGBA bytes (alpha 255).</summary>
    public byte[] GetFrameRgba(int frame)
    {
        var o = new byte[Width * Height * 4];
        long plane = (long)Frames * Height * Width;
        for (int p = 0; p < Width * Height; p++)
        {
            for (int c = 0; c < 3; c++)
            {
                float v = Data[c * plane + (long)frame * Width * Height + p];
                o[p * 4 + c] = (byte)Math.Clamp((int)MathF.Round((v + 1f) * 127.5f), 0, 255);
            }
            o[p * 4 + 3] = 255;
        }
        return o;
    }
}

/// <summary>Progress of a generation: the stage, how far through it (0..1), and a short message.</summary>
public readonly record struct WanProgress(string Stage, double Fraction, string Message);

/// <summary>
/// Wan 2.2 TI2V-5B, text to video and image to video: umT5 text encoding (streamed), the optional VAE encoding of the first frame, the DiT with classifier-free guidance and the UniPC
/// flow sampler, then the VAE decode. The stages run one after another so only ONE large model is on the device at a time (the 10 GB bf16 DiT, 5.6 GB 8-bit one, or 3.1 GB 4-bit
/// one; the 1.4 GB VAE; the umT5's single layer being streamed), which is what lets a 12 GB card, and later a browser tab, run a 5B-parameter video model.
/// Every stage is gated against the reference implementation (see <c>Plans/wan22-ti2v-5b-2026-10-22.md</c>).
/// </summary>
public sealed class Wan22Pipeline
{
    private readonly Accelerator _acc;
    private readonly Wan22Paths _paths;
    private readonly IModelSource _source;
    private readonly string _repo;
    private UnigramTokenizer? _tok;

    /// <summary>The reference's default negative prompt (Chinese; it lists the usual video defects).</summary>
    public const string DefaultNegativePrompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走";

    /// <summary>Files from a model source (the hub with its OPFS cache or range streaming in the browser, a local folder on the desktop): <paramref name="paths"/> are names within <paramref name="repo"/>.</summary>
    public Wan22Pipeline(Accelerator accelerator, IModelSource source, string repo, Wan22Paths paths) { _acc = accelerator; _source = source; _repo = repo; _paths = paths; }

    /// <summary>Files on disk: <paramref name="paths"/> are file paths (<see cref="Wan22Paths.InFolder"/>).</summary>
    public Wan22Pipeline(Accelerator accelerator, Wan22Paths paths) : this(accelerator, new LocalFolderModelSource(""), "", paths) { }

    private async Task<GpuWeights> LoadWeightsAsync(string file, CancellationToken ct)
    {
        await using var stream = await _source.OpenAsync(_repo, file, ct);
        using var empty = System.Text.Json.JsonDocument.Parse("{}");
        return await GpuWeights.LoadAsync(_acc, stream, empty.RootElement, file, cancellationToken: ct);
    }

    private async Task<int[]> TokenizeAsync(string text, CancellationToken ct)
    {
        if (_tok == null) { using var ms = new MemoryStream(await _source.FetchBytesAsync(_repo, _paths.Tokenizer, ct)); _tok = UnigramTokenizer.FromTokenizerJson(ms); }
        var ids = _tok.Encode(UnigramTokenizer.CleanWhitespace(text));
        return ids.Length > WanDiT_TextLength ? ids[..WanDiT_TextLength] : ids;         // the reference truncates to 512 tokens
    }
    private const int WanDiT_TextLength = 512;

    /// <summary>
    /// Standard normal samples, seeded: xoshiro256** (seeded through splitmix64, so the same on every platform and runtime) and Box-Muller. Not torch's generator, so a seed here is not a seed there;
    /// <see cref="WanRequest.InitialNoise"/> takes another implementation's noise to reproduce its run.
    /// </summary>
    public static float[] Gaussian(long count, long seed)
    {
        ulong sm = unchecked((ulong)seed);
        ulong SplitMix() { sm += 0x9E3779B97F4A7C15UL; ulong z = sm; z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL; z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL; return z ^ (z >> 31); }
        ulong s0 = SplitMix(), s1 = SplitMix(), s2 = SplitMix(), s3 = SplitMix();
        static ulong Rotl(ulong x, int k) => (x << k) | (x >> (64 - k));
        double Next()                                                                  // uniform in (0, 1]
        {
            ulong r = Rotl(s1 * 5, 7) * 9, t = s1 << 17;
            s2 ^= s0; s3 ^= s1; s1 ^= s2; s0 ^= s3; s2 ^= t; s3 = Rotl(s3, 45);
            return ((r >> 11) + 1) * (1.0 / 9007199254740992.0);
        }
        var x = new float[count];
        for (long i = 0; i < count; i += 2)
        {
            double u1 = Next(), u2 = Next(), r = Math.Sqrt(-2.0 * Math.Log(u1));
            x[i] = (float)(r * Math.Cos(2 * Math.PI * u2));
            if (i + 1 < count) x[i + 1] = (float)(r * Math.Sin(2 * Math.PI * u2));
        }
        return x;
    }

    /// <summary>
    /// The picture as the VAE wants it: scaled until it covers the output (the larger of the two ratios), centre-cropped, bilinear, channel-first [3, 1, H, W] in [-1, 1].
    /// </summary>
    public static float[] PrepareImage(byte[] rgb, int w, int h, int outW, int outH)
    {
        double scale = Math.Max((double)outW / w, (double)outH / h);
        double sw = w * scale, sh = h * scale, ox = (sw - outW) / 2, oy = (sh - outH) / 2;
        var o = new float[3L * outH * outW];
        for (int y = 0; y < outH; y++)
            for (int x = 0; x < outW; x++)
            {
                double fx = Math.Clamp((x + ox + 0.5) / scale - 0.5, 0, w - 1), fy = Math.Clamp((y + oy + 0.5) / scale - 0.5, 0, h - 1);
                int x0 = (int)fx, y0 = (int)fy, x1 = Math.Min(x0 + 1, w - 1), y1 = Math.Min(y0 + 1, h - 1);
                double ax = fx - x0, ay = fy - y0;
                for (int c = 0; c < 3; c++)
                {
                    double v = rgb[(y0 * w + x0) * 3 + c] * (1 - ax) * (1 - ay) + rgb[(y0 * w + x1) * 3 + c] * ax * (1 - ay) + rgb[(y1 * w + x0) * 3 + c] * (1 - ax) * ay + rgb[(y1 * w + x1) * 3 + c] * ax * ay;
                    o[((long)c * outH + y) * outW + x] = (float)(v / 127.5 - 1.0);
                }
            }
        return o;
    }

    public async Task<WanVideo> GenerateAsync(WanRequest req, IProgress<WanProgress>? progress = null, CancellationToken ct = default)
    {
        if (req.Width % 32 != 0 || req.Height % 32 != 0) throw new ArgumentException("width and height must be multiples of 32");
        if ((req.Frames - 1) % 4 != 0 || req.Frames < 1) throw new ArgumentException("the frame count must be 4n + 1");
        int Fz = (req.Frames - 1) / 4 + 1, h = req.Height / 16, w = req.Width / 16, per = h * w;
        bool i2v = req.ImageRgb != null;
        void Report(string stage, double f, string msg) => progress?.Report(new WanProgress(stage, f, msg));

        // 1. text: the prompt and the negative prompt through umT5, one layer of its weights on the device at a time
        Report("text", 0, "encoding the prompt");
        float[] ctx, ctxNeg; int ctxRows, ctxNegRows;
        var ids = await TokenizeAsync(req.Prompt, ct); var nids = await TokenizeAsync(req.NegativePrompt ?? DefaultNegativePrompt, ct);
        ctxRows = ids.Length; ctxNegRows = nids.Length;
        using (var t5 = await UmT5Encoder.CreateAsync(_acc, _source, _repo, _paths.TextEncoder, ct))
        {
            var both = await t5.EncodeManyAsync(new[] { ids, nids }, ct);                  // one pass over the layers for both prompts
            ctx = both[0]; ctxNeg = both[1];
        }
        Report("text", 1, $"{ctxRows} + {ctxNegRows} tokens");

        // 2. the first frame (image to video): the VAE encodes it to the latent that replaces the first latent frame at every step
        float[]? z0 = null;
        if (i2v)
        {
            Report("image", 0, "encoding the picture");
            var img = PrepareImage(req.ImageRgb!, req.ImageWidth, req.ImageHeight, req.Width, req.Height);
            using var vw = await LoadWeightsAsync(_paths.Vae, ct);
            using var vae = new WanVae(_acc, vw);
            z0 = await vae.EncodeImageAsync(img, req.Height, req.Width);        // [48, 1, h, w]
            Report("image", 1, "done");
        }

        // 3. the denoising loop
        var latent = req.InitialNoise != null ? (float[])req.InitialNoise.Clone() : Gaussian(48L * Fz * per, req.Seed);
        if (latent.Length != 48L * Fz * per) throw new ArgumentException("InitialNoise has the wrong length for this size");
        void PutPicture(float[] l) { if (z0 != null) for (int c = 0; c < 48; c++) Array.Copy(z0, (long)c * per, l, (long)c * Fz * per, per); }
        PutPicture(latent);
        Report("denoise", 0, "loading the transformer");
        using (var dw = await LoadWeightsAsync(_paths.Dit, ct))
        using (var dit = new WanDiT(_acc, new DiTKernels(_acc), dw))
        {
            dit.SetGrid(Fz, h, w);
            dit.SetContext(0, ctx, ctxRows); dit.SetContext(1, ctxNeg, ctxNegRows);
            var sched = new FlowUniPCScheduler(); sched.SetTimesteps(req.Steps, req.Shift);
            using var lIn = _acc.Allocate1D<float>(latent.Length); using var lOut = _acc.Allocate1D<float>(latent.Length);
            for (int s = 0; s < req.Steps; s++)
            {
                ct.ThrowIfCancellationRequested();
                lIn.View.CopyFromCPU(latent);
                dit.Forward(lIn.View, sched.Timesteps[s], i2v, 0, lOut.View); var nc = await lOut.View.CopyToHostAsync();
                dit.Forward(lIn.View, sched.Timesteps[s], i2v, 1, lOut.View); var nu = await lOut.View.CopyToHostAsync();
                float g = (float)req.GuidanceScale;
                for (int i = 0; i < nc.Length; i++) nc[i] = nu[i] + g * (nc[i] - nu[i]);           // classifier-free guidance, in place
                latent = sched.Step(nc, latent);
                PutPicture(latent);
                Report("denoise", (s + 1.0) / req.Steps, $"step {s + 1} of {req.Steps}");
            }
        }

        // 4. decode
        Report("decode", 0, "decoding the video");
        float[] video;
        using (var vw = await LoadWeightsAsync(_paths.Vae, ct))
        using (var vae = new WanVae(_acc, vw))
            { vae.Trace = req.Trace; video = await vae.DecodeAsync(latent, Fz, h, w); }
        Report("decode", 1, "done");
        return new WanVideo { Data = video, Width = req.Width, Height = req.Height, Frames = 4 * Fz - 3, Latent = latent };
    }
}
