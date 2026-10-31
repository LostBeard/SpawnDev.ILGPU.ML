using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnDev.ILGPU.ML.Trellis2;
using SpawnDev.ILGPU.ML.Wan;

namespace SpawnDev.ILGPU.ML.Ltx;

/// <summary>The LTX-Video 2B 0.9.8 distilled files (the exports of tools/ltx-harness/reference: DiT bf16, VAE fp16, T5-v1.1-XXL bf16, the tokenizer), as names the pipeline model source resolves.</summary>
public sealed class LtxPaths
{
    /// <summary><c>ltx_dit_bf16.safetensors</c> (3.9 GB).</summary>
    public required string Dit { get; init; }
    /// <summary><c>ltx_vae_f16.safetensors</c> (1.1 GB, the decoder).</summary>
    public required string Vae { get; init; }
    /// <summary><c>ltx_vae_encoder_f16.safetensors</c> (1.4 GB, the encoder: loaded only for image / video conditioning).</summary>
    public string? VaeEncoder { get; init; }
    /// <summary><c>ltx_upsampler_f16.safetensors</c> (253 MB, the x2 spatial latent upscaler: loaded only for <see cref="LtxRequest.Upscale"/>).</summary>
    public string? Upsampler { get; init; }
    /// <summary><c>t5_xxl_enc_bf16.safetensors</c> (9.5 GB, streamed layer by layer).</summary>
    public required string TextEncoder { get; init; }
    /// <summary>The <c>tokenizer.json</c> of google/t5-v1_1-xxl.</summary>
    public required string Tokenizer { get; init; }

    /// <summary>The DiT file for a precision: "bf16" (3.6 GB) or "q8" (2.0 GB).</summary>
    public static string DitFile(string precision) => $"ltx_dit_{precision}.safetensors";
    /// <summary>The T5 file for a precision: "bf16" (8.9 GB) or "q8" (5.0 GB).</summary>
    public static string TextEncoderFile(string precision) => $"t5_xxl_enc_{precision}.safetensors";

    /// <summary>
    /// The file names a model repo holds (relative, forward slashes). <see cref="LtxPipeline.DefaultRepo"/> on models.spawndev.com serves the bf16 and 8-bit DiT, the 8-bit
    /// T5 (the 9 GB bf16 one is not hosted), the VAE decoder and encoder, the latent upscaler and the tokenizer.
    /// </summary>
    public static LtxPaths InRepo(string dit = "ltx_dit_bf16.safetensors", string textEncoder = "t5_xxl_enc_q8.safetensors") => new()
    {
        Dit = dit, Vae = "ltx_vae_f16.safetensors", VaeEncoder = "ltx_vae_encoder_f16.safetensors", Upsampler = "ltx_upsampler_f16.safetensors", TextEncoder = textEncoder,
        Tokenizer = "google/t5-v1_1-xxl/tokenizer.json",
    };

    /// <summary>The conventional layout of one folder on disk: the file names the exporters write.</summary>
    public static LtxPaths InFolder(string folder, string dit = "ltx_dit_bf16.safetensors", string textEncoder = "t5_xxl_enc_bf16.safetensors") => new()
    {
        Dit = Path.Combine(folder, dit), Vae = Path.Combine(folder, "ltx_vae_f16.safetensors"), VaeEncoder = Path.Combine(folder, "ltx_vae_encoder_f16.safetensors"),
        Upsampler = Path.Combine(folder, "ltx_upsampler_f16.safetensors"), TextEncoder = Path.Combine(folder, textEncoder), Tokenizer = Path.Combine(folder, "google", "t5-v1_1-xxl", "tokenizer.json"),
    };
}

/// <summary>
/// A picture or a clip the video is conditioned on (LTXConditionPipeline's <c>image</c> / <c>video</c> with <c>frame_index</c> and <c>strength</c>): at frame 0 it replaces the first latent
/// frames (image-to-video, video continuation); at a later frame it is added as extra tokens placed at that time (keyframes; a clip longer than 2 latent frames also replaces the latent
/// frames after its first two). Give either RGB bytes (resized to cover the output, centre-cropped) or ready pixels at the output size.
/// </summary>
public sealed class LtxCondition
{
    /// <summary>RGB bytes, 3 per pixel, row-major: <see cref="Frames"/> pictures of <see cref="Width"/> x <see cref="Height"/> back to back.</summary>
    public byte[]? Rgb { get; init; }
    public int Width { get; init; }
    public int Height { get; init; }
    /// <summary>Instead of <see cref="Rgb"/>: [3, Frames, H, W] channel-first in [-1, 1] at the request's output size.</summary>
    public float[]? Pixels { get; init; }
    /// <summary>1 for a picture; a clip is 8n + 1 frames (longer clips are cut to fit the video after <see cref="FrameIndex"/>, to 8n + 1).</summary>
    public int Frames { get; init; } = 1;
    /// <summary>The pixel frame the condition starts at (0 = the first frame).</summary>
    public int FrameIndex { get; init; }
    /// <summary>1 = hard conditioning (the frames stay the condition, the model only sees them); below 1 the condition is blended with the starting noise and denoised from (1 - strength).</summary>
    public float Strength { get; init; } = 1f;

    /// <summary>One picture at <paramref name="frameIndex"/> (0: image-to-video).</summary>
    public static LtxCondition Image(byte[] rgb, int width, int height, int frameIndex = 0, float strength = 1f)
        => new() { Rgb = rgb, Width = width, Height = height, FrameIndex = frameIndex, Strength = strength };
}

/// <summary>
/// One denoising step's x0 estimate (<c>x - sigma * v</c>, the clean latent the sampler is heading for), for a live preview: [128, <see cref="Frames"/>, <see cref="Height"/>,
/// <see cref="Width"/>] channel-first, model space (<see cref="LatentPreview.Ltx"/> turns it into colours). <see cref="Pass"/> is "first pass" / "second pass" in the two-pass recipe.
/// </summary>
public sealed record LtxPreview(string Pass, int Step, int Steps, float[] X0, int Frames, int Height, int Width);

/// <summary>What to generate.</summary>
public sealed class LtxRequest
{
    public required string Prompt { get; init; }
    /// <summary>Output size, multiples of 32 (the VAE compresses 32x spatially, the DiT works on every latent position). 704x480 is the model's native size; the 0.9.8 recipe runs a first pass at 2/3 of the final size.</summary>
    public int Width { get; init; } = 704;
    public int Height { get; init; } = 480;
    /// <summary>Frame count, 8n + 1 (121 = five seconds at 24 fps).</summary>
    public int Frames { get; init; } = 121;
    /// <summary>The sigmas of the denoising steps; null = the distilled recipe's first pass (7 steps, guidance 1, one DiT forward per step). A terminal 0 is appended.</summary>
    public float[]? Sigmas { get; init; }
    /// <summary>Frames per second the clip is meant to play at: it sets the temporal scale of the position embedding (8 / frame rate). The diffusers reference uses 25; the model was trained on 24-30.</summary>
    public float FrameRate { get; init; } = 24f;
    public long Seed { get; init; } = 42;
    /// <summary>The decode timestep and noise scale of the 0.9.8 recipe (the VAE is timestep-conditioned and decodes a slightly noised latent).</summary>
    public float DecodeTimestep { get; init; } = 0.05f;
    public float DecodeNoiseScale { get; init; } = 0.025f;
    /// <summary>Starting noise [128, (F - 1) / 8 + 1, H / 32, W / 32] to use instead of one drawn from <see cref="Seed"/> (to reproduce another implementation's run exactly).</summary>
    public float[]? InitialNoise { get; init; }
    /// <summary>
    /// A model-space latent to start from (the reference pipelines' <c>latents</c> with a schedule that starts below 1): the start is <c>sigma0 * noise + (1 - sigma0) * Latents</c>, sigma0
    /// the first of <see cref="Sigmas"/>, before the conditions are applied. The recipe's second pass starts from the upsampled first pass this way.
    /// </summary>
    public float[]? Latents { get; init; }

    /// <summary>
    /// The distilled model's multi-scale recipe (configs/ltxv-2b-0.9.8-distilled.yaml): a first pass at 2/3 of <see cref="Width"/> x <see cref="Height"/> (times 0.6666666, rounded
    /// down to multiples of 32, as the official code), the x2 latent upscaler with AdaIN, and a second pass at twice the first-pass size with <see cref="SecondPassSigmas"/>, then the
    /// decode. The video comes out at the second-pass size (896x576 for 704x480: 480 x 0.6666666 floors to 319, then 288; the official script then resizes the frames to the request). Conditions must be given as
    /// <see cref="LtxCondition.Rgb"/> (each pass resizes them); the noise overrides are not used.
    /// </summary>
    public bool Upscale { get; init; }
    /// <summary>The second pass's sigmas; null = the recipe's (0.9094, 0.7250, 0.4219).</summary>
    public float[]? SecondPassSigmas { get; init; }
    /// <summary>The upscaler's AdaIN blend (each channel's statistics moved to the first pass's): 1 in the recipe.</summary>
    public float AdainFactor { get; init; } = 1f;
    /// <summary>
    /// Return the frames as RGBA bytes (<see cref="WanVideo.FramesRgba"/>, converted on the device, read a frame at a time) instead of the float data: a quarter of the memory and
    /// no host array the size of the whole clip (in the browser, a few clips of floats exhaust the .NET heap).
    /// </summary>
    public bool RgbaFrames { get; init; }

    /// <summary>
    /// Pictures / clips to condition on (empty or null: text to video). With conditions the pipeline follows LTXConditionPipeline: per-token timesteps, the causal time positions
    /// (pixel frame <c>max(8 f - 7, 0)</c>) and the image-conditioning noise; without, LTXPipeline (the text-to-video reference, unchanged).
    /// </summary>
    public IReadOnlyList<LtxCondition>? Conditions { get; init; }
    /// <summary>The noise added to hard-conditioned tokens before every step, times (t / 1000)^2 (LTXConditionPipeline's <c>image_cond_noise_scale</c>; it helps the motion continue the picture).</summary>
    public float ImageCondNoiseScale { get; init; } = 0.15f;

    /// <summary>The prompt's T5 output [rows, 4096] (the real tokens only) to use instead of encoding <see cref="Prompt"/> (the reference's <c>prompt_embeds</c>): no text encoder is loaded.</summary>
    public float[]? PromptEmbedding { get; init; }

    /// <summary>
    /// The noise of the conditioned run, to reproduce another implementation's draws exactly (each null entry is drawn from <see cref="Seed"/>): the VAE posterior sample of each condition
    /// [128, latent frames, h, w], the noise a condition at a later frame is blended with [128, its latent frames (at most 2), h, w], and the image-conditioning noise of each step [tokens, 128].
    /// </summary>
    public float[]?[]? PosteriorNoise { get; init; }
    public float[]?[]? ConditionNoise { get; init; }
    public float[]?[]? StepNoise { get; init; }
}

/// <summary>
/// LTX-Video 2B 0.9.8 distilled, text to video and image / video conditioned: T5-v1.1-XXL text encoding (streamed layer by layer), the VAE encoding of the conditions, the DiT with the
/// distilled flow-matching Euler sampler (one forward per step, the latent stays on the device), then the timestep-conditioned VAE decode. The stages run one after another so only ONE
/// large model is on the device at a time. Returns the same container as the Wan pipeline (frames in [-1, 1], channel-first). Every stage is gated against the diffusers reference
/// (LTXPipeline for text to video, LTXConditionPipeline with conditions; see <c>Plans/ltx-video-2b-2026-10-22.md</c>).
/// </summary>
public sealed class LtxPipeline : IDisposable
{
    private readonly Accelerator _acc;
    private readonly LtxPaths _paths;
    private readonly IModelSource _source;
    private readonly string _repo;
    private UnigramTokenizer? _tok;

    /// <summary>
    /// Called after every denoising step with the sampler's x0 estimate (a live preview: the latent is read back once per step, which costs a few milliseconds). Null = no preview.
    /// </summary>
    public Action<LtxPreview>? Preview { get; set; }
    private string _pass = "";

    /// <summary>Test hook of the conditioned sampler: (step, the DiT's token-major input [tokens, 128], its prediction) after each step's forward; valid only during the call.</summary>
    public Action<int, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>? StepTap { get; set; }

    /// <summary>The model repo the SpawnDev models host serves the exports from (https://models.spawndev.com/&lt;repo&gt;/resolve/main/&lt;file&gt;).</summary>
    public const string DefaultRepo = "LostBeard/LTX-Video-2B-0.9.8-distilled-ILGPU";
    private const int TextLength = 128;
    private const int TemporalRatio = 8, SpatialRatio = 32;

    public LtxPipeline(Accelerator accelerator, IModelSource source, string repo, LtxPaths paths) { _acc = accelerator; _source = source; _repo = repo; _paths = paths; }

    /// <summary>Files on disk: <paramref name="paths"/> are file paths (<see cref="LtxPaths.InFolder"/>).</summary>
    public LtxPipeline(Accelerator accelerator, LtxPaths paths) : this(accelerator, new LocalFolderModelSource(""), "", paths) { }

    private async Task<GpuWeights> LoadWeightsAsync(string file, CancellationToken ct)
    {
        await using var stream = await _source.OpenAsync(_repo, file, ct);
        using var empty = System.Text.Json.JsonDocument.Parse("{}");
        return await GpuWeights.LoadAsync(_acc, stream, empty.RootElement, file, cancellationToken: ct);
    }

    /// <summary>
    /// Keep the DiT, the VAE decoder and encoder and the latent upscaler on the device between generations instead of loading them for every clip (the T5 is streamed as
    /// always). In the browser (RTX 4070) a 704x480 x97 clip with a cached prompt went from 35 s to 30 s; it costs their size in device memory (about 5 GB with the 8-bit DiT)
    /// on top of every stage's own peak, so leave it off on an 8 GB GPU. <see cref="Dispose"/> (or setting it back to false) frees them.
    /// </summary>
    public bool KeepLoaded { get => _keep; set { _keep = value; if (!value) ReleaseKept(); } }
    private bool _keep;
    private readonly Dictionary<string, GpuWeights> _kept = new();

    /// <summary>The weights of <paramref name="file"/> for one stage: kept ones are shared (the lease does not free them), others are freed with the lease.</summary>
    private async Task<WeightsLease> AcquireAsync(string file, CancellationToken ct)
    {
        if (_keep && _kept.TryGetValue(file, out var w)) return new WeightsLease(w, owned: false);
        var loaded = await LoadWeightsAsync(file, ct);
        if (_keep) { _kept[file] = loaded; return new WeightsLease(loaded, owned: false); }
        return new WeightsLease(loaded, owned: true);
    }

    private sealed class WeightsLease(GpuWeights w, bool owned) : IDisposable
    {
        public GpuWeights W { get; } = w;
        public void Dispose() { if (owned) W.Dispose(); }
    }

    private void ReleaseKept()
    {
        foreach (var w in _kept.Values) w.Dispose();
        _kept.Clear();
    }

    /// <summary>Frees the weights <see cref="KeepLoaded"/> kept on the device (the accelerator is the application's and is not touched).</summary>
    public void Dispose() => ReleaseKept();

    // the last few prompts' T5 outputs: the encoder streams 5 GB of weights per prompt (in the browser about half of a short clip's time), so a new seed, another mode or an
    // extension with the same prompt skips it (an embedding is at most 128 x 4096 floats)
    private readonly List<(string Prompt, float[] Embedding)> _prompts = new();
    private const int PromptCacheSize = 8;

    /// <summary>The prompt's T5 output [tokens, 4096] (cached by prompt text).</summary>
    private async Task<float[]> EncodePromptAsync(string prompt, Action<string, double, string> report, CancellationToken ct)
    {
        int hit = _prompts.FindIndex(p => p.Prompt == prompt);
        if (hit >= 0)
        {
            var e = _prompts[hit];
            _prompts.RemoveAt(hit); _prompts.Add(e);
            report("text", 1, "the prompt's encoding is cached");
            return e.Embedding;
        }
        report("text", 0, "encoding the prompt");
        var ids = await TokenizeAsync(prompt, ct);
        float[] ctx;
        using (var t5 = await UmT5Encoder.CreateAsync(_acc, _source, _repo, _paths.TextEncoder, ct))
            ctx = await t5.EncodeAsync(ids, ct);
        _prompts.Add((prompt, ctx));
        if (_prompts.Count > PromptCacheSize) _prompts.RemoveAt(0);
        report("text", 1, $"{ids.Length} tokens");
        return ctx;
    }

    /// <summary>The T5 token ids of a prompt as the reference makes them: truncated to 128 tokens including the end-of-sequence token (the tokenizer cuts before adding it).</summary>
    private async Task<int[]> TokenizeAsync(string text, CancellationToken ct)
    {
        if (_tok == null) { using var ms = new MemoryStream(await _source.FetchBytesAsync(_repo, _paths.Tokenizer, ct)); _tok = UnigramTokenizer.FromTokenizerJson(ms); }
        var ids = _tok.Encode(UnigramTokenizer.CleanWhitespace(text));
        if (ids.Length <= TextLength) return ids;
        var cut = new int[TextLength];
        Array.Copy(ids, cut, TextLength - 1);
        cut[TextLength - 1] = ids[^1];
        return cut;
    }

    /// <summary>Standard normal samples, seeded (<see cref="SeededNoise"/>). Not torch's generator, so a seed here is not a seed there; <see cref="LtxRequest.InitialNoise"/> takes another implementation's noise.</summary>
    public static float[] Gaussian(long count, long seed) => SeededNoise.Gaussian(count, seed);

    /// <summary>The recipe's first-pass size for a requested size: <c>int(side * 0.6666666)</c> rounded down to a multiple of 32 (the official inference script's arithmetic).</summary>
    public static (int Width, int Height) FirstPassSize(int width, int height)
    {
        static int Side(int v) { int x = (int)(v * 0.6666666); return x - x % SpatialRatio; }
        return (Side(width), Side(height));
    }

    /// <summary>The x2 spatial latent upscaler on a model-space latent [128, Fz, h, w] -> [128, Fz, 2h, 2w] (<see cref="LtxLatentUpsampler"/>; the weights are loaded for the call).</summary>
    public async Task<float[]> UpsampleLatentAsync(float[] latent, int Fz, int h, int w, float adainFactor = 1f, CancellationToken ct = default)
    {
        if (_paths.Upsampler == null) throw new InvalidOperationException("the latent upscaler needs LtxPaths.Upsampler");
        using var uw = await AcquireAsync(_paths.Upsampler, ct);
        using var up = new LtxLatentUpsampler(_acc, uw.W);
        return await up.UpsampleAsync(latent, Fz, h, w, adainFactor);
    }

    /// <summary>The whole pipeline: <see cref="DenoiseAsync"/> (with <see cref="LtxRequest.Upscale"/>: both passes and the upscaler between them), then the VAE decode.</summary>
    public async Task<WanVideo> GenerateAsync(LtxRequest req, IProgress<WanProgress>? progress = null, CancellationToken ct = default)
    {
        if (req.Upscale) return await GenerateMultiScaleAsync(req, progress, ct);
        var latent = await DenoiseAsync(req, progress, ct);
        int Fz = (req.Frames - 1) / TemporalRatio + 1, h = req.Height / SpatialRatio, w = req.Width / SpatialRatio;
        return await DecodeAsync(req, latent, Fz, h, w, progress, ct);
    }

    private async Task<WanVideo> DecodeAsync(LtxRequest req, float[] latent, int Fz, int h, int w, IProgress<WanProgress>? progress, CancellationToken ct)
    {
        progress?.Report(new WanProgress("decode", 0, "decoding the video"));
        float[] video = Array.Empty<float>();
        byte[][]? rgba = null;
        var decodeNoise = req.DecodeNoiseScale != 0f ? Gaussian(latent.Length, req.Seed + 104729L) : null;
        using (var vw = await AcquireAsync(_paths.Vae, ct))
        using (var vae = new LtxVae(_acc, vw.W))
        {
            if (req.RgbaFrames) rgba = await vae.DecodeRgbaFramesAsync(latent, Fz, h, w, req.DecodeTimestep, req.DecodeNoiseScale, decodeNoise);
            else video = await vae.DecodeAsync(latent, Fz, h, w, req.DecodeTimestep, req.DecodeNoiseScale, decodeNoise);
        }
        progress?.Report(new WanProgress("decode", 1, "done"));
        return new WanVideo { Data = video, FramesRgba = rgba, Width = SpatialRatio * w, Height = SpatialRatio * h, Frames = TemporalRatio * (Fz - 1) + 1, Latent = latent };
    }

    private static LtxRequest Pass(LtxRequest r, int width, int height, float[]? sigmas, float[]? latents, long seed, float[] embedding) => new()
    {
        Prompt = r.Prompt, Width = width, Height = height, Frames = r.Frames, Sigmas = sigmas, FrameRate = r.FrameRate, Seed = seed, DecodeTimestep = r.DecodeTimestep,
        DecodeNoiseScale = r.DecodeNoiseScale, Conditions = r.Conditions, ImageCondNoiseScale = r.ImageCondNoiseScale, PromptEmbedding = embedding, Latents = latents, RgbaFrames = r.RgbaFrames,
    };

    private async Task<WanVideo> GenerateMultiScaleAsync(LtxRequest req, IProgress<WanProgress>? progress, CancellationToken ct)
    {
        if (req.InitialNoise != null || req.Latents != null || req.PosteriorNoise != null || req.ConditionNoise != null || req.StepNoise != null)
            throw new ArgumentException("Upscale runs two passes with their own noise: InitialNoise, Latents and the noise overrides are not used with it");
        if (req.Conditions?.Any(c => c.Rgb == null) == true) throw new ArgumentException("with Upscale every condition must be given as Rgb (each pass resizes it to its own size)");
        var (w1, h1) = FirstPassSize(req.Width, req.Height);
        if (w1 < SpatialRatio || h1 < SpatialRatio) throw new ArgumentException($"{req.Width}x{req.Height} is too small for the two-pass recipe (its first pass would be {w1}x{h1})");
        int Fz = (req.Frames - 1) / TemporalRatio + 1;
        void Report(string stage, double f, string msg) => progress?.Report(new WanProgress(stage, f, msg));

        // the prompt once for both passes
        var embedding = req.PromptEmbedding ?? await EncodePromptAsync(req.Prompt, Report, ct);
        var inner = progress == null ? null : new Progress<WanProgress>(p => progress.Report(p.Stage == "denoise" ? p with { Message = "first pass: " + p.Message } : p));
        _pass = "first pass";
        var first = await DenoiseAsync(Pass(req, w1, h1, req.Sigmas, null, req.Seed, embedding), inner, ct);
        Report("upscale", 0, "x2 latent upscaler");
        var up = await UpsampleLatentAsync(first, Fz, h1 / SpatialRatio, w1 / SpatialRatio, req.AdainFactor, ct);
        Report("upscale", 1, "done");
        var inner2 = progress == null ? null : new Progress<WanProgress>(p => progress.Report(p.Stage == "denoise" ? p with { Message = "second pass: " + p.Message } : p));
        _pass = "second pass";
        var second = await DenoiseAsync(Pass(req, 2 * w1, 2 * h1, req.SecondPassSigmas ?? LtxSampler.DistilledSecondPass, up, req.Seed + 7919L, embedding), inner2, ct);
        _pass = "";
        return await DecodeAsync(req, second, Fz, 2 * h1 / SpatialRatio, 2 * w1 / SpatialRatio, progress, ct);
    }

    /// <summary>
    /// Text (and conditions) to the denoised latent [128, (F - 1) / 8 + 1, H / 32, W / 32] in model space, before the decode (the reference's <c>output_type="latent"</c>).
    /// </summary>
    public async Task<float[]> DenoiseAsync(LtxRequest req, IProgress<WanProgress>? progress = null, CancellationToken ct = default)
    {
        if (req.Width % SpatialRatio != 0 || req.Height % SpatialRatio != 0) throw new ArgumentException("width and height must be multiples of 32");
        if ((req.Frames - 1) % TemporalRatio != 0 || req.Frames < 1) throw new ArgumentException("the frame count must be 8n + 1");
        int Fz = (req.Frames - 1) / TemporalRatio + 1, h = req.Height / SpatialRatio, w = req.Width / SpatialRatio, N = Fz * h * w;
        void Report(string stage, double f, string msg) => progress?.Report(new WanProgress(stage, f, msg));
        var sigmas = LtxSampler.WithTerminal(req.Sigmas ?? LtxSampler.DistilledFirstPass);

        // 1. text: the prompt through T5, one layer of its weights on the device at a time
        float[] ctx; int rows;
        if (req.PromptEmbedding != null)
        {
            if (req.PromptEmbedding.Length % 4096 != 0 || req.PromptEmbedding.Length == 0) throw new ArgumentException("the prompt embedding must be [rows, 4096]");
            ctx = req.PromptEmbedding; rows = ctx.Length / 4096;
        }
        else
        {
            ctx = await EncodePromptAsync(req.Prompt, Report, ct);
            rows = ctx.Length / 4096;
        }

        var noise = req.InitialNoise != null ? (float[])req.InitialNoise.Clone() : Gaussian(128L * N, req.Seed);
        if (noise.Length != 128L * N) throw new ArgumentException("InitialNoise has the wrong length for this size");
        if (req.Latents != null)
        {
            if (req.Latents.Length != noise.Length) throw new ArgumentException("Latents must be [128, (F - 1) / 8 + 1, H / 32, W / 32]");
            float s0 = sigmas[0];
            for (long i = 0; i < noise.Length; i++) noise[i] = s0 * noise[i] + (1f - s0) * req.Latents[i];        // prepare_latents: sigma * noise + (1 - sigma) * latents
        }
        if (req.Conditions is { Count: > 0 }) return await DenoiseConditionedAsync(req, ctx, rows, noise, sigmas, Fz, h, w, Report, ct);

        // 2. the denoising loop: the latent [128, Fz, h, w] stays on the device, the Euler step runs there too
        int steps = sigmas.Length - 1;
        Report("denoise", 0, "loading the transformer");
        var lk = new LtxKernels(_acc);
        using var dw = await AcquireAsync(_paths.Dit, ct);
        using var dit = new LtxDiT(_acc, new DiTKernels(_acc), dw.W);
        dit.SetGrid(Fz, h, w, (float)TemporalRatio / req.FrameRate, SpatialRatio, SpatialRatio);
        dit.SetContext(0, ctx, rows);
        using var lat = _acc.Allocate1D(noise); using var v = _acc.Allocate1D<float>(noise.Length);
        for (int s = 0; s < steps; s++)
        {
            ct.ThrowIfCancellationRequested();
            dit.Forward(lat.View, LtxSampler.Timestep(sigmas[s]), 0, v.View);
            if (Preview != null)
            {
                var xs = await lat.View.CopyToHostAsync(); var vs = await v.View.CopyToHostAsync();
                float sg = sigmas[s];
                for (long i = 0; i < xs.Length; i++) xs[i] -= sg * vs[i];
                Preview(new LtxPreview(_pass, s + 1, steps, xs, Fz, h, w));
            }
            lk.Axpy(lat.View, v.View, sigmas[s + 1] - sigmas[s], noise.Length);
            await _acc.SynchronizeAsync();                                               // one step at a time: the browser queue must not run ahead
            Report("denoise", (s + 1.0) / steps, $"step {s + 1} of {steps}");
        }
        return await lat.View.CopyToHostAsync();                                         // the small latent crosses once, for the decode
    }

    /// <summary>The token positions of a latent block of <paramref name="frames"/> x h x w starting at pixel frame <paramref name="frameIndex"/>: [tokens, 3] (time in seconds, pixel row, pixel column).</summary>
    private static float[] Coords(int frames, int h, int w, int frameIndex, float frameRate)
    {
        var c = new float[frames * h * w * 3];
        float inv = (float)(1.0 / frameRate);                                            // video_coords[:, 0] * (1.0 / frame_rate), in float32
        int n = 0;
        for (int f = 0; f < frames; f++)
        {
            float t = Math.Max(TemporalRatio * f + 1 - TemporalRatio, 0) + frameIndex;    // _scale_video_ids: the causal fix, then the frame index
            for (int i = 0; i < h; i++)
                for (int j = 0; j < w; j++, n++) { c[3 * n] = t * inv; c[3 * n + 1] = SpatialRatio * i; c[3 * n + 2] = SpatialRatio * j; }
        }
        return c;
    }

    /// <summary>[128, frames * hw] channel-major -> [frames * hw, 128] packed tokens.</summary>
    private static float[] Pack(float[] x, int tokens)
    {
        var o = new float[x.Length];
        for (int c = 0; c < 128; c++) for (int n = 0; n < tokens; n++) o[(long)n * 128 + c] = x[(long)c * tokens + n];
        return o;
    }

    /// <summary>The condition's pixels [3, frames, H, W] in [-1, 1] at the output size, and the frame count after the reference's trim.</summary>
    private static (float[] Pixels, int Frames) ConditionPixels(LtxCondition c, LtxRequest req)
    {
        int frames = c.Frames;
        if ((frames - 1) % TemporalRatio != 0) throw new ArgumentException($"a condition clip must have 8n + 1 frames, not {frames}");
        int keep = Math.Min(frames, req.Frames - c.FrameIndex);                          // trim_conditioning_sequence
        if (keep < 1) throw new ArgumentException($"the condition at frame {c.FrameIndex} starts after the video ends");
        keep = (keep - 1) / TemporalRatio * TemporalRatio + 1;
        int H = req.Height, W = req.Width;
        long plane = (long)H * W;
        var o = new float[3 * keep * plane];
        for (int f = 0; f < keep; f++)
        {
            float[] frame;
            if (c.Pixels != null)
            {
                if (c.Pixels.Length != 3L * frames * plane) throw new ArgumentException("condition Pixels must be [3, Frames, Height, Width] at the output size");
                frame = new float[3 * plane];
                for (int ch = 0; ch < 3; ch++) Array.Copy(c.Pixels, (ch * (long)frames + f) * plane, frame, ch * plane, plane);
            }
            else
            {
                if (c.Rgb == null || c.Width < 1 || c.Height < 1 || c.Rgb.Length != 3L * frames * c.Width * c.Height) throw new ArgumentException("a condition needs Rgb (Frames pictures of Width x Height) or Pixels");
                frame = Wan22Pipeline.PrepareImage(c.Rgb.AsSpan(f * 3 * c.Width * c.Height, 3 * c.Width * c.Height).ToArray(), c.Width, c.Height, W, H);
            }
            for (int ch = 0; ch < 3; ch++) Array.Copy(frame, ch * plane, o, (ch * (long)keep + f) * plane, plane);
        }
        return (o, keep);
    }

    /// <summary>
    /// LTXConditionPipeline's sampler. Tokens are packed [N, 128] in (frame, row, column) order: the extra tokens of the conditions at later frames first, then the video's. Each token has a
    /// conditioning strength m (per latent frame or per condition); a step at timestep t gives it the timestep <c>min(t, (1 - m) * 1000)</c>, updates it only once
    /// <c>t / 1000 - 1e-6 &lt; 1 - m</c>, with the flow-matching Euler step from its own sigma to the next lower sigma of the schedule; before every step the hard-conditioned tokens
    /// (m = 1) are reset to their clean latent plus the image-conditioning noise times (t / 1000)^2.
    /// </summary>
    private async Task<float[]> DenoiseConditionedAsync(LtxRequest req, float[] ctx, int rows, float[] noise, float[] sigmas, int Fz, int h, int w, Action<string, double, string> Report, CancellationToken ct)
    {
        int hw = h * w, N = Fz * hw, steps = sigmas.Length - 1;
        var main = Pack(noise, N);
        var frameMask = new float[Fz];
        var extras = new List<(float[] Tokens, float[] Coords, float Strength)>();

        // 2. the conditions through the VAE encoder (normalised latents), blended into the starting latent or kept as extra tokens
        Report("conditions", 0, "encoding the conditions");
        if (_paths.VaeEncoder == null) throw new InvalidOperationException("conditioning needs the VAE encoder (LtxPaths.VaeEncoder)");
        using (var ew = await AcquireAsync(_paths.VaeEncoder, ct))
        using (var enc = new LtxVaeEncoder(_acc, ew.W))
        {
            int extraIndex = 0;
            for (int i = 0; i < req.Conditions!.Count; i++)
            {
                var c = req.Conditions[i];
                float s = c.Strength;
                var (pixels, frames) = ConditionPixels(c, req);
                int nc = (frames - 1) / TemporalRatio + 1;
                var post = req.PosteriorNoise?.ElementAtOrDefault(i) ?? Gaussian(128L * nc * hw, req.Seed + 1000 + i);
                var cl = Pack(await enc.EncodeAsync(pixels, frames, req.Height, req.Width, post), nc * hw);     // [nc * hw, 128]
                if (c.FrameIndex == 0)
                {
                    int f = Math.Min(nc, Fz);
                    for (long k = 0; k < (long)f * hw * 128; k++) main[k] += s * (cl[k] - main[k]);                // torch.lerp(noise, condition, strength)
                    for (int j = 0; j < f; j++) frameMask[j] = s;
                }
                else
                {
                    if (frames > 1)
                    {
                        const int prefix = 2;                                                     // num_prefix_latent_frames
                        if (nc < prefix) throw new ArgumentException($"a condition clip at a later frame needs at least {prefix} latent frames ({8 * prefix - 7} frames)");
                        if (nc > prefix)
                        {
                            int start = c.FrameIndex / TemporalRatio + prefix, end = Math.Min(start + nc - prefix, Fz);
                            for (int j = start; j < end; j++)
                            {
                                for (int k = 0; k < hw * 128; k++) { long d = ((long)j * hw) * 128 + k; main[d] += s * (cl[((long)(j - start + prefix) * hw) * 128 + k] - main[d]); }
                                frameMask[j] = s;
                            }
                            cl = cl.AsSpan(0, prefix * hw * 128).ToArray();
                            nc = prefix;
                        }
                    }
                    var n2 = req.ConditionNoise?.ElementAtOrDefault(extraIndex) ?? Gaussian(128L * nc * hw, req.Seed + 2000 + i);
                    var n2p = Pack(n2, nc * hw);
                    for (long k = 0; k < cl.Length; k++) cl[k] = n2p[k] + s * (cl[k] - n2p[k]);
                    extras.Add((cl, Coords(nc, h, w, c.FrameIndex, req.FrameRate), s));
                    extraIndex++;
                }
            }
        }
        Report("conditions", 1, $"{req.Conditions.Count} condition(s)");

        // the token sequence: extras, then the video; its positions and per-token strengths
        int extraTokens = extras.Sum(e => e.Tokens.Length / 128), Nt = extraTokens + N;
        var tokens = new float[(long)Nt * 128];
        var coords = new float[Nt * 3];
        var mask = new float[Nt];
        int at = 0;
        foreach (var e in extras)
        {
            int n = e.Tokens.Length / 128;
            Array.Copy(e.Tokens, 0, tokens, (long)at * 128, e.Tokens.Length);
            Array.Copy(e.Coords, 0, coords, at * 3, n * 3);
            for (int k = 0; k < n; k++) mask[at + k] = e.Strength;
            at += n;
        }
        Array.Copy(main, 0, tokens, (long)at * 128, main.Length);
        Array.Copy(Coords(Fz, h, w, 0, req.FrameRate), 0, coords, at * 3, N * 3);
        for (int k = 0; k < N; k++) mask[at + k] = frameMask[k / hw];

        // runs of tokens with the same strength (the timestep is a function of it)
        var runs = new List<(int Start, int Count, float M)>();
        for (int k = 0; k < Nt; k++)
            if (runs.Count > 0 && runs[^1].M == mask[k]) runs[^1] = (runs[^1].Start, runs[^1].Count + 1, mask[k]);
            else runs.Add((k, 1, mask[k]));
        bool anyHard = runs.Any(r => r.M > 1f - 1e-6f);

        // 3. the denoising loop on the device
        Report("denoise", 0, "loading the transformer");
        var lk = new LtxKernels(_acc);
        using var dw = await AcquireAsync(_paths.Dit, ct);
        using var dit = new LtxDiT(_acc, new DiTKernels(_acc), dw.W);
        dit.SetPositions(coords);
        dit.SetContext(0, ctx, rows);
        using var lat = _acc.Allocate1D(tokens); using var init = _acc.Allocate1D(tokens); using var v = _acc.Allocate1D<float>(tokens.Length);
        using var stepNoise = anyHard && req.ImageCondNoiseScale > 0 ? _acc.Allocate1D<float>(tokens.Length) : null;
        for (int s = 0; s < steps; s++)
        {
            ct.ThrowIfCancellationRequested();
            float t = LtxSampler.Timestep(sigmas[s]);
            if (stepNoise != null)
            {
                var sn = req.StepNoise?.ElementAtOrDefault(s) ?? Gaussian((long)Nt * 128, req.Seed + 3000 + s);
                if (sn.Length != tokens.Length) throw new ArgumentException("StepNoise entries must be [tokens, 128]");
                stepNoise.View.CopyFromCPU(sn);
                float scale = req.ImageCondNoiseScale * (t / 1000f) * (t / 1000f);
                foreach (var r in runs.Where(r => r.M > 1f - 1e-6f))
                    lk.AddScaled(init.View.SubView((long)r.Start * 128, (long)r.Count * 128), stepNoise.View.SubView((long)r.Start * 128, (long)r.Count * 128), scale,
                        lat.View.SubView((long)r.Start * 128, (long)r.Count * 128), r.Count * 128);
            }
            var segs = new List<LtxTimestepSegment>();
            foreach (var r in runs)
            {
                double tt = Math.Min(t, (1f - r.M) * 1000f);
                if (segs.Count > 0 && segs[^1].Timestep == tt) segs[^1] = segs[^1] with { Count = segs[^1].Count + r.Count };
                else segs.Add(new LtxTimestepSegment(r.Start, r.Count, tt));
            }
            dit.Forward(lat.View, segs, 0, v.View, tokenMajor: true);
            StepTap?.Invoke(s, lat.View, v.View);
            if (Preview != null)
            {
                var xs = await lat.View.CopyToHostAsync(); var vs = await v.View.CopyToHostAsync();
                var x0 = new float[128L * N];
                foreach (var r in runs)
                {
                    float sg = Math.Min(t, (1f - r.M) * 1000f) / 1000f;                       // the token's own sigma: 0 for a hard-conditioned token
                    for (int k = Math.Max(r.Start, extraTokens); k < r.Start + r.Count; k++)
                        for (int c = 0; c < 128; c++) x0[(long)c * N + k - extraTokens] = xs[(long)k * 128 + c] - sg * vs[(long)k * 128 + c];
                }
                Preview(new LtxPreview(_pass, s + 1, steps, x0, Fz, h, w));
            }
            foreach (var r in runs)
            {
                if (!(t / 1000f - 1e-6f < 1f - r.M)) continue;                                // tokens_to_denoise_mask: a conditioned token waits until the schedule reaches its strength
                float cur = Math.Min(t, (1f - r.M) * 1000f) / 1000f;
                float next = 0f;
                foreach (var sg in sigmas) if (sg < cur - 1e-6f && sg > next) next = sg;          // the next lower sigma of the schedule (FlowMatchEulerDiscreteScheduler, per-token)
                lk.Axpy(lat.View.SubView((long)r.Start * 128, (long)r.Count * 128), v.View.SubView((long)r.Start * 128, (long)r.Count * 128), next - cur, r.Count * 128);
            }
            await _acc.SynchronizeAsync();
            Report("denoise", (s + 1.0) / steps, $"step {s + 1} of {steps}");
        }

        // the video's tokens back to [128, Fz, h, w]
        var all = await lat.View.CopyToHostAsync();
        var latent = new float[128L * N];
        for (int n = 0; n < N; n++) for (int c = 0; c < 128; c++) latent[(long)c * N + n] = all[((long)extraTokens + n) * 128 + c];
        return latent;
    }
}
