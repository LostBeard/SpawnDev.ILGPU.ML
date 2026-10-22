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
    /// <summary><c>t5_xxl_enc_bf16.safetensors</c> (9.5 GB, streamed layer by layer).</summary>
    public required string TextEncoder { get; init; }
    /// <summary>The <c>tokenizer.json</c> of google/t5-v1_1-xxl.</summary>
    public required string Tokenizer { get; init; }

    /// <summary>The DiT file for a precision: "bf16" (3.6 GB) or "q8" (2.0 GB).</summary>
    public static string DitFile(string precision) => $"ltx_dit_{precision}.safetensors";
    /// <summary>The T5 file for a precision: "bf16" (8.9 GB) or "q8" (5.0 GB).</summary>
    public static string TextEncoderFile(string precision) => $"t5_xxl_enc_{precision}.safetensors";

    /// <summary>The file names the exporters write and a model repo holds (relative, forward slashes).</summary>
    public static LtxPaths InRepo(string dit = "ltx_dit_bf16.safetensors", string textEncoder = "t5_xxl_enc_bf16.safetensors") => new()
    {
        Dit = dit, Vae = "ltx_vae_f16.safetensors", TextEncoder = textEncoder, Tokenizer = "google/t5-v1_1-xxl/tokenizer.json",
    };

    /// <summary>The conventional layout of one folder on disk: the file names the exporters write.</summary>
    public static LtxPaths InFolder(string folder, string dit = "ltx_dit_bf16.safetensors", string textEncoder = "t5_xxl_enc_bf16.safetensors") => new()
    {
        Dit = Path.Combine(folder, dit), Vae = Path.Combine(folder, "ltx_vae_f16.safetensors"), TextEncoder = Path.Combine(folder, textEncoder),
        Tokenizer = Path.Combine(folder, "google", "t5-v1_1-xxl", "tokenizer.json"),
    };
}

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
}

/// <summary>
/// LTX-Video 2B 0.9.8 distilled, text to video: T5-v1.1-XXL text encoding (streamed layer by layer), the DiT with the distilled flow-matching Euler sampler (one forward per step, the latent
/// stays on the device), then the timestep-conditioned VAE decode. The stages run one after another so only ONE large model is on the device at a time. Returns the same container as
/// the Wan pipeline (frames in [-1, 1], channel-first). Every stage is gated against the diffusers reference (see <c>Plans/ltx-video-2b-2026-10-22.md</c>).
/// </summary>
public sealed class LtxPipeline
{
    private readonly Accelerator _acc;
    private readonly LtxPaths _paths;
    private readonly IModelSource _source;
    private readonly string _repo;
    private UnigramTokenizer? _tok;

    /// <summary>The model repo the SpawnDev models host serves the exports from (https://models.spawndev.com/&lt;repo&gt;/resolve/main/&lt;file&gt;).</summary>
    public const string DefaultRepo = "LostBeard/LTX-Video-2B-0.9.8-distilled-ILGPU";
    private const int TextLength = 128;

    public LtxPipeline(Accelerator accelerator, IModelSource source, string repo, LtxPaths paths) { _acc = accelerator; _source = source; _repo = repo; _paths = paths; }

    /// <summary>Files on disk: <paramref name="paths"/> are file paths (<see cref="LtxPaths.InFolder"/>).</summary>
    public LtxPipeline(Accelerator accelerator, LtxPaths paths) : this(accelerator, new LocalFolderModelSource(""), "", paths) { }

    private async Task<GpuWeights> LoadWeightsAsync(string file, CancellationToken ct)
    {
        await using var stream = await _source.OpenAsync(_repo, file, ct);
        using var empty = System.Text.Json.JsonDocument.Parse("{}");
        return await GpuWeights.LoadAsync(_acc, stream, empty.RootElement, file, cancellationToken: ct);
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

    public async Task<WanVideo> GenerateAsync(LtxRequest req, IProgress<WanProgress>? progress = null, CancellationToken ct = default)
    {
        if (req.Width % 32 != 0 || req.Height % 32 != 0) throw new ArgumentException("width and height must be multiples of 32");
        if ((req.Frames - 1) % 8 != 0 || req.Frames < 1) throw new ArgumentException("the frame count must be 8n + 1");
        int Fz = (req.Frames - 1) / 8 + 1, h = req.Height / 32, w = req.Width / 32, N = Fz * h * w;
        void Report(string stage, double f, string msg) => progress?.Report(new WanProgress(stage, f, msg));
        var sigmas = LtxSampler.WithTerminal(req.Sigmas ?? LtxSampler.DistilledFirstPass);
        int steps = sigmas.Length - 1;

        // 1. text: the prompt through T5, one layer of its weights on the device at a time
        Report("text", 0, "encoding the prompt");
        var ids = await TokenizeAsync(req.Prompt, ct);
        float[] ctx;
        using (var t5 = await UmT5Encoder.CreateAsync(_acc, _source, _repo, _paths.TextEncoder, ct))
            ctx = await t5.EncodeAsync(ids, ct);
        Report("text", 1, $"{ids.Length} tokens");

        // 2. the denoising loop: the latent [128, Fz, h, w] stays on the device, the Euler step runs there too
        var latent = req.InitialNoise != null ? (float[])req.InitialNoise.Clone() : Gaussian(128L * N, req.Seed);
        if (latent.Length != 128L * N) throw new ArgumentException("InitialNoise has the wrong length for this size");
        Report("denoise", 0, "loading the transformer");
        var lk = new LtxKernels(_acc);
        using (var dw = await LoadWeightsAsync(_paths.Dit, ct))
        using (var dit = new LtxDiT(_acc, new DiTKernels(_acc), dw))
        {
            dit.SetGrid(Fz, h, w, 8f / req.FrameRate, 32f, 32f);
            dit.SetContext(0, ctx, ids.Length);
            using var lat = _acc.Allocate1D(latent); using var v = _acc.Allocate1D<float>(latent.Length);
            for (int s = 0; s < steps; s++)
            {
                ct.ThrowIfCancellationRequested();
                dit.Forward(lat.View, LtxSampler.Timestep(sigmas[s]), 0, v.View);
                lk.Axpy(lat.View, v.View, sigmas[s + 1] - sigmas[s], latent.Length);
                await _acc.SynchronizeAsync();                                           // one step at a time: the browser queue must not run ahead
                Report("denoise", (s + 1.0) / steps, $"step {s + 1} of {steps}");
            }
            latent = await lat.View.CopyToHostAsync();                                   // the small latent crosses once, for the decode
        }

        // 3. decode
        Report("decode", 0, "decoding the video");
        float[] video;
        var decodeNoise = req.DecodeNoiseScale != 0f ? Gaussian(latent.Length, req.Seed + 104729L) : null;
        using (var vw = await LoadWeightsAsync(_paths.Vae, ct))
        using (var vae = new LtxVae(_acc, vw))
            video = await vae.DecodeAsync(latent, Fz, h, w, req.DecodeTimestep, req.DecodeNoiseScale, decodeNoise);
        Report("decode", 1, "done");
        return new WanVideo { Data = video, Width = req.Width, Height = req.Height, Frames = 8 * (Fz - 1) + 1, Latent = latent };
    }
}
