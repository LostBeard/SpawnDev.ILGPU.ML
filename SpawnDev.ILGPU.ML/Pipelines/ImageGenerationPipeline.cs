using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Preprocessing;

namespace SpawnDev.ILGPU.ML.Pipelines;

/// <summary>
/// Text-to-Image Generation: text prompt → generated image.
/// Implements the Stable Diffusion pipeline:
///   1. Text encoding (CLIP) → text embeddings
///   2. Noise generation → initial latent
///   3. Iterative denoising (UNet) with scheduler
///   4. VAE decoding → pixel image
///
/// Models: Stable Diffusion 1.5, SD 2.1, SDXL, SD Turbo.
/// SD Turbo is recommended for browser (single-step generation).
/// </summary>
public class ImageGenerationPipeline : IPipeline<ImageGenerationInput, ImageGenerationResult>
{
    private readonly Accelerator _accelerator;
    private InferenceSession? _textEncoder;
    private InferenceSession? _unet;
    private InferenceSession? _vaeDecoder;
    private BPETokenizer? _tokenizer;
    private float[]? _alphasCumprod;

    public bool IsReady => _textEncoder != null && _unet != null && _vaeDecoder != null;
    public string ModelName { get; private set; } = "";
    public string BackendName => _accelerator.AcceleratorType.ToString();

    /// <summary>Number of denoising steps (20-50 for SD 1.5, 1 for SD Turbo).</summary>
    public int NumInferenceSteps { get; set; } = 20;

    /// <summary>Guidance scale for classifier-free guidance (7.5 typical, 0 for SD Turbo).</summary>
    public float GuidanceScale { get; set; } = 7.5f;

    /// <summary>Output image width (must be multiple of 8, typically 512).</summary>
    public int Width { get; set; } = 512;

    /// <summary>Output image height (must be multiple of 8, typically 512).</summary>
    public int Height { get; set; } = 512;

    /// <summary>Random seed for reproducible generation. Null = random.</summary>
    public int? Seed { get; set; }

    /// <summary>Scheduler type: "ddim" or "euler".</summary>
    public string Scheduler { get; set; } = "ddim";

    /// <summary>Progress callback: (currentStep, totalSteps).</summary>
    public event Action<int, int>? OnProgress;

    private ImageGenerationPipeline(Accelerator accelerator) => _accelerator = accelerator;

    public static async Task<ImageGenerationPipeline> CreateAsync(
        Accelerator accelerator, HttpClient http, string? modelId, PipelineOptions options)
    {
        var pipe = new ImageGenerationPipeline(accelerator);
        var basePath = modelId ?? options.ModelPath ?? "models/sd-turbo";
        pipe.ModelName = basePath;

        // SD has 3 sub-models + tokenizer
        // TODO: Load text_encoder, unet, vae_decoder from sub-directories
        // pipe._textEncoder = await InferenceSession.CreateAsync(accelerator, http, $"{basePath}/text_encoder");
        // pipe._unet = await InferenceSession.CreateAsync(accelerator, http, $"{basePath}/unet");
        // pipe._vaeDecoder = await InferenceSession.CreateAsync(accelerator, http, $"{basePath}/vae_decoder");
        // pipe._tokenizer = load tokenizer...

        pipe._alphasCumprod = DiffusionScheduler.ComputeAlphasCumprod();

        return pipe;
    }

    /// <summary>
    /// Generate an image from a text prompt.
    /// </summary>
    public async Task<ImageGenerationResult> RunAsync(ImageGenerationInput input)
    {
        if (!IsReady) throw new InvalidOperationException("Model not loaded");

        var sw = System.Diagnostics.Stopwatch.StartNew();

        // Step 1: Tokenize and encode text
        // var tokens = _tokenizer!.EncodeCLIP(input.Prompt, 77);
        // var textEmbeddings = RunTextEncoder(tokens);
        // If using classifier-free guidance, also encode empty prompt
        // var uncondEmbeddings = RunTextEncoder(_tokenizer.EncodeCLIP("", 77));

        // Step 2: Generate initial noise latent
        int latentH = Height / 8;
        int latentW = Width / 8;
        var latent = DiffusionScheduler.GenerateNoise(4, latentH, latentW, Seed);

        // Step 3: Get timesteps
        var timesteps = DiffusionScheduler.GetTimesteps(NumInferenceSteps);

        // Step 4: Denoising loop
        for (int step = 0; step < timesteps.Length; step++)
        {
            OnProgress?.Invoke(step, timesteps.Length);

            // Prepare UNet input: concat(latent, latent) for classifier-free guidance
            // var noisePred = RunUNet(latent, timesteps[step], textEmbeddings);

            // Apply guidance: noise_pred = uncond + guidance_scale * (cond - uncond)

            // Scheduler step: update latent
            int prevTimestep = step + 1 < timesteps.Length ? timesteps[step + 1] : -1;
            // latent = DiffusionScheduler.DDIMStep(noisePred, latent, timesteps[step], prevTimestep, _alphasCumprod!);
        }

        // Step 5: VAE decode latent → pixel image
        // Scale latent: latent = latent / 0.18215 (SD scaling factor)
        // var pixels = RunVAEDecoder(latent);

        // Step 6: Convert [-1,1] to [0,255] RGBA
        // var rgba = LatentToRGBA(pixels, Width, Height);

        sw.Stop();

        return new ImageGenerationResult
        {
            ImageRGBA = Array.Empty<byte>(), // TODO: actual pixels
            Width = Width,
            Height = Height,
            Prompt = input.Prompt,
            Seed = Seed ?? 0,
            NumSteps = NumInferenceSteps,
            InferenceTimeMs = sw.Elapsed.TotalMilliseconds,
        };
    }

    public void Dispose()
    {
        _textEncoder?.Dispose();
        _unet?.Dispose();
        _vaeDecoder?.Dispose();
    }
}

/// <summary>Input for image generation.</summary>
public class ImageGenerationInput
{
    /// <summary>Text description of the desired image.</summary>
    public string Prompt { get; set; } = "";

    /// <summary>Negative prompt (what to avoid).</summary>
    public string NegativePrompt { get; set; } = "";

    /// <summary>Override number of steps (null = use pipeline default).</summary>
    public int? NumSteps { get; set; }

    /// <summary>Override guidance scale (null = use pipeline default).</summary>
    public float? GuidanceScale { get; set; }

    /// <summary>Override seed (null = use pipeline default).</summary>
    public int? Seed { get; set; }
}

/// <summary>Result from image generation.</summary>
public class ImageGenerationResult
{
    public byte[] ImageRGBA { get; init; } = Array.Empty<byte>();
    public int Width { get; init; }
    public int Height { get; init; }
    public string Prompt { get; init; } = "";
    public int Seed { get; init; }
    public int NumSteps { get; init; }
    public double InferenceTimeMs { get; init; }
}
