using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnDev.ILGPU.ML.Tensors;

namespace SpawnDev.ILGPU.ML.Pipelines;

/// <summary>
/// Text-to-Image Generation: text prompt → generated image.
/// Implements the Stable Diffusion pipeline:
///   1. Text encoding (CLIP) → text embeddings
///   2. Noise generation → initial latent
///   3. Iterative denoising (UNet) with scheduler
///   4. VAE decoding → pixel image
///
/// SD Turbo is recommended for browser (single-step, no guidance, ~2.5 GB).
///
/// Usage:
///   var hub = new ModelHub(js);
///   var pipe = await ImageGenerationPipeline.CreateAsync(accelerator, hub,
///       ModelHub.KnownModels.SDTurbo, onProgress: (stage, pct) => UpdateUI(stage, pct));
///   pipe.NumInferenceSteps = 1; // SD Turbo: single step
///   pipe.GuidanceScale = 0f;    // SD Turbo: no guidance
///   var result = await pipe.RunAsync(new ImageGenerationInput { Prompt = "a photo of a cat" });
///   // result.ImageRGBA is 512x512 RGBA pixels
/// </summary>
public class ImageGenerationPipeline : IPipeline<ImageGenerationInput, ImageGenerationResult>
{
    private readonly Accelerator _accelerator;
    private InferenceSession? _textEncoder;
    private InferenceSession? _unet;
    private InferenceSession? _vaeDecoder;
    private BPETokenizer? _tokenizer;
    private float[]? _alphasCumprod;

    public bool IsReady => _textEncoder != null && _unet != null && _vaeDecoder != null && _tokenizer != null;
    public string ModelName { get; private set; } = "";
    public string BackendName => _accelerator.AcceleratorType.ToString();

    /// <summary>Number of denoising steps (20-50 for SD 1.5, 1 for SD Turbo).</summary>
    public int NumInferenceSteps { get; set; } = 1;

    /// <summary>Guidance scale for classifier-free guidance (7.5 typical, 0 for SD Turbo).</summary>
    public float GuidanceScale { get; set; } = 0f;

    /// <summary>Output image width (must be multiple of 8, typically 512).</summary>
    public int Width { get; set; } = 512;

    /// <summary>Output image height (must be multiple of 8, typically 512).</summary>
    public int Height { get; set; } = 512;

    /// <summary>Random seed for reproducible generation. Null = random.</summary>
    public int? Seed { get; set; }

    /// <summary>Scheduler type: "ddim" or "euler".</summary>
    public string Scheduler { get; set; } = "euler";

    /// <summary>Progress callback: (currentStep, totalSteps).</summary>
    public event Action<int, int>? OnProgress;

    private ImageGenerationPipeline(Accelerator accelerator) => _accelerator = accelerator;

    /// <summary>
    /// Create an SD-Turbo pipeline from HuggingFace Hub with OPFS caching.
    /// Downloads 3 ONNX models (~2.5 GB total) + tokenizer on first call.
    /// Subsequent calls load from browser cache.
    ///
    /// Models from schmuell/sd-turbo-ort-web (FP16, WebGPU-optimized):
    ///   text_encoder/model.onnx (681 MB) — CLIP text encoder
    ///   unet/model.onnx (1,733 MB) — Single-step denoiser
    ///   vae_decoder/model.onnx (99 MB) — Latent → pixel decoder
    /// </summary>
    public static async Task<ImageGenerationPipeline> CreateAsync(
        Accelerator accelerator, ModelHub hub, string? repoId = null,
        Action<string, int>? onProgress = null)
    {
        var pipe = new ImageGenerationPipeline(accelerator);
        repoId ??= ModelHub.KnownModels.SDTurbo;
        pipe.ModelName = repoId;

        // Load tokenizer
        onProgress?.Invoke("tokenizer", 0);
        var tokenizerBytes = await hub.LoadAsync(repoId, "tokenizer/tokenizer.json");
        var tokenizerJson = System.Text.Encoding.UTF8.GetString(tokenizerBytes);
        pipe._tokenizer = BPETokenizer.LoadFromTokenizerJson(tokenizerJson);
        onProgress?.Invoke("tokenizer", 100);

        // Load 3 ONNX sub-models (sequential — browser memory limited)
        onProgress?.Invoke("text_encoder", 0);
        var textEncoderBytes = await hub.LoadAsync(repoId, "text_encoder/model.onnx");
        pipe._textEncoder = InferenceSession.CreateFromOnnx(accelerator, textEncoderBytes,
            inputShapes: new Dictionary<string, int[]> { ["input_ids"] = new[] { 1, 77 } });
        onProgress?.Invoke("text_encoder", 100);

        onProgress?.Invoke("unet", 0);
        var unetBytes = await hub.LoadAsync(repoId, "unet/model.onnx");
        pipe._unet = InferenceSession.CreateFromOnnx(accelerator, unetBytes,
            inputShapes: new Dictionary<string, int[]>
            {
                ["sample"] = new[] { 1, 4, 64, 64 },
                ["timestep"] = new[] { 1 },
                ["encoder_hidden_states"] = new[] { 1, 77, 1024 },
            });
        onProgress?.Invoke("unet", 100);

        onProgress?.Invoke("vae_decoder", 0);
        var vaeBytes = await hub.LoadAsync(repoId, "vae_decoder/model.onnx");
        pipe._vaeDecoder = InferenceSession.CreateFromOnnx(accelerator, vaeBytes,
            inputShapes: new Dictionary<string, int[]> { ["latent_sample"] = new[] { 1, 4, 64, 64 } });
        onProgress?.Invoke("vae_decoder", 100);

        pipe._alphasCumprod = DiffusionScheduler.ComputeAlphasCumprod();

        return pipe;
    }

    /// <summary>
    /// Create pipeline from HttpClient (non-HuggingFace, direct URL loading).
    /// </summary>
    public static async Task<ImageGenerationPipeline> CreateAsync(
        Accelerator accelerator, HttpClient http, string? modelId, PipelineOptions options)
    {
        var pipe = new ImageGenerationPipeline(accelerator);
        var basePath = modelId ?? options.ModelPath ?? "models/sd-turbo";
        pipe.ModelName = basePath;
        pipe._alphasCumprod = DiffusionScheduler.ComputeAlphasCumprod();
        return pipe;
    }

    /// <summary>
    /// Generate an image from a text prompt.
    /// For SD Turbo: single-step generation (NumInferenceSteps=1, GuidanceScale=0).
    /// </summary>
    public async Task<ImageGenerationResult> RunAsync(ImageGenerationInput input)
    {
        if (!IsReady) throw new InvalidOperationException("Pipeline not loaded. Call CreateAsync first.");

        var sw = System.Diagnostics.Stopwatch.StartNew();
        int steps = input.NumSteps ?? NumInferenceSteps;
        float guidance = input.GuidanceScale ?? GuidanceScale;
        int seed = input.Seed ?? Seed ?? Environment.TickCount;

        // ═══════════════════════════════════════════════════════════
        //  Step 1: Tokenize with CLIP BPE (pad to 77 tokens)
        // ═══════════════════════════════════════════════════════════
        var tokenIds = _tokenizer!.EncodeCLIP(input.Prompt, maxLength: 77);
        var tokenFloats = tokenIds.Select(t => (float)t).ToArray();

        using var tokenBuf = _accelerator.Allocate1D(tokenFloats);
        var tokenTensor = new Tensor(tokenBuf.View, new[] { 1, 77 });

        // ═══════════════════════════════════════════════════════════
        //  Step 2: Text encode → [1, 77, 1024] embeddings
        // ═══════════════════════════════════════════════════════════
        OnProgress?.Invoke(0, steps + 2); // +2 for text encode + VAE decode

        var textInputs = new Dictionary<string, Tensor>
        {
            [_textEncoder!.InputNames[0]] = tokenTensor,
        };
        var textOutputs = await _textEncoder.RunAsync(textInputs);
        var textEmbeddings = textOutputs[_textEncoder.OutputNames[0]]; // [1, 77, 1024]

        // ═══════════════════════════════════════════════════════════
        //  Step 3: Generate initial Gaussian noise latent [1, 4, 64, 64]
        // ═══════════════════════════════════════════════════════════
        int latentH = Height / 8;
        int latentW = Width / 8;
        var noiseData = DiffusionScheduler.GenerateNoise(4, latentH, latentW, seed);

        // Scale initial noise by first sigma (for Euler scheduler)
        if (Scheduler == "euler" && _alphasCumprod != null)
        {
            var timesteps = DiffusionScheduler.GetTimesteps(steps);
            var sigmas = DiffusionScheduler.TimestepsToSigmas(timesteps, _alphasCumprod);
            noiseData = DiffusionScheduler.ScaleNoise(noiseData, sigmas[0]);
        }

        using var latentBuf = _accelerator.Allocate1D(noiseData);
        var latentTensor = new Tensor(latentBuf.View, new[] { 1, 4, latentH, latentW });

        // ═══════════════════════════════════════════════════════════
        //  Step 4: UNet denoising (1 step for SD Turbo)
        // ═══════════════════════════════════════════════════════════
        var timestepValues = DiffusionScheduler.GetTimesteps(steps);

        for (int step = 0; step < steps; step++)
        {
            OnProgress?.Invoke(step + 1, steps + 2);

            // Timestep as float tensor [1]
            using var tBuf = _accelerator.Allocate1D(new float[] { timestepValues[step] });
            var tTensor = new Tensor(tBuf.View, new[] { 1 });

            var unetInputs = new Dictionary<string, Tensor>
            {
                [_unet!.InputNames[0]] = latentTensor,   // sample [1,4,64,64]
                [_unet.InputNames[1]] = tTensor,          // timestep [1]
                [_unet.InputNames[2]] = textEmbeddings,   // encoder_hidden_states [1,77,1024]
            };

            var unetOutputs = await _unet.RunAsync(unetInputs);
            var noisePred = unetOutputs[_unet.OutputNames[0]]; // out_sample [1,4,64,64]

            if (steps == 1)
            {
                // SD Turbo single-step: output IS the denoised latent, no scheduler step needed
                // Copy denoised output to latent tensor for VAE
                new ElementWiseKernels(_accelerator).Scale(
                    noisePred.Data.SubView(0, noiseData.Length),
                    latentTensor.Data.SubView(0, noiseData.Length),
                    noiseData.Length, 1f);
                await _accelerator.SynchronizeAsync();
            }
            else
            {
                // Multi-step: apply scheduler (DDIM or Euler)
                var noisePredCpu = await ReadTensorToCpu(noisePred, noiseData.Length);
                var latentCpu = await ReadTensorToCpu(latentTensor, noiseData.Length);

                float[] updated;
                if (Scheduler == "euler" && _alphasCumprod != null)
                {
                    var sigmas = DiffusionScheduler.TimestepsToSigmas(timestepValues, _alphasCumprod);
                    updated = DiffusionScheduler.EulerStep(noisePredCpu, latentCpu, sigmas[step], sigmas[step + 1]);
                }
                else
                {
                    int prevT = step + 1 < timestepValues.Length ? timestepValues[step + 1] : -1;
                    updated = DiffusionScheduler.DDIMStep(noisePredCpu, latentCpu, timestepValues[step], prevT, _alphasCumprod!);
                }

                latentTensor.Data.SubView(0, updated.Length).CopyFromCPU(updated);
                await _accelerator.SynchronizeAsync();
            }
        }

        // ═══════════════════════════════════════════════════════════
        //  Step 5: Scale latent for VAE (1 / 0.18215)
        // ═══════════════════════════════════════════════════════════
        const float vaeScaleFactor = 1f / 0.18215f;
        new ElementWiseKernels(_accelerator).ScaleInPlace(
            latentTensor.Data.SubView(0, noiseData.Length),
            noiseData.Length, vaeScaleFactor);
        await _accelerator.SynchronizeAsync();

        // ═══════════════════════════════════════════════════════════
        //  Step 6: VAE decode → [1, 3, 512, 512] RGB
        // ═══════════════════════════════════════════════════════════
        OnProgress?.Invoke(steps + 1, steps + 2);

        var vaeInputs = new Dictionary<string, Tensor>
        {
            [_vaeDecoder!.InputNames[0]] = latentTensor,
        };
        var vaeOutputs = await _vaeDecoder.RunAsync(vaeInputs);
        var imageOutput = vaeOutputs[_vaeDecoder.OutputNames[0]]; // [1, 3, 512, 512]

        // ═══════════════════════════════════════════════════════════
        //  Step 7: Convert NCHW [-1,1] → RGBA [0,255]
        // ═══════════════════════════════════════════════════════════
        int imagePixels = Width * Height;
        var rgbData = await ReadTensorToCpu(imageOutput, 3 * imagePixels);

        var rgba = new byte[4 * imagePixels];
        for (int i = 0; i < imagePixels; i++)
        {
            // NCHW layout: R at [0*HW+i], G at [1*HW+i], B at [2*HW+i]
            float r = (rgbData[0 * imagePixels + i] + 1f) * 0.5f * 255f;
            float g = (rgbData[1 * imagePixels + i] + 1f) * 0.5f * 255f;
            float b = (rgbData[2 * imagePixels + i] + 1f) * 0.5f * 255f;

            rgba[i * 4 + 0] = (byte)Math.Clamp((int)(r + 0.5f), 0, 255);
            rgba[i * 4 + 1] = (byte)Math.Clamp((int)(g + 0.5f), 0, 255);
            rgba[i * 4 + 2] = (byte)Math.Clamp((int)(b + 0.5f), 0, 255);
            rgba[i * 4 + 3] = 255; // Full alpha
        }

        OnProgress?.Invoke(steps + 2, steps + 2);
        sw.Stop();

        return new ImageGenerationResult
        {
            ImageRGBA = rgba,
            Width = Width,
            Height = Height,
            Prompt = input.Prompt,
            Seed = seed,
            NumSteps = steps,
            InferenceTimeMs = sw.Elapsed.TotalMilliseconds,
        };
    }

    private async Task<float[]> ReadTensorToCpu(Tensor tensor, int count)
    {
        using var readBuf = _accelerator.Allocate1D<float>(count);
        new ElementWiseKernels(_accelerator).Scale(
            tensor.Data.SubView(0, count), readBuf.View, count, 1f);
        await _accelerator.SynchronizeAsync();
        return await readBuf.CopyToHostAsync<float>(0, count);
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

    /// <summary>Negative prompt (what to avoid). Not used by SD Turbo.</summary>
    public string NegativePrompt { get; set; } = "";

    /// <summary>Override number of steps (null = use pipeline default). SD Turbo: 1.</summary>
    public int? NumSteps { get; set; }

    /// <summary>Override guidance scale (null = use pipeline default). SD Turbo: 0.</summary>
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
