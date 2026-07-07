using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Graph;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnDev.ILGPU.ML.Tensors;
using SpawnDev.ILGPU.ML.Tiling;

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
///   var hubStream = new HubModelStream(webTorrentClient, httpClient);
///   var pipe = await ImageGenerationPipeline.CreateAsync(accelerator, hubStream,
///       ModelHub.KnownModels.SDTurbo, onProgress: (stage, pct) => UpdateUI(stage, pct));
///   pipe.NumInferenceSteps = 1; // SD Turbo: single step
///   pipe.GuidanceScale = 0f;    // SD Turbo: no guidance
///   var result = await pipe.RunAsync(new ImageGenerationInput { Prompt = "a photo of a cat" });
///   // result.ImageRGBA is 512x512 RGBA pixels
/// </summary>
public class ImageGenerationPipeline : IPipeline<ImageGenerationInput, ImageGenerationResult>
{
    private readonly Accelerator _accelerator;
    private ElementWiseKernels? _elementWise;
    private ElementWiseKernels ElementWise => _elementWise ??= new ElementWiseKernels(_accelerator);   // one instance per pipeline (kernels lazy-load once, not per step)
    private InferenceSession? _textEncoder;
    private InferenceSession? _unet;
    private InferenceSession? _vaeDecoder;
    /// <summary>The VAE decoder session (for tiled-decode weight extraction / introspection).</summary>
    public InferenceSession? VaeDecoder => _vaeDecoder;
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

    /// <summary>Seam-free tiled VAE decode grid (NxN). 0 = auto (tile 2x2 when the output is >=512 on a side, so
    /// the decode's GPU peak stays bounded by one tile instead of the full-res up_blocks intermediates). -1 =
    /// force the full single-pass decode. Set explicitly to override. Added 2026-07-07 (Tuvok): the full 512x512
    /// VAE decode's peak resident set OOM-kills the GPU process at up_blocks.3 on constrained machines; tiling the
    /// up-blocks avoids the 256MiB full-res intermediates entirely. The VAE_TILE_EXACT env var still overrides.</summary>
    public int VaeTileGrid { get; set; } = 0;

    /// <summary>Scheduler type: "ddim" or "euler".</summary>
    public string Scheduler { get; set; } = "euler";

    /// <summary>Progress callback: (currentStep, totalSteps).</summary>
    public event Action<int, int>? OnProgress;

    /// <summary>Graph capture (CUDA graphs / WebGPU dispatch plans) for the three sub-models: first
    /// call per model captures, every later call replays - across steps AND across generations (the
    /// shapes are fixed). ON by default per the pipeline rule (the always-on switch lives here, not
    /// in consumers); no-op on non-capture backends. Opt out for one-shot use.</summary>
    // DEFAULT OFF for THIS pipeline (unlike depth): SD-class activation volumes guarantee pool
    // misses during the capture pass, and ILGPU's AllocateWithReclaim (flush/sync/alloc mid-capture)
    // corrupts the CUDA context under VRAM pressure - 0xC0000005, bisect-proven per sub-model
    // 2026-07-03. Direct generation is correct + fast (single-step); capture needs the pool-priming
    // work extended to SD scale first (tracked: Plans/sd-capture-pool-priming.md). Opt IN via
    // SDTURBO_FORCE_CAPTURE=1 for that development.
    public bool EnableGraphCapture { get; set; } = Environment.GetEnvironmentVariable("SDTURBO_FORCE_CAPTURE") == "1";
    // SDTURBO_CAPTURE scopes which sub-models capture (diagnostic): "clip"/"unet"/"vae" (comma list)
    // or unset = all. With ML_NO_SESSION_CAPTURE=1 nothing captures regardless.
    private static bool CapScope(string m)
    { var s = Environment.GetEnvironmentVariable("SDTURBO_CAPTURE"); return string.IsNullOrEmpty(s) || s.Contains(m, StringComparison.OrdinalIgnoreCase); }
    private Graph.SessionGraphCapture? _textEncoderCap, _unetCap, _vaeCap;
    private Graph.SessionGraphCapture TextEncoderCap => _textEncoderCap ??= new Graph.SessionGraphCapture(_textEncoder!, _accelerator) { Enabled = EnableGraphCapture && CapScope("clip") };
    private Graph.SessionGraphCapture UnetCap => _unetCap ??= new Graph.SessionGraphCapture(_unet!, _accelerator) { Enabled = EnableGraphCapture && CapScope("unet") };
    private Graph.SessionGraphCapture VaeCap => _vaeCap ??= new Graph.SessionGraphCapture(_vaeDecoder!, _accelerator) { Enabled = EnableGraphCapture && CapScope("vae") };

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
        Accelerator accelerator, HubModelStream hubStream, string? repoId = null,
        Action<string, int>? onProgress = null)
    {
        var pipe = new ImageGenerationPipeline(accelerator);
        repoId ??= ModelHub.KnownModels.SDTurbo;
        pipe.ModelName = repoId;

        // CLIP tokenizer: SD-Turbo (and CLIP generally) ships the classic vocab.json + merges.txt pair,
        // NOT a consolidated tokenizer.json — that file does not exist in schmuell/sd-turbo-ort-web, so
        // the old "tokenizer/tokenizer.json" open 404'd at the hub before /generate could ever load.
        // Read both small text files from the hub and build the BPE tokenizer (EncodeCLIP uses the
        // hardcoded CLIP 49406/49407 special-token ids, so vocab+merges is sufficient).
        onProgress?.Invoke("tokenizer", 0);
        string vocabJson, mergesText;
        var vocabFile = await hubStream.OpenAsync(repoId, "tokenizer/vocab.json");
        await using (vocabFile.Stream)
        using (var r = new System.IO.StreamReader(vocabFile.Stream))
            vocabJson = await r.ReadToEndAsync();
        // NOTE: do NOT remove the per-file torrents after load. RemoveAsync disposes the model's AsyncFSMemory
        //   Blobs, but the WebTorrent store's cached File references them (new File([blob]) is a reference, not a
        //   copy), so disposing mid-load raced an in-flight read -> intermittent NotReadableError. The download
        //   works fine on AsyncFSMemory alone (the browser spills the accumulated Blobs to disk).
        var mergesFile = await hubStream.OpenAsync(repoId, "tokenizer/merges.txt");
        await using (mergesFile.Stream)
        using (var r = new System.IO.StreamReader(mergesFile.Stream))
            mergesText = await r.ReadToEndAsync();
        pipe._tokenizer = BPETokenizer.Load(vocabJson, mergesText);
        onProgress?.Invoke("tokenizer", 100);

        // 3 ONNX sub-models streamed weight-by-weight straight to the GPU - never materialized whole in
        // memory (the U-Net alone is 1.7GB; a byte[] of that OOMs Blazor WASM, which is why /generate
        // never ran on the old hub.LoadAsync path). Sequential: browser memory is limited even while
        // streaming. The "upload" sub-progress drives each per-model bar so the long U-Net stream doesn't
        // look frozen (same lesson as the text-gen model-load progress fix).
        onProgress?.Invoke("text_encoder", 0);
        var teModel = await hubStream.OpenAsync(repoId, "text_encoder/model.onnx");
        await using (teModel.Stream)
            pipe._textEncoder = await InferenceSession.CreateFromOnnxStreamAsync(accelerator, teModel.Stream,
                onProgress: (s, p) => { Console.WriteLine($"[GenLoad {Environment.TickCount64}ms] text_encoder/{s} {p}%"); onProgress?.Invoke($"text_encoder:{s}", p); },
                inputShapes: new Dictionary<string, int[]> { ["input_ids"] = new[] { 1, 77 } });
        onProgress?.Invoke("text_encoder", 100);

        onProgress?.Invoke("unet", 0);
        var unetModel = await hubStream.OpenAsync(repoId, "unet/model.onnx");
        await using (unetModel.Stream)
            pipe._unet = await InferenceSession.CreateFromOnnxStreamAsync(accelerator, unetModel.Stream,
                onProgress: (s, p) => { Console.WriteLine($"[GenLoad {Environment.TickCount64}ms] unet/{s} {p}%"); onProgress?.Invoke($"unet:{s}", p); },
                inputShapes: new Dictionary<string, int[]>
                {
                    ["sample"] = new[] { 1, 4, 64, 64 },
                    ["timestep"] = new[] { 1 },
                    ["encoder_hidden_states"] = new[] { 1, 77, 1024 },
                });
        onProgress?.Invoke("unet", 100);

        onProgress?.Invoke("vae_decoder", 0);
        var vaeModel = await hubStream.OpenAsync(repoId, "vae_decoder/model.onnx");
        await using (vaeModel.Stream)
            pipe._vaeDecoder = await InferenceSession.CreateFromOnnxStreamAsync(accelerator, vaeModel.Stream,
                onProgress: (s, p) => { Console.WriteLine($"[GenLoad {Environment.TickCount64}ms] vae_decoder/{s} {p}%"); onProgress?.Invoke($"vae_decoder:{s}", p); },
                inputShapes: new Dictionary<string, int[]> { ["latent_sample"] = new[] { 1, 4, 64, 64 } });
        // VAE fp16-activation storage via the precision-AWARE pass-through (approach i): Conv/InstanceNorm/
        // Sigmoid/Mul/Add/Relu read+write fp16 DIRECTLY (no fp32 temp). The mechanism is correct (image
        // bit-near-identical + sharp) and PROVEN to cut the working set on the controlled de-risk graph
        // (MixedPrecisionExecutorTests, F16 peak < F32 all 6 backends). BUT on the FULL SD-Turbo VAE it does NOT
        // yet reduce peak: the working-set peak (~2194 MiB) is set by ops that still fall back to fp32 — the
        // mid-block attention (MatMul/Softmax/Transpose) + Reshape/Resize — and each fallback op that reads a
        // half input spawns an fp32 convert-temp. Measured (RTX 4070, "a nice house"): peak TOTAL fp32=3507 →
        // F16=3698 MiB, peak LIVE 2194 unchanged. So F16 is OPT-IN (VAE_ACT_F16=1) until those fallback ops are
        // covered (the real win). Default stays F32 = no regression. See the plan doc.
        if (Environment.GetEnvironmentVariable("VAE_ACT_F16") == "1")
            pipe._vaeDecoder.Executor.ActivationDtype = ActivationPrecision.F16;
        onProgress?.Invoke("vae_decoder", 100);

        pipe._alphasCumprod = DiffusionScheduler.ComputeAlphasCumprod();

        // Warm shape-readback cache. SD-Turbo's tensor shapes are FIXED across generations (latent
        // 1x4x64x64, tokens 1x77, embeddings 1x77x1024), so every generation reads back the IDENTICAL
        // shape constants (Reshape/Slice/Concat dims). On the browser GPU backends each such readback is
        // a ~345ms mapAsync round-trip, and the UNet alone triggers thousands (measured raw, CUDA/no-
        // interp: text_encoder 897x, unet 2746x, vae 71x). CacheShapeReadbacks makes generation 1 record
        // the proven-stable set and every LATER generation serve those values from the CPU cache — the
        // GPU round-trip is skipped entirely on the warm path. Same lever GgufGenerator uses for its
        // fixed-shape decode loop; the sub-models each run once per generation and their outputs are fully
        // consumed before the next generation, so the fixed-shape recycling is safe here.
        pipe._textEncoder!.CacheShapeReadbacks = true;
        pipe._unet!.CacheShapeReadbacks = true;
        pipe._vaeDecoder!.CacheShapeReadbacks = true;

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
        var textOutputs = await TimedRun("text_encoder", _textEncoder!, TextEncoderCap, textInputs);
        var textEmbeddings = textOutputs[_textEncoder.OutputNames[0]]; // [1, 77, 1024]

        // ═══════════════════════════════════════════════════════════
        //  Step 3: Generate initial Gaussian noise latent [1, 4, 64, 64]
        // ═══════════════════════════════════════════════════════════
        int latentH = Height / 8;
        int latentW = Width / 8;
        var noiseData = DiffusionScheduler.GenerateNoise(4, latentH, latentW, seed);

        // Euler sigmas for the selected (trailing) timesteps. sigmas[0] = sigma_max (the Euler init
        // scale), sigmas[^1] = 0. SD-Turbo's single step starts from FULL noise, so the init latent is
        // noise * sigma_max — without this the UNet denoises near-zero input and emits a flat image.
        var timestepValues = DiffusionScheduler.GetTimesteps(steps);
        float[]? sigmas = _alphasCumprod != null
            ? DiffusionScheduler.TimestepsToSigmas(timestepValues, _alphasCumprod)
            : null;
        bool euler = Scheduler == "euler" && sigmas != null;
        if (euler)
            noiseData = DiffusionScheduler.ScaleNoise(noiseData, sigmas![0]);

        using var latentBuf = _accelerator.Allocate1D(noiseData);
        var latentTensor = new Tensor(latentBuf.View, new[] { 1, 4, latentH, latentW });

        // ═══════════════════════════════════════════════════════════
        //  Step 4: Denoising. SD-Turbo = ONE Euler step from full noise. The UNet predicts epsilon
        //  (scheduler_config prediction_type="epsilon"), so each step:
        //    (a) scale_model_input — feed the UNet sample/sqrt(sigma^2+1) (it's trained on ~unit
        //        variance; the raw latent is sigma-times too large → out-of-distribution → garbage),
        //    (b) UNet → epsilon,
        //    (c) Euler step on the ORIGINAL sample: x_{t-1} = sample + epsilon*(sigma_next - sigma);
        //        final step sigma_next=0 ⇒ x0 = sample - sigma*epsilon (the denoised latent).
        //  The old code SHORT-CIRCUITED steps==1 by copying the UNet output into the latent — i.e. it
        //  treated the predicted NOISE as the image. That, plus a clean-end timestep (GetTimesteps,
        //  now trailing) and no scale_model_input, is why /generate produced garbage.
        // ═══════════════════════════════════════════════════════════
        for (int step = 0; step < steps; step++)
        {
            OnProgress?.Invoke(step + 1, steps + 2);

            // (a) scale_model_input (Euler): unet_input = sample / sqrt(sigma^2 + 1).
            using var scaledBuf = euler ? _accelerator.Allocate1D<float>(noiseData.Length) : null;
            Tensor unetSample = latentTensor;
            if (euler)
            {
                float c = 1f / MathF.Sqrt(sigmas![step] * sigmas[step] + 1f);
                ElementWise.Scale(
                    latentTensor.Data.SubView(0, noiseData.Length), scaledBuf!.View, noiseData.Length, c);
                await _accelerator.SynchronizeAsync();
                unetSample = new Tensor(scaledBuf.View, new[] { 1, 4, latentH, latentW });
            }

            using var tBuf = _accelerator.Allocate1D(new float[] { timestepValues[step] });
            var tTensor = new Tensor(tBuf.View, new[] { 1 });

            var unetInputs = new Dictionary<string, Tensor>
            {
                [_unet!.InputNames[0]] = unetSample,      // sample [1,4,64,64] (scale_model_input applied)
                [_unet.InputNames[1]] = tTensor,          // timestep [1]
                [_unet.InputNames[2]] = textEmbeddings,   // encoder_hidden_states [1,77,1024]
            };

            var unetOutputs = await TimedRun("unet", _unet!, UnetCap, unetInputs);
            var noisePred = unetOutputs[_unet.OutputNames[0]]; // epsilon [1,4,64,64]

            // (b,c) scheduler step on the ORIGINAL (unscaled) latent.
            if (euler)
            {
                // GPU Euler step: latent += epsilon * (sigma_next - sigma) - ONE axpy dispatch. The
                // old path read BOTH tensors to the host, looped, and re-uploaded - two readbacks +
                // an upload per denoise step for math the GPU does in microseconds (zero-copy law).
                float dt = sigmas![step + 1] - sigmas[step];
                ElementWise.AddScaledInPlace(
                    latentTensor.Data.SubView(0, noiseData.Length),
                    noisePred.Data.SubView(0, noiseData.Length),
                    noiseData.Length, dt);
                await _accelerator.SynchronizeAsync();
            }
            else
            {
                // DDIM keeps the host step (alpha-cumprod indexing; not on the SD-Turbo hot path).
                var noisePredCpu = await ReadTensorToCpu(noisePred, noiseData.Length);
                var latentCpu = await ReadTensorToCpu(latentTensor, noiseData.Length);
                float[] updated = DiffusionScheduler.DDIMStep(noisePredCpu, latentCpu, timestepValues[step],
                    step + 1 < timestepValues.Length ? timestepValues[step + 1] : -1, _alphasCumprod!);
                latentTensor.Data.SubView(0, updated.Length).CopyFromCPU(updated);
                await _accelerator.SynchronizeAsync();
            }
        }

        // ═══════════════════════════════════════════════════════════
        //  Step 5: Scale latent for VAE (1 / 0.18215)
        // ═══════════════════════════════════════════════════════════
        const float vaeScaleFactor = 1f / 0.18215f;
        ElementWise.ScaleInPlace(
            latentTensor.Data.SubView(0, noiseData.Length),
            noiseData.Length, vaeScaleFactor);
        await _accelerator.SynchronizeAsync();

        // DIAGNOSTIC (desktop only, opt-in): dump the EXACT post-scale latent fed to our VAE as raw float32,
        // so an ONNX Runtime oracle can decode the identical [1,4,latH,latW] tensor and we can diff its image
        // vs ours — isolates a VAE-op fidelity bug from upstream (UNet/latent). Env: SDTURBO_DUMP_LATENT=path.
        var _dumpLatentPath = Environment.GetEnvironmentVariable("SDTURBO_DUMP_LATENT");
        if (!string.IsNullOrEmpty(_dumpLatentPath))
        {
            var _lat = await ReadTensorToCpu(latentTensor, noiseData.Length);
            var _bytes = new byte[_lat.Length * sizeof(float)];
            Buffer.BlockCopy(_lat, 0, _bytes, 0, _bytes.Length);
            await File.WriteAllBytesAsync(_dumpLatentPath, _bytes);
            Console.WriteLine($"[DUMP] VAE-input latent [1,4,{latentH},{latentW}] ({_lat.Length} floats) -> {_dumpLatentPath}");
        }

        // ═══════════════════════════════════════════════════════════
        //  Step 6: VAE decode → [1, 3, 512, 512] RGB
        // ═══════════════════════════════════════════════════════════
        OnProgress?.Invoke(steps + 1, steps + 2);

        // Measurement aid: reset the (global, cross-session) peak counters right before VAE decode so the
        // reported peak is VAE-ONLY — disambiguates whether the pipeline peak is the UNet or the VAE.
        if (Environment.GetEnvironmentVariable("VAE_PEAK_ONLY") == "1")
            SpawnDev.ILGPU.ML.Tensors.BufferPool.ResetPeaks();
        // Experiment knob: the deferred-release backlog cap (default 512 MiB) sets how many dead buffers stay
        // resident between drains — it DOMINATES the VAE peak. A lower cap trades a few more GPU drains (slightly
        // slower) for a lower peak. VAE_BYTECAP_MB overrides it for the VAE decode.
        long _savedByteCap = Graph.GraphExecutor.MaxPendingReleaseBytes;
        if (int.TryParse(Environment.GetEnvironmentVariable("VAE_BYTECAP_MB"), out var capMb) && capMb > 0)
            Graph.GraphExecutor.MaxPendingReleaseBytes = (long)capMb * 1024 * 1024;

        int imagePixels = Width * Height;                // latentH/latentW computed above (Height/8, Width/8)
        float[] rgbData;                                 // NCHW [3, Height, Width] in [-1,1]

        // Tiled VAE decode (opt-in, for low-VRAM): VAE_TILE_LATENT=N decodes the latent in NxN-ish overlapping
        // tiles, bounding the GPU peak to one tile's decode (the full peak scales with spatial AREA). Default
        // (unset / 0) = the full single-pass decode, unchanged.
        // Exact-stat SEAM-FREE tiled decode (VAE_TILE_EXACT=N → NxN tile grid). Runs the decoder HEAD whole and
        // the UP-BLOCKS tiled with global GroupNorm stats + halo-refreshed convs, so the result is bit-near-
        // identical to the full decode (no seams) at a much lower GPU peak. VAE_EXACT_VERIFY=1 additionally runs
        // the full fp32 decode and reports tiled-vs-full max/mean abs diff (the correctness gate).
        // Effective exact-tile grid: the VAE_TILE_EXACT env var overrides (desktop); else the VaeTileGrid
        // property, where 0 = AUTO (2x2 when the output is >=512 on a side, so the VAE decode's GPU peak is
        // bounded by one tile instead of the full-res up_blocks intermediates that OOM-kill the GPU process),
        // -1 = force the full single-pass decode. Env vars are null in Blazor WASM, so the property is how the
        // browser demo gets tiling.
        int exactTile;
        if (int.TryParse(Environment.GetEnvironmentVariable("VAE_TILE_EXACT"), out var ext)) exactTile = ext;
        else if (VaeTileGrid == -1) exactTile = 0;
        else if (VaeTileGrid > 0) exactTile = VaeTileGrid;
        else exactTile = Math.Max(Width, Height) >= 512 ? 2 : 0;
        bool exactVerify = Environment.GetEnvironmentVariable("VAE_EXACT_VERIFY") == "1";
        if (exactTile > 0 || exactVerify)
        {
            int grid = exactTile > 0 ? exactTile : 2;
            rgbData = await ExactTiledVaeDecodeAsync(latentTensor, 3 * imagePixels, grid, exactVerify);
            Graph.GraphExecutor.MaxPendingReleaseBytes = _savedByteCap;
            goto haveRgb;
        }

        int tileLatent = int.TryParse(Environment.GetEnvironmentVariable("VAE_TILE_LATENT"), out var tl) ? tl : 0;
        if (tileLatent > 0 && tileLatent < Math.Max(latentH, latentW))
        {
            int overlapLatent = int.TryParse(Environment.GetEnvironmentVariable("VAE_TILE_OVERLAP"), out var ov)
                ? ov : Math.Max(2, tileLatent / 2);
            // The genuine working set is what bounds VRAM; tiling shrinks it ONLY if the deferred-release backlog
            // is also bounded (else the byte-cap refills with more small buffers). So tiling auto-lowers the cap
            // (unless VAE_BYTECAP_MB overrode it) — together they cut the SD VAE peak LIVE ~896→450 MiB.
            if (capMb <= 0) Graph.GraphExecutor.MaxPendingReleaseBytes = 96L * 1024 * 1024;
            rgbData = await TiledVaeDecodeAsync(latentTensor, latentH, latentW, tileLatent, overlapLatent);
        }
        else
        {
            // Captured (replay after gen 1). Tiled/diagnostic VAE paths above stay DIRECT - they
            // mutate GraphExecutor break/capture globals mid-run, which a replayed plan ignores.
            var vaeOutputs = await TimedRun("vae_decoder", _vaeDecoder!, VaeCap, new Dictionary<string, Tensor>
                { [_vaeDecoder!.InputNames[0]] = latentTensor });
            rgbData = await ReadTensorToCpu(vaeOutputs[_vaeDecoder.OutputNames[0]], 3 * imagePixels);
        }
        Graph.GraphExecutor.MaxPendingReleaseBytes = _savedByteCap;

        haveRgb:
        // ═══════════════════════════════════════════════════════════
        //  Step 7: Convert NCHW [-1,1] → RGBA [0,255]
        // ═══════════════════════════════════════════════════════════

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

    /// <summary>
    /// Exact-stat SEAM-FREE tiled VAE decode. Phase A runs the decoder HEAD whole (post_quant_conv → conv_in →
    /// mid_block, cheap at 64²) via the session with <see cref="Graph.GraphExecutor.BreakAtNode"/>, capturing the
    /// mid-block output. Phase B runs up_blocks 0-3 + conv_norm_out + conv_out tiled (<see cref="TiledVaeUpDecoder"/>)
    /// with GLOBAL per-group GroupNorm stats + halo-refreshed convs — exact, so no brightness seams (unlike the
    /// approximate <see cref="TiledVaeDecodeAsync"/>), at a GPU peak bounded by one tile. Returns NCHW [3,H,W] in
    /// [-1,1]. When <paramref name="verify"/>, also runs the full fp32 decode and prints tiled-vs-full max/mean
    /// abs diff (the correctness gate) before returning the TILED result.
    /// </summary>
    private async Task<float[]> ExactTiledVaeDecodeAsync(Tensor latent, int outCount, int grid, bool verify)
    {
        var exec = _vaeDecoder!.Executor;
        string inName = _vaeDecoder.InputNames[0], finalName = _vaeDecoder.OutputNames[0];

        // Locate the mid-block output node (Phase-A → Phase-B boundary) in the compiled graph.
        int midIdx = -1;
        for (int i = 0; i < _vaeDecoder.NodeCount; i++)
        {
            var (_, _, outs) = _vaeDecoder.GetNode(i);
            if (outs.Length > 0 && outs[0] == TiledVaeUpDecoder.MidBlockOutputName) { midIdx = i; break; }
        }
        if (midIdx < 0)
            throw new InvalidOperationException($"VAE mid-block output '{TiledVaeUpDecoder.MidBlockOutputName}' not found in graph.");

        // ── Phase A: run the head only (break right after the mid node), capturing just the mid output. ──
        var savedCap = Graph.GraphExecutor.CapturedOutputs;
        var savedNames = Graph.GraphExecutor.CaptureOutputNames;
        var savedBreak = Graph.GraphExecutor.BreakAtNode;
        var savedDtype = exec.ActivationDtype;
        float[] mid;
        try
        {
            exec.ActivationDtype = ActivationPrecision.F32;     // fp32 head → clean (matches the tiled fp32 activations)
            Graph.GraphExecutor.CapturedOutputs = new Dictionary<string, float[]>();
            Graph.GraphExecutor.CaptureOutputNames = new HashSet<string> { TiledVaeUpDecoder.MidBlockOutputName };
            Graph.GraphExecutor.BreakAtNode = midIdx + 1;       // loop breaks when nodeIdx >= this (after running midIdx)
            await _vaeDecoder.RunAsync(new Dictionary<string, Tensor> { [inName] = latent });
            var midKey = Graph.GraphExecutor.CapturedOutputs.Keys.FirstOrDefault(k => k.EndsWith(TiledVaeUpDecoder.MidBlockOutputName))
                ?? throw new InvalidOperationException("Phase-A capture missed the mid-block output.");
            mid = Graph.GraphExecutor.CapturedOutputs[midKey];
        }
        finally
        {
            Graph.GraphExecutor.CapturedOutputs = savedCap;
            Graph.GraphExecutor.CaptureOutputNames = savedNames;
            Graph.GraphExecutor.BreakAtNode = savedBreak;
        }

        // ── Verify reference: capture the full fp32 decode + per-stage boundary tensors FIRST. ──
        Dictionary<string, float[]>? refStages = null;
        float[]? full = null;
        if (verify)
        {
            var boundaryNames = new List<string> { finalName };
            for (int b = 0; b <= 3; b++) for (int r = 0; r <= 2; r++) boundaryNames.Add($"/decoder/up_blocks.{b}/resnets.{r}/Add_output_0");
            for (int b = 0; b <= 2; b++) boundaryNames.Add($"/decoder/up_blocks.{b}/upsamplers.0/conv/Conv_output_0");
            try
            {
                exec.ActivationDtype = ActivationPrecision.F32;
                Graph.GraphExecutor.CapturedOutputs = new Dictionary<string, float[]>();
                Graph.GraphExecutor.CaptureOutputNames = new HashSet<string>(boundaryNames);
                var refOut = await _vaeDecoder.RunAsync(new Dictionary<string, Tensor> { [inName] = latent });
                full = await ReadTensorToCpu(refOut[finalName], outCount);
                refStages = new Dictionary<string, float[]>();
                foreach (var kv in Graph.GraphExecutor.CapturedOutputs)
                {
                    string outName = kv.Key[(kv.Key.IndexOf('_', kv.Key.IndexOf('_') + 1) + 1)..]; // strip "NNN_OpType_"
                    refStages[outName] = kv.Value;
                }
            }
            finally { Graph.GraphExecutor.CapturedOutputs = savedCap; Graph.GraphExecutor.CaptureOutputNames = savedNames; }
        }

        // ── Phase B: tiled up-blocks (compare each stage to the reference when verifying). ──
        float[] tiled;
        using (var dec = new TiledVaeUpDecoder(_vaeDecoder, _accelerator))
        {
            if (verify && refStages != null)
                dec.OnStage = (name, cur) =>
                {
                    if (!refStages.TryGetValue(name, out var refv)) return;
                    double mx = 0, sm = 0; int n = Math.Min(refv.Length, cur.Length);
                    for (int i = 0; i < n; i++) { double d = Math.Abs((double)cur[i] - refv[i]); if (d > mx) mx = d; sm += d; }
                    Console.WriteLine($"[VAE_EXACT]   stage {name} [{cur.Length}]: maxAbs={mx:E3} meanAbs={sm / n:E3}");
                };
            tiled = await dec.DecodeUpBlocksAsync(mid, grid, grid);
        }

        if (verify && full != null)
        {
            double maxAbs = 0, sum = 0; int n = Math.Min(full.Length, tiled.Length);
            for (int i = 0; i < n; i++) { double d = Math.Abs((double)tiled[i] - full[i]); if (d > maxAbs) maxAbs = d; sum += d; }
            Console.WriteLine($"[VAE_EXACT] grid={grid}x{grid} tiled-vs-full(fp32): maxAbs={maxAbs:E4} meanAbs={sum / n:E4} (range [-1,1])");
        }
        exec.ActivationDtype = savedDtype;
        return tiled;
    }

    /// <summary>
    /// Tiled VAE decode: split the latent into overlapping tiles, decode each independently (the
    /// InferenceSession recompiles for the tile shape — pure CPU, weights shared, kernels accelerator-cached),
    /// and linearly blend the overlapping image tiles. Bounds the GPU working set to ONE tile's decode (the full
    /// decode's peak scales with spatial AREA, so e.g. a half-size tile is ~¼ the peak). Per-tile GroupNorm stats
    /// differ slightly from a global decode; the linear ramp blend across the overlap hides the seam (the
    /// standard diffusers VAE-tiling technique). Returns NCHW [3, H, W] in [-1,1], identical layout to the full
    /// decode. Edge tiles get a full (weight-1) contribution on their image-boundary side (no neighbor to blend).
    ///
    /// ⚠ QUALITY TRADEOFF (opt-in, low-VRAM only): per-tile GroupNorm normalizes brightness/contrast over each
    /// tile's content, so tiles can differ in brightness — the linear blend SMOOTHS the transition but cannot
    /// fully remove the mismatch. On SD-Turbo (single-step, less robust) the seams are VISIBLE (measured maxAbs
    /// ~114/255 vs the full decode at 40-latent tiles). The seam-free fix is exact GLOBAL GroupNorm stats, which
    /// is impractical for a memory-bounded decode (it needs the full feature map). So this is an escape hatch for
    /// cards that would otherwise OOM (it cut peak LIVE 896→450 MiB), NOT the default. Larger tiles + more
    /// overlap reduce the seams at the cost of less VRAM saving.
    /// </summary>
    private async Task<float[]> TiledVaeDecodeAsync(Tensor latent, int latentH, int latentW, int tileLatent, int overlapLatent)
    {
        const int up = 8;                                    // VAE upsample factor (latent → image)
        int imgH = latentH * up, imgW = latentW * up;
        var lat = await ReadTensorToCpu(latent, 4 * latentH * latentW);   // [4, latentH, latentW]

        var acc = new float[3 * imgH * imgW];                // weighted RGB accumulator (NCHW)
        var wsum = new float[imgH * imgW];                   // per-pixel weight sum
        var ys = TilePositions(latentH, tileLatent, overlapLatent);
        var xs = TilePositions(latentW, tileLatent, overlapLatent);
        int overlapPx = overlapLatent * up;
        string inName = _vaeDecoder!.InputNames[0], outName = _vaeDecoder.OutputNames[0];

        foreach (int ly in ys)
            foreach (int lx in xs)
            {
                int th = Math.Min(tileLatent, latentH - ly), tw = Math.Min(tileLatent, latentW - lx);
                var tileData = new float[4 * th * tw];
                for (int c = 0; c < 4; c++)
                    for (int yy = 0; yy < th; yy++)
                        for (int xx = 0; xx < tw; xx++)
                            tileData[(c * th + yy) * tw + xx] = lat[(c * latentH + (ly + yy)) * latentW + (lx + xx)];

                using var tileBuf = _accelerator.Allocate1D(tileData);
                var outs = await _vaeDecoder.RunAsync(new Dictionary<string, Tensor>
                    { [inName] = new Tensor(tileBuf.View, new[] { 1, 4, th, tw }, inName) });
                int oh = th * up, ow = tw * up;
                var img = await ReadTensorToCpu(outs[outName], 3 * oh * ow);   // [3, oh, ow]

                int oy = ly * up, ox = lx * up;
                bool blendTop = ly > 0, blendBottom = ly + th < latentH;
                bool blendLeft = lx > 0, blendRight = lx + tw < latentW;
                for (int yy = 0; yy < oh; yy++)
                {
                    float wy = EdgeRamp(yy, oh, overlapPx, blendTop, blendBottom);
                    int gy = oy + yy;
                    for (int xx = 0; xx < ow; xx++)
                    {
                        float w = wy * EdgeRamp(xx, ow, overlapPx, blendLeft, blendRight);
                        int gx = ox + xx, gi = gy * imgW + gx;
                        wsum[gi] += w;
                        for (int c = 0; c < 3; c++)
                            acc[(c * imgH + gy) * imgW + gx] += w * img[(c * oh + yy) * ow + xx];
                    }
                }
            }

        var rgb = new float[3 * imgH * imgW];
        for (int p = 0; p < imgH * imgW; p++)
        {
            float w = wsum[p] > 0 ? wsum[p] : 1f;
            for (int c = 0; c < 3; c++) rgb[c * imgH * imgW + p] = acc[c * imgH * imgW + p] / w;
        }
        return rgb;
    }

    /// <summary>Overlapping tile start positions covering [0,size): stride = tile-overlap, with the last tile
    /// flush to the edge so the full extent is covered.</summary>
    private static List<int> TilePositions(int size, int tile, int overlap)
    {
        if (tile >= size) return new List<int> { 0 };
        int stride = Math.Max(1, tile - overlap);
        var ps = new List<int>();
        for (int p = 0; p + tile < size; p += stride) ps.Add(p);
        if (ps.Count == 0 || ps[^1] != size - tile) ps.Add(size - tile);
        return ps;
    }

    /// <summary>Linear blend weight along one axis: 1 in the tile interior, ramping 0→1 over the leading overlap
    /// (when there's a previous tile to blend with) and 1→0 over the trailing overlap (when there's a next tile).
    /// Image-boundary edges (blendStart/blendEnd false) stay at weight 1.</summary>
    private static float EdgeRamp(int i, int len, int overlapPx, bool blendStart, bool blendEnd)
    {
        float w = 1f;
        if (blendStart && overlapPx > 0 && i < overlapPx) w *= (i + 0.5f) / overlapPx;
        if (blendEnd && overlapPx > 0 && i >= len - overlapPx) w *= (len - i - 0.5f) / overlapPx;
        return w;
    }

    // Per-sub-model perf decomposition (SDTURBO_NODE_TIMING=1): wall time + the GraphExecutor
    // readback/drain split (the DAv3 attribution tool) so we SEE whether the WebGPU generation cost is
    // per-node shape readbacks, sync drains, or raw dispatch/compute — not guess it. No-op unless set.
    // The readback/drain COUNTS are graph-structural (identical on CUDA and WebGPU); only the per-op MS
    // differs by backend, so a CUDA run already reveals the shape of the problem.
    private static readonly bool _nodeTiming = Environment.GetEnvironmentVariable("SDTURBO_NODE_TIMING") == "1";
    private static async Task<Dictionary<string, Tensor>> TimedRun(string tag, InferenceSession sess,
        Graph.SessionGraphCapture cap, Dictionary<string, Tensor> inputs)
    {
        if (!_nodeTiming) return await cap.RunAsync(inputs);
        var sw = System.Diagnostics.Stopwatch.StartNew();
        var outs = await cap.RunAsync(inputs);
        sw.Stop();
        Console.WriteLine($"[NODETIME] {tag,-12} nodes={sess.NodeCount,4} wall={sw.Elapsed.TotalMilliseconds,9:F1}ms" +
            $" | readback {Graph.GraphExecutor.LastRunReadbackCount,4}x {Graph.GraphExecutor.LastRunReadbackMs,8:F1}ms" +
            $" | drain {Graph.GraphExecutor.LastRunSyncDrainCount,4}x {Graph.GraphExecutor.LastRunSyncDrainMs,8:F1}ms" +
            $" | shapeResolved={Graph.GraphExecutor.LastRunShapeInterpResolved} execTotal={Graph.GraphExecutor.LastRunTotalMs:F1}ms");
        return outs;
    }

    private async Task<float[]> ReadTensorToCpu(Tensor tensor, int count)
    {
        using var readBuf = _accelerator.Allocate1D<float>(count);
        ElementWise.Scale(
            tensor.Data.SubView(0, count), readBuf.View, count, 1f);
        await _accelerator.SynchronizeAsync();
        return await readBuf.CopyToHostAsync<float>(0, count);
    }

    public void Dispose()
    {
        _textEncoderCap?.Dispose();
        _unetCap?.Dispose();
        _vaeCap?.Dispose();
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
