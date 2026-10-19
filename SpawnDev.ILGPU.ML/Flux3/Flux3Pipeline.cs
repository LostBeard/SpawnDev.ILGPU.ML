using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Flux2;
using SpawnDev.ILGPU.ML.Hub;

namespace SpawnDev.ILGPU.ML.Flux3;

/// <summary>What to draw with FLUX 3.</summary>
public sealed class Flux3Request
{
    /// <summary>The caption the DiT is conditioned on. The model is trained on full scene captions (the authors' prompt upsampler writes them): a short prompt gives generic pictures.</summary>
    public string Caption { get; init; } = "";
    public int Seed { get; init; }
    /// <summary>Picture width and height in pixels, multiples of 16, 64..4096 (0 = 1024).</summary>
    public int Width { get; init; }
    public int Height { get; init; }
    /// <summary>Euler steps of the distilled flow-map model (16).</summary>
    public int Steps { get; init; } = 16;
    /// <summary>The schedule's shift: t = a * s / (1 + (a - 1) s). The reference code samples with 16.</summary>
    public float Alpha { get; init; } = 16f;
    /// <summary>Start latents [(Height/16) * (Width/16), 128] to use instead of seeded noise (tests feed the reference's).</summary>
    public float[]? Noise { get; init; }
    /// <summary>Up to <see cref="Flux3Pipeline.MaxReferences"/> pictures the result is conditioned on (edits, compositions): each is prepared like the reference code does (above 1024 x 1024 pixels it is shrunk, then centre-cropped to multiples of 16), encoded by the
    /// autoencoder and fed to the transformer as <c>image_cond</c> tokens with the position id l = -(1 + i). Needs the VAE encoder (<see cref="Flux3Pipeline.VaeEncoderFactory"/>).</summary>
    public IReadOnlyList<Flux2ReferenceImage>? References { get; init; }
    /// <summary>Called with (stage, step, steps) as the run moves along.</summary>
    public Action<string, int, int>? Progress { get; init; }
}

/// <summary>
/// FLUX 3 Image Klein (distilled, flow map) text to image: the Qwen3-VL text tower encodes the caption, the 5.6B flow-map transformer integrates 16 Euler steps from noise
/// (<see cref="Flux3Transformer"/>), the 16x autoencoder decodes. The three models never share the GPU: each is loaded, run and disposed in turn. Code Apache-2.0; weights under the FLUX Kommunity License.
/// </summary>
public sealed class Flux3Pipeline : IDisposable
{
    private readonly Accelerator _acc;
    private readonly Qwen2Tokenizer _tokenizer;
    private readonly Func<Task<Flux2Weights>> _textWeights, _ditWeights;
    private readonly Flux2Pipeline _vae;       // the autoencoder is the FLUX.2 one in layout: decoder graph, batch-norm statistics, tiled decode (its own weights)
    private readonly Flux3Config _config;

    public Flux3Pipeline(Accelerator accelerator, Qwen2Tokenizer tokenizer, Func<Task<Flux2Weights>> textEncoderWeights, Func<Task<Flux2Weights>> transformerWeights,
        InferenceSession vaeDecoder, Flux2VaeNorm vaeNorm, Flux3Config? config = null)
    {
        _acc = accelerator; _tokenizer = tokenizer; _textWeights = textEncoderWeights; _ditWeights = transformerWeights; _config = config ?? new Flux3Config();
        _vae = new Flux2Pipeline(accelerator, null!, null!, null!, vaeDecoder, vaeNorm);
    }

    /// <summary>The most reference pictures one request takes.</summary>
    public const int MaxReferences = 10;

    /// <summary>Builds the VAE encoder's session for a picture of (height, width) pixels (the graph is fixed-shape): needed for <see cref="Flux3Request.References"/>. <see cref="CreateAsync"/> sets it; null = references unsupported.</summary>
    public Func<int, int, Task<InferenceSession>>? VaeEncoderFactory { get => _vae.VaeEncoderFactory; set => _vae.VaeEncoderFactory = value; }

    public const string TokenizerFile = "tokenizer.json", NormFile = "vae_norm.json", VaeFile = "vae_decoder.onnx", VaeEncoderFile = "vae_encoder.onnx";

    /// <summary>The stage files for 4-bit (default: 5.7 GB in all, 3.5 GB on the GPU at a time) or 8-bit weights (near lossless, 10.3 GB, 6.2 GB at a time: a desktop GPU's choice).</summary>
    public static (string Encoder, string Transformer) StageFiles(int bits) => bits == 8 ? ("text_encoder_q8.safetensors", "transformer_q8.safetensors") : ("text_encoder_q4.safetensors", "transformer_q4.safetensors");

    /// <summary>Files a run reads, as (repo, path): for storage planning (<see cref="RangeStreamModelSource.PlanCacheAsync(IEnumerable{ValueTuple{string, string}}, long, CancellationToken)"/>).</summary>
    public static IReadOnlyList<(string Repo, string Path)> HubFiles(string repo, bool references = false, int bits = 4)
    {
        var (enc, tr) = StageFiles(bits);
        return references ? new[] { (repo, enc), (repo, tr), (repo, VaeFile), (repo, VaeEncoderFile) } : new[] { (repo, enc), (repo, tr), (repo, VaeFile) };
    }

    /// <summary>
    /// Build the pipeline from a model source holding the export (<c>LostBeard/flux-3-image-klein-q4-ILGPU</c> layout: the stage files, <see cref="VaeFile"/>, <see cref="NormFile"/>, <see cref="TokenizerFile"/>).
    /// Only the tokenizer, the batch-norm statistics and the small decoder are loaded now (the decoder stays resident, ~200 MB); the two big stages stream in when a run needs them and leave when it is done with them.
    /// </summary>
    public static async Task<Flux3Pipeline> CreateAsync(Accelerator accelerator, IModelSource source, string repo, Action<string, int>? onProgress = null, CancellationToken cancellationToken = default, int bits = 4)
    {
        onProgress?.Invoke("tokenizer", 0);
        var tok = new Qwen2Tokenizer(System.Text.Encoding.UTF8.GetString(await source.FetchBytesAsync(repo, TokenizerFile, cancellationToken)));
        var norm = Flux2VaeNorm.FromJson(System.Text.Encoding.UTF8.GetString(await source.FetchBytesAsync(repo, NormFile, cancellationToken)));
        onProgress?.Invoke("tokenizer", 100);
        InferenceSession vae;
        await using (var s = await source.OpenAsync(repo, VaeFile, cancellationToken))
            vae = await InferenceSession.CreateFromOnnxStreamAsync(accelerator, s, onProgress: (st, pct) => onProgress?.Invoke("vae:" + st, pct),
                inputShapes: new Dictionary<string, int[]> { ["latents"] = new[] { 1, 32, 64, 64 } }, ct: cancellationToken);
        var (encoderFile, transformerFile) = StageFiles(bits);
        var pipe = new Flux3Pipeline(accelerator, tok,
            () => Flux2Weights.LoadAsync(accelerator, source, repo, encoderFile, cancellationToken),
            () => Flux2Weights.LoadAsync(accelerator, source, repo, transformerFile, cancellationToken), vae, norm);
        pipe.VaeEncoderFactory = async (h, w) =>
        {
            await using var es = await source.OpenAsync(repo, VaeEncoderFile, cancellationToken);
            return await InferenceSession.CreateFromOnnxStreamAsync(accelerator, es, inputShapes: new Dictionary<string, int[]> { ["image"] = new[] { 1, 3, h, w } }, ct: cancellationToken);
        };
        return pipe;
    }

    /// <summary>
    /// The reference pictures as <c>image_cond</c> inputs: their autoencoder tokens one after the other [N, 128] and their position ids [N, 4] = (0, row / 2, column / 2, -(1 + i)).
    /// </summary>
    public async Task<(float[] Tokens, float[] Ids)> EncodeReferencesAsync(IReadOnlyList<Flux2ReferenceImage>? references, Action<int, int>? progress = null)
    {
        if (references == null || references.Count == 0) return (Array.Empty<float>(), Array.Empty<float>());
        if (references.Count > MaxReferences) throw new ArgumentException($"At most {MaxReferences} reference images");
        var tokens = new List<float>(); var ids = new List<float>();
        for (int r = 0; r < references.Count; r++)
        {
            progress?.Invoke(r, references.Count);
            var (t, rows, cols) = await _vae.EncodeReferenceAsync(references[r]);
            tokens.AddRange(t); ids.AddRange(ImageIds(rows, cols, -(1 + r)));
        }
        return (tokens.ToArray(), ids.ToArray());
    }

    /// <summary>The Qwen3-VL chat template for a caption: user turn, generation prompt, no thinking block.</summary>
    public static string ChatPrompt(string caption) => "<|im_start|>user\n" + caption + "<|im_end|>\n<|im_start|>assistant\n";

    /// <summary>Token ids of the chat-templated caption padded on the right with the pad token to a multiple of 80 (at most 4096), and the number of real tokens.</summary>
    public (int[] Ids, int Real) Tokenize(string caption)
    {
        var ids = _tokenizer.Encode(ChatPrompt(caption));
        int real = Math.Min(ids.Length, Flux3TextEncoder.MaxTokens);
        int len = Flux3TextEncoder.PaddedLength(real);
        var padded = new int[len];
        for (int i = 0; i < len; i++) padded[i] = i < real ? ids[i] : _tokenizer.PadId;
        return (padded, real);
    }

    /// <summary>Position ids of a rows x columns grid of picture tokens as the reference builds them: (0, row / 2, column / 2, 0).</summary>
    public static float[] ImageIds(int rows, int columns, float l = 0f)
    {
        var ids = new float[rows * columns * 4];
        for (int y = 0; y < rows; y++)
            for (int x = 0; x < columns; x++) { int o = (y * columns + x) * 4; ids[o + 1] = y * 0.5f; ids[o + 2] = x * 0.5f; ids[o + 3] = l; }
        return ids;
    }

    /// <summary>The Euler schedule: t_i = a * s_i / (1 + (a - 1) s_i) with s linear from 1 to 0 (steps + 1 values).</summary>
    public static double[] Schedule(int steps, double alpha)
    {
        var t = new double[steps + 1];
        for (int i = 0; i <= steps; i++) { double s = 1.0 - (double)i / steps; t[i] = alpha * s / (1.0 + (alpha - 1.0) * s); }
        return t;
    }

    private string? _cachedCaption;
    private (float[] Rows, int Tokens)? _cachedText;
    private Flux2ReferenceImage[]? _cachedRefs;
    private (float[] Tokens, float[] Ids) _cachedRefTokens;

    private async Task<(float[] Rows, int Tokens)> CacheTextAsync(string caption)
    {
        var encoded = await EncodeTextAsync(caption);
        _cachedCaption = caption; _cachedText = encoded;
        return encoded;
    }

    /// <summary>The caption's conditioning rows [tokens, 20480] and the token count (the text encoder is loaded for the call and freed after it).</summary>
    public async Task<(float[] Rows, int Tokens)> EncodeTextAsync(string caption)
    {
        var (ids, real) = Tokenize(caption);
        using var w = await _textWeights();
        using var enc = new Flux3TextEncoder(_acc, w, ids.Length);
        enc.Encode(ids, real);
        await _acc.SynchronizeAsync();
        return (await enc.Output.SubView(0, (long)ids.Length * Flux3TextEncoder.EmbedWidth).CopyToHostAsync(), ids.Length);
    }

    public Task<Flux2Image> RunAsync(Flux3Request request) => RunAsync(request, null);

    /// <summary>Draw with precomputed text rows (tests feed the reference's); null encodes the caption.</summary>
    public async Task<Flux2Image> RunAsync(Flux3Request request, (float[] Rows, int Tokens)? text)
    {
        int width = request.Width > 0 ? request.Width : 1024, height = request.Height > 0 ? request.Height : 1024;
        if (width % 16 != 0 || height % 16 != 0 || width < 64 || height < 64 || width > 4096 || height > 4096) throw new ArgumentException("Width and height must be multiples of 16 between 64 and 4096");
        int rows = height / 16, cols = width / 16, nImg = rows * cols;
        void Report(string s, int i, int n) => request.Progress?.Invoke(s, i, n);

        var clock = System.Diagnostics.Stopwatch.StartNew(); var timings = new List<(string Stage, double Seconds)>(); double mark = 0;
        void Lap(string stage) { double now = clock.Elapsed.TotalSeconds; timings.Add((stage, now - mark)); mark = now; }
        Report("text encoder", 0, 1);
        // the conditioning of the last caption is kept (host memory, 6.5 MB for 80 tokens): further pictures of the same caption (other seeds) skip loading and running the 2.3 GB text encoder
        var (textRows, nTxt) = text ?? (_cachedCaption == request.Caption && _cachedText != null ? _cachedText.Value : await CacheTextAsync(request.Caption));

        // likewise the reference pictures' tokens, while the same picture objects are passed
        (float[] condRows, float[] condIdArray) refs;
        if (request.References != null && _cachedRefs != null && request.References.Count == _cachedRefs.Length && request.References.Zip(_cachedRefs, ReferenceEquals).All(x => x)) refs = _cachedRefTokens;
        else { refs = await EncodeReferencesAsync(request.References, (i, n) => Report("reference", i, n)); _cachedRefs = request.References?.ToArray(); _cachedRefTokens = refs; }
        var (condRows, condIdArray) = refs;
        Lap("text + references");
        int nCond = condRows.Length / Flux3Config.InChannels;

        var latents = request.Noise?.ToArray() ?? Flux2Pipeline.GaussianNoise(nImg * Flux3Config.InChannels, request.Seed);
        var imageIds = ImageIds(rows, cols);
        var textIds = new float[nTxt * 4]; for (int i = 0; i < nTxt; i++) textIds[i * 4 + 3] = i;
        var schedule = Schedule(request.Steps, request.Alpha);
        Report("transformer", 0, 1);
        var w = _resident ?? await _ditWeights();
        Lap("transformer weights (load)");
        try
        {
            if (KeepTransformerResident) _resident = w;
            using var tr = new Flux3Transformer(_acc, w, _config, nImg, nCond, nTxt);
            // the caption, references and rotary tables go up once; the latents stay on the device through every step (an Euler kernel updates them): nothing crosses to the CPU until the picture is done
            tr.Prepare(condRows, textRows, imageIds, condIdArray, textIds);
            tr.UploadLatents(latents);
            for (int i = 0; i < request.Steps; i++)
            {
                Report("denoise", i, request.Steps);
                tr.Step((float)schedule[i], (float)schedule[i + 1]);
                tr.Euler((float)(schedule[i + 1] - schedule[i]));
                await _acc.SynchronizeAsync();                       // a wait, not a read-back: paces the command queue and lets the page report the step
            }
            latents = await tr.Latents.CopyToHostAsync();
            Lap($"{request.Steps} steps");
        }
        finally
        {
            if (!KeepTransformerResident) { if (ReferenceEquals(w, _resident)) _resident = null; w.Dispose(); }
        }
        Report("vae", 0, 1);
        var picture = await _vae.DecodeAsync(_vae.ToVaeInput(latents, rows, cols), rows, cols);
        Lap("vae decode");
        LastTimings = timings;
        return picture;
    }

    /// <summary>Seconds each stage of the last <see cref="RunAsync(Flux3Request, ValueTuple{float[], int}?)"/> took (text and references: 0 when cached; transformer load: 0 when resident; the steps; the VAE decode).</summary>
    public IReadOnlyList<(string Stage, double Seconds)> LastTimings { get; private set; } = Array.Empty<(string, double)>();

    /// <summary>
    /// Keep the transformer's weights on the device after a run, so the next picture skips the load (3.5 GB at 4 bits: about a minute in a browser). Off by default because the weights hold GPU memory until
    /// <see cref="ReleaseResident"/> (or <see cref="Dispose"/>) frees them.
    /// </summary>
    public bool KeepTransformerResident { get; set; }

    private Flux2Weights? _resident;

    /// <summary>Free the transformer weights kept by <see cref="KeepTransformerResident"/>.</summary>
    public void ReleaseResident() { _resident?.Dispose(); _resident = null; }

    public void Dispose() { ReleaseResident(); _vae.Dispose(); }
}
