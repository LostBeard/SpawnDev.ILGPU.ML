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

        Report("text encoder", 0, 1);
        var (textRows, nTxt) = text ?? await EncodeTextAsync(request.Caption);

        var latents = request.Noise?.ToArray() ?? Flux2Pipeline.GaussianNoise(nImg * Flux3Config.InChannels, request.Seed);
        var imageIds = ImageIds(rows, cols);
        var textIds = new float[nTxt * 4]; for (int i = 0; i < nTxt; i++) textIds[i * 4 + 3] = i;
        var schedule = Schedule(request.Steps, request.Alpha);
        Report("transformer", 0, 1);
        using (var w = await _ditWeights())
        using (var tr = new Flux3Transformer(_acc, w, _config, nImg, 0, nTxt))
        {
            for (int i = 0; i < request.Steps; i++)
            {
                Report("denoise", i, request.Steps);
                tr.Forward(latents, Array.Empty<float>(), textRows, (float)schedule[i], (float)schedule[i + 1], imageIds, Array.Empty<float>(), textIds);
                await _acc.SynchronizeAsync();
                var velocity = await tr.Output.SubView(0, (long)nImg * Flux3Config.InChannels).CopyToHostAsync();
                float dt = (float)(schedule[i + 1] - schedule[i]);
                for (int k = 0; k < latents.Length; k++) latents[k] += dt * velocity[k];
            }
        }
        Report("vae", 0, 1);
        return await _vae.DecodeAsync(_vae.ToVaeInput(latents, rows, cols), rows, cols);
    }

    public void Dispose() => _vae.Dispose();
}
