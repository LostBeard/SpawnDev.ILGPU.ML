using System.Text.Json;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.Tensors;
using SpawnDev.ILGPU.ML.Trellis2;

namespace SpawnDev.ILGPU.ML.Flux2;

/// <summary>The batch-norm statistics FLUX.2's VAE keeps for its 128-channel packed latents (export_flux2_vae.py writes vae_norm.json).</summary>
public sealed class Flux2VaeNorm
{
    public float[] Mean { get; }
    public float[] Std { get; }

    public Flux2VaeNorm(float[] mean, float[] variance, float eps)
    {
        Mean = mean;
        Std = variance.Select(v => MathF.Sqrt(v + eps)).ToArray();
    }

    public static Flux2VaeNorm FromJson(string json)
    {
        using var doc = JsonDocument.Parse(json);
        var r = doc.RootElement;
        return new Flux2VaeNorm(r.GetProperty("mean").EnumerateArray().Select(x => x.GetSingle()).ToArray(),
                                r.GetProperty("var").EnumerateArray().Select(x => x.GetSingle()).ToArray(), r.GetProperty("eps").GetSingle());
    }
}

/// <summary>One picture from <see cref="Flux2Pipeline"/>.</summary>
public sealed class Flux2Image
{
    public int Width { get; init; }
    public int Height { get; init; }
    /// <summary>RGBA, 8 bits per channel.</summary>
    public byte[] Rgba { get; init; } = Array.Empty<byte>();
}

/// <summary>What to draw.</summary>
public sealed class Flux2Request
{
    public string Prompt { get; init; } = "";
    public int Seed { get; init; }
    /// <summary>Side of the square picture in pixels; a multiple of 16.</summary>
    public int Size { get; init; } = 512;
    /// <summary>Denoising steps; the klein checkpoints are distilled for 4.</summary>
    public int Steps { get; init; } = 4;
    /// <summary>Start latents [(Size/16)^2, 128] to use instead of seeded noise (tests feed the reference's).</summary>
    public float[]? Noise { get; init; }
    /// <summary>Called with (stage, step, steps) as the run moves along.</summary>
    public Action<string, int, int>? Progress { get; init; }
}

/// <summary>
/// FLUX.2 klein 4B text to image: Qwen3-4B encodes the prompt, the 3.9B flow transformer denoises (4 Euler steps from noise), the VAE decodes. The three models never share the GPU:
/// each is loaded, run and disposed in turn (the encoder's and transformer's weights come from the factories, so a caller may stream, quantise or cache them as it likes), which is what lets a
/// 4B model run in a browser tab's memory. Apache-2.0 weights (black-forest-labs/FLUX.2-klein-4B).
/// </summary>
public sealed class Flux2Pipeline : IDisposable
{
    private readonly Accelerator _acc;
    private readonly Qwen2Tokenizer _tokenizer;
    private readonly Func<Task<Flux2Weights>> _encoderWeights, _transformerWeights;
    private readonly InferenceSession _vae;
    private readonly Flux2VaeNorm _norm;
    private ElementWiseKernels? _elementWise;
    private ElementWiseKernels ElementWise => _elementWise ??= new ElementWiseKernels(_acc);

    public Flux2Pipeline(Accelerator accelerator, Qwen2Tokenizer tokenizer, Func<Task<Flux2Weights>> encoderWeights, Func<Task<Flux2Weights>> transformerWeights, InferenceSession vaeDecoder, Flux2VaeNorm vaeNorm)
    {
        _acc = accelerator; _tokenizer = tokenizer; _encoderWeights = encoderWeights; _transformerWeights = transformerWeights; _vae = vaeDecoder; _norm = vaeNorm;
    }

    /// <summary>Files a run reads, as (repo, path): for storage planning (<see cref="RangeStreamModelSource.PlanCacheAsync(IEnumerable{ValueTuple{string, string}}, long, CancellationToken)"/>).</summary>
    public static IReadOnlyList<(string Repo, string Path)> HubFiles(string repo) => new[] { (repo, EncoderFile), (repo, TransformerFile), (repo, VaeFile) };

    public const string EncoderFile = "text_encoder_q4.safetensors", TransformerFile = "transformer_q4.safetensors", VaeFile = "vae_decoder.onnx", TokenizerFile = "tokenizer.json", NormFile = "vae_norm.json";

    /// <summary>
    /// Build the pipeline from a model source (the 4-bit export of black-forest-labs/FLUX.2-klein-4B: <see cref="EncoderFile"/>, <see cref="TransformerFile"/>, <see cref="VaeFile"/>, <see cref="TokenizerFile"/>, <see cref="NormFile"/>).
    /// Only the tokenizer, the batch-norm statistics and the small VAE decoder are loaded now (the VAE stays resident, ~200 MB); the two big stages stream in when a run needs them and leave when it is done with them.
    /// <paramref name="size"/> is the picture side the VAE graph is built for.
    /// </summary>
    public static async Task<Flux2Pipeline> CreateAsync(Accelerator accelerator, IModelSource source, string repo, int size = 512, Action<string, int>? onProgress = null, CancellationToken cancellationToken = default)
    {
        onProgress?.Invoke("tokenizer", 0);
        var tok = new Qwen2Tokenizer(System.Text.Encoding.UTF8.GetString(await source.FetchBytesAsync(repo, TokenizerFile, cancellationToken)));
        var norm = Flux2VaeNorm.FromJson(System.Text.Encoding.UTF8.GetString(await source.FetchBytesAsync(repo, NormFile, cancellationToken)));
        onProgress?.Invoke("tokenizer", 100);
        int lat = size / 8;
        InferenceSession vae;
        await using (var s = await source.OpenAsync(repo, VaeFile, cancellationToken))
            vae = await InferenceSession.CreateFromOnnxStreamAsync(accelerator, s, onProgress: (st, pct) => onProgress?.Invoke("vae:" + st, pct),
                inputShapes: new Dictionary<string, int[]> { ["latents"] = new[] { 1, 32, lat, lat } }, ct: cancellationToken);
        return new Flux2Pipeline(accelerator, tok,
            () => Flux2Weights.LoadAsync(accelerator, source, repo, EncoderFile, cancellationToken),
            () => Flux2Weights.LoadAsync(accelerator, source, repo, TransformerFile, cancellationToken), vae, norm);
    }

    /// <summary>The position ids of the image tokens, [tokens, 4] = (0, row, column, 0), and of the text tokens, (0, 0, 0, index).</summary>
    public static int[] ImageIds(int side)
    {
        var ids = new int[side * side * 4];
        for (int y = 0; y < side; y++)
            for (int x = 0; x < side; x++) { int o = (y * side + x) * 4; ids[o + 1] = y; ids[o + 2] = x; }
        return ids;
    }

    public static int[] TextIds(int count)
    {
        var ids = new int[count * 4];
        for (int i = 0; i < count; i++) ids[i * 4 + 3] = i;
        return ids;
    }

    /// <summary>Standard normal noise from a seeded generator (Box-Muller); not torch's stream, so a seed gives its own picture, not the reference's.</summary>
    public static float[] GaussianNoise(int count, int seed)
    {
        var rng = new Random(seed);
        var n = new float[count];
        for (int i = 0; i < count; i += 2)
        {
            double u1 = 1.0 - rng.NextDouble(), u2 = rng.NextDouble();
            double r = Math.Sqrt(-2.0 * Math.Log(u1));
            n[i] = (float)(r * Math.Cos(2 * Math.PI * u2));
            if (i + 1 < count) n[i + 1] = (float)(r * Math.Sin(2 * Math.PI * u2));
        }
        return n;
    }

    /// <summary>
    /// Packed latents [side*side, 128] back to the VAE's input [32, 2*side, 2*side]: undo the batch-norm (x * std + mean per packed channel), then unpatchify
    /// (channel c*4 + py*2 + px of cell (y, x) becomes channel c at (2y + py, 2x + px)).
    /// </summary>
    public float[] ToVaeInput(float[] packed, int side)
    {
        int cells = side * side, outSide = side * 2;
        var o = new float[32 * outSide * outSide];
        for (int cell = 0; cell < cells; cell++)
        {
            int y = cell / side, x = cell % side;
            for (int ch = 0; ch < 128; ch++)
            {
                float v = packed[cell * 128 + ch] * _norm.Std[ch] + _norm.Mean[ch];
                int c = ch / 4, py = (ch / 2) % 2, px = ch % 2;
                o[((long)c * outSide + 2 * y + py) * outSide + 2 * x + px] = v;
            }
        }
        return o;
    }

    /// <summary>Decode VAE input latents [32, 2*side, 2*side] (<see cref="ToVaeInput"/>) to RGBA; <paramref name="image"/> also receives the raw [-1, 1] planes when given.</summary>
    public async Task<Flux2Image> DecodeAsync(float[] vaeInput, int side, float[]? image = null)
    {
        int lat = side * 2, size = lat * 8, pixels = size * size;
        using var latBuf = _acc.Allocate1D(vaeInput);
        var outputs = await _vae.RunAsync(new Dictionary<string, Tensor> { [_vae.InputNames[0]] = new Tensor(latBuf.View, new[] { 1, 32, lat, lat }, _vae.InputNames[0]) });
        var vaeOut = outputs[_vae.OutputNames[0]];
        if (image != null) vaeOut.Data.SubView(0, 3L * pixels).CopyToCPU(image);
        using var rgbaBuf = _acc.Allocate1D<int>(pixels);
        ElementWise.NchwDenormToRgba(vaeOut.Data.SubView(0, 3 * pixels), rgbaBuf.View, pixels);
        await _acc.SynchronizeAsync();
        var packed = await rgbaBuf.CopyToHostAsync<int>(0, pixels);
        var rgba = new byte[4 * pixels];
        Buffer.BlockCopy(packed, 0, rgba, 0, 4 * pixels);
        return new Flux2Image { Width = size, Height = size, Rgba = rgba };
    }

    public async Task<Flux2Image> RunAsync(Flux2Request request)
    {
        if (request.Size % 16 != 0 || request.Size < 64) throw new ArgumentException("Size must be a multiple of 16");
        int side = request.Size / 16, nImg = side * side, nTxt = Flux2TextEncoder.SeqLen;
        void Report(string s, int i, int n) => request.Progress?.Invoke(s, i, n);

        // 1. text
        Report("text encoder", 0, 1);
        var (ids, real) = _tokenizer.EncodePadded(request.Prompt);
        float[] embeds;
        using (var w = await _encoderWeights())
        using (var enc = new Flux2TextEncoder(_acc, w))
        {
            enc.Encode(ids, real);
            await _acc.SynchronizeAsync();
            embeds = await enc.Output.CopyToHostAsync();
        }

        // 2. denoise
        var latents = request.Noise?.ToArray() ?? GaussianNoise(nImg * Flux2Transformer.InChannels, request.Seed);
        var sigmas = Flux2Scheduler.Sigmas(nImg, request.Steps);
        var imgIds = ImageIds(side); var txtIds = TextIds(nTxt);
        using (var w = await _transformerWeights())
        using (var tr = new Flux2Transformer(_acc, w, nTxt, nImg))
        {
            for (int i = 0; i < request.Steps; i++)
            {
                Report("denoise", i, request.Steps);
                tr.Forward(latents, embeds, Flux2Scheduler.Timestep(sigmas, i), imgIds, txtIds);
                await _acc.SynchronizeAsync();
                var pred = await tr.Output.SubView(0, (long)nImg * Flux2Transformer.InChannels).CopyToHostAsync();
                float dt = (float)(sigmas[i + 1] - sigmas[i]);
                for (int k = 0; k < latents.Length; k++) latents[k] += dt * pred[k];
            }
        }

        // 3. decode
        Report("vae", 0, 1);
        return await DecodeAsync(ToVaeInput(latents, side), side);
    }

    /// <summary>Frees the resident VAE decoder. The accelerator is not touched.</summary>
    public void Dispose() { _vae.Dispose(); _elementWise?.Dispose(); }
}
