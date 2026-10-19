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
    /// <summary>The stage files for 4-bit (default, 4.4 GB, 2.6 GB on the GPU at a time) or 8-bit weights (near lossless, 7.9 GB, 4.4 GB at a time: a desktop GPU's choice).</summary>
    public static (string Encoder, string Transformer) StageFiles(int bits) => bits == 8 ? ("text_encoder_q8.safetensors", "transformer_q8.safetensors") : (EncoderFile, TransformerFile);

    /// <summary>
    /// Build the pipeline from a model source (the 4-bit export of black-forest-labs/FLUX.2-klein-4B: <see cref="EncoderFile"/>, <see cref="TransformerFile"/>, <see cref="VaeFile"/>, <see cref="TokenizerFile"/>, <see cref="NormFile"/>).
    /// Only the tokenizer, the batch-norm statistics and the small VAE decoder are loaded now (the VAE stays resident, ~200 MB); the two big stages stream in when a run needs them and leave when it is done with them.
    /// <paramref name="size"/> is the picture side the VAE graph is built for.
    /// </summary>
    public static async Task<Flux2Pipeline> CreateAsync(Accelerator accelerator, IModelSource source, string repo, int size = 512, Action<string, int>? onProgress = null, CancellationToken cancellationToken = default, int bits = 4)
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
        var (encoderFile, transformerFile) = StageFiles(bits);
        return new Flux2Pipeline(accelerator, tok,
            () => Flux2Weights.LoadAsync(accelerator, source, repo, encoderFile, cancellationToken),
            () => Flux2Weights.LoadAsync(accelerator, source, repo, transformerFile, cancellationToken), vae, norm) { VaeTileLatent = lat };
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
        if (lat > VaeTileLatent) return await DecodeTiledAsync(vaeInput, lat, image);
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

    /// <summary>
    /// Side of the latent square the VAE graph is built for (64 = 512 px pictures). Larger pictures are decoded in overlapping tiles of this size, so the decoder's memory (about 1.2 GB at 512 px, four times that
    /// at 1024) never grows with the picture: what lets a browser tab draw 1024 px. Set by <see cref="CreateAsync"/> from its size argument.
    /// </summary>
    public int VaeTileLatent { get; set; } = 64;

    /// <summary>Latent cells of overlap between neighbouring tiles (16 = 128 px), feathered linearly so no seam shows.</summary>
    public int VaeTileOverlap { get; set; } = 16;

    private async Task<Flux2Image> DecodeTiledAsync(float[] vaeInput, int lat, float[]? image)
    {
        int T = VaeTileLatent, O = Math.Min(VaeTileOverlap, T / 2), stride = T - O, size = lat * 8, tilePx = T * 8, ramp = O * 8;
        var sum = new float[3 * size * size]; var wsum = new float[size * size];
        var starts = new List<int>(); for (int x = 0; x + T < lat; x += stride) starts.Add(x); starts.Add(lat - T);
        var tileIn = new float[32 * T * T];
        foreach (int y0 in starts)
            foreach (int x0 in starts)
            {
                for (int c = 0; c < 32; c++)
                    for (int y = 0; y < T; y++)
                        Array.Copy(vaeInput, ((long)c * lat + y0 + y) * lat + x0, tileIn, (c * T + y) * T, T);
                using var latBuf = _acc.Allocate1D(tileIn);
                var outputs = await _vae.RunAsync(new Dictionary<string, Tensor> { [_vae.InputNames[0]] = new Tensor(latBuf.View, new[] { 1, 32, T, T }, _vae.InputNames[0]) });
                var tile = await outputs[_vae.OutputNames[0]].Data.SubView(0, 3L * tilePx * tilePx).CopyToHostAsync();
                int px0 = x0 * 8, py0 = y0 * 8;
                for (int y = 0; y < tilePx; y++)
                {
                    float wy = Feather(y, tilePx, ramp, y0 == 0, y0 == lat - T);
                    for (int x = 0; x < tilePx; x++)
                    {
                        float w = wy * Feather(x, tilePx, ramp, x0 == 0, x0 == lat - T);
                        long o = (long)(py0 + y) * size + px0 + x;
                        wsum[o] += w;
                        for (int c = 0; c < 3; c++) sum[c * (long)size * size + o] += w * tile[((long)c * tilePx + y) * tilePx + x];
                    }
                }
            }
        long pixels = (long)size * size;
        var rgba = new byte[4 * pixels];
        for (long i = 0; i < pixels; i++)
        {
            float inv = 1f / wsum[i];
            for (int c = 0; c < 3; c++)
            {
                float v = sum[c * pixels + i] * inv;
                if (image != null) image[c * pixels + i] = v;
                rgba[i * 4 + c] = (byte)Math.Clamp((int)((v + 1f) * 0.5f * 255f + 0.5f), 0, 255);
            }
            rgba[i * 4 + 3] = 255;
        }
        return new Flux2Image { Width = size, Height = size, Rgba = rgba };
    }

    /// <summary>Blend weight of pixel <paramref name="i"/> of a tile: a linear ramp over <paramref name="ramp"/> pixels at each side that has a neighbour, 1 elsewhere.</summary>
    private static float Feather(int i, int n, int ramp, bool first, bool last)
    {
        float w = 1f;
        if (!first && i < ramp) w = Math.Min(w, (i + 1f) / (ramp + 1f));
        if (!last && n - 1 - i < ramp) w = Math.Min(w, (n - i) / (ramp + 1f));
        return w;
    }

    public async Task<Flux2Image> RunAsync(Flux2Request request) => (await RunManyAsync(request, new[] { request.Seed }))[0];

    /// <summary>
    /// Draw one prompt with several seeds: the prompt is encoded once and the transformer's weights stay on the device for every seed, so each extra picture costs only its denoise steps and a VAE
    /// decode (a few seconds on a desktop GPU) instead of another 4 GB of weight loading. <paramref name="onImage"/> is called with (index, picture) as each one is finished. <see cref="Flux2Request.Seed"/> is ignored;
    /// <see cref="Flux2Request.Noise"/> applies to a single seed only.
    /// </summary>
    public async Task<List<Flux2Image>> RunManyAsync(Flux2Request request, int[] seeds, Func<int, Flux2Image, Task>? onImage = null)
    {
        if (request.Size % 16 != 0 || request.Size < 64) throw new ArgumentException("Size must be a multiple of 16");
        if (seeds.Length == 0) throw new ArgumentException("at least one seed");
        if (request.Noise != null && seeds.Length != 1) throw new ArgumentException("Noise can only be given for a single seed");
        int side = request.Size / 16, nImg = side * side, nTxt = Flux2TextEncoder.SeqLen;
        void Report(string s, int i, int n) => request.Progress?.Invoke(s, i, n);

        // 1. text (the embeddings of the last prompt are kept, so drawing the same prompt again never loads the encoder)
        float[] embeds;
        if (_cachedEmbeds != null && _cachedPrompt == request.Prompt) embeds = _cachedEmbeds;
        else
        {
            Report("text encoder", 0, 1);
            var (ids, real) = _tokenizer.EncodePadded(request.Prompt);
            using (var w = await _encoderWeights())
            using (var enc = new Flux2TextEncoder(_acc, w))
            {
                enc.Encode(ids, real);
                await _acc.SynchronizeAsync();
                embeds = await enc.Output.CopyToHostAsync();
            }
            _cachedPrompt = request.Prompt; _cachedEmbeds = embeds;
        }

        // 2. denoise and decode each seed; the transformer stays loaded for all of them
        var sigmas = Flux2Scheduler.Sigmas(nImg, request.Steps);
        var imgIds = ImageIds(side); var txtIds = TextIds(nTxt);
        var images = new List<Flux2Image>();
        Flux2Weights tw; Flux2Transformer tr;
        if (_residentTr != null && _residentTokens == nImg) { tw = _residentWeights!; tr = _residentTr; }
        else
        {
            ReleaseResident();
            Report("transformer", 0, 1);
            tw = await _transformerWeights();
            tr = new Flux2Transformer(_acc, tw, nTxt, nImg);
        }
        try
        {
        for (int n = 0; n < seeds.Length; n++)
        {
            var latents = request.Noise?.ToArray() ?? GaussianNoise(nImg * Flux2Transformer.InChannels, seeds[n]);
            for (int i = 0; i < request.Steps; i++)
            {
                Report(seeds.Length > 1 ? $"denoise {n + 1}/{seeds.Length}" : "denoise", i, request.Steps);
                tr.Forward(latents, embeds, Flux2Scheduler.Timestep(sigmas, i), imgIds, txtIds);
                await _acc.SynchronizeAsync();
                var pred = await tr.Output.SubView(0, (long)nImg * Flux2Transformer.InChannels).CopyToHostAsync();
                float dt = (float)(sigmas[i + 1] - sigmas[i]);
                for (int k = 0; k < latents.Length; k++) latents[k] += dt * pred[k];
            }
            Report("vae", n, seeds.Length);
            var image = await DecodeAsync(ToVaeInput(latents, side), side);
            images.Add(image);
            if (onImage != null) await onImage(n, image);
        }
        }
        finally
        {
            if (KeepTransformerResident) { _residentWeights = tw; _residentTr = tr; _residentTokens = nImg; }
            else { if (ReferenceEquals(tr, _residentTr)) { _residentTr = null; _residentWeights = null; } tr.Dispose(); tw.Dispose(); }
        }
        return images;
    }

    /// <summary>
    /// Keep the transformer's weights on the device after a run (the next run of the same size skips the 2.4 GB load: about 60 s in a browser). Off by default because the weights hold GPU memory until
    /// <see cref="ReleaseResident"/> (or <see cref="Dispose"/>) frees them; a page that goes on to run another big model should release first.
    /// </summary>
    public bool KeepTransformerResident { get; set; }

    private Flux2Weights? _residentWeights;
    private Flux2Transformer? _residentTr;
    private int _residentTokens;
    private string? _cachedPrompt;
    private float[]? _cachedEmbeds;

    /// <summary>Free the transformer kept by <see cref="KeepTransformerResident"/> (the cached prompt embeddings stay: they are host memory).</summary>
    public void ReleaseResident()
    {
        _residentTr?.Dispose(); _residentWeights?.Dispose();
        _residentTr = null; _residentWeights = null; _residentTokens = 0;
    }

    /// <summary>Frees the resident VAE decoder and any kept transformer. The accelerator is not touched.</summary>
    public void Dispose() { ReleaseResident(); _vae.Dispose(); _elementWise?.Dispose(); }
}
