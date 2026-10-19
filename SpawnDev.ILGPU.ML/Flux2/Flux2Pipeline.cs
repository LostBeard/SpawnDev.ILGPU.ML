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
    /// <summary>Side of the square picture in pixels; a multiple of 16. Ignored when <see cref="References"/> are given (the picture then has the first reference's size) or <see cref="Width"/> and <see cref="Height"/> are set.</summary>
    public int Size { get; init; } = 512;
    /// <summary>Picture width in pixels (a multiple of 16); 0 = <see cref="Size"/>, or the first reference's width.</summary>
    public int Width { get; init; }
    /// <summary>Picture height in pixels (a multiple of 16); 0 = <see cref="Size"/>, or the first reference's height.</summary>
    public int Height { get; init; }
    /// <summary>
    /// Pictures to edit or compose from (up to 10; the model is trained for several). Each is encoded by the VAE and its tokens are appended to the picture's own in the transformer's sequence with their own
    /// position (t = 10 + 10 i); the prompt then says what to do ("give the knight a red plume"). Needs the VAE encoder (<see cref="Flux2Pipeline.VaeEncoderFile"/>).
    /// </summary>
    public IReadOnlyList<Flux2ReferenceImage>? References { get; init; }
    /// <summary>Denoising steps; the klein checkpoints are distilled for 4.</summary>
    public int Steps { get; init; } = 4;
    /// <summary>Start latents [(Height/16) * (Width/16), 128] to use instead of seeded noise (tests feed the reference's).</summary>
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
    /// <summary>Builds the VAE encoder's session for a picture of (height, width) pixels (the graph is fixed-shape): needed for <see cref="Flux2Request.References"/>. <see cref="CreateAsync"/> sets it; null = references unsupported.</summary>
    public Func<int, int, Task<InferenceSession>>? VaeEncoderFactory { get; set; }
    private ElementWiseKernels? _elementWise;
    private ElementWiseKernels ElementWise => _elementWise ??= new ElementWiseKernels(_acc);

    public Flux2Pipeline(Accelerator accelerator, Qwen2Tokenizer tokenizer, Func<Task<Flux2Weights>> encoderWeights, Func<Task<Flux2Weights>> transformerWeights, InferenceSession vaeDecoder, Flux2VaeNorm vaeNorm)
    {
        _acc = accelerator; _tokenizer = tokenizer; _encoderWeights = encoderWeights; _transformerWeights = transformerWeights; _vae = vaeDecoder; _norm = vaeNorm;
    }

    /// <summary>Files a run reads, as (repo, path): for storage planning (<see cref="RangeStreamModelSource.PlanCacheAsync(IEnumerable{ValueTuple{string, string}}, long, CancellationToken)"/>).</summary>
    public static IReadOnlyList<(string Repo, string Path)> HubFiles(string repo, bool references = false)
        => references ? new[] { (repo, EncoderFile), (repo, TransformerFile), (repo, VaeFile), (repo, VaeEncoderFile) } : new[] { (repo, EncoderFile), (repo, TransformerFile), (repo, VaeFile) };

    public const string EncoderFile = "text_encoder_q4.safetensors", TransformerFile = "transformer_q4.safetensors", VaeFile = "vae_decoder.onnx", VaeEncoderFile = "vae_encoder.onnx", TokenizerFile = "tokenizer.json", NormFile = "vae_norm.json";
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
        Func<int, int, Task<InferenceSession>> encoderFactory = async (h, w) =>
        {
            await using var es = await source.OpenAsync(repo, VaeEncoderFile, cancellationToken);
            return await InferenceSession.CreateFromOnnxStreamAsync(accelerator, es, inputShapes: new Dictionary<string, int[]> { ["image"] = new[] { 1, 3, h, w } }, ct: cancellationToken);
        };
        return new Flux2Pipeline(accelerator, tok,
            () => Flux2Weights.LoadAsync(accelerator, source, repo, encoderFile, cancellationToken),
            () => Flux2Weights.LoadAsync(accelerator, source, repo, transformerFile, cancellationToken), vae, norm) { VaeTileLatent = lat, VaeEncoderFactory = encoderFactory };
    }

    /// <summary>The position ids of the image tokens, [tokens, 4] = (0, row, column, 0), and of the text tokens, (0, 0, 0, index).</summary>
    public static int[] ImageIds(int side) => ImageIds(side, side);

    /// <summary>Position ids of a rows x columns grid of image tokens, (t, row, column, 0); <paramref name="t"/> is 0 for the picture itself and 10 + 10 i for reference i.</summary>
    public static int[] ImageIds(int rows, int columns, int t = 0)
    {
        var ids = new int[rows * columns * 4];
        for (int y = 0; y < rows; y++)
            for (int x = 0; x < columns; x++) { int o = (y * columns + x) * 4; ids[o] = t; ids[o + 1] = y; ids[o + 2] = x; }
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
    public float[] ToVaeInput(float[] packed, int side) => ToVaeInput(packed, side, side);

    /// <summary><see cref="ToVaeInput(float[], int)"/> for a rows x columns grid of tokens: [32, 2*rows, 2*columns].</summary>
    public float[] ToVaeInput(float[] packed, int rows, int columns)
    {
        int cells = rows * columns, outH = rows * 2, outW = columns * 2;
        var o = new float[(long)32 * outH * outW];
        for (int cell = 0; cell < cells; cell++)
        {
            int y = cell / columns, x = cell % columns;
            for (int ch = 0; ch < 128; ch++)
            {
                float v = packed[cell * 128 + ch] * _norm.Std[ch] + _norm.Mean[ch];
                int c = ch / 4, py = (ch / 2) % 2, px = ch % 2;
                o[((long)c * outH + 2 * y + py) * outW + 2 * x + px] = v;
            }
        }
        return o;
    }

    /// <summary>
    /// A reference picture as transformer tokens: prepared like the reference pipeline (<see cref="Flux2ReferencePrep"/>), encoded by the VAE (the mean of the latent distribution), patchified 2 x 2
    /// (channel c * 4 + dy * 2 + dx), batch-normalised with the same statistics the decoder path undoes, packed row by row into [rows * columns, 128].
    /// </summary>
    public async Task<(float[] Tokens, int Rows, int Columns)> EncodeReferenceAsync(Flux2ReferenceImage reference)
    {
        var factory = VaeEncoderFactory ?? throw new InvalidOperationException("This pipeline has no VAE encoder (VaeEncoderFactory): reference images need " + VaeEncoderFile + ".");
        var (chw, w, h) = Flux2ReferencePrep.Prepare(reference);
        // the encoder graph is fixed-shape: the last one is kept, so several pictures of one size (or the same picture again) rebuild nothing
        if (_encoder == null || _encoderKey != (h, w))
        {
            _encoder?.Dispose(); _encoder = null;
            _encoder = await factory(h, w); _encoderKey = (h, w);
        }
        var enc = _encoder;
        using var img = _acc.Allocate1D(chw);
        var outputs = await enc.RunAsync(new Dictionary<string, Tensor> { [enc.InputNames[0]] = new Tensor(img.View, new[] { 1, 3, h, w }, enc.InputNames[0]) });
        int lh = h / 8, lw = w / 8;
        var lat = await outputs[enc.OutputNames[0]].Data.SubView(0, 32L * lh * lw).CopyToHostAsync();
        int rows = h / 16, cols = w / 16;
        var tokens = new float[(long)rows * cols * 128];
        for (int c = 0; c < 32; c++)
            for (int y = 0; y < rows; y++)
                for (int x = 0; x < cols; x++)
                    for (int py = 0; py < 2; py++)
                        for (int px = 0; px < 2; px++)
                        {
                            int ch = c * 4 + py * 2 + px;
                            float v = lat[((long)c * lh + 2 * y + py) * lw + 2 * x + px];
                            tokens[((long)y * cols + x) * 128 + ch] = (v - _norm.Mean[ch]) / _norm.Std[ch];
                        }
        return (tokens, rows, cols);
    }

    /// <summary>Decode VAE input latents [32, 2*side, 2*side] (<see cref="ToVaeInput"/>) to RGBA; <paramref name="image"/> also receives the raw [-1, 1] planes when given.</summary>
    public Task<Flux2Image> DecodeAsync(float[] vaeInput, int side, float[]? image = null) => DecodeAsync(vaeInput, side, side, image);

    /// <summary><see cref="DecodeAsync(float[], int, float[])"/> for a rows x columns grid of tokens (a picture of 16 * columns x 16 * rows pixels); a non-square or smaller-than-the-graph picture goes through the tiled decode.</summary>
    public async Task<Flux2Image> DecodeAsync(float[] vaeInput, int rows, int columns, float[]? image = null)
    {
        if (rows != columns) return await DecodeTiledAsync(vaeInput, rows * 2, columns * 2, image);
        int side = rows;
        int lat = side * 2, size = lat * 8, pixels = size * size;
        if (lat > VaeTileLatent) return await DecodeTiledAsync(vaeInput, lat, lat, image);
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

    private async Task<Flux2Image> DecodeTiledAsync(float[] vaeInput, int latH, int latW, float[]? image)
    {
        int T = VaeTileLatent, O = Math.Min(VaeTileOverlap, T / 2), stride = T - O, tilePx = T * 8, ramp = O * 8;
        int height = latH * 8, width = latW * 8;
        var sum = new float[3L * height * width]; var wsum = new float[(long)height * width];
        List<int> Starts(int len) { var l = new List<int>(); if (len <= T) { l.Add(0); return l; } for (int x = 0; x + T < len; x += stride) l.Add(x); l.Add(len - T); return l; }
        var ys = Starts(latH); var xs = Starts(latW);
        var tileIn = new float[32 * T * T];
        foreach (int y0 in ys)
            foreach (int x0 in xs)
            {
                for (int c = 0; c < 32; c++)
                    for (int y = 0; y < T; y++)
                    {
                        int sy = Math.Min(y0 + y, latH - 1);                                  // a picture smaller than the graph is padded by repeating its last row / column
                        for (int x = 0; x < T; x++) tileIn[(c * T + y) * T + x] = vaeInput[((long)c * latH + sy) * latW + Math.Min(x0 + x, latW - 1)];
                    }
                using var latBuf = _acc.Allocate1D(tileIn);
                var outputs = await _vae.RunAsync(new Dictionary<string, Tensor> { [_vae.InputNames[0]] = new Tensor(latBuf.View, new[] { 1, 32, T, T }, _vae.InputNames[0]) });
                var tile = await outputs[_vae.OutputNames[0]].Data.SubView(0, 3L * tilePx * tilePx).CopyToHostAsync();
                int px0 = x0 * 8, py0 = y0 * 8;
                bool firstY = y0 == 0, lastY = latH <= T || y0 == latH - T, firstX = x0 == 0, lastX = latW <= T || x0 == latW - T;
                for (int y = 0; y < tilePx && py0 + y < height; y++)
                {
                    float wy = Feather(y, tilePx, ramp, firstY, lastY);
                    for (int x = 0; x < tilePx && px0 + x < width; x++)
                    {
                        float w = wy * Feather(x, tilePx, ramp, firstX, lastX);
                        long o = (long)(py0 + y) * width + px0 + x;
                        wsum[o] += w;
                        for (int c = 0; c < 3; c++) sum[c * (long)height * width + o] += w * tile[((long)c * tilePx + y) * tilePx + x];
                    }
                }
            }
        long pixels = (long)height * width;
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
        return new Flux2Image { Width = width, Height = height, Rgba = rgba };
    }

    /// <summary>Blend weight of pixel <paramref name="i"/> of a tile: a linear ramp over <paramref name="ramp"/> pixels at each side that has a neighbour, 1 elsewhere.</summary>
    private static float Feather(int i, int n, int ramp, bool first, bool last)
    {
        float w = 1f;
        if (!first && i < ramp) w = Math.Min(w, (i + 1f) / (ramp + 1f));
        if (!last && n - 1 - i < ramp) w = Math.Min(w, (n - i) / (ramp + 1f));
        return w;
    }

    /// <summary>The most reference pictures one request may carry.</summary>
    public const int MaxReferences = 10;

    public async Task<Flux2Image> RunAsync(Flux2Request request) => (await RunManyAsync(request, new[] { request.Seed }))[0];

    /// <summary>
    /// Draw one prompt with several seeds: the prompt is encoded once and the transformer's weights stay on the device for every seed, so each extra picture costs only its denoise steps and a VAE
    /// decode (a few seconds on a desktop GPU) instead of another 4 GB of weight loading. <paramref name="onImage"/> is called with (index, picture) as each one is finished. <see cref="Flux2Request.Seed"/> is ignored;
    /// <see cref="Flux2Request.Noise"/> applies to a single seed only.
    /// </summary>
    public async Task<List<Flux2Image>> RunManyAsync(Flux2Request request, int[] seeds, Func<int, Flux2Image, Task>? onImage = null)
    {
        if (seeds.Length == 0) throw new ArgumentException("at least one seed");
        if (request.Noise != null && seeds.Length != 1) throw new ArgumentException("Noise can only be given for a single seed");
        var refs = request.References;
        if (refs != null && (refs.Count == 0 || refs.Count > MaxReferences)) throw new ArgumentException($"References: 1 to {MaxReferences} pictures");
        // the picture's size: Width x Height when given, else the first reference's, else Size x Size
        int pxW, pxH;
        if (request.Width > 0 || request.Height > 0) { pxW = request.Width > 0 ? request.Width : request.Size; pxH = request.Height > 0 ? request.Height : request.Size; }
        else if (refs != null) (pxW, pxH) = Flux2ReferencePrep.PreparedSize(refs[0].Width, refs[0].Height);
        else { pxW = pxH = request.Size; }
        if (pxW % 16 != 0 || pxH % 16 != 0 || pxW < 64 || pxH < 64) throw new ArgumentException("The picture's sides must be multiples of 16 and at least 64");
        int rows = pxH / 16, cols = pxW / 16, nImg = rows * cols, nTxt = Flux2TextEncoder.SeqLen;
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
        var sigmas = Flux2Scheduler.Sigmas(nImg, request.Steps);          // the schedule shift depends on the picture's own token count, not on the references'
        var txtIds = TextIds(nTxt);
        // the references: encoded now (the encoder's graph is built per size and freed again), appended to the picture's tokens with their own positions
        var allIds = new List<int>(ImageIds(rows, cols));
        var refTokens = new List<float[]>(); int nRef = 0;
        if (refs != null)
            for (int r = 0; r < refs.Count; r++)
            {
                Report("reference " + (r + 1), r, refs.Count);
                var (tokens, rr, cc) = await EncodeReferenceAsync(refs[r]);
                refTokens.Add(tokens); nRef += rr * cc;
                allIds.AddRange(ImageIds(rr, cc, 10 + 10 * r));
            }
        var imgIds = allIds.ToArray();
        var input = new float[(long)(nImg + nRef) * Flux2Transformer.InChannels];            // [the picture's tokens | the references' tokens]
        { long o = (long)nImg * Flux2Transformer.InChannels; foreach (var t in refTokens) { Array.Copy(t, 0, input, o, t.Length); o += t.Length; } }
        var images = new List<Flux2Image>();
        Flux2Weights tw; Flux2Transformer tr;
        if (_residentTr != null && _residentTokens == nImg + nRef) { tw = _residentWeights!; tr = _residentTr; }
        else
        {
            ReleaseResident();
            Report("transformer", 0, 1);
            tw = await _transformerWeights();
            tr = new Flux2Transformer(_acc, tw, nTxt, nImg + nRef);
        }
        try
        {
        for (int n = 0; n < seeds.Length; n++)
        {
            var latents = request.Noise?.ToArray() ?? GaussianNoise(nImg * Flux2Transformer.InChannels, seeds[n]);
            for (int i = 0; i < request.Steps; i++)
            {
                Report(seeds.Length > 1 ? $"denoise {n + 1}/{seeds.Length}" : "denoise", i, request.Steps);
                Array.Copy(latents, input, latents.Length);
                tr.Forward(input, embeds, Flux2Scheduler.Timestep(sigmas, i), imgIds, txtIds);
                await _acc.SynchronizeAsync();
                var pred = await tr.Output.SubView(0, (long)nImg * Flux2Transformer.InChannels).CopyToHostAsync();
                float dt = (float)(sigmas[i + 1] - sigmas[i]);
                for (int k = 0; k < latents.Length; k++) latents[k] += dt * pred[k];
            }
            Report("vae", n, seeds.Length);
            var image = await DecodeAsync(ToVaeInput(latents, rows, cols), rows, cols);
            images.Add(image);
            if (onImage != null) await onImage(n, image);
        }
        }
        finally
        {
            if (KeepTransformerResident) { _residentWeights = tw; _residentTr = tr; _residentTokens = nImg + nRef; }
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

    private InferenceSession? _encoder;
    private (int Height, int Width) _encoderKey;

    /// <summary>Frees the resident VAE decoder and encoder and any kept transformer. The accelerator is not touched.</summary>
    public void Dispose() { ReleaseResident(); _encoder?.Dispose(); _encoder = null; _vae.Dispose(); _elementWise?.Dispose(); }
}
