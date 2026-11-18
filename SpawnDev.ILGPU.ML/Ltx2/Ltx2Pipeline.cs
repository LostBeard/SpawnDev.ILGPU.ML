using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnDev.ILGPU.ML.SafeTensors;
using SpawnDev.ILGPU.ML.Wan;
using FV = ILGPU.Runtime.ArrayView1D<float, ILGPU.Stride1D.Dense>;

namespace SpawnDev.ILGPU.ML.Ltx2;

/// <summary>The files of an LTX-2 19B run, as names the pipeline's model source resolves (ComfyUI's own files work as they are).</summary>
public sealed class Ltx2Paths
{
    /// <summary>The checkpoint (<c>ltx-2-19b-dev-fp8.safetensors</c>, 27 GB): the transformer (streamed block by block), the text connectors, the video and audio VAEs, the vocoder.</summary>
    public required string Checkpoint { get; init; }
    /// <summary>Gemma 3 12B (<c>gemma_3_12B_it_fp4_mixed.safetensors</c>, 9.4 GB, streamed layer by layer); it carries its SentencePiece model (<c>spiece_model</c>).</summary>
    public required string TextEncoder { get; init; }
    /// <summary>The distilled LoRA (<c>ltx-2-19b-distilled-lora-384.safetensors</c>): needed for <see cref="Ltx2Request.Distilled"/>.</summary>
    public string? DistilledLora { get; init; }
    /// <summary>The repository id the source resolves the files in (a local folder source ignores it).</summary>
    public string Repo { get; init; } = "";

    /// <summary>ComfyUI's layout under one folder (a <see cref="LocalFolderModelSource"/> on it): <c>lightricks/</c> and <c>comfy/split_files/text_encoders/</c>.</summary>
    public static Ltx2Paths InFolder() => new()
    {
        Checkpoint = "lightricks/ltx-2-19b-dev-fp8.safetensors",
        TextEncoder = "comfy/split_files/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors",
        DistilledLora = "lightricks/ltx-2-19b-distilled-lora-384.safetensors",
    };
}

/// <summary>
/// A picture the clip is guided by (ComfyUI's LTXVAddGuide): <see cref="Pixels"/> [3, Height, Width] channel-first in [-1, 1] at the request's size, held at pixel frame
/// <see cref="FrameIndex"/> (0 = the first frame: image-to-video; negative counts from the end, -1 = the last frame). <see cref="Strength"/> 1 keeps the picture as it is
/// (its latent frame is never noised); below 1 it is denoised from (1 - strength) and its pull on the other frames is weakened (ln(strength) on the attention).
/// </summary>
public sealed class Ltx2Keyframe
{
    public required float[] Pixels { get; init; }
    public int FrameIndex { get; init; }
    public float Strength { get; init; } = 1f;

    /// <summary>From RGB bytes (3 per pixel, row-major) at the request's size.</summary>
    public static Ltx2Keyframe FromRgb(byte[] rgb, int width, int height, int frameIndex = 0, float strength = 1f)
    {
        if (rgb.Length != 3 * width * height) throw new ArgumentException($"rgb: 3 x {width} x {height} bytes");
        int n = width * height;
        var p = new float[3 * n];
        for (int i = 0; i < n; i++) for (int c = 0; c < 3; c++) p[c * n + i] = rgb[3 * i + c] / 127.5f - 1f;
        return new Ltx2Keyframe { Pixels = p, FrameIndex = frameIndex, Strength = strength };
    }
}

/// <summary>What to generate.</summary>
public sealed class Ltx2Request
{
    public required string Prompt { get; init; }
    /// <summary>The negative prompt (used when <see cref="Cfg"/> is not 1).</summary>
    public string NegativePrompt { get; init; } = "";
    /// <summary>Output size, multiples of 32.</summary>
    public int Width { get; init; } = 768;
    public int Height { get; init; } = 512;
    /// <summary>Frame count, 8 n + 1 (121 = 4.84 s at 25 fps). The audio runs as long: one audio latent frame per video frame at 25 fps.</summary>
    public int Frames { get; init; } = 121;
    public float FrameRate { get; init; } = 25f;
    /// <summary>Denoising steps of ComfyUI's LTXVScheduler (<see cref="Ltx2Schedule.Sigmas"/>); ignored with <see cref="Sigmas"/> or <see cref="Distilled"/>.</summary>
    public int Steps { get; init; } = 20;
    /// <summary>Classifier-free guidance; 1 = none (one model row per step instead of two).</summary>
    public float Cfg { get; init; } = 4f;
    /// <summary>The distilled recipe: the distilled LoRA (<see cref="Ltx2Paths.DistilledLora"/>) at <see cref="LoraStrength"/>, <see cref="Ltx2Schedule.Distilled"/>'s 8 steps, no guidance.</summary>
    public bool Distilled { get; init; }
    public float LoraStrength { get; init; } = 1f;
    /// <summary>
    /// How the LoRA meets the checkpoint's 8-bit (F8_E4M3) weights. False (default): exact, the LoRA's full effect in float32. True: ComfyUI's result, the patched
    /// weight re-rounded to 8 bits with its seeded rounding (<see cref="Ltx2Transformer.ComfyCompatibleFp8Lora"/>): pick it to reproduce a ComfyUI render.
    /// </summary>
    public bool ComfyCompatibleFp8Lora { get; init; }
    /// <summary>The sigmas to run (ending in 0), instead of the schedule.</summary>
    public float[]? Sigmas { get; init; }
    public long Seed { get; init; } = 42;
    /// <summary>The start noise [packed length] (video [128, F, H, W] with the guide frames appended, then audio [8, Ta, 16]) instead of one drawn from <see cref="Seed"/>.</summary>
    public float[]? InitialNoise { get; init; }
    /// <summary>Pictures to guide the clip by, in order (each adds one latent frame to the run; cropped from the result).</summary>
    public IReadOnlyList<Ltx2Keyframe>? Keyframes { get; init; }
    /// <summary>True (default): the frames come as RGBA bytes, converted on the device a frame at a time; false: one float array [3, Frames, Height, Width] in [-1, 1].</summary>
    public bool RgbaFrames { get; init; } = true;
    /// <summary>False: skip the audio decode.</summary>
    public bool Audio { get; init; } = true;
}

/// <summary>The result of <see cref="Ltx2Pipeline.GenerateAsync"/>.</summary>
public sealed class Ltx2Clip
{
    public required WanVideo Video { get; init; }
    /// <summary>The stereo waveform [2, samples] channel-first (left, then right) at <see cref="AudioSampleRate"/>; null when the request skipped it.</summary>
    public float[]? Audio { get; init; }
    public int AudioSampleRate { get; init; } = Ltx2AudioDecoder.SampleRate;
    /// <summary>The final latents: video [128, F, H, W] (guide frames cropped) and audio [8, Ta, 16].</summary>
    public required float[] VideoLatent { get; init; }
    public required float[] AudioLatent { get; init; }
    /// <summary>Seconds per stage: "encode" (Gemma + connectors), "sample", "video decode", "audio decode".</summary>
    public required IReadOnlyDictionary<string, double> Seconds { get; init; }
}

/// <summary>
/// LTX-2 19B end to end on one accelerator: prompt -> Gemma 3 12B -> text connectors -> the audio-video transformer under ComfyUI's sampler (LTXVScheduler sigmas,
/// classifier-free guidance, Euler; keyframe guides by masked sampling) -> video and audio decode. Every stage is the one gated against ComfyUI
/// (<see cref="Gemma3TextEncoder"/>, <see cref="Ltx2TextConnectors"/>, <see cref="Ltx2AvModel"/>, <see cref="Ltx2VideoEncoder"/>, <see cref="Ltx2VideoDecoder"/>,
/// <see cref="Ltx2AudioDecoder"/>); the stages are opened one after another so only one large model is on the device at a time.
/// <para>The start noise is this library's seeded generator, not torch's: a seed gives its own clip, not ComfyUI's (<see cref="Ltx2Request.InitialNoise"/> takes
/// another implementation's noise).</para>
/// </summary>
public sealed class Ltx2Pipeline
{
    public const int ContextTokens = 1024, SpatialRatio = 32, TemporalRatio = 8;

    private readonly Accelerator _acc;
    private readonly IModelSource _source;
    private readonly Ltx2Paths _paths;

    /// <summary>Transformer blocks kept on the device between steps (the rest stream from the checkpoint every step): default 5 GB.</summary>
    public long TransformerResidentBytes { get; set; } = 5_000_000_000;
    /// <summary>Gemma layers kept on the device during the encode: default 4 GB.</summary>
    public long TextEncoderResidentBytes { get; set; } = 4_000_000_000;
    /// <summary>
    /// Host RAM (bytes) for keeping streamed block images between sampling steps: the transformer's blocks first, what is left for the distilled LoRA's. A cached
    /// block is read from the disk once per clip instead of once per step (it is still uploaded every step). Default 0: every step re-reads everything it streams
    /// (at 768 x 512 x 121 on a 12 GB card: 16.2 GB of checkpoint blocks, plus 6.7 GB of LoRA blocks in the distilled recipe). It pays where the disk sets the pace:
    /// MEASURED 2026-11-18 from a SATA drive with 6 GB, a distilled clip 458 -> 424 s; a two-row call (GPU-bound there) does not change.
    /// </summary>
    public long HostCacheBytes { get; set; }
    /// <summary>
    /// Host RAM (bytes) the sampler's read-ahead may hold in streamed block images (the transformer's; the distilled LoRA's read-ahead gets half as much): the reader
    /// runs that far ahead of the device instead of one block (<see cref="StreamedModelWeights.ReadAheadBytes"/>). Default 0: one block ahead. For a slow drive.
    /// MEASURED 2026-11-18, RTX 4070, 768 x 512 x 121 from a SATA drive with 2 GB: a two-row call 44.0 -> 40.6 s, a distilled clip 458 -> 439 s, the same output bit
    /// for bit; from an NVMe drive it changes nothing. It costs host memory: 3.1 to 5.3 GB committed after a clip against 1.1 to 1.3 GB (each block image is a
    /// fresh large array), which is why it is not the default.
    /// </summary>
    public long ReadAheadBytes { get; set; }
    /// <summary>
    /// True (default): the transformer computes with float32-exact products (what the ComfyUI float32 gates are held to). False: its GEMMs and attention run on the
    /// tensor cores with bf16 activations where the card has them: about half the GPU work, a result 5.8e-3 (relative RMS, one call) from the exact one. MEASURED
    /// 2026-11-18, RTX 4070, 768 x 512 x 121, files on an NVMe drive: a two-row call 38.8 s exact against 20.9 s; a distilled clip 240 s against 169 s. From a SATA
    /// drive the disk hides most of it (a two-row call 44.0 against 39.7 s): the fast call needs its 16 GB of streamed blocks faster than that drive reads them.
    /// </summary>
    public bool PreciseActivations { get; set; } = true;

    /// <summary>
    /// With <see cref="PreciseActivations"/> on a card with tensor cores: run the transformer's attentions on the tensor cores with q, k, v and the probabilities kept as
    /// two bf16 halves each (three passes per product, fp32 sums, keys centred). MEASURED 2026-11-18, RTX 4070, 768 x 512 x 121: the GPU work of a two-row call
    /// 34.2 s -> 23.7 s (video self-attention 14.3 s -> 5.1 s); one attention is 4e-6 to 9e-6 (relative RMS) from float64 where the fp32 kernel is 3e-7 and the
    /// bf16 one 3e-3; the transformer against ComfyUI's float32 taps is 5.9e-6 at the final projection (1.2e-5 with the fp32 attention kernel: both inside
    /// the reference's own rounding). False by default: it is a little less precise per attention, so it is yours to turn on.
    /// </summary>
    public bool ExactTensorCoreAttention { get; set; }

    public Ltx2Pipeline(Accelerator accelerator, IModelSource source, Ltx2Paths paths)
    {
        _acc = accelerator; _source = source; _paths = paths;
    }

    /// <summary>Gemma's SentencePiece tokenizer from the text encoder file's <c>spiece_model</c> tensor (read by offset, without the rest of the file).</summary>
    public static async Task<SentencePieceTokenizer> LoadTokenizerAsync(IModelSource source, string repo, string file, CancellationToken ct = default)
    {
        await using var stream = await source.OpenAsync(repo, file, ct);
        if (!stream.CanSeek) throw new NotSupportedException($"'{file}': the tokenizer is read from the safetensors file by offset; the model source must give a seekable stream");
        var lenBytes = new byte[8];
        await stream.ReadExactlyAsync(lenBytes, ct);
        long headerSize = BitConverter.ToInt64(lenBytes);
        if (headerSize <= 0 || headerSize > 100_000_000) throw new InvalidDataException($"'{file}': invalid SafeTensors header size {headerSize}");
        var headerBytes = new byte[headerSize];
        await stream.ReadExactlyAsync(headerBytes, ct);
        long dataStart = 8 + headerSize;
        var (tensors, _) = SafeTensorsFileReader.ParseHeader(file, headerBytes, dataStart, stream.Length - dataStart);
        var spm = tensors.FirstOrDefault(t => t.Name == "spiece_model") ?? throw new InvalidDataException($"'{file}' has no spiece_model tensor (Gemma's SentencePiece model)");
        if (spm.DType != "U8" || spm.DataLength <= 0 || spm.DataLength > 64 << 20) throw new InvalidDataException($"'{file}': spiece_model is {spm.DType}, {spm.DataLength} bytes; expected U8 bytes of a SentencePiece model");
        var bytes = new byte[spm.DataLength];
        stream.Seek(spm.DataOffset, SeekOrigin.Begin);
        await stream.ReadExactlyAsync(bytes, ct);
        return SentencePieceTokenizer.FromSentencePieceModel(bytes);
    }

    /// <summary>The ids the encoder takes for a prompt: BOS, then the prompt's pieces (ComfyUI's ids for the template prompt, id for id), cut to 1024.</summary>
    public static int[] PromptIds(SentencePieceTokenizer tokenizer, string prompt)
    {
        var pieces = tokenizer.Encode(prompt.Trim());
        var ids = new int[Math.Min(pieces.Length + 1, Gemma3TextEncoder.MaxPositions)];
        ids[0] = tokenizer.BosId;
        Array.Copy(pieces, 0, ids, 1, ids.Length - 1);
        return ids;
    }

    /// <summary>
    /// The conditioning of the prompts, [prompts, 1024, 7680] on the device in the given order (Gemma 3 encode, then the connectors; both are released before returning).
    /// The caller disposes the buffer.
    /// </summary>
    public async Task<MemoryBuffer1D<float, Stride1D.Dense>> EncodePromptsAsync(IReadOnlyList<string> prompts, IProgress<WanProgress>? progress = null, CancellationToken ct = default)
    {
        long row = (long)ContextTokens * Ltx2AvModel.ContextWidth;
        var context = _acc.Allocate1D<float>(prompts.Count * row);
        try
        {
            var tokenizer = await LoadTokenizerAsync(_source, _paths.Repo, _paths.TextEncoder, ct);
            using var te = await Gemma3TextEncoder.OpenAsync(_acc, _source, _paths.Repo, _paths.TextEncoder, TextEncoderResidentBytes, ct);
            using var tc = await Ltx2TextConnectors.OpenAsync(_acc, _source, _paths.Repo, _paths.Checkpoint, ct);
            for (int i = 0; i < prompts.Count; i++)
            {
                progress?.Report(new WanProgress("encode", (double)i / prompts.Count, $"encoding prompt {i + 1} of {prompts.Count}"));
                var ids = PromptIds(tokenizer, prompts[i]);
                var features = await te.EncodeAsync(ids, ct);
                var cond = await tc.RunAsync(features, ids.Length, ct);
                context.View.SubView(i * row, row).CopyFrom(cond.SubView(0, row));
            }
            await _acc.SynchronizeAsync();                                                   // the copies read the stages' buffers: drained before those are disposed
            progress?.Report(new WanProgress("encode", 1, "prompts encoded"));
            return context;
        }
        catch { await _acc.SynchronizeAsync(); context.Dispose(); throw; }
    }

    /// <summary>The whole run: encode, sample, decode.</summary>
    public async Task<Ltx2Clip> GenerateAsync(Ltx2Request req, IProgress<WanProgress>? progress = null, CancellationToken ct = default)
    {
        if (req.Width % SpatialRatio != 0 || req.Height % SpatialRatio != 0 || req.Width < SpatialRatio || req.Height < SpatialRatio) throw new ArgumentException("width and height: multiples of 32");
        if (req.Frames < 1 || (req.Frames - 1) % TemporalRatio != 0) throw new ArgumentException("frames: 8 n + 1");
        if (req.Distilled && _paths.DistilledLora == null) throw new InvalidOperationException("the distilled recipe needs Ltx2Paths.DistilledLora");
        float cfg = req.Distilled ? 1f : req.Cfg;
        int B = cfg == 1f ? 1 : 2;
        int F = (req.Frames - 1) / TemporalRatio + 1, H = req.Height / SpatialRatio, W = req.Width / SpatialRatio, hw = H * W;
        int Ta = (int)Math.Round(req.Frames * 25.0 / req.FrameRate);                          // 25 audio latent frames per second (16 kHz, hop 160, x4): the clip's duration
        var keys = req.Keyframes ?? Array.Empty<Ltx2Keyframe>();
        int gf = keys.Count, Fall = F + gf, nv = Fall * hw, V = Ltx2AvModel.Channels * nv, A = Ltx2AvModel.AudioChannels * Ta * Ltx2AvModel.AudioBins, P = V + A;
        var seconds = new Dictionary<string, double>();
        var sw = System.Diagnostics.Stopwatch.StartNew();

        // 1. the conditioning: [uncond, cond] (ComfyUI's batch order) or [cond]
        using var context = await EncodePromptsAsync(B == 2 ? new[] { req.NegativePrompt, req.Prompt } : new[] { req.Prompt }, progress, ct);
        seconds["encode"] = sw.Elapsed.TotalSeconds; sw.Restart();

        // 2. keyframe guides: each picture's latent frame appended after the video's own, its tokens at the picture's place in time (ComfyUI's LTXVAddGuide)
        Ltx2Guide? guide = null;
        float[]? clean = null;
        if (gf > 0)
        {
            clean = new float[P];
            var boxes = new float[3 * gf * hw * 2];
            var frameMask = new float[Fall];
            var strengths = new float[gf * hw];
            Array.Fill(frameMask, 1f);
            using var encoder = await Ltx2VideoEncoder.OpenAsync(_acc, _source, _paths.Repo, _paths.Checkpoint, ct);
            for (int g = 0; g < gf; g++)
            {
                var k = keys[g];
                if (k.Pixels.Length != 3 * req.Width * req.Height) throw new ArgumentException($"keyframe {g}: [3, {req.Height}, {req.Width}] pixels");
                if (!(k.Strength > 0f && k.Strength <= 1f)) throw new ArgumentException($"keyframe {g}: strength in (0, 1]");
                int frame = k.FrameIndex >= 0 ? k.FrameIndex : req.Frames + k.FrameIndex;
                if (frame < 0 || frame >= req.Frames) throw new ArgumentException($"keyframe {g}: frame {k.FrameIndex} of {req.Frames}");
                var z = await encoder.EncodeAsync(k.Pixels, 1, req.Height, req.Width);        // [128, 1, H, W]: the posterior mean
                for (int c = 0; c < Ltx2AvModel.Channels; c++) Array.Copy(z, c * hw, clean, c * nv + (F + g) * hw, hw);
                frameMask[F + g] = 1f - k.Strength;
                for (int t = 0; t < hw; t++)
                {
                    int j = g * hw + t, r = t / W, col = t % W;
                    boxes[(0 * gf * hw + j) * 2] = frame; boxes[(0 * gf * hw + j) * 2 + 1] = frame + 1;
                    boxes[(1 * gf * hw + j) * 2] = SpatialRatio * r; boxes[(1 * gf * hw + j) * 2 + 1] = SpatialRatio * (r + 1);
                    boxes[(2 * gf * hw + j) * 2] = SpatialRatio * col; boxes[(2 * gf * hw + j) * 2 + 1] = SpatialRatio * (col + 1);
                    strengths[j] = k.Strength;
                }
            }
            await _acc.SynchronizeAsync();
            guide = new Ltx2Guide(gf, boxes, keys[0].Strength, frameMask, strengths);
        }

        // 3. sampling
        var sigmas = req.Sigmas ?? (req.Distilled ? Ltx2Schedule.Distilled : Ltx2Schedule.Sigmas(req.Steps, F * hw));   // the shift goes by the video's own tokens
        if (sigmas.Length < 2 || sigmas[^1] != 0f) throw new ArgumentException("sigmas: at least one step, ending in 0");
        var noise = req.InitialNoise ?? SeededNoise.Gaussian(P, req.Seed);
        if (noise.Length != P) throw new ArgumentException($"initial noise: {P} values (video [128, {Fall}, {H}, {W}] then audio [8, {Ta}, 16])");
        var start = noise;
        if (clean != null || sigmas[0] != 1f)
        {
            // the sampler's start state: sigma noise + (1 - sigma) latent (the latent: zeros but for the guide frames)
            start = new float[P];
            float s0 = sigmas[0];
            for (int i = 0; i < P; i++) start[i] = s0 * noise[i] + (1f - s0) * (clean?[i] ?? 0f);
        }
        float[] latent;
        using (var model = await Ltx2AvModel.OpenAsync(_acc, _source, _paths.Repo, _paths.Checkpoint, B, Fall, H, W, Ta, ContextTokens, TransformerResidentBytes, ct))
        {
            Ltx2Lora? lora = null;
            long cacheForBlocks = Math.Min(HostCacheBytes, model.Transformer.Weights.StreamedBytesPerForward);
            model.Transformer.Weights.HostCacheBytes = cacheForBlocks;
            model.Transformer.Weights.ReadAheadBytes = ReadAheadBytes;
            model.Transformer.PreciseActivations = PreciseActivations;
            if (ExactTensorCoreAttention) model.Transformer.ExactTensorCoreAttention = true;      // otherwise the kernels' own default (env TRELLIS2_MMA_EXACT_ATTN)
            try
            {
                if (req.Distilled)
                {
                    lora = await Ltx2Lora.OpenAsync(_acc, _source, _paths.Repo, _paths.DistilledLora!, req.LoraStrength, 1L << 30, ct);
                    lora.Weights.HostCacheBytes = HostCacheBytes - cacheForBlocks;
                    lora.Weights.ReadAheadBytes = ReadAheadBytes / 2;
                    model.Transformer.AddLora(lora);
                    model.Transformer.ComfyCompatibleFp8Lora = req.ComfyCompatibleFp8Lora;
                }
                int steps = sigmas.Length - 1;
                progress?.Report(new WanProgress("sample", 0, $"{steps} steps, {nv} video + {Ta} audio tokens"));
                // the sampler runs on the device: the state crosses to the host once, at the end (the decoders take host latents)
                using var startDev = _acc.Allocate1D(start);
                using var cleanDev = clean != null ? _acc.Allocate1D(clean) : null;
                try
                {
                    var x = await model.SampleAsync(startDev.View, context.View, sigmas, cfg, req.FrameRate, guide, cleanDev != null ? cleanDev.View : default(FV),
                        (i, _) => { progress?.Report(new WanProgress("sample", (i + 1.0) / steps, $"step {i + 1} of {steps}")); return Task.CompletedTask; }, ct);
                    latent = await x.CopyToHostAsync();
                }
                finally { await _acc.SynchronizeAsync(); }                                   // nothing queued may still read the two uploads when they are disposed
            }
            finally
            {
                await _acc.SynchronizeAsync();
                if (lora != null) { model.Transformer.RemoveLora(lora); lora.Dispose(); }
            }
        }
        seconds["sample"] = sw.Elapsed.TotalSeconds; sw.Restart();

        // 4. the latents (guide frames cropped: ComfyUI's LTXVCropGuides) and the decodes
        var videoLatent = new float[Ltx2AvModel.Channels * F * hw];
        for (int c = 0; c < Ltx2AvModel.Channels; c++) Array.Copy(latent, c * nv, videoLatent, c * F * hw, F * hw);
        var audioLatent = new float[A];
        Array.Copy(latent, V, audioLatent, 0, A);
        progress?.Report(new WanProgress("decode", 0, "decoding the video"));
        float[] video = Array.Empty<float>(); byte[][]? rgba = null;
        using (var vae = await Ltx2VideoDecoder.OpenAsync(_acc, _source, _paths.Repo, _paths.Checkpoint, ct))
        {
            if (req.RgbaFrames) rgba = await vae.DecodeRgbaFramesAsync(videoLatent, F, H, W);
            else video = await vae.DecodeAsync(videoLatent, F, H, W);
            await _acc.SynchronizeAsync();
        }
        seconds["video decode"] = sw.Elapsed.TotalSeconds; sw.Restart();
        float[]? wav = null;
        if (req.Audio)
        {
            progress?.Report(new WanProgress("decode", 0.9, "decoding the audio"));
            using var ad = await Ltx2AudioDecoder.OpenAsync(_acc, _source, _paths.Repo, _paths.Checkpoint, ct);
            wav = await ad.DecodeAsync(audioLatent, Ta);
            await _acc.SynchronizeAsync();
            seconds["audio decode"] = sw.Elapsed.TotalSeconds;
        }
        progress?.Report(new WanProgress("decode", 1, "done"));
        return new Ltx2Clip
        {
            Video = new WanVideo { Data = video, FramesRgba = rgba, Width = req.Width, Height = req.Height, Frames = req.Frames, Latent = videoLatent },
            Audio = wav, VideoLatent = videoLatent, AudioLatent = audioLatent, Seconds = seconds,
        };
    }
}
