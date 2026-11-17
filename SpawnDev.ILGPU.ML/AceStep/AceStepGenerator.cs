using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Flux2;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.SkinTokens;
using SpawnDev.ILGPU.ML.Trellis2;
using FV = ILGPU.Runtime.ArrayView1D<float, ILGPU.Stride1D.Dense>;

namespace SpawnDev.ILGPU.ML.AceStep;

/// <summary>
/// ACE-Step 1.5 text-to-music with the audio codes off, as ComfyUI's audio_ace_step_1_5_split template runs it (measured on SpawnGen's dumps; Plans/ace-step-1.5-2026-11-16.md):
/// <list type="number">
/// <item>caption: Qwen3-0.6B (qwen_0.6b_ace15) last layer + final norm of the caption ids, the prompt real up to its first &lt;|endoftext|&gt; (keys after it masked);</item>
/// <item>lyrics: the 0.6B's token embedding rows of the lyric ids;</item>
/// <item>condition tokens: <see cref="AceStepConditionEncoder"/> of (caption, lyrics, the reference latent: text-to-music = the silence latent);</item>
/// <item>sampling: x = noise at sigma 1; per step v = <see cref="AceStepDiT"/>(x, context = [silence latent | ones], t = r = sigma), x += (sigma_next - sigma) v (Euler);
/// the last step's denoised x - sigma v is the latent (cfg 1: one call per step);</item>
/// <item>audio: <see cref="OobleckDecoder"/> of the latent.</item>
/// </list>
/// Tensors are time-major [frames, 64] on the device except the latent handed to the VAE ([64, frames], ComfyUI's layout).
/// </summary>
public sealed class AceStepGenerator : IDisposable
{
    /// <summary>Qwen's &lt;|endoftext|&gt; (also its pad token): the caption prompt is real up to the first one.</summary>
    public const int EndOfText = 151643;
    public const int LatentChannels = 64, TextWidth = 1024;

    private readonly Accelerator _acc;
    private readonly Flux2Weights _teWeights;
    private readonly Flux2TextEncoder _te;
    private readonly AceStepConditionEncoder _enc;
    private readonly AceStepDiT _dit;
    private readonly OobleckDecoder _vae;
    private readonly DiTKernels _k;
    private readonly SkinTokensKernels _u;
    private readonly AceStepKernels _a;
    private readonly int _maxFrames, _maxCaption, _maxLyrics;

    public AceStepDiT DiT => _dit;
    public AceStepConditionEncoder ConditionEncoder => _enc;
    public OobleckDecoder Vae => _vae;

    private AceStepGenerator(Accelerator acc, Flux2Weights teWeights, AceStepConditionEncoder enc, AceStepDiT dit, OobleckDecoder vae, int maxFrames, int maxCaption, int maxLyrics)
    {
        _acc = acc; _teWeights = teWeights; _enc = enc; _dit = dit; _vae = vae; _maxFrames = maxFrames; _maxCaption = maxCaption; _maxLyrics = maxLyrics;
        _te = new Flux2TextEncoder(acc, teWeights, new[] { 28 }, finalNorm: true, maxTokens: maxCaption) { PreciseActivations = true };
        _k = new DiTKernels(acc); _u = new SkinTokensKernels(acc); _a = new AceStepKernels(acc);
    }

    /// <summary>
    /// Load ComfyUI's three files: the turbo model (<paramref name="ditFile"/>: DiT + condition encoder), the 0.6B text encoder, the VAE. Buffers are sized for up to
    /// <paramref name="maxFrames"/> latent frames (25 per second), <paramref name="maxCaptionTokens"/> caption ids and <paramref name="maxLyricTokens"/> lyric ids.
    /// </summary>
    public static async Task<AceStepGenerator> LoadAsync(Accelerator accelerator, IModelSource source, string repo, string ditFile, string textEncoderFile, string vaeFile,
        int maxFrames, int maxCaptionTokens = 256, int maxLyricTokens = 2048, CancellationToken ct = default)
    {
        Flux2Weights? te = null; AceStepConditionEncoder? enc = null; AceStepDiT? dit = null; OobleckDecoder? vae = null;
        try
        {
            te = await Flux2Weights.LoadAsync(accelerator, source, repo, textEncoderFile, 28, finalNorm: true, ct);
            enc = await AceStepConditionEncoder.LoadAsync(accelerator, source, repo, ditFile, Math.Max(maxLyricTokens, maxFrames), ct);
            dit = await AceStepDiT.LoadAsync(accelerator, source, repo, ditFile, maxFrames, maxLyricTokens + 1 + maxCaptionTokens, ct);
            vae = await OobleckDecoder.LoadAsync(accelerator, source, repo, vaeFile, ct: ct);
            return new AceStepGenerator(accelerator, te, enc, dit, vae, maxFrames, maxCaptionTokens, maxLyricTokens);
        }
        catch { vae?.Dispose(); dit?.Dispose(); enc?.Dispose(); te?.Dispose(); throw; }
    }

    /// <summary>
    /// The latent [64, frames] (ComfyUI's layout) for the caption / lyric ids (ComfyUI's TextEncodeAceStepAudio1.5 ids, templates included), the starting noise
    /// <paramref name="noise"/> [64, frames] (ComfyUI's layout, sigma 1), the silence latent <paramref name="silence"/> [frames, 64] (time-major) and the sigma schedule
    /// (descending, ending at 0). <paramref name="onStep"/> sees (step, sigma) before each model call.
    /// </summary>
    public async Task<float[]> GenerateLatentAsync(int[] captionIds, int[] lyricIds, float[] noise, float[] silence, int frames, float[] sigmas, Action<int, float>? onStep = null, CancellationToken ct = default)
    {
        if (frames > _maxFrames || captionIds.Length > _maxCaption || lyricIds.Length > _maxLyrics) throw new ArgumentException("larger than the generator was loaded for");
        if (noise.Length != LatentChannels * frames || silence.Length != LatentChannels * frames) throw new ArgumentException("noise / silence: 64 x frames values each");
        if (sigmas.Length < 2 || sigmas[^1] != 0f) throw new ArgumentException("sigmas: descending, ending at 0");
        int nc = captionIds.Length, nl = lyricIds.Length, ne = nl + 1 + nc, D = AceStepDiT.Dim;
        int real = Array.IndexOf(captionIds, EndOfText) is int first and >= 0 ? first : nc;

        using var lyricIdBuf = _acc.Allocate1D(lyricIds);
        using var lyrics = _acc.Allocate1D<float>((long)nl * TextWidth);
        using var enc = _acc.Allocate1D<float>((long)ne * D);
        using var sil = _acc.Allocate1D(silence);
        using var ctx = _acc.Allocate1D<float>((long)frames * 2 * LatentChannels);
        using var nz = _acc.Allocate1D(noise);
        using var x = _acc.Allocate1D<float>((long)frames * LatentChannels);
        using var v = _acc.Allocate1D<float>((long)frames * LatentChannels);
        using var den = _acc.Allocate1D<float>((long)frames * LatentChannels);
        using var latent = _acc.Allocate1D<float>((long)frames * LatentChannels);

        // conditioning: caption states, lyric embeddings, the condition tokens (reference = the silence latent)
        var caption = _te.Encode(captionIds, real);
        _u.Embed(lyricIdBuf.View, _teWeights.Vector("embed_tokens.weight"), lyrics.View, nl, TextWidth);
        _enc.Encode(caption.SubView(0, (long)nc * TextWidth), nc, lyrics.View, nl, sil.View, frames, enc.View);
        // context [frames, 128] = [silence | ones]
        var ones = new float[frames * LatentChannels]; Array.Fill(ones, 1f);
        using (var onesBuf = _acc.Allocate1D(ones))
        {
            _a.Concat(sil.View, onesBuf.View, ctx.View, frames, LatentChannels, LatentChannels);
            await _acc.SynchronizeAsync();
        }
        _k.Transpose(nz.View, x.View, LatentChannels, frames);                                   // [64, T] -> [T, 64]
        for (int i = 0; i < sigmas.Length - 1; i++)
        {
            ct.ThrowIfCancellationRequested();
            float s = sigmas[i], next = sigmas[i + 1];
            onStep?.Invoke(i, s);
            _dit.Forward(x.View, ctx.View, enc.View, frames, ne, s, s, v.View);
            if (next == 0f) { _a.Axpy(x.View, v.View, den.View, -s, frames * LatentChannels); break; }   // the last step: denoised = x - sigma v
            _a.AxpyInPlace(x.View, v.View, next - s, frames * LatentChannels);                   // Euler: x += (next - sigma) v
            await _acc.SynchronizeAsync();
        }
        _k.Transpose(den.View, latent.View, frames, LatentChannels);                             // [T, 64] -> [64, T]
        return await latent.View.CopyToHostAsync();
    }

    /// <summary>Stereo audio [2, 1920 frames] at 48 kHz for a latent [64, frames] (ComfyUI's layout).</summary>
    public async Task<float[]> DecodeAsync(float[] latent, int frames, CancellationToken ct = default)
    {
        using var l = _acc.Allocate1D(latent);
        using var audio = await _vae.DecodeAsync(l.View, frames, ct);
        return await audio.View.CopyToHostAsync();
    }

    public void Dispose()
    {
        _te.Dispose(); _teWeights.Dispose(); _enc.Dispose(); _dit.Dispose(); _vae.Dispose();
    }
}
