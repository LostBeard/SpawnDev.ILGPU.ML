using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Hub;
using FV = ILGPU.Runtime.ArrayView1D<float, ILGPU.Stride1D.Dense>;

namespace SpawnDev.ILGPU.ML.Ltx2;

/// <summary>
/// LTX-2's model call in ComfyUI's own form (what SpawnGen's sampler calls): the packed audio-video latent of each batch item, <c>[video [128, F, H, W] row-major, then audio
/// [8, Ta, 16] row-major]</c> (128 F H W + 128 Ta values), sigma per item, the conditioning [B, L, 7680] (video connector | audio connector) and the frame rate, in; the DENOISED
/// packed latent <c>x - sigma v</c> out (MEASURED: ComfyUI's call output equals x - sigma v exactly). Wraps <see cref="Ltx2Transformer"/> (sigma x 1000 as its timestep).
/// <para>Token layouts (MEASURED bit-exact against ComfyUI's patchify inputs by SpawnGen): video token f H W + h W + w, feature = channel; audio token = latent frame,
/// feature c 16 + bin.</para>
/// </summary>
public sealed class Ltx2AvModel : IDisposable
{
    public const int Channels = 128, AudioChannels = 8, AudioBins = 16, ContextWidth = 2 * Ltx2Transformer.Caption;

    private readonly Accelerator _acc;
    private readonly Ltx2Transformer _t;
    private readonly Flux2.Flux2Kernels _f;
    private readonly int _B, _F, _H, _W, _Ta, _L;
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _video, _audio, _vctx, _actx;

    /// <summary>Values of one item's packed latent: 128 F H W + 128 Ta.</summary>
    public int PackedLength => Channels * _F * _H * _W + AudioChannels * _Ta * AudioBins;

    /// <summary>The transformer (probes, streaming timers, precision).</summary>
    public Ltx2Transformer Transformer => _t;

    private Ltx2AvModel(Accelerator acc, Ltx2Transformer t, int B, int F, int H, int W, int Ta, int L)
    {
        _acc = acc; _t = t; _B = B; _F = F; _H = H; _W = W; _Ta = Ta; _L = L; _f = new Flux2.Flux2Kernels(acc);
        _video = acc.Allocate1D<float>((long)B * F * H * W * Channels); _audio = acc.Allocate1D<float>((long)B * Ta * Channels);
        _vctx = acc.Allocate1D<float>((long)B * L * Ltx2Transformer.Caption); _actx = acc.Allocate1D<float>((long)B * L * Ltx2Transformer.Caption);
    }

    /// <summary>
    /// Open the transformer of an LTX-2 checkpoint for calls of <paramref name="batch"/> items of a [128, F, H, W] video latent and an [8, Ta, 16] audio latent with
    /// <paramref name="contextTokens"/> conditioning rows (1024). <paramref name="residentBytes"/>: transformer blocks kept on the device; the rest stream per call.
    /// </summary>
    public static async Task<Ltx2AvModel> OpenAsync(Accelerator accelerator, IModelSource source, string repo, string file, int batch, int F, int H, int W, int Ta, int contextTokens,
        long residentBytes, CancellationToken ct = default)
    {
        var t = await Ltx2Transformer.OpenAsync(accelerator, source, repo, file, batch, F * H * W, Ta, contextTokens, residentBytes, ct);
        try { return new Ltx2AvModel(accelerator, t, batch, F, H, W, Ta, contextTokens); }
        catch { t.Dispose(); throw; }
    }

    /// <summary>
    /// One model call: <paramref name="packed"/> [B, PackedLength], <paramref name="sigmas"/> [B], <paramref name="context"/> [B, L, 7680] on the device. Returns the denoised packed
    /// latents [B, PackedLength] (x - sigma v).
    /// </summary>
    public async Task<float[]> DenoiseAsync(float[] packed, IReadOnlyList<float> sigmas, FV context, float fps, CancellationToken ct = default)
    {
        int B = _B, P = PackedLength, nv = _F * _H * _W;
        if (packed.Length != (long)B * P) throw new ArgumentException($"packed: {B} x {P} values");
        if (sigmas.Count != B) throw new ArgumentException($"{B} sigmas");
        if (context.Length < (long)B * _L * ContextWidth) throw new ArgumentException($"context: [{B}, {_L}, {ContextWidth}]");
        // tokens: video [nv, 128] (transpose of [128, nv]); audio [Ta, 8 x 16] (permute [8, Ta, 16])
        var v = new float[(long)B * nv * Channels]; var a = new float[(long)B * _Ta * Channels];
        for (int b = 0; b < B; b++)
        {
            int o = b * P;
            for (int c = 0; c < Channels; c++) for (int t = 0; t < nv; t++) v[((long)b * nv + t) * Channels + c] = packed[o + c * nv + t];
            int oa = o + Channels * nv;
            for (int c = 0; c < AudioChannels; c++) for (int f = 0; f < _Ta; f++) for (int m = 0; m < AudioBins; m++)
                a[((long)b * _Ta + f) * Channels + c * AudioBins + m] = packed[oa + (c * _Ta + f) * AudioBins + m];
        }
        _video.View.CopyFromCPU(v); _audio.View.CopyFromCPU(a);
        _f.CopyCols(context, _vctx.View, B * _L, Ltx2Transformer.Caption, ContextWidth, Ltx2Transformer.Caption, 0);
        _f.CopyCols(context.SubView(Ltx2Transformer.Caption, context.Length - Ltx2Transformer.Caption), _actx.View, B * _L, Ltx2Transformer.Caption, ContextWidth, Ltx2Transformer.Caption, 0);
        var ts = sigmas.Select(s => s * 1000f).ToArray();
        var (vo, ao) = await _t.CallAsync(_video.View, _audio.View, _vctx.View, _actx.View, ts, _F, _H, _W, _Ta, fps, ct);
        var vel = await vo.CopyToHostAsync(); var avel = await ao.CopyToHostAsync();
        var den = new float[packed.Length];
        for (int b = 0; b < B; b++)
        {
            int o = b * P; float s = sigmas[b];
            for (int c = 0; c < Channels; c++) for (int t = 0; t < nv; t++)
            { int i = o + c * nv + t; den[i] = packed[i] - s * vel[((long)b * nv + t) * Channels + c]; }
            int oa = o + Channels * nv;
            for (int c = 0; c < AudioChannels; c++) for (int f = 0; f < _Ta; f++) for (int m = 0; m < AudioBins; m++)
            { int i = oa + (c * _Ta + f) * AudioBins + m; den[i] = packed[i] - s * avel[((long)b * _Ta + f) * Channels + c * AudioBins + m]; }
        }
        return den;
    }

    /// <summary>
    /// The template's sampler (MEASURED against ComfyUI's two calls: step to 3.8e-7, final latent to 6e-8 from its own outputs): classifier-free guidance on the denoised
    /// latents, <c>D = uncond + cfg (cond - uncond)</c>, then Euler <c>x += (sigma_next - sigma) (x - D) / sigma</c> over <paramref name="sigmaSchedule"/> (ends at 0). The batch is
    /// [uncond, cond] (ComfyUI's order): <paramref name="context"/> is [2, L, 7680] with the negative prompt's conditioning first. Needs a model opened with batch 2.
    /// Returns the final packed latent [PackedLength].
    /// </summary>
    public async Task<float[]> SampleAsync(float[] noise, FV context, IReadOnlyList<float> sigmaSchedule, float cfg, float fps, Action<int, float[]>? onStep = null, CancellationToken ct = default)
    {
        // batch 1: no guidance (cfg 1: ComfyUI calls the conditioned model only); batch 2: [uncond, cond]
        if (_B is not (1 or 2)) throw new InvalidOperationException("a model opened with batch 1 (no guidance) or 2 ([uncond, cond])");
        if (_B == 1 && cfg != 1f) throw new ArgumentException("guidance (cfg != 1) needs a model opened with batch 2");
        int P = PackedLength;
        if (noise.Length != P) throw new ArgumentException($"noise: {P} values");
        var x = (float[])noise.Clone();
        for (int i = 0; i + 1 < sigmaSchedule.Count; i++)
        {
            float s = sigmaSchedule[i], next = sigmaSchedule[i + 1];
            var batch = new float[_B * P]; for (int b = 0; b < _B; b++) Array.Copy(x, 0, batch, b * P, P);
            var d = await DenoiseAsync(batch, _B == 1 ? new[] { s } : new[] { s, s }, context, fps, ct);
            for (int k = 0; k < P; k++)
            {
                float den = _B == 1 ? d[k] : d[k] + (d[P + k] - d[k]) * cfg;       // ComfyUI's cfg: uncond + (cond - uncond) * cfg
                float slope = (x[k] - den) / s;                                       // its Euler: d = (x - denoised) / sigma; x = x + d (sigma_next - sigma)
                x[k] = x[k] + slope * (next - s);
            }
            onStep?.Invoke(i, x);
        }
        return x;
    }

    public void Dispose()
    {
        _video.Dispose(); _audio.Dispose(); _vctx.Dispose(); _actx.Dispose();
        _t.Dispose();
    }
}
