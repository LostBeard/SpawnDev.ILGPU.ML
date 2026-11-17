using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Hub;
using FV = ILGPU.Runtime.ArrayView1D<float, ILGPU.Stride1D.Dense>;

namespace SpawnDev.ILGPU.ML.Ltx2;

/// <summary>
/// A keyframe guide on an LTX-2 call (ComfyUI's LTXVAddGuide): <paramref name="GuideFrames"/> latent frames appended after the video's own, <paramref name="KeyframeBoxes"/>
/// keyframe_idxs [3, guide tokens, 2] (pixel-space [start, end) per axis t / h / w), <paramref name="Strength"/> the guide's strength, <paramref name="FrameMask"/> the
/// denoise_mask per latent frame ([F] or [B, F]; a guide frame at 1 - its strength).
/// <para>Several guides (chained LTXVAddGuide) are one record: their frames in add order, their boxes concatenated, and <paramref name="TokenStrengths"/> [guide tokens] =
/// each guide token's attention strength (its guide's strength, times the guide's attention mask at latent resolution when it has one; 0 = blocked). Null: every guide
/// token at <paramref name="Strength"/>. MEASURED on ComfyUI's attention masks: a noisy query's score to a guide key gets ln(the key's strength), a guide query's score to a
/// noisy key ln(its own), guide tokens among themselves nothing (also between two guides).</para>
/// </summary>
public sealed record Ltx2Guide(int GuideFrames, float[] KeyframeBoxes, float Strength, float[] FrameMask, float[]? TokenStrengths = null);

/// <summary>
/// LTX-2's model call in ComfyUI's own form (what SpawnGen's sampler calls): the packed audio-video latent of each batch item, <c>[video [128, F, H, W] row-major, then audio
/// [8, Ta, 16] row-major]</c> (128 F H W + 128 Ta values), sigma per item, the conditioning [B, L, 7680] (video connector | audio connector) and the frame rate, in; the DENOISED
/// packed latent <c>x - sigma v</c> out (MEASURED: ComfyUI's call output equals x - sigma v exactly). Wraps <see cref="Ltx2Transformer"/> (sigma x 1000 as its timestep).
/// <para>Token layouts (MEASURED bit-exact against ComfyUI's patchify inputs by SpawnGen): video token f H W + h W + w, feature = channel; audio token = latent frame,
/// feature c 16 + bin.</para>
/// <para>The call and the sampler run on the device: the device-view overloads take and return views (nothing crosses to the host); the float[] overloads upload, run the
/// same path, and read the result back.</para>
/// </summary>
public sealed class Ltx2AvModel : IDisposable
{
    public const int Channels = 128, AudioChannels = 8, AudioBins = 16, ContextWidth = 2 * Ltx2Transformer.Caption;

    private readonly Accelerator _acc;
    private readonly Ltx2Transformer _t;
    private readonly Flux2.Flux2Kernels _f;
    private readonly Ltx2Kernels _l;
    private readonly int _B, _F, _H, _W, _Ta, _L;
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _video, _audio, _vctx, _actx;
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _batch, _den, _x, _sig, _fmask;      // call input [B, P], denoised [B, P], the sampler's state [P], sigma [B], frame mask [F]

    /// <summary>Values of one item's packed latent: 128 F H W + 128 Ta.</summary>
    public int PackedLength => Channels * _F * _H * _W + AudioChannels * _Ta * AudioBins;

    /// <summary>The transformer (probes, streaming timers, precision).</summary>
    public Ltx2Transformer Transformer => _t;

    private Ltx2AvModel(Accelerator acc, Ltx2Transformer t, int B, int F, int H, int W, int Ta, int L)
    {
        _acc = acc; _t = t; _B = B; _F = F; _H = H; _W = W; _Ta = Ta; _L = L; _f = new Flux2.Flux2Kernels(acc); _l = new Ltx2Kernels(acc);
        _video = acc.Allocate1D<float>((long)B * F * H * W * Channels); _audio = acc.Allocate1D<float>((long)B * Ta * Channels);
        _vctx = acc.Allocate1D<float>((long)B * L * Ltx2Transformer.Caption); _actx = acc.Allocate1D<float>((long)B * L * Ltx2Transformer.Caption);
        long P = PackedLength;
        _batch = acc.Allocate1D<float>(B * P); _den = acc.Allocate1D<float>(B * P); _x = acc.Allocate1D<float>(P);
        _sig = acc.Allocate1D<float>(B); _fmask = acc.Allocate1D<float>(F);
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
    /// One model call on the device: <paramref name="packed"/> [B, PackedLength], <paramref name="sigmas"/> [B], <paramref name="context"/> [B, L, 7680]. Returns the denoised
    /// packed latents [B, PackedLength] (x - sigma v) as a device view, valid until the next call.
    /// <para>With a keyframe <paramref name="guide"/> (ComfyUI's LTXVAddGuide conditioning): the packed video latent holds <see cref="Ltx2Guide.GuideFrames"/> appended guide
    /// frames; <see cref="Ltx2Guide.FrameMask"/> is ComfyUI's denoise_mask per latent frame (each frame's timestep is sigma x mask; MEASURED: the denoised output still uses
    /// the global sigma, x - sigma v, exactly); <see cref="Ltx2Guide.KeyframeBoxes"/> is keyframe_idxs [3, guide tokens, 2] (the guide tokens' pixel boxes for RoPE);
    /// a guide token strength &lt; 1 (<see cref="Ltx2Guide.Strength"/> / <see cref="Ltx2Guide.TokenStrengths"/>) adds its logarithm to the self-attention between noisy and
    /// guide tokens.</para>
    /// </summary>
    public async Task<FV> DenoiseAsync(FV packed, IReadOnlyList<float> sigmas, FV context, float fps, Ltx2Guide? guide = null, CancellationToken ct = default)
    {
        int B = _B, P = PackedLength, nv = _F * _H * _W;
        if (packed.Length < (long)B * P) throw new ArgumentException($"packed: {B} x {P} values");
        if (sigmas.Count != B) throw new ArgumentException($"{B} sigmas");
        if (context.Length < (long)B * _L * ContextWidth) throw new ArgumentException($"context: [{B}, {_L}, {ContextWidth}]");
        // a caller feeding the previous call's result back in: the result buffer is written below, so the input moves to the call buffer first
        if (SameBuffer(packed, _den.View)) { _batch.View.CopyFrom(packed.SubView(0, (long)B * P)); packed = _batch.View; }
        // tokens: video [nv, 128] (transpose of [128, nv]); audio [Ta, 8 x 16] (permute [8, Ta, 16])
        _l.UnpackTokens(packed, _video.View, _audio.View, B, nv, _Ta);
        _f.CopyCols(context, _vctx.View, B * _L, Ltx2Transformer.Caption, ContextWidth, Ltx2Transformer.Caption, 0);
        _f.CopyCols(context.SubView(Ltx2Transformer.Caption, context.Length - Ltx2Transformer.Caption), _actx.View, B * _L, Ltx2Transformer.Caption, ContextWidth, Ltx2Transformer.Caption, 0);
        var ts = sigmas.Select(s => s * 1000f).ToArray();
        float[]? boxes = null; (int, float[])? bias = null; IReadOnlyList<float>? mults = guide?.FrameMask;
        if (guide != null)
        {
            int gf = guide.GuideFrames, hw = _H * _W, ng = gf * hw, grid = (_F - gf) * hw;
            if (gf < 1 || gf >= _F) throw new ArgumentException($"guide frames {gf} of {_F} latent frames");
            if (guide.KeyframeBoxes.Length != 3 * ng * 2) throw new ArgumentException($"keyframe boxes: [3, {ng}, 2] (the guide's tokens)");
            boxes = new float[_F * hw * 6];
            Array.Copy(Ltx2Transformer.GridBoxes(_F - gf, _H, _W), boxes, grid * 6);
            for (int j = 0; j < ng; j++)
                for (int ax = 0; ax < 3; ax++)
                {
                    boxes[(grid + j) * 6 + 2 * ax] = guide.KeyframeBoxes[(ax * ng + j) * 2];
                    boxes[(grid + j) * 6 + 2 * ax + 1] = guide.KeyframeBoxes[(ax * ng + j) * 2 + 1];
                }
            if (guide.TokenStrengths != null && guide.TokenStrengths.Length != ng) throw new ArgumentException($"token strengths: one per guide token ({ng})");
            var tb = new float[nv]; bool any = false;
            for (int j = 0; j < ng; j++)
            {
                float st = guide.TokenStrengths?[j] ?? guide.Strength;
                if (st >= 1f) continue;                                                      // strength 1: ComfyUI builds no mask (ln 1 = 0)
                tb[grid + j] = st > 0f ? Math.Max(MathF.Log(st), Trellis2.DiTKernels.BlockedTokenBias) : Trellis2.DiTKernels.BlockedTokenBias;
                any = true;
            }
            if (any) bias = (grid, tb);
        }
        var (vo, ao) = await _t.CallAsync(_video.View, _audio.View, _vctx.View, _actx.View, ts, mults, boxes, bias, _F, _H, _W, _Ta, fps, ct);
        _sig.View.CopyFromCPU(sigmas.ToArray());                                             // B scalars
        _l.DenoisedPacked(packed, vo, ao, _sig.View, _den.View, B, nv, _Ta);
        return _den.View;
    }

    /// <summary>The model call from and to host arrays: uploads <paramref name="packed"/> [B, PackedLength], runs the device call, reads the denoised latents back.</summary>
    public Task<float[]> DenoiseAsync(float[] packed, IReadOnlyList<float> sigmas, FV context, float fps, CancellationToken ct = default)
        => DenoiseAsync(packed, sigmas, context, fps, null, ct);

    /// <summary>The model call with a keyframe guide, from and to host arrays (see the device overload).</summary>
    public async Task<float[]> DenoiseAsync(float[] packed, IReadOnlyList<float> sigmas, FV context, float fps, Ltx2Guide? guide, CancellationToken ct = default)
    {
        if (packed.Length != (long)_B * PackedLength) throw new ArgumentException($"packed: {_B} x {PackedLength} values");
        _batch.View.CopyFromCPU(packed);
        var den = await DenoiseAsync(_batch.View, sigmas, context, fps, guide, ct);
        return await den.CopyToHostAsync();
    }

    /// <summary>
    /// The template's sampler on the device (MEASURED against ComfyUI's two calls: step to 3.8e-7, final latent to 6e-8 from its own outputs): classifier-free guidance on the
    /// denoised latents, <c>D = uncond + cfg (cond - uncond)</c>, then Euler <c>x += (sigma_next - sigma) (x - D) / sigma</c> over <paramref name="sigmaSchedule"/> (ends at 0).
    /// The batch is [uncond, cond] (ComfyUI's order): <paramref name="context"/> is [2, L, 7680] with the negative prompt's conditioning first; a model opened with batch 1 runs
    /// without guidance (cfg 1). <paramref name="noise"/> [PackedLength] is the start state. Returns the final packed latent [PackedLength] as a device view, valid until the
    /// next sampling run. <paramref name="onStep"/> gets the state after each step (the same view).
    /// </summary>
    public Task<FV> SampleAsync(FV noise, FV context, IReadOnlyList<float> sigmaSchedule, float cfg, float fps, Func<int, FV, Task>? onStep = null, CancellationToken ct = default)
        => SampleAsync(noise, context, sigmaSchedule, cfg, fps, null, default, onStep, ct);

    /// <summary>
    /// The sampler with a keyframe <paramref name="guide"/> (ComfyUI's masked sampling, MEASURED 2026-11-17 on its guided runs at strengths 1 and 0.6): with m the latent
    /// frame's denoise mask (<see cref="Ltx2Guide.FrameMask"/>, the same for every batch row; audio 1) and <paramref name="latent"/> [PackedLength] the clean latent (the
    /// guide frames' encodings; free frames are not read), the sampler keeps its own state x, the model sees <c>m x + (1 - m) latent</c>, and the step uses
    /// <c>D = m D + (1 - m) latent</c> (a guide at strength 1 stays the clean latent throughout; the final latent is the last D). <paramref name="noise"/> is the start
    /// state, guide frames included.
    /// </summary>
    public async Task<FV> SampleAsync(FV noise, FV context, IReadOnlyList<float> sigmaSchedule, float cfg, float fps, Ltx2Guide? guide, FV latent, Func<int, FV, Task>? onStep = null,
        CancellationToken ct = default)
    {
        // batch 1: no guidance (cfg 1: ComfyUI calls the conditioned model only); batch 2: [uncond, cond]
        if (_B is not (1 or 2)) throw new InvalidOperationException("a model opened with batch 1 (no guidance) or 2 ([uncond, cond])");
        if (_B == 1 && cfg != 1f) throw new ArgumentException("guidance (cfg != 1) needs a model opened with batch 2");
        int P = PackedLength, nv = _F * _H * _W, hw = _H * _W;
        if (noise.Length < P) throw new ArgumentException($"noise: {P} values");
        if (guide != null)
        {
            if (latent.Length < P) throw new ArgumentException($"latent: {P} values (the clean latent the masked frames keep)");
            var fm = guide.FrameMask;
            if (fm.Length != _F && fm.Length != _B * _F) throw new ArgumentException($"frame mask: [{_F}] or [{_B}, {_F}]");
            for (int i = _F; i < fm.Length; i++) if (fm[i] != fm[i % _F]) throw new ArgumentException("the sampler needs the same frame mask for every batch row");
            _fmask.View.CopyFromCPU(fm.AsSpan(0, _F).ToArray());                             // F scalars
        }
        if (!SameBuffer(noise, _x.View)) _x.View.CopyFrom(noise.SubView(0, P));            // the previous run's result as the start state: already in place
        else if (((IContiguousArrayView)noise.BaseView).Index != 0) throw new ArgumentException("noise is a slice of this model's sampler state that does not start at its first value");
        var sig = new float[_B];
        for (int i = 0; i + 1 < sigmaSchedule.Count; i++)
        {
            float s = sigmaSchedule[i], next = sigmaSchedule[i + 1];
            if (guide != null) _l.BlendRows(_x.View, latent, _fmask.View, _batch.View, _B, P, nv, hw);
            else for (int b = 0; b < _B; b++) _batch.View.SubView((long)b * P, P).CopyFrom(_x.View);
            Array.Fill(sig, s);
            var den = await DenoiseAsync(_batch.View, sig, context, fps, guide, ct);
            // ComfyUI's cfg and Euler: d = (x - denoised) / sigma; x = x + d (sigma_next - sigma). 1 / sigma on the host (fp32 division is not exact on every backend)
            if (guide != null) _l.EulerCfgStep(_x.View, den, P, _B, cfg, 1f / s, next - s, latent, _fmask.View, nv, hw);
            else _l.EulerCfgStep(_x.View, den, P, _B, cfg, 1f / s, next - s);
            if (onStep != null) await onStep(i, _x.View);
        }
        return _x.View;
    }

    /// <summary>The sampler from and to host arrays (see the device overload): returns the final packed latent [PackedLength]; <paramref name="onStep"/> gets a copy of the state after each step.</summary>
    public Task<float[]> SampleAsync(float[] noise, FV context, IReadOnlyList<float> sigmaSchedule, float cfg, float fps, Action<int, float[]>? onStep = null, CancellationToken ct = default)
        => SampleAsync(noise, context, sigmaSchedule, cfg, fps, null, null, onStep, ct);

    /// <summary>The guided sampler from and to host arrays (see the device overload): <paramref name="latent"/> [PackedLength] is the clean latent (needed with a guide).</summary>
    public async Task<float[]> SampleAsync(float[] noise, FV context, IReadOnlyList<float> sigmaSchedule, float cfg, float fps, Ltx2Guide? guide, float[]? latent,
        Action<int, float[]>? onStep = null, CancellationToken ct = default)
    {
        int P = PackedLength;
        if (noise.Length != P) throw new ArgumentException($"noise: {P} values");
        if (guide != null && (latent == null || latent.Length != P)) throw new ArgumentException($"latent: {P} values (the clean latent the masked frames keep)");
        using var start = _acc.Allocate1D(noise);
        using var clean = guide != null ? _acc.Allocate1D(latent!) : null;
        try
        {
            var x = await SampleAsync(start.View, context, sigmaSchedule, cfg, fps, guide, clean != null ? clean.View : default,
                onStep == null ? null : async (i, v) => onStep(i, await v.CopyToHostAsync()), ct);
            return await x.CopyToHostAsync();
        }
        finally { await _acc.SynchronizeAsync(); }                                           // nothing queued may still read the two uploads when they are disposed
    }

    private static bool SameBuffer(FV a, FV b) => a.IsValid && b.IsValid && ((IContiguousArrayView)a.BaseView).Buffer == ((IContiguousArrayView)b.BaseView).Buffer;

    public void Dispose()
    {
        _video.Dispose(); _audio.Dispose(); _vctx.Dispose(); _actx.Dispose();
        _batch.Dispose(); _den.Dispose(); _x.Dispose(); _sig.Dispose(); _fmask.Dispose();
        _t.Dispose();
    }
}
