using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.SkinTokens;
using SpawnDev.ILGPU.ML.Trellis2;
using FV = ILGPU.Runtime.ArrayView1D<float, ILGPU.Stride1D.Dense>;

namespace SpawnDev.ILGPU.ML.Ltx2;

/// <summary>
/// The LTX-2 19B audio + video transformer (diffusers' LTX2VideoTransformer3DModel, Apache-2.0; the checkpoint's own config: split RoPE, no gated attention, no prompt adaLN):
/// 48 blocks of a video stream (32 x 128 = 4096) and an audio stream (32 x 64 = 2048). Each block: video and audio self-attention (adaLN-modulated RMS norm, gated
/// residual), text cross-attention (plain RMS norm, ungated), audio-to-video then video-to-audio cross-attention (the second reads the video after the first's update, as
/// ComfyUI runs it; audio head size, time-only RoPE), and the gated feed-forwards. The q / k norms use epsilon 1e-5. Output: velocity for both streams (x0 = x - sigma v).
/// <para>
/// One call runs a batch (the guidance pair: [cond, uncond]) through each block while the block's weights are loaded, so a step streams the 19B weights once. Blocks stream
/// from the checkpoint as stored (fp8 or bf16); the base set (timestep embedders, projections) stays resident.
/// </para>
/// </summary>
public sealed class Ltx2Transformer : IDisposable
{
    public const int Dv = 4096, Da = 2048, Hv = 32, HeadV = 128, Ha = 32, HeadA = 64, Layers = 48, InChannels = 128, Caption = 3840, TimeChannels = 256, Fv = 4 * Dv, Fa = 4 * Da;
    public const float Eps = 1e-6f;
    /// <summary>
    /// The epsilon of the attention q / k RMS norms: 1e-5 (MEASURED 2026-11-17 against ComfyUI's block 0, float64 reference: 1e-6 leaves the video stream 1.9e-3 off, 1e-5 3e-7).
    /// The block norms use <see cref="Eps"/>.
    /// </summary>
    public const float QkEps = 1e-5f;
    public const double RopeTheta = 10_000.0;
    /// <summary>The positional normalisers of the checkpoint config (positional_embedding_max_pos [20, 2048, 2048]; audio [20]).</summary>
    public const float MaxTime = 20f, MaxSpace = 2048f;
    /// <summary>The VAE's compression: 8 frames and 32 x 32 pixels per latent voxel; the audio latent's frame is 4 mel frames of 160 samples at 16 kHz.</summary>
    public const int TimeScale = 8, SpaceScale = 32, AudioScale = 4, AudioHop = 160, AudioRate = 16000;

    private readonly Accelerator _acc;
    private readonly StreamedModelWeights _w;
    private readonly DiTKernels _k;
    private readonly Ltx.LtxKernels _x;
    private readonly Ltx2Kernels _l;
    private readonly SkinTokensKernels _u;
    private readonly Flux2.Flux2Kernels _f;
    private readonly List<IDisposable> _owned = new();
    private readonly int _B, _nv, _na, _L;
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _v, _a, _vn, _an, _vctx, _actx, _q, _kv, _part, _o, _hid, _scores, _xs;
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _sin256, _t1, _t1a, _temb, _tembS;
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _tV, _tVe, _tA, _tAe, _tCaV, _tCaA, _tGateV, _tGateA;     // the six embedders' outputs (mod rows, embedded timestep)
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _modV, _modA, _caV, _caA, _gateV, _gateA, _headV, _headA;
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _ropeV, _ropeVs, _ropeA, _ropeAs, _ropeVx, _ropeVxs, _outV, _outA;

    /// <summary>Called with ("block{k}_video" [B nv, 4096] | "block{k}_audio" [B na, 2048] | "adaln" ... ) as a call goes.</summary>
    public Action<string, FV, int, int>? Probe;

    /// <summary>CUDA: fp32-class activations through the tensor-core GEMMs (on: the ComfyUI reference runs float32).</summary>
    public bool PreciseActivations { get => _k.PreciseActivations; set => _k.PreciseActivations = value; }

    /// <summary>The streamed weights (timers, prefetch, host cache).</summary>
    public StreamedModelWeights Weights => _w;

    private Ltx2Transformer(Accelerator accelerator, StreamedModelWeights weights, int batch, int videoTokens, int audioTokens, int contextTokens)
    {
        _acc = accelerator; _w = weights; _B = batch; _nv = videoTokens; _na = audioTokens; _L = contextTokens;
        _k = new DiTKernels(accelerator) { PreciseActivations = true }; _x = new Ltx.LtxKernels(accelerator); _l = new Ltx2Kernels(accelerator); _u = new SkinTokensKernels(accelerator); _f = new Flux2.Flux2Kernels(accelerator);
        long B = batch, nv = videoTokens, na = audioTokens, L = contextTokens, rows = Math.Max(nv, Math.Max(na, L));
        _v = Alloc(B * nv * Dv); _a = Alloc(B * na * Da); _vn = Alloc(B * nv * Dv); _an = Alloc(B * na * Da);
        _vctx = Alloc(B * L * Dv); _actx = Alloc(B * L * Da);
        _q = Alloc(B * rows * Dv); _kv = Alloc(B * rows * 2 * Dv); _part = Alloc(B * rows * Dv); _o = Alloc(B * rows * Dv);
        _hid = Alloc(B * Math.Max(nv * Fv, Math.Max(na * Fa, L * Dv))); _xs = Alloc(B * Math.Max(nv * Fv, Math.Max(na * Fa, L * Caption)));
        _scores = Alloc(Math.Max(Hv, Ha) * Math.Max(Math.Max(nv * nv, na * na), Math.Max(Math.Max(nv, na) * L, nv * na)));
        _sin256 = Alloc(B * TimeChannels); _t1 = Alloc(B * Dv); _t1a = Alloc(B * Dv); _temb = Alloc(B * Dv); _tembS = Alloc(B * Dv);
        _tV = Alloc(B * 6 * Dv); _tVe = Alloc(B * Dv); _tA = Alloc(B * 6 * Da); _tAe = Alloc(B * Da);
        _tCaV = Alloc(B * 4 * Dv); _tCaA = Alloc(B * 4 * Da); _tGateV = Alloc(B * Dv); _tGateA = Alloc(B * Da);
        _modV = Alloc(B * 6 * Dv); _modA = Alloc(B * 6 * Da); _caV = Alloc(B * 4 * Dv); _caA = Alloc(B * 4 * Da); _gateV = Alloc(B * Dv); _gateA = Alloc(B * Da); _headV = Alloc(B * 2 * Dv); _headA = Alloc(B * 2 * Da);
        _ropeV = Alloc(nv * Dv / 2); _ropeVs = Alloc(nv * Dv / 2); _ropeA = Alloc(na * Da / 2); _ropeAs = Alloc(na * Da / 2); _ropeVx = Alloc(nv * Da / 2); _ropeVxs = Alloc(nv * Da / 2);
        _outV = Alloc(B * nv * InChannels); _outA = Alloc(B * na * InChannels);
    }

    private MemoryBuffer1D<float, Stride1D.Dense> Alloc(long n) { var b = _acc.Allocate1D<float>(Math.Max(1, n)); _owned.Add(b); return b; }

    /// <summary>
    /// Open the transformer of an LTX-2 checkpoint (any prefix: the original layout's <c>model.diffusion_model.</c>) for calls of <paramref name="batch"/> items of
    /// <paramref name="videoTokens"/> video and <paramref name="audioTokens"/> audio tokens and <paramref name="contextTokens"/> text rows. <paramref name="residentBytes"/>: blocks kept
    /// on the device as stored, in order; the rest stream per call.
    /// </summary>
    public static async Task<Ltx2Transformer> OpenAsync(Accelerator accelerator, IModelSource source, string repo, string file, int batch, int videoTokens, int audioTokens, int contextTokens,
        long residentBytes, CancellationToken ct = default)
    {
        static int BlockOf(string n)
        {
            if (n.StartsWith("transformer_blocks.", StringComparison.Ordinal)) { int dot = n.IndexOf('.', 19); return int.Parse(n.AsSpan(19, dot - 19)); }
            if (n.Contains("_embeddings_connector.", StringComparison.Ordinal)) return StreamedModelWeights.HostOnly;   // Ltx2TextConnectors loads them
            return -1;
        }
        var w = await StreamedModelWeights.OpenAsync(accelerator, source, repo, file, "patchify_proj.weight", BlockOf, Layers, residentBytes, null, ct);
        try
        {
            if (!w.HeaderNames.Contains($"transformer_blocks.{Layers - 1}.audio_ff.net.2.weight") || w.HeaderNames.Contains($"transformer_blocks.{Layers}.ff.net.2.weight"))
                throw new InvalidDataException($"'{file}' is not LTX-2's 48-block audio-video transformer");
            return new Ltx2Transformer(accelerator, w, batch, videoTokens, audioTokens, contextTokens);
        }
        catch { w.Dispose(); throw; }
    }

    // nn.Linear over M rows (+ bias when the checkpoint has one); an 8-bit matrix runs exact (its decode without the scale, the input times the scale)
    private void Lin(GpuWeights w, FV x, string name, FV o, int M, int K, int N, DiTKernels.Act act = DiTKernels.Act.None)
    {
        float s = _w.ActivationScale(w, name + ".weight");
        if (s != 1f) { var xs = _xs.View.SubView(0, (long)M * K); _f.ScaleInto(x, xs, s, M * K); x = xs; }
        string bias = name + ".bias";
        _k.Linear(x, _w.MatrixExact(w, name + ".weight"), w.Has(bias) ? w.BF16(bias) : null, o, M, K, N, act);
    }

    /// <summary>The 256-wide sinusoid of each timestep (diffusers Timesteps(256, flip_sin_to_cos = True, shift 0): [cos | sin], frequency exp(-ln(1e4) i / 128)).</summary>
    public static float[] Sinusoid(IReadOnlyList<float> t)
    {
        const int half = TimeChannels / 2;
        var o = new float[t.Count * TimeChannels];
        for (int b = 0; b < t.Count; b++)
            for (int i = 0; i < half; i++)
            {
                float f = MathF.Exp(-MathF.Log(10000f) * i / half);
                float a = t[b] * f;
                o[b * TimeChannels + i] = (float)Math.Cos(a); o[b * TimeChannels + half + i] = (float)Math.Sin(a);
            }
        return o;
    }

    // one adaLN embedder: emb = linear_2(silu(linear_1(sinusoid))) [B, D]; mod = linear(silu(emb)) [B, N D]
    private void Embedder(string name, int D, FV mod, FV emb)
    {
        var b = _w.Base;
        Lin(b, _sin256.View, name + ".emb.timestep_embedder.linear_1", _t1.View, _B, TimeChannels, D, DiTKernels.Act.Silu);
        Lin(b, _t1.View, name + ".emb.timestep_embedder.linear_2", emb, _B, D, D);
        _k.Silu(emb, _tembS.View, _B * D);
        Lin(b, _tembS.View, name + ".linear", mod, _B, D, b.Shape(name + ".linear.weight")[0]);
    }

    /// <summary>
    /// The RoPE tables of a call (MEASURED against ComfyUI's dumped tables, 6e-8): video tokens (f, h, w) of a [F, H, W] latent, audio tokens of <paramref name="audioFrames"/>
    /// latent frames, the video frame rate <paramref name="fps"/>.
    /// </summary>
    private void UploadRope(int F, int H, int W, int audioFrames, float fps)
    {
        int nv = F * H * W;
        var tv = new float[nv]; var hv = new float[nv]; var wv = new float[nv];
        for (int f = 0, i = 0; f < F; f++)
            for (int h = 0; h < H; h++)
                for (int x = 0; x < W; x++, i++)
                {
                    float t0 = Math.Max(TimeScale * f + 1 - TimeScale, 0), t1 = Math.Max(TimeScale * (f + 1) + 1 - TimeScale, 0);   // causal: frame 0 is one pixel frame
                    tv[i] = (t0 + t1) / 2f / fps / MaxTime;
                    hv[i] = (SpaceScale * h + SpaceScale / 2f) / MaxSpace; wv[i] = (SpaceScale * x + SpaceScale / 2f) / MaxSpace;
                }
        // video self: 682 frequencies per axis, frequency-major with the axis fastest, 2 identity entries first
        int Fq = Dv / 6, pad = Dv / 2 - 3 * Fq;
        var (ct, st) = Ltx2Kernels.SplitRopeTables(tv, Fq, 0, RopeTheta); var (ch, sh) = Ltx2Kernels.SplitRopeTables(hv, Fq, 0, RopeTheta); var (cw, sw) = Ltx2Kernels.SplitRopeTables(wv, Fq, 0, RopeTheta);
        var cos = new float[nv * Dv / 2]; var sin = new float[nv * Dv / 2];
        for (int i = 0; i < nv; i++)
        {
            int r = i * Dv / 2;
            for (int j = 0; j < pad; j++) { cos[r + j] = 1f; sin[r + j] = 0f; }
            for (int j = 0; j < Fq; j++)
            {
                cos[r + pad + 3 * j] = ct[i * Fq + j]; sin[r + pad + 3 * j] = st[i * Fq + j];
                cos[r + pad + 3 * j + 1] = ch[i * Fq + j]; sin[r + pad + 3 * j + 1] = sh[i * Fq + j];
                cos[r + pad + 3 * j + 2] = cw[i * Fq + j]; sin[r + pad + 3 * j + 2] = sw[i * Fq + j];
            }
        }
        _ropeV.View.SubView(0, cos.Length).CopyFromCPU(cos); _ropeVs.View.SubView(0, sin.Length).CopyFromCPU(sin);
        // audio self and cross: 1024 frequencies of the audio frame's time
        var ta = new float[audioFrames];
        for (int f = 0; f < audioFrames; f++)
        {
            float s0 = Math.Max(AudioScale * f + 1 - AudioScale, 0), s1 = Math.Max(AudioScale * (f + 1) + 1 - AudioScale, 0);
            ta[f] = (s0 + s1) / 2f * AudioHop / AudioRate / MaxTime;
        }
        var (ca, sa) = Ltx2Kernels.SplitRopeTables(ta, Da / 2, 0, RopeTheta);
        _ropeA.View.SubView(0, ca.Length).CopyFromCPU(ca); _ropeAs.View.SubView(0, sa.Length).CopyFromCPU(sa);
        // video cross (to audio): 1024 frequencies of the video token's time
        var (cx, sx) = Ltx2Kernels.SplitRopeTables(tv, Da / 2, 0, RopeTheta);
        _ropeVx.View.SubView(0, cx.Length).CopyFromCPU(cx); _ropeVxs.View.SubView(0, sx.Length).CopyFromCPU(sx);
    }

    /// <summary>
    /// One call: <paramref name="video"/> [B, F H W, 128] and <paramref name="audio"/> [B, audioFrames, 128] (the packed latents), the text <paramref name="videoContext"/> /
    /// <paramref name="audioContext"/> [B, L, 3840] (the connectors' outputs), the timesteps <paramref name="timesteps"/> (sigma x 1000) per item. Returns the velocities
    /// ([B, F H W, 128], [B, audioFrames, 128]) on the device, valid until the next call.
    /// </summary>
    public async Task<(FV Video, FV Audio)> CallAsync(FV video, FV audio, FV videoContext, FV audioContext, IReadOnlyList<float> timesteps, int F, int H, int W, int audioFrames, float fps,
        CancellationToken ct = default)
    {
        if (F * H * W != _nv || audioFrames != _na) throw new ArgumentException($"this transformer was opened for {_nv} video / {_na} audio tokens");
        if (timesteps.Count != _B) throw new ArgumentException($"{_B} timesteps (one per batch item)");
        int B = _B, nv = _nv, na = _na, L = _L;
        var b0 = _w.Base;
        UploadRope(F, H, W, audioFrames, fps);
        _sin256.View.SubView(0, B * TimeChannels).CopyFromCPU(Sinusoid(timesteps));
        Embedder("adaln_single", Dv, _tV.View, _tVe.View);
        Embedder("audio_adaln_single", Da, _tA.View, _tAe.View);
        Embedder("av_ca_video_scale_shift_adaln_single", Dv, _tCaV.View, _temb.View);
        Embedder("av_ca_audio_scale_shift_adaln_single", Da, _tCaA.View, _temb.View);
        Embedder("av_ca_a2v_gate_adaln_single", Dv, _tGateV.View, _temb.View);
        Embedder("av_ca_v2a_gate_adaln_single", Da, _tGateA.View, _temb.View);
        Probe?.Invoke("adaln_video", _tV.View, B, 6 * Dv); Probe?.Invoke("adaln_video_emb", _tVe.View, B, Dv);
        Probe?.Invoke("adaln_audio", _tA.View, B, 6 * Da); Probe?.Invoke("adaln_audio_emb", _tAe.View, B, Da);

        // text: the caption projections (Linear, GELU-tanh, Linear)
        Lin(b0, videoContext, "caption_projection.linear_1", _hid.View, B * L, Caption, Dv, DiTKernels.Act.GeluTanh);
        Lin(b0, _hid.View, "caption_projection.linear_2", _vctx.View, B * L, Dv, Dv);
        Lin(b0, audioContext, "audio_caption_projection.linear_1", _hid.View, B * L, Caption, Da, DiTKernels.Act.GeluTanh);
        Lin(b0, _hid.View, "audio_caption_projection.linear_2", _actx.View, B * L, Da, Da);
        Probe?.Invoke("caption_video", _vctx.View, B * L, Dv); Probe?.Invoke("caption_audio", _actx.View, B * L, Da);
        Lin(b0, video, "patchify_proj", _v.View, B * nv, InChannels, Dv);
        Lin(b0, audio, "audio_patchify_proj", _a.View, B * na, InChannels, Da);
        Probe?.Invoke("patchify_video", _v.View, B * nv, Dv); Probe?.Invoke("patchify_audio", _a.View, B * na, Da);

        for (int k = 0; k < Layers; k++)
        {
            var (w, streamed) = await _w.BlockAsync(k, ct);
            try { Block(w, $"transformer_blocks.{k}."); }
            finally { if (streamed) { await _acc.SynchronizeAsync(); w.Dispose(); } }
            Probe?.Invoke($"block{k}_video", _v.View, B * nv, Dv);
            Probe?.Invoke($"block{k}_audio", _a.View, B * na, Da);
        }

        // output heads: LayerNorm (no affine), (1 + scale) + shift with shift = table[0] + embedded timestep, scale = table[1] + embedded timestep
        for (int b = 0; b < B; b++)
        {
            for (int j = 0; j < 2; j++)
            {
                _x.Add(b0.F32("scale_shift_table").SubView(j * Dv, Dv), _tVe.View.SubView(b * Dv, Dv), _headV.View.SubView((2 * b + j) * Dv, Dv), Dv);
                _x.Add(b0.F32("audio_scale_shift_table").SubView(j * Da, Da), _tAe.View.SubView(b * Da, Da), _headA.View.SubView((2 * b + j) * Da, Da), Da);
            }
            _k.LayerNormModulate(_v.View.SubView((long)b * nv * Dv, (long)nv * Dv), _vn.View.SubView((long)b * nv * Dv, (long)nv * Dv), _headV.View, 2 * b * Dv, (2 * b + 1) * Dv, nv, Dv, Eps);
            _k.LayerNormModulate(_a.View.SubView((long)b * na * Da, (long)na * Da), _an.View.SubView((long)b * na * Da, (long)na * Da), _headA.View, 2 * b * Da, (2 * b + 1) * Da, na, Da, Eps);
        }
        Lin(b0, _vn.View, "proj_out", _outV.View, B * nv, Dv, InChannels);
        Lin(b0, _an.View, "audio_proj_out", _outA.View, B * na, Da, InChannels);
        return (_outV.View.SubView(0, (long)B * nv * InChannels), _outA.View.SubView(0, (long)B * na * InChannels));
    }

    // mod rows of one block: table [N, D] (fp32) + the embedder's [B, N D] (per item)
    private void Mod(FV table, FV temb, FV mod, int N, int D, int tableRow0 = 0, int tembN = -1)
    {
        int tn = tembN < 0 ? N : tembN;
        for (int b = 0; b < _B; b++)
            _x.Add(table.SubView((long)tableRow0 * D, (long)N * D), temb.SubView((long)b * tn * D, (long)N * D), mod.SubView((long)b * N * D, (long)N * D), N * D);
    }

    // self- or cross-attention of one stream: q from xq [B Nq, Dq-in], k / v from xkv [B Nk, Dkv-in]; heads H x D (inner H D); optional RoPE tables per side; out [B Nq, Dout] into o
    private void Attn(GpuWeights w, string p, FV xq, int Nq, int dq, FV xkv, int Nk, int dkv, int H, int D, int dout, FV o,
        (FV Cos, FV Sin)? ropeQ, (FV Cos, FV Sin)? ropeK)
    {
        int B = _B, C = H * D;
        Lin(w, xq, p + "to_q", _q.View, B * Nq, dq, C);
        Lin(w, xkv, p + "to_k", _part.View, B * Nk, dkv, C); _f.CopyCols(_part.View, _kv.View, B * Nk, C, C, 2 * C, 0);
        Lin(w, xkv, p + "to_v", _part.View, B * Nk, dkv, C); _f.CopyCols(_part.View, _kv.View, B * Nk, C, C, 2 * C, C);
        _u.RmsNormHeads(_q.View, w.BF16(p + "q_norm.weight"), B * Nq, C, 0, 1, C, QkEps);               // over the whole inner dim, with weights
        _u.RmsNormHeads(_kv.View, w.BF16(p + "k_norm.weight"), B * Nk, 2 * C, 0, 1, C, QkEps);
        for (int b = 0; b < B; b++)
        {
            var qb = _q.View.SubView((long)b * Nq * C, (long)Nq * C); var kvb = _kv.View.SubView((long)b * Nk * 2 * C, (long)Nk * 2 * C);
            if (ropeQ is { } rq) _l.RopeSplit(qb, rq.Cos, rq.Sin, Nq, C, 0, H, D);
            if (ropeK is { } rk) _l.RopeSplit(kvb, rk.Cos, rk.Sin, Nk, 2 * C, 0, H, D);
            _k.Attention(qb, C, 0, kvb, 2 * C, 0, C, _scores.View, _o.View.SubView((long)b * Nq * C, (long)Nq * C), Nq, Nk, H, D);
        }
        Lin(w, _o.View, p + "to_out.0", o, B * Nq, C, dout);
    }

    private void Block(GpuWeights w, string p)
    {
        int B = _B, nv = _nv, na = _na, L = _L;
        Mod(w.F32(p + "scale_shift_table"), _tV.View, _modV.View, 6, Dv);
        Mod(w.F32(p + "audio_scale_shift_table"), _tA.View, _modA.View, 6, Da);
        // rows 0..3 of the per-block cross tables + the scale/shift embedders; row 4 + the gate embedders
        Mod(w.F32(p + "scale_shift_table_a2v_ca_video"), _tCaV.View, _caV.View, 4, Dv);
        Mod(w.F32(p + "scale_shift_table_a2v_ca_audio"), _tCaA.View, _caA.View, 4, Da);
        Mod(w.F32(p + "scale_shift_table_a2v_ca_video"), _tGateV.View, _gateV.View, 1, Dv, 4);
        Mod(w.F32(p + "scale_shift_table_a2v_ca_audio"), _tGateA.View, _gateA.View, 1, Da, 4);

        // 1. self-attention (video, then audio): adaLN 0 shift, 1 scale, 2 gate
        for (int b = 0; b < B; b++) _x.RmsNormModulate(Rows(_v, b, nv, Dv), Rows(_vn, b, nv, Dv), _modV.View, b * 6 * Dv, b * 6 * Dv + Dv, nv, Dv, Eps);
        Attn(w, p + "attn1.", _vn.View, nv, Dv, _vn.View, nv, Dv, Hv, HeadV, Dv, _part.View, (_ropeV.View, _ropeVs.View), (_ropeV.View, _ropeVs.View));
        for (int b = 0; b < B; b++) _k.GateResidual(Rows(_v, b, nv, Dv), _part.View.SubView((long)b * nv * Dv, (long)nv * Dv), _modV.View, b * 6 * Dv + 2 * Dv, nv * Dv, Dv);
        for (int b = 0; b < B; b++) _x.RmsNormModulate(Rows(_a, b, na, Da), Rows(_an, b, na, Da), _modA.View, b * 6 * Da, b * 6 * Da + Da, na, Da, Eps);
        Attn(w, p + "audio_attn1.", _an.View, na, Da, _an.View, na, Da, Ha, HeadA, Da, _part.View, (_ropeA.View, _ropeAs.View), (_ropeA.View, _ropeAs.View));
        for (int b = 0; b < B; b++) _k.GateResidual(Rows(_a, b, na, Da), _part.View.SubView((long)b * na * Da, (long)na * Da), _modA.View, b * 6 * Da + 2 * Da, na * Da, Da);

        // 2. text cross-attention: plain RMS norm, no RoPE, ungated
        _l.RmsNormPlain(_v.View, _vn.View, B * nv, Dv, Eps);
        Attn(w, p + "attn2.", _vn.View, nv, Dv, _vctx.View, L, Dv, Hv, HeadV, Dv, _part.View, null, null);
        _k.GateResidual(_v.View, _part.View, _part.View, -1, B * nv * Dv, Dv);
        _l.RmsNormPlain(_a.View, _an.View, B * na, Da, Eps);
        Attn(w, p + "audio_attn2.", _an.View, na, Da, _actx.View, L, Da, Ha, HeadA, Da, _part.View, null, null);
        _k.GateResidual(_a.View, _part.View, _part.View, -1, B * na * Da, Da);

        // 3. audio <-> video. a2v first; then v2a reads the video stream AFTER a2v's update (MEASURED 2026-11-17 against ComfyUI's block 0, float64 reference: the
        //    pre-update video, as diffusers' code reads it, leaves the audio stream 2.4e-4 off; the updated one 1.1e-7).
        //    video mods: 0 a2v scale, 1 a2v shift, 2 v2a scale, 3 v2a shift; audio the same
        // a2v: q = video modulated by its a2v pair, k / v = audio modulated by its a2v pair
        for (int b = 0; b < B; b++)
        {
            _x.RmsNormModulate(Rows(_v, b, nv, Dv), Rows(_vn, b, nv, Dv), _caV.View, b * 4 * Dv + Dv, b * 4 * Dv, nv, Dv, Eps);
            _x.RmsNormModulate(Rows(_a, b, na, Da), Rows(_an, b, na, Da), _caA.View, b * 4 * Da + Da, b * 4 * Da, na, Da, Eps);
        }
        Attn(w, p + "audio_to_video_attn.", _vn.View, nv, Dv, _an.View, na, Da, Ha, HeadA, Dv, _part.View, (_ropeVx.View, _ropeVxs.View), (_ropeA.View, _ropeAs.View));
        for (int b = 0; b < B; b++) _k.GateResidual(Rows(_v, b, nv, Dv), _part.View.SubView((long)b * nv * Dv, (long)nv * Dv), _gateV.View, b * Dv, nv * Dv, Dv);
        // v2a: q = audio modulated by its v2a pair, k / v = the (a2v-updated) video modulated by its v2a pair
        for (int b = 0; b < B; b++)
        {
            _x.RmsNormModulate(Rows(_v, b, nv, Dv), Rows(_vn, b, nv, Dv), _caV.View, b * 4 * Dv + 3 * Dv, b * 4 * Dv + 2 * Dv, nv, Dv, Eps);
            _x.RmsNormModulate(Rows(_a, b, na, Da), Rows(_an, b, na, Da), _caA.View, b * 4 * Da + 3 * Da, b * 4 * Da + 2 * Da, na, Da, Eps);
        }
        Attn(w, p + "video_to_audio_attn.", _an.View, na, Da, _vn.View, nv, Dv, Ha, HeadA, Da, _part.View, (_ropeA.View, _ropeAs.View), (_ropeVx.View, _ropeVxs.View));
        for (int b = 0; b < B; b++) _k.GateResidual(Rows(_a, b, na, Da), _part.View.SubView((long)b * na * Da, (long)na * Da), _gateA.View, b * Da, na * Da, Da);

        // 4. feed-forwards: adaLN 3 shift, 4 scale, 5 gate
        for (int b = 0; b < B; b++) _x.RmsNormModulate(Rows(_v, b, nv, Dv), Rows(_vn, b, nv, Dv), _modV.View, b * 6 * Dv + 3 * Dv, b * 6 * Dv + 4 * Dv, nv, Dv, Eps);
        Lin(w, _vn.View, p + "ff.net.0.proj", _hid.View, B * nv, Dv, Fv, DiTKernels.Act.GeluTanh);
        Lin(w, _hid.View, p + "ff.net.2", _part.View, B * nv, Fv, Dv);
        for (int b = 0; b < B; b++) _k.GateResidual(Rows(_v, b, nv, Dv), _part.View.SubView((long)b * nv * Dv, (long)nv * Dv), _modV.View, b * 6 * Dv + 5 * Dv, nv * Dv, Dv);
        for (int b = 0; b < B; b++) _x.RmsNormModulate(Rows(_a, b, na, Da), Rows(_an, b, na, Da), _modA.View, b * 6 * Da + 3 * Da, b * 6 * Da + 4 * Da, na, Da, Eps);
        Lin(w, _an.View, p + "audio_ff.net.0.proj", _hid.View, B * na, Da, Fa, DiTKernels.Act.GeluTanh);
        Lin(w, _hid.View, p + "audio_ff.net.2", _part.View, B * na, Fa, Da);
        for (int b = 0; b < B; b++) _k.GateResidual(Rows(_a, b, na, Da), _part.View.SubView((long)b * na * Da, (long)na * Da), _modA.View, b * 6 * Da + 5 * Da, na * Da, Da);
    }

    private static FV Rows(MemoryBuffer1D<float, Stride1D.Dense> buf, int b, int n, int C) => buf.View.SubView((long)b * n * C, (long)n * C);

    public void Dispose()
    {
        foreach (var o in _owned) o.Dispose();
        _owned.Clear();
        _w.Dispose();
    }
}
