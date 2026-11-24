using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.SkinTokens;
using SpawnDev.ILGPU.ML.Trellis2;
using System.Text.Json;
using FV = ILGPU.Runtime.ArrayView1D<float, ILGPU.Stride1D.Dense>;
using BV = ILGPU.Runtime.ArrayView1D<ILGPU.BFloat16, ILGPU.Stride1D.Dense>;

namespace SpawnDev.ILGPU.ML.AceStep;

/// <summary>
/// ACE-Step 1.5's diffusion transformer (<c>AceStepDiTModel</c>, the <c>decoder.*</c> tensors of ComfyUI's acestep_v1.5_turbo; MIT source: ace-step/ACE-Step-1.5,
/// acestep/models/turbo/modeling_acestep_v15_turbo.py). One call:
/// <code>
/// temb, proj = time_embed(t) + time_embed_r(t - r)          sinusoid(1000 t, 256: [cos, sin]) -> linear_1 -> SiLU -> linear_2 = temb; time_proj(SiLU(temb)) = proj [6, D]
/// h = proj_in(pad_even([context | x]))                        Conv1d(192 -> 2048, k2 s2): patchify + GEMM
/// enc = condition_embedder(encoder_hidden_states)             Linear 2048 -> 2048 + bias
/// 24 x { mod = scale_shift_table + proj                       shift, scale, gate, c_shift, c_scale, c_gate
///        h += gate * o(attn(rope(qk_norm(q, k, v of RMS(h)(1 + scale) + shift))))   GQA 16 / 8 x 128, full attention (<see cref="SlidingWindow"/>)
///        h += o(attn(q_norm(q(RMS(h))), k_norm(k(enc)), v(enc)))                    cross-attention: no RoPE, no modulation, no gate
///        h += c_gate * down(silu(gate(m)) up(m)),  m = RMS(h)(1 + c_scale) + c_shift }
/// out = proj_out(RMS(h)(1 + scale) + shift), (shift, scale) = scale_shift_table[2] + temb   ConvTranspose1d(2048 -> 64, k2 s2): GEMM + unpatchify, cropped
/// </code>
/// Inputs and output are time-major as ComfyUI's decoder sees them: x [T, 64], context [T, 128], encoder states [Ne, 2048], out [T, 64] (the velocity).
/// </summary>
public sealed class AceStepDiT : IDisposable
{
    public const int Dim = 2048, Heads = 16, KvHeads = 8, HeadDim = 128, Mlp = 6144, Layers = 24, LatentChannels = 64, ContextChannels = 128, Window = 128;
    private const float Eps = 1e-6f, RopeTheta = 1e6f;
    private const int TimeChannels = 256;

    private readonly Accelerator _acc;
    private readonly GpuWeights _w;
    private readonly DiTKernels _k;
    private readonly SkinTokensKernels _u;
    private readonly AceStepKernels _a;
    private readonly MemoryBuffer1D<BFloat16, Stride1D.Dense> _projOutT;
    private readonly int _maxFrames, _maxTokens, _maxEnc;
    private readonly List<IDisposable> _buffers = new();
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _patches, _h, _n, _m, _q, _kk, _v, _att, _o, _g, _up, _gu, _enc, _encK, _encV, _scores, _cos, _sin, _y;
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _sin256, _t1, _tA, _tembT, _tembR, _temb, _projT, _projR, _proj, _mod, _modOut;

    /// <summary>Intermediate checkpoints for tests: name, device view, rows, columns.</summary>
    public Action<string, FV, int, int>? Probe;
    /// <summary>
    /// The even layers' bidirectional self-attention window: -1 (default) = full attention, as ComfyUI's implementation runs ACE-Step (measured on its condition encoder,
    /// <see cref="AceStepConditionEncoder.SlidingWindow"/>; at 10 s the DiT's 125 tokens fit any window). <see cref="Window"/> gives ACE-Step's own code's |i - j| &lt;= 128.
    /// </summary>
    public int SlidingWindow { get; set; } = -1;

    /// <summary>CUDA: fp32-class activations through the tensor-core GEMMs (<see cref="DiTKernels.PreciseActivations"/>); on by default (the reference is float32).</summary>
    public bool PreciseActivations { get => _k.PreciseActivations; set => _k.PreciseActivations = value; }

    private AceStepDiT(Accelerator acc, GpuWeights w, BFloat16[] projOutT, int maxFrames, int maxEnc)
    {
        _acc = acc; _w = w; _maxFrames = maxFrames; _maxTokens = (maxFrames + 1) / 2; _maxEnc = maxEnc;
        _k = new DiTKernels(acc) { PreciseActivations = true }; _u = new SkinTokensKernels(acc); _a = new AceStepKernels(acc);
        MemoryBuffer1D<float, Stride1D.Dense> A(long n) { var b = acc.Allocate1D<float>(Math.Max(1, n)); _buffers.Add(b); return b; }
        long N = _maxTokens, E = maxEnc, M = Math.Max(N, E);
        _patches = A(N * 2 * (ContextChannels + LatentChannels));
        _h = A(N * Dim); _n = A(M * Dim); _m = A(N * Dim); _q = A(N * Dim); _kk = A(N * KvHeads * HeadDim); _v = A(N * KvHeads * HeadDim);
        _att = A(N * Dim); _o = A(N * Dim); _g = A(N * Mlp); _up = A(N * Mlp); _gu = A(N * Mlp);
        _enc = A(E * Dim); _encK = A(E * KvHeads * HeadDim); _encV = A(E * KvHeads * HeadDim);
        _scores = A((long)Heads * N * M); _cos = A(N * HeadDim / 2); _sin = A(N * HeadDim / 2); _y = A(N * 2 * LatentChannels);
        _sin256 = A(TimeChannels); _t1 = A(Dim); _tA = A(Dim); _tembT = A(Dim); _tembR = A(Dim); _temb = A(Dim); _projT = A(6 * Dim); _projR = A(6 * Dim); _proj = A(6 * Dim);
        _mod = A(6 * Dim); _modOut = A(2 * Dim);
        _projOutT = acc.Allocate1D(projOutT);
        UploadRope(_maxTokens);
    }

    /// <summary>Load the DiT (and nothing else) from ComfyUI's acestep_v1.5_turbo file; buffers sized for <paramref name="maxFrames"/> latent frames and <paramref name="maxEncoderTokens"/> condition tokens.</summary>
    public static async Task<AceStepDiT> LoadAsync(Accelerator accelerator, IModelSource source, string repo, string file, int maxFrames, int maxEncoderTokens, CancellationToken ct = default)
    {
        GpuWeights w;
        await using (var stream = await source.OpenAsync(repo, file, ct))
        {
            using var empty = JsonDocument.Parse("{}");
            w = await GpuWeights.LoadAsync(accelerator, stream, empty.RootElement, file, n => n.StartsWith("decoder.", StringComparison.Ordinal), ct);
        }
        try
        {
            var shape = w.Shape("decoder.proj_in.1.weight");
            if (shape[0] != Dim || shape[1] != ContextChannels + LatentChannels || shape[2] != 2) throw new InvalidDataException($"'{file}': proj_in [{string.Join(", ", shape)}] is not ACE-Step 1.5's");
            // proj_out (ConvTranspose1d [2048, 64, 2]) as a GEMM matrix [128, 2048]: row c 2 + k, column o
            var wo = await w.BF16("decoder.proj_out.1.weight").CopyToHostAsync();
            var t = new BFloat16[2 * LatentChannels * Dim];
            for (int o = 0; o < Dim; o++) for (int r = 0; r < 2 * LatentChannels; r++) t[(long)r * Dim + o] = wo[(long)o * 2 * LatentChannels + r];
            return new AceStepDiT(accelerator, w, t, maxFrames, maxEncoderTokens);
        }
        catch { w.Dispose(); throw; }
    }

    // rotate_half tables for positions 0 .. n - 1 (Qwen3RotaryEmbedding, theta 1e6), float32 arguments as the reference evaluates them
    private void UploadRope(int n)
    {
        int half = HeadDim / 2;
        var c = new float[n * half]; var s = new float[n * half];
        for (int k = 0; k < half; k++)
        {
            float inv = 1f / MathF.Pow(RopeTheta, 2f * k / HeadDim);
            for (int p = 0; p < n; p++) { float ang = p * inv; c[p * half + k] = (float)Math.Cos(ang); s[p * half + k] = (float)Math.Sin(ang); }
        }
        _cos.View.SubView(0, c.Length).CopyFromCPU(c); _sin.View.SubView(0, s.Length).CopyFromCPU(s);
    }

    private static float[] Sinusoid(float t)
    {
        int half = TimeChannels / 2;
        var e = new float[TimeChannels];
        for (int i = 0; i < half; i++)
        {
            float f = MathF.Exp(-MathF.Log(10000f) * i / half), a = t * 1000f * f;
            e[i] = (float)Math.Cos(a); e[half + i] = (float)Math.Sin(a);
        }
        return e;
    }

    private BV B(string name) => _w.BF16(name);
    private void Lin(FV x, string name, FV o, int rows, int K, int N, bool bias = false) => _k.Linear(x, B(name + ".weight"), bias ? B(name + ".bias") : null, o, rows, K, N);

    // TimestepEmbedding: temb [D], proj [6 D]
    private void TimeEmbed(string p, float t, FV temb, FV proj)
    {
        _sin256.View.CopyFromCPU(Sinusoid(t));
        _u.Gemv(_sin256.View, B(p + ".linear_1.weight"), B(p + ".linear_1.bias"), _t1.View, TimeChannels, Dim);
        _k.Silu(_t1.View, _tA.View, Dim);
        _u.Gemv(_tA.View, B(p + ".linear_2.weight"), B(p + ".linear_2.bias"), temb, Dim, Dim);
        _k.Silu(temb, _tA.View, Dim);
        _u.Gemv(_tA.View, B(p + ".time_proj.weight"), B(p + ".time_proj.bias"), proj, Dim, 6 * Dim);
    }

    /// <summary>
    /// One model call: velocity [frames, 64] (time-major) into <paramref name="output"/> for the noisy latent <paramref name="x"/> [frames, 64], the context
    /// <paramref name="context"/> [frames, 128] (source latents | chunk mask), the condition encoder's states <paramref name="encoderStates"/> [encTokens, 2048],
    /// timestep <paramref name="t"/> (sigma) and <paramref name="tr"/> (r; time_embed_r sees t - r).
    /// </summary>
    public void Forward(FV x, FV context, FV encoderStates, int frames, int encTokens, float t, float tr, FV output)
    {
        if (frames > _maxFrames || encTokens > _maxEnc) throw new ArgumentException($"{frames} frames / {encTokens} condition tokens: the DiT was sized for {_maxFrames} / {_maxEnc}");
        int N = (frames + 1) / 2, D = Dim, kv = KvHeads * HeadDim, rep = Heads / KvHeads;
        TimeEmbed("decoder.time_embed", t, _tembT.View, _projT.View);
        TimeEmbed("decoder.time_embed_r", t - tr, _tembR.View, _projR.View);
        _a.Add(_tembT.View, _tembR.View, _temb.View, D);                                   // separate output buffers: WebGPU cannot bind one buffer twice
        _a.Add(_projT.View, _projR.View, _proj.View, 6 * D);
        _a.Patchify(context, x, _patches.View, frames, ContextChannels, LatentChannels);
        Lin(_patches.View, "decoder.proj_in.1", _h.View, N, 2 * (ContextChannels + LatentChannels), D, bias: true);
        Lin(encoderStates, "decoder.condition_embedder", _enc.View, encTokens, D, D, bias: true);
        Probe?.Invoke("proj_in", _h.View, N, D);
        for (int l = 0; l < Layers; l++)
        {
            // browser: a command buffer per block (DeviceSync.Submit; a no-op on desktop). A whole forward in ONE command buffer can outrun a slow
            // GPU's 2 s watchdog (2026-11-23, Lore: FLUX.2 lost the device on an RX 590 in Chrome until its encoder and transformer submitted per block).
            _acc.Submit();
            string b = $"decoder.layers.{l}.";
            _k.AddBf16(B(b + "scale_shift_table"), _proj.View, _mod.View, 6 * D);
            // self-attention
            _u.RmsNorm(_h.View, _n.View, B(b + "self_attn_norm.weight"), N, D, Eps);
            _a.Modulate(_n.View, _m.View, _mod.View, 0, D, N, D);
            Lin(_m.View, b + "self_attn.q_proj", _q.View, N, D, D);
            Lin(_m.View, b + "self_attn.k_proj", _kk.View, N, D, kv);
            Lin(_m.View, b + "self_attn.v_proj", _v.View, N, D, kv);
            _u.RmsNormHeads(_q.View, B(b + "self_attn.q_norm.weight"), N, D, 0, Heads, HeadDim, Eps);
            _u.RmsNormHeads(_kk.View, B(b + "self_attn.k_norm.weight"), N, kv, 0, KvHeads, HeadDim, Eps);
            _u.RopeTables(_q.View, _cos.View, _sin.View, N, D, 0, Heads, HeadDim);
            _u.RopeTables(_kk.View, _cos.View, _sin.View, N, kv, 0, KvHeads, HeadDim);
            _a.Gqa(_q.View, D, _kk.View, kv, _v.View, kv, _scores.View, _att.View, N, N, Heads, HeadDim, rep, l % 2 == 0 ? SlidingWindow : -1);
            Lin(_att.View, b + "self_attn.o_proj", _o.View, N, D, D);
            _k.GateResidual(_h.View, _o.View, _mod.View, 2 * D, N * D, D);
            // cross-attention to the condition tokens
            _u.RmsNorm(_h.View, _n.View, B(b + "cross_attn_norm.weight"), N, D, Eps);
            Lin(_n.View, b + "cross_attn.q_proj", _q.View, N, D, D);
            Lin(_enc.View, b + "cross_attn.k_proj", _encK.View, encTokens, D, kv);
            Lin(_enc.View, b + "cross_attn.v_proj", _encV.View, encTokens, D, kv);
            _u.RmsNormHeads(_q.View, B(b + "cross_attn.q_norm.weight"), N, D, 0, Heads, HeadDim, Eps);
            _u.RmsNormHeads(_encK.View, B(b + "cross_attn.k_norm.weight"), encTokens, kv, 0, KvHeads, HeadDim, Eps);
            _a.Gqa(_q.View, D, _encK.View, kv, _encV.View, kv, _scores.View, _att.View, N, encTokens, Heads, HeadDim, rep, -1);
            Lin(_att.View, b + "cross_attn.o_proj", _o.View, N, D, D);
            _k.GateResidual(_h.View, _o.View, _mod.View, -1, N * D, D);
            // MLP
            _u.RmsNorm(_h.View, _n.View, B(b + "mlp_norm.weight"), N, D, Eps);
            _a.Modulate(_n.View, _m.View, _mod.View, 3 * D, 4 * D, N, D);
            Lin(_m.View, b + "mlp.gate_proj", _g.View, N, D, Mlp);
            Lin(_m.View, b + "mlp.up_proj", _up.View, N, D, Mlp);
            _a.SiluMul(_g.View, _up.View, _gu.View, N * Mlp);
            Lin(_gu.View, b + "mlp.down_proj", _o.View, N, Mlp, D);
            _k.GateResidual(_h.View, _o.View, _mod.View, 5 * D, N * D, D);
            Probe?.Invoke($"layer{l}", _h.View, N, D);
        }
        // head: (shift, scale) = scale_shift_table[2, D] + temb
        _k.AddBf16(B("decoder.scale_shift_table").SubView(0, D), _temb.View, _modOut.View.SubView(0, D), D);
        _k.AddBf16(B("decoder.scale_shift_table").SubView(D, D), _temb.View, _modOut.View.SubView(D, D), D);
        _u.RmsNorm(_h.View, _n.View, B("decoder.norm_out.weight"), N, D, Eps);
        _a.Modulate(_n.View, _m.View, _modOut.View, 0, D, N, D);
        _k.Linear(_m.View, _projOutT.View, null, _y.View, N, D, 2 * LatentChannels);
        _a.Unpatchify(_y.View, B("decoder.proj_out.1.bias"), output, frames, LatentChannels);
    }

    public void Dispose()
    {
        foreach (var b in _buffers) b.Dispose();
        _buffers.Clear();
        _projOutT.Dispose();
        _w.Dispose();
    }
}
