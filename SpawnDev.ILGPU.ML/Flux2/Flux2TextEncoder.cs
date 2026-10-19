using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.SkinTokens;
using SpawnDev.ILGPU.ML.Trellis2;

namespace SpawnDev.ILGPU.ML.Flux2;

using FV = ArrayView1D<float, Stride1D.Dense>;

/// <summary>
/// FLUX.2 klein's text encoder: Qwen3-4B (36 layers, width 2560, 32 query heads and 8 key/value heads of 128, per-head RMS norm on q and k, rotary positions with theta 1e6, SwiGLU of 9728)
/// run over the chat-templated prompt padded to 512 tokens. The pipeline never uses the model's output: it reads the hidden states after layers 9, 18 and 27 and stacks them into one
/// [512, 3 * 2560 = 7680] conditioning matrix, so only the first 27 layers are loaded and run (no final norm, no head).
/// <code>
/// h = embed(ids)
/// 27 x { h += o_proj(attention(rope(qk_norm(qkv(rms(h))))))  ;  h += down(swiglu(gate_up(rms'(h)))) }       causal, padding keys masked
/// out[:, 0:2560] = h after layer 9,   out[:, 2560:5120] = after layer 18,   out[:, 5120:7680] = after layer 27
/// </code>
/// bf16 weights, fp32 activations (<see cref="DiTKernels"/>). Weights from tools/unirig-harness/reference/export_flux2.py (text_encoder.safetensors).
/// Checked against the reference's own prompt embeddings (golden_klein.py).
/// </summary>
public sealed class Flux2TextEncoder : IDisposable
{
    public const int Layers = 27, Hidden = 2560, QHeads = 32, KvHeads = 8, HeadDim = 128, Rep = QHeads / KvHeads, Inter = 9728, SeqLen = 512;
    public const int QCols = QHeads * HeadDim, KvCols = KvHeads * HeadDim, QkvCols = QCols + 2 * KvCols;
    public const float RopeTheta = 1_000_000f, RmsEps = 1e-6f;
    /// <summary>Width of the conditioning rows: three hidden states side by side.</summary>
    public const int EmbedWidth = 3 * Hidden;
    private static readonly int[] TapAfterLayer = { 9, 18, 27 };

    private readonly Accelerator _acc;
    private readonly Flux2Weights _w;
    private readonly DiTKernels _k;
    private readonly SkinTokensKernels _u;
    private readonly Flux2Kernels _f;
    private readonly List<IDisposable> _owned = new();
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _h, _hn, _qkv, _attn, _proj, _gateUp, _act, _scores, _kc, _vc, _dummy, _invFreq, _out;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _ids;

    /// <summary>
    /// Round the rotary table to bf16 as a model converted with <c>.to(bfloat16)</c> does. The reference loads the checkpoint with <c>dtype=bfloat16</c>; <see cref="RopeTable"/> tells which of the two
    /// matches (tests/golden decide; the default is what matches).
    /// </summary>
    public static bool RoundRopeToBf16 = false;

    public Flux2TextEncoder(Accelerator accelerator, Flux2Weights weights)
    {
        _acc = accelerator; _w = weights;
        _k = new DiTKernels(accelerator); _u = new SkinTokensKernels(accelerator); _f = new Flux2Kernels(accelerator);
        long T = SeqLen;
        _h = Alloc<float>(T * Hidden); _hn = Alloc<float>(T * Hidden); _qkv = Alloc<float>(T * QkvCols); _attn = Alloc<float>(T * QCols);
        _proj = Alloc<float>(T * Hidden); _gateUp = Alloc<float>(T * 2 * Inter); _act = Alloc<float>(T * Inter);
        _scores = Alloc<float>((long)QHeads * T * T); _kc = Alloc<float>(T * KvCols); _vc = Alloc<float>(T * KvCols);
        _dummy = Alloc<float>(1); _out = Alloc<float>(T * EmbedWidth); _ids = Alloc<int>(T);
        _invFreq = _acc.Allocate1D(RopeTable()); _owned.Add(_invFreq);
    }

    private static float[] RopeTable()
    {
        if (RoundRopeToBf16) return SkinTokensKernels.RopeTable(RopeTheta, HeadDim);
        var t = new float[HeadDim / 2];
        for (int k = 0; k < t.Length; k++) t[k] = 1f / MathF.Pow(RopeTheta, (2f * k) / HeadDim);
        return t;
    }

    private MemoryBuffer1D<T, Stride1D.Dense> Alloc<T>(long n) where T : unmanaged { var b = _acc.Allocate1D<T>(Math.Max(1, n)); _owned.Add(b); return b; }

    /// <summary>The conditioning rows of the last <see cref="Encode"/>, [512, 7680] on the device.</summary>
    public FV Output => _out.View;

    /// <summary>Intermediate checkpoints for tests: called with the name and the device view after the embedding ("embed") and after every layer ("layer{n}", 1-based) with the live rows.</summary>
    public Action<string, FV, int, int>? Probe;

    /// <summary>
    /// Encode one padded prompt: <paramref name="ids"/> is 512 token ids, the first <paramref name="real"/> of them the prompt (the rest padding). Returns the conditioning matrix on the device
    /// (also <see cref="Output"/>), [512, 7680] row-major.
    /// </summary>
    public FV Encode(int[] ids, int real)
    {
        if (ids.Length != SeqLen) throw new ArgumentException($"{SeqLen} token ids expected");
        _ids.View.CopyFromCPU(ids);
        _w.Embed(_ids.View, "embed_tokens.weight", _h.View, SeqLen, Hidden);
        Probe?.Invoke("embed", _h.View, SeqLen, Hidden);
        int tap = 0;
        for (int l = 0; l < Layers; l++)
        {
            string b = $"layers.{l}.";
            _u.RmsNorm(_h.View, _hn.View, _w.Vector(b + "input_layernorm.weight"), SeqLen, Hidden, RmsEps);
            _k.Linear(_hn.View, _w.Matrix(b + "self_attn.qkv_proj.weight"), null, _qkv.View, SeqLen, Hidden, QkvCols);
            _u.RmsNormHeads(_qkv.View, _w.Vector(b + "self_attn.q_norm.weight"), SeqLen, QkvCols, 0, QHeads, HeadDim, RmsEps);
            _u.RmsNormHeads(_qkv.View, _w.Vector(b + "self_attn.k_norm.weight"), SeqLen, QkvCols, QCols, KvHeads, HeadDim, RmsEps);
            _u.Rope(_qkv.View, _invFreq.View, SeqLen, QkvCols, 0, QHeads, HeadDim, 0);
            _u.Rope(_qkv.View, _invFreq.View, SeqLen, QkvCols, QCols, KvHeads, HeadDim, 0);
            _u.StoreKv(_qkv.View, _kc.View, _vc.View, 0, SeqLen, QCols, KvCols);
            _f.PaddedCausalGqa(_qkv.View, _kc.View, _vc.View, _scores.View, _attn.View, SeqLen, real, QHeads, HeadDim, Rep);
            _k.Linear(_attn.View, _w.Matrix(b + "self_attn.o_proj.weight"), null, _proj.View, SeqLen, QCols, Hidden);
            _k.GateResidual(_h.View, _proj.View, _dummy.View, -1, SeqLen * Hidden, Hidden);
            _u.RmsNorm(_h.View, _hn.View, _w.Vector(b + "post_attention_layernorm.weight"), SeqLen, Hidden, RmsEps);
            _k.Linear(_hn.View, _w.Matrix(b + "mlp.gate_up_proj.weight"), null, _gateUp.View, SeqLen, Hidden, 2 * Inter);
            _u.Swiglu(_gateUp.View, _act.View, SeqLen, Inter);
            _k.Linear(_act.View, _w.Matrix(b + "mlp.down_proj.weight"), null, _proj.View, SeqLen, Inter, Hidden);
            _k.GateResidual(_h.View, _proj.View, _dummy.View, -1, SeqLen * Hidden, Hidden);
            Probe?.Invoke($"layer{l + 1}", _h.View, SeqLen, Hidden);
            if (tap < TapAfterLayer.Length && l + 1 == TapAfterLayer[tap])
            {
                _f.CopyCols(_h.View, _out.View, SeqLen, Hidden, Hidden, EmbedWidth, tap * Hidden);
                tap++;
            }
        }
        return _out.View;
    }

    public void Dispose()
    {
        foreach (var o in _owned) o.Dispose();
        _owned.Clear();
    }
}
