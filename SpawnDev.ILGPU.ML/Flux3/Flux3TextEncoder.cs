using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Flux2;
using SpawnDev.ILGPU.ML.SkinTokens;
using SpawnDev.ILGPU.ML.Trellis2;

namespace SpawnDev.ILGPU.ML.Flux3;

using FV = ArrayView1D<float, Stride1D.Dense>;

/// <summary>
/// FLUX 3 Image Klein's text encoder: the TEXT TOWER of Qwen3-VL-4B-Instruct (36 layers, width 2560, 32 query / 8 key-value heads of 128, per-head RMS norm on q and k, SwiGLU of 9728, rotary theta 5e6) run over the
/// chat-templated caption, padded on the right to a multiple of 80 tokens (at most 4096), and read at the hidden states after layers 4, 8, 12, 16, 20, 24, 28 and 32, stacked into one
/// [tokens, 8 * 2560 = 20480] conditioning matrix: only the first 32 layers are loaded and run (no final norm, no head, no vision tower).
/// <list type="bullet">
/// <item>The reference passes an attention mask: padding KEYS are masked, causal otherwise; padding query rows still see the whole prompt and all of them go to the DiT.</item>
/// <item>Positions are <c>cumsum(mask) - 1</c> with the padding rows put at position <b>1</b>, not at their index. With text only the three multimodal rotary axes carry the same position, which makes the
/// interleaved mrope the plain rotary embedding.</item>
/// </list>
/// bf16 (or 4 / 8-bit block) weights, fp32 activations. Weights from tools/unirig-harness/reference/flux3/quantize_flux3_text.py, in FLUX.2's text-encoder layout.
/// </summary>
public sealed class Flux3TextEncoder : IDisposable
{
    public const int Layers = 32, Hidden = 2560, QHeads = 32, KvHeads = 8, HeadDim = 128, Rep = QHeads / KvHeads, Inter = 9728, PadMultiple = 80, MaxTokens = 4096;
    public const int QCols = QHeads * HeadDim, KvCols = KvHeads * HeadDim, QkvCols = QCols + 2 * KvCols;
    public const float RopeTheta = 5_000_000f, RmsEps = 1e-6f;
    public static readonly int[] TapAfterLayer = { 4, 8, 12, 16, 20, 24, 28, 32 };
    public const int EmbedWidth = 8 * Hidden;

    private readonly Accelerator _acc;
    private readonly Flux2Weights _w;
    private readonly DiTKernels _k;
    private readonly SkinTokensKernels _u;
    private readonly Flux2Kernels _f;
    private readonly List<IDisposable> _owned = new();
    private readonly int _capacity;
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _h, _hn, _qkv, _attn, _proj, _gateUp, _act, _scores, _kc, _vc, _dummy, _invFreq, _out;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _ids, _pos;

    /// <summary>The padded length a caption of <paramref name="realTokens"/> tokens is run at: the next multiple of 80, at most 4096.</summary>
    public static int PaddedLength(int realTokens) => Math.Min((realTokens + PadMultiple - 1) / PadMultiple * PadMultiple, MaxTokens);

    /// <param name="capacity">the longest padded sequence this instance will encode (its attention scores take 32 x capacity^2 floats)</param>
    public Flux3TextEncoder(Accelerator accelerator, Flux2Weights weights, int capacity)
    {
        _acc = accelerator; _w = weights; _capacity = capacity;
        _k = new DiTKernels(accelerator); _u = new SkinTokensKernels(accelerator); _f = new Flux2Kernels(accelerator);
        long T = capacity;
        _h = Alloc<float>(T * Hidden); _hn = Alloc<float>(T * Hidden); _qkv = Alloc<float>(T * QkvCols); _attn = Alloc<float>(T * QCols);
        _proj = Alloc<float>(T * Hidden); _gateUp = Alloc<float>(T * 2 * Inter); _act = Alloc<float>(T * Inter);
        _scores = Alloc<float>((long)QHeads * T * T); _kc = Alloc<float>(T * KvCols); _vc = Alloc<float>(T * KvCols);
        _dummy = Alloc<float>(1); _out = Alloc<float>(T * EmbedWidth); _ids = Alloc<int>(T); _pos = Alloc<int>(T);
        var inv = new float[HeadDim / 2];
        for (int k = 0; k < inv.Length; k++) inv[k] = 1f / MathF.Pow(RopeTheta, (2f * k) / HeadDim);
        _invFreq = _acc.Allocate1D(inv); _owned.Add(_invFreq);
    }

    private MemoryBuffer1D<T, Stride1D.Dense> Alloc<T>(long n) where T : unmanaged { var b = _acc.Allocate1D<T>(Math.Max(1, n)); _owned.Add(b); return b; }

    /// <summary>The conditioning rows of the last <see cref="Encode"/>, [tokens, 20480] on the device.</summary>
    public FV Output => _out.View;

    /// <summary>Intermediate checkpoints for tests: the name ("embed", "layer{n}") and the device view with the live rows.</summary>
    public Action<string, FV, int, int>? Probe;

    /// <summary>
    /// Encode one padded caption: <paramref name="ids"/> are the token ids (their count a multiple of 80, the first <paramref name="real"/> of them the caption, the rest padding).
    /// Returns the conditioning matrix [ids.Length, 20480] on the device (also <see cref="Output"/>).
    /// </summary>
    public FV Encode(int[] ids, int real)
    {
        int T = ids.Length;
        if (T > _capacity) throw new ArgumentException($"{T} tokens exceed this encoder's capacity {_capacity}");
        _ids.View.SubView(0, T).CopyFromCPU(ids);
        var pos = new int[T]; for (int i = 0; i < T; i++) pos[i] = i < real ? i : 1;
        _pos.View.SubView(0, T).CopyFromCPU(pos);
        var posV = _pos.View.SubView(0, T);
        _w.Embed(_ids.View.SubView(0, T), "embed_tokens.weight", _h.View, T, Hidden);
        Probe?.Invoke("embed", _h.View, T, Hidden);
        int tap = 0;
        for (int l = 0; l < Layers; l++)
        {
            string b = $"layers.{l}.";
            _u.RmsNorm(_h.View, _hn.View, _w.Vector(b + "input_layernorm.weight"), T, Hidden, RmsEps);
            _k.Linear(_hn.View, _w.Matrix(b + "self_attn.qkv_proj.weight"), null, _qkv.View, T, Hidden, QkvCols);
            _u.RmsNormHeads(_qkv.View, _w.Vector(b + "self_attn.q_norm.weight"), T, QkvCols, 0, QHeads, HeadDim, RmsEps);
            _u.RmsNormHeads(_qkv.View, _w.Vector(b + "self_attn.k_norm.weight"), T, QkvCols, QCols, KvHeads, HeadDim, RmsEps);
            _f.RopeHalfPositions(_qkv.View, _invFreq.View, posV, T, QkvCols, 0, QHeads, HeadDim);
            _f.RopeHalfPositions(_qkv.View, _invFreq.View, posV, T, QkvCols, QCols, KvHeads, HeadDim);
            _u.StoreKv(_qkv.View, _kc.View, _vc.View, 0, T, QCols, KvCols);
            _f.PaddedCausalGqa(_qkv.View, _kc.View, _vc.View, _scores.View, _attn.View, T, real, QHeads, HeadDim, Rep);
            _k.Linear(_attn.View, _w.Matrix(b + "self_attn.o_proj.weight"), null, _proj.View, T, QCols, Hidden);
            _k.GateResidual(_h.View, _proj.View, _dummy.View, -1, T * Hidden, Hidden);
            _u.RmsNorm(_h.View, _hn.View, _w.Vector(b + "post_attention_layernorm.weight"), T, Hidden, RmsEps);
            _k.Linear(_hn.View, _w.Matrix(b + "mlp.gate_up_proj.weight"), null, _gateUp.View, T, Hidden, 2 * Inter);
            _u.Swiglu(_gateUp.View, _act.View, T, Inter);
            _k.Linear(_act.View, _w.Matrix(b + "mlp.down_proj.weight"), null, _proj.View, T, Inter, Hidden);
            _k.GateResidual(_h.View, _proj.View, _dummy.View, -1, T * Hidden, Hidden);
            Probe?.Invoke($"layer{l + 1}", _h.View, T, Hidden);
            if (tap < TapAfterLayer.Length && l + 1 == TapAfterLayer[tap])
            {
                _f.CopyCols(_h.View, _out.View, T, Hidden, Hidden, EmbedWidth, tap * Hidden);
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
