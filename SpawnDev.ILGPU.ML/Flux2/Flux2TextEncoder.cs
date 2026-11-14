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
/// <para>The same class runs the other Qwen text encoders ComfyUI's templates use, sized from the weights: Z-Image reads Qwen3-4B after layer 35; Qwen-Image reads Qwen2.5-VL-7B's
/// language model (28 layers, width 3584, 28 / 4 heads, biases on q / k / v, no q / k norms) after its final norm, from 8-bit float matrices with one scale each that stay 8-bit
/// on the device (q, k, v and gate, up then run as separate GEMMs: their scales differ).</para>
/// </summary>
public sealed class Flux2TextEncoder : IDisposable
{
    // FLUX.2 klein's encoder (Qwen3-4B, first 27 layers)
    public const int Layers = 27, Hidden = 2560, QHeads = 32, KvHeads = 8, HeadDim = 128, Rep = QHeads / KvHeads, Inter = 9728, SeqLen = 512;
    public const int QCols = QHeads * HeadDim, KvCols = KvHeads * HeadDim, QkvCols = QCols + 2 * KvCols;
    public const float RopeTheta = 1_000_000f, RmsEps = 1e-6f;
    /// <summary>Width of the conditioning rows: three hidden states side by side.</summary>
    public const int EmbedWidth = 3 * Hidden;

    private readonly int[] _taps;
    private readonly int _layers, _hidden, _qHeads, _kvHeads, _inter, _qCols, _kvCols, _qkvCols;
    private readonly bool _qkNorm, _fusedQkv, _fusedGateUp, _qkvBias, _finalNorm;

    /// <summary>How many layers run (the last tap).</summary>
    public int LayerCount => _layers;
    /// <summary>This model's hidden width (FLUX.2 / Z-Image: 2560; Qwen-Image: 3584).</summary>
    public int HiddenSize => _hidden;
    /// <summary>Width of this encoder's conditioning rows: one hidden state per tap (FLUX.2: <see cref="EmbedWidth"/>).</summary>
    public int Width => _taps.Length * _hidden;

    private readonly Accelerator _acc;
    private readonly Flux2Weights _w;
    private readonly DiTKernels _k;
    private readonly SkinTokensKernels _u;
    private readonly Flux2Kernels _f;
    private readonly List<IDisposable> _owned = new();
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _h, _hn, _qkv, _attn, _proj, _gateUp, _act, _scores, _kc, _vc, _dummy, _invFreq, _out, _part, _xs;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _ids;

    /// <summary>
    /// Round the rotary table to bf16 as a model converted with <c>.to(bfloat16)</c> does. The reference loads the checkpoint with <c>dtype=bfloat16</c>; <see cref="RopeTable"/> tells which of the two
    /// matches (tests/golden decide; the default is what matches).
    /// </summary>
    public static bool RoundRopeToBf16 = false;

    public Flux2TextEncoder(Accelerator accelerator, Flux2Weights weights) : this(accelerator, weights, new[] { 9, 18, 27 }) { }

    /// <summary>
    /// The same stack reading the hidden states after the given layers (1-based, ascending), side by side: FLUX.2 { 9, 18, 27 }; Z-Image { 35 } (the reference's
    /// <c>hidden_states[-2]</c> of the 36-layer model, no final norm). The weights must hold the first <c>tapAfterLayer[^1]</c> layers. <paramref name="finalNorm"/>: the last tap
    /// goes through the model's final RMS norm (<c>norm.weight</c>), as Hugging Face's last hidden state does (Qwen-Image reads it after all 28 layers).
    /// </summary>
    public Flux2TextEncoder(Accelerator accelerator, Flux2Weights weights, int[] tapAfterLayer, bool finalNorm = false)
    {
        if (tapAfterLayer.Length == 0 || tapAfterLayer[0] < 1 || tapAfterLayer.Zip(tapAfterLayer.Skip(1)).Any(p => p.Second <= p.First)) throw new ArgumentException("taps: ascending 1-based layer numbers", nameof(tapAfterLayer));
        _acc = accelerator; _w = weights; _taps = (int[])tapAfterLayer.Clone(); _layers = _taps[^1]; _finalNorm = finalNorm;
        // the dimensions, from layer 0's tensors (fused by the adapters when they could be: q | k | v and gate | up of one precision and scale)
        _fusedQkv = weights.Has("layers.0.self_attn.qkv_proj.weight"); _fusedGateUp = weights.Has("layers.0.mlp.gate_up_proj.weight");
        _hidden = weights.Shape("layers.0.input_layernorm.weight")[0];
        if (_fusedQkv)
        {
            // the fused form is the Qwen3 adapter's (q | k | v of Qwen3-4B); its split follows from the head counts
            if (_hidden != Hidden || weights.Shape("layers.0.self_attn.qkv_proj.weight")[0] != QkvCols) throw new NotSupportedException("a fused q|k|v of a model other than Qwen3-4B");
            _qCols = QCols; _kvCols = KvCols;
        }
        else { _qCols = weights.Shape("layers.0.self_attn.q_proj.weight")[0]; _kvCols = weights.Shape("layers.0.self_attn.k_proj.weight")[0]; }
        _qkvCols = _qCols + 2 * _kvCols; _qHeads = _qCols / HeadDim; _kvHeads = _kvCols / HeadDim;
        _inter = _fusedGateUp ? weights.Shape("layers.0.mlp.gate_up_proj.weight")[0] / 2 : weights.Shape("layers.0.mlp.gate_proj.weight")[0];
        _qkNorm = weights.Has("layers.0.self_attn.q_norm.weight");
        _qkvBias = weights.Has("layers.0.self_attn.q_proj.bias") || weights.Has("layers.0.self_attn.qkv_proj.bias");
        if (_finalNorm && !weights.Has("norm.weight")) throw new InvalidDataException("finalNorm: the weights hold no norm.weight");

        _k = new DiTKernels(accelerator); _u = new SkinTokensKernels(accelerator); _f = new Flux2Kernels(accelerator);
        long T = SeqLen;
        _h = Alloc<float>(T * _hidden); _hn = Alloc<float>(T * _hidden); _qkv = Alloc<float>(T * _qkvCols); _attn = Alloc<float>(T * _qCols);
        _proj = Alloc<float>(T * _hidden); _gateUp = Alloc<float>(T * 2 * _inter); _act = Alloc<float>(T * _inter);
        _scores = Alloc<float>((long)_qHeads * T * T); _kc = Alloc<float>(T * _kvCols); _vc = Alloc<float>(T * _kvCols);
        _dummy = Alloc<float>(1); _out = Alloc<float>(T * Width); _ids = Alloc<int>(T);
        _part = Alloc<float>(_fusedQkv && _fusedGateUp ? 1 : T * Math.Max(_qCols, _inter));
        // a scaled 8-bit file (ComfyUI's fp8_scaled: Qwen-Image's Qwen2.5-VL) multiplies each matrix's input by the matrix's scale (Flux2Weights.MatrixExact): one buffer of the widest input
        _xs = Alloc<float>(_w.ActivationScale(_fusedQkv ? "layers.0.self_attn.qkv_proj.weight" : "layers.0.self_attn.q_proj.weight") != 1f ? T * Math.Max(Math.Max(_hidden, _qCols), _inter) : 1);
        _invFreq = _acc.Allocate1D(RopeTable()); _owned.Add(_invFreq);
    }

    /// <summary>
    /// CUDA: fp32-class activations through the tensor-core GEMMs (<see cref="DiTKernels.PreciseActivations"/>). Off by default here (FLUX.2's reference is a bf16 run); the encoders of
    /// models checked against ComfyUI's fp32 runs turn it on (Z-Image, Qwen-Image). Other backends compute in fp32 either way.
    /// </summary>
    public bool PreciseActivations { get => _k.PreciseActivations; set => _k.PreciseActivations = value; }

    // nn.Linear over the 512 rows; an 8-bit scaled matrix runs exact: its unscaled decode, with the input times its scale
    private void Lin(FV x, string weight, ArrayView1D<BFloat16, Stride1D.Dense>? bias, FV o, int K, int N)
    {
        float s = _w.ActivationScale(weight);
        if (s != 1f) { var xs = _xs.View.SubView(0, (long)SeqLen * K); _f.ScaleInto(x, xs, s, SeqLen * K); x = xs; }
        _k.Linear(x, _w.MatrixExact(weight), bias, o, SeqLen, K, N);
    }

    private static float[] RopeTable()
    {
        if (RoundRopeToBf16) return SkinTokensKernels.RopeTable(RopeTheta, HeadDim);
        var t = new float[HeadDim / 2];
        for (int k = 0; k < t.Length; k++) t[k] = 1f / MathF.Pow(RopeTheta, (2f * k) / HeadDim);
        return t;
    }

    private MemoryBuffer1D<T, Stride1D.Dense> Alloc<T>(long n) where T : unmanaged { var b = _acc.Allocate1D<T>(Math.Max(1, n)); _owned.Add(b); return b; }

    /// <summary>The conditioning rows of the last <see cref="Encode"/>, [512, <see cref="Width"/>] on the device.</summary>
    public FV Output => _out.View;

    /// <summary>Intermediate checkpoints for tests: called with the name and the device view after the embedding ("embed") and after every layer ("layer{n}", 1-based) with the live rows.</summary>
    public Action<string, FV, int, int>? Probe;

    // a projection whose output is a column range of a wider matrix: its own GEMM, then copied into place (a fused matrix's GEMM writes the whole row at once)
    private void LinearInto(FV x, string name, FV dst, int K, int N, int dstStride, int dstOff)
    {
        ArrayView1D<BFloat16, Stride1D.Dense>? bias = _w.Has(name + ".bias") ? _w.Vector(name + ".bias") : null;
        Lin(x, name + ".weight", bias, _part.View, K, N);
        _f.CopyCols(_part.View, dst, SeqLen, N, N, dstStride, dstOff);
    }

    /// <summary>
    /// Encode one padded prompt: <paramref name="ids"/> is 512 token ids, the first <paramref name="real"/> of them the prompt (the rest padding). Returns the conditioning matrix on the device
    /// (also <see cref="Output"/>), [512, <see cref="Width"/>] row-major.
    /// </summary>
    public FV Encode(int[] ids, int real)
    {
        if (ids.Length != SeqLen) throw new ArgumentException($"{SeqLen} token ids expected");
        int H = _hidden, rep = _qHeads / _kvHeads;
        _ids.View.CopyFromCPU(ids);
        _w.Embed(_ids.View, "embed_tokens.weight", _h.View, SeqLen, H);
        Probe?.Invoke("embed", _h.View, SeqLen, H);
        int tap = 0;
        for (int l = 0; l < _layers; l++)
        {
            string b = $"layers.{l}.";
            _u.RmsNorm(_h.View, _hn.View, _w.Vector(b + "input_layernorm.weight"), SeqLen, H, RmsEps);
            if (_fusedQkv) Lin(_hn.View, b + "self_attn.qkv_proj.weight", _qkvBias ? _w.Vector(b + "self_attn.qkv_proj.bias") : null, _qkv.View, H, _qkvCols);
            else
            {
                LinearInto(_hn.View, b + "self_attn.q_proj", _qkv.View, H, _qCols, _qkvCols, 0);
                LinearInto(_hn.View, b + "self_attn.k_proj", _qkv.View, H, _kvCols, _qkvCols, _qCols);
                LinearInto(_hn.View, b + "self_attn.v_proj", _qkv.View, H, _kvCols, _qkvCols, _qCols + _kvCols);
            }
            if (_qkNorm)
            {
                _u.RmsNormHeads(_qkv.View, _w.Vector(b + "self_attn.q_norm.weight"), SeqLen, _qkvCols, 0, _qHeads, HeadDim, RmsEps);
                _u.RmsNormHeads(_qkv.View, _w.Vector(b + "self_attn.k_norm.weight"), SeqLen, _qkvCols, _qCols, _kvHeads, HeadDim, RmsEps);
            }
            _u.Rope(_qkv.View, _invFreq.View, SeqLen, _qkvCols, 0, _qHeads, HeadDim, 0);
            _u.Rope(_qkv.View, _invFreq.View, SeqLen, _qkvCols, _qCols, _kvHeads, HeadDim, 0);
            _u.StoreKv(_qkv.View, _kc.View, _vc.View, 0, SeqLen, _qCols, _kvCols);
            _f.PaddedCausalGqa(_qkv.View, _kc.View, _vc.View, _scores.View, _attn.View, SeqLen, real, _qHeads, HeadDim, rep);
            Lin(_attn.View, b + "self_attn.o_proj.weight", null, _proj.View, _qCols, H);
            _k.GateResidual(_h.View, _proj.View, _dummy.View, -1, SeqLen * H, H);
            _u.RmsNorm(_h.View, _hn.View, _w.Vector(b + "post_attention_layernorm.weight"), SeqLen, H, RmsEps);
            if (_fusedGateUp) Lin(_hn.View, b + "mlp.gate_up_proj.weight", null, _gateUp.View, H, 2 * _inter);
            else
            {
                LinearInto(_hn.View, b + "mlp.gate_proj", _gateUp.View, H, _inter, 2 * _inter, 0);
                LinearInto(_hn.View, b + "mlp.up_proj", _gateUp.View, H, _inter, 2 * _inter, _inter);
            }
            _u.Swiglu(_gateUp.View, _act.View, SeqLen, _inter);
            Lin(_act.View, b + "mlp.down_proj.weight", null, _proj.View, _inter, H);
            _k.GateResidual(_h.View, _proj.View, _dummy.View, -1, SeqLen * H, H);
            Probe?.Invoke($"layer{l + 1}", _h.View, SeqLen, H);
            if (tap < _taps.Length && l + 1 == _taps[tap])
            {
                var src = _h.View;
                if (_finalNorm && tap == _taps.Length - 1) { _u.RmsNorm(_h.View, _hn.View, _w.Vector("norm.weight"), SeqLen, H, RmsEps); src = _hn.View; }
                _f.CopyCols(src, _out.View, SeqLen, H, H, Width, tap * H);
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
