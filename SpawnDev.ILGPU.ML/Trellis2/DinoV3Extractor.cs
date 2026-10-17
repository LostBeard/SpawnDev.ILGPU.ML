using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Trellis2;

using FV = ArrayView1D<float, Stride1D.Dense>;
using HV = ArrayView1D<global::ILGPU.Half, Stride1D.Dense>;

/// <summary>
/// The image-conditioning features of TRELLIS.2 (DinoV3FeatureExtractor.extract_features): DINOv3 ViT-L/16 (24 layers, 1024 wide, 16 heads of 64,
/// 4 register tokens, rotary position embedding on the patch tokens only, layer scale, exact-GELU MLP), run through every layer and then
/// <c>F.layer_norm</c> WITHOUT parameters (the model's own final norm is not used). Input: a normalised RGB image [3, S, S]; output: tokens
/// [1 + 4 + (S/16)^2, 1024] (1029 tokens at 512 px). Weights are the Hugging Face checkpoint (fp32) converted once to fp16 for the tensor-core
/// GEMMs; both key layouts are accepted (<c>layer.N.*</c> and <c>model.layer.N.*</c>). This class takes ownership of the weights and frees the
/// fp32 copy after the conversion.
/// </summary>
public sealed class DinoV3Extractor : IDisposable
{
    private const int C = 1024, Heads = 16, HeadDim = 64, Layers = 24, Patch = 16, Registers = 4, Mlp = 4096;
    private readonly Accelerator _acc;
    private readonly SparseKernels _sk;
    private readonly DiTKernels _dk;
    private GpuWeights? _raw;
    private readonly List<IDisposable> _owned = new();
    private readonly string _prefix;
    private readonly bool _timm;

    private sealed class LayerWeights
    {
        public HV Qkv, QkvBias, O, OBias, Up, UpBias, Down, DownBias;
        public FV Norm1W, Norm1B, Norm2W, Norm2B, Scale1, Scale2;
    }
    private readonly LayerWeights[] _layers = new LayerWeights[Layers];
    private HV _patchW, _patchB;
    private FV _cls, _reg;

    // ---- kernels ----
    private Action<Index1D, FV, HV>? _toHalf;
    private static void ToHalfImpl(Index1D i, FV src, HV dst) => dst[i] = PrecisionConvert.ConvertFromSingle<global::ILGPU.Half>(src[i]);
    private Action<Index1D, FV, FV, int, int>? _patches;
    private static void PatchesImpl(Index1D idx, FV img, FV o, int size, int pw)
    {
        int p = idx / (3 * Patch * Patch), k = idx % (3 * Patch * Patch);
        int ci = k / (Patch * Patch), ky = k / Patch % Patch, kx = k % Patch;
        int py = p / pw, px = p % pw;
        o[idx] = img[((long)ci * size + py * Patch + ky) * size + px * Patch + kx];
    }
    private Action<Index1D, FV, FV, FV, FV, int>? _assemble;
    private static void AssembleImpl(Index1D idx, FV cls, FV reg, FV patch, FV tokens, int numPatches)
    {
        int row = idx / C, c = idx % C;
        float v = row == 0 ? cls[c] : row <= Registers ? reg[(row - 1) * C + c] : patch[(long)(row - 1 - Registers) * C + c];
        tokens[idx] = v;
    }
    private Action<Index1D, FV, FV, int, int, float>? _ropeTable;
    private static void RopeTableImpl(Index1D idx, FV cosT, FV sinT, int ph, int pw, float theta)
    {
        int d = idx % HeadDim, p = idx / HeadDim;
        int m = d % (HeadDim / 2);
        int axis = m / (HeadDim / 4), k = m % (HeadDim / 4);
        int py = p / pw, px = p % pw;
        float coord = axis == 0 ? 2f * ((py + 0.5f) / ph) - 1f : 2f * ((px + 0.5f) / pw) - 1f;
        float invFreq = 1f / MathF.Pow(theta, (float)k / (HeadDim / 4));
        float angle = 6.283185307179586f * coord * invFreq;
        cosT[idx] = MathF.Cos(angle); sinT[idx] = MathF.Sin(angle);
    }
    private Action<Index1D, FV, FV, FV, int, int>? _rope;
    private static void RopeImpl(Index1D idx, FV qkv, FV cosT, FV sinT, int nPrefix, int n)
    {
        // idx over (patch token, which in {q,k}, head, d < HeadDim/2)
        int d = idx % (HeadDim / 2); int r = idx / (HeadDim / 2);
        int h = r % Heads; r /= Heads;
        int which = r % 2; int p = r / 2;
        long b = (long)(nPrefix + p) * 3 * C + which * C + h * HeadDim;
        float x1 = qkv[b + d], x2 = qkv[b + d + HeadDim / 2];
        long t = (long)p * HeadDim;
        qkv[b + d] = x1 * cosT[t + d] - x2 * sinT[t + d];
        qkv[b + d + HeadDim / 2] = x2 * cosT[t + d + HeadDim / 2] + x1 * sinT[t + d + HeadDim / 2];
    }

    public DinoV3Extractor(Accelerator accelerator, SparseKernels sparse, DiTKernels dit, GpuWeights weights)
    {
        _acc = accelerator; _sk = sparse; _dk = dit; _raw = weights;
        _timm = weights.Has("patch_embed.proj.weight");          // timm layout (blocks.N.attn.qkv, gamma_1, mlp.fc1) vs transformers layout
        _prefix = weights.Has("model.layer.0.norm1.weight") ? "model.layer." : "layer.";
        _toHalf = _acc.LoadAutoGroupedStreamKernel<Index1D, FV, HV>(ToHalfImpl);
        Convert();
        _raw!.Dispose(); _raw = null;   // the fp32 originals are no longer needed
    }

    private HV Half(long n) { var b = _acc.Allocate1D<global::ILGPU.Half>(n); b.MemSetToZero(); _owned.Add(b); return b.View; }
    private FV Float(long n) { var b = _acc.Allocate1D<float>(n); _owned.Add(b); return b.View; }
    private void ToHalf(FV src, HV dst, long count) => _toHalf!((int)count, src, dst);
    private FV F32Copy(string name)
    {
        var src = _raw!.F32(name); var dst = Float(src.Length); dst.CopyFrom(src); return dst;
    }

    private void Convert()
    {
        var w = _raw!;
        _cls = F32Copy(_timm ? "cls_token" : "embeddings.cls_token");
        _reg = F32Copy(_timm ? "reg_token" : "embeddings.register_tokens");
        // patch embedding: Conv2d [1024, 3, 16, 16] == Linear [1024, 768] on the flattened (c, ky, kx) patch
        string patchName = _timm ? "patch_embed.proj" : "embeddings.patch_embeddings";
        _patchW = Half(C * 3 * Patch * Patch); ToHalf(w.F32(patchName + ".weight"), _patchW, C * 3 * Patch * Patch);
        _patchB = Half(C); ToHalf(w.F32(patchName + ".bias"), _patchB, C);
        for (int i = 0; i < Layers; i++)
        {
            var l = new LayerWeights();
            l.Qkv = Half(3L * C * C); l.QkvBias = Half(3L * C);                      // zero-initialised: the key projection has no bias
            if (_timm)
            {
                string p = $"blocks.{i}.";
                ToHalf(w.F32(p + "attn.qkv.weight"), l.Qkv, 3L * C * C);             // fused [q; k; v] rows
                if (w.Has(p + "attn.qkv.bias")) ToHalf(w.F32(p + "attn.qkv.bias"), l.QkvBias, 3L * C);
                else
                {
                    if (w.Has(p + "attn.q_bias")) ToHalf(w.F32(p + "attn.q_bias"), l.QkvBias.SubView(0, C), C);
                    if (w.Has(p + "attn.v_bias")) ToHalf(w.F32(p + "attn.v_bias"), l.QkvBias.SubView(2L * C, C), C);
                }
                l.O = Half((long)C * C); ToHalf(w.F32(p + "attn.proj.weight"), l.O, (long)C * C);
                l.OBias = Half(C); ToHalf(w.F32(p + "attn.proj.bias"), l.OBias, C);
                l.Up = Half((long)Mlp * C); ToHalf(w.F32(p + "mlp.fc1.weight"), l.Up, (long)Mlp * C);
                l.UpBias = Half(Mlp); ToHalf(w.F32(p + "mlp.fc1.bias"), l.UpBias, Mlp);
                l.Down = Half((long)C * Mlp); ToHalf(w.F32(p + "mlp.fc2.weight"), l.Down, (long)C * Mlp);
                l.DownBias = Half(C); ToHalf(w.F32(p + "mlp.fc2.bias"), l.DownBias, C);
                l.Norm1W = F32Copy(p + "norm1.weight"); l.Norm1B = F32Copy(p + "norm1.bias");
                l.Norm2W = F32Copy(p + "norm2.weight"); l.Norm2B = F32Copy(p + "norm2.bias");
                l.Scale1 = F32Copy(p + "gamma_1"); l.Scale2 = F32Copy(p + "gamma_2");
            }
            else
            {
                string p = $"{_prefix}{i}.";
                ToHalf(w.F32(p + "attention.q_proj.weight"), l.Qkv.SubView(0, (long)C * C), (long)C * C);
                ToHalf(w.F32(p + "attention.k_proj.weight"), l.Qkv.SubView((long)C * C, (long)C * C), (long)C * C);
                ToHalf(w.F32(p + "attention.v_proj.weight"), l.Qkv.SubView(2L * C * C, (long)C * C), (long)C * C);
                ToHalf(w.F32(p + "attention.q_proj.bias"), l.QkvBias.SubView(0, C), C);
                ToHalf(w.F32(p + "attention.v_proj.bias"), l.QkvBias.SubView(2L * C, C), C);
                l.O = Half((long)C * C); ToHalf(w.F32(p + "attention.o_proj.weight"), l.O, (long)C * C);
                l.OBias = Half(C); ToHalf(w.F32(p + "attention.o_proj.bias"), l.OBias, C);
                l.Up = Half((long)Mlp * C); ToHalf(w.F32(p + "mlp.up_proj.weight"), l.Up, (long)Mlp * C);
                l.UpBias = Half(Mlp); ToHalf(w.F32(p + "mlp.up_proj.bias"), l.UpBias, Mlp);
                l.Down = Half((long)C * Mlp); ToHalf(w.F32(p + "mlp.down_proj.weight"), l.Down, (long)C * Mlp);
                l.DownBias = Half(C); ToHalf(w.F32(p + "mlp.down_proj.bias"), l.DownBias, C);
                l.Norm1W = F32Copy(p + "norm1.weight"); l.Norm1B = F32Copy(p + "norm1.bias");
                l.Norm2W = F32Copy(p + "norm2.weight"); l.Norm2B = F32Copy(p + "norm2.bias");
                l.Scale1 = F32Copy(p + "layer_scale1.lambda1"); l.Scale2 = F32Copy(p + "layer_scale2.lambda1");
            }
            _layers[i] = l;
        }
        _acc.Fence();
    }

    /// <summary>
    /// <paramref name="image"/> is [3, S, S] (channel-first, ImageNet-normalised); <paramref name="o"/> receives [tokens, 1024].
    /// </summary>
    public void Extract(FV image, int size, FV o)
    {
        if (size % Patch != 0) throw new ArgumentException("image size must be a multiple of 16");
        _patches ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, int, int>(PatchesImpl);
        _assemble ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, FV, FV, int>(AssembleImpl);
        _ropeTable ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, int, int, float>(RopeTableImpl);
        _rope ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, FV, int, int>(RopeImpl);
        int pw = size / Patch, P = pw * pw, N = 1 + Registers + P;
        using var patchIn = _acc.Allocate1D<float>((long)P * 3 * Patch * Patch);
        using var patchOut = _acc.Allocate1D<float>((long)P * C);
        using var h = _acc.Allocate1D<float>((long)N * C);
        using var hn = _acc.Allocate1D<float>((long)N * C);
        using var qkv = _acc.Allocate1D<float>((long)N * 3 * C);
        using var attn = _acc.Allocate1D<float>((long)N * C);
        using var proj = _acc.Allocate1D<float>((long)N * C);
        using var mlp = _acc.Allocate1D<float>((long)N * Mlp);
        using var scores = _acc.Allocate1D<float>(_dk.UsesFlashAttention(HeadDim) ? 1 : (long)Heads * N * N);   // flash kernels need no score matrix
        using var cosT = _acc.Allocate1D<float>((long)P * HeadDim);
        using var sinT = _acc.Allocate1D<float>((long)P * HeadDim);

        _patches(P * 3 * Patch * Patch, image, patchIn.View, size, pw);
        _sk.Linear(patchIn.View, _patchW, _patchB, patchOut.View, P, 3 * Patch * Patch, C);
        _assemble(N * C, _cls, _reg, patchOut.View, h.View, P);
        _ropeTable(P * HeadDim, cosT.View, sinT.View, pw, pw, 100f);

        for (int i = 0; i < Layers; i++)
        {
            var l = _layers[i];
            _sk.LayerNormF32(h.View, hn.View, l.Norm1W, l.Norm1B, N, C, 1e-5f);
            _sk.Linear(hn.View, l.Qkv, l.QkvBias, qkv.View, N, C, 3 * C);
            _rope((P) * Heads * 2 * (HeadDim / 2), qkv.View, cosT.View, sinT.View, 1 + Registers, N);
            _dk.Attention(qkv.View, 3 * C, 0, qkv.View, 3 * C, C, 2 * C, scores.View, attn.View, N, N, Heads, HeadDim);
            _sk.Linear(attn.View, l.O, l.OBias, proj.View, N, C, C);
            _dk.GateResidual(h.View, proj.View, l.Scale1, 0, N * C, C);          // layer scale, then the residual add
            _sk.LayerNormF32(h.View, hn.View, l.Norm2W, l.Norm2B, N, C, 1e-5f);
            _sk.Linear(hn.View, l.Up, l.UpBias, mlp.View, N, C, Mlp, act: 3);
            _sk.Linear(mlp.View, l.Down, l.DownBias, proj.View, N, Mlp, C);
            _dk.GateResidual(h.View, proj.View, l.Scale2, 0, N * C, C);
        }
        _dk.LayerNormPlain(h.View, o, N, C, 1e-5f);                                  // F.layer_norm(hidden_states, hidden_states.shape[-1:])
        _acc.Fence();
    }

    public void Dispose()
    {
        foreach (var d in _owned) d.Dispose();
        _owned.Clear();
        _raw?.Dispose();
    }
}
