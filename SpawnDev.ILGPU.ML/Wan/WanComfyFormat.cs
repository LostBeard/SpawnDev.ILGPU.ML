using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Trellis2;

namespace SpawnDev.ILGPU.ML.Wan;

/// <summary>
/// ComfyUI's repackaged Wan 2.2 files (Comfy-Org/Wan_2.2_ComfyUI_Repackaged, Apache-2.0: the official weights in ComfyUI's tensor layout) loaded as they are:
/// <see cref="Wan22Pipeline.LoadWeightsAsync(Accelerator, Hub.IModelSource, string, string, CancellationToken)"/> recognises the layout and rewrites the tensors
/// into this library's on the device, so a ComfyUI workflow's own files run without a separate export. Read from the files' headers only (2026-11-12).
/// VAE (<c>split_files/vae/wan2.2_vae.safetensors</c>): the same tensor names as our export; the convolutions are torch's [co, ci, kt, kh, kw] where ours are the
/// gather layout [co, taps, ci] (<c>tools/wan-harness/reference/export_vae_f16.py</c>), the RMS-norm gammas are fp16 [C, 1, 1(, 1)] where ours are fp32 [C], and
/// the latent normalisation (<c>latents_mean</c> / <c>latents_std</c>, the constants of the reference's <c>Wan2_2_VAE</c>) is not in the file.
/// </summary>
public static class WanComfyFormat
{
    // ---------------------------------------------------------------------------------------------------------------------------------------------------
    // DiT (split_files/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors): the reference's names with q / k / v unfused (our export fuses
    // self_attn.qkv = cat(q, k, v) and cross_attn.kv = cat(k, v): tools/wan-harness/reference/export_dit_bf16.py), fp16 where ours is the reference's bf16,
    // modulation [1, 6, C] / head.modulation [1, 2, C] / patch_embedding [C, 48, 1, 2, 2] where ours is [6, C] / [2, C] / [C, 192] (the same element order).
    // ---------------------------------------------------------------------------------------------------------------------------------------------------

    /// <summary>True when <paramref name="w"/> is ComfyUI's Wan 2.2 DiT layout (unfused q / k / v) rather than ours.</summary>
    public static bool IsComfyDit(GpuWeights w) => w.Has("blocks.0.self_attn.q.weight") && !w.Has("blocks.0.self_attn.qkv.weight");

    private static Action<Index1D, ArrayView1D<global::ILGPU.Half, Stride1D.Dense>, ArrayView1D<BFloat16, Stride1D.Dense>>? _toBf16;
    private static Accelerator? _toBf16Acc;
    // fp16 -> bf16 (ILGPU's (BFloat16)float cast rounds to nearest even; a bf16 value that went through fp16 comes back exactly unless fp16 made it subnormal)
    private static void ToBf16Impl(Index1D i, ArrayView1D<global::ILGPU.Half, Stride1D.Dense> src, ArrayView1D<BFloat16, Stride1D.Dense> dst) => dst[i] = (BFloat16)(float)src[i];
    private static Action<Index1D, ArrayView1D<global::ILGPU.Half, Stride1D.Dense>, ArrayView1D<BFloat16, Stride1D.Dense>, ArrayView1D<BFloat16, Stride1D.Dense>>? _toBf16HiLo;
    private static Accelerator? _toBf16HiLoAcc;
    private static void ToBf16HiLoImpl(Index1D i, ArrayView1D<global::ILGPU.Half, Stride1D.Dense> src, ArrayView1D<BFloat16, Stride1D.Dense> hi, ArrayView1D<BFloat16, Stride1D.Dense> lo)
    {
        float v = (float)src[i];
        var h = (BFloat16)v;
        hi[i] = h; lo[i] = (BFloat16)(v - (float)h);
    }
    /// <summary>fp16 -> bf16 hi + bf16 lo with hi + lo = the fp16 value exactly (the remainder has at most 3 significant bits).</summary>
    internal static void ToBf16HiLo(Accelerator acc, ArrayView1D<global::ILGPU.Half, Stride1D.Dense> src, ArrayView1D<BFloat16, Stride1D.Dense> hi, ArrayView1D<BFloat16, Stride1D.Dense> lo)
    {
        if (!ReferenceEquals(_toBf16HiLoAcc, acc)) { _toBf16HiLo = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<global::ILGPU.Half, Stride1D.Dense>, ArrayView1D<BFloat16, Stride1D.Dense>, ArrayView1D<BFloat16, Stride1D.Dense>>(ToBf16HiLoImpl); _toBf16HiLoAcc = acc; }
        _toBf16HiLo!((int)src.Length, src, hi, lo);
    }
    internal static void ToBf16(Accelerator acc, ArrayView1D<global::ILGPU.Half, Stride1D.Dense> src, ArrayView1D<BFloat16, Stride1D.Dense> dst)
    {
        if (!ReferenceEquals(_toBf16Acc, acc)) { _toBf16 = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<global::ILGPU.Half, Stride1D.Dense>, ArrayView1D<BFloat16, Stride1D.Dense>>(ToBf16Impl); _toBf16Acc = acc; }
        _toBf16!((int)src.Length, src, dst);
    }

    /// <summary>
    /// Rewrite ComfyUI's Wan 2.2 DiT layout into ours, in place: q / k / v fused, every tensor fp16 -> bf16, the three reshapes. One tensor at a time with a device
    /// wait before its fp16 buffer is freed, so the peak is the fp16 model plus one tensor (the 10 GB file fits a 12 GB card).
    /// </summary>
    public static async Task AdaptDitAsync(Accelerator accelerator, GpuWeights w)
    {
        async Task Retire(IDisposable? old) { if (old == null) return; await accelerator.SynchronizeAsync(); old.Dispose(); }
        int blocks = 0;
        while (w.Has($"blocks.{blocks}.self_attn.q.weight")) blocks++;
        for (int b = 0; b < blocks; b++)
        {
            foreach (var (fused, parts) in new[] { ("self_attn.qkv", new[] { "self_attn.q", "self_attn.k", "self_attn.v" }), ("cross_attn.kv", new[] { "cross_attn.k", "cross_attn.v" }) })
                foreach (var suffix in new[] { ".weight", ".bias" })
                {
                    string p = $"blocks.{b}.";
                    var names = parts.Select(x => p + x + suffix).ToArray();
                    long total = names.Sum(n => w.F16(n).Length);
                    var dst = accelerator.Allocate1D<BFloat16>(total);
                    long off = 0;
                    foreach (var n in names) { var src = w.F16(n); ToBf16(accelerator, src, dst.View.SubView(off, src.Length)); off += src.Length; }
                    var s0 = w.Shape(names[0]);
                    var shape = s0.Length == 2 ? new[] { s0[0] * names.Length, s0[1] } : new[] { (int)total };
                    w.SetBF16(p + fused + suffix, dst, shape);
                    foreach (var n in names) await Retire(w.Remove(n));
                }
        }
        foreach (var name in w.Names.ToList())
        {
            if (!w.IsF16(name)) continue;
            var s = w.Shape(name);
            int[] shape = name.EndsWith("modulation", StringComparison.Ordinal) && s.Length == 3 && s[0] == 1 ? new[] { s[1], s[2] }
                : name == "patch_embedding.weight" && s.Length == 5 ? new[] { s[0], s[1] * s[2] * s[3] * s[4] } : s;
            var src = w.F16(name);
            var dst = accelerator.Allocate1D<BFloat16>(src.Length);
            ToBf16(accelerator, src, dst.View);
            await Retire(w.SetBF16(name, dst, shape));
        }
        await accelerator.SynchronizeAsync();
    }

    /// <summary>True when <paramref name="w"/> is ComfyUI's Wan 2.2 VAE layout (torch-shaped convolutions, no latent normalisation) rather than ours.</summary>
    public static bool IsComfyVae(GpuWeights w) => !w.Has("latents_mean") && w.Has("conv1.weight") && w.Shape("conv1.weight").Length == 5;

    private static Action<Index1D, ArrayView1D<BFloat16, Stride1D.Dense>, ArrayView1D<global::ILGPU.Half, Stride1D.Dense>>? _toF16;
    private static Accelerator? _toF16Acc;
    // bf16 -> fp16 (exact for a VAE's weights: fp16 has 3 more mantissa bits; only values outside fp16's range or below its subnormals would change)
    private static void ToF16Impl(Index1D i, ArrayView1D<BFloat16, Stride1D.Dense> src, ArrayView1D<global::ILGPU.Half, Stride1D.Dense> dst) => dst[i] = (global::ILGPU.Half)(float)src[i];

    /// <summary>
    /// Rewrite ComfyUI's Wan VAE layout into ours, in place (see the class summary): Wan 2.2's <c>wan2.2_vae.safetensors</c> (fp16, 48 latent channels) and the Wan 2.1 VAE
    /// (16 channels: <c>wan_2.1_vae.safetensors</c>, Qwen-Image's <c>qwen_image_vae.safetensors</c>, bf16 tensors made fp16 first). The latent normalisation constants follow the channel
    /// count. Returns once the device has finished with the old buffers.
    /// </summary>
    public static async Task AdaptVaeAsync(Accelerator accelerator, GpuWeights w)
    {
        var sk = new SparseKernels(accelerator);
        var displaced = new List<IDisposable>();
        try
        {
            foreach (var name in w.Names.Where(w.IsBF16).ToList())
            {
                if (!ReferenceEquals(_toF16Acc, accelerator)) { _toF16 = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<BFloat16, Stride1D.Dense>, ArrayView1D<global::ILGPU.Half, Stride1D.Dense>>(ToF16Impl); _toF16Acc = accelerator; }
                var src = w.BF16(name);
                var dst = accelerator.Allocate1D<global::ILGPU.Half>(src.Length);
                _toF16!((int)src.Length, src, dst.View);
                if (w.SetF16(name, dst, w.Shape(name)) is { } old) displaced.Add(old);
            }
            foreach (var name in w.Names.ToList())
            {
                var s = w.Shape(name);
                if (name.EndsWith(".gamma", StringComparison.Ordinal) && w.IsF16(name))
                {
                    // fp16 [C, 1, 1(, 1)] -> fp32 [C]: tiny, through the host
                    var h = await w.F16(name).CopyToHostAsync();
                    var f = new float[h.Length];
                    for (int i = 0; i < h.Length; i++) f[i] = (float)h[i];
                    if (w.SetF32(name, accelerator.Allocate1D(f), new[] { f.Length }) is { } old) displaced.Add(old);
                }
                else if (name.EndsWith(".weight", StringComparison.Ordinal) && s.Length >= 4 && w.IsF16(name))
                {
                    // torch conv [co, ci, k...] -> the gather layout [co, taps, ci] (taps = the product of the kernel dims; a 1-tap kernel is the same element order)
                    int co = s[0], ci = s[1], taps = 1;
                    for (int d = 2; d < s.Length; d++) taps *= s[d];
                    if (taps == 1) { w.SetShape(name, new[] { co, 1, ci }); continue; }
                    var dst = accelerator.Allocate1D<global::ILGPU.Half>((long)co * taps * ci);
                    sk.PermuteConvWeight(w.F16(name), dst.View, ci, co, taps);
                    if (w.SetF16(name, dst, new[] { co, taps, ci }) is { } old) displaced.Add(old);
                }
            }
            bool v21 = w.Shape("conv2.weight")[0] == 16;                       // post_quant_conv: the latent's channels
            if (w.SetF32("latents_mean", accelerator.Allocate1D(v21 ? LatentsMean21 : LatentsMean), new[] { v21 ? 16 : 48 }) is { } m) displaced.Add(m);
            if (w.SetF32("latents_std", accelerator.Allocate1D(v21 ? LatentsStd21 : LatentsStd), new[] { v21 ? 16 : 48 }) is { } d2) displaced.Add(d2);
            await accelerator.SynchronizeAsync();        // the permutes have read the old buffers
        }
        finally
        {
            foreach (var b in displaced) b.Dispose();
        }
    }

    // ---------------------------------------------------------------------------------------------------------------------------------------------------
    // umT5 text encoder (split_files/text_encoders/umt5_xxl_fp16.safetensors, umt5_xxl_fp8_e4m3fn_scaled.safetensors): the Hugging Face T5 names
    //   shared.weight [256384, 4096]                                              -> token_embedding.weight (fp16 in both files; read row by row, never on the device)
    //   encoder.block.N.layer.0.layer_norm.weight                                 -> blocks.N.norm1.weight
    //   encoder.block.N.layer.0.SelfAttention.q / k / v.weight                    -> blocks.N.attn.qkv.weight (fused, rows q | k | v)
    //   encoder.block.N.layer.0.SelfAttention.o.weight                            -> blocks.N.attn.o.weight
    //   encoder.block.N.layer.0.SelfAttention.relative_attention_bias.weight      -> blocks.N.pos_embedding.embedding.weight
    //   encoder.block.N.layer.1.layer_norm.weight                                 -> blocks.N.norm2.weight
    //   encoder.block.N.layer.1.DenseReluDense.wi_0 / wi_1 / wo.weight            -> blocks.N.ffn.gate / ffn.fc1 / ffn.fc2.weight (wi_0 is the GELU branch)
    //   encoder.final_layer_norm.weight                                           -> norm.weight
    // fp16 file: every tensor fp16 (the reference's bf16 weights fit fp16 exactly unless fp16 makes them subnormal). fp8 file: the 7 matrices of a block are
    // F8_E4M3 with one fp32 scalar each (<name>.scale_weight: the weight is value * scale), the norms / bias tables / embedding fp16. spiece_model (the
    // SentencePiece model) and the empty scaled_fp8 marker are not weights. The layers are in name order (0, 1, 10, ...), the fp8 file grouped by dtype first.
    // ---------------------------------------------------------------------------------------------------------------------------------------------------

    /// <summary>True when the tensor names are a Hugging Face layout T5 encoder (ComfyUI's umT5 files) rather than our export's.</summary>
    public static bool IsComfyT5(IEnumerable<string> names)
    {
        bool shared = false, q = false;
        foreach (var n in names)
        {
            if (n == "token_embedding.weight") return false;
            shared |= n == "shared.weight";
            q |= n == "encoder.block.0.layer.0.SelfAttention.q.weight";
        }
        return shared && q;
    }

    private static readonly (string Theirs, string Ours)[] T5BlockNames =
    {
        ("layer.0.layer_norm.weight", "norm1.weight"),
        ("layer.0.SelfAttention.q.weight", "attn.qkv.weight"), ("layer.0.SelfAttention.k.weight", "attn.qkv.weight"), ("layer.0.SelfAttention.v.weight", "attn.qkv.weight"),
        ("layer.0.SelfAttention.o.weight", "attn.o.weight"),
        ("layer.0.SelfAttention.relative_attention_bias.weight", "pos_embedding.embedding.weight"),
        ("layer.1.layer_norm.weight", "norm2.weight"),
        ("layer.1.DenseReluDense.wi_0.weight", "ffn.gate.weight"), ("layer.1.DenseReluDense.wi_1.weight", "ffn.fc1.weight"), ("layer.1.DenseReluDense.wo.weight", "ffn.fc2.weight"),
    };
    private const string ScaleSuffix = ".scale_weight", WeightSuffix = ".weight";

    /// <summary>
    /// Our export's name for a tensor of a Hugging Face layout T5 encoder file (see <see cref="IsComfyT5"/>): the tensor it becomes, or becomes part of (q, k and v
    /// map to the fused <c>attn.qkv</c>; a <c>scale_weight</c> maps to its matrix). Null for a tensor that is not a weight of the encoder (the SentencePiece blob, markers).
    /// </summary>
    public static string? T5OurName(string name)
    {
        if (name == "shared.weight") return "token_embedding.weight";
        if (name == "encoder.final_layer_norm.weight") return "norm.weight";
        const string Prefix = "encoder.block.";
        if (!name.StartsWith(Prefix, StringComparison.Ordinal)) return null;
        int dot = name.IndexOf('.', Prefix.Length);
        if (dot < 0 || !int.TryParse(name.AsSpan(Prefix.Length, dot - Prefix.Length), System.Globalization.NumberStyles.None, System.Globalization.CultureInfo.InvariantCulture, out int block)) return null;
        string rest = name[(dot + 1)..];
        if (rest.EndsWith(ScaleSuffix, StringComparison.Ordinal)) rest = rest[..^ScaleSuffix.Length] + WeightSuffix;
        foreach (var (theirs, ours) in T5BlockNames)
            if (rest == theirs) return $"blocks.{block}.{ours}";
        return null;
    }

    private static Action<Index1D, ArrayView1D<uint, Stride1D.Dense>, ArrayView1D<BFloat16, Stride1D.Dense>, float>? _f8ToBf16;
    private static Accelerator? _f8ToBf16Acc;
    // F8_E4M3 (the "fn" variant: sign, 4 exponent bits biased by 7, 3 mantissa bits; no infinities, S.1111.111 is NaN; exponent 0 is subnormal, mantissa * 2^-9)
    // times the tensor's scale, to bf16 (round to nearest even)
    private static void F8ToBf16Impl(Index1D i, ArrayView1D<uint, Stride1D.Dense> src, ArrayView1D<BFloat16, Stride1D.Dense> dst, float scale)
    {
        uint b = (src[i >> 2] >> ((i & 3) << 3)) & 0xFFu;
        uint e = (b >> 3) & 15u, m = b & 7u;
        float mag = Interop.IntAsFloat(((e + 120u) << 23) | (m << 20));
        if (e == 0u) mag = m * 0.001953125f;
        if ((b & 0x7Fu) == 0x7Fu) mag = float.NaN;
        if ((b & 0x80u) != 0u) mag = -mag;
        dst[i] = (BFloat16)(mag * scale);
    }

    /// <summary>
    /// Decode F8_E4M3 values (stored four bytes to a word: <see cref="GpuWeights.F8Words"/>) times <paramref name="scale"/> into bf16, rounded to nearest even;
    /// <paramref name="dst"/>'s length is the value count.
    /// </summary>
    public static void DecodeF8E4M3(Accelerator accelerator, ArrayView1D<uint, Stride1D.Dense> packed, ArrayView1D<BFloat16, Stride1D.Dense> dst, float scale)
    {
        if (packed.Length * 4 < dst.Length) throw new ArgumentException($"{dst.Length} values need {(dst.Length + 3) / 4} words, got {packed.Length}");
        if (!ReferenceEquals(_f8ToBf16Acc, accelerator)) { _f8ToBf16 = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<uint, Stride1D.Dense>, ArrayView1D<BFloat16, Stride1D.Dense>, float>(F8ToBf16Impl); _f8ToBf16Acc = accelerator; }
        _f8ToBf16!((int)dst.Length, packed, dst, scale);
    }

    /// <summary>
    /// Rewrite the tensors of a Hugging Face layout T5 encoder held in <paramref name="w"/> (one layer of ComfyUI's umT5 file, or the final norm) into our export's
    /// layout, in place: our names, q / k / v fused, everything bf16 (fp16 converted; F8_E4M3 decoded times its <c>scale_weight</c>). One matrix at a time with a
    /// device wait before its source is freed.
    /// The 8-bit matrices become bf16(value * scale): the engine's weight precision for this encoder (the reference's weights ARE bf16). MEASURED on the host
    /// (2026-11-12, 21 matrices): value * scale is 2.65e-2 relative RMS from the fp16 file's weights (the 8-bit format's own rounding) and the bf16 rounding of
    /// value * scale is 1.1e-3 to 1.9e-3, which adds 0.2% to that error in quadrature.
    /// </summary>
    public static Task AdaptT5Async(Accelerator accelerator, GpuWeights w) => AdaptT5Async(accelerator, w, null);

    /// <summary>
    /// The same; with <paramref name="exactScaled"/>, an 8-bit scaled file's matrices are NOT folded into bf16(f8 * scale): each fused group keeps its 8-bit values as they are (exact in bf16)
    /// and <paramref name="exactScaled"/>[our name] lists its parts' (rows, scale) in order, for the caller to apply the scales exactly (UmT5Encoder: the input times the scale for one part,
    /// the output columns of each part for a fused q | k | v).
    /// </summary>
    public static async Task AdaptT5Async(Accelerator accelerator, GpuWeights w, Dictionary<string, (int Rows, float Scale)[]>? exactScaled)
    {
        async Task Retire(IDisposable? old) { if (old == null) return; await accelerator.SynchronizeAsync(); old.Dispose(); }
        // our name -> its source tensors in the fused order (the table's order: q, k, v)
        var groups = new Dictionary<string, List<string>>();
        var scales = new Dictionary<string, string>();                 // matrix name -> its scale_weight
        foreach (var name in w.Names.ToList())
        {
            if (T5OurName(name) is not { } ours) continue;
            if (name.EndsWith(ScaleSuffix, StringComparison.Ordinal)) { scales[name[..^ScaleSuffix.Length] + WeightSuffix] = name; continue; }
            if (!groups.TryGetValue(ours, out var list)) groups[ours] = list = new List<string>();
            list.Add(name);
        }
        int Order(string n) => Array.FindIndex(T5BlockNames, t => n.EndsWith("." + t.Theirs, StringComparison.Ordinal));
        foreach (var (ours, sources) in groups)
        {
            sources.Sort((a, b) => Order(a).CompareTo(Order(b)));
            if (ours.EndsWith("attn.qkv.weight", StringComparison.Ordinal) && sources.Count != 3) throw new InvalidDataException($"{ours}: needs q, k and v, got {string.Join(", ", sources)}");
            long total = 0;
            foreach (var n in sources) { long c = 1; foreach (int d in w.Shape(n)) c *= d; total += c; }
            if (exactScaled != null && sources.All(w.IsF8) && sources.All(n => scales.ContainsKey(n)))
            {
                // exact: the 8-bit values fused as they are (four to a word: every part's count is a multiple of 4), the scales returned
                var words = accelerator.Allocate1D<uint>(total / 4);
                long woff = 0;
                var parts = new (int Rows, float Scale)[sources.Count];
                for (int i = 0; i < sources.Count; i++)
                {
                    var src = w.F8Words(sources[i]);
                    words.View.SubView(woff, src.Length).CopyFrom(src); woff += src.Length;
                    parts[i] = (w.Shape(sources[i])[0], (await w.F32(scales[sources[i]]).CopyToHostAsync())[0]);
                }
                var s0x = w.Shape(sources[0]);
                foreach (var n in sources) await Retire(w.Remove(n));
                await Retire(w.SetF8(ours, words, sources.Count == 1 ? s0x : new[] { s0x[0] * sources.Count, s0x[1] }));
                exactScaled[ours] = parts;
                continue;
            }
            var dst = accelerator.Allocate1D<BFloat16>(total);
            long off = 0;
            foreach (var n in sources)
            {
                long c = 1; foreach (int d in w.Shape(n)) c *= d;
                var part = dst.View.SubView(off, c);
                if (w.IsF8(n))
                {
                    // ComfyUI's "scaled fp8" files carry one fp32 scale_weight per matrix; its all-in-one FLUX.1 checkpoints store plain 8-bit floats with no scale at all
                    // (value * 1). A file that has scales must have this matrix's: a missing one is an error, not a silent 1.
                    float scale = 1f;
                    if (scales.TryGetValue(n, out var sn) && w.IsF32(sn)) scale = (await w.F32(sn).CopyToHostAsync())[0];      // one scalar
                    else if (scales.Count > 0) throw new InvalidDataException($"{n}: an F8_E4M3 matrix without its fp32 scale_weight");
                    DecodeF8E4M3(accelerator, w.F8Words(n), part, scale);
                }
                else if (w.IsF16(n)) ToBf16(accelerator, w.F16(n), part);
                else if (w.IsBF16(n)) part.CopyFrom(w.BF16(n));
                else throw new NotSupportedException($"{n}: expected fp16, bf16 or F8_E4M3");
                off += c;
            }
            var s0 = w.Shape(sources[0]);
            int[] shape = sources.Count == 1 ? s0 : new[] { s0[0] * sources.Count, s0[1] };
            foreach (var n in sources) await Retire(w.Remove(n));
            w.SetBF16(ours, dst, shape);
        }
        foreach (var sn in scales.Values) await Retire(w.Remove(sn));
        await accelerator.SynchronizeAsync();
    }

    // the reference Wan2_2_VAE's latent normalisation (decode: z * std + mean), as our export stores it (fp32, shortest round-trip literals)
    // the Wan 2.1 VAE's (Wan 2.1, Qwen-Image: diffusers' AutoencoderKLWan / AutoencoderKLQwenImage defaults)
    private static readonly float[] LatentsMean21 = { -0.7571f, -0.7089f, -0.9113f, 0.1075f, -0.1745f, 0.9653f, -0.1517f, 1.5508f, 0.4134f, -0.0715f, 0.5517f, -0.3632f, -0.1922f, -0.9497f, 0.2503f, -0.2921f };
    private static readonly float[] LatentsStd21 = { 2.8184f, 1.4541f, 2.3275f, 2.6558f, 1.2196f, 1.7708f, 2.6052f, 2.0743f, 3.2687f, 2.1526f, 2.8652f, 1.5579f, 1.6382f, 1.1253f, 2.8251f, 1.9160f };
    private static readonly float[] LatentsMean = { -0.2289f, -0.0052f, -0.1323f, -0.2339f, -0.2799f, 0.0174f, 0.1838f, 0.1557f, -0.1382f, 0.0542f, 0.2813f, 0.0891f, 0.157f, -0.0098f, 0.0375f, -0.1825f, -0.2246f, -0.1207f, -0.0698f, 0.5109f, 0.2665f, -0.2108f, -0.2158f, 0.2502f, -0.2055f, -0.0322f, 0.1109f, 0.1567f, -0.0729f, 0.0899f, -0.2799f, -0.123f, -0.0313f, -0.1649f, 0.0117f, 0.0723f, -0.2839f, -0.2083f, -0.052f, 0.3748f, 0.0152f, 0.1957f, 0.1433f, -0.2944f, 0.3573f, -0.0548f, -0.1681f, -0.0667f };
    private static readonly float[] LatentsStd = { 0.4765f, 1.0364f, 0.4514f, 1.1677f, 0.5313f, 0.499f, 0.4818f, 0.5013f, 0.8158f, 1.0344f, 0.5894f, 1.0901f, 0.6885f, 0.6165f, 0.8454f, 0.4978f, 0.5759f, 0.3523f, 0.7135f, 0.6804f, 0.5833f, 1.4146f, 0.8986f, 0.5659f, 0.7069f, 0.5338f, 0.4889f, 0.4917f, 0.4069f, 0.4999f, 0.6866f, 0.4093f, 0.5709f, 0.6065f, 0.6415f, 0.4944f, 0.5726f, 1.2042f, 0.5458f, 1.6887f, 0.3971f, 1.06f, 0.3943f, 0.5537f, 0.5444f, 0.4089f, 0.7468f, 0.7744f };
}
