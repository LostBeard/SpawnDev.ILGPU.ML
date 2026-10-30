using System.Text.Json;
using System.Text.RegularExpressions;
using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.StableDiffusion;

using HV = ArrayView1D<global::ILGPU.Half, Stride1D.Dense>;

/// <summary>The element type a saved file stores.</summary>
public enum SdSaveDType { F16, F32, BF16 }

/// <summary>How <see cref="SdCheckpoint.SaveAsync"/> and the component SaveAsync methods write a .safetensors file.</summary>
public sealed class SdSaveOptions
{
    /// <summary>The stored element type (default fp16, what the weights are held as: no rounding).</summary>
    public SdSaveDType DType { get; set; } = SdSaveDType.F16;
    /// <summary>String pairs for the header's <c>__metadata__</c> (null = none).</summary>
    public IReadOnlyDictionary<string, string>? Metadata { get; set; }
    /// <summary>
    /// Write ComfyUI's empty <c>v_pred</c> tensor, which marks a v-prediction model (its loaders and ours read it). Null = the checkpoint's <see cref="SdCheckpoint.Prediction"/>
    /// for <see cref="SdCheckpoint.SaveAsync"/>, no marker for a component saved alone.
    /// </summary>
    public bool? VPrediction { get; set; }
}

/// <summary>
/// Writes resident fp16 weights as a .safetensors file in the layout ComfyUI and the LDM single files use, streaming front to back (no seeking: a browser OPFS or download
/// stream works). Every tensor goes through one reused device staging buffer: a kernel undoes the load-time layout (the convolution gather order, the transposed OpenCLIP
/// text projection, the fused q | k | v rows) and converts to the stored type, then <c>CopyToStreamAsync</c> sends the bytes out (GPU -> JS -> stream in the browser,
/// never through the .NET heap). Names: UNet <c>model.diffusion_model.*</c>, VAE <c>first_stage_model.*</c>, text encoders <c>cond_stage_model.transformer.text_model.*</c>
/// (SD 1.x), <c>cond_stage_model.model.*</c> (SD 2.x, OpenCLIP), <c>conditioner.embedders.0.transformer.text_model.*</c> + <c>conditioner.embedders.1.model.*</c> (SDXL).
/// Tensors the loaders do not read (EMA copies, <c>alphas_cumprod</c>, <c>position_ids</c>, <c>logit_scale</c>) are not written, as ComfyUI's CheckpointSave does not.
/// </summary>
internal static class SdSafeTensorsWriter
{
    /// <summary>One tensor of the file. Layout 0 as held, 1 gather -> file ([cout][taps][cin] held, [cout][cin][taps] in the file: A = cin, B = taps),
    /// 2 transposed (held [A][B], the file [B][A]). <c>Offset</c> = the first element of the held tensor it starts at (a fused part's rows).</summary>
    internal readonly record struct Out(string Name, SdWeights Store, string Tensor, long Offset, int[] Shape, int Layout, int A, int B);

    /// <summary>The tensors of a UNet or VAE store under <paramref name="prefix"/>: the held (LDM) names, fused parts split back, 4-D convolutions back to [cout, cin, kH, kW].</summary>
    public static List<Out> Ldm(SdWeights w, string prefix)
    {
        var list = new List<Out>();
        foreach (var (src, t, rowOff, rows) in w.Sources())
        {
            var ts = w.Shape(t);
            long cols = 1; for (int d = 1; d < ts.Length; d++) cols *= ts[d];
            var shape = (int[])ts.Clone(); if (shape.Length > 0) shape[0] = rows;
            bool gather = shape.Length == 4 && shape[2] * shape[3] > 1;
            list.Add(new Out(prefix + src, w, t, rowOff * cols, shape, gather ? 1 : 0, gather ? shape[1] : 1, gather ? shape[2] * shape[3] : 1));
        }
        return list;
    }

    private static readonly Regex QkvPart = new(@"^(.+\.)(q|k|v)\.(weight|bias)$", RegexOptions.Compiled);

    /// <summary>Where a normalised text tensor lives: whole, a part fused into q|k|v at load, or a third of an OpenCLIP in_proj (a tower loaded from OpenCLIP names, saved as HF).</summary>
    private static (string Tensor, long Offset, int[] Shape)? Part(SdWeights w, string n)
    {
        var loc = w.Locate(n);
        if (loc is { } l)
        {
            var ts = w.Shape(l.Name);
            long cols = 1; for (int d = 1; d < ts.Length; d++) cols *= ts[d];
            var shape = (int[])ts.Clone(); shape[0] = l.Rows;
            return (l.Name, l.RowOffset * cols, shape);
        }
        var m = QkvPart.Match(n);
        if (!m.Success) return null;
        string qkv = m.Groups[1].Value + "qkv." + m.Groups[3].Value;
        if (!w.Has(qkv)) return null;
        var s = w.Shape(qkv);
        int C = s[0] / 3, part = "qkv".IndexOf(m.Groups[2].Value[0]);
        long c2 = 1; for (int d = 1; d < s.Length; d++) c2 *= s[d];
        var shp = (int[])s.Clone(); shp[0] = C;
        return (qkv, (long)part * C * c2, shp);
    }

    private static void Add(List<Out> list, SdWeights w, string name, string n, bool transpose = false)
    {
        var p = Part(w, n) ?? throw new InvalidDataException($"text encoder tensor '{n}' is not loaded: cannot write '{name}'");
        if (!transpose) { list.Add(new Out(name, w, p.Tensor, p.Offset, p.Shape, 0, 1, 1)); return; }
        if (p.Shape.Length != 2) throw new InvalidDataException($"'{n}' is not 2-D");
        list.Add(new Out(name, w, p.Tensor, p.Offset, new[] { p.Shape[1], p.Shape[0] }, 2, p.Shape[0], p.Shape[1]));
    }

    private static int Layers(SdWeights w, string p) { int i = 0; while (Part(w, $"{p}layers.{i}.ln1.weight") != null) i++; return i; }

    private static void HfTower(List<Out> list, SdWeights w, string p, string pre)
    {
        Add(list, w, pre + "embeddings.token_embedding.weight", p + "tok");
        Add(list, w, pre + "embeddings.position_embedding.weight", p + "pos");
        int L = Layers(w, p);
        for (int i = 0; i < L; i++)
            foreach (var (hf, local) in new[] { ("self_attn.q_proj", "q"), ("self_attn.k_proj", "k"), ("self_attn.v_proj", "v"), ("self_attn.out_proj", "out"),
                ("layer_norm1", "ln1"), ("layer_norm2", "ln2"), ("mlp.fc1", "fc1"), ("mlp.fc2", "fc2") })
                foreach (var suf in new[] { "weight", "bias" })
                    Add(list, w, $"{pre}encoder.layers.{i}.{hf}.{suf}", $"{p}layers.{i}.{local}.{suf}");
        Add(list, w, pre + "final_layer_norm.weight", p + "final_ln.weight");
        Add(list, w, pre + "final_layer_norm.bias", p + "final_ln.bias");
    }

    private static void OpenClipTower(List<Out> list, SdWeights w, string p, string pre)
    {
        Add(list, w, pre + "token_embedding.weight", p + "tok");
        Add(list, w, pre + "positional_embedding", p + "pos");
        int L = Layers(w, p);
        for (int i = 0; i < L; i++)
            foreach (var (oc, local) in new[] { ("ln_1", "ln1"), ("attn.out_proj", "out"), ("ln_2", "ln2"), ("mlp.c_fc", "fc1"), ("mlp.c_proj", "fc2") })
                foreach (var suf in new[] { "weight", "bias" })
                    Add(list, w, $"{pre}transformer.resblocks.{i}.{oc}.{suf}", $"{p}layers.{i}.{local}.{suf}");
        for (int i = 0; i < L; i++)
        {
            Add(list, w, $"{pre}transformer.resblocks.{i}.attn.in_proj_weight", $"{p}layers.{i}.qkv.weight");
            Add(list, w, $"{pre}transformer.resblocks.{i}.attn.in_proj_bias", $"{p}layers.{i}.qkv.bias");
        }
        Add(list, w, pre + "ln_final.weight", p + "final_ln.weight");
        Add(list, w, pre + "ln_final.bias", p + "final_ln.bias");
        if (Part(w, p + "proj.weight") != null) Add(list, w, pre + "text_projection", p + "proj.weight", transpose: true);
    }

    /// <summary>The text encoder tensors under the names of a <paramref name="family"/> checkpoint.</summary>
    public static List<Out> Text(SdWeights w, SdFamily family)
    {
        var list = new List<Out>();
        switch (family)
        {
            case SdFamily.SD1: HfTower(list, w, "l.", "cond_stage_model.transformer.text_model."); break;
            case SdFamily.SD2: OpenClipTower(list, w, "h.", "cond_stage_model.model."); break;
            default:
                HfTower(list, w, "l.", "conditioner.embedders.0.transformer.text_model.");
                OpenClipTower(list, w, "g.", "conditioner.embedders.1.model.");
                break;
        }
        return list;
    }

    // ---------------------------------------------------------------------------------------------------
    // writing
    // ---------------------------------------------------------------------------------------------------
    private const int Chunk = 1 << 24;          // elements per conversion launch: no launch grows with the tensor

    // file index j -> held index (relative to the tensor's first element)
    private static int HeldIndex(int j, int layout, int a, int b)
    {
        if (layout == 1)
        {
            int t = j % b, ci = j / b % a, co = j / (a * b);       // file [co][ci][t] <- held [co][t][ci]
            return (co * b + t) * a + ci;
        }
        if (layout == 2)
        {
            int r = j / a, c = j % a;                              // file [b][a] <- held [a][b]
            return c * b + r;
        }
        return j;
    }
    private static void ToF16Impl(Index1D i, HV src, HV dst, int srcOff, int layout, int a, int b, int at)
        => dst[at + i] = src[srcOff + HeldIndex(at + i, layout, a, b)];
    private static void ToF32Impl(Index1D i, HV src, ArrayView1D<float, Stride1D.Dense> dst, int srcOff, int layout, int a, int b, int at)
        => dst[at + i] = PrecisionConvert.ConvertToSingle(src[srcOff + HeldIndex(at + i, layout, a, b)]);
    private static void ToBf16Impl(Index1D i, HV src, ArrayView1D<BFloat16, Stride1D.Dense> dst, int srcOff, int layout, int a, int b, int at)
        => dst[at + i] = PrecisionConvert.ConvertFromSingle<BFloat16>(PrecisionConvert.ConvertToSingle(src[srcOff + HeldIndex(at + i, layout, a, b)]));

    private static string DTypeName(SdSaveDType t) => t switch { SdSaveDType.F32 => "F32", SdSaveDType.BF16 => "BF16", _ => "F16" };

    public static async Task WriteAsync(Accelerator acc, Stream target, IReadOnlyList<Out> tensors, SdSaveOptions o, bool vPrediction, CancellationToken ct)
    {
        int es = o.DType == SdSaveDType.F32 ? 4 : 2;
        string dt = DTypeName(o.DType);
        var ordered = tensors.OrderBy(t => t.Name, StringComparer.Ordinal).ToList();
        for (int i = 1; i < ordered.Count; i++)
            if (ordered[i].Name == ordered[i - 1].Name) throw new InvalidOperationException($"two tensors named '{ordered[i].Name}'");

        // header
        var json = new MemoryStream();
        long total = 0, largest = 0;
        using (var jw = new Utf8JsonWriter(json))
        {
            jw.WriteStartObject();
            if (o.Metadata is { Count: > 0 })
            {
                jw.WriteStartObject("__metadata__");
                foreach (var (k, v) in o.Metadata) jw.WriteString(k, v);
                jw.WriteEndObject();
            }
            foreach (var t in ordered)
            {
                long n = 1; foreach (var d in t.Shape) n *= d;
                if (n > int.MaxValue) throw new NotSupportedException($"'{t.Name}': {n} elements (32-bit kernel indexing)");
                jw.WriteStartObject(t.Name);
                jw.WriteString("dtype", dt);
                jw.WriteStartArray("shape"); foreach (var d in t.Shape) jw.WriteNumberValue(d); jw.WriteEndArray();
                jw.WriteStartArray("data_offsets"); jw.WriteNumberValue(total); total += n * es; jw.WriteNumberValue(total); jw.WriteEndArray();
                jw.WriteEndObject();
                largest = Math.Max(largest, n * es);
            }
            if (vPrediction)
            {
                // ComfyUI's marker: an empty float32 tensor
                jw.WriteStartObject("v_pred");
                jw.WriteString("dtype", "F32");
                jw.WriteStartArray("shape"); jw.WriteNumberValue(0); jw.WriteEndArray();
                jw.WriteStartArray("data_offsets"); jw.WriteNumberValue(total); jw.WriteNumberValue(total); jw.WriteEndArray();
                jw.WriteEndObject();
            }
            jw.WriteEndObject();
        }
        int hlen = (int)json.Length, padded = (hlen + 7) / 8 * 8;
        var head = new byte[8 + padded];
        BitConverter.TryWriteBytes(head.AsSpan(0, 8), (long)padded);
        json.GetBuffer().AsSpan(0, hlen).CopyTo(head.AsSpan(8));
        head.AsSpan(8 + hlen).Fill((byte)' ');
        await target.WriteAsync(head, ct);

        // data
        if (largest == 0) return;
        using var staging = acc.Allocate1D<uint>((largest + 3) / 4);
        var f16 = o.DType == SdSaveDType.F16 ? acc.LoadAutoGroupedStreamKernel<Index1D, HV, HV, int, int, int, int, int>(ToF16Impl) : null;
        var f32 = o.DType == SdSaveDType.F32 ? acc.LoadAutoGroupedStreamKernel<Index1D, HV, ArrayView1D<float, Stride1D.Dense>, int, int, int, int, int>(ToF32Impl) : null;
        var bf16 = o.DType == SdSaveDType.BF16 ? acc.LoadAutoGroupedStreamKernel<Index1D, HV, ArrayView1D<BFloat16, Stride1D.Dense>, int, int, int, int, int>(ToBf16Impl) : null;
        foreach (var t in ordered)
        {
            ct.ThrowIfCancellationRequested();
            int n = 1; foreach (var d in t.Shape) n *= d;
            if (n == 0) continue;
            var src = t.Store[t.Tensor];
            if (t.Offset + n > src.Length) throw new InvalidOperationException($"'{t.Name}': elements {t.Offset}..{t.Offset + n} of '{t.Tensor}' ({src.Length})");
            if (t.Offset > int.MaxValue) throw new NotSupportedException($"'{t.Name}': offset {t.Offset} (32-bit kernel indexing)");
            int off = (int)t.Offset;
            for (int at = 0; at < n; at += Chunk)
            {
                int k = Math.Min(Chunk, n - at);
                if (f16 != null) f16(k, src, staging.View.Cast<uint, global::ILGPU.Half>(), off, t.Layout, t.A, t.B, at);
                else if (f32 != null) f32(k, src, staging.View.Cast<uint, float>(), off, t.Layout, t.A, t.B, at);
                else bf16!(k, src, staging.View.Cast<uint, BFloat16>(), off, t.Layout, t.A, t.B, at);
            }
            long bytes = (long)n * es;
            if (bytes % 4 == 0)
            {
                await staging.View.SubView(0, bytes / 4).CopyToStreamAsync(target, cancellationToken: ct);
            }
            else
            {
                // an odd-length fp16 / bf16 tensor (the VAE's 3-channel conv_out bias): a browser readback moves whole 4-byte words, so read the padded word span
                // (a few bytes; bulk tensors never take this path) and write the tensor's own bytes
                if (bytes > 4096) throw new NotSupportedException($"'{t.Name}': {bytes} bytes is not a multiple of 4");
                var words = await staging.View.SubView(0, (bytes + 3) / 4).CopyToHostAsync();
                var raw = new byte[words.Length * 4];
                Buffer.BlockCopy(words, 0, raw, 0, raw.Length);
                await target.WriteAsync(raw.AsMemory(0, (int)bytes), ct);
            }
        }
    }
}
