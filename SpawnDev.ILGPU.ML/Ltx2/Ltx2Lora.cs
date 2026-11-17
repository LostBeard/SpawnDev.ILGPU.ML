using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.Trellis2;

namespace SpawnDev.ILGPU.ML.Ltx2;

/// <summary>
/// A LoRA for the LTX-2 transformer in the PEFT layout LTX-2's LoRAs ship in (<c>diffusion_model.&lt;module&gt;.lora_A.weight</c> [r, in], <c>.lora_B.weight</c> [out, r], no
/// alpha: the update is <c>strength (B A)</c>, as ComfyUI applies a LoRA without an alpha). Applied without merging: <see cref="Ltx2Transformer"/> adds
/// <c>strength ((x A^T) B^T)</c> to each patched Linear's output, which equals ComfyUI's on-the-fly patch <c>x (W + strength B A)^T</c> up to rounding order and keeps the
/// base weights exact and unduplicated. The per-block tensors stream with their block (<see cref="Weights"/>); the top-level ones stay resident.
/// </summary>
public sealed class Ltx2Lora : IDisposable
{
    /// <summary>The LoRA's tensors (prefix-free names: <c>transformer_blocks.N.attn1.to_q.lora_A.weight</c>, ...), blocks as the transformer's.</summary>
    public StreamedModelWeights Weights { get; }

    /// <summary>The multiplier of the update (ComfyUI's strength_model).</summary>
    public float Strength { get; set; }

    /// <summary>The LoRA's rank (the first dimension of its first lora_A).</summary>
    public int Rank { get; }

    private Ltx2Lora(StreamedModelWeights w, float strength, int rank) { Weights = w; Strength = strength; Rank = rank; }

    /// <summary>
    /// Open a LoRA file for the LTX-2 transformer. <paramref name="residentBytes"/>: per-block LoRA tensors kept on the device (as stored); the rest stream with their block.
    /// Tensors outside the transformer (the text projection's <c>text_embedding_projection.*</c>) are not loaded: ComfyUI's LoraLoaderModelOnly patches the model only.
    /// </summary>
    public static async Task<Ltx2Lora> OpenAsync(Accelerator accelerator, IModelSource source, string repo, string file, float strength, long residentBytes, CancellationToken ct = default)
        => await OpenAsync(accelerator, source, repo, file, strength, residentBytes, null, ct);

    /// <summary>
    /// The same with <paramref name="include"/> (prefix-free tensor name -> true to apply): a LoRA on part of the model (tests apply the subset a reference measured).
    /// </summary>
    public static async Task<Ltx2Lora> OpenAsync(Accelerator accelerator, IModelSource source, string repo, string file, float strength, long residentBytes, Func<string, bool>? include,
        CancellationToken ct = default)
    {
        int BlockOf(string n)
        {
            if (include != null && !include(n)) return StreamedModelWeights.HostOnly;
            if (n.StartsWith("transformer_blocks.", StringComparison.Ordinal)) { int dot = n.IndexOf('.', 19); return int.Parse(n.AsSpan(19, dot - 19)); }
            return -1;
        }
        // the anchor (which finds the file's prefix) is any lora_A tensor: a LoRA may cover only some blocks, or none of block 0
        var anchor = await AnchorAsync(source, repo, file, ct);
        var w = await StreamedModelWeights.OpenAsync(accelerator, source, repo, file, anchor, BlockOf, Ltx2Transformer.Layers, residentBytes, null, ct);
        try
        {
            foreach (var n in w.HeaderNames)
                if (n.EndsWith(".alpha", StringComparison.Ordinal) || (!n.EndsWith(".lora_A.weight", StringComparison.Ordinal) && !n.EndsWith(".lora_B.weight", StringComparison.Ordinal)))
                    throw new NotSupportedException($"'{file}': {n} is not a PEFT lora_A / lora_B weight (alpha-scaled or other LoRA layouts are not handled here)");
            int rank = w.Base.Names.Where(n => n.EndsWith(".lora_A.weight", StringComparison.Ordinal)).Select(n => w.Base.Shape(n)[0]).DefaultIfEmpty(0).Max();
            for (int k = 0; k < Ltx2Transformer.Layers; k++)                                  // and the first applied block's (a few top-level layers use smaller ranks)
            {
                string bp = $"transformer_blocks.{k}.";
                if (!w.HeaderNames.Any(n => n.StartsWith(bp, StringComparison.Ordinal) && n.EndsWith(".lora_A.weight", StringComparison.Ordinal) && (include == null || include(n)))) continue;
                var (bk, streamed) = await w.BlockAsync(k, ct);
                rank = Math.Max(rank, bk.Names.Where(n => n.EndsWith(".lora_A.weight", StringComparison.Ordinal)).Select(n => bk.Shape(n)[0]).DefaultIfEmpty(0).Max());
                if (streamed) { await accelerator.SynchronizeAsync(); bk.Dispose(); }
                break;
            }
            return new Ltx2Lora(w, strength, rank);
        }
        catch { w.Dispose(); throw; }
    }

    // a prefix-free lora_A name from the file's header: from "transformer_blocks." on, else what follows "diffusion_model.", else the name itself
    private static async Task<string> AnchorAsync(IModelSource source, string repo, string file, CancellationToken ct)
    {
        await using var s = await source.OpenAsync(repo, file, ct);
        var lenBytes = new byte[8]; await s.ReadExactlyAsync(lenBytes, ct);
        long hl = BitConverter.ToInt64(lenBytes);
        if (hl <= 0 || hl > 100_000_000) throw new InvalidDataException($"'{file}': invalid SafeTensors header size {hl}");
        var header = new byte[hl]; await s.ReadExactlyAsync(header, ct);
        using var doc = System.Text.Json.JsonDocument.Parse(header);
        string? any = null;
        foreach (var t in doc.RootElement.EnumerateObject())
        {
            if (!t.Name.EndsWith(".lora_A.weight", StringComparison.Ordinal)) continue;
            int b = t.Name.IndexOf("transformer_blocks.", StringComparison.Ordinal);
            if (b >= 0) return t.Name[b..];
            any ??= t.Name;
        }
        if (any == null) throw new InvalidDataException($"'{file}' holds no lora_A tensor: not a PEFT LoRA");
        int d = any.IndexOf("diffusion_model.", StringComparison.Ordinal);
        return d >= 0 ? any[(d + "diffusion_model.".Length)..] : any;
    }

    public void Dispose() => Weights.Dispose();
}
