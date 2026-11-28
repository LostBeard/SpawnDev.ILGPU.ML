using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Trellis2;

namespace SpawnDev.ILGPU.ML.Anima;

/// <summary>
/// A LoRA for <see cref="AnimaDiT"/> in the PEFT layout Anima's LoRAs ship in (e.g. circlestone-labs anima-turbo-lora-v0.2): <c>diffusion_model.&lt;module&gt;.lora_A.weight</c>
/// [r, in], <c>.lora_B.weight</c> [out, r], no alpha. The update is <c>strength (B A)</c> in full precision, applied WITHOUT merging: each patched Linear's output gets
/// <c>strength ((x A^T) B^T)</c> (before its activation), which equals ComfyUI's patched weight <c>x (W + strength B A)^T</c> up to rounding order and leaves the
/// bf16 base weights untouched (the strength can change between calls; several LoRAs add up).
/// <para>
/// MEASURED against ComfyUI's LoraLoaderModelOnly (SpawnGen's run 10, float64 reference tools/anima-ref/check_lora.py): strength 1.0 / 0.5 x B A, kept exact,
/// matches the denoised output to 8.2e-6 / 2.2e-6; the same merge re-rounded to bf16 misses by 0.21 (the update is mostly below a bf16 step of W), strength 0.5 on
/// the 1.0 run by 0.60.
/// </para>
/// Only the DiT blocks are patched (<c>blocks.N.*</c>: attention q / k / v / output, the MLP, the adaLN-LoRA modulation pairs). The file's
/// <c>llm_adapter.blocks.*</c> tensors are not applied: ComfyUI's adapter output is the same with and without the LoRA (SpawnGen's run 10), and in
/// anima-turbo-lora-v0.2 every adapter lora_B is zero, so applying them would change nothing either.
/// </summary>
public sealed class AnimaLora : IDisposable
{
    private readonly Dictionary<string, (string A, string B, int Rank)> _mods = new(StringComparer.Ordinal);
    private readonly bool _ownsWeights;

    /// <summary>The LoRA file's tensors on the device.</summary>
    public GpuWeights Weights { get; }
    internal Flux2.Flux2Weights Mat { get; }
    /// <summary>The multiplier of the update (ComfyUI's strength_model). Read at every call.</summary>
    public float Strength { get; set; }
    /// <summary>The largest rank in the file.</summary>
    public int Rank { get; }
    /// <summary>Patched DiT Linears.</summary>
    public int Modules => _mods.Count;

    /// <summary>A LoRA over its weights already on the device.</summary>
    /// <param name="accelerator">the application's accelerator (never disposed here)</param>
    /// <param name="weights">the LoRA file (any prefix before <c>blocks.</c>, e.g. <c>diffusion_model.</c>)</param>
    /// <param name="strength">the multiplier of the update</param>
    /// <param name="ownsWeights">dispose <paramref name="weights"/> with this LoRA</param>
    public AnimaLora(Accelerator accelerator, GpuWeights weights, float strength = 1f, bool ownsWeights = true)
    {
        Weights = weights; Strength = strength; _ownsWeights = ownsWeights;
        Mat = new Flux2.Flux2Weights(accelerator, weights, ownsWeights: false);
        const string tailA = ".lora_A.weight", tailB = ".lora_B.weight";
        foreach (var n in weights.Names)
        {
            if (!n.EndsWith(tailA, StringComparison.Ordinal) && !n.EndsWith(tailB, StringComparison.Ordinal))
                throw new NotSupportedException($"AnimaLora: {n} is not a PEFT lora_A / lora_B weight (alpha-scaled or other LoRA layouts are not handled)");
            if (!n.EndsWith(tailA, StringComparison.Ordinal)) continue;
            string m = n[..^tailA.Length];
            int dot = m.StartsWith("blocks.", StringComparison.Ordinal) ? -1 : m.IndexOf(".blocks.", StringComparison.Ordinal);
            if ((dot < 0 && !m.StartsWith("blocks.", StringComparison.Ordinal)) || m.Contains("llm_adapter.", StringComparison.Ordinal)) continue;   // not a DiT block Linear (the adapter's: see the summary)
            int at = dot + 1;
            string key = m[at..];
            string bName = m + tailB;
            if (!weights.Has(bName)) throw new InvalidDataException($"AnimaLora: {n} has no {bName}");
            int r = weights.Shape(n)[0];
            if (weights.Shape(bName)[1] != r) throw new InvalidDataException($"AnimaLora: {m}: A rank {r}, B [{string.Join(", ", weights.Shape(bName))}]");
            _mods[key] = (n, bName, r);
            Rank = Math.Max(Rank, r);
        }
        if (_mods.Count == 0) throw new InvalidDataException("AnimaLora: no blocks.* lora_A / lora_B pairs in the weights");
    }

    /// <summary>The A / B tensor names and rank for the DiT Linear <paramref name="module"/> (<c>blocks.N.self_attn.q_proj</c>, without <c>net.</c> or <c>.weight</c>).</summary>
    internal bool TryGet(string module, out (string A, string B, int Rank) m) => _mods.TryGetValue(module, out m);

    /// <summary>Frees the weights when owned (drain the queue first in a browser).</summary>
    public void Dispose()
    {
        Mat.Dispose();
        if (_ownsWeights) Weights.Dispose();
    }
}
