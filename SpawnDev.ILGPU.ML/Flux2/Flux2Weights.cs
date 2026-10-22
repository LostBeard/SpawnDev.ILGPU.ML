using System.Text.Json;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.SkinTokens;
using SpawnDev.ILGPU.ML.Trellis2;

namespace SpawnDev.ILGPU.ML.Flux2;

using FV = ArrayView1D<float, Stride1D.Dense>;
using IV = ArrayView1D<int, Stride1D.Dense>;
using BV = ArrayView1D<BFloat16, Stride1D.Dense>;

/// <summary>
/// The weights of one FLUX.2 stage, as the encoder and the transformer read them: bf16 tensors straight from the device, and matrices stored as 4-bit blocks
/// (quantize_flux2.py: 5 bits per weight, 2.5 GB for the 3.9B transformer instead of 7.75) dequantised one matrix at a time into a bf16 scratch just before its
/// GEMM, so the normal tensor-core / tiled GEMM runs on them unchanged. <see cref="Matrix"/> returns a view that is only valid until the next call (it may be the shared scratch).
/// </summary>
public sealed class Flux2Weights : IDisposable
{
    private readonly GpuWeights _w;
    private readonly Flux2Kernels _f;
    private readonly SkinTokensKernels _u;
    private MemoryBuffer1D<uint, Stride1D.Dense>? _scratch;
    private readonly int _wordsPerBlock;                        // 5 = 4-bit blocks, 9 = 8-bit blocks (the file's metadata "quant" says which)

    /// <summary>4 or 8: the bits per quantised weight of this stage (0 when it holds no quantised matrix).</summary>
    public int QuantBits { get; }

    private readonly bool _owns;

    /// <summary>
    /// Wrap a loaded stage. Disposing this disposes <paramref name="weights"/> too, unless <paramref name="ownsWeights"/> is false (a model that keeps the <see cref="GpuWeights"/> itself and only
    /// borrows the block dequantisation: the Wan 2.2 DiT reads bf16, 8-bit and 4-bit exports through it).
    /// </summary>
    public Flux2Weights(Accelerator accelerator, GpuWeights weights, bool ownsWeights = true)
    {
        _owns = ownsWeights;
        _w = weights; _f = new Flux2Kernels(accelerator); _u = new SkinTokensKernels(accelerator);
        weights.Metadata.TryGetValue("quant", out var quant);
        QuantBits = quant != null && quant.StartsWith("q8", StringComparison.Ordinal) ? 8 : 4;
        _wordsPerBlock = QuantBits == 8 ? 9 : 5;
        long most = 0;
        foreach (var name in weights.Names)
            if (weights.IsU32(name)) { var s = weights.Shape(name); most = Math.Max(most, (long)s[0] * (s[1] / _wordsPerBlock * 32)); }
        if (most > 0) _scratch = accelerator.Allocate1D<uint>(most / 2);
    }

    /// <summary>Stream one stage's safetensors file from a model source (browser OPFS / hub, or HttpClient + a file store on desktop) onto the device.</summary>
    public static async Task<Flux2Weights> LoadAsync(Accelerator accelerator, IModelSource source, string repo, string file, CancellationToken cancellationToken = default)
    {
        await using var stream = await source.OpenAsync(repo, file, cancellationToken);
        using var empty = JsonDocument.Parse("{}");
        return new Flux2Weights(accelerator, await GpuWeights.LoadAsync(accelerator, stream, empty.RootElement, $"{repo}/{file}", cancellationToken: cancellationToken));
    }

    public bool IsQuantized(string name) => _w.IsU32(name);
    /// <summary>Device bytes held (weights plus the dequantisation scratch).</summary>
    public long Bytes => _w.Bytes + (_scratch?.Length ?? 0) * 4;

    /// <summary>A weight matrix [N, K] as bf16: the stored tensor itself, or its dequantisation in the shared scratch (valid until the next call).</summary>
    public BV Matrix(string name)
    {
        if (!_w.IsU32(name)) return _w.BF16(name);
        var s = _w.Shape(name);
        int n = s[0], k = s[1] / _wordsPerBlock * 32;
        if (QuantBits == 8) _f.DequantQ8(_w.U32(name), _scratch!.View, n, k); else _f.DequantQ4(_w.U32(name), _scratch!.View, n, k);
        return _scratch.View.SubView(0, (long)n * k / 2).Cast<uint, BFloat16>();
    }

    /// <summary>A small bf16 tensor (norm weight, bias).</summary>
    public BV Vector(string name) => _w.BF16(name);

    /// <summary>o[t, :] = row ids[t] of the embedding table <paramref name="name"/> ([vocab, C]), fp32.</summary>
    public void Embed(IV ids, string name, FV o, int count, int C)
    {
        if (_w.IsU32(name)) { if (QuantBits == 8) _f.DequantRows8(_w.U32(name), ids, o, count, C); else _f.DequantRows(_w.U32(name), ids, o, count, C); }
        else _u.Embed(ids, _w.BF16(name), o, count, C);
    }

    public void Dispose()
    {
        _scratch?.Dispose(); _scratch = null;
        if (_owns) _w.Dispose();
    }
}
