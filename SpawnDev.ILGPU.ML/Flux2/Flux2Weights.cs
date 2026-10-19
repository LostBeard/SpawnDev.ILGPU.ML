using ILGPU;
using ILGPU.Runtime;
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

    /// <summary>Wrap a loaded stage. Disposing this disposes <paramref name="weights"/> too.</summary>
    public Flux2Weights(Accelerator accelerator, GpuWeights weights)
    {
        _w = weights; _f = new Flux2Kernels(accelerator); _u = new SkinTokensKernels(accelerator);
        long most = 0;
        foreach (var name in weights.Names)
            if (weights.IsU32(name)) { var s = weights.Shape(name); most = Math.Max(most, (long)s[0] * (s[1] / 5 * 32)); }
        if (most > 0) _scratch = accelerator.Allocate1D<uint>(most / 2);
    }

    public bool IsQuantized(string name) => _w.IsU32(name);
    /// <summary>Device bytes held (weights plus the dequantisation scratch).</summary>
    public long Bytes => _w.Bytes + (_scratch?.Length ?? 0) * 4;

    /// <summary>A weight matrix [N, K] as bf16: the stored tensor itself, or its dequantisation in the shared scratch (valid until the next call).</summary>
    public BV Matrix(string name)
    {
        if (!_w.IsU32(name)) return _w.BF16(name);
        var s = _w.Shape(name);
        int n = s[0], k = s[1] / 5 * 32;
        _f.DequantQ4(_w.U32(name), _scratch!.View, n, k);
        return _scratch.View.SubView(0, (long)n * k / 2).Cast<uint, BFloat16>();
    }

    /// <summary>A small bf16 tensor (norm weight, bias).</summary>
    public BV Vector(string name) => _w.BF16(name);

    /// <summary>o[t, :] = row ids[t] of the embedding table <paramref name="name"/> ([vocab, C]), fp32.</summary>
    public void Embed(IV ids, string name, FV o, int count, int C)
    {
        if (_w.IsU32(name)) _f.DequantRows(_w.U32(name), ids, o, count, C);
        else _u.Embed(ids, _w.BF16(name), o, count, C);
    }

    public void Dispose()
    {
        _scratch?.Dispose(); _scratch = null;
        _w.Dispose();
    }
}
