using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Trellis2;

using FV = ArrayView1D<float, Stride1D.Dense>;

/// <summary>
/// TRELLIS.2 sparse-structure flow (trellis2/models/sparse_structure_flow.py): the dense latent [inCh, R, R, R]
/// (R = 16, 8 channels) as R^3 tokens through <see cref="FlowDiT"/>, with the reference's channel-first layout at the
/// boundary. Batch 1; classifier-free guidance runs the model twice (the sampler does that).
/// </summary>
public sealed class SparseStructureFlow : IDisposable
{
    private readonly Accelerator _acc;
    private readonly DiTKernels _k;
    private readonly FlowDiT _dit;
    private readonly int _res;
    private MemoryBuffer1D<float, Stride1D.Dense>? _tokIn, _tokOut;

    /// <summary>Latent resolution per axis (16).</summary>
    public int Resolution => _res;
    public int InChannels => _dit.InChannels;
    public int OutChannels => _dit.OutChannels;
    public int CondChannels => _dit.CondChannels;

    /// <summary>Test hook, see <see cref="FlowDiT.Tap"/>.</summary>
    public Action<string, FV, int>? Tap { get => _dit.Tap; set => _dit.Tap = value; }

    public SparseStructureFlow(Accelerator accelerator, DiTKernels kernels, GpuWeights weights)
    {
        _acc = accelerator; _k = kernels;
        _res = weights.Args.GetProperty("resolution").GetInt32();
        _dit = new FlowDiT(accelerator, kernels, weights);
        int n = _res * _res * _res;
        using var coords = accelerator.Allocate1D<int>(n * 3);
        _k.GridCoords(coords.View, _res);
        _dit.SetTokens(coords.View, n);
        _tokIn = accelerator.Allocate1D<float>((long)n * _dit.InChannels);
        _tokOut = accelerator.Allocate1D<float>((long)n * _dit.OutChannels);
    }

    /// <summary>
    /// One velocity prediction. <paramref name="x"/> is [inCh, R^3] (the reference's [1, C, R, R, R]),
    /// <paramref name="cond"/> is [condTokens, condCh]; <paramref name="o"/> receives [outCh, R^3].
    /// </summary>
    public void Forward(FV x, float t, FV cond, int condTokens, FV o)
    {
        int n = _res * _res * _res;
        _k.Transpose(x, _tokIn!.View, _dit.InChannels, n);                  // [C, N] -> [N, C]
        _dit.Forward(_tokIn.View, t, cond, condTokens, _tokOut!.View);
        _k.Transpose(_tokOut.View, o, n, _dit.OutChannels);                 // [N, C] -> [C, N]
        _acc.Fence();
    }

    public void Dispose()
    {
        _dit.Dispose();
        _tokIn?.Dispose(); _tokOut?.Dispose();
    }
}
