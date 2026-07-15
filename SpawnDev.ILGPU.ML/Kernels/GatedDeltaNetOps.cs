using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Kernels;

/// <summary>
/// The elementwise/reduction helper kernels around the Qwen3-Next Gated DeltaNet scan
/// (<see cref="GatedDeltaNetKernel"/>): the causal depthwise conv+SiLU, per-head L2-norm, and the gated
/// RMSNorm. Kept small and separate so each can be bit-verified against the numpy reference independently.
/// </summary>
public sealed class GatedDeltaNetOps : IDisposable
{
    private readonly Accelerator _accelerator;

    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>>? _conv;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>>? _l2;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>>? _zero;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>>? _grms;

    private MemoryBuffer1D<int, Stride1D.Dense>? _cp, _lp, _gp;
    private readonly List<IDisposable> _old = new();

    public GatedDeltaNetOps(Accelerator accelerator) => _accelerator = accelerator;

    // ── Causal depthwise conv (kernel L) + SiLU over the token axis, per channel of the CONCATENATED [q|k|v],
    //    fusing the split: the C=qDim+kDim+vDim channels of `qkv` are conv'd, then routed to the q/k/v outputs.
    //    qkv: [seq, C]; weight: [C, L] (ne=[L,C] → w[c*L+j]); no bias (qwen35). out[t,c]=silu(Σ_j w[c*L+j]*qkv[t-(L-1)+j,c]).
    /// <summary>Causal depthwise conv1d (kernel L) + SiLU over [q|k|v]'s C=qDim+kDim+vDim channels, split to the
    /// three contiguous outputs. <paramref name="weight"/> is [C,L] (GGUF ne=[L,C]). Zero-pads at seq start.</summary>
    public void CausalConvSiluSplit(ArrayView1D<float, Stride1D.Dense> qkv, ArrayView1D<float, Stride1D.Dense> weight,
        ArrayView1D<float, Stride1D.Dense> q, ArrayView1D<float, Stride1D.Dense> k, ArrayView1D<float, Stride1D.Dense> v,
        int seq, int qDim, int kDim, int vDim, int L)
    {
        _conv ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>>(ConvImpl);
        int C = qDim + kDim + vDim;
        var p = new[] { seq, C, L, qDim, kDim };
        if (_cp != null) _old.Add(_cp);
        _cp = _accelerator.Allocate1D(p);
        _conv(seq * C, qkv, weight, q, k, v, _cp.View);
    }

    private static void ConvImpl(Index1D idx, ArrayView1D<float, Stride1D.Dense> qkv,
        ArrayView1D<float, Stride1D.Dense> weight, ArrayView1D<float, Stride1D.Dense> q,
        ArrayView1D<float, Stride1D.Dense> k, ArrayView1D<float, Stride1D.Dense> v,
        ArrayView1D<int, Stride1D.Dense> p)
    {
        int seq = p[0], C = p[1], L = p[2], qDim = p[3], kDim = p[4];
        int vDim = C - qDim - kDim;
        int t = idx / C, c = idx % C;
        int wBase = c * L;
        float acc = 0f;
        for (int j = 0; j < L; j++)
        {
            int tt = t - (L - 1) + j;
            if (tt >= 0) acc += weight[wBase + j] * qkv[tt * C + c];
        }
        float silu = acc / (1f + MathF.Exp(-acc));   // acc * sigmoid(acc)
        // Route to the contiguous q / k / v output (channels [0,qDim) / [qDim,qDim+kDim) / rest).
        if (c < qDim) q[t * qDim + c] = silu;
        else if (c < qDim + kDim) k[t * kDim + (c - qDim)] = silu;
        else v[t * vDim + (c - qDim - kDim)] = silu;
    }

    /// <summary>Zero the first <paramref name="n"/> elements of <paramref name="v"/> (recurrent-state reset for
    /// a fresh prefill; browser-safe write-index==thread-index fill).</summary>
    public void Zero(ArrayView1D<float, Stride1D.Dense> v, int n)
    {
        _zero ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>>(
            (Index1D i, ArrayView1D<float, Stride1D.Dense> x) => x[i] = 0f);
        _zero(n, v);
    }

    // ── Per-head L2 normalize IN PLACE: x[seq, H, D] → each length-D head vector scaled by 1/sqrt(sum(sq)+eps).
    /// <summary>L2-normalize each length-<paramref name="D"/> head vector in place (eps 1e-6). Layout [seq,H,D].</summary>
    public void L2NormHeads(ArrayView1D<float, Stride1D.Dense> x, int rows, int D)
    {
        _l2 ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>>(L2Impl);
        var p = new[] { D };
        if (_lp != null) _old.Add(_lp);
        _lp = _accelerator.Allocate1D(p);
        _l2(rows, x, _lp.View);   // one thread per (seq*H) head vector
    }

    private static void L2Impl(Index1D idx, ArrayView1D<float, Stride1D.Dense> x, ArrayView1D<int, Stride1D.Dense> p)
    {
        int D = p[0];
        int baseI = idx * D;
        float s2 = 0f;
        for (int d = 0; d < D; d++) { float val = x[baseI + d]; s2 += val * val; }
        float inv = 1f / MathF.Sqrt(s2 + 1e-6f);
        for (int d = 0; d < D; d++) x[baseI + d] = x[baseI + d] * inv;
    }

    // ── Gated RMSNorm: out[t, hv*D + d] = rmsnorm(scan[t,hv,:], normW) * silu(z[t, hv*D + d]).
    //    scan: [seq, numVHeads, D]; normW: [D]; z: [seq, numVHeads*D]; out: [seq, numVHeads*D]. One thread/head-vec.
    /// <summary>Gated RMSNorm per value-head: rmsnorm(scan_head, <paramref name="normW"/>[D], eps 1e-6) elementwise-×
    /// silu(<paramref name="z"/>_head). Produces [seq, numVHeads*D].</summary>
    public void GatedRmsNorm(ArrayView1D<float, Stride1D.Dense> scan, ArrayView1D<float, Stride1D.Dense> normW,
        ArrayView1D<float, Stride1D.Dense> z, ArrayView1D<float, Stride1D.Dense> outp, int rows, int D)
    {
        _grms ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>>(GrmsImpl);
        var p = new[] { D };
        if (_gp != null) _old.Add(_gp);
        _gp = _accelerator.Allocate1D(p);
        _grms(rows, scan, normW, z, outp, _gp.View);   // one thread per (seq*numVHeads) head vector
    }

    private static void GrmsImpl(Index1D idx, ArrayView1D<float, Stride1D.Dense> scan,
        ArrayView1D<float, Stride1D.Dense> normW, ArrayView1D<float, Stride1D.Dense> z,
        ArrayView1D<float, Stride1D.Dense> outp, ArrayView1D<int, Stride1D.Dense> p)
    {
        int D = p[0];
        int baseI = idx * D;
        float s2 = 0f;
        for (int d = 0; d < D; d++) { float val = scan[baseI + d]; s2 += val * val; }
        float inv = 1f / MathF.Sqrt(s2 / D + 1e-6f);
        for (int d = 0; d < D; d++)
        {
            float normed = scan[baseI + d] * inv * normW[d];
            float g = z[baseI + d];
            float silu = g / (1f + MathF.Exp(-g));
            outp[baseI + d] = normed * silu;
        }
    }

    public void Dispose()
    {
        _cp?.Dispose(); _lp?.Dispose(); _gp?.Dispose();
        _cp = _lp = _gp = null;
        foreach (var d in _old) d.Dispose();
        _old.Clear();
    }
}
