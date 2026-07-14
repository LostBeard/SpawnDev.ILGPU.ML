using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Kernels;

/// <summary>
/// LFM2 "double-gated short convolution" (the LIV conv mixer), fused into one pass.
///
/// Given the in_proj output BCx (split into three equal H-wide chunks B, C, x - in that order, verified
/// against llama.cpp src/models/lfm2.cpp), this computes:
///     Bx[t,c]  = B[t,c] * x[t,c]
///     conv[t,c]= sum_{k=0..L-1} W[c,k] * Bx[t-(L-1)+k, c]     (causal depthwise, Bx[&lt;0]=0)
///     y[t,c]   = C[t,c] * conv[t,c]
/// No activation is applied anywhere in the shortconv path.
///
/// Layouts (length-major, matching the graph's [1,seq,*] activations):
///   BCx   : [seq, 3H]  - element (t, j) at t*3H + j.  B=(t,c), C=(t,H+c), x=(t,2H+c).
///   weight: [H, L]     - W[c,k] at c*L + k  (GGUF shortconv.conv.weight, ggml ne=[L,H] -> our [H,L]).
///   y     : [seq, H]   - (t,c) at t*H + c.
///
/// One thread per output (t,c); no shared memory / atomics / barriers, so it compiles on every backend
/// (same shape as Conv1DKernel's flat kernel). Decode-time conv-state (prepending the previous L-1 Bx
/// values across a step boundary) is layered on separately; a full-sequence forward needs no external state.
/// </summary>
public sealed class ShortConvKernel : IDisposable
{
    private readonly Accelerator _accelerator;
    private Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>,   // bcx   [seq*3H]
        ArrayView1D<float, Stride1D.Dense>,   // weight[H*L]
        ArrayView1D<float, Stride1D.Dense>,   // y     [seq*H]
        ArrayView1D<int, Stride1D.Dense>>? _kernel;   // params [seq, H, L]

    private MemoryBuffer1D<int, Stride1D.Dense>? _paramsBuf;
    private readonly List<MemoryBuffer1D<int, Stride1D.Dense>> _oldParamsBufs = new();

    public ShortConvKernel(Accelerator accelerator) => _accelerator = accelerator;

    /// <summary>Run the fused shortconv. <paramref name="bcx"/> is the in_proj output [seq,3H];
    /// <paramref name="weight"/> is [H,L]; <paramref name="y"/> receives [seq,H].</summary>
    public void Forward(
        ArrayView1D<float, Stride1D.Dense> bcx,
        ArrayView1D<float, Stride1D.Dense> weight,
        ArrayView1D<float, Stride1D.Dense> y,
        int seq, int H, int L)
    {
        _kernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>>(ShortConvImpl);

        var p = new[] { seq, H, L };
        if (_paramsBuf != null) _oldParamsBufs.Add(_paramsBuf);
        _paramsBuf = _accelerator.Allocate1D(p);

        _kernel(seq * H, bcx, weight, y, _paramsBuf.View);
    }

    private static void ShortConvImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> bcx,
        ArrayView1D<float, Stride1D.Dense> weight,
        ArrayView1D<float, Stride1D.Dense> y,
        ArrayView1D<int, Stride1D.Dense> p)
    {
        int seq = p[0], H = p[1], L = p[2];
        int t = idx / H;
        int c = idx % H;

        int wBase = c * L;
        float acc = 0f;
        // Causal depthwise conv over Bx = B*x; taps align so ox=t reads inputs t-(L-1)..t.
        for (int k = 0; k < L; k++)
        {
            int tt = t - (L - 1) + k;
            if (tt >= 0)
            {
                int rowBase = tt * 3 * H;
                float b = bcx[rowBase + c];
                float x = bcx[rowBase + 2 * H + c];
                acc += weight[wBase + k] * (b * x);
            }
        }
        float cGate = bcx[t * 3 * H + H + c];
        y[idx] = cGate * acc;
    }

    public void Dispose()
    {
        _paramsBuf?.Dispose(); _paramsBuf = null;
        foreach (var b in _oldParamsBufs) b.Dispose();
        _oldParamsBufs.Clear();
    }
}
