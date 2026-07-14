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

    // State-aware variant (KV-decode): reads the previous (L-1) tokens' bcx from a persistent state
    // buffer for the taps that fall before the current chunk (tt<0), instead of zero-padding.
    private Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>,   // bcx   [seq*3H]
        ArrayView1D<float, Stride1D.Dense>,   // weight[H*L]
        ArrayView1D<float, Stride1D.Dense>,   // y     [seq*H]
        ArrayView1D<float, Stride1D.Dense>,   // state [(L-1)*3H]  (prev tokens' bcx; ignored when stateRows=0)
        ArrayView1D<int, Stride1D.Dense>>? _stateKernel;   // params [seq, H, L, stateRows]
    private MemoryBuffer1D<int, Stride1D.Dense>? _stateParamsBuf;
    private readonly List<MemoryBuffer1D<int, Stride1D.Dense>> _oldStateParamsBufs = new();

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

    /// <summary>Run the fused shortconv with a persistent conv-STATE buffer (KV-decode). For output taps
    /// that fall before the current chunk (tt&lt;0) the kernel reads from <paramref name="state"/> — the
    /// previous <paramref name="stateRows"/> tokens' bcx laid out [stateRows,3H] — instead of zero-padding.
    /// Pass <paramref name="stateRows"/>=0 (state ignored) for a fresh sequence start (zero-pad, identical
    /// to <see cref="Forward"/>). Produces exactly <paramref name="seq"/> outputs.</summary>
    public void ForwardWithState(
        ArrayView1D<float, Stride1D.Dense> bcx,
        ArrayView1D<float, Stride1D.Dense> weight,
        ArrayView1D<float, Stride1D.Dense> y,
        ArrayView1D<float, Stride1D.Dense> state,
        int seq, int H, int L, int stateRows)
    {
        _stateKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>>(ShortConvStateImpl);

        var p = new[] { seq, H, L, stateRows };
        if (_stateParamsBuf != null) _oldStateParamsBufs.Add(_stateParamsBuf);
        _stateParamsBuf = _accelerator.Allocate1D(p);

        _stateKernel(seq * H, bcx, weight, y, state, _stateParamsBuf.View);
    }

    private static void ShortConvStateImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> bcx,
        ArrayView1D<float, Stride1D.Dense> weight,
        ArrayView1D<float, Stride1D.Dense> y,
        ArrayView1D<float, Stride1D.Dense> state,
        ArrayView1D<int, Stride1D.Dense> p)
    {
        int seq = p[0], H = p[1], L = p[2], stateRows = p[3];
        int t = idx / H;
        int c = idx % H;

        int wBase = c * L;
        float acc = 0f;
        // Causal depthwise conv over Bx = B*x. tap k reads token tt = t-(L-1)+k. For tt<0 (before the
        // current chunk) read the previous tokens from `state` at row (tt+stateRows); if that's still <0
        // (no state / not enough history) it's a true sequence-start → zero-pad.
        for (int k = 0; k < L; k++)
        {
            int tt = t - (L - 1) + k;
            float b, x;
            if (tt >= 0)
            {
                int rowBase = tt * 3 * H;
                b = bcx[rowBase + c];
                x = bcx[rowBase + 2 * H + c];
            }
            else
            {
                int st = tt + stateRows;   // tt in [-(L-1),-1] → st in [stateRows-(L-1), stateRows-1]
                if (st >= 0)
                {
                    int rowBase = st * 3 * H;
                    b = state[rowBase + c];
                    x = state[rowBase + 2 * H + c];
                }
                else { b = 0f; x = 0f; }   // zero-pad (fresh sequence, no prior history)
            }
            acc += weight[wBase + k] * (b * x);
        }
        float cGate = bcx[t * 3 * H + H + c];   // gate C is always the CURRENT token's chunk-1
        y[idx] = cGate * acc;
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
        _stateParamsBuf?.Dispose(); _stateParamsBuf = null;
        foreach (var b in _oldStateParamsBufs) b.Dispose();
        _oldStateParamsBufs.Clear();
    }
}
