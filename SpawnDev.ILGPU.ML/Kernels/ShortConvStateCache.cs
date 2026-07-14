using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Kernels;

/// <summary>
/// Per-conv-layer state cache for LFM2 (and any short-conv/SSM mixer) incremental KV-decode.
///
/// A short-conv layer's causal depthwise conv needs the previous <c>L-1</c> tokens' <c>bcx</c> (the in_proj
/// output) to compute the current token. During full-recompute the whole sequence is present so the conv
/// zero-pads only at the true sequence start. During KV-decode each step feeds ONLY the new token(s), so the
/// conv would zero-pad every step and lose all history → wrong output (the model diverges from full-recompute
/// after the prefill). This cache holds the last <c>L-1</c> <c>bcx</c> rows per conv-layer and feeds them to
/// <see cref="ShortConvKernel.ForwardWithState"/> so decode-step conv sees the real history.
///
/// Analogous to <see cref="GGUFDecodeKVCache"/> for attention. Prefill (pastLen==0) runs the plain zero-pad
/// conv and snapshots the tail as the initial state; decode (pastLen&gt;0) prepends the state. Ping-pong state
/// buffers keep the update browser-safe (the kernel reads the ACTIVE buffer while the update writes the OTHER,
/// then they swap — never an in-place overlapping copy, never a method-local buffer feeding a pending dispatch).
/// </summary>
public sealed class ShortConvStateCache : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly ShortConvKernel _kernel;

    private sealed class LayerState
    {
        public int RowWidth;      // 3H
        public int StateRows;     // L-1
        public MemoryBuffer1D<float, Stride1D.Dense> Active = null!;   // [(L-1)*3H] previous tokens' bcx
        public MemoryBuffer1D<float, Stride1D.Dense> Alt = null!;      // ping-pong scratch for the update
    }

    private readonly Dictionary<int, LayerState> _layers = new();

    public ShortConvStateCache(Accelerator accelerator, ShortConvKernel kernel)
    {
        _accelerator = accelerator;
        _kernel = kernel;
    }

    /// <summary>Number of conv-layers that have populated state.</summary>
    public int NumLayers => _layers.Count;

    /// <summary>Drop all state so the next call to <see cref="Forward"/> starts a fresh sequence (zero-pad).
    /// Call when reusing the cache for a new, unrelated generation.</summary>
    public void Reset()
    {
        foreach (var ls in _layers.Values) { ls.Active.Dispose(); ls.Alt.Dispose(); }
        _layers.Clear();
    }

    private LayerState Ensure(int layer, int rowWidth, int stateRows)
    {
        if (_layers.TryGetValue(layer, out var ls))
            return ls;
        int elems = Math.Max(1, stateRows * rowWidth);
        ls = new LayerState
        {
            RowWidth = rowWidth,
            StateRows = stateRows,
            Active = _accelerator.Allocate1D<float>(elems),
            Alt = _accelerator.Allocate1D<float>(elems),
        };
        ls.Active.MemSetToZero();   // fresh state = zeros (irrelevant on the pastLen==0 first call, which zero-pads)
        _layers[layer] = ls;
        return ls;
    }

    /// <summary>
    /// Run the short-conv for one decode/prefill step with conv-state. <paramref name="bcx"/> is the in_proj
    /// output [seq,3H]; <paramref name="weight"/> is [H,L]; <paramref name="output"/> receives [seq,H].
    /// <paramref name="pastLen"/> is the decode cursor: 0 = fresh sequence (prefill; zero-pad + snapshot),
    /// &gt;0 = continue (prepend the cached previous-token state). After the conv, the state is updated to the
    /// last <c>L-1</c> rows of the virtual sequence [prevState ++ bcx].
    /// </summary>
    public void Forward(int layer,
        ArrayView1D<float, Stride1D.Dense> bcx,
        ArrayView1D<float, Stride1D.Dense> weight,
        ArrayView1D<float, Stride1D.Dense> output,
        int seq, int H, int L, int pastLen)
    {
        int rowW = 3 * H;
        int stateRows = L - 1;
        var ls = Ensure(layer, rowW, stateRows);
        bool useState = pastLen > 0;   // decode: real history; prefill/first: zero-pad

        // Conv with (or without) the prepended state. stateRows=0 → identical to the plain zero-pad Forward.
        _kernel.ForwardWithState(bcx, weight, output, ls.Active.View, seq, H, L, useState ? stateRows : 0);

        if (stateRows <= 0) return;   // L==1 (degenerate) needs no history

        // Update state = last stateRows rows of the virtual sequence [ (useState? prevState : nothing) ++ bcx ].
        // Write into the ALT buffer (never in-place), then swap. Each row copy is a GPU→GPU CopyFrom (all backends).
        int prevRows = useState ? stateRows : 0;
        int totalRows = prevRows + seq;
        // Only a sequence shorter than the history window (a <L-1-token prefill) leaves leading state rows
        // unfilled; zero the whole ALT buffer first in that rare case so those rows are zeros, not stale.
        if (totalRows < stateRows) ls.Alt.MemSetToZero();
        var dst = ls.Alt.View;
        for (int i = 0; i < stateRows; i++)
        {
            int virt = totalRows - stateRows + i;   // source row in the virtual sequence
            if (virt < 0) continue;                 // pre-zeroed above
            var dstRow = dst.SubView((long)i * rowW, rowW);
            if (virt < prevRows)
                dstRow.CopyFrom(ls.Active.View.SubView((long)virt * rowW, rowW)); // from the previous state
            else
                dstRow.CopyFrom(bcx.SubView((long)(virt - prevRows) * rowW, rowW)); // from this step's bcx
        }
        (ls.Active, ls.Alt) = (ls.Alt, ls.Active);   // ping-pong: ALT becomes the live state
    }

    public void Dispose() => Reset();
}
