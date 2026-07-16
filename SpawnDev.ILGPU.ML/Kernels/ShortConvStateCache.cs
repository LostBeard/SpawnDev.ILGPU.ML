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
/// conv and snapshots the tail as the initial state; decode (pastLen&gt;0) prepends the state. The update is
/// staged through a SCRATCH buffer and committed back into the SAME Active buffer: browser-safe (never an
/// in-place overlapping copy, never a method-local buffer feeding a pending dispatch) AND binding-stable, so
/// WebGPU decode capture/replay - which re-executes a FROZEN command plan per token and never re-runs this
/// C# - stays correct. It was ping-pong until 2026-07-16; the swap rebound the live state every step, which a
/// replayed plan cannot follow, so LFM2 froze its conv history at the capture step and degenerated into token
/// soup in the browser while CUDA (capture off) was fine. Do NOT reintroduce a swap here.
/// </summary>
public sealed class ShortConvStateCache : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly ShortConvKernel _kernel;

    private sealed class LayerState
    {
        public int RowWidth;      // 3H
        public int StateRows;     // L-1
        // FIXED (never swapped - see the binding-stability note in Forward): Active is always the live state,
        // Scratch always the staging buffer, so a captured WebGPU decode plan replays correctly.
        public MemoryBuffer1D<float, Stride1D.Dense> Active = null!;   // [(L-1)*3H] previous tokens' bcx
        public MemoryBuffer1D<float, Stride1D.Dense> Scratch = null!;  // staging for the update (never in-place)
    }

    private readonly Dictionary<int, LayerState> _layers = new();

    public ShortConvStateCache(Accelerator accelerator, ShortConvKernel kernel)
    {
        _accelerator = accelerator;
        _kernel = kernel;
    }

    /// <summary>Number of conv-layers that have populated state.</summary>
    public int NumLayers => _layers.Count;

    /// <summary>
    /// The absolute sequence position this state describes: it holds the bcx of tokens
    /// [StatePos-(L-1) .. StatePos-1], i.e. the history a step at cursor <c>pastLen == StatePos</c> needs.
    ///
    /// This exists because the conv state, unlike the KV cache, is NOT position-addressed. `GGUFDecodeKVCache`
    /// writes row `pastLen`, so a consumer may jump the cursor anywhere already computed (that is what the
    /// KV-PREFIX CACHE does: reuse the longest common prefix P, set the cursor to P, prefill only the suffix).
    /// This cache is a shift register holding history for exactly ONE cursor, so a jump to any other P feeds
    /// the conv layers a history describing DIFFERENT tokens - the same prompt asked twice then returns a
    /// different answer, and on some backends outright garbage (measured on CUDA/WebGPU/WebGL/CPU 2026-07-16).
    /// Callers that move the cursor must check this and force a full prefill when it does not match.
    /// </summary>
    public int StatePos { get; private set; }

    /// <summary>A GPU-side copy of every layer's live conv state. See <see cref="CreateSnapshot"/>.</summary>
    public sealed class Snapshot : IDisposable
    {
        internal readonly Dictionary<int, MemoryBuffer1D<float, Stride1D.Dense>> Layers = new();
        internal int StatePos;   // the cursor the snapshotted buffers describe - restored with them
        public void Dispose()
        {
            foreach (var b in Layers.Values) b.Dispose();
            Layers.Clear();
        }
    }

    /// <summary>
    /// Copy every layer's live state so it can be put back with <see cref="RestoreSnapshot"/>.
    ///
    /// Why this exists: unlike <see cref="GGUFDecodeKVCache"/> - which writes row <c>pastLen</c> and is therefore
    /// IDEMPOTENT when a step is re-run at the same cursor - this cache is a SHIFT REGISTER: every
    /// <see cref="Forward"/> advances the history by one, regardless of pastLen. WebGPU decode capture
    /// (<see cref="WebGPUDecodeCapture.TryCaptureAsync"/>) runs the decode graph SIX times to discover its
    /// cursor-dependent patch points (warm/probe/capture at P0, then again at P0+1), which would shift the conv
    /// history six times with a throwaway token and leave replay starting from a corrupted state (LFM2 then
    /// decodes fluent but WRONG text - 2026-07-16). Capture snapshots around its probes so the net effect is
    /// exactly the one real step it reports. Any other multi-run-at-one-cursor caller must do the same.
    /// </summary>
    public Snapshot CreateSnapshot()
    {
        var snap = new Snapshot { StatePos = StatePos };
        foreach (var (layer, ls) in _layers)
        {
            var buf = _accelerator.Allocate1D<float>(ls.Active.Length);
            buf.View.CopyFrom(ls.Active.View);
            snap.Layers[layer] = buf;
        }
        return snap;
    }

    /// <summary>Put a <see cref="CreateSnapshot"/> state back (layers absent from the snapshot are untouched).
    /// Restores <see cref="StatePos"/> too - the buffers and the cursor they describe travel together.</summary>
    public void RestoreSnapshot(Snapshot snap)
    {
        foreach (var (layer, buf) in snap.Layers)
            if (_layers.TryGetValue(layer, out var ls))
                ls.Active.View.CopyFrom(buf.View);
        StatePos = snap.StatePos;
    }

    /// <summary>Drop all state so the next call to <see cref="Forward"/> starts a fresh sequence (zero-pad).
    /// Call when reusing the cache for a new, unrelated generation.</summary>
    public void Reset()
    {
        // ZERO the buffers, do NOT dispose them. Reset runs from SYNC paths (ResetGGUFDecode, called at the
        // start of every full prefill), where a dispatch from the previous generation may still be pending:
        // disposing a buffer a pending dispatch reads corrupts it on the browser backends (see the
        // "never dispose before flush" / Wasm SharedArrayBuffer rules in the project CLAUDE.md). It surfaced
        // as the SECOND generation returning garbage on WebGL even after a full prefill (2026-07-16).
        // Reusing the allocation is also what ResetGGUFDecode's contract promises, and it is cheaper.
        foreach (var ls in _layers.Values) { ls.Active.MemSetToZero(); ls.Scratch.MemSetToZero(); }
        StatePos = 0;
    }

    private LayerState Ensure(int layer, int rowWidth, int stateRows)
    {
        if (_layers.TryGetValue(layer, out var ls))
        {
            // Reset() keeps the allocations (browser-safe), so a layer can outlive a generation. Geometry is
            // fixed per model/session, but rebuild rather than silently reuse a wrong-sized buffer if it ever
            // changes.
            if (ls.RowWidth == rowWidth && ls.StateRows == stateRows) return ls;
            ls.Active.Dispose(); ls.Scratch.Dispose();
            _layers.Remove(layer);
        }
        int elems = Math.Max(1, stateRows * rowWidth);
        ls = new LayerState
        {
            RowWidth = rowWidth,
            StateRows = stateRows,
            Active = _accelerator.Allocate1D<float>(elems),
            Scratch = _accelerator.Allocate1D<float>(elems),
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

        // The state now describes the history a step at cursor (pastLen + seq) needs. Callers that MOVE the
        // cursor (KV-prefix reuse) must compare against this - see the StatePos docs. Set on BOTH paths: an
        // L==1 model keeps no history, so its (empty) state is valid at every cursor and must not be treated
        // as stale.
        StatePos = pastLen + seq;

        if (stateRows <= 0) return;   // L==1 (degenerate) needs no history

        // Update state = last stateRows rows of the virtual sequence [ (useState? prevState : nothing) ++ bcx ].
        // Staged through SCRATCH (never in-place: the conv dispatch above still reads Active), then COMMITTED
        // back into the SAME Active buffer. Each copy is a GPU→GPU CopyFrom (all backends).
        //
        // Active/Scratch are FIXED - deliberately NOT ping-pong. A ping-pong swap rebinds which buffer is the
        // live state each step, and WebGPU decode capture/replay (WebGPUDecodeCapture) records ONE command plan
        // with FROZEN bindings and re-executes it per token, patching only values affine in the decode cursor.
        // The C# here does not run on replay, so a swap would never happen again: every replay would read the
        // capture-time Active and write the capture-time Scratch, freezing the conv history at the capture step.
        // LFM2 then decodes coherently for a few tokens and degenerates into token soup - exactly the browser
        // bug (2026-07-16); qwen has no conv state, so its plan is fully described by the affine patches and
        // replayed fine, which is why only LFM2 broke. Keep the bindings stable. Cost: one extra copy of
        // (L-1)*3H floats per conv layer per step (48KB at LFM2's H=2048, L=3).
        int prevRows = useState ? stateRows : 0;
        int totalRows = prevRows + seq;
        // Only a sequence shorter than the history window (a <L-1-token prefill) leaves leading state rows
        // unfilled; zero the whole SCRATCH buffer first in that rare case so those rows are zeros, not stale.
        if (totalRows < stateRows) ls.Scratch.MemSetToZero();
        var dst = ls.Scratch.View;
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
        // Commit: SCRATCH -> ACTIVE. Ordered after the reads above within the same queue, so the conv dispatch
        // and the row copies both saw the OLD state before it is overwritten.
        ls.Active.View.CopyFrom(ls.Scratch.View);
    }

    /// <summary>Release every layer's buffers. NOT Reset() - that deliberately KEEPS the allocations (it runs
    /// from sync paths where a dispatch may still reference them); disposal only happens here, when the session
    /// is done with the cache.</summary>
    public void Dispose()
    {
        foreach (var ls in _layers.Values) { ls.Active.Dispose(); ls.Scratch.Dispose(); }
        _layers.Clear();
        StatePos = 0;
    }
}
