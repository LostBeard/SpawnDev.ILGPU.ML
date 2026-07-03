using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Kernels;

/// <summary>
/// Additional element-wise kernels needed for full pipeline support.
/// These supplement the existing ElementWiseKernels with operations needed
/// by encoder-decoder models, attention masking, and text generation.
/// </summary>
public class MissingElementWiseKernels : IDisposable
{
    private readonly Accelerator _accelerator;

    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>? _expKernel;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>? _logKernel;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>? _ceilKernel;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>? _floorKernel;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>? _roundKernel;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>? _signKernel;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>? _reciprocalKernel;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>? _minKernel;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>? _maxKernel;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>? _whereKernel;

    // DepthToSpace / PixelShuffle
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>>? _depthToSpaceKernel;

    // Expand (broadcast copy)
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>>? _expandKernel;

    // TopK (indices stored as float to avoid Wasm Int32Array alignment issues)
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int>? _topKKernel;
    private MemoryBuffer1D<float, Stride1D.Dense>? _topKIdxBuf;
    private readonly List<MemoryBuffer1D<float, Stride1D.Dense>> _oldTopKIdxBufs = new();

    private MemoryBuffer1D<int, Stride1D.Dense>? _paramsBuf;
    // Deferred disposal: see TransposeKernel for the rationale (WebGPU
    // command-encoder may still reference the prior _paramsBuf at next sync).
    private readonly List<MemoryBuffer1D<int, Stride1D.Dense>> _oldParamsBufs = new();

    public MissingElementWiseKernels(Accelerator accelerator) => _accelerator = accelerator;

    public void Dispose()
    {
        _paramsBuf?.Dispose();
        foreach (var buf in _oldParamsBufs) buf.Dispose();
        _oldParamsBufs.Clear();
        _topKIdxBuf?.Dispose();
        foreach (var buf in _oldTopKIdxBufs) buf.Dispose();
        _oldTopKIdxBufs.Clear();
    }

    // ──────────────────────────────────────────────
    //  Unary ops
    // ──────────────────────────────────────────────

    private static void ExpImpl(Index1D i, ArrayView1D<float, Stride1D.Dense> input, ArrayView1D<float, Stride1D.Dense> output) => output[i] = MathF.Exp(input[i]);
    private static void LogImpl(Index1D i, ArrayView1D<float, Stride1D.Dense> input, ArrayView1D<float, Stride1D.Dense> output) => output[i] = MathF.Log(input[i]);
    private static void CeilImpl(Index1D i, ArrayView1D<float, Stride1D.Dense> input, ArrayView1D<float, Stride1D.Dense> output) => output[i] = MathF.Ceiling(input[i]);
    private static void FloorImpl(Index1D i, ArrayView1D<float, Stride1D.Dense> input, ArrayView1D<float, Stride1D.Dense> output) => output[i] = MathF.Floor(input[i]);
    private static void RoundImpl(Index1D i, ArrayView1D<float, Stride1D.Dense> input, ArrayView1D<float, Stride1D.Dense> output) => output[i] = MathF.Round(input[i]);
    private static void SignImpl(Index1D i, ArrayView1D<float, Stride1D.Dense> input, ArrayView1D<float, Stride1D.Dense> output) => output[i] = input[i] > 0 ? 1f : (input[i] < 0 ? -1f : 0f);
    private static void ReciprocalImpl(Index1D i, ArrayView1D<float, Stride1D.Dense> input, ArrayView1D<float, Stride1D.Dense> output) => output[i] = 1f / input[i];

    public void Exp(ArrayView1D<float, Stride1D.Dense> input, ArrayView1D<float, Stride1D.Dense> output, int count) { _expKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(ExpImpl); _expKernel(count, input, output); }
    public void Log(ArrayView1D<float, Stride1D.Dense> input, ArrayView1D<float, Stride1D.Dense> output, int count) { _logKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(LogImpl); _logKernel(count, input, output); }
    public void Ceil(ArrayView1D<float, Stride1D.Dense> input, ArrayView1D<float, Stride1D.Dense> output, int count) { _ceilKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(CeilImpl); _ceilKernel(count, input, output); }
    public void Floor(ArrayView1D<float, Stride1D.Dense> input, ArrayView1D<float, Stride1D.Dense> output, int count) { _floorKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(FloorImpl); _floorKernel(count, input, output); }
    public void Round(ArrayView1D<float, Stride1D.Dense> input, ArrayView1D<float, Stride1D.Dense> output, int count) { _roundKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(RoundImpl); _roundKernel(count, input, output); }
    public void Sign(ArrayView1D<float, Stride1D.Dense> input, ArrayView1D<float, Stride1D.Dense> output, int count) { _signKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(SignImpl); _signKernel(count, input, output); }
    public void Reciprocal(ArrayView1D<float, Stride1D.Dense> input, ArrayView1D<float, Stride1D.Dense> output, int count) { _reciprocalKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(ReciprocalImpl); _reciprocalKernel(count, input, output); }

    // ──────────────────────────────────────────────
    //  Binary ops
    // ──────────────────────────────────────────────

    private static void MinImpl(Index1D i, ArrayView1D<float, Stride1D.Dense> a, ArrayView1D<float, Stride1D.Dense> b, ArrayView1D<float, Stride1D.Dense> output) => output[i] = MathF.Min(a[i], b[i]);
    private static void MaxImpl(Index1D i, ArrayView1D<float, Stride1D.Dense> a, ArrayView1D<float, Stride1D.Dense> b, ArrayView1D<float, Stride1D.Dense> output) => output[i] = MathF.Max(a[i], b[i]);

    public void Min(ArrayView1D<float, Stride1D.Dense> a, ArrayView1D<float, Stride1D.Dense> b, ArrayView1D<float, Stride1D.Dense> output, int count) { _minKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(MinImpl); _minKernel(count, a, b, output); }
    public void Max(ArrayView1D<float, Stride1D.Dense> a, ArrayView1D<float, Stride1D.Dense> b, ArrayView1D<float, Stride1D.Dense> output, int count) { _maxKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(MaxImpl); _maxKernel(count, a, b, output); }

    // ──────────────────────────────────────────────
    //  Where (conditional select)
    // ──────────────────────────────────────────────

    /// <summary>Where: output[i] = condition[i] != 0 ? x[i] : y[i]</summary>
    private static void WhereImpl(Index1D i,
        ArrayView1D<float, Stride1D.Dense> condition,
        ArrayView1D<float, Stride1D.Dense> x,
        ArrayView1D<float, Stride1D.Dense> y,
        ArrayView1D<float, Stride1D.Dense> output)
    {
        output[i] = condition[i] != 0f ? x[i] : y[i];
    }

    public void Where(ArrayView1D<float, Stride1D.Dense> condition, ArrayView1D<float, Stride1D.Dense> x, ArrayView1D<float, Stride1D.Dense> y, ArrayView1D<float, Stride1D.Dense> output, int count)
    {
        _whereKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(WhereImpl);
        _whereKernel(count, condition, x, y, output);
    }

    // ──────────────────────────────────────────────
    //  DepthToSpace (pixel shuffle for super-res)
    // ──────────────────────────────────────────────

    /// <summary>
    /// DepthToSpace: [N, C*r*r, H, W] → [N, C, H*r, W*r]
    /// Rearranges depth data into spatial blocks.
    /// params: [C, H, W, r, mode] (output channels, input height, input width, blocksize, mode)
    /// mode: 0=DCR (default), 1=CRD
    /// </summary>
    private static void DepthToSpaceImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<int, Stride1D.Dense> p)
    {
        int outC = p[0]; int inH = p[1]; int inW = p[2]; int r = p[3]; int mode = p[4];
        int outH = inH * r;
        int outW = inW * r;
        int outHW = outH * outW;

        int oc = idx / outHW;
        int rem = idx % outHW;
        int oy = rem / outW;
        int ox = rem % outW;

        // Map output (oc, oy, ox) back to input (ic, iy, ix)
        int iy = oy / r;
        int ix = ox / r;
        int by = oy % r;
        int bx = ox % r;
        // DCR: ic = oc * r² + by * r + bx
        // CRD: ic = oc * r² + bx * r + by  (column-row-depth ordering)
        int ic = mode == 0
            ? oc * r * r + by * r + bx
            : oc * r * r + bx * r + by;

        output[idx] = input[ic * inH * inW + iy * inW + ix];
    }

    public void DepthToSpace(
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output,
        int outC, int inH, int inW, int blockSize, int mode = 0)
    {
        int totalOutput = outC * inH * blockSize * inW * blockSize;
        // Persistent buffer avoids use-after-dispose on async backends (WebGPU, Wasm)
        var paramsData = new int[] { outC, inH, inW, blockSize, mode };
        EnsureParamsBuf(paramsData.Length);
        // Copy into an EXACT-size subview: EnsureParamsBuf only grows the persistent buffer, so a smaller
        // params payload (a lower-rank op following a higher-rank one) is shorter than _paramsBuf, and
        // ILGPU's CopyFromCPU requires data.Length == view.Length — copying the whole (larger) view throws
        // ArgumentOutOfRange. Sub-viewing to paramsData.Length keeps the lengths matched.
        _paramsBuf!.View.SubView(0, paramsData.Length).CopyFromCPU(paramsData);

        _depthToSpaceKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>>(DepthToSpaceImpl);
        _depthToSpaceKernel(totalOutput, input, output, _paramsBuf.View);
    }

    // ──────────────────────────────────────────────
    //  Expand (broadcast copy to larger shape)
    // ──────────────────────────────────────────────

    /// <summary>
    /// Expand: broadcast input to output shape.
    /// params: [rank, inputShape..., outputShape..., inputStrides...]
    /// One thread per output element.
    /// </summary>
    private static void ExpandImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<int, Stride1D.Dense> p)
    {
        int rank = p[0];

        // Decompose output linear index to multi-dimensional coords
        // Then map each coord to input index (broadcasting: if input dim is 1, use 0)
        int remaining = idx;
        int inputIdx = 0;

        for (int d = rank - 1; d >= 0; d--)
        {
            int outDim = p[1 + rank + d];           // output shape[d]
            int inStride = p[1 + 2 * rank + d];     // input stride[d]
            int inDim = p[1 + d];                    // input shape[d]

            int coord = remaining % outDim;
            remaining /= outDim;

            // Broadcasting: if input dim is 1, always use index 0
            int inCoord = inDim == 1 ? 0 : coord;
            inputIdx += inCoord * inStride;
        }

        output[idx] = input[inputIdx];
    }

    public void Expand(
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output,
        int[] inputShape, int[] outputShape)
    {
        int rank = outputShape.Length;
        int totalOutput = 1;
        for (int i = 0; i < rank; i++) totalOutput *= outputShape[i];

        // Compute input strides
        var inputStrides = new int[rank];
        // Pad input shape to match rank (prepend 1s)
        var paddedInput = new int[rank];
        int offset = rank - inputShape.Length;
        for (int i = 0; i < rank; i++)
            paddedInput[i] = i < offset ? 1 : inputShape[i - offset];

        int stride = 1;
        for (int i = rank - 1; i >= 0; i--)
        {
            inputStrides[i] = paddedInput[i] == 1 ? 0 : stride;
            stride *= paddedInput[i];
        }

        // Pack params: [rank, inputShape..., outputShape..., inputStrides...]
        var paramsData = new int[1 + 3 * rank];
        paramsData[0] = rank;
        for (int i = 0; i < rank; i++)
        {
            paramsData[1 + i] = paddedInput[i];
            paramsData[1 + rank + i] = outputShape[i];
            paramsData[1 + 2 * rank + i] = inputStrides[i];
        }
        ArrayView1D<int, Stride1D.Dense> paramsView;
        if (Graph.GraphExecutor.UseCaptureParamSlots)
        {
            // CUDA-graph capture: stable per-forward slot (no per-call CopyFromCPU/alloc mid-capture).
            paramsView = CaptureParamArena.Shared(_accelerator).RentStableSlot(paramsData);
        }
        else
        {
            // Persistent buffer avoids use-after-dispose on async backends (WebGPU, Wasm)
            EnsureParamsBuf(paramsData.Length);
            // Copy into an EXACT-size subview: EnsureParamsBuf only grows the persistent buffer, so a smaller
            // params payload (a lower-rank op following a higher-rank one) is shorter than _paramsBuf, and
            // ILGPU's CopyFromCPU requires data.Length == view.Length — copying the whole (larger) view throws
            // ArgumentOutOfRange. Sub-viewing to paramsData.Length keeps the lengths matched.
            _paramsBuf!.View.SubView(0, paramsData.Length).CopyFromCPU(paramsData);
            paramsView = _paramsBuf.View.SubView(0, paramsData.Length);
        }

        _expandKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>>(ExpandImpl);
        _expandKernel(totalOutput, input, output, paramsView);
    }

    // ──────────────────────────────────────────────
    //  TopK (for text generation sampling + detection)
    // ──────────────────────────────────────────────

    /// <summary>
    /// TopK: for each row, find the K largest values and their indices.
    /// Input: [rows, cols], Output values: [rows, K], Output indices: [rows, K]
    /// Uses simple selection (fine for K ≤ ~50, which covers all ML use cases).
    /// </summary>
    // Indices stored as float to avoid Wasm Int32Array alignment issues; caller gets float-cast ints.
    //
    // ONE thread per OUTPUT slot, each writing EXACTLY ONE element at its OWN index. The previous
    // version was one-thread-per-row writing all k slots in a loop — multi-store-per-thread, which
    // silently corrupts on the WebGL Transform-Feedback path (a vertex captures only one output, AND
    // only at its own vertex index — no scatter). So the dispatch is rows*k threads (thread g owns
    // output slot g) and each thread finds its (row, ki) result SELF-CONTAINED — no cross-thread/
    // cross-slot reads — via ki+1 rounds of "largest element strictly below the previous round's
    // pick" under a strict total order (value descending, index ascending on ties). That order makes
    // duplicate values deterministic and matches the old greedy+dedup result on all backends.
    private static void TopKStageImpl(Index1D g,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> outputValues,
        ArrayView1D<float, Stride1D.Dense> outputIndices,
        int cols, int k)
    {
        int row = g / k;
        int ki = g % k;
        int rowStart = row * cols;

        // Round 0 finds the global max; each later round finds the largest element strictly below
        // the previous round's pick. After ki+1 rounds, (bestVal, bestIdx) is the ki-th largest
        // (0-indexed). Strict order is value descending, index ascending on ties; using `v > bestVal`
        // (strict) with ascending-c iteration keeps the lowest index on a tie automatically.
        //
        // Deliberately written with ONLY the constructs proven to run correctly on the interpreted
        // ILGPU Wasm backend: a float `-inf` accumulator plus `if (v > bestVal)`. Combining a `bool`
        // local guard (`if (isBelow) { ... }`) with a nested `if (bestIdx < 0) ... else if ...`
        // selection mis-executes on that backend (the body never fires → all -inf), even though each
        // construct works in isolation — so the inner selection is kept flat: one `if (v > bestVal)`.
        float bestVal = float.NegativeInfinity;
        int bestIdx = 0;
        for (int c = 0; c < cols; c++)
        {
            float v = input[rowStart + c];
            if (v > bestVal) { bestVal = v; bestIdx = c; }
        }

        for (int r = 1; r <= ki; r++)
        {
            float prevVal = bestVal;
            int prevIdx = bestIdx;
            bestVal = float.NegativeInfinity;
            bestIdx = 0;
            for (int c = 0; c < cols; c++)
            {
                float v = input[rowStart + c];
                // Candidate must be strictly below (prevVal, prevIdx): either a smaller value, or
                // the same value at a higher index. Two flat branches, each a single `if (v > bestVal)`.
                if (v < prevVal)
                {
                    if (v > bestVal) { bestVal = v; bestIdx = c; }
                }
                else if (v == prevVal)
                {
                    if (c > prevIdx)
                    {
                        if (v > bestVal) { bestVal = v; bestIdx = c; }
                    }
                }
            }
        }

        outputValues[g] = bestVal;
        outputIndices[g] = (float)bestIdx;
    }

    public void TopK(
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> outputValues,
        ArrayView1D<float, Stride1D.Dense> outputIndices,
        int rows, int cols, int k)
    {
        _topKKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            int, int>(TopKStageImpl);

        ArrayView1D<float, Stride1D.Dense> idxView;
        if (outputIndices.Length >= rows * k)
        {
            idxView = outputIndices;
        }
        else
        {
            // Persist the temp buffer — no using/Synchronize-inside-Execute (unsafe on Wasm async dispatch)
            int needed = rows * k;
            if (_topKIdxBuf == null || _topKIdxBuf.Length < needed)
            {
                if (_topKIdxBuf != null) _oldTopKIdxBufs.Add(_topKIdxBuf);
                _topKIdxBuf = _accelerator.Allocate1D<float>(needed);
            }
            idxView = _topKIdxBuf.View;
        }
        // One thread per output slot (rows*k): thread g owns slot g and writes it at its own index
        // (WebGL Transform-Feedback requires write-index == thread-index — no scatter). Each thread
        // is self-contained (see TopKStageImpl), so a single dispatch suffices.
        _topKKernel(rows * k, input, outputValues, idxView, cols, k);
    }

    /// <summary>
    /// Allocates a FRESH params buffer of exactly the requested size, retiring the previous one for
    /// deferred disposal. The old grow-only + overwrite-in-place design was a batching hazard on the
    /// async backends: a pending dispatch in an un-submitted WebGPU encoder / on the Wasm worker pool
    /// still references the buffer, so the next call's CopyFromCPU handed it the WRONG params (the
    /// DAv3 Slice_4 params-content corruption class; see SliceKernel/GatherKernel).
    /// </summary>
    private void EnsureParamsBuf(int minSize)
    {
        if (_paramsBuf != null) _oldParamsBufs.Add(_paramsBuf);
        _paramsBuf = _accelerator.Allocate1D<int>(minSize);
    }
}
