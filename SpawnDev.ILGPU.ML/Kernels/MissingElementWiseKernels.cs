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
    // TopK sort path: init, one bitonic compare-exchange step, gather. Scratch = (value, index) ping-pong pairs.
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int, int, int>? _topKSortInitKernel;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int, int, int>? _topKSortStepKernel;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int, int>? _topKSortGatherKernel;
    private MemoryBuffer1D<float, Stride1D.Dense>? _topKSortScratch;   // 4 x rows*P: valuesA, indicesA, valuesB, indicesB

    /// <summary>One buffer per distinct param set - see <see cref="ParamBufferCache{T}"/>.</summary>
    private readonly ParamBufferCache<int> _params = new();
    // Deferred disposal: see TransposeKernel for the rationale (WebGPU
    // command-encoder may still reference the prior _paramsBuf at next sync).
    private readonly List<MemoryBuffer1D<int, Stride1D.Dense>> _oldParamsBufs = new();

    public MissingElementWiseKernels(Accelerator accelerator) => _accelerator = accelerator;

    public void Dispose()
    {
        _params.Dispose();
        foreach (var buf in _oldParamsBufs) buf.Dispose();
        _oldParamsBufs.Clear();
        _topKIdxBuf?.Dispose();
        _topKSortScratch?.Dispose();
        _nonZeroScratch?.Dispose();
        _logSoftmaxStats?.Dispose();
        _nonZeroScratchB?.Dispose();
        foreach (var buf in _oldNonZeroScratch) buf.Dispose();
        _oldNonZeroScratch.Clear();
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
        // One buffer per DISTINCT param set, holding exactly these values - so there is no write here at
        // all, and the exact-size dance below is unnecessary. The previous version allocated a fresh buffer
        // per call, which is a cuMemAlloc and therefore ILLEGAL inside a CUDA graph-capture window
        // (uncatchable access violation), and then REWROTE it - the very corruption the old comment warned
        // about. A cache hit rewrites nothing.
        var paramsView = _params.Get(_accelerator, paramsData);

        _depthToSpaceKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>>(DepthToSpaceImpl);
        _depthToSpaceKernel(totalOutput, input, output, paramsView);
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
            // One buffer per distinct param set - no write, so nothing a pending dispatch reads is
            // rewritten, and no per-call cuMemAlloc to break CUDA graph capture.
            paramsView = _params.Get(_accelerator, paramsData);
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

    // ── Sort path (any K, any axis, largest or smallest) ─────────────────────────────────────────────────────
    // The selection kernel above costs k(k+1)/2 passes over the row, so it is only usable for small K: RaCo-ALIKED
    // asks for k=2304 of a 65,536-wide row (~1.7e12 reads). This path sorts each row's (value, index) pairs with a
    // bitonic network over the row padded to P = 2^L, L(L+1)/2 steps of rows*P threads, and gathers the first K.
    // Every thread writes ONLY its own slot (read src, write dst, ping-pong), which the WebGL Transform-Feedback
    // path requires. Order: best value first, lower index first among equals (ONNX's sorted=1 order). Padding is
    // the worst value with an index past every real one, so it always sorts last.

    // t = row * P + p; row = o * inner + i addresses input element (o * axisLen + p) * inner + i.
    private static void TopKSortInitImpl(Index1D t,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> vals,
        ArrayView1D<float, Stride1D.Dense> idxs,
        int axisLen, int inner, int P, int largest)
    {
        int row = t / P;
        int p = t - row * P;
        int o = row / inner;
        int i = row - o * inner;
        if (p < axisLen)
        {
            vals[t] = input[(o * axisLen + p) * inner + i];
        }
        else
        {
            vals[t] = largest != 0 ? float.NegativeInfinity : float.PositiveInfinity;
        }
        idxs[t] = (float)p;
    }

    // One compare-exchange step (block size kStage, partner distance jStage). Thread t keeps the better or the worse
    // of (self, partner): the lower position of a pair takes the better one inside a best-first block, the worse
    // one inside a worst-first block.
    private static void TopKSortStepImpl(Index1D t,
        ArrayView1D<float, Stride1D.Dense> srcVals,
        ArrayView1D<float, Stride1D.Dense> srcIdxs,
        ArrayView1D<float, Stride1D.Dense> dstVals,
        ArrayView1D<float, Stride1D.Dense> dstIdxs,
        int P, int kStage, int jStage, int largest)
    {
        int row = t / P;
        int p = t - row * P;
        int partner = p ^ jStage;
        int pt = row * P + partner;
        float av = srcVals[t];
        float ai = srcIdxs[t];
        float bv = srcVals[pt];
        float bi = srcIdxs[pt];
        int aBetter = 0;
        if (largest != 0)
        {
            if (av > bv) aBetter = 1;
            if (av == bv && ai < bi) aBetter = 1;
        }
        else
        {
            if (av < bv) aBetter = 1;
            if (av == bv && ai < bi) aBetter = 1;
        }
        int bestFirst = (p & kStage) == 0 ? 1 : 0;
        int isLow = p < partner ? 1 : 0;
        int wantBetter = bestFirst == isLow ? 1 : 0;
        if (wantBetter == aBetter)
        {
            dstVals[t] = av;
            dstIdxs[t] = ai;
        }
        else
        {
            dstVals[t] = bv;
            dstIdxs[t] = bi;
        }
    }

    // t walks the OUTPUT [outer, k, inner]: slot j of row (o, i) is sorted position j.
    private static void TopKSortGatherImpl(Index1D t,
        ArrayView1D<float, Stride1D.Dense> vals,
        ArrayView1D<float, Stride1D.Dense> idxs,
        ArrayView1D<float, Stride1D.Dense> outputValues,
        ArrayView1D<float, Stride1D.Dense> outputIndices,
        int k, int inner, int P)
    {
        int ki = k * inner;
        int o = t / ki;
        int rem = t - o * ki;
        int j = rem / inner;
        int i = rem - j * inner;
        int src = (o * inner + i) * P + j;
        outputValues[t] = vals[src];
        outputIndices[t] = idxs[src];
    }

    /// <summary>
    /// TopK over <paramref name="axisLen"/> of an [outer, axisLen, inner] input into [outer, k, inner] values and
    /// (float) indices, best first and lower index first among equals. Small K on the last axis (largest) takes the
    /// one-dispatch selection kernel; everything else the bitonic sort path. <paramref name="outputIndices"/> may be
    /// default when the model does not use the indices.
    /// </summary>
    public void TopK(
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> outputValues,
        ArrayView1D<float, Stride1D.Dense> outputIndices,
        int outer, int axisLen, int inner, int k, bool largest)
    {
        int rows = outer * inner;
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

        int L = 0;
        while ((1 << L) < axisLen) L++;
        int P = 1 << L;
        // Selection reads the row k(k+1)/2 times in one dispatch; the sort reads it ~L(L+1)/2 times over L(L+1)/2
        // dispatches. Selection wins whenever k <= L, and it only exists for the last axis and largest.
        if (inner == 1 && largest && k <= L)
        {
            _topKKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
                ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>,
                int, int>(TopKStageImpl);
            // One thread per output slot (rows*k): thread g owns slot g and writes it at its own index
            // (WebGL Transform-Feedback requires write-index == thread-index — no scatter). Each thread
            // is self-contained (see TopKStageImpl), so a single dispatch suffices.
            _topKKernel(rows * k, input, outputValues, idxView, axisLen, k);
            return;
        }

        _topKSortInitKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            int, int, int, int>(TopKSortInitImpl);
        _topKSortStepKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            int, int, int, int>(TopKSortStepImpl);
        _topKSortGatherKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            int, int, int>(TopKSortGatherImpl);

        long n = (long)rows * P;
        if (n * 4 > int.MaxValue) throw new NotSupportedException($"TopK: {rows} rows x {P} padded length is too large");
        if (_topKSortScratch == null || _topKSortScratch.Length < n * 4)
        {
            // Retired, not disposed: a dispatch still queued on WebGPU/Wasm may reference it (see the class notes).
            if (_topKSortScratch != null) _oldTopKIdxBufs.Add(_topKSortScratch);
            _topKSortScratch = _accelerator.Allocate1D<float>(n * 4);
        }
        int ni = (int)n;
        var sv = _topKSortScratch.View;
        var aV = sv.SubView(0, ni); var aI = sv.SubView(ni, ni);
        var bV = sv.SubView(2 * ni, ni); var bI = sv.SubView(3 * ni, ni);
        int lg = largest ? 1 : 0;
        _topKSortInitKernel(ni, input, aV, aI, axisLen, inner, P, lg);
        for (int kStage = 2; kStage <= P; kStage <<= 1)
            for (int jStage = kStage >> 1; jStage > 0; jStage >>= 1)
            {
                _topKSortStepKernel(ni, aV, aI, bV, bI, P, kStage, jStage, lg);
                (aV, bV) = (bV, aV);
                (aI, bI) = (bI, aI);
            }
        _topKSortGatherKernel(outer * k * inner, aV, aI, outputValues, idxView, k, inner, P);
    }

    // ──────────────────────────────────────────────
    //  NonZero (data-dependent output size)
    // ──────────────────────────────────────────────
    // Inclusive prefix count of the non-zero flags (Hillis-Steele, log2(n) ping-pong steps, one thread per element
    // writing only its own slot - WebGL Transform-Feedback safe), ONE int readback of the total, then a gather per
    // output row: slot j finds the j-th non-zero element by binary search over the prefix (monotonic). Integer
    // counts are exact at any size.

    private static void NonZeroFlagImpl(Index1D i, ArrayView1D<float, Stride1D.Dense> input, ArrayView1D<int, Stride1D.Dense> flags)
    {
        flags[i] = input[i] != 0f ? 1 : 0;
    }

    private static void NonZeroScanStepImpl(Index1D i, ArrayView1D<int, Stride1D.Dense> src, ArrayView1D<int, Stride1D.Dense> dst, int offset)
    {
        int v = src[i];
        if (i >= offset) v += src[i - offset];
        dst[i] = v;
    }

    // Row d of the [rank, count] output: coordinate d of the j-th non-zero element, (index / stride) % dim.
    // `row` is that row's own SubView, so thread j writes slot j: WebGL's Transform Feedback writes a thread's value at
    // its OWN index, and writing row d at d*count + j there put row 1 on top of row 0. The search runs a fixed
    // `steps` = ceil(log2 n) + 1 iterations (a uniform loop bound).
    private static void NonZeroGatherRowImpl(Index1D j, ArrayView1D<int, Stride1D.Dense> prefix, ArrayView1D<float, Stride1D.Dense> row,
        int n, int stride, int dim, int steps)
    {
        int target = j + 1;
        int lo = 0, hi = n - 1;
        for (int s = 0; s < steps; s++)
        {
            int mid = (lo + hi) / 2;
            int geq = prefix[mid] >= target ? 1 : 0;
            if (lo < hi)
            {
                if (geq == 1) hi = mid;
                else lo = mid + 1;
            }
        }
        row[j] = (float)((lo / stride) % dim);
    }

    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>>? _nonZeroFlagKernel;
    private Action<Index1D, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, int>? _nonZeroScanKernel;
    private Action<Index1D, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int, int, int>? _nonZeroGatherKernel;
    // Ping-pong pair as TWO buffers: two SubViews of one buffer overlap after WebGPU's binding alignment (aliasing).
    private MemoryBuffer1D<int, Stride1D.Dense>? _nonZeroScratch, _nonZeroScratchB;
    private readonly List<MemoryBuffer1D<int, Stride1D.Dense>> _oldNonZeroScratch = new();

    /// <summary>
    /// Phase 1 of NonZero: the inclusive prefix count of <paramref name="input"/>'s non-zero elements. Returns the view
    /// holding it; its LAST element is the total, which the caller reads back (one int) to size the output.
    /// </summary>
    public ArrayView1D<int, Stride1D.Dense> NonZeroPrefix(ArrayView1D<float, Stride1D.Dense> input, int n)
    {
        _nonZeroFlagKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>>(NonZeroFlagImpl);
        _nonZeroScanKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, int>(NonZeroScanStepImpl);
        if (_nonZeroScratch == null || _nonZeroScratchB == null || _nonZeroScratch.Length < n)
        {
            // Retired, not disposed: a queued dispatch may still reference them (see the class notes).
            if (_nonZeroScratch != null) _oldNonZeroScratch.Add(_nonZeroScratch);
            if (_nonZeroScratchB != null) _oldNonZeroScratch.Add(_nonZeroScratchB);
            _nonZeroScratch = _accelerator.Allocate1D<int>(n);
            _nonZeroScratchB = _accelerator.Allocate1D<int>(n);
        }
        var a = _nonZeroScratch.View.SubView(0, n);
        var b = _nonZeroScratchB.View.SubView(0, n);
        _nonZeroFlagKernel(n, input, a);
        for (int offset = 1; offset < n; offset <<= 1)
        {
            _nonZeroScanKernel(n, a, b, offset);
            (a, b) = (b, a);
        }
        return a;
    }

    /// <summary>
    /// Phase 2 of NonZero: writes the [rank, <paramref name="count"/>] coordinates (row-major, as float) from the
    /// prefix <see cref="NonZeroPrefix"/> returned.
    /// </summary>
    public void NonZeroGather(ArrayView1D<int, Stride1D.Dense> prefix, int n, int[] shape, int count, ArrayView1D<float, Stride1D.Dense> output)
    {
        if (count == 0) return;
        _nonZeroGatherKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int, int, int>(NonZeroGatherRowImpl);
        int rank = shape.Length;
        if (rank == 0)
            return;   // rank-0 input: output [0, count] holds no values
        int steps = 1;
        while ((1 << (steps - 1)) < n) steps++;
        int stride = 1;
        for (int d = rank - 1; d >= 0; d--)
        {
            _nonZeroGatherKernel(count, prefix, output.SubView(d * count, count), n, stride, Math.Max(1, shape[d]), steps);
            stride *= shape[d];
        }
    }

    // ──────────────────────────────────────────────
    //  LogSoftmax along any axis, computed in log space
    // ──────────────────────────────────────────────
    // [outer, axisDim, inner]: per column (outer, inner) the log-sum-exp max + log(sum exp(x - max)), then
    // out = x - lse. Never log(softmax(x)): that underflows for very negative x (-31.4 where -31.1 is right, -inf past
    // ~1e-38). Both kernels write only their own slot and loop over the uniform axisDim (WebGL-safe).
    private static void LogSoftmaxStatsImpl(Index1D c, ArrayView1D<float, Stride1D.Dense> input, ArrayView1D<float, Stride1D.Dense> lse,
        int axisDim, int inner)
    {
        int o = c / inner;
        int i = c - o * inner;
        int baseIdx = o * axisDim * inner + i;
        float mx = float.NegativeInfinity;
        for (int a = 0; a < axisDim; a++)
        {
            float v = input[baseIdx + a * inner];
            if (v > mx) mx = v;
        }
        float sum = 0f;
        for (int a = 0; a < axisDim; a++)
            sum += MathF.Exp(input[baseIdx + a * inner] - mx);
        lse[c] = mx + MathF.Log(sum);
    }

    private static void LogSoftmaxApplyImpl(Index1D t, ArrayView1D<float, Stride1D.Dense> input, ArrayView1D<float, Stride1D.Dense> lse,
        ArrayView1D<float, Stride1D.Dense> output, int axisDim, int inner)
    {
        int block = axisDim * inner;
        int o = t / block;
        int i = (t - o * block) % inner;
        output[t] = input[t] - lse[o * inner + i];
    }

    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int>? _logSoftmaxStatsKernel;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int>? _logSoftmaxApplyKernel;
    private MemoryBuffer1D<float, Stride1D.Dense>? _logSoftmaxStats;

    /// <summary>LogSoftmax of an [outer, axisDim, inner] tensor along axisDim. <paramref name="output"/> must not alias
    /// <paramref name="input"/>.</summary>
    public void LogSoftmax(ArrayView1D<float, Stride1D.Dense> input, ArrayView1D<float, Stride1D.Dense> output, int outer, int axisDim, int inner)
    {
        _logSoftmaxStatsKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int>(LogSoftmaxStatsImpl);
        _logSoftmaxApplyKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int>(LogSoftmaxApplyImpl);
        int cols = outer * inner;
        if (_logSoftmaxStats == null || _logSoftmaxStats.Length < cols)
        {
            // Retired, not disposed: a queued dispatch may still reference it (see the class notes).
            if (_logSoftmaxStats != null) _oldTopKIdxBufs.Add(_logSoftmaxStats);
            _logSoftmaxStats = _accelerator.Allocate1D<float>(cols);
        }
        var lse = _logSoftmaxStats.View.SubView(0, cols);
        _logSoftmaxStatsKernel(cols, input, lse, axisDim, inner);
        _logSoftmaxApplyKernel(cols * axisDim, input, lse, output, axisDim, inner);
    }

    /// <summary>
    /// Allocates a FRESH params buffer of exactly the requested size, retiring the previous one for
    /// deferred disposal. The old grow-only + overwrite-in-place design was a batching hazard on the
    /// async backends: a pending dispatch in an un-submitted WebGPU encoder / on the Wasm worker pool
    /// still references the buffer, so the next call's CopyFromCPU handed it the WRONG params (the
    /// DAv3 Slice_4 params-content corruption class; see SliceKernel/GatherKernel).
    /// </summary>
}
