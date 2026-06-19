using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Kernels;

/// <summary>
/// General N-dimensional transpose (permutation of axes).
/// Maps ONNX Transpose op.
///
/// For common 2D/3D cases, consider using ElementWiseKernels.TransposeLastTwo
/// which is more efficient (no params buffer needed).
///
/// This kernel handles arbitrary permutations up to 6 dimensions via a params
/// buffer that encodes the shape and permutation.
/// </summary>
public class TransposeKernel : IDisposable
{
    private readonly Accelerator _accelerator;

    // params: [rank, shape[0..rank-1], perm[0..rank-1], inStrides[0..rank-1], outStrides[0..rank-1]]
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<int, Stride1D.Dense>>? _transposeKernel;

    // Content-keyed paramsBuf cache (key = the packed [rank,shape,perm,inStrides,outStrides] ints).
    // The params for a given (shape,perm) are CONSTANT, so this allocates + uploads the tiny params
    // buffer ONCE per distinct transpose configuration and reuses it on every later call with the same
    // shape/perm. This kills a per-call `Allocate1D` (cudaMalloc) + synchronous `CopyFromCPU` that was
    // forcing a mid-layer device sync — on gemma4 seq=1 decode the 192 per-token transposes were 98.9%
    // of CPU dispatch time (~1.6s/token), because each malloc-sync blocked on the preceding projection
    // matmul. Each cache entry is written exactly ONCE and never overwritten, so it PRESERVES the Wasm
    // safety invariant the old per-call-fresh code guaranteed (a queued dispatch can never read a buffer
    // that a later CopyFromCPU has clobbered — there is no later CopyFromCPU on a cache hit). Same
    // per-call-alloc pathology exists in PadKernel/Conv2DKernel/PoolingKernels/NormalizationKernels;
    // they're not on the gemma4 decode path but want the same treatment.
    private readonly List<MemoryBuffer1D<int, Stride1D.Dense>> _allParamsBufs = new();
    private readonly Dictionary<string, MemoryBuffer1D<int, Stride1D.Dense>> _paramsCache = new();

    public TransposeKernel(Accelerator accelerator) => _accelerator = accelerator;

    public void Dispose()
    {
        foreach (var buf in _allParamsBufs) buf.Dispose();
        _allParamsBufs.Clear();
    }

    /// <summary>
    /// General transpose: for each output element, compute its source index
    /// by reversing the permutation through the strides.
    /// </summary>
    private static void TransposeImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<int, Stride1D.Dense> p)
    {
        int rank = p[0];
        // p layout: [rank, inShape[rank], perm[rank], inStrides[rank], outStrides[rank]]
        // offsets:   0      1              1+rank       1+2*rank         1+3*rank

        // Decompose output linear index into output coordinates
        int remaining = idx;
        int srcIdx = 0;
        for (int d = 0; d < rank; d++)
        {
            int outStride = p[1 + 3 * rank + d];
            int coord = remaining / outStride;
            remaining = remaining % outStride;

            // This output dim d corresponds to input dim perm[d]
            int srcDim = p[1 + rank + d];
            int inStride = p[1 + 2 * rank + srcDim];
            srcIdx += coord * inStride;
        }

        output[idx] = input[srcIdx];
    }

    /// <summary>
    /// Transpose input with given shape by the specified permutation.
    /// perm[i] = which input dimension maps to output dimension i.
    /// E.g., perm=[2,0,1] on shape [A,B,C] → output shape [C,A,B].
    /// </summary>
    public void Transpose(ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output,
        int[] inputShape, int[] perm)
    {
        EnsureLoaded();
        int rank = inputShape.Length;
        if (perm.Length != rank) throw new ArgumentException("Perm length must match rank");

        // Compute strides
        var inStrides = new int[rank];
        var outShape = new int[rank];
        var outStrides = new int[rank];

        inStrides[rank - 1] = 1;
        for (int i = rank - 2; i >= 0; i--)
            inStrides[i] = inStrides[i + 1] * inputShape[i + 1];

        for (int i = 0; i < rank; i++)
            outShape[i] = inputShape[perm[i]];

        outStrides[rank - 1] = 1;
        for (int i = rank - 2; i >= 0; i--)
            outStrides[i] = outStrides[i + 1] * outShape[i + 1];

        int totalElements = 1;
        for (int i = 0; i < rank; i++) totalElements *= inputShape[i];

        // Pack params: [rank, inShape..., perm..., inStrides..., outStrides...]
        int paramsSize = 1 + 4 * rank;
        var paramsData = new int[paramsSize];
        paramsData[0] = rank;
        for (int i = 0; i < rank; i++) paramsData[1 + i] = inputShape[i];
        for (int i = 0; i < rank; i++) paramsData[1 + rank + i] = perm[i];
        for (int i = 0; i < rank; i++) paramsData[1 + 2 * rank + i] = inStrides[i];
        for (int i = 0; i < rank; i++) paramsData[1 + 3 * rank + i] = outStrides[i];

        // Cache the GPU params buffer by content — the (shape,perm) for a given transpose node is
        // constant across decode steps, so this allocates + uploads once and reuses thereafter,
        // eliminating the per-call cudaMalloc + sync HtoD that dominated gemma4 decode. Each buffer
        // is written exactly once on its miss and never overwritten (Wasm-safe; see field comment).
        var key = string.Join(",", paramsData);
        if (!_paramsCache.TryGetValue(key, out var paramsBuf))
        {
            paramsBuf = _accelerator.Allocate1D<int>(paramsSize);
            paramsBuf.View.CopyFromCPU(paramsData);
            _paramsCache[key] = paramsBuf;
            _allParamsBufs.Add(paramsBuf);
        }

        _transposeKernel!(totalElements, input, output, paramsBuf.View);
    }

    private void EnsureLoaded()
    {
        _transposeKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>>(TransposeImpl);
    }

    // ── Generic transpose over any element type (additive; the f32 path above is untouched). ──
    // Transpose is pure index remap (no arithmetic), so it works for any unmanaged T. Used to transpose a
    // NATIVE low-precision weight (e.g. BFloat16 [N,K] -> [K,N]) at load WITHOUT upcasting to f32 — keeps the
    // bytes low-p end to end (no-needless-conversion). One compiled kernel per concrete T, cached.
    private readonly Dictionary<Type, object> _transposeKernelT = new();

    private static void TransposeImplT<T>(Index1D idx,
        ArrayView1D<T, Stride1D.Dense> input, ArrayView1D<T, Stride1D.Dense> output,
        ArrayView1D<int, Stride1D.Dense> p) where T : unmanaged
    {
        int rank = p[0];
        int remaining = idx;
        int srcIdx = 0;
        for (int d = 0; d < rank; d++)
        {
            int outStride = p[1 + 3 * rank + d];
            int coord = remaining / outStride;
            remaining = remaining % outStride;
            int srcDim = p[1 + rank + d];
            int inStride = p[1 + 2 * rank + srcDim];
            srcIdx += coord * inStride;
        }
        output[idx] = input[srcIdx];
    }

    /// <summary>Generic transpose: <paramref name="output"/> = <paramref name="input"/> permuted by
    /// <paramref name="perm"/> (perm[i] = which input dim maps to output dim i), for any unmanaged element
    /// type. Keeps low-precision weights native (no f32 upcast). Shares the content-keyed params-buffer cache.</summary>
    public void Transpose<T>(ArrayView1D<T, Stride1D.Dense> input, ArrayView1D<T, Stride1D.Dense> output,
        int[] inputShape, int[] perm) where T : unmanaged
    {
        int rank = inputShape.Length;
        if (perm.Length != rank) throw new ArgumentException("Perm length must match rank");

        var inStrides = new int[rank];
        var outShape = new int[rank];
        var outStrides = new int[rank];
        inStrides[rank - 1] = 1;
        for (int i = rank - 2; i >= 0; i--) inStrides[i] = inStrides[i + 1] * inputShape[i + 1];
        for (int i = 0; i < rank; i++) outShape[i] = inputShape[perm[i]];
        outStrides[rank - 1] = 1;
        for (int i = rank - 2; i >= 0; i--) outStrides[i] = outStrides[i + 1] * outShape[i + 1];
        int totalElements = 1;
        for (int i = 0; i < rank; i++) totalElements *= inputShape[i];

        int paramsSize = 1 + 4 * rank;
        var paramsData = new int[paramsSize];
        paramsData[0] = rank;
        for (int i = 0; i < rank; i++) paramsData[1 + i] = inputShape[i];
        for (int i = 0; i < rank; i++) paramsData[1 + rank + i] = perm[i];
        for (int i = 0; i < rank; i++) paramsData[1 + 2 * rank + i] = inStrides[i];
        for (int i = 0; i < rank; i++) paramsData[1 + 3 * rank + i] = outStrides[i];

        var key = "T:" + string.Join(",", paramsData);
        if (!_paramsCache.TryGetValue(key, out var paramsBuf))
        {
            paramsBuf = _accelerator.Allocate1D<int>(paramsSize);
            paramsBuf.View.CopyFromCPU(paramsData);
            _paramsCache[key] = paramsBuf;
            _allParamsBufs.Add(paramsBuf);
        }

        if (!_transposeKernelT.TryGetValue(typeof(T), out var k))
            _transposeKernelT[typeof(T)] = k = _accelerator.LoadAutoGroupedStreamKernel<Index1D,
                ArrayView1D<T, Stride1D.Dense>, ArrayView1D<T, Stride1D.Dense>,
                ArrayView1D<int, Stride1D.Dense>>(TransposeImplT<T>);
        ((Action<Index1D, ArrayView1D<T, Stride1D.Dense>, ArrayView1D<T, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>>)k)(totalElements, input, output, paramsBuf.View);
    }
}
