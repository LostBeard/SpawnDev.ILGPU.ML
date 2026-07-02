using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Kernels;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace SpawnDev.ILGPU.ML;

/// <summary>
/// Tiled matrix multiplication using ILGPU shared memory.
/// C[M,N] = A[M,K] × B[K,N]
///
/// Uses 16×16 tiles to stay within WebGPU's 256 invocations/workgroup limit.
/// 1D shared memory with manual 2D indexing (Allocate2D not supported on WebGPU backend).
///
/// Future home: SpawnDev.ILGPU.ML
/// </summary>
public class MatMulKernel
{
    private const int TILE = 16; // 16×16 = 256 threads = WebGPU max workgroup size

    private const int REG_TILE = 64; // RegisterBlockedMatMul uses 64×64 output tiles

    private readonly Accelerator _accelerator;
    private readonly bool _useSimpleKernels; // CPU/WebGL can't handle 256-thread groups
    private RegisterBlockedMatMul? _regBlockedMatMul; // lazy-init, used for large matrices

    // LoadStreamKernel returns Action<KernelConfig, ...>
    private Action<KernelConfig,
                   ArrayView1D<float, Stride1D.Dense>,
                   ArrayView1D<float, Stride1D.Dense>,
                   ArrayView1D<float, Stride1D.Dense>,
                   int, int, int, int>?  // M, K, N, numTilesN
        _matMulKernel;

    private Action<KernelConfig,
                   ArrayView1D<float, Stride1D.Dense>,
                   ArrayView1D<float, Stride1D.Dense>,
                   ArrayView1D<float, Stride1D.Dense>,
                   int, int, int, int>?  // M, K, N, numTilesN
        _batchedMatMulKernel;

    public MatMulKernel(Accelerator accelerator, bool forceSimpleKernels = false)
    {
        _accelerator = accelerator;
        // CPU backend has max group dim of 16 per axis — can't do 256-thread tiled kernels
        // WebGL has no shared memory — must use simple kernels
        // forceSimpleKernels: bypass tiled kernels for debugging (test if tiled MatMul is the bug)
        _useSimpleKernels = forceSimpleKernels || accelerator.MaxNumThreadsPerGroup < TILE * TILE;
    }

    // ─────────────────────────────────────────────────────────────
    //  GPU Kernels (1D shared memory, manual 2D indexing)
    // ─────────────────────────────────────────────────────────────

    /// <summary>
    /// Tiled matrix multiply: C[M,N] = A[M,K] × B[K,N].
    /// Each workgroup computes one TILE×TILE block of C.
    /// Uses 1D shared memory with manual row-major indexing.
    /// </summary>
    /// <summary>
    /// Tiled matrix multiply: C[M,N] = A[M,K] × B[K,N].
    /// Uses 1D grid + 1D group (256 threads) to avoid 2D index mapping issues on WebGPU.
    /// Each workgroup computes one TILE×TILE output tile.
    /// numTilesN passed as parameter for 2D tile index derivation from 1D grid.
    /// </summary>
    /// <summary>
    /// Inner k-tile accumulation, extracted as a helper. No explicit MethodImpl
    /// attribute - JIT decides per call site. Direct attempt with NoInlining
    /// produced a 11.6x first-compile win on Wasm (4611ms -> 396ms for the qkv
    /// MatMul) but tripped a WGSL validation error on WebGPU at the Gemm
    /// dispatch path (the fn-definition path bug flagged in
    /// feedback_methodimpl_inlining_directives.md). Without the attribute the
    /// helper still gets emitted as a separate method body that the JIT may
    /// elect not to inline at large call sites - giving up the explicit lock
    /// but keeping the structural separation that helps compile time on
    /// backends where the JIT chooses fn-call form. Re-evaluate adding back
    /// NoInlining when Geordi closes the WGSL fn-definition path.
    /// </summary>
    private static float TiledMatMulInnerAccumulate(
        ArrayView1D<float, Stride1D.Dense> aTile,
        ArrayView1D<float, Stride1D.Dense> bTile,
        int aRowOffset, int ty, float sum)
    {
        for (int k = 0; k < TILE; k++)
            sum += aTile[aRowOffset + k] * bTile[k * TILE + ty];
        return sum;
    }

    private static void TiledMatMulImpl(
        ArrayView1D<float, Stride1D.Dense> A,
        ArrayView1D<float, Stride1D.Dense> B,
        ArrayView1D<float, Stride1D.Dense> C,
        int M, int K, int N,
        int numTilesN)
    {
        var aTile = SharedMemory.Allocate<float>(TILE * TILE);
        var bTile = SharedMemory.Allocate<float>(TILE * TILE);

        // 1D grid → 2D tile index
        int tileIdx = Grid.IdxX;
        int tileRow = tileIdx / numTilesN;
        int tileCol = tileIdx % numTilesN;

        // 1D group (256 threads) → 2D local index
        int localIdx = Group.IdxX;
        int tx = localIdx / TILE;  // row within tile (0..15)
        int ty = localIdx % TILE;  // col within tile (0..15)

        int row = tileRow * TILE + tx;
        int col = tileCol * TILE + ty;
        int txTimesT = tx * TILE;

        float sum = 0f;

        int numKTiles = (K + TILE - 1) / TILE;
        for (int t = 0; t < numKTiles; t++)
        {
            int aCol = t * TILE + ty;
            aTile[txTimesT + ty] = (row < M && aCol < K) ? A[row * K + aCol] : 0f;

            int bRow = t * TILE + tx;
            bTile[txTimesT + ty] = (bRow < K && col < N) ? B[bRow * N + col] : 0f;

            Group.Barrier();

            sum = TiledMatMulInnerAccumulate(aTile, bTile, txTimesT, ty, sum);

            Group.Barrier();
        }

        if (row < M && col < N)
            C[row * N + col] = sum;
    }

    /// <summary>
    /// Batched tiled matrix multiply: C[b,M,N] = A[b,M,K] × B[b,K,N].
    /// Batch index from grid Z dimension.
    /// </summary>
    /// <summary>
    /// Batched tiled matrix multiply. Batch index from Grid.IdxY (1D grid per batch).
    /// </summary>
    private static void BatchedTiledMatMulImpl(
        ArrayView1D<float, Stride1D.Dense> A,
        ArrayView1D<float, Stride1D.Dense> B,
        ArrayView1D<float, Stride1D.Dense> C,
        int M, int K, int N,
        int numTilesN)
    {
        var aTile = SharedMemory.Allocate<float>(TILE * TILE);
        var bTile = SharedMemory.Allocate<float>(TILE * TILE);

        int batch = Grid.IdxY;
        int tileIdx = Grid.IdxX;
        int tileRow = tileIdx / numTilesN;
        int tileCol = tileIdx % numTilesN;

        int localIdx = Group.IdxX;
        int tx = localIdx / TILE;
        int ty = localIdx % TILE;

        int row = tileRow * TILE + tx;
        int col = tileCol * TILE + ty;
        int txTimesT = tx * TILE;

        int aOffset = batch * M * K;
        int bOffset = batch * K * N;
        int cOffset = batch * M * N;

        float sum = 0f;

        int numKTiles = (K + TILE - 1) / TILE;
        for (int t = 0; t < numKTiles; t++)
        {
            int aCol = t * TILE + ty;
            aTile[txTimesT + ty] = (row < M && aCol < K) ? A[aOffset + row * K + aCol] : 0f;

            int bRow = t * TILE + tx;
            bTile[txTimesT + ty] = (bRow < K && col < N) ? B[bOffset + bRow * N + col] : 0f;

            Group.Barrier();

            // Reuses TiledMatMulInnerAccumulate - same helper definition shared with the
            // non-batched TiledMatMulImpl. Per the no-attribute compile bisect (2026-05-05
            // commit b54f983), JIT-decides on this helper gives ~4x first-compile win
            // on Wasm without breaking WebGPU's WGSL fn-definition path.
            sum = TiledMatMulInnerAccumulate(aTile, bTile, txTimesT, ty, sum);

            Group.Barrier();
        }

        if (row < M && col < N)
            C[cOffset + row * N + col] = sum;
    }

    // ─────────────────────────────────────────────────────────────
    //  Simple (non-tiled) kernels — no shared memory, uses LoadAutoGroupedStreamKernel
    //  Avoids WGSL redeclaration bug in LoadStreamKernel
    // ─────────────────────────────────────────────────────────────

    /// <summary>Simple MatMul: one thread per output element. No shared memory.</summary>
    private static void SimpleMatMulImpl(
        Index1D idx,
        ArrayView1D<float, Stride1D.Dense> A,
        ArrayView1D<float, Stride1D.Dense> B,
        ArrayView1D<float, Stride1D.Dense> C,
        int M, int K, int N)
    {
        int col = idx % N;
        int row = idx / N;
        if (row >= M) return;
        float sum = 0f;
        for (int k = 0; k < K; k++)
            sum += A[row * K + k] * B[k * N + col];
        C[idx] = sum;
    }

    /// <summary>Simple batched MatMul: one thread per output element across all batches.</summary>
    private static void SimpleBatchedMatMulImpl(
        Index1D idx,
        ArrayView1D<float, Stride1D.Dense> A,
        ArrayView1D<float, Stride1D.Dense> B,
        ArrayView1D<float, Stride1D.Dense> C,
        int batchSize, int M, int K, int N)
    {
        int elementsPerBatch = M * N;
        int batch = idx / elementsPerBatch;
        int local = idx % elementsPerBatch;
        int col = local % N;
        int row = local / N;
        if (batch >= batchSize || row >= M) return;
        int aOff = batch * M * K;
        int bOff = batch * K * N;
        float sum = 0f;
        for (int k = 0; k < K; k++)
            sum += A[aOff + row * K + k] * B[bOff + k * N + col];
        C[idx] = sum;
    }

    /// <summary>
    /// Simple MatMul with fp16 (ILGPU.Half) weights in B: C[fp32] = A[fp32] × B[fp16]. Upconverts each
    /// weight to float and accumulates in fp32 — ORT-style mixed precision: HALF the weight memory, no
    /// accuracy loss vs the all-fp32 kernel. One thread per output element, no shared memory (works on
    /// every backend incl. WGSL; the f16 spike proved ILGPU.Half storage + fp32 compute everywhere).
    /// </summary>
    private static void SimpleMatMulLowPWeightImpl<T>(
        Index1D idx,
        ArrayView1D<float, Stride1D.Dense> A,
        ArrayView1D<T, Stride1D.Dense> B,
        ArrayView1D<float, Stride1D.Dense> C,
        int M, int K, int N)
        where T : unmanaged, INumber<T>
    {
        int col = idx % N;
        int row = idx / N;
        if (row >= M) return;
        float sum = 0f;
        for (int k = 0; k < K; k++)
            sum += A[row * K + k] * PrecisionConvert.ConvertToSingle(B[k * N + col]);
        C[idx] = sum;
    }

    /// <summary>Batched MatMul with native low-precision weights in B (ILGPU.Half / BFloat16 / Float8E*) —
    /// e.g. SD-Turbo attention projections (2D weight, rank-3 activation -> batched path; batch=1 so bOff=0
    /// reads the shared weight). Converts each weight to float in-register (PrecisionConvert), fp32 accumulate.
    /// Mirrors SimpleBatchedMatMulImpl.</summary>
    private static void SimpleBatchedMatMulLowPWeightImpl<T>(
        Index1D idx,
        ArrayView1D<float, Stride1D.Dense> A,
        ArrayView1D<T, Stride1D.Dense> B,
        ArrayView1D<float, Stride1D.Dense> C,
        int batchSize, int M, int K, int N)
        where T : unmanaged, INumber<T>
    {
        int elementsPerBatch = M * N;
        int batch = idx / elementsPerBatch;
        int local = idx % elementsPerBatch;
        int col = local % N;
        int row = local / N;
        if (batch >= batchSize || row >= M) return;
        int aOff = batch * M * K;
        int bOff = batch * K * N;
        float sum = 0f;
        for (int k = 0; k < K; k++)
            sum += A[aOff + row * K + k] * PrecisionConvert.ConvertToSingle(B[bOff + k * N + col]);
        C[idx] = sum;
    }

    /// <summary>
    /// MatMul with a TRANSPOSED native low-precision weight: C[M,N] = A[M,K] (fp32) × B[N,K]^T, i.e. B is
    /// stored row-major [N,K] (the ONNX Gemm <c>transB=1</c> layout that Linear/Dense layers export). Reads
    /// B row <c>n</c> contiguously (coalesced) and converts each element to float in-register via
    /// PrecisionConvert - so the weight stays native (no f32 transpose temp; Rule 4 zero-copy). fp32 accumulate.
    /// Mirror of <see cref="SimpleMatMulLowPWeightImpl{T}"/> with B indexed [n,k] instead of [k,n].
    /// </summary>
    private static void SimpleMatMulLowPWeightTransBImpl<T>(
        Index1D idx,
        ArrayView1D<float, Stride1D.Dense> A,
        ArrayView1D<T, Stride1D.Dense> B,
        ArrayView1D<float, Stride1D.Dense> C,
        int M, int K, int N)
        where T : unmanaged, INumber<T>
    {
        int col = idx % N;
        int row = idx / N;
        if (row >= M) return;
        float sum = 0f;
        for (int k = 0; k < K; k++)
            sum += A[row * K + k] * PrecisionConvert.ConvertToSingle(B[col * K + k]);
        C[idx] = sum;
    }

    private readonly Dictionary<Type, object> _simpleMatMulLowPWeightTransBKernels = new();

    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, int, int, int>? _simpleMatMulKernel;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, int, int, int, int>? _simpleBatchedMatMulKernel;
    // One compiled low-p-weight kernel per concrete weight type T (Half / BFloat16 / Float8E*), cached.
    // object-typed because each delegate is T-specific; lazily loaded on first use of that type.
    private readonly Dictionary<Type, object> _simpleMatMulLowPWeightKernels = new();
    private readonly Dictionary<Type, object> _simpleBatchedMatMulLowPWeightKernels = new();

    // ─────────────────────────────────────────────────────────────
    //  Public API
    // ─────────────────────────────────────────────────────────────

    /// <summary>
    /// Matrix multiply: C = A × B. All buffers are flat row-major.
    /// A[M,K] × B[K,N] → C[M,N].
    /// Auto-selects register-blocked path (4×4 per thread, 64×64 tiles) for large matrices
    /// when the hardware supports it (≥256 threads/group). Falls back to 16×16 tiled or simple.
    /// </summary>
    public void MatMul(
        ArrayView1D<float, Stride1D.Dense> A,
        ArrayView1D<float, Stride1D.Dense> B,
        ArrayView1D<float, Stride1D.Dense> C,
        int M, int K, int N)
    {
        var accelerator = _accelerator;
        EnsureKernelsLoaded(accelerator);

        if (_useSimpleKernels || M == 1)
        {
            // GEMV (M == 1, the LLM decode shape): the simple thread-per-output kernel
            // IS the right GEMV for row-major B[K,N] - consecutive threads n read
            // consecutive B addresses (coalesced) and broadcast-read A[k]. The 16x16
            // tiled kernel pads the single row to a 16-row tile (15/16 of every
            // 256-thread group idle), and its shared-memory staging buys nothing at
            // M = 1: there is no A-row reuse and GEMV touches each B element once.
            _simpleMatMulKernel!(M * N, A, B, C, M, K, N);
        }
        else if (M >= REG_TILE && N >= REG_TILE)
        {
            // Large matrices: use register-blocked path (16 results/thread vs 1)
            _regBlockedMatMul ??= new RegisterBlockedMatMul(accelerator);
            _regBlockedMatMul.MatMul(A, B, C, M, K, N);
        }
        else
        {
            int numTilesM = (M + TILE - 1) / TILE;
            int numTilesN = (N + TILE - 1) / TILE;
            int totalTiles = numTilesM * numTilesN;
            _matMulKernel!(new KernelConfig(totalTiles, TILE * TILE), A, B, C, M, K, N, numTilesN);
        }
    }

    /// <summary>
    /// Matrix multiply with fp16 weights: C[M,N] = A[M,K] (fp32) × B[K,N] (fp16 ILGPU.Half). Simple
    /// (non-tiled) path — proves the f16-weight pipeline; tiled/register-blocked fp16 variants follow.
    /// Half the weight memory, fp32 accumulate (no accuracy loss vs the all-fp32 MatMul).
    /// </summary>
    public void MatMulHalfWeight(
        ArrayView1D<float, Stride1D.Dense> A,
        ArrayView1D<global::ILGPU.Half, Stride1D.Dense> B,
        ArrayView1D<float, Stride1D.Dense> C,
        int M, int K, int N)
        => MatMulLowPWeight(A, B, C, M, K, N);

    /// <summary>
    /// Matrix multiply with NATIVE low-precision weights: C[M,N] = A[M,K] (fp32) × B[K,N] (low-p
    /// <typeparamref name="T"/> = ILGPU.Half / BFloat16 / Float8E*). The weight stays native in GPU memory
    /// (no f32 temp buffer); each element is converted to float in-register via PrecisionConvert and the
    /// product accumulated in fp32 (no accuracy loss vs the all-fp32 MatMul). Simple (non-tiled) path.
    /// </summary>
    public void MatMulLowPWeight<T>(
        ArrayView1D<float, Stride1D.Dense> A,
        ArrayView1D<T, Stride1D.Dense> B,
        ArrayView1D<float, Stride1D.Dense> C,
        int M, int K, int N)
        where T : unmanaged, INumber<T>
    {
        // Large matrices: register-blocked low-p path (16 results/thread, weight decoded once on the shared-mem
        // load) — the tiled throughput the per-element kernel forfeited. M==1 (the LLM-decode GEMV) and small /
        // CPU / WebGL (no 256-thread group) fall through to the simple per-element kernel below.
        if (!_useSimpleKernels && M >= REG_TILE && N >= REG_TILE)
        {
            _regBlockedMatMul ??= new RegisterBlockedMatMul(_accelerator);
            _regBlockedMatMul.MatMulLowPWeight(A, B, C, M, K, N);
            return;
        }
        if (!_simpleMatMulLowPWeightKernels.TryGetValue(typeof(T), out var k))
            _simpleMatMulLowPWeightKernels[typeof(T)] = k = _accelerator.LoadAutoGroupedStreamKernel<Index1D,
                ArrayView1D<float, Stride1D.Dense>, ArrayView1D<T, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>, int, int, int>(SimpleMatMulLowPWeightImpl<T>);
        ((Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<T, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, int, int, int>)k)(M * N, A, B, C, M, K, N);
    }

    /// <summary>
    /// Matrix multiply with a TRANSPOSED native low-precision weight: C[M,N] = A[M,K] (fp32) × B[N,K]^T
    /// (B stored row-major [N,K] = ONNX Gemm <c>transB=1</c>). The weight stays native (no f32 transpose
    /// temp); each element converted to float in-register via PrecisionConvert, fp32 accumulate. Simple
    /// (non-tiled) path - the same simple-kernel basis as <see cref="MatMulLowPWeight{T}"/>.
    /// </summary>
    public void MatMulLowPWeightTransB<T>(
        ArrayView1D<float, Stride1D.Dense> A,
        ArrayView1D<T, Stride1D.Dense> B,
        ArrayView1D<float, Stride1D.Dense> C,
        int M, int K, int N)
        where T : unmanaged, INumber<T>
    {
        if (!_simpleMatMulLowPWeightTransBKernels.TryGetValue(typeof(T), out var k))
            _simpleMatMulLowPWeightTransBKernels[typeof(T)] = k = _accelerator.LoadAutoGroupedStreamKernel<Index1D,
                ArrayView1D<float, Stride1D.Dense>, ArrayView1D<T, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>, int, int, int>(SimpleMatMulLowPWeightTransBImpl<T>);
        ((Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<T, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, int, int, int>)k)(M * N, A, B, C, M, K, N);
    }

    /// <summary>
    /// Batched matrix multiply with fp16 (ILGPU.Half) weights: C[b] = A[b] × B (fp16), fp32 accumulate.
    /// For SD-Turbo attention (2D weight shared across batch=1). Simple (non-tiled) path.
    /// </summary>
    public void BatchedMatMulHalfWeight(
        ArrayView1D<float, Stride1D.Dense> A,
        ArrayView1D<global::ILGPU.Half, Stride1D.Dense> B,
        ArrayView1D<float, Stride1D.Dense> C,
        int batchSize, int M, int K, int N)
        => BatchedMatMulLowPWeight(A, B, C, batchSize, M, K, N);

    /// <summary>
    /// Batched matrix multiply with NATIVE low-precision weights: C[b] = A[b] × B (low-p
    /// <typeparamref name="T"/>), fp32 accumulate. For SD-Turbo attention (2D weight shared across batch=1).
    /// Weight stays native; converted in-register via PrecisionConvert. Simple (non-tiled) path.
    /// </summary>
    public void BatchedMatMulLowPWeight<T>(
        ArrayView1D<float, Stride1D.Dense> A,
        ArrayView1D<T, Stride1D.Dense> B,
        ArrayView1D<float, Stride1D.Dense> C,
        int batchSize, int M, int K, int N)
        where T : unmanaged, INumber<T>
    {
        if (!_simpleBatchedMatMulLowPWeightKernels.TryGetValue(typeof(T), out var k))
            _simpleBatchedMatMulLowPWeightKernels[typeof(T)] = k = _accelerator.LoadAutoGroupedStreamKernel<Index1D,
                ArrayView1D<float, Stride1D.Dense>, ArrayView1D<T, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>, int, int, int, int>(SimpleBatchedMatMulLowPWeightImpl<T>);
        ((Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<T, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, int, int, int, int>)k)(batchSize * M * N, A, B, C, batchSize, M, K, N);
    }

    /// <summary>
    /// Batched matrix multiply: C[b] = A[b] × B[b] for b in [0, batchSize).
    /// </summary>
    public void BatchedMatMul(
        ArrayView1D<float, Stride1D.Dense> A,
        ArrayView1D<float, Stride1D.Dense> B,
        ArrayView1D<float, Stride1D.Dense> C,
        int batchSize, int M, int K, int N)
    {
        var accelerator = _accelerator;
        EnsureKernelsLoaded(accelerator);

        if (_useSimpleKernels || M == 1)
        {
            // Batched GEMV (M == 1 per batch entry = decode-time attention, one row of
            // Q against K^T/V per head): same routing rationale as MatMul - the tiled
            // kernel pads each batch entry's single row to a 16-row tile (15/16 idle).
            _simpleBatchedMatMulKernel!(batchSize * M * N, A, B, C, batchSize, M, K, N);
        }
        else if (M >= REG_TILE && N >= REG_TILE)
        {
            // Large batched matrices (prefill-shaped attention scores/probs·V, einsum contractions):
            // register-blocked path, same gate as the non-batched MatMul above. Before this route existed,
            // batched matmuls could ONLY hit the 16×16 one-result-per-thread tiled kernel - the ~100 GFLOPS
            // floor the DAv3 per-op profile exposed as the dominant cost.
            _regBlockedMatMul ??= new RegisterBlockedMatMul(accelerator);
            _regBlockedMatMul.BatchedMatMul(A, B, C, batchSize, M, K, N);
        }
        else
        {
            int numTilesM = (M + TILE - 1) / TILE;
            int numTilesN = (N + TILE - 1) / TILE;
            int totalTiles = numTilesM * numTilesN;
            var gridDim = new Index2D(totalTiles, batchSize);
            var groupDim = new Index2D(TILE * TILE, 1);
            _batchedMatMulKernel!(new KernelConfig(gridDim, groupDim), A, B, C, M, K, N, numTilesN);
        }
    }

    private void EnsureKernelsLoaded(Accelerator accelerator)
    {
        _simpleMatMulKernel ??= accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, int, int, int>(SimpleMatMulImpl);
        _simpleBatchedMatMulKernel ??= accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, int, int, int, int>(SimpleBatchedMatMulImpl);
        // Low-p-weight kernels are lazy per concrete T (see MatMulLowPWeight / BatchedMatMulLowPWeight).

        _matMulKernel ??= accelerator.LoadStreamKernel<
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            int, int, int, int>(TiledMatMulImpl);

        _batchedMatMulKernel ??= accelerator.LoadStreamKernel<
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            int, int, int, int>(BatchedTiledMatMulImpl);
    }

    // ─────────────────────────────────────────────────────────────
    //  Validation
    // ─────────────────────────────────────────────────────────────

    /// <summary>
    /// Validate with known data: A=all 1s, B=all 1s → C[i,j] should equal K.
    /// Prints the first few output values for debugging.
    /// </summary>
    public async Task ValidateKnownAsync(int M, int K, int N)
    {
        
        var accelerator = _accelerator;
        EnsureKernelsLoaded(accelerator);

        var aData = new float[M * K];
        var bData = new float[K * N];
        System.Array.Fill(aData, 1f);
        System.Array.Fill(bData, 1f);

        using var aBuf = accelerator.Allocate1D(aData);
        using var bBuf = accelerator.Allocate1D(bData);
        using var cBuf = accelerator.Allocate1D<float>(M * N);

        MatMul(aBuf.View, bBuf.View, cBuf.View, M, K, N);
        await accelerator.SynchronizeAsync();

        var gpuC = await cBuf.CopyToHostAsync<float>(0, M * N);

        // Print first row (cols 0..N-1) — should all be K
        var firstRow = string.Join(", ", gpuC.Take(Math.Min(N, 40)).Select(v => v.ToString("F1")));
        if (InferenceSession.VerboseLogging) Console.WriteLine($"[MatMul] Known {M}x{K}x{K}x{N}: first row (expect all {K}.0): [{firstRow}]");

        // Check row 0 col 0 vs col 16
        if (N > 16)
            if (InferenceSession.VerboseLogging) Console.WriteLine($"[MatMul]   C[0,0]={gpuC[0]:F1}, C[0,16]={gpuC[16]:F1}, C[0,31]={gpuC[Math.Min(31, N - 1)]:F1}");
    }

    /// <summary>
    /// GPU-only benchmark: runs MatMul at full size, reports timing. No CPU reference.
    /// </summary>
    public async Task BenchmarkAsync(int M, int K, int N, int warmup = 2, int runs = 5)
    {
        
        var accelerator = _accelerator;
        EnsureKernelsLoaded(accelerator);

        var rng = new Random(42);
        var aData = new float[M * K];
        var bData = new float[K * N];
        for (int i = 0; i < aData.Length; i++) aData[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < bData.Length; i++) bData[i] = (float)(rng.NextDouble() * 2 - 1);

        using var aBuf = accelerator.Allocate1D(aData);
        using var bBuf = accelerator.Allocate1D(bData);
        using var cBuf = accelerator.Allocate1D<float>(M * N);

        // Warmup
        for (int i = 0; i < warmup; i++)
        {
            MatMul(aBuf.View, bBuf.View, cBuf.View, M, K, N);
            await accelerator.SynchronizeAsync();
        }

        // Timed runs
        var sw = System.Diagnostics.Stopwatch.StartNew();
        for (int i = 0; i < runs; i++)
        {
            MatMul(aBuf.View, bBuf.View, cBuf.View, M, K, N);
            await accelerator.SynchronizeAsync();
        }
        sw.Stop();

        double avgMs = sw.Elapsed.TotalMilliseconds / runs;
        double gflops = 2.0 * M * K * N / (avgMs * 1e6); // 2*M*K*N FLOPs for matmul
        if (InferenceSession.VerboseLogging) Console.WriteLine($"[MatMul] Benchmark {M}x{K} x {K}x{N}: {avgMs:F1}ms avg, {gflops:F1} GFLOPS");
    }

    /// <summary>
    /// Validate the MatMul kernel against CPU reference.
    /// Returns (maxError, avgError).
    /// </summary>
    public async Task<(float maxError, float avgError)> ValidateAsync(
        int M = 384, int K = 1536, int N = 384, float tolerance = 1e-3f)
    {
        
        var accelerator = _accelerator;
        EnsureKernelsLoaded(accelerator);

        var rng = new Random(42);
        var aData = new float[M * K];
        var bData = new float[K * N];
        for (int i = 0; i < aData.Length; i++) aData[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < bData.Length; i++) bData[i] = (float)(rng.NextDouble() * 2 - 1);

        // CPU reference
        var cpuC = new float[M * N];
        for (int r = 0; r < M; r++)
            for (int c = 0; c < N; c++)
            {
                float s = 0;
                for (int k = 0; k < K; k++)
                    s += aData[r * K + k] * bData[k * N + c];
                cpuC[r * N + c] = s;
            }

        // GPU compute
        using var aBuf = accelerator.Allocate1D(aData);
        using var bBuf = accelerator.Allocate1D(bData);
        using var cBuf = accelerator.Allocate1D<float>(M * N);

        MatMul(aBuf.View, bBuf.View, cBuf.View, M, K, N);
        await accelerator.SynchronizeAsync();

        var gpuC = await cBuf.CopyToHostAsync<float>(0, M * N);

        float maxErr = 0f, sumErr = 0f;
        for (int i = 0; i < cpuC.Length; i++)
        {
            float err = MathF.Abs(cpuC[i] - gpuC[i]);
            if (err > maxErr) maxErr = err;
            sumErr += err;
        }
        float avgErr = sumErr / cpuC.Length;

        if (InferenceSession.VerboseLogging) Console.WriteLine($"[MatMul] Validate {M}x{K} x {K}x{N}: maxErr={maxErr:E3}, avgErr={avgErr:E3}");

        if (maxErr > tolerance)
            if (InferenceSession.VerboseLogging) Console.WriteLine($"[MatMul] WARNING: maxErr {maxErr:E3} exceeds tolerance {tolerance:E3}!");

        return (maxErr, avgErr);
    }
}
