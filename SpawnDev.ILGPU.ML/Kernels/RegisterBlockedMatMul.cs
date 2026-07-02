using ILGPU;
using ILGPU.Runtime;
using System.Numerics;

namespace SpawnDev.ILGPU.ML.Kernels;

/// <summary>
/// Register-blocked tiled matrix multiplication.
/// Each thread computes a 4x4 block of output elements using data cached in registers.
/// 16x16 thread block = 256 threads (WebGPU max), each computing 4x4 = 16 results.
/// Output tile: 64x64 (16 threads × 4 results per thread in each dimension).
///
/// vs. current TiledMatMul: 16x16 tile, 1 result per thread = 256 results per workgroup
/// vs. this: 64x64 tile, 16 results per thread = 4096 results per workgroup (16x improvement)
///
/// The register blocking multiplies arithmetic intensity: each shared memory load
/// feeds 4x as many computations, hiding memory latency and boosting throughput.
///
/// Target: 200+ GFLOPS (current tiled: 92-101 GFLOPS)
/// </summary>
public class RegisterBlockedMatMul
{
    private const int BLOCK = 16;   // Thread block: 16x16 = 256 threads
    private const int REG = 4;      // Each thread computes REG x REG output elements
    private const int TILE = BLOCK * REG; // Output tile: 64x64

    private readonly Accelerator _accelerator;
    private Action<KernelConfig, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, int, int, int, int>? _kernel;

    public RegisterBlockedMatMul(Accelerator accelerator) => _accelerator = accelerator;

    /// <summary>
    /// C[M,N] = A[M,K] × B[K,N]
    /// Uses register blocking for maximum throughput on large matrices.
    /// Falls back to simple kernel for matrices smaller than one tile.
    /// </summary>
    public void MatMul(
        ArrayView1D<float, Stride1D.Dense> A,
        ArrayView1D<float, Stride1D.Dense> B,
        ArrayView1D<float, Stride1D.Dense> C,
        int M, int K, int N)
    {
        // For small matrices, register blocking overhead isn't worth it
        if (M < TILE || N < TILE || _accelerator.MaxNumThreadsPerGroup < BLOCK * BLOCK)
        {
            // Fall back to simple per-element kernel
            var simple = _accelerator.LoadAutoGroupedStreamKernel<Index1D,
                ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>, int, int, int>(SimpleMatMulImpl);
            simple(M * N, A, B, C, M, K, N);
            return;
        }

        int numTilesM = (M + TILE - 1) / TILE;
        int numTilesN = (N + TILE - 1) / TILE;
        int totalTiles = numTilesM * numTilesN;

        _kernel ??= _accelerator.LoadStreamKernel<ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            int, int, int, int>(RegBlockedImpl);

        var config = new KernelConfig(
            new Index1D(totalTiles),      // grid: one workgroup per output tile
            new Index1D(BLOCK * BLOCK));  // group: 256 threads

        _kernel(config, A, B, C, M, K, N, numTilesN);
    }

    /// <summary>
    /// Register-blocked tiled MatMul kernel.
    /// Each thread:
    ///   - Participates in loading BLOCK-wide strips of A and B into shared memory
    ///   - Computes REG×REG output elements using register accumulation
    ///   - Writes REG×REG results to global memory
    /// </summary>
    private static void RegBlockedImpl(
        ArrayView1D<float, Stride1D.Dense> A,
        ArrayView1D<float, Stride1D.Dense> B,
        ArrayView1D<float, Stride1D.Dense> C,
        int M, int K, int N, int numTilesN)
    {
        // Shared memory tiles: TILE rows × BLOCK columns
        var aTile = SharedMemory.Allocate<float>(TILE * BLOCK);
        var bTile = SharedMemory.Allocate<float>(BLOCK * TILE);

        // 1D grid → 2D tile index
        int tileIdx = Grid.IdxX;
        int tileRow = tileIdx / numTilesN;
        int tileCol = tileIdx % numTilesN;

        // Thread position within block
        int localIdx = Group.IdxX;
        int threadRow = localIdx / BLOCK; // 0..15
        int threadCol = localIdx % BLOCK; // 0..15

        // This thread computes output elements at:
        // rows: tileRow * TILE + threadRow * REG + (0..REG-1)
        // cols: tileCol * TILE + threadCol * REG + (0..REG-1)

        // Register accumulators: REG × REG
        float c00 = 0, c01 = 0, c02 = 0, c03 = 0;
        float c10 = 0, c11 = 0, c12 = 0, c13 = 0;
        float c20 = 0, c21 = 0, c22 = 0, c23 = 0;
        float c30 = 0, c31 = 0, c32 = 0, c33 = 0;

        int numKTiles = (K + BLOCK - 1) / BLOCK;

        for (int t = 0; t < numKTiles; t++)
        {
            // Collaboratively load A tile: TILE rows × BLOCK columns
            // Each of 256 threads loads REG elements (covers TILE rows)
            for (int r = 0; r < REG; r++)
            {
                int aRow = tileRow * TILE + threadRow * REG + r;
                int aCol = t * BLOCK + threadCol;
                int sIdx = (threadRow * REG + r) * BLOCK + threadCol;
                aTile[sIdx] = (aRow < M && aCol < K) ? A[aRow * K + aCol] : 0f;
            }

            // Collaboratively load B tile: BLOCK rows × TILE columns
            for (int r = 0; r < REG; r++)
            {
                int bRow = t * BLOCK + threadRow;
                int bCol = tileCol * TILE + threadCol * REG + r;
                int sIdx = threadRow * TILE + threadCol * REG + r;
                bTile[sIdx] = (bRow < K && bCol < N) ? B[bRow * N + bCol] : 0f;
            }

            Group.Barrier();

            // Compute REG×REG output block using data from shared memory
            for (int k = 0; k < BLOCK; k++)
            {
                // Load REG values from A tile column k
                float a0 = aTile[(threadRow * REG + 0) * BLOCK + k];
                float a1 = aTile[(threadRow * REG + 1) * BLOCK + k];
                float a2 = aTile[(threadRow * REG + 2) * BLOCK + k];
                float a3 = aTile[(threadRow * REG + 3) * BLOCK + k];

                // Load REG values from B tile row k
                float b0 = bTile[k * TILE + threadCol * REG + 0];
                float b1 = bTile[k * TILE + threadCol * REG + 1];
                float b2 = bTile[k * TILE + threadCol * REG + 2];
                float b3 = bTile[k * TILE + threadCol * REG + 3];

                // 16 multiply-adds — all from registers, no memory access
                c00 += a0 * b0; c01 += a0 * b1; c02 += a0 * b2; c03 += a0 * b3;
                c10 += a1 * b0; c11 += a1 * b1; c12 += a1 * b2; c13 += a1 * b3;
                c20 += a2 * b0; c21 += a2 * b1; c22 += a2 * b2; c23 += a2 * b3;
                c30 += a3 * b0; c31 += a3 * b1; c32 += a3 * b2; c33 += a3 * b3;
            }

            Group.Barrier();
        }

        // Write REG×REG results to global memory
        int baseRow = tileRow * TILE + threadRow * REG;
        int baseCol = tileCol * TILE + threadCol * REG;

        if (baseRow + 0 < M && baseCol + 0 < N) C[(baseRow + 0) * N + baseCol + 0] = c00;
        if (baseRow + 0 < M && baseCol + 1 < N) C[(baseRow + 0) * N + baseCol + 1] = c01;
        if (baseRow + 0 < M && baseCol + 2 < N) C[(baseRow + 0) * N + baseCol + 2] = c02;
        if (baseRow + 0 < M && baseCol + 3 < N) C[(baseRow + 0) * N + baseCol + 3] = c03;

        if (baseRow + 1 < M && baseCol + 0 < N) C[(baseRow + 1) * N + baseCol + 0] = c10;
        if (baseRow + 1 < M && baseCol + 1 < N) C[(baseRow + 1) * N + baseCol + 1] = c11;
        if (baseRow + 1 < M && baseCol + 2 < N) C[(baseRow + 1) * N + baseCol + 2] = c12;
        if (baseRow + 1 < M && baseCol + 3 < N) C[(baseRow + 1) * N + baseCol + 3] = c13;

        if (baseRow + 2 < M && baseCol + 0 < N) C[(baseRow + 2) * N + baseCol + 0] = c20;
        if (baseRow + 2 < M && baseCol + 1 < N) C[(baseRow + 2) * N + baseCol + 1] = c21;
        if (baseRow + 2 < M && baseCol + 2 < N) C[(baseRow + 2) * N + baseCol + 2] = c22;
        if (baseRow + 2 < M && baseCol + 3 < N) C[(baseRow + 2) * N + baseCol + 3] = c23;

        if (baseRow + 3 < M && baseCol + 0 < N) C[(baseRow + 3) * N + baseCol + 0] = c30;
        if (baseRow + 3 < M && baseCol + 1 < N) C[(baseRow + 3) * N + baseCol + 1] = c31;
        if (baseRow + 3 < M && baseCol + 2 < N) C[(baseRow + 3) * N + baseCol + 2] = c32;
        if (baseRow + 3 < M && baseCol + 3 < N) C[(baseRow + 3) * N + baseCol + 3] = c33;
    }

    private static void SimpleMatMulImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> A,
        ArrayView1D<float, Stride1D.Dense> B,
        ArrayView1D<float, Stride1D.Dense> C,
        int M, int K, int N)
    {
        int row = idx / N;
        int col = idx % N;
        float sum = 0f;
        for (int k = 0; k < K; k++)
            sum += A[row * K + k] * B[k * N + col];
        C[idx] = sum;
    }

    // ── Batched variant: C[b] = A[b] × B[b], register-blocked ──
    // The batched matmuls are attention's scores (Q·Kᵀ) and probs·V — until this entry existed they could
    // only run on the 16×16 one-result-per-thread tiled kernel (MatMulKernel.BatchedMatMul had no
    // register-blocked route), which is exactly the ~100 GFLOPS-class floor the DAv3 profile exposed.
    // Same 64×64-tile / 4×4-register core as RegBlockedImpl; batch index from Grid.IdxY, flat per-batch
    // offsets. Grid stays (tiles, batch) 2D — the layout the existing batched tiled kernel already uses
    // on every backend including WebGPU.

    private Action<KernelConfig, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, int, int, int, int>? _batchedKernel;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, int, int, int, int>? _simpleBatchedKernel;

    /// <summary>
    /// Batched C[b,M,N] = A[b,M,K] × B[b,K,N] for b in [0, batchSize), register-blocked (4×4 per thread,
    /// 64×64 output tiles). Small matrices and sub-256-thread hardware fall back to the simple
    /// one-thread-per-output batched kernel.
    /// </summary>
    public void BatchedMatMul(
        ArrayView1D<float, Stride1D.Dense> A,
        ArrayView1D<float, Stride1D.Dense> B,
        ArrayView1D<float, Stride1D.Dense> C,
        int batchSize, int M, int K, int N)
    {
        if (M < TILE || N < TILE || _accelerator.MaxNumThreadsPerGroup < BLOCK * BLOCK)
        {
            _simpleBatchedKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
                ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>, int, int, int, int>(SimpleBatchedMatMulImpl);
            _simpleBatchedKernel(batchSize * M * N, A, B, C, M, K, N, batchSize);
            return;
        }

        int numTilesM = (M + TILE - 1) / TILE;
        int numTilesN = (N + TILE - 1) / TILE;

        _batchedKernel ??= _accelerator.LoadStreamKernel<ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            int, int, int, int>(RegBlockedBatchedImpl);

        var config = new KernelConfig(
            new Index2D(numTilesM * numTilesN, batchSize),
            new Index2D(BLOCK * BLOCK, 1));
        _batchedKernel(config, A, B, C, M, K, N, numTilesN);
    }

    // Identical to RegBlockedImpl except the Grid.IdxY batch index offsets A/B/C by whole matrices.
    private static void RegBlockedBatchedImpl(
        ArrayView1D<float, Stride1D.Dense> A,
        ArrayView1D<float, Stride1D.Dense> B,
        ArrayView1D<float, Stride1D.Dense> C,
        int M, int K, int N, int numTilesN)
    {
        var aTile = SharedMemory.Allocate<float>(TILE * BLOCK);
        var bTile = SharedMemory.Allocate<float>(BLOCK * TILE);

        int batch = Grid.IdxY;
        int aOff = batch * M * K;
        int bOff = batch * K * N;
        int cOff = batch * M * N;

        int tileIdx = Grid.IdxX;
        int tileRow = tileIdx / numTilesN;
        int tileCol = tileIdx % numTilesN;
        int localIdx = Group.IdxX;
        int threadRow = localIdx / BLOCK;
        int threadCol = localIdx % BLOCK;

        float c00 = 0, c01 = 0, c02 = 0, c03 = 0;
        float c10 = 0, c11 = 0, c12 = 0, c13 = 0;
        float c20 = 0, c21 = 0, c22 = 0, c23 = 0;
        float c30 = 0, c31 = 0, c32 = 0, c33 = 0;

        int numKTiles = (K + BLOCK - 1) / BLOCK;
        for (int t = 0; t < numKTiles; t++)
        {
            for (int r = 0; r < REG; r++)
            {
                int aRow = tileRow * TILE + threadRow * REG + r;
                int aCol = t * BLOCK + threadCol;
                int sIdx = (threadRow * REG + r) * BLOCK + threadCol;
                aTile[sIdx] = (aRow < M && aCol < K) ? A[aOff + aRow * K + aCol] : 0f;
            }
            for (int r = 0; r < REG; r++)
            {
                int bRow = t * BLOCK + threadRow;
                int bCol = tileCol * TILE + threadCol * REG + r;
                int sIdx = threadRow * TILE + threadCol * REG + r;
                bTile[sIdx] = (bRow < K && bCol < N) ? B[bOff + bRow * N + bCol] : 0f;
            }

            Group.Barrier();

            for (int k = 0; k < BLOCK; k++)
            {
                float a0 = aTile[(threadRow * REG + 0) * BLOCK + k];
                float a1 = aTile[(threadRow * REG + 1) * BLOCK + k];
                float a2 = aTile[(threadRow * REG + 2) * BLOCK + k];
                float a3 = aTile[(threadRow * REG + 3) * BLOCK + k];
                float b0 = bTile[k * TILE + threadCol * REG + 0];
                float b1 = bTile[k * TILE + threadCol * REG + 1];
                float b2 = bTile[k * TILE + threadCol * REG + 2];
                float b3 = bTile[k * TILE + threadCol * REG + 3];
                c00 += a0 * b0; c01 += a0 * b1; c02 += a0 * b2; c03 += a0 * b3;
                c10 += a1 * b0; c11 += a1 * b1; c12 += a1 * b2; c13 += a1 * b3;
                c20 += a2 * b0; c21 += a2 * b1; c22 += a2 * b2; c23 += a2 * b3;
                c30 += a3 * b0; c31 += a3 * b1; c32 += a3 * b2; c33 += a3 * b3;
            }

            Group.Barrier();
        }

        int baseRow = tileRow * TILE + threadRow * REG;
        int baseCol = tileCol * TILE + threadCol * REG;
        if (baseRow + 0 < M && baseCol + 0 < N) C[cOff + (baseRow + 0) * N + baseCol + 0] = c00;
        if (baseRow + 0 < M && baseCol + 1 < N) C[cOff + (baseRow + 0) * N + baseCol + 1] = c01;
        if (baseRow + 0 < M && baseCol + 2 < N) C[cOff + (baseRow + 0) * N + baseCol + 2] = c02;
        if (baseRow + 0 < M && baseCol + 3 < N) C[cOff + (baseRow + 0) * N + baseCol + 3] = c03;
        if (baseRow + 1 < M && baseCol + 0 < N) C[cOff + (baseRow + 1) * N + baseCol + 0] = c10;
        if (baseRow + 1 < M && baseCol + 1 < N) C[cOff + (baseRow + 1) * N + baseCol + 1] = c11;
        if (baseRow + 1 < M && baseCol + 2 < N) C[cOff + (baseRow + 1) * N + baseCol + 2] = c12;
        if (baseRow + 1 < M && baseCol + 3 < N) C[cOff + (baseRow + 1) * N + baseCol + 3] = c13;
        if (baseRow + 2 < M && baseCol + 0 < N) C[cOff + (baseRow + 2) * N + baseCol + 0] = c20;
        if (baseRow + 2 < M && baseCol + 1 < N) C[cOff + (baseRow + 2) * N + baseCol + 1] = c21;
        if (baseRow + 2 < M && baseCol + 2 < N) C[cOff + (baseRow + 2) * N + baseCol + 2] = c22;
        if (baseRow + 2 < M && baseCol + 3 < N) C[cOff + (baseRow + 2) * N + baseCol + 3] = c23;
        if (baseRow + 3 < M && baseCol + 0 < N) C[cOff + (baseRow + 3) * N + baseCol + 0] = c30;
        if (baseRow + 3 < M && baseCol + 1 < N) C[cOff + (baseRow + 3) * N + baseCol + 1] = c31;
        if (baseRow + 3 < M && baseCol + 2 < N) C[cOff + (baseRow + 3) * N + baseCol + 2] = c32;
        if (baseRow + 3 < M && baseCol + 3 < N) C[cOff + (baseRow + 3) * N + baseCol + 3] = c33;
    }

    private static void SimpleBatchedMatMulImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> A,
        ArrayView1D<float, Stride1D.Dense> B,
        ArrayView1D<float, Stride1D.Dense> C,
        int M, int K, int N, int batchSize)
    {
        int elementsPerBatch = M * N;
        int batch = idx / elementsPerBatch;
        if (batch >= batchSize) return;
        int local = idx % elementsPerBatch;
        int row = local / N;
        int col = local % N;
        int aOff = batch * M * K;
        int bOff = batch * K * N;
        float sum = 0f;
        for (int k = 0; k < K; k++)
            sum += A[aOff + row * K + k] * B[bOff + k * N + col];
        C[idx] = sum;
    }

    // ── Native low-precision weight B (ILGPU.Half / BFloat16 / Float8E*) ──
    // Same 64×64-tile / 4×4-register GEMM, but B is read in its native low-p type and converted to float ONCE
    // as it stages into the float shared-memory tile (PrecisionConvert) — so the decode is amortized over the
    // 4× register reuse and the hot register-block math is byte-identical to the f32 kernel. The weight stays
    // native in GPU memory (no f32 temp); fp32 accumulate. This is the tiled counterpart to
    // MatMulKernel.SimpleMatMulLowPWeightImpl — the throughput fp16/bf16 SD weights forfeited on the per-element
    // kernel. One compiled kernel per concrete T, cached.
    private readonly Dictionary<Type, object> _lowPKernels = new();
    private readonly Dictionary<Type, object> _simpleLowPKernels = new();

    /// <summary>C[M,N] = A[M,K] (fp32) × B[K,N] (native low-p <typeparamref name="T"/>), register-blocked.
    /// Decodes each weight to float in-register as it stages into shared memory. Falls back to a simple low-p
    /// kernel below one tile (the large-matrix gate normally lives in the caller).</summary>
    public void MatMulLowPWeight<T>(
        ArrayView1D<float, Stride1D.Dense> A,
        ArrayView1D<T, Stride1D.Dense> B,
        ArrayView1D<float, Stride1D.Dense> C,
        int M, int K, int N)
        where T : unmanaged, INumber<T>
    {
        if (M < TILE || N < TILE || _accelerator.MaxNumThreadsPerGroup < BLOCK * BLOCK)
        {
            if (!_simpleLowPKernels.TryGetValue(typeof(T), out var sk))
                _simpleLowPKernels[typeof(T)] = sk = _accelerator.LoadAutoGroupedStreamKernel<Index1D,
                    ArrayView1D<float, Stride1D.Dense>, ArrayView1D<T, Stride1D.Dense>,
                    ArrayView1D<float, Stride1D.Dense>, int, int, int>(SimpleMatMulLowPImpl<T>);
            ((Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<T, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>, int, int, int>)sk)(M * N, A, B, C, M, K, N);
            return;
        }

        int numTilesM = (M + TILE - 1) / TILE;
        int numTilesN = (N + TILE - 1) / TILE;
        if (!_lowPKernels.TryGetValue(typeof(T), out var k))
            _lowPKernels[typeof(T)] = k = _accelerator.LoadStreamKernel<
                ArrayView1D<float, Stride1D.Dense>, ArrayView1D<T, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>, int, int, int, int>(RegBlockedLowPImpl<T>);
        var config = new KernelConfig(new Index1D(numTilesM * numTilesN), new Index1D(BLOCK * BLOCK));
        ((Action<KernelConfig, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<T, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, int, int, int, int>)k)(config, A, B, C, M, K, N, numTilesN);
    }

    // Register-blocked GEMM with a native low-p weight B. Identical to RegBlockedImpl except B is read as T and
    // decoded to float as it enters the (float) shared tile — the ONE line that differs.
    private static void RegBlockedLowPImpl<T>(
        ArrayView1D<float, Stride1D.Dense> A,
        ArrayView1D<T, Stride1D.Dense> B,
        ArrayView1D<float, Stride1D.Dense> C,
        int M, int K, int N, int numTilesN)
        where T : unmanaged, INumber<T>
    {
        var aTile = SharedMemory.Allocate<float>(TILE * BLOCK);
        var bTile = SharedMemory.Allocate<float>(BLOCK * TILE);

        int tileIdx = Grid.IdxX;
        int tileRow = tileIdx / numTilesN;
        int tileCol = tileIdx % numTilesN;
        int localIdx = Group.IdxX;
        int threadRow = localIdx / BLOCK;
        int threadCol = localIdx % BLOCK;

        float c00 = 0, c01 = 0, c02 = 0, c03 = 0;
        float c10 = 0, c11 = 0, c12 = 0, c13 = 0;
        float c20 = 0, c21 = 0, c22 = 0, c23 = 0;
        float c30 = 0, c31 = 0, c32 = 0, c33 = 0;

        int numKTiles = (K + BLOCK - 1) / BLOCK;
        for (int t = 0; t < numKTiles; t++)
        {
            for (int r = 0; r < REG; r++)
            {
                int aRow = tileRow * TILE + threadRow * REG + r;
                int aCol = t * BLOCK + threadCol;
                int sIdx = (threadRow * REG + r) * BLOCK + threadCol;
                aTile[sIdx] = (aRow < M && aCol < K) ? A[aRow * K + aCol] : 0f;
            }
            for (int r = 0; r < REG; r++)
            {
                int bRow = t * BLOCK + threadRow;
                int bCol = tileCol * TILE + threadCol * REG + r;
                int sIdx = threadRow * TILE + threadCol * REG + r;
                // Decode the native low-p weight to float ONCE as it enters shared memory (amortized over reuse).
                bTile[sIdx] = (bRow < K && bCol < N) ? PrecisionConvert.ConvertToSingle(B[bRow * N + bCol]) : 0f;
            }

            Group.Barrier();

            for (int k = 0; k < BLOCK; k++)
            {
                float a0 = aTile[(threadRow * REG + 0) * BLOCK + k];
                float a1 = aTile[(threadRow * REG + 1) * BLOCK + k];
                float a2 = aTile[(threadRow * REG + 2) * BLOCK + k];
                float a3 = aTile[(threadRow * REG + 3) * BLOCK + k];
                float b0 = bTile[k * TILE + threadCol * REG + 0];
                float b1 = bTile[k * TILE + threadCol * REG + 1];
                float b2 = bTile[k * TILE + threadCol * REG + 2];
                float b3 = bTile[k * TILE + threadCol * REG + 3];
                c00 += a0 * b0; c01 += a0 * b1; c02 += a0 * b2; c03 += a0 * b3;
                c10 += a1 * b0; c11 += a1 * b1; c12 += a1 * b2; c13 += a1 * b3;
                c20 += a2 * b0; c21 += a2 * b1; c22 += a2 * b2; c23 += a2 * b3;
                c30 += a3 * b0; c31 += a3 * b1; c32 += a3 * b2; c33 += a3 * b3;
            }

            Group.Barrier();
        }

        int baseRow = tileRow * TILE + threadRow * REG;
        int baseCol = tileCol * TILE + threadCol * REG;
        if (baseRow + 0 < M && baseCol + 0 < N) C[(baseRow + 0) * N + baseCol + 0] = c00;
        if (baseRow + 0 < M && baseCol + 1 < N) C[(baseRow + 0) * N + baseCol + 1] = c01;
        if (baseRow + 0 < M && baseCol + 2 < N) C[(baseRow + 0) * N + baseCol + 2] = c02;
        if (baseRow + 0 < M && baseCol + 3 < N) C[(baseRow + 0) * N + baseCol + 3] = c03;
        if (baseRow + 1 < M && baseCol + 0 < N) C[(baseRow + 1) * N + baseCol + 0] = c10;
        if (baseRow + 1 < M && baseCol + 1 < N) C[(baseRow + 1) * N + baseCol + 1] = c11;
        if (baseRow + 1 < M && baseCol + 2 < N) C[(baseRow + 1) * N + baseCol + 2] = c12;
        if (baseRow + 1 < M && baseCol + 3 < N) C[(baseRow + 1) * N + baseCol + 3] = c13;
        if (baseRow + 2 < M && baseCol + 0 < N) C[(baseRow + 2) * N + baseCol + 0] = c20;
        if (baseRow + 2 < M && baseCol + 1 < N) C[(baseRow + 2) * N + baseCol + 1] = c21;
        if (baseRow + 2 < M && baseCol + 2 < N) C[(baseRow + 2) * N + baseCol + 2] = c22;
        if (baseRow + 2 < M && baseCol + 3 < N) C[(baseRow + 2) * N + baseCol + 3] = c23;
        if (baseRow + 3 < M && baseCol + 0 < N) C[(baseRow + 3) * N + baseCol + 0] = c30;
        if (baseRow + 3 < M && baseCol + 1 < N) C[(baseRow + 3) * N + baseCol + 1] = c31;
        if (baseRow + 3 < M && baseCol + 2 < N) C[(baseRow + 3) * N + baseCol + 2] = c32;
        if (baseRow + 3 < M && baseCol + 3 < N) C[(baseRow + 3) * N + baseCol + 3] = c33;
    }

    private static void SimpleMatMulLowPImpl<T>(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> A,
        ArrayView1D<T, Stride1D.Dense> B,
        ArrayView1D<float, Stride1D.Dense> C,
        int M, int K, int N)
        where T : unmanaged, INumber<T>
    {
        int row = idx / N;
        int col = idx % N;
        float sum = 0f;
        for (int k = 0; k < K; k++)
            sum += A[row * K + k] * PrecisionConvert.ConvertToSingle(B[k * N + col]);
        C[idx] = sum;
    }
}
