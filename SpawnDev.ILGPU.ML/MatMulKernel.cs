using ILGPU;
using ILGPU.Runtime;

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

    private readonly Accelerator _accelerator;
    private readonly bool _useSimpleKernels; // CPU/WebGL can't handle 256-thread groups

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

        float sum = 0f;

        int numKTiles = (K + TILE - 1) / TILE;
        for (int t = 0; t < numKTiles; t++)
        {
            int aCol = t * TILE + ty;
            aTile[tx * TILE + ty] = (row < M && aCol < K) ? A[row * K + aCol] : 0f;

            int bRow = t * TILE + tx;
            bTile[tx * TILE + ty] = (bRow < K && col < N) ? B[bRow * N + col] : 0f;

            Group.Barrier();

            for (int k = 0; k < TILE; k++)
                sum += aTile[tx * TILE + k] * bTile[k * TILE + ty];

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

        int aOffset = batch * M * K;
        int bOffset = batch * K * N;
        int cOffset = batch * M * N;

        float sum = 0f;

        int numKTiles = (K + TILE - 1) / TILE;
        for (int t = 0; t < numKTiles; t++)
        {
            int aCol = t * TILE + ty;
            aTile[tx * TILE + ty] = (row < M && aCol < K) ? A[aOffset + row * K + aCol] : 0f;

            int bRow = t * TILE + tx;
            bTile[tx * TILE + ty] = (bRow < K && col < N) ? B[bOffset + bRow * N + col] : 0f;

            Group.Barrier();

            for (int k = 0; k < TILE; k++)
                sum += aTile[tx * TILE + k] * bTile[k * TILE + ty];

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

    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, int, int, int>? _simpleMatMulKernel;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, int, int, int, int>? _simpleBatchedMatMulKernel;

    // ─────────────────────────────────────────────────────────────
    //  Public API
    // ─────────────────────────────────────────────────────────────

    /// <summary>
    /// Matrix multiply: C = A × B. All buffers are flat row-major.
    /// A[M,K] × B[K,N] → C[M,N].
    /// </summary>
    public void MatMul(
        ArrayView1D<float, Stride1D.Dense> A,
        ArrayView1D<float, Stride1D.Dense> B,
        ArrayView1D<float, Stride1D.Dense> C,
        int M, int K, int N)
    {
        var accelerator = _accelerator;
        EnsureKernelsLoaded(accelerator);

        if (_useSimpleKernels)
        {
            _simpleMatMulKernel!(M * N, A, B, C, M, K, N);
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

        // WORKAROUND: Use simple kernel until SpawnDev.ILGPU 4.4.1+ with 2D grid fix.
        // The tiled version works correctly with the fix (committed in SpawnDev.ILGPU aeeb457).
        // Re-enable tiled batched MatMul after updating SpawnDev.ILGPU NuGet.
        _simpleBatchedMatMulKernel!(batchSize * M * N, A, B, C, batchSize, M, K, N);
    }

    private void EnsureKernelsLoaded(Accelerator accelerator)
    {
        _simpleMatMulKernel ??= accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, int, int, int>(SimpleMatMulImpl);
        _simpleBatchedMatMulKernel ??= accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, int, int, int, int>(SimpleBatchedMatMulImpl);

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
        Console.WriteLine($"[MatMul] Known {M}x{K}x{K}x{N}: first row (expect all {K}.0): [{firstRow}]");

        // Check row 0 col 0 vs col 16
        if (N > 16)
            Console.WriteLine($"[MatMul]   C[0,0]={gpuC[0]:F1}, C[0,16]={gpuC[16]:F1}, C[0,31]={gpuC[Math.Min(31, N - 1)]:F1}");
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
        Console.WriteLine($"[MatMul] Benchmark {M}x{K} x {K}x{N}: {avgMs:F1}ms avg, {gflops:F1} GFLOPS");
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

        Console.WriteLine($"[MatMul] Validate {M}x{K} x {K}x{N}: maxErr={maxErr:E3}, avgErr={avgErr:E3}");

        if (maxErr > tolerance)
            Console.WriteLine($"[MatMul] WARNING: maxErr {maxErr:E3} exceeds tolerance {tolerance:E3}!");

        return (maxErr, avgErr);
    }
}
