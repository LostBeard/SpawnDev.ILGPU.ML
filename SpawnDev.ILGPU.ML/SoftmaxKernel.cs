using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML;

/// <summary>
/// Row-wise Softmax using auto-grouped kernels (no shared memory).
/// Two passes:
///   Pass 1: One thread per row — sequential max + exp + sum.
///   Pass 2: One thread per element — normalize by row sum.
///
/// For attention (1369 tokens per row), sequential pass is ~4K FLOPs per row — fast enough.
/// Avoids WGSL variable redeclaration bug with multiple LoadStreamKernel calls.
///
/// Future home: SpawnDev.ILGPU.ML
/// </summary>
public class SoftmaxKernel
{
    private readonly Accelerator _accelerator;

    // Pass 1: compute per-row max and sum(exp) — no in-place data writes (TF compatible)
    private Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        int>?
        _softmaxStatsKernel;

    // Pass 2: compute exp(x-max)/sum per element — one write per thread (TF compatible)
    private Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        int, int>?
        _softmaxApplyKernel;

    // Persistent temp buffers (reused across calls, resized as needed)
    private MemoryBuffer1D<float, Stride1D.Dense>? _rowMaxesBuf;
    private MemoryBuffer1D<float, Stride1D.Dense>? _rowSumsBuf;
    private int _rowStatsCapacity;

    public SoftmaxKernel(Accelerator accelerator) => _accelerator = accelerator;

    /// <summary>
    /// Pass 1: One thread per row. Find max and compute sum(exp(x-max)).
    /// Writes exactly 2 values per thread (rowMaxes[row], rowSums[row]) — TF compatible.
    /// Does NOT write exp values to data (avoids multi-write-in-loop TF bug).
    /// </summary>
    private static void SoftmaxStatsImpl(
        Index1D row,
        ArrayView1D<float, Stride1D.Dense> data,
        ArrayView1D<float, Stride1D.Dense> rowMaxes,
        ArrayView1D<float, Stride1D.Dense> rowSums,
        int cols)
    {
        int offset = row * cols;

        // Find row max
        float max = data[offset];
        for (int i = 1; i < cols; i++)
        {
            float v = data[offset + i];
            if (v > max) max = v;
        }
        rowMaxes[row] = max;

        // Double-precision accumulation for exp sum
        double sum = 0.0;
        for (int i = 0; i < cols; i++)
            sum += (double)MathF.Exp(data[offset + i] - max);

        rowSums[row] = (float)sum;
    }

    /// <summary>
    /// Pass 2: One thread per element. Compute exp(x-max)/sum in-place.
    /// Writes exactly 1 value per thread — TF compatible.
    /// Recomputes exp per element (cheap, avoids pass 1 multi-write).
    /// </summary>
    private static void SoftmaxApplyImpl(
        Index1D idx,
        ArrayView1D<float, Stride1D.Dense> data,
        ArrayView1D<float, Stride1D.Dense> rowMaxes,
        ArrayView1D<float, Stride1D.Dense> rowSums,
        int cols, int rowOffset)
    {
        int localRow = idx / cols;
        int globalRow = localRow + rowOffset;
        data[idx] = MathF.Exp(data[idx] - rowMaxes[globalRow]) / rowSums[globalRow];
    }

    /// <summary>
    /// In-place row-wise softmax. data: [rows, cols] flat.
    /// Uses internal temp buffer for row sums.
    /// </summary>
    // WebGPU 1D dispatch limit: 65535 workgroups × 64 threads = 4,194,240 elements
    private const int MAX_DISPATCH = 65535 * 64;

    public void Forward(
        ArrayView1D<float, Stride1D.Dense> data,
        int rows, int cols)
    {
        var accelerator = _accelerator;
        EnsureLoaded(accelerator);

        // Reuse persistent temp buffers (don't allocate+dispose per call — GPU is async)
        if (_rowSumsBuf == null || _rowStatsCapacity < rows)
        {
            _rowMaxesBuf?.Dispose();
            _rowSumsBuf?.Dispose();
            _rowMaxesBuf = accelerator.Allocate1D<float>(rows);
            _rowSumsBuf = accelerator.Allocate1D<float>(rows);
            _rowStatsCapacity = rows;
        }

        // Pass 1: one thread per row — compute max and sum(exp) without modifying data
        _softmaxStatsKernel!(rows, data, _rowMaxesBuf.View, _rowSumsBuf.View, cols);

        // Pass 2: one thread per element — compute exp(x-max)/sum in-place
        int rowsPerBatch = MAX_DISPATCH / cols;
        if (rowsPerBatch < 1) rowsPerBatch = 1;
        for (int rowStart = 0; rowStart < rows; rowStart += rowsPerBatch)
        {
            int batchRows = Math.Min(rowsPerBatch, rows - rowStart);
            int count = batchRows * cols;
            _softmaxApplyKernel!(count, data.SubView(rowStart * cols, count), _rowMaxesBuf.View, _rowSumsBuf.View, cols, rowStart);
        }
    }

    private void EnsureLoaded(Accelerator accelerator)
    {
        _softmaxStatsKernel ??= accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            int>(SoftmaxStatsImpl);

        _softmaxApplyKernel ??= accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            int, int>(SoftmaxApplyImpl);
    }

    public async Task DiagnosticAsync()
    {
        
        var accelerator = _accelerator;
        EnsureLoaded(accelerator);

        var input = new float[] { 1f, 2f, 3f, 4f };
        using var dataBuf = accelerator.Allocate1D(input);
        Forward(dataBuf.View, 1, 4);
        await accelerator.SynchronizeAsync();
        var gpuOut = await dataBuf.CopyToHostAsync<float>(0, 4);
        Console.WriteLine($"[Softmax] Diagnostic [1,2,3,4]: [{string.Join(", ", gpuOut.Select(v => v.ToString("F4")))}] (expect [0.0321, 0.0871, 0.2369, 0.6439])");
    }

    public async Task<(float maxError, float avgError)> ValidateAsync(int rows = 96, int cols = 1369)
    {
        
        var accelerator = _accelerator;
        EnsureLoaded(accelerator);

        var rng = new Random(42);
        var inputData = new float[rows * cols];
        for (int i = 0; i < inputData.Length; i++) inputData[i] = (float)(rng.NextDouble() * 4 - 2);

        var cpuOut = new float[rows * cols];
        for (int r = 0; r < rows; r++)
        {
            float max = float.MinValue;
            for (int c = 0; c < cols; c++) max = MathF.Max(max, inputData[r * cols + c]);
            float sum = 0;
            for (int c = 0; c < cols; c++) { cpuOut[r * cols + c] = MathF.Exp(inputData[r * cols + c] - max); sum += cpuOut[r * cols + c]; }
            for (int c = 0; c < cols; c++) cpuOut[r * cols + c] /= sum;
        }

        var gpuInput = (float[])inputData.Clone();
        using var dataBuf = accelerator.Allocate1D(gpuInput);
        Forward(dataBuf.View, rows, cols);
        await accelerator.SynchronizeAsync();
        var gpuOut = await dataBuf.CopyToHostAsync<float>(0, rows * cols);

        float maxErr = 0f, sumErr = 0f;
        for (int i = 0; i < cpuOut.Length; i++)
        {
            float err = MathF.Abs(cpuOut[i] - gpuOut[i]);
            if (err > maxErr) maxErr = err;
            sumErr += err;
        }
        Console.WriteLine($"[Softmax] Validate {rows}x{cols}: maxErr={maxErr:E3}, avgErr={sumErr / cpuOut.Length:E3}");
        return (maxErr, sumErr / cpuOut.Length);
    }
}
