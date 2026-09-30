using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Kernels;

/// <summary>
/// General ONNX Einsum for one or two inputs on the accelerator: one thread per output element decodes its output-label
/// values, then sums the product of the inputs over every combination of the contracted labels. Covers outer products
/// (<c>i,j-&gt;ij</c>), contractions, transposes, repeated labels (diagonals: a label's stride is the SUM of the strides
/// of every position it occupies) and reductions (<c>ij-&gt;i</c>).
/// </summary>
/// <remarks>
/// Replaces the CPU contraction for everything the matmul / broadcast fast paths do not claim. MEASURED 2026-09-30,
/// DAv3 Small 518 on WebGPU: its 26 outer-product einsums (<c>i,j-&gt;ij</c>, <c>m,d-&gt;md</c>, RoPE positions x
/// frequencies) ran the CPU loop at ~70 ms each - 1.8 s of a 3.2 s uncaptured forward - and uploaded the result.
/// <para>
/// Label table (ints, one write-once device buffer per distinct content - see <see cref="ContentParamBuffers{T}"/>):
/// <c>[nOut, nContracted, (size, strideA, strideB) per output label in output order, (size, strideA, strideB) per
/// contracted label]</c>. A label absent from an input has stride 0 there.
/// </para>
/// </remarks>
public sealed class EinsumKernel : IDisposable
{
    readonly Accelerator _accelerator;
    Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, int>? _kernel;
    ContentParamBuffers<int>? _tables;
    MemoryBuffer1D<float, Stride1D.Dense>? _unitB;

    public EinsumKernel(Accelerator accelerator) => _accelerator = accelerator;

    /// <summary>
    /// output[o] = sum over contracted labels of a[..] * b[..] (b = 1 when <paramref name="b"/> is null).
    /// </summary>
    public void Run(ArrayView1D<float, Stride1D.Dense> a, ArrayView1D<float, Stride1D.Dense>? b,
        ArrayView1D<float, Stride1D.Dense> output, int outputCount, int[] table)
    {
        _kernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, int>(EinsumImpl);
        ArrayView1D<int, Stride1D.Dense> tableView = Graph.GraphExecutor.UseCaptureParamSlots
            ? CaptureParamArena.Shared(_accelerator).RentStableSlot(table)
            : (_tables ??= new ContentParamBuffers<int>(_accelerator)).Get(table);
        // A single-input einsum still binds a second buffer: WebGPU forbids binding one buffer to two read_write slots.
        _unitB ??= _accelerator.Allocate1D(new float[] { 1f });
        _kernel(new Index1D(outputCount), a, b ?? _unitB.View, output, tableView, b.HasValue ? 1 : 0);
    }

    static void EinsumImpl(Index1D o, ArrayView1D<float, Stride1D.Dense> a, ArrayView1D<float, Stride1D.Dense> b,
        ArrayView1D<float, Stride1D.Dense> output, ArrayView1D<int, Stride1D.Dense> table, int hasB)
    {
        int nOut = table[0];
        int nContracted = table[1];
        int rem = o;
        int offA = 0, offB = 0;
        for (int d = nOut - 1; d >= 0; d--)
        {
            int size = table[2 + 3 * d];
            int v = rem % size;
            rem /= size;
            offA += v * table[3 + 3 * d];
            offB += v * table[4 + 3 * d];
        }
        int cBase = 2 + 3 * nOut;
        int combos = 1;
        for (int c = 0; c < nContracted; c++) combos *= table[cBase + 3 * c];
        float sum = 0f;
        for (int ci = 0; ci < combos; ci++)
        {
            int r = ci;
            int ia = offA, ib = offB;
            for (int c = nContracted - 1; c >= 0; c--)
            {
                int size = table[cBase + 3 * c];
                int v = r % size;
                r /= size;
                ia += v * table[cBase + 3 * c + 1];
                ib += v * table[cBase + 3 * c + 2];
            }
            sum += a[ia] * (hasB != 0 ? b[ib] : 1f);
        }
        output[o] = sum;
    }

    /// <summary>
    /// Builds the label table for <paramref name="inputLabels"/> -&gt; <paramref name="outputLabels"/> over row-major inputs
    /// of the given shapes. Contracted labels are every input label missing from the output, in first-seen order.
    /// </summary>
    public static int[] BuildTable(char[][] inputLabels, int[][] inputShapes, char[] outputLabels, IReadOnlyDictionary<char, int> dimSizes)
    {
        int StrideOf(int input, char label)
        {
            if (input >= inputLabels.Length) return 0;
            var labels = inputLabels[input];
            var shape = inputShapes[input];
            int stride = 1, sum = 0;
            for (int p = labels.Length - 1; p >= 0; p--)
            {
                if (labels[p] == label) sum += stride;
                stride *= p < shape.Length ? shape[p] : 1;
            }
            return sum;
        }
        var contracted = new List<char>();
        foreach (var labels in inputLabels)
            foreach (var c in labels)
                if (Array.IndexOf(outputLabels, c) < 0 && !contracted.Contains(c)) contracted.Add(c);
        var table = new int[2 + 3 * (outputLabels.Length + contracted.Count)];
        table[0] = outputLabels.Length;
        table[1] = contracted.Count;
        int k = 2;
        foreach (var label in outputLabels.Concat(contracted))
        {
            table[k++] = dimSizes.TryGetValue(label, out var n) ? n : 1;
            table[k++] = StrideOf(0, label);
            table[k++] = StrideOf(1, label);
        }
        return table;
    }

    public void Dispose()
    {
        _tables?.Dispose(); _tables = null;
        _unitB?.Dispose(); _unitB = null;
    }
}
