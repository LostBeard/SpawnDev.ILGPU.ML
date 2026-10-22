using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Kernels;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    /// <summary>
    /// An embedding lookup at the CLIP text encoder's real size: a [49408, 1024] table (row r holds the value r in every column, so a wrong row shows as a wrong number), looked up by 77 token ids
    /// including 49406 (BOS), 320, 1125 and ids around powers of two, through both Gather kernels (float token ids, as the graph executor stores them, and int ids). Every returned row must be the
    /// requested one. Reported on an AMD RX 590 (OpenCL, SpawnDev.ILGPU.ML 5.4.0): the SD-Turbo text encoder's Gather returned row id + 8192 for every id but BOS, so the whole image was noise; this
    /// test shows the same thing without a model. Runs on every backend.
    /// </summary>
    [TestMethod(Timeout = 300000)]
    public async Task Gather_ClipSizedEmbeddingTable_ReturnsTheRequestedRows() => await RunTest(async accelerator =>
    {
        const int rows = 49408, inner = 1024;
        var table = new float[(long)rows * inner];
        for (int r = 0; r < rows; r++) Array.Fill(table, (float)r, r * inner, inner);
        var ids = new List<int> { 49406, 320, 1125, 539, 320, 736, 3055, 525, 0, 49407, 8191, 8192, 8193, 16384, 40000, 1, 2, 4096, 24576, 32768, 49152 };
        var rng = new Random(5);
        while (ids.Count < 77) ids.Add(rng.Next(rows));
        int n = ids.Count;
        using var tableBuf = accelerator.Allocate1D(table);
        using var idsInt = accelerator.Allocate1D(ids.ToArray());
        using var idsFloat = accelerator.Allocate1D(ids.Select(i => (float)i).ToArray());
        using var outFloat = accelerator.Allocate1D<float>((long)n * inner);
        using var outInt = accelerator.Allocate1D<float>((long)n * inner);
        var gather = new GatherKernel(accelerator);
        gather.GatherAxis0Float(tableBuf.View, idsFloat.View, outFloat.View, n, inner, rows);
        gather.GatherAxis0(tableBuf.View, idsInt.View, outInt.View, n, inner);
        var gotFloat = await outFloat.View.CopyToHostAsync(); var gotInt = await outInt.View.CopyToHostAsync();
        var bad = new List<string>();
        foreach (var (name, got) in new[] { ("float ids", gotFloat), ("int ids", gotInt) })
            for (int i = 0; i < n; i++)
                for (int c = 0; c < inner; c += 127)
                    if (got[(long)i * inner + c] != ids[i]) { bad.Add($"{name}: position {i}, id {ids[i]}, column {c}: row {got[(long)i * inner + c]}"); break; }
        if (bad.Count > 0) throw new Exception($"{bad.Count} wrong rows on {accelerator.AcceleratorType}: {string.Join("; ", bad.Take(8))}");
        return $"{n} ids, both kernels, every row correct on {accelerator.AcceleratorType}";
    });
}
