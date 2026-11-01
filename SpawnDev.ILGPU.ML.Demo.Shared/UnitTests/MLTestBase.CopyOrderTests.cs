using ILGPU;
using ILGPU.Runtime;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    /// <summary>
    /// A device-to-device copy sees the host writes made before it, on every backend: write buffer A from the host (and a sub-range of it), copy A into B
    /// (whole and into a sub-view at an offset, as <see cref="SpawnDev.ILGPU.ML.Retrieval.DeviceVectorIndex"/>'s growth does), read B. Found when the
    /// vector index's rows came out all zero on WebGL after it grew (2026-11-01).
    /// </summary>
    [TestMethod(Timeout = 120000)]
    public async Task DeviceCopy_SeesEarlierHostWrites() => await RunTest(async accelerator =>
    {
        var reports = new List<string>();
        foreach (int n in new[] { 4096, 8192 * 384 })      // small, and the vector index's growth copy (8192 rows of 384)
            reports.Add(await CopyOrderAt(accelerator, n));
        var all = string.Join(" | ", reports);
        Console.WriteLine("[CopyOrder] " + all);
        if (all.Contains("WRONG")) throw new Exception(all);
        return all;
    });

    private static async Task<string> CopyOrderAt(Accelerator accelerator, int N)
    {
        var data = Enumerable.Range(1, N).Select(i => (float)i).ToArray();
        using var a = accelerator.Allocate1D<float>(N);
        a.MemSetToZero();
        a.View.CopyFromCPU(data);
        a.View.SubView(100, 4).CopyFromCPU(new[] { -1f, -2f, -3f, -4f });
        using var b = accelerator.Allocate1D<float>(2 * N);
        b.MemSetToZero();
        b.View.SubView(N, N).CopyFrom(a.View);                    // into the second half, at an offset
        var got = await b.View.CopyToHostAsync();
        int wrong = 0, first = -1;
        for (int i = 0; i < N; i++)
        {
            float want = i >= 100 && i < 104 ? -(i - 99) : i + 1;
            if (got[N + i] != want) { wrong++; if (first < 0) first = i; }
            if (got[i] != 0f) { wrong++; if (first < 0) first = i; }
        }
        // a KERNEL reading the copy's destination must see what the copy wrote (a host read of it is not the same path on every backend)
        using var c = accelerator.Allocate1D<float>(N);
        var copyKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int>(CopyOrderReadImpl);
        copyKernel(N, b.View, c.View, N);
        var viaKernel = await c.View.CopyToHostAsync();
        int kernelWrong = 0, kFirst = -1;
        for (int i = 0; i < N; i++)
        {
            float want = i >= 100 && i < 104 ? -(i - 99) : i + 1;
            if (viaKernel[i] != want) { kernelWrong++; if (kFirst < 0) kFirst = i; }
        }
        wrong += kernelWrong;
        // a host write made AFTER the device copy, into another buffer, must reach the next kernel too (the vector index's query after its growth)
        using var d = accelerator.Allocate1D<float>(N);
        var later = Enumerable.Range(0, N).Select(i => 0.5f + i).ToArray();
        d.View.CopyFromCPU(later);
        copyKernel(N, d.View, c.View, 0);
        var afterCopy = await c.View.CopyToHostAsync();
        int lateWrong = 0;
        for (int i = 0; i < N; i++) if (afterCopy[i] != later[i]) lateWrong++;
        wrong += lateWrong;
        if (lateWrong > 0) kernelWrong += lateWrong;
        return $"{accelerator.AcceleratorType} {N} floats{(lateWrong > 0 ? $" [a host write after the copy: {lateWrong} WRONG, first {afterCopy.FirstOrDefault()}]" : "")}: {(wrong == 0 ? "the copy saw every host write, and a kernel reading its destination sees the copy" : $"{wrong} values WRONG (host read of the copy {wrong - kernelWrong}, a kernel reading it {kernelWrong}{(kFirst >= 0 ? $", first at {kFirst}: {viaKernel[kFirst]}" : "")})")}";
    }

    // c[i] = b[offset + i]
    private static void CopyOrderReadImpl(Index1D i, ArrayView1D<float, Stride1D.Dense> b, ArrayView1D<float, Stride1D.Dense> c, int offset) => c[i] = b[offset + (int)i];
}
