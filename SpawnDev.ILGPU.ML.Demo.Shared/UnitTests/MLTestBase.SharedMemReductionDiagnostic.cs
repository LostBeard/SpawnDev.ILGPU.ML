using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// INVESTIGATION HARNESS (not a PMT test): isolates the SHARED-MEMORY TREE REDUCTION — the one
/// cross-thread mechanism the decode-path GEMV (FusedDequantMatMul) uses, and the CPU analog of the
/// Wasm "write shared memory, read back stale, retry until it sticks" visibility bug.
///
/// Each group of G=64 threads loads 64 distinct values (tid+1, sum 1..64 = 2080, float-exact) into
/// shared memory and tree-reduces them — the exact pattern as GemvDequant*Impl. Any missing fence /
/// stale cross-thread read corrupts the sum to something != 2080, caught immediately. Fast: one tiny
/// kernel dispatch + a 4096-float readback per rep (no GGUF, no sessions), so thousands of reps run in
/// seconds. KVRACE_BURN spins background CPU burners to perturb scheduling like a loaded PMT run.
/// </summary>
public partial class MLTestBase
{
    private const int ShRedGroup = 64;

    private static void SharedMemReduceImpl(
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output)
    {
        int g = Grid.IdxX;
        int tid = Group.IdxX;
        var sh = SharedMemory.Allocate<float>(ShRedGroup);
        sh[tid] = input[g * ShRedGroup + tid];
        Group.Barrier();
        for (int stride = ShRedGroup / 2; stride > 0; stride >>= 1)
        {
            if (tid < stride) sh[tid] += sh[tid + stride];
            Group.Barrier();
        }
        if (tid == 0) output[g] = sh[0];
    }

    public async Task DiagnoseSharedMemReduction(int reps)
    {
        int numGroups = int.TryParse(Environment.GetEnvironmentVariable("KVRACE_GROUPS"), out var gg) ? gg : 256;
        const int G = ShRedGroup;
        const float expected = G * (G + 1) / 2f; // sum 1..64 = 2080, exact in float

        int burn = int.TryParse(Environment.GetEnvironmentVariable("KVRACE_BURN"), out var bb) ? bb : 0;
        using var burnCts = new System.Threading.CancellationTokenSource();
        var burners = new List<Task>();
        for (int b = 0; b < burn; b++)
            burners.Add(Task.Run(() => { double x = 1.0001; var ct = burnCts.Token; while (!ct.IsCancellationRequested) { x = Math.Sin(x) + 1.0001; } GC.KeepAlive(x); }));

        var (context, acc) = await CreateAcceleratorAsync();
        Console.WriteLine($"[ShMemRed:{BackendName}] reps={reps}, groups={numGroups}, G={G}, expected={expected}, burners={burn}, procs={Environment.ProcessorCount}");

        var input = new float[(long)numGroups * G];
        for (int g = 0; g < numGroups; g++)
            for (int t = 0; t < G; t++)
                input[g * G + t] = t + 1;

        int badReps = 0; long badElems = 0; string firstBad = "";
        try
        {
            using var inBuf = acc.Allocate1D(input);
            using var outBuf = acc.Allocate1D<float>(numGroups);
            var kernel = acc.LoadStreamKernel<ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(SharedMemReduceImpl);

            for (int r = 0; r < reps; r++)
            {
                kernel(new KernelConfig(numGroups, G), inBuf.View, outBuf.View);
                await acc.SynchronizeAsync();
                var outArr = await outBuf.CopyToHostAsync<float>(0, numGroups);
                bool repBad = false;
                for (int g = 0; g < numGroups; g++)
                {
                    if (outArr[g] != expected)
                    {
                        badElems++; repBad = true;
                        if (firstBad == "") firstBad = $"rep{r} group{g}: got {outArr[g]} want {expected}";
                    }
                }
                if (repBad) { badReps++; Console.WriteLine($"  BAD rep{r}: {CountBad(outArr, expected)} groups wrong (first: {FirstBad(outArr, expected)})"); }
                if (r > 0 && r % 500 == 0)
                    Console.WriteLine($"  ...{r}/{reps} (badReps={badReps} badElems={badElems})");
            }
        }
        finally
        {
            try { await acc.SynchronizeAsync(); } catch { }
            burnCts.Cancel();
            try { await Task.WhenAll(burners); } catch { }
            try { acc.Dispose(); } catch { }
            try { context.Dispose(); } catch { }
        }

        Console.WriteLine($"[ShMemRed:{BackendName}] DONE. badReps={badReps}/{reps}, badElems={badElems}. {(firstBad == "" ? "" : "first: " + firstBad)}");
        Console.WriteLine(badReps == 0
            ? "[ShMemRed] reduction DETERMINISTIC + correct this run."
            : "[ShMemRed] SHARED-MEMORY REDUCTION NON-DETERMINISTIC — fencing/visibility bug in CPU group barrier / shared memory.");
    }

    private static int CountBad(float[] a, float expected)
    { int c = 0; foreach (var v in a) if (v != expected) c++; return c; }
    private static string FirstBad(float[] a, float expected)
    { for (int i = 0; i < a.Length; i++) if (a[i] != expected) return $"group{i}={a[i]}"; return ""; }
}
