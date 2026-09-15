using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Kernels;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// INVESTIGATION HARNESS (not a PMT test): prices the InstanceNorm PASS 1 LAUNCH SHAPE.
///
/// 🔴 WHY IT EXISTS. On 2026-09-15 the overnight full sweep lost the whole WebGPU lane to one
/// <c>DXGI_ERROR_DEVICE_HUNG</c> raised by <c>InstanceNorm_StyleMosaicShape_MatchesCpu</c>
/// (N=1, C=32, spatial=50176) - 345 WebGPU failures, all downstream <c>DEVICE_REMOVED</c> from that
/// single device loss. A scoped re-run of the same test passed, so the hang is INTERMITTENT and a
/// pass/fail re-run can never settle it. What settles it is the DISPATCH DURATION: Pass 1
/// (<c>InstanceNormMeanVarImpl</c>) launches exactly N*C threads, each looping <c>spatial</c> twice
/// serially, so its wall time grows with <c>spatial</c> and is bounded by nothing. A dispatch that
/// drifts past the Windows TDR budget is removed by the OS, not by this engine.
///
/// A COUNT IS NOT A COST and neither is a hang: measure the milliseconds.
///
/// Decomposition, all through the PUBLIC kernel API:
///   - <c>InstanceNormPartialStats</c>  - ONE serial pass over spatial, one thread per slice. This is
///                                       the launch shape under suspicion, isolated.
///   - <c>InstanceNormApplyWithStats</c>- one thread per ELEMENT, no loop. The well-shaped sibling.
///   - <c>InstanceNorm</c>              - the production call (Pass 1 + Pass 2).
/// Pass 1 should read as roughly 2x the partial-stats row (it makes two passes over spatial).
///
///   dotnet run --project SpawnDev.ILGPU.ML.DemoConsole -c Release -- INORMBENCH [Cuda|OpenCL|CPU] [reps]
///
/// ⚠️ Release PUBLISH only for browser numbers (global Rule 5a). Desktop CUDA/OpenCL is the fast signal;
/// it shares the launch shape with WebGPU, which is where the device actually died.
/// </summary>
public partial class MLTestBase
{
    /// <summary>The shapes that matter: the two neighbours that bracket the overnight device loss, plus
    /// the scaling rows that show the cost is in <c>spatial</c> (the serial loop) and not in N*C.</summary>
    private static readonly (string Name, int N, int C, int Spatial)[] INormBenchShapes =
    {
        ("InstanceNorm_MatchesCpu            ", 1,  3,    64),   // passes: trivially small
        ("InstanceNorm_StyleTransferDims     ", 1,  3, 50176),   // passes: 3 threads  x 50176
        ("InstanceNorm_StyleMosaicShape  HUNG", 1, 32, 50176),   // DEVICE_HUNG overnight: 32 threads x 50176
        ("StyleMosaic quarter-res            ", 1, 32, 12544),   // same threads, 1/4 the serial loop
        ("SD VAE-ish                         ", 1, 64, 50176),   // more slices, same loop length
    };

    /// <summary>
    /// Prices InstanceNorm Pass 1 ON THE BACKEND THAT IS ACTUALLY RUNNING - every lane, browser included -
    /// and prints WHICH path it took.
    /// </summary>
    /// <remarks>
    /// 🔴 THE MISTAKE THIS EXISTS TO PREVENT. On 2026-09-15 I measured the cooperative Pass 1 at 39x on CUDA,
    /// watched two scoped gates go green, and reported the WebGPU device hang as fixed. The next gate hung the
    /// device again on the same test. **The card I measured was not the device that was dying**, and a green
    /// intermittent gate is not evidence - see [[fb-a-device-hang-is-one-event]]. A desktop benchmark cannot
    /// speak for a browser backend, which is the same lesson as "an engine benchmark is not a product
    /// measurement" one layer down.
    ///
    /// Read it with `PMT_CONSOLE_LOG=INormPath` - PMT summarises browser console lines away by default, which
    /// is exactly how a diagnostic runs every sweep and has its verdict discarded.
    ///
    /// Deliberately NOT an assertion on a DURATION: timings on a shared browser device are not stable enough
    /// to gate on, and a flaky red is worse than no gate.
    ///
    /// ⭐ But it does assert the PATH, because "which path ran" is the exact thing I assumed instead of
    /// checking, and an assumption is what this whole episode was made of. On any backend that has real
    /// groups, a 50,176-element slice MUST take the cooperative kernel; if `CoopStatsApplies` ever stops
    /// selecting it - a changed threshold, a misreported limit, a lost `AcceleratorType` case - the one-thread
    /// -per-slice kernel silently returns, correct and slow, and no correctness test anywhere would notice.
    /// WebGL is excluded by design (maxGroup=1, no shared memory in the TF path), so the assertion is keyed on
    /// the reported group size rather than on a backend name.
    /// </remarks>
    [TestMethod(Timeout = 120000)]
    public async Task InstanceNormPass1_LaunchShape_PerBackend() => await RunTest(async accelerator =>
    {
        var norm = new NormalizationKernels(accelerator);
        // The two shapes that bracket the device loss, and nothing heavier - this runs on every lane.
        foreach (var (name, N, C, spatial) in new[]
                 {
                     ("small   ", 1, 3, 64),
                     ("styleXfer", 1, 3, 50176),
                     ("styleMosaic", 1, 32, 50176),
                 })
        {
            int total = N * C * spatial;
            var input = new float[total];
            for (int i = 0; i < total; i++) input[i] = (i % 97) * 0.01f - 0.5f;
            var ones = new float[C]; for (int c = 0; c < C; c++) ones[c] = 1f;
            var zeros = new float[C];

            using var inBuf = accelerator.Allocate1D(input);
            using var outBuf = accelerator.Allocate1D<float>(total);
            using var sBuf = accelerator.Allocate1D(ones);
            using var bBuf = accelerator.Allocate1D(zeros);

            norm.InstanceNorm(inBuf.View, outBuf.View, sBuf.View, bBuf.View, N, C, spatial);
            await accelerator.SynchronizeAsync();
            bool coop = NormalizationKernels.LastStatsPathWasCooperative;
            int group = NormalizationKernels.LastStatsGroupSize;

            var sw = System.Diagnostics.Stopwatch.StartNew();
            const int reps = 5;
            for (int r = 0; r < reps; r++)
            {
                norm.InstanceNorm(inBuf.View, outBuf.View, sBuf.View, bBuf.View, N, C, spatial);
                await accelerator.SynchronizeAsync();
            }
            sw.Stop();

            Console.WriteLine($"[INormPath] {BackendName,-8} {name,-12} N={N} C={C} spatial={spatial} "
                            + $"path={(coop ? $"COOP(T={group})" : "SERIAL")} "
                            + $"maxGroup={accelerator.MaxNumThreadsPerGroup} "
                            + $"{sw.Elapsed.TotalMilliseconds / reps:F2} ms/call");

            // The invariant, per backend and per shape. Asserted as INTENT, not as a mirror of
            // CoopStatsApplies - a test that just restates the implementation cannot catch the
            // implementation changing.
            //
            //   Cuda / OpenCL / WebGPU - real GPUs, a long slice MUST divide across a group.
            //   CPU                    - MUST stay serial. ILGPU emulates groups with real threads and
            //                            barriers; MEASURED 7-13x WORSE cooperative. maxGroup is 64 there,
            //                            so a naive "has groups => must be cooperative" rule gets it wrong,
            //                            which is exactly the regression this pins.
            //   WebGL                  - MUST stay serial; it cannot run shared memory + barriers at all.
            //   Wasm                   - MUST stay serial. MEASURED both ways: coop 9.03 vs serial 7.30 and
            //                            coop 28.97 vs serial 29.95 ms/call - no gain, inside noise.
            //
            // The generalisation: the cooperative kernel wins where threads hide MEMORY LATENCY, i.e. a real
            // GPU. Where "threads" are OS threads or WASM workers, the barriers cost more than they buy.
            var t = accelerator.AcceleratorType;
            bool mustBeCooperative = t is AcceleratorType.Cuda or AcceleratorType.OpenCL or AcceleratorType.WebGPU;
            bool mustBeSerial = t is AcceleratorType.CPU or AcceleratorType.WebGL or AcceleratorType.Wasm;

            if (spatial >= 50176 && mustBeCooperative && !coop)
                throw new Exception(
                    $"{BackendName}: a {spatial}-element slice took the SERIAL one-thread-per-slice Pass 1 on a "
                  + $"real GPU backend (MaxNumThreadsPerGroup={accelerator.MaxNumThreadsPerGroup}). The "
                  + "cooperative path stopped being selected - correct but unboundedly slow, and no "
                  + "correctness test would catch it.");
            if (mustBeSerial && coop)
                throw new Exception(
                    $"{BackendName}: took the COOPERATIVE Pass 1, which is a deliberate exclusion here - "
                  + "WebGL cannot run it, and on the CPU accelerator it measured 7-13x SLOWER than the serial "
                  + "kernel because groups and barriers are emulated with real threads.");
        }
        norm.Dispose();
    });

    /// <summary>
    /// Prices <see cref="MLTestBase.AssertCloseGpu"/> itself against ELEMENT COUNT, per backend.
    /// </summary>
    /// <remarks>
    /// 🔴 WHY. `CompareOnGpuAsync`'s kernel does `Atomic.Add(ref results[0], ..)` and
    /// `Atomic.Max(ref results[1], ..)` - EVERY element contending on the SAME TWO ADDRESSES. That is a fully
    /// serialized reduction wearing the clothes of a parallel one, and its cost scales with element count.
    /// `InstanceNorm_StyleMosaicShape_MatchesCpu` compares 1,605,632 elements and hung the D3D12 device;
    /// `InstanceNorm_StyleTransferDims_MatchesCpu` compares 150,528 and passes. 10.7x.
    ///
    /// This is the TEST HARNESS, not the engine - so it can fail any test that compares a big tensor, on a
    /// lane shared with every other test, and the blame lands on whatever kernel the test happened to be
    /// exercising. It cost me a wrong diagnosis today.
    ///
    /// Sizes ascend and each prints BEFORE the next runs, so if the device dies partway the log still says at
    /// which size - a measurement that survives its own subject crashing.
    /// Read with `PMT_CONSOLE_LOG=CmpCost`.
    /// </remarks>
    [TestMethod(Timeout = 180000)]
    public async Task AssertCloseGpu_CostVsElementCount_PerBackend() => await RunTest(async accelerator =>
    {
        foreach (int count in new[] { 50_176, 150_528, 401_408, 802_816, 1_605_632 })
        {
            // 🔴 `actual` MUST DIFFER FROM `expected`, ELEMENT BY ELEMENT, AND THE RUNNING MAX MUST KEEP
            // MOVING. The first version of this benchmark allocated `actual` FROM `expected`, so every
            // absDiff was exactly 0 - and `Atomic.Max` against a max that never changes retires its CAS
            // immediately, so the contention being measured did not happen. It reported 43.5 ms where the
            // real test pays 12,063 ms, and on the strength of that I told the Captain the atomics were
            // exonerated. A comparison benchmark whose two inputs are equal measures nothing.
            //
            // Ascending-magnitude noise keeps `Atomic.Max` genuinely racing: the max is beaten repeatedly
            // and from many threads at once, which is the production case.
            var expected = new float[count];
            var actual = new float[count];
            for (int i = 0; i < count; i++)
            {
                expected[i] = (i % 89) * 0.01f;
                actual[i] = expected[i] + (i % 1021) * 1e-7f;   // varied, non-zero, max keeps climbing
            }

            using var actualBuf = accelerator.Allocate1D(actual);
            var sw = System.Diagnostics.Stopwatch.StartNew();
            await AssertCloseGpu(accelerator, actualBuf.View, expected, 1e-2f, $"cmp{count}: ");
            sw.Stop();
            Console.WriteLine($"[CmpCost] {BackendName,-8} n={count,9} {sw.Elapsed.TotalMilliseconds,10:F1} ms");
        }
    });

    public async Task DiagnoseInstanceNormPass1(int reps)
    {
        var (context, acc) = await CreateAcceleratorAsync();
        Console.WriteLine($"[INormBench:{BackendName}] reps={reps} per shape (first rep excluded - kernel compile)");
        Console.WriteLine($"[INormBench:{BackendName}] {"shape",-36} {"N",3} {"C",4} {"spatial",8} {"slices",7} | {"pass1-ish",10} {"apply",10} {"InstanceNorm",13}");
        try
        {
            foreach (var (name, N, C, spatial) in INormBenchShapes)
            {
                int slices = N * C;
                long total = (long)slices * spatial;
                var input = new float[total];
                var rng = new Random(9151);
                for (long i = 0; i < total; i++) input[i] = (float)(rng.NextDouble() * 10 - 5);
                var ones = new float[C]; for (int c = 0; c < C; c++) ones[c] = 1f;
                var zeros = new float[C];

                using var inBuf = acc.Allocate1D(input);
                using var outBuf = acc.Allocate1D<float>(total);
                using var sBuf = acc.Allocate1D(ones);
                using var bBuf = acc.Allocate1D(zeros);
                using var sums = acc.Allocate1D<float>(slices);
                using var sqs = acc.Allocate1D<float>(slices);
                using var means = acc.Allocate1D<float>(slices);
                using var invs = acc.Allocate1D<float>(slices);

                var norm = new NormalizationKernels(acc);

                // Every row is timed the same way: one warm dispatch (kernel compile + first-touch), then
                // `reps` timed dispatches each followed by a full synchronize. Timing a dispatch without
                // the synchronize measures the ENQUEUE, not the work.
                var partial = await TimeDispatchAsync(acc, reps,
                    () => norm.InstanceNormPartialStats(inBuf.View, sums.View, sqs.View, N, C, spatial));
                var apply = await TimeDispatchAsync(acc, reps,
                    () => norm.InstanceNormApplyWithStats(outBuf.View, sBuf.View, bBuf.View, means.View, invs.View, N, C, spatial));
                var full = await TimeDispatchAsync(acc, reps,
                    () => norm.InstanceNorm(inBuf.View, outBuf.View, sBuf.View, bBuf.View, N, C, spatial));

                Console.WriteLine($"[INormBench:{BackendName}] {name,-36} {N,3} {C,4} {spatial,8} {slices,7} | {partial,8:F2}ms {apply,8:F2}ms {full,11:F2}ms");
                norm.Dispose();
            }
        }
        finally
        {
            try { await acc.SynchronizeAsync(); } catch { }
            try { acc.Dispose(); } catch { }
            try { context.Dispose(); } catch { }
        }
        Console.WriteLine($"[INormBench:{BackendName}] DONE. A Pass-1 row in the hundreds of ms on a discrete card is the TDR exposure:");
        Console.WriteLine($"[INormBench:{BackendName}] the same kernel runs on a shared browser device behind a compositor and 30+ other tabs.");
    }

    /// <summary>Mean wall-ms of <paramref name="reps"/> dispatch+synchronize pairs, after one untimed warm
    /// dispatch. The synchronize is INSIDE the timed region deliberately - the enqueue is not the cost.</summary>
    private static async Task<double> TimeDispatchAsync(Accelerator acc, int reps, Action dispatch)
    {
        dispatch();
        await acc.SynchronizeAsync();
        var sw = System.Diagnostics.Stopwatch.StartNew();
        for (int r = 0; r < reps; r++)
        {
            dispatch();
            await acc.SynchronizeAsync();
        }
        sw.Stop();
        return sw.Elapsed.TotalMilliseconds / reps;
    }
}
