using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Tensors;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// Gate on <see cref="BufferPool"/> OWNERSHIP: no graph run may READ a tensor whose buffer is already back
/// in the free pool (use-after-return).
///
/// <para>
/// WHY THIS FILE EXISTS: the pool keys ownership on the tensor NAME. <c>Rent(shape, name)</c> writes
/// <c>_namedBuffers[name] = buffer</c> unconditionally and <c>Return(tensor)</c> pools whatever that name
/// maps to at the moment it is called, so the record can diverge from the tensor actually being returned.
/// Whether that MATTERS depends on liveness, which the pool does not track.
/// </para>
///
/// <para>
/// ⚠️ SO THIS ASSERTS THE NARROW, UNAMBIGUOUS INVARIANT: no node may READ a tensor whose buffer is already
/// in the free bucket. A returned buffer still reads perfectly - it just holds whatever Rented that bucket
/// next - which is why this kind of defect surfaces as values that change between otherwise identical runs
/// rather than as an error.
/// </para>
///
/// <para>
/// ⚠️ HISTORY, because it is the point: this file was first written asserting ZERO divergences of ANY kind,
/// on the theory that ZipVoice's long-utterance failure was buffer aliasing. It failed 18 of 20 on
/// KNOWN-CORRECT graphs (tiny_loop 21 divergences, tiny_scan 23, both matching onnxruntime exactly), and the
/// aliasing theory was wrong: the real defects were a compile-time-resolved Slice under an If and a missing
/// entry in the executor's unary shape-adoption allowlist. A record divergence is a LEAD, not a verdict, and
/// a gate that fails on correct code teaches people to ignore it.
/// </para>
/// </summary>
public abstract partial class MLTestBase
{
    /// <summary>
    /// Run one control-flow fixture with pool-ownership detection ON and require a clean run.
    /// </summary>
    private async Task PoolOwnershipCleanFor(string name)
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");

        var json = await http.GetStringAsync($"references/controlflow/{name}.json");
        var modelBytes = await http.GetByteArrayAsync($"references/controlflow/{name}.onnx");
        using var doc = JsonDocument.Parse(json);
        var root = doc.RootElement;

        var inputShapes = new Dictionary<string, int[]>();
        foreach (var p in root.GetProperty("inputs").EnumerateObject())
            inputShapes[p.Name] = p.Value.GetProperty("shape").EnumerateArray().Select(e => e.GetInt32()).ToArray();

        await RunTest(async accelerator =>
        {
            bool prior = BufferPool.TracePoolOwnership;
            BufferPool.TracePoolOwnership = true;
            BufferPool.ResetPoolOwnershipTrace();
            try
            {
                using var session = InferenceSession.CreateFromFile(accelerator, modelBytes,
                    inputShapes: inputShapes.ToDictionary(kv => kv.Key,
                        kv => kv.Value.Length == 0 ? new[] { 1 } : kv.Value));

                var feeds = new Dictionary<string, Tensor>();
                var buffers = new List<MemoryBuffer1D<float, Stride1D.Dense>>();
                try
                {
                    foreach (var p in root.GetProperty("inputs").EnumerateObject())
                    {
                        var data = p.Value.GetProperty("data").EnumerateArray().Select(FixtureFloat).ToArray();
                        if (data.Length == 0) data = new[] { 0f };
                        var shape = inputShapes[p.Name];
                        if (shape.Length == 0) shape = new[] { 1 };
                        var buf = accelerator.Allocate1D(data);
                        buffers.Add(buf);
                        feeds[p.Name] = new Tensor(buf.View, shape);
                    }

                    // ⚠️ Run it TWICE. The defect needs a name to be Rented while an earlier tensor of that
                    // name is still live, and the second run against a warm executor (and, for control flow,
                    // a CACHED subgraph plan) is the cheapest way to reach that state. A single run can be
                    // clean while every run after it is not - which is exactly how this reached production.
                    await session.RunAsync(feeds);
                    await session.RunAsync(feeds);
                }
                finally
                {
                    foreach (var b in buffers) b.Dispose();
                }

                List<string> violations;
                lock (BufferPool.PoolOwnershipViolations)
                    violations = BufferPool.PoolOwnershipViolations.ToList();

                // ⚠️ ONLY USE-AFTER-RETURN is asserted, and the distinction is the whole point of this test.
                //
                // USE-AFTER-RETURN means a node READ a tensor whose buffer is sitting in the free bucket. That
                // is unambiguous: its contents belong to whatever Rents that bucket next.
                //
                // ALIEN-RETURN and REBIND-LIVE only say the pool's NAME->buffer record diverged from the
                // tensor being returned. MEASURED 2026-09-01: tiny_loop produces 21 and tiny_scan 23, and both
                // match onnxruntime exactly - they are views/handoffs carrying a name they never Rented. An
                // earlier version of this test asserted zero of ALL kinds and failed 18/20 on correct graphs.
                // Asserting a condition that healthy code violates trains people to ignore the gate, so those
                // two are REPORTED and not failed on.
                var readAfterFree = violations.Where(v => v.StartsWith("USE-AFTER-RETURN", StringComparison.Ordinal)).ToList();
                if (readAfterFree.Count > 0)
                    throw new Exception(
                        $"{name}: {readAfterFree.Count} use-after-return(s) - a node read a tensor whose buffer "
                        + "was already back in the free pool, so its contents belong to another tensor. First few:\n  "
                        + string.Join("\n  ", readAfterFree.Take(5)));

                Console.WriteLine($"[PoolOwnership] {name}: 0 use-after-return over 2 runs "
                                + $"({violations.Count} record-divergence note(s), not asserted)");
            }
            finally
            {
                BufferPool.TracePoolOwnership = prior;
                BufferPool.ResetPoolOwnershipTrace();
            }
        });
    }

    [TestMethod(Timeout = 180000)]
    public async Task PoolOwnership_If_NoAliasedBuffers() => await PoolOwnershipCleanFor("tiny_if");

    [TestMethod(Timeout = 180000)]
    public async Task PoolOwnership_Loop_NoAliasedBuffers() => await PoolOwnershipCleanFor("tiny_loop");

    [TestMethod(Timeout = 180000)]
    public async Task PoolOwnership_Scan_NoAliasedBuffers() => await PoolOwnershipCleanFor("tiny_scan");
}
