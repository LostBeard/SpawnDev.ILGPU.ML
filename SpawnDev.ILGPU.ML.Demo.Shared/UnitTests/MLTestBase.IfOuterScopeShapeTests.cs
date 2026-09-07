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
/// Gate for an <c>If</c> whose branch CAPTURES AN OUTER-SCOPE TENSOR whose shape varies, run at several
/// outer shapes in one session so the executor LRU evicts underneath it.
///
/// <para>
/// WHAT THIS COVERS. ONNX subgraphs capture from the enclosing scope implicitly, and
/// <c>OuterScope.Add</c> supplies "every tensor the subgraph references but does not itself produce".
/// Those captured tensors go into <c>subInputs</c>, which is BOTH what the branch executes against and
/// what keys <c>SubgraphRunner</c>'s plan cache. This runs one session across five distinct outer shapes
/// (forcing four LRU evictions, since <c>MaxShapeExecutors</c> is 3 and the creation shape uses the base
/// executor, which is never in that cache), checks every case against onnxruntime, and checks that a
/// repeated shape reproduces its own first reading BIT-FOR-BIT rather than merely within tolerance.
/// </para>
///
/// <para>
/// 🔴 WHAT THIS DOES **NOT** COVER, AND WHY. It is NOT a guard for the 5.2.11 plan-cache defect (a cached
/// plan bound to a pool belonging to a different, still-cached executor). <b>MEASURED 2026-09-06: with
/// the fix - <c>if (!ReferenceEquals(candidate.ConstantsPool, ctx.Pool)) continue;</c> - DISABLED, this
/// test still passed 8 of 8.</b> It was written believing the cache key covered only the subgraph's
/// declared inputs, so a branch reading a dynamic outer-scope tensor would collide across shapes. That
/// premise is wrong: <c>IfOperator.ExecuteAsync</c> calls <c>OuterScope.Add</c> BEFORE
/// <c>SubgraphRunner.ExecuteAsync</c>, so the captured tensor's shape is in the signature and the plan is
/// correctly rebuilt per outer shape. A branch that captures a varying-shape tensor can never collide.
/// </para>
///
/// <para>
/// Reproducing the real defect needs the opposite shape of fixture: a branch whose captures are ALL
/// shape-INVARIANT (ZipVoice's relative-position <c>If</c> captures <c>[1]</c> scalars, which is why one
/// entry served every utterance length) while the OUTER graph shape varies, so executors churn and evict
/// while the subgraph signature stays constant. That defect is currently gated only at the SpawnDev.AI
/// level, by audio hashes in the voice gate. See the CHANGELOG's 5.2.11 entry.
/// </para>
///
/// <para>
/// ⚠️ The fixture's data must DEPEND ON N. With a plain <c>arange</c>, row <i>i</i> holds the same values
/// at every shape, so a stale plan reading a larger shape's leading rows would return exactly the right
/// numbers. Regenerate with <c>tools/gen_subgraph_plan_cache_reference.py</c>.
/// </para>
/// </summary>
public abstract partial class MLTestBase
{
    /// <summary>
    /// Run one session across the fixture's shape sequence, checking every case against onnxruntime AND
    /// checking that a repeated shape reproduces its own first reading exactly.
    /// </summary>
    [TestMethod(Timeout = 180000)]
    public async Task ControlFlow_IfOuterScopeCapture_AcrossExecutorEviction()
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");

        const string name = "if_outer_scope_dynamic";
        var json = await http.GetStringAsync($"references/controlflow/{name}.json");
        var modelBytes = await http.GetByteArrayAsync($"references/controlflow/{name}.onnx");
        using var doc = JsonDocument.Parse(json);
        var cases = doc.RootElement.GetProperty("cases").EnumerateArray().ToArray();

        // Guard the fixture's own preconditions, so a regenerated or hand-edited fixture that can no longer
        // reach an eviction fails LOUDLY here instead of passing green and guarding nothing.
        var distinctShapes = cases.Select(c => c.GetProperty("n").GetInt32()).Distinct().Count();
        var repeats = cases.Length - distinctShapes;
        if (distinctShapes < 5)
            throw new Exception(
                $"{name}: only {distinctShapes} distinct shapes. The creation shape uses the BASE executor, "
                + "which is never in the LRU, so 5 distinct shapes are needed for 4 non-base ones to pass "
                + "MaxShapeExecutors=3 and force an eviction. Without an eviction this fixture cannot "
                + "exercise the defect.");
        if (repeats < 1)
            throw new Exception($"{name}: no shape is run twice, so history independence is never checked");

        await RunTest(async accelerator =>
        {
            // ONE session for the whole sequence. The plan cache lives on the session's registry, so a
            // session per shape would destroy the very state under test.
            var firstShapes = new Dictionary<string, int[]>();
            foreach (var p in cases[0].GetProperty("inputs").EnumerateObject())
                firstShapes[p.Name] = p.Value.GetProperty("shape").EnumerateArray()
                    .Select(e => e.GetInt32()).ToArray();

            using var session = InferenceSession.CreateFromFile(accelerator, modelBytes,
                inputShapes: firstShapes);

            // Keyed by the case's N, so the repeat compares against the FIRST reading at that shape - the
            // same comparison the ZipVoice audio hashes made, and the one word-overlap was too coarse for.
            var firstReading = new Dictionary<int, float[]>();

            foreach (var c in cases)
            {
                var index = c.GetProperty("index").GetInt32();
                var n = c.GetProperty("n").GetInt32();

                var feeds = new Dictionary<string, Tensor>();
                var buffers = new List<MemoryBuffer1D<float, Stride1D.Dense>>();
                try
                {
                    foreach (var p in c.GetProperty("inputs").EnumerateObject())
                    {
                        var data = p.Value.GetProperty("data").EnumerateArray()
                            .Select(FixtureFloat).ToArray();
                        var shape = p.Value.GetProperty("shape").EnumerateArray()
                            .Select(e => e.GetInt32()).ToArray();
                        if (data.Length == 0) data = new[] { 0f };
                        if (shape.Length == 0) shape = new[] { 1 };
                        var buf = accelerator.Allocate1D(data);
                        buffers.Add(buf);
                        feeds[p.Name] = new Tensor(buf.View, shape);
                    }

                    var outputs = await session.RunAsync(feeds);

                    var expected = c.GetProperty("outputs").GetProperty("Y").GetProperty("data")
                        .EnumerateArray().Select(FixtureFloat).ToArray();

                    if (!outputs.TryGetValue("Y", out var got))
                        throw new Exception($"{name} case {index} (N={n}): no output named 'Y'");

                    // SIZE first. A plan built for another shape is most likely to declare that other
                    // shape's length, and a count mismatch names the defect far more clearly than a value.
                    if (got.ElementCount != expected.Length)
                        throw new Exception(
                            $"{name} case {index} (N={n}): output holds {got.ElementCount} values, "
                            + $"{expected.Length} expected. A cached subgraph plan built for a DIFFERENT "
                            + "outer shape is what causes this.");

                    using var host = accelerator.Allocate1D<float>(expected.Length);
                    await host.View.CopyFromAsync(got.Data.SubView(0, expected.Length));
                    await accelerator.SynchronizeAsync();
                    var values = await host.CopyToHostAsync<float>(0, expected.Length);

                    double worst = 0; int worstAt = -1;
                    for (var i = 0; i < expected.Length; i++)
                    {
                        var d = Math.Abs(values[i] - expected[i]);
                        if (d > worst) { worst = d; worstAt = i; }
                    }
                    if (worst > 1e-4)
                        throw new Exception(
                            $"{name} case {index} (N={n}): max |d| {worst:E3} at {worstAt} "
                            + $"(ORT {expected[worstAt]:F6} vs ours {values[worstAt]:F6})");

                    // HISTORY INDEPENDENCE. Running this shape again, after other shapes have evicted it,
                    // must reproduce its own first reading EXACTLY - not merely "close enough to ORT".
                    // That is the property the defect broke, and tolerance-based checks miss it: a plan
                    // bound to another executor's pool can still land inside 1e-4 and be a different answer.
                    if (firstReading.TryGetValue(n, out var before))
                    {
                        for (var i = 0; i < values.Length; i++)
                            if (BitConverter.SingleToInt32Bits(values[i])
                                != BitConverter.SingleToInt32Bits(before[i]))
                                throw new Exception(
                                    $"{name} case {index} (N={n}): element {i} changed between the first "
                                    + $"and second run of the SAME shape ({before[i]:R} -> {values[i]:R}). "
                                    + "The output depends on execution history - a cached subgraph plan is "
                                    + "being reused across shape executors.");
                        Console.WriteLine($"[IfOuterScope] case {index} N={n}: reproduced its first "
                                        + "reading bit-for-bit");
                    }
                    else
                    {
                        firstReading[n] = values;
                        Console.WriteLine($"[IfOuterScope] case {index} N={n}: {expected.Length} "
                                        + $"values, max |d| vs ORT {worst:E2}");
                    }
                }
                finally
                {
                    foreach (var b in buffers) b.Dispose();
                }
            }

            if (!firstReading.Values.Any(v => v.Length > 0))
                throw new Exception($"{name}: no case produced any output");
        });
    }
}
