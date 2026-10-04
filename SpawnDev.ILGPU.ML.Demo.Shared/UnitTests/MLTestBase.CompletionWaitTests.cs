using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Tensors;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    /// <summary>
    /// <see cref="InferenceSession.SkipCompletionWait"/>: a run that SUBMITS instead of awaiting completion must still be
    /// correct for a caller that only does more GPU work. Run A is submitted, its input buffer is overwritten with
    /// garbage straight away (before A has necessarily executed), run B follows on the same session, and only then are
    /// both outputs read. WebGPU's single queue must keep A's dispatches ahead of the overwrite and B's work - a skip
    /// that broke the order shows as a corrupted A. Elsewhere the flag is ignored (the run still waits).
    /// </summary>
    [TestMethod(Timeout = 180000)]
    public async Task Session_SkipCompletionWait_QueueOrderedAndCorrect()
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");
        var json = await http.GetStringAsync("references/fusion/erf_gelu.json");
        var modelBytes = await http.GetByteArrayAsync("references/fusion/erf_gelu.onnx");
        using var doc = JsonDocument.Parse(json);
        var x = doc.RootElement.GetProperty("inputs").GetProperty("X").GetProperty("data").EnumerateArray().Select(FixtureFloat).ToArray();
        var expected = doc.RootElement.GetProperty("outputs").GetProperty("Y").GetProperty("data").EnumerateArray().Select(FixtureFloat).ToArray();

        await RunTest(async accelerator =>
        {
            using var ms = new MemoryStream(modelBytes);
            using var session = await InferenceSession.CreateFromStreamAsync(accelerator, ms,
                inputShapes: new Dictionary<string, int[]> { ["X"] = new[] { 8, 16 } });
            session.SkipCompletionWait = true;
            using var xa = accelerator.Allocate1D(x);
            using var xb = accelerator.Allocate1D(x);
            var outA = await session.RunAsync(new Dictionary<string, Tensor> { ["X"] = new Tensor(xa.View, new[] { 8, 16 }) });
            bool skipped = session.LastRunCompletionWaitSkipped;
            bool expectSkip = accelerator.AcceleratorType == AcceleratorType.WebGPU;
            if (skipped != expectSkip)
                throw new Exception($"completion wait skipped={skipped} on {BackendName}, expected {expectSkip} (WebGPU only)");
            // Garbage into A's input, right after the run returned. On a browser buffer through CopyFromJS = an IMMEDIATE
            // queue.writeBuffer (a small CopyFromCPU would be recorded in the pending batch instead). It must land after
            // A's dispatches - SpawnDev.ILGPU submits pending work before any host write (FlushBeforeHostWrite), which is
            // what makes skipping the completion wait safe for a caller that keeps writing inputs.
            var garbage = Enumerable.Repeat(1e6f, x.Length).ToArray();
            if (xa.Buffer is SpawnDev.ILGPU.IBrowserMemoryBuffer browserBuffer)
            {
                using var js = new SpawnDev.SpawnJS.JSObjects.Float32Array(garbage);
                browserBuffer.CopyFromJS(js);
            }
            else xa.CopyFromCPU(garbage);
            var outB = await session.RunAsync(new Dictionary<string, Tensor> { ["X"] = new Tensor(xb.View, new[] { 8, 16 }) });
            foreach (var (name, outs) in new[] { ("A", outA), ("B", outB) })
            {
                using var host = accelerator.Allocate1D<float>(expected.Length);
                await host.View.CopyFromAsync(outs["Y"].Data.SubView(0, expected.Length));
                await accelerator.SynchronizeAsync();
                var got = await host.CopyToHostAsync<float>(0, expected.Length);
                double worst = 0, scale = expected.Max(v => Math.Abs((double)v));
                for (int i = 0; i < expected.Length; i++) worst = Math.Max(worst, Math.Abs(got[i] - expected[i]) / scale);
                if (worst > 1e-5)
                    throw new Exception($"run {name}: max |d| / max|y| {worst:E2} vs onnxruntime on {BackendName} (completion wait skipped: {skipped})");
            }
            session.ReturnOutputs(outA);
            session.ReturnOutputs(outB);
        });
    }
}
