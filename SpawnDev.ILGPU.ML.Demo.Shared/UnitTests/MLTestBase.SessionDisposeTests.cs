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

public abstract partial class MLTestBase
{
    /// <summary>
    /// Disposing a session releases the scratch its operators' KERNELS hold, not only its pool.
    /// </summary>
    /// <remarks>
    /// SoftmaxKernel, ReductionKernels, TopK's (and Sign's, DepthToSpace's) own kernels and GatherND's params buffer were
    /// not IDisposable, so the registry's <c>(x as IDisposable)?.Dispose()</c> did nothing and their buffers outlived
    /// every session. The desktop GC finalized them eventually, which hid it; a browser did not - SpawnScene's
    /// RaCo-ALIKED extractor left ~10 MB of WebGPU storage per create/run/dispose cycle (TopK's 11 MB sort scratch
    /// among it), ~500 MB over one run (2026-10-03).
    /// <para>
    /// The disposed sessions are deliberately kept REFERENCED, so whatever a disposed session still owns stays
    /// reachable and is counted on every backend - the leak cannot hide behind a GC. The accelerator's live buffers
    /// must not grow from one cycle to the next.
    /// </para>
    /// </remarks>
    [TestMethod(Timeout = 180000)]
    public async Task Session_Dispose_ReleasesKernelScratch()
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");
        var json = await http.GetStringAsync("references/lifecycle/kernel_scratch.json");
        var modelBytes = await http.GetByteArrayAsync("references/lifecycle/kernel_scratch.onnx");
        using var doc = JsonDocument.Parse(json);
        var x = doc.RootElement.GetProperty("inputs").GetProperty("X").GetProperty("data").EnumerateArray().Select(FixtureFloat).ToArray();
        var expectedTop = doc.RootElement.GetProperty("outputs").GetProperty("topv").GetProperty("data").EnumerateArray().Select(FixtureFloat).ToArray();

        await RunTest(async accelerator =>
        {
            var held = new List<InferenceSession>();   // keeps disposed sessions reachable: see remarks
            int afterFirst = -1;
            for (int cycle = 0; cycle < 4; cycle++)
            {
                var session = InferenceSession.CreateFromFile(accelerator, modelBytes,
                    inputShapes: new Dictionary<string, int[]> { ["X"] = new[] { 4, 64 } });
                using (var xBuf = accelerator.Allocate1D(x))
                {
                    var outs = await session.RunAsync(new Dictionary<string, Tensor> { ["X"] = new Tensor(xBuf.View, new[] { 4, 64 }) });
                    if (cycle == 0)
                    {
                        using var host = accelerator.Allocate1D<float>(expectedTop.Length);
                        await host.View.CopyFromAsync(outs["topv"].Data.SubView(0, expectedTop.Length));
                        await accelerator.SynchronizeAsync();
                        var got = await host.CopyToHostAsync<float>(0, expectedTop.Length);
                        double worst = 0;
                        for (int i = 0; i < expectedTop.Length; i++) worst = Math.Max(worst, Math.Abs(got[i] - expectedTop[i]));
                        if (worst > 1e-5) throw new Exception($"kernel_scratch topv: max |d| {worst:E2} vs onnxruntime on {BackendName}");
                    }
                    session.ReturnOutputs(outs);
                    await accelerator.SynchronizeAsync();
                }
                session.Dispose();
                held.Add(session);
                int live = LiveDeviceBuffers(accelerator);
                if (cycle == 0) afterFirst = live;
                else if (live > afterFirst)
                    throw new Exception($"cycle {cycle}: {live} live device buffers vs {afterFirst} after the first cycle on {BackendName} - " +
                        "a disposed session still owns device memory (an operator kernel not released by OperatorRegistry.Dispose?)");
            }
            GC.KeepAlive(held);
        });
    }

    /// <summary>Live (not disposed) device buffers of the accelerator, after a full GC: ILGPU's child-object list, read
    /// by reflection (not exposed), as CudaGraphCapture's census does.</summary>
    private static int LiveDeviceBuffers(Accelerator a)
    {
        GC.Collect(); GC.WaitForPendingFinalizers(); GC.Collect();
        var f = typeof(Accelerator).GetField("childObjects",
            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)
            ?? throw new UnsupportedTestException("Accelerator.childObjects not found (ILGPU internals changed)");
        var list = (System.Collections.IEnumerable)f.GetValue(a)!;
        int n = 0;
        lock (list)
            foreach (var wrObj in list)
                if (wrObj is WeakReference<AcceleratorObject> wr && wr.TryGetTarget(out var t) && t is MemoryBuffer mb && !mb.IsDisposed)
                    n++;
        return n;
    }
}
