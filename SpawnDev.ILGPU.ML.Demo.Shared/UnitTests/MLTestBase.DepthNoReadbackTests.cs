using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// The no-readback depth path: <see cref="Kernels.ImagePostprocessKernel.MinMax"/> into a device view and
/// <see cref="DepthEstimationPipeline.EstimateGpuRawAsync(ArrayView1D{int, Stride1D.Dense}, int, int, ArrayView1D{float, Stride1D.Dense}, ArrayView1D{float, Stride1D.Dense}, int, int)"/>
/// into caller buffers. A video consumer (Anaglyphohol) runs it every frame and never syncs, so these gates
/// queue several frames - and several input SHAPES, enough to evict a per-shape executor - before reading
/// anything back.
/// </summary>
public abstract partial class MLTestBase
{
    /// <summary>
    /// MinMax into a caller view at an OFFSET inside a larger buffer: exact min/max (comparisons only, no
    /// arithmetic), and the neighbours of the 2-float window untouched (a kernel writing result[0..1] of the
    /// base buffer instead of the view would overwrite the sentinels and miss the window).
    /// </summary>
    [TestMethod]
    public async Task MinMax_IntoDeviceView_MatchesHost() => await RunTest(async accelerator =>
    {
        int count = 518 * 518 + 37;   // video-path size, deliberately not 1024-aligned
        var data = RandomFloats(count, seed: 778, scale: 5f);
        data[321] = -42.5f; data[count - 1] = 99.25f;   // known extremes at awkward positions
        float expMin = data[0], expMax = data[0];
        foreach (var v in data) { if (v < expMin) expMin = v; if (v > expMax) expMax = v; }

        const float Sentinel = 7777.5f;
        using var buf = accelerator.Allocate1D(data);
        using var result = accelerator.Allocate1D(new[] { Sentinel, Sentinel, Sentinel, Sentinel, Sentinel, Sentinel });
        using var post = new Kernels.ImagePostprocessKernel(accelerator);
        post.MinMax(buf.View, count, result.View.SubView(2, 2));
        await accelerator.SynchronizeAsync();
        var r = await result.CopyToHostAsync<float>(0, 6);
        if (r[2] != expMin || r[3] != expMax)
            throw new Exception($"{BackendName}: device min/max ({r[2]}, {r[3]}) != host ({expMin}, {expMax})");
        if (r[0] != Sentinel || r[1] != Sentinel || r[4] != Sentinel || r[5] != Sentinel)
            throw new Exception($"{BackendName}: MinMax wrote outside its 2-float window: [{string.Join(", ", r)}]");
    });

    /// <summary>
    /// Caller buffers vs the allocating overload, bit-exact, with NO sync between frames: two different frames
    /// back to back, then a run of NativeAspect input shapes (more than InferenceSession keeps executors for, so
    /// one is evicted while earlier frames may still be queued). Also: an oversized output keeps its tail, and a
    /// too-small output throws without breaking the pipeline.
    /// </summary>
    [TestMethod(Timeout = 600000, Category = "HeavyModel")]
    public async Task Depth_NoReadback_CallerBuffers_MatchAllocating() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient();
        if (http == null)
            throw new UnsupportedTestException("HttpClient not available for this backend");

        var onnxBytes = await InferenceSession.DownloadBytesChunkedAsync(http,
            HuggingFaceClient.GetDownloadUrl("onnx-community/depth-anything-v2-small", "onnx/model.onnx"));
        using var session = InferenceSession.CreateFromOnnx(
            accelerator, onnxBytes,
            inputShapes: new Dictionary<string, int[]> { ["pixel_values"] = new[] { 1, 3, 224, 224 } });
        using var pipeline = new DepthEstimationPipeline(session, accelerator);
        pipeline.EnableGraphCapture = false;   // the per-frame plain forward this overload is for

        int w = 64, h = 48;
        int[] Frame(int seed)
        {
            var px = new int[w * h];
            var rng = new Random(seed);
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                    px[y * w + x] = (int)(x * 255f / w) | ((int)(y * 255f / h) << 8) | (rng.Next(0, 256) << 16)
                        | unchecked((int)0xFF000000);
            return px;
        }

        // References: the allocating overload (it reads min/max back, so each call is fully synchronized).
        async Task<(float[] Depth, float Min, float Max)> Reference(ArrayView1D<int, Stride1D.Dense> rgba)
        {
            var (raw, mn, mx, ow, oh) = await pipeline.EstimateGpuRawAsync(rgba, w, h, w, h);
            using (raw) return (await raw.CopyToHostAsync<float>(0, ow * oh), mn, mx);
        }

        const float Sentinel = -12345.25f;
        int n = w * h, tail = 17;
        async Task Compare(string what, MemoryBuffer1D<float, Stride1D.Dense> outBuf, MemoryBuffer1D<float, Stride1D.Dense> mmBuf,
            (float[] Depth, float Min, float Max) expected)
        {
            var got = await outBuf.CopyToHostAsync<float>(0, n + tail);
            var mm = await mmBuf.CopyToHostAsync<float>(0, 2);
            if (mm[0] != expected.Min || mm[1] != expected.Max)
                throw new Exception($"{BackendName} {what}: device min/max ({mm[0]}, {mm[1]}) != allocating path ({expected.Min}, {expected.Max})");
            for (int i = 0; i < n; i++)
                if (BitConverter.SingleToInt32Bits(got[i]) != BitConverter.SingleToInt32Bits(expected.Depth[i]))
                    throw new Exception($"{BackendName} {what}: depth[{i}] = {got[i]}, allocating path = {expected.Depth[i]}");
            for (int i = n; i < n + tail; i++)
                if (got[i] != Sentinel)
                    throw new Exception($"{BackendName} {what}: wrote past the {w}x{h} grid at [{i}] = {got[i]}");
        }
        MemoryBuffer1D<float, Stride1D.Dense> NewOut() => accelerator.Allocate1D(Enumerable.Repeat(Sentinel, n + tail).ToArray());

        using var frameA = accelerator.Allocate1D(Frame(1));
        using var frameB = accelerator.Allocate1D(Frame(2));
        await Reference(frameA.View);   // warm (compile) outside the comparison

        // 1. Two different frames back to back into caller buffers, nothing read until both are queued.
        using var outA = NewOut();
        using var outB = NewOut();
        using var mmA = accelerator.Allocate1D<float>(2);
        using var mmB = accelerator.Allocate1D<float>(2);
        var sizeA = await pipeline.EstimateGpuRawAsync(frameA.View, w, h, outA.View, mmA.View, w, h);
        var sizeB = await pipeline.EstimateGpuRawAsync(frameB.View, w, h, outB.View, mmB.View, w, h);
        if (sizeA != (w, h) || sizeB != (w, h))
            throw new Exception($"{BackendName}: returned sizes {sizeA} / {sizeB}, expected ({w}, {h})");
        var refA = await Reference(frameA.View);
        var refB = await Reference(frameB.View);
        await Compare("frame A", outA, mmA, refA);
        await Compare("frame B", outB, mmB, refB);
        // NEGATIVE CONTROL: the two frames really differ, so a comparison against the wrong frame would fail.
        if (refA.Depth.AsSpan().SequenceEqual(refB.Depth))
            throw new Exception($"{BackendName}: frames A and B gave identical depth - the comparison above proves nothing");

        // 2. Too small an output is refused, and the pipeline still works afterwards.
        using (var small = accelerator.Allocate1D<float>(n - 1))
        {
            bool threw = false;
            try { await pipeline.EstimateGpuRawAsync(frameA.View, w, h, small.View, mmA.View, w, h); }
            catch (ArgumentException) { threw = true; }
            if (!threw) throw new Exception($"{BackendName}: a {n - 1}-float output for a {w}x{h} grid did not throw");
        }

        // 3. NativeAspect across more input shapes than the session caches executors for, all queued unsynced.
        pipeline.ResizeMode = DepthResizeMode.NativeAspect;
        int[] longSides = { 112, 140, 168, 196, 224 };
        var outs = new List<(MemoryBuffer1D<float, Stride1D.Dense> Out, MemoryBuffer1D<float, Stride1D.Dense> Mm, int Side)>();
        try
        {
            foreach (int side in longSides)
            {
                pipeline.ProcessResolution = side;
                var o = NewOut();
                var m = accelerator.Allocate1D<float>(2);
                outs.Add((o, m, side));
                await pipeline.EstimateGpuRawAsync(frameB.View, w, h, o.View, m.View, w, h);
            }
            foreach (var (o, m, side) in outs)
            {
                pipeline.ProcessResolution = side;
                await Compare($"NativeAspect long side {side}", o, m, await Reference(frameB.View));
            }
        }
        finally
        {
            foreach (var (o, m, _) in outs) { o.Dispose(); m.Dispose(); }
        }
        Console.WriteLine($"[DepthNoReadback] {BackendName}: caller buffers bit-exact vs allocating path (2 frames, {longSides.Length} shapes)");
    });
}
