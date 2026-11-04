using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Graph;
using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnDev.ILGPU.ML.Tensors;
using SpawnDev.UnitTesting;
using System.Text.Json;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    /// <summary>
    /// CLAP audio tower (Xenova/clap-htsat-unfused audio_model.onnx, 118 MB) pool memory per forward, five forwards of the fixture's tone clip: with
    /// <see cref="BufferPool.PlannedReuse"/> the first runs record and later ones replay a plan, so the line shows the pool before and after planning,
    /// the plan counters, and that every embedding is bit-identical. Asked by Riker (AssistZ, 2026-11-04): whether reuse engages for
    /// ClapAudioClassifier and what cut to expect. CUDA / OpenCL / WebGPU; read the browser line with PMT_CONSOLE_LOG=ClapPool. On WebGPU it also gates every live buffer (WebGPUBufferAccounting), not just the pool's.
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task Bench_ClapAudioTower_PoolPeak() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is not (AcceleratorType.Cuda or AcceleratorType.OpenCL or AcceleratorType.WebGPU))
            throw new UnsupportedTestException("a CUDA / OpenCL / WebGPU measurement");
        var assets = GetHttpClient() ?? throw new UnsupportedTestException("HttpClient not available");
        using var doc = JsonDocument.Parse(await assets.GetStringAsync("test-audio/clap-fixture.json"));
        var clip = doc.RootElement.GetProperty("clips")[0];
        var feats = clip.GetProperty("input_features").EnumerateArray().SelectMany(r => r.EnumerateArray().Select(v => v.GetSingle())).ToArray();
        using var http = CreateHuggingFaceHttpClient();
        http.Timeout = TimeSpan.FromMinutes(20);
        var bytes = await InferenceSession.DownloadBytesChunkedAsync(http, "https://models.spawndev.com/Xenova/clap-htsat-unfused/resolve/main/onnx/audio_model.onnx");
        bool trackSaved = BufferPool.TrackPeaks;
        BufferPool.TrackPeaks = true;
        try
        {
            using var session = InferenceSession.CreateFromFile(accelerator, bytes);
            session.OutputsLiveUntilNextRun = true;   // as ClapAudioClassifier runs it
            using var featBuf = accelerator.Allocate1D(feats);
            var input = new Dictionary<string, Tensor> { [session.InputNames[0]] = new Tensor(featBuf.View, new[] { 1, 1, ClapFeatureExtractor.Frames, ClapFeatureExtractor.Mels }) };
            const double MiB = 1048576.0;
            long plans0 = BufferPool.PlannedReusePlans, rents0 = BufferPool.PlannedReuseRents, dev0 = BufferPool.PlannedReuseDeviations;
            float[]? first = null;
            var parts = new List<string>();
            for (int run = 1; run <= 5; run++)
            {
                BufferPool.ResetPeaks();
                var outs = await session.RunAsync(input);
                var emb = await outs[session.OutputNames[0]].Data.SubView(0, 512).CopyToHostAsync();
                if (first == null) first = emb;
                else if (!emb.SequenceEqual(first)) throw new Exception($"run {run}: the embedding differs from run 1");
                var pool = session.Executor.Pool;   // shape-recompiled executors share this pool
                parts.Add($"run {run}: peak total {BufferPool.PeakTotalBytes / MiB:F0} MiB, peak live {BufferPool.PeakLiveBytes / MiB:F0} MiB, planned slots {pool.PlannedSlotBytes / MiB:F0} MiB");
            }
            // A long run, as AssistZ does one forward per audio event: does anything grow per forward? (Riker saw Chrome's dedicated GPU memory climb
            // ~160 MB over 130 events, on local.3 and local.5 alike.) Device allocations by the pool must stay flat once the plan is in place.
            long allocs0 = BufferPool.TotalDeviceAllocations;
            // Buffers OUTSIDE the pool (kernel params, scratch) are invisible to TotalDeviceAllocations: on WebGPU the backend's own accounting
            // sees every buffer. MEASURED 2026-11-04 (Riker, AssistZ): PadKernel allocated a fresh 88-byte params buffer per call and kept it -
            // 12 per CLAP forward, Chrome's dedicated GPU memory +0.95 MB per audio event - while the pool counter stayed flat.
            bool webgpu = accelerator.AcceleratorType == AcceleratorType.WebGPU;
            int live0 = webgpu ? global::SpawnDev.ILGPU.WebGPU.WebGPUBufferAccounting.LiveBufferCount : 0;
            const int LongRuns = 200;
            for (int run = 0; run < LongRuns; run++)
            {
                var outs = await session.RunAsync(input);
                await outs[session.OutputNames[0]].Data.SubView(0, 512).CopyToHostAsync();
            }
            long grew = BufferPool.TotalDeviceAllocations - allocs0;
            int liveGrew = webgpu ? global::SpawnDev.ILGPU.WebGPU.WebGPUBufferAccounting.LiveBufferCount - live0 : 0;
            parts.Add($"{LongRuns} more forwards: pool device allocations +{grew}" + (webgpu ? $", live WebGPU buffers +{liveGrew}" : "")
                + $", planned slots {session.Executor.Pool.PlannedSlotBytes / MiB:F0} MiB, free {session.PooledFreeBytes / MiB:F0} MiB");
            if (grew > 0) throw new Exception($"the pool allocated {grew} device buffers over {LongRuns} identical forwards: " + string.Join(" | ", parts));
            if (liveGrew > 0)
                throw new Exception($"{liveGrew} live WebGPU buffers more after {LongRuns} identical forwards (outside the pool): top creation sites need "
                    + "WebGPUBufferAccounting.CaptureCreationSites; " + string.Join(" | ", parts));
            var line = $"{accelerator.AcceleratorType}: weights {bytes.Length / MiB:F0} MiB, planned reuse {(BufferPool.PlannedReuse ? "on" : "off")} "
                + $"(plans +{BufferPool.PlannedReusePlans - plans0}, planned rents +{BufferPool.PlannedReuseRents - rents0}, deviations +{BufferPool.PlannedReuseDeviations - dev0}) | "
                + string.Join(" | ", parts);
            Console.WriteLine("[ClapPool] " + line);
            return line;
        }
        finally { BufferPool.TrackPeaks = trackSaved; }
    });

    /// <summary>
    /// CLAP audio tower, captured (SessionGraphCapture: CUDA graph / WebGPU dispatch plan) vs direct, at the classifier's fixed input
    /// [1,1,1001,64]: steady-state direct forward, the capture call, steady-state replay, and how many embedding values the replay changes
    /// (must be 0). AssistZ runs one forward per audio event, so a replay that is faster and bit-identical is a free win for it.
    /// Prints "[Benchmark] ClapCapture [...]"; read the browser line with PMT_CONSOLE_LOG=ClapCapture.
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task Bench_ClapAudioTower_CaptureVsDirect() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is not (AcceleratorType.Cuda or AcceleratorType.WebGPU))
            throw new UnsupportedTestException($"graph capture is CUDA and WebGPU only; {accelerator.AcceleratorType} has nothing to replay");
        var assets = GetHttpClient() ?? throw new UnsupportedTestException("HttpClient not available");
        using var doc = JsonDocument.Parse(await assets.GetStringAsync("test-audio/clap-fixture.json"));
        var clips = doc.RootElement.GetProperty("clips");
        float[] Feats(int i) => clips[i].GetProperty("input_features").EnumerateArray().SelectMany(r => r.EnumerateArray().Select(v => v.GetSingle())).ToArray();
        var featsA = Feats(0);
        var featsB = Feats(clips.GetArrayLength() - 1);
        using var http = CreateHuggingFaceHttpClient();
        http.Timeout = TimeSpan.FromMinutes(20);
        var bytes = await InferenceSession.DownloadBytesChunkedAsync(http, "https://models.spawndev.com/Xenova/clap-htsat-unfused/resolve/main/onnx/audio_model.onnx");
        using var session = InferenceSession.CreateFromFile(accelerator, bytes);
        session.OutputsLiveUntilNextRun = true;
        using var featBuf = accelerator.Allocate1D<float>(featsA.Length);
        var shape = new[] { 1, 1, ClapFeatureExtractor.Frames, ClapFeatureExtractor.Mels };
        string name = session.InputNames[0], outName = session.OutputNames[0];
        async Task<float[]> Run(Func<Dictionary<string, Tensor>, Task<Dictionary<string, Tensor>>> run, float[] feats)
        {
            featBuf.CopyFromCPU(feats);
            var outs = await run(new Dictionary<string, Tensor> { [name] = new Tensor(featBuf.View, shape) });
            return await outs[outName].Data.SubView(0, 512).CopyToHostAsync();
        }
        const int Steps = 30;
        var direct = (Func<Dictionary<string, Tensor>, Task<Dictionary<string, Tensor>>>)session.RunAsync;
        // compile + warm, and past planned reuse's record/plan runs: timed earlier, the direct side carried the plan build (64 ms on WebGPU
        // in the first measurement against 32 ms steady state)
        for (int i = 0; i < 8; i++) await Run(direct, (i & 1) == 0 ? featsA : featsB);
        var sw = System.Diagnostics.Stopwatch.StartNew();
        for (int i = 0; i < Steps; i++) await Run(direct, (i & 1) == 0 ? featsA : featsB);
        double directMs = sw.Elapsed.TotalMilliseconds / Steps;
        var directA = await Run(direct, featsA);
        var directB = await Run(direct, featsB);

        using var capture = new SessionGraphCapture(session, accelerator);
        var captured = (Func<Dictionary<string, Tensor>, Task<Dictionary<string, Tensor>>>)capture.RunAsync;
        sw.Restart();
        int setupCalls = 0;
        while (!capture.IsCaptured && setupCalls < 4) { await Run(captured, featsA); setupCalls++; }
        double setupMs = sw.Elapsed.TotalMilliseconds;
        if (!capture.IsCaptured) throw new Exception($"capture never went live after {setupCalls} calls: {capture.CaptureStatus}");
        await Run(captured, featsB);                                              // first replay
        sw.Restart();
        for (int i = 0; i < Steps; i++) await Run(captured, (i & 1) == 0 ? featsA : featsB);
        double replayMs = sw.Elapsed.TotalMilliseconds / Steps;
        // Replays must not grow anything per call (Riker's PadKernel climb was invisible to the pool counters): count every live WebGPU buffer.
        bool webgpu = accelerator.AcceleratorType == AcceleratorType.WebGPU;
        int live0 = webgpu ? global::SpawnDev.ILGPU.WebGPU.WebGPUBufferAccounting.LiveBufferCount : 0;
        long allocs0 = BufferPool.TotalDeviceAllocations;
        for (int i = 0; i < 100; i++) await Run(captured, (i & 1) == 0 ? featsA : featsB);
        int liveGrew = webgpu ? global::SpawnDev.ILGPU.WebGPU.WebGPUBufferAccounting.LiveBufferCount - live0 : 0;
        long allocsGrew = BufferPool.TotalDeviceAllocations - allocs0;
        var replayA = await Run(captured, featsA);
        var replayB = await Run(captured, featsB);
        int differ = 0;
        for (int i = 0; i < 512; i++)
        {
            if (BitConverter.SingleToInt32Bits(replayA[i]) != BitConverter.SingleToInt32Bits(directA[i])) differ++;
            if (BitConverter.SingleToInt32Bits(replayB[i]) != BitConverter.SingleToInt32Bits(directB[i])) differ++;
        }
        bool distinct = !directA.SequenceEqual(directB);   // the two clips must give different embeddings, or the replay check proves nothing
        var line = $"{accelerator.AcceleratorType}: direct {directMs:F1} ms/forward, capture setup {setupMs:F0} ms over {setupCalls} call(s) "
            + $"({capture.CaptureStatus}), replay {replayMs:F1} ms/forward ({directMs / replayMs:F2}x), replay differs from direct in {differ} of 1024 values, "
            + $"100 more replays: pool device allocations +{allocsGrew}" + (webgpu ? $", live WebGPU buffers +{liveGrew}" : "");
        Console.WriteLine($"[Benchmark] ClapCapture [{accelerator.AcceleratorType}] {line}");
        if (!distinct) throw new Exception("the two fixture clips gave identical embeddings: the input-change check is vacuous: " + line);
        if (differ != 0) throw new Exception("captured replay is not bit-identical to the direct forward: " + line);
        if (liveGrew > 0 || allocsGrew > 0) throw new Exception("replays grow device memory: " + line);
        return line;
    });
}
