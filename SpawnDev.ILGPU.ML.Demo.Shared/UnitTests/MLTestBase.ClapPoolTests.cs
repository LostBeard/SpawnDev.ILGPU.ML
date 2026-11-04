using ILGPU.Runtime;
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
}
