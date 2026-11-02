using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnDev.UnitTesting;
using System.Diagnostics;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    /// <summary>
    /// A measurement for SpawnDev.AI's lost WebGPU device (LongHistoryIsTrimmedRatherThanFailing, qwen3:0.6b on the batch decoder, turn 5 at ~3.1k tokens):
    /// <see cref="GgufBatchDecoder"/>'s prefill wall time (one prefill + one wait) for ~600 new tokens over a growing reused prefix up to ~3.6k, and a full 3.6k
    /// prefill; on CUDA also where the time goes (ProfileKernels: a wait per kernel). Qwen3-0.6B Q8_0, maxSeq 4096.
    /// </summary>
    [TestMethod(Timeout = 900000, Category = "HeavyModel")]
    public async Task Bench_BatchDecoder_LongPrefill() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is not (AcceleratorType.Cuda or AcceleratorType.WebGPU)) throw new UnsupportedTestException("CUDA and WebGPU");
        var http = GetHttpClient() ?? throw new UnsupportedTestException("HttpClient not available");
        var hub = new ModelsHostStreamLoader(http);
        var src = await hub.OpenAsync("Qwen/Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q8_0.gguf", deselect: false);
        var dec = await GgufBatchDecoder.CreateAsync(accelerator, src.Stream, maxSeq: 4096, maxBatch: 1);
        if (accelerator.AcceleratorType == AcceleratorType.WebGPU) dec.CaptureWebGPU = true;
        var tok = SentencePieceTokenizer.FromGGUF(dec.Model)!;
        var paragraph = tok.Encode(string.Concat(Enumerable.Repeat("The lighthouse keeper watched the storm roll in over the grey water, counting the waves and the seconds between the flashes of light. ", 200)));
        var notes = new List<string>();
        var stops = new HashSet<int> { tok.EosId };
        // warm: compile the kernels and the capture
        await dec.GenerateAsync(paragraph[..64], 1, 4, 0f, 1f, 1, stops);
        int[] convo = Array.Empty<int>();
        for (int turn = 1; turn <= 6; turn++)
        {
            convo = convo.Concat(paragraph[..600]).ToArray();
            var sw = Stopwatch.StartNew();
            await dec.PrefillAsync(convo, 0f, reusePrefix: true);
            await accelerator.SynchronizeAsync();
            double ms = sw.Elapsed.TotalMilliseconds;
            notes.Add($"turn {turn}: {convo.Length - dec.LastReusedPrefix} new over {dec.LastReusedPrefix} reused: {ms:F0} ms");
            // the generation's record so the next turn reuses this one's rows (generate one token)
            await dec.GenerateAsync(convo, 1, 1, 0f, 1f, 1, stops, reusePrefix: true);
            convo = convo.Concat(paragraph[600..610]).ToArray();
        }
        foreach (int chunk in new[] { 100000, 256 })
        {
            dec.PrefillChunk = chunk;
            var sw = Stopwatch.StartNew();
            await dec.PrefillAsync(paragraph[..3600], 0f, reusePrefix: false);
            await accelerator.SynchronizeAsync();
            notes.Add($"full prefill of 3600, {(chunk > 4096 ? "one forward" : $"chunks of {chunk}")}: {sw.Elapsed.TotalMilliseconds:F0} ms");
            // 600 new over 3000 reused, timed alone
            await dec.PrefillAsync(paragraph[..3000], 0f, reusePrefix: false);
            await dec.GenerateAsync(paragraph[..3000], 1, 1, 0f, 1f, 1, stops, reusePrefix: true);
            await accelerator.SynchronizeAsync();
            sw.Restart();
            await dec.PrefillAsync(paragraph[..3600], 0f, reusePrefix: true);
            await accelerator.SynchronizeAsync();
            notes.Add($"600 over {dec.LastReusedPrefix}, {(chunk > 4096 ? "one forward" : $"chunks of {chunk}")}: {sw.Elapsed.TotalMilliseconds:F0} ms");
        }
        dec.PrefillChunk = 256;
        if (accelerator.AcceleratorType == AcceleratorType.Cuda)
        {
            dec.ProfileKernels = true;
            dec.PrefillChunk = 100000;
            await dec.PrefillAsync(paragraph[..3000], 0f, reusePrefix: false);
            await dec.GenerateAsync(paragraph[..3000], 1, 1, 0f, 1f, 1, stops, reusePrefix: true);
            dec.KernelTimes.Clear();
            await dec.PrefillAsync(paragraph[..3600], 0f, reusePrefix: true);     // 600 new over 2999 reused, one forward
            dec.ProfileKernels = false;
            notes.Add("600 over ~3k, kernels: " + string.Join(", ", dec.KernelTimes.OrderByDescending(k => k.Value).Take(8).Select(k => $"{k.Key} {k.Value:F1} ms")));
        }
        // qwen3:4b (Q4_K_M), the model SpawnDev.AI's reasoning test lost the device on: a 600-token prefill as one forward and in chunks
        dec.Dispose();
        var src4 = await hub.OpenAsync("Qwen/Qwen3-4B-GGUF", "Qwen3-4B-Q4_K_M.gguf", deselect: false);
        using (var d4 = await GgufBatchDecoder.CreateAsync(accelerator, src4.Stream, maxSeq: 4096, maxBatch: 1))
        {
            await d4.GenerateAsync(paragraph[..64], 1, 2, 0f, 1f, 1, stops);
            foreach (int chunk in new[] { 100000, 256 })
            {
                d4.PrefillChunk = chunk;
                var sw = Stopwatch.StartNew();
                await d4.PrefillAsync(paragraph[..600], 0f, reusePrefix: false);
                await accelerator.SynchronizeAsync();
                notes.Add($"qwen3-4b 600-token prefill, {(chunk > 4096 ? "one forward" : $"chunks of {chunk}")}: {sw.Elapsed.TotalMilliseconds:F0} ms");
            }
        }
        var report = $"{accelerator.AcceleratorType}: " + string.Join(" | ", notes);
        Console.WriteLine("[LongPrefill] " + report);
        return report;   // a measurement (browser lanes print it with PMT_CONSOLE_LOG=LongPrefill)
    });
}
