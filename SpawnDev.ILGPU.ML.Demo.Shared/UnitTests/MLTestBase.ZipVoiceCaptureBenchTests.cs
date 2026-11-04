using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.UnitTesting;
using System.Text;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    /// <summary>
    /// ZipVoice decoder step, captured vs direct (int8 graphs, the fidelity test's fixed inputs): the steady-state direct step, the capture setup
    /// (calls until the plan is live: observe + warm + record), the steady-state replayed step, and how many output values the replay changes
    /// (must be 0). CUDA and WebGPU (the backends with graph capture). Prints "[Benchmark] ZipVoiceCapture [...]"; read it with
    /// PMT_CONSOLE_LOG=ZipVoiceCapture on the browser lane. Added 2026-11-04 (capture made faithful in 5.8.2-local.4).
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task Bench_ZipVoice_DecoderCaptureVsDirect() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is not (AcceleratorType.WebGPU or AcceleratorType.Cuda))
            throw new UnsupportedTestException($"graph capture is CUDA and WebGPU only; {accelerator.AcceleratorType} has nothing to replay");
        using var http = CreateHuggingFaceHttpClient();
        var encoderBytes = await InferenceSession.DownloadBytesChunkedAsync(http, HuggingFaceClient.GetDownloadUrl(ZipVoiceRepo, "zipvoice_distill/text_encoder_int8.onnx"));
        var decoderBytes = await InferenceSession.DownloadBytesChunkedAsync(http, HuggingFaceClient.GetDownloadUrl(ZipVoiceRepo, "zipvoice_distill/fm_decoder_int8.onnx"));
        var tokensBytes = await InferenceSession.DownloadBytesChunkedAsync(http, HuggingFaceClient.GetDownloadUrl(ZipVoiceRepo, "zipvoice_distill/tokens.txt"));
        var vocoderBytes = await InferenceSession.DownloadBytesChunkedAsync(http, ArchiveMemberUrl(VocoderArchive, VocoderMember));
        var tokenizer = ZipVoiceTokenizer.CreateFromTokens(Encoding.UTF8.GetString(tokensBytes));
        using var graphs = IlgpuZipVoiceGraphs.Create(accelerator, encoderBytes, decoderBytes, vocoderBytes);
        var encoding = await graphs.RunEncoderAsync(tokenizer.Encode("Paint the sockets in the wall dull green."), tokenizer.Encode(LibrivoxTranscript), 96, 1.0f);
        int numFrames = encoding.NumFrames, featDim = encoding.FeatDim, count = numFrames * featDim;
        var x = Pseudo(count, 11);
        var speech = Pseudo(count, 22);
        Task<float[]> Step() => graphs.RunDecoderAsync(0.25f, x, encoding.TextCondition, speech, 1.0f, numFrames, featDim);
        const int Steps = 8;

        graphs.EnableGraphCapture = false;
        await Step(); await Step();                                   // compile + warm
        var sw = System.Diagnostics.Stopwatch.StartNew();
        float[] direct = Array.Empty<float>();
        for (int i = 0; i < Steps; i++) direct = await Step();       // RunDecoderAsync reads the output back: each step is complete
        double directMs = sw.Elapsed.TotalMilliseconds / Steps;

        graphs.EnableGraphCapture = true;
        graphs.AllowControlFlowCapture = true;
        sw.Restart();
        int setupCalls = 0;
        while (!graphs.DecoderCaptured && setupCalls < 8) { await Step(); setupCalls++; }
        double setupMs = sw.Elapsed.TotalMilliseconds;
        if (!graphs.DecoderCaptured)
            throw new Exception($"capture never went live after {setupCalls} calls: {graphs.DecoderCaptureStatus}");
        await Step();                                                 // first replay
        sw.Restart();
        float[] replay = Array.Empty<float>();
        for (int i = 0; i < Steps; i++) replay = await Step();
        double replayMs = sw.Elapsed.TotalMilliseconds / Steps;
        int differ = 0;
        for (int i = 0; i < count; i++) if (BitConverter.SingleToInt32Bits(replay[i]) != BitConverter.SingleToInt32Bits(direct[i])) differ++;
        var line = $"{accelerator.AcceleratorType}: direct {directMs:F1} ms/step, capture setup {setupMs:F0} ms over {setupCalls} calls, replay {replayMs:F1} ms/step "
            + $"({directMs / replayMs:F2}x), replay differs from direct in {differ} of {count} values";
        Console.WriteLine($"[Benchmark] ZipVoiceCapture [{accelerator.AcceleratorType}] {line}");
        if (differ != 0) throw new Exception("captured replay is not bit-identical to the direct step: " + line);
        return line;
    });
}
