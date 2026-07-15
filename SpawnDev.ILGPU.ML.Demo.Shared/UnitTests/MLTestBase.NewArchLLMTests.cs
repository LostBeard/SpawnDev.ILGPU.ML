using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// End-to-end WebGPU (+ all-backend) gate for the NEW GGUF architectures being wired into the
/// SpawnDev.AI demo: qwen3 (standard transformer + useRMSNorm) and LFM2 (ShortConv hybrid).
/// Each loads a small instruct GGUF via the hub, runs a real greedy decode, and asserts the
/// factual oracle (" Paris"). RunTest fans this across every PMT backend, so the WebGPU pass
/// PROVES the arch (and, for LFM2, the ShortConv kernel's WGSL) works in the browser - the exact
/// path the deployed gh-pages demo takes. (qwen3.5/GatedDeltaNet is verified separately.)
/// Verified GREEN on WebGPU + WebGL + CUDA + OpenCL. HeavyModel (excluded from the default CI sweep;
/// run manually with PMT_EXCLUDE_CATEGORIES=). The CPU-desktop lane is the unoptimized correctness
/// REFERENCE (~12s/token) and is killed by PMT's 10-min console cap on multi-hundred-M LLM decode -
/// an expected backend impracticality (same class as WasmHeavy), NOT a correctness gap; raise
/// PMT_CONSOLE_TIMEOUT_MS to run the CPU E2E deliberately.
/// </summary>
public abstract partial class MLTestBase
{
    // Shared driver: stream a GGUF instruct model from the hub, greedy-decode the France prompt,
    // assert the answer mentions Paris. Network/hub outage → UnsupportedTestException (not a failure).
    private async Task GgufLLM_AnswersParis(Accelerator accelerator, string repoId, string file, string tag,
        string userSuffix = "", int maxTokens = 16)
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");

        var client = new SpawnDev.WebTorrent.WebTorrentClient();
        try
        {
            var hub = new SpawnDev.ILGPU.ML.Hub.HubModelStream(client, http) { PrepareTimeout = TimeSpan.FromMinutes(8) };
            using var cts = new System.Threading.CancellationTokenSource(TimeSpan.FromMinutes(9));

            var model = await hub.OpenAsync(repoId, file, deselect: false, cts.Token);
            if (model.Length < 100_000_000)
                throw new Exception($"[{tag}] hub GGUF length={model.Length}, expected a real model");

            await using (model.Stream)
            using (var pipe = await GgufTextGenerationPipeline.CreateFromStreamAsync(
                accelerator, model.Stream, maxSeqLen: 512, ct: cts.Token))
            {
                Console.WriteLine($"[{tag}] loaded arch={pipe.Architecture} format={pipe.ChatFormat}");
                var messages = new[] { ("user", "What is the capital of France? Answer in one short sentence." + userSuffix) };
                var answer = await pipe.GenerateAsync(messages,
                    config: new GenerationConfig { MaxNewTokens = maxTokens, Strategy = "greedy" }, ct: cts.Token);
                Console.WriteLine($"[{tag}] answer='{answer.Trim()}'");

                if (string.IsNullOrWhiteSpace(answer))
                    throw new Exception($"[{tag}] produced empty output");
                if (!answer.Contains("Paris", StringComparison.OrdinalIgnoreCase))
                    throw new Exception($"[{tag}] answer did not mention Paris (oracle): '{answer.Trim()}'");
            }
        }
        catch (UnsupportedTestException) { throw; }
        catch (Exception ex) when (ex.Message.Contains("No connection") || ex.Message.Contains("network")
            || ex.Message.Contains("magnet") || ex.Message.Contains("preparing") || ex is TimeoutException)
        {
            throw new UnsupportedTestException($"[{tag}] hub/network unavailable: {ex.Message}");
        }
        finally { await client.DisposeAsync(); }
    }

    // qwen3 (0.6B, Q8_0) - standard transformer arch (shares qwen2.5's op set + the useRMSNorm fix).
    // qwen3 is a REASONING model (emits <think>…</think> first); "/no_think" gives the direct answer so the
    // oracle lands quickly. Verified: the WGSL forward is coherent + bit-consistent across all backends.
    [TestMethod(Timeout = 900000, Category = "HeavyModel,WasmHeavy,HeavyCpu", RetryCount = 2)]
    public async Task Pipeline_Qwen3_ViaHubStream_AnswersParis() => await RunTest(async accelerator =>
        await GgufLLM_AnswersParis(accelerator, "Qwen/Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q8_0.gguf", "Qwen3",
            userSuffix: " /no_think", maxTokens: 48));

    // LFM2 (1.2B instruct, Q4_K_M) - ShortConv hybrid. The WebGPU pass proves ShortConv's WGSL in-browser.
    [TestMethod(Timeout = 900000, Category = "HeavyModel,WasmHeavy,HeavyCpu", RetryCount = 2)]
    public async Task Pipeline_Lfm2_ViaHubStream_AnswersParis() => await RunTest(async accelerator =>
        await GgufLLM_AnswersParis(accelerator, "LiquidAI/LFM2-1.2B-GGUF", "LFM2-1.2B-Q4_K_M.gguf", "LFM2"));
}
