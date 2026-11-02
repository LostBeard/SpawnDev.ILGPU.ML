using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnDev.ILGPU.ML.Tensors;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    /// <summary>
    /// <see cref="WebGPUDecodeCapture.TryCaptureAsync"/>'s answer for the step that builds it (<see cref="WebGPUDecodeCapture.StepOutputs"/>) against that step run
    /// directly, on the session after the prompt's prefill: the same argmax and logits (within 1e-3), and the same step run twice at one cursor giving the same
    /// logits (a KV row rewrite is idempotent). The capture used to hand back its P0+1 probe's logits (the token fed twice): every first answer after a load on
    /// WebGPU came from a wrong second token (SmolLM2 "1111...", 2026-11-01). SmolLM2-360M (llama), Qwen2.5-0.5B (qwen2).
    /// </summary>
    [TestMethod(Timeout = 900000, Category = "HeavyModel")]
    public async Task WebGPUDecodeCapture_StepOutputs_MatchTheDirectStep() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType != AcceleratorType.WebGPU) throw new UnsupportedTestException("WebGPU");
        var http = GetHttpClient() ?? throw new UnsupportedTestException("HttpClient not available");
        var hub = new ModelsHostStreamLoader(http);
        var q = new List<(string, string)> { ("user", "Name three primary colors and say why they are called primary.") };
        var notes = new List<string>();
        bool ok = true;
        foreach (var (repo, file) in new[] { ("HuggingFaceTB/SmolLM2-360M-Instruct-GGUF", "smollm2-360m-instruct-q8_0.gguf"), ("Qwen/Qwen2.5-0.5B-Instruct-GGUF", "qwen2.5-0.5b-instruct-q8_0.gguf") })
        {
            var m = await hub.OpenAsync(repo, file, deselect: false);
            await using (m.Stream)
            using (var pipe = await GgufTextGenerationPipeline.CreateFromStreamAsync(accelerator, m.Stream, maxSeqLen: 1024, engine: GgufEngine.Generator))
            {
                var session = pipe.Session;
                var (ids, _) = ChatTemplates.BuildChatPrompt(session.GgufModel!, pipe.Tokenizer, q, thinking: false);
                using var inBuf = accelerator.Allocate1D<float>(1024);
                async Task<float[]> Step(int[] tokens)
                {
                    inBuf.View.SubView(0, tokens.Length).CopyFromCPU(tokens.Select(t => (float)t).ToArray());
                    var o = await session.RunDecodeStepAsync(new Dictionary<string, Tensor> { ["input_ids"] = new Tensor(inBuf.View.SubView(0, tokens.Length), new[] { 1, tokens.Length }, "input_ids") });
                    return await LastRow(o);
                }
                async Task<float[]> LastRow(IReadOnlyDictionary<string, Tensor> o)
                {
                    var t = o.TryGetValue("logits", out var l) ? l : o.Values.First();
                    int vocab = t.Shape[^1], rows = t.ElementCount / vocab;
                    return await t.Data.SubView((long)(rows - 1) * vocab, vocab).CopyToHostAsync();
                }
                static int ArgMax(float[] a) { int b = 0; for (int i = 1; i < a.Length; i++) if (a[i] > a[b]) b = i; return b; }
                static float MaxDiff(float[] a, float[] b) { float d = 0; for (int i = 0; i < a.Length; i++) d = Math.Max(d, Math.Abs(a[i] - b[i])); return d; }

                session.ResetGGUFDecode();
                var pre = await Step(ids);
                int t1 = ArgMax(pre);
                var direct = await Step(new[] { t1 });                           // the reference: the step at P0 = prompt length
                int d2 = ArgMax(direct);

                // the same step twice at the same cursor
                session.ResetGGUFDecode(); await Step(ids);
                await Step(new[] { t1 });
                session.SetGGUFDecodePastLen(ids.Length);
                var again = await Step(new[] { t1 });

                // the capture pass
                session.ResetGGUFDecode(); await Step(ids);
                var cap = await WebGPUDecodeCapture.TryCaptureAsync(session, t1, null);
                string capNote;
                if (cap == null) { capNote = "capture returned null (WRONG)"; ok = false; }
                else
                {
                    var c = await LastRow(cap.StepOutputs);
                    bool good = ArgMax(c) == d2 && MaxDiff(c, direct) <= 1e-3f;
                    ok &= good;
                    capNote = $"capture step argmax {ArgMax(c)} '{pipe.Tokenizer.Decode(new[] { ArgMax(c) })}', max |diff| {MaxDiff(c, direct):G3}{(good ? "" : " (WRONG)")}";
                    cap.Dispose();
                }
                ok &= ArgMax(again) == d2 && MaxDiff(again, direct) <= 1e-3f;
                notes.Add($"{file}: prefill -> {t1} '{pipe.Tokenizer.Decode(new[] { t1 })}'; direct step -> {d2} '{pipe.Tokenizer.Decode(new[] { d2 })}'; " +
                    $"same step again -> {ArgMax(again)} (max |diff| {MaxDiff(again, direct):G3}); {capNote}");
            }
        }
        var report = string.Join(" || ", notes);
        Console.WriteLine("[CaptureStepDiag] " + report);
        if (!ok) throw new Exception(report);
        return report;
    });
}
