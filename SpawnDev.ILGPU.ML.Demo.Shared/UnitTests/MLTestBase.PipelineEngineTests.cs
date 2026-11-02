using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    /// <summary>
    /// <see cref="GgufTextGenerationPipeline"/> on the batch decoder (<see cref="GgufEngine.Auto"/> picks it for these models on CUDA, OpenCL and WebGPU) against
    /// the same pipeline on <see cref="GgufGenerator"/>: Qwen2.5-0.5B (qwen2), Qwen3-0.6B (qwen3), SmolLM2-360M (llama). The same text for greedy, greedy +
    /// penalty, a short answer (the stop reasons) and a conversation's second turn (the batch engine reusing the first turn's rows); seeded top-k reproducible;
    /// the streamed text equal to the returned one; seeded top-p reproducible on the batch engine (the second run reusing the whole prompt); stop strings on
    /// both engines (the text cut at the match, StopString); a request cancelled after a few tokens (Cancelled, the partial text), then a request that is still
    /// right. Session must throw on the batch engine.
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task GgufTextGenerationPipeline_BatchEngine_MatchesGenerator() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is not (AcceleratorType.Cuda or AcceleratorType.OpenCL or AcceleratorType.WebGPU))
            throw new UnsupportedTestException("the batch engine is chosen on CUDA, OpenCL and WebGPU");
        var http = GetHttpClient() ?? throw new UnsupportedTestException("HttpClient not available");
        var hub = new ModelsHostStreamLoader(http);
        var notes = new List<string>();
        bool ok = true;
        void Note(bool pass, string s) { ok &= pass; notes.Add(pass ? s : s + " (WRONG)"); }
        foreach (var (repo, file) in new[] { ("Qwen/Qwen2.5-0.5B-Instruct-GGUF", "qwen2.5-0.5b-instruct-q8_0.gguf"), ("Qwen/Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q8_0.gguf"),
            ("HuggingFaceTB/SmolLM2-360M-Instruct-GGUF", "smollm2-360m-instruct-q8_0.gguf") })
        {
            var mg = await hub.OpenAsync(repo, file, deselect: false);
            var mb = await hub.OpenAsync(repo, file, deselect: false);
            await using (mg.Stream)
            await using (mb.Stream)
            using (var gen = await GgufTextGenerationPipeline.CreateFromStreamAsync(accelerator, mg.Stream, maxSeqLen: 1024, engine: GgufEngine.Generator))
            using (var bat = await GgufTextGenerationPipeline.CreateFromStreamAsync(accelerator, mb.Stream, maxSeqLen: 1024))
            {
                if (accelerator.AcceleratorType == AcceleratorType.WebGPU) gen.EnableWebGPUDecodeCapture = true;
                string tag = file.Split('-')[0].ToLowerInvariant() + (file.Contains("Qwen3") ? "3" : "");
                bool chose = bat.Engine == GgufEngine.BatchDecoder && gen.Engine == GgufEngine.Generator;
                bool sessionThrows = false; try { _ = bat.Session; } catch (InvalidOperationException) { sessionThrows = true; }
                Note(chose && sessionThrows, $"{file}: Auto -> {bat.Engine}{(bat.EngineReason.Length > 0 ? " (" + bat.EngineReason + ")" : "")}, Session {(sessionThrows ? "throws" : "DOES NOT THROW")}");
                var q = new List<(string, string)> { ("user", "Name three primary colors and say why they are called primary.") };
                async Task Same(string what, GenerationConfig cfg, List<(string, string)> m)
                {
                    string a = await gen.GenerateAsync(m, cfg, thinking: false);
                    var ra = gen.LastStopReason;
                    string streamed = "";
                    string b = await bat.GenerateAsync(m, cfg, thinking: false, onToken: (_, soFar) => { streamed = soFar; return Task.CompletedTask; });
                    var rb = bat.LastStopReason;
                    bool same = a == b && ra == rb, stream = streamed == b;
                    int at = 0; while (at < Math.Min(a.Length, b.Length) && a[at] == b[at]) at++;
                    Note(same && stream, $"{file} {what}: {(same ? "same text" : $"DIFFERS at char {at} ({ra} / {rb})")}{(stream ? "" : ", streamed text DIFFERS")}"
                        + (same ? "" : $" | generator: {Clip(a[Math.Max(0, at - 10)..])} | batch: {Clip(b[Math.Max(0, at - 10)..])}"));
                }
                await Same("greedy", new GenerationConfig { MaxNewTokens = 48, Strategy = "greedy" }, q);
                // an answer that ends on its own (the stop reasons must agree: EOS, or the template's stop token)
                await Same("a short answer", new GenerationConfig { MaxNewTokens = 24, Strategy = "greedy" }, new List<(string, string)> { ("user", "Reply with only the word: yes") });
                await Same("greedy + penalty 1.3", new GenerationConfig { MaxNewTokens = 48, Strategy = "greedy", RepetitionPenalty = 1.3f }, q);
                // seeded top-k: the same text twice on the batch engine (sampled text is NOT compared across engines: their kernels give logits a hair apart, and a
                // draw that lands within that hair of a boundary flips; MEASURED 2026-11-01, top-k 3 T 1.5 seed 11: the uniform 0.4729 against a boundary at 0.4738
                // on the batch engine's logits; the generator's put it the other side. The draw rule itself: GgufBatchDecoder_TopK_MatchesHostSampler)
                var k3 = new GenerationConfig { MaxNewTokens = 48, Strategy = "top_k", TopK = 3, Temperature = 1.5f, Seed = 11 };
                string k1 = await bat.GenerateAsync(q, k3, thinking: false), k2 = await bat.GenerateAsync(q, k3, thinking: false);
                Note(k1 == k2, $"{file} top-k 3 T 1.5 seed 11 twice: {(k1 == k2 ? "same" : "DIFFERS")}");
                // turn 2: the batch engine reuses turn 1's rows (its answer is the same text, re-tokenised)
                string turn1 = await bat.GenerateAsync(q, new GenerationConfig { MaxNewTokens = 32, Strategy = "greedy" }, thinking: false);
                var q2 = new List<(string, string)>(q) { ("assistant", turn1), ("user", "Which one is your favourite?") };
                await gen.GenerateAsync(q, new GenerationConfig { MaxNewTokens = 32, Strategy = "greedy" }, thinking: false);
                await Same("turn 2 (greedy)", new GenerationConfig { MaxNewTokens = 40, Strategy = "greedy" }, q2);
                Note(bat.LastReusedPrefix > 0, $"{file} turn 2: batch engine reused {bat.LastReusedPrefix} rows");
                // seeded top-p on the batch engine: the same text twice (the second run reuses all of the prompt but its last token)
                var topP = new GenerationConfig { MaxNewTokens = 48, Strategy = "top_p", TopP = 0.9f, Temperature = 0.7f, RepetitionPenalty = 1.15f, Seed = 1234 };
                string p1 = await bat.GenerateAsync(q, topP, thinking: false);
                string p2 = await bat.GenerateAsync(q, topP, thinking: false);
                Note(p1 == p2 && bat.LastReusedPrefix > 0, $"{file} top-p seed 1234 twice: {(p1 == p2 ? "same" : "DIFFERS")} (reused {bat.LastReusedPrefix})");

                // a cancel after a few tokens: Cancelled with the partial text; the next request is still right
                using (var cts = new CancellationTokenSource())
                {
                    int n = 0;
                    string partial = await bat.GenerateAsync(q, new GenerationConfig { MaxNewTokens = 48, Strategy = "greedy" }, thinking: false,
                        onToken: (_, _) => { if (++n == 5) cts.Cancel(); return Task.CompletedTask; }, ct: cts.Token);
                    var reason = bat.LastStopReason;
                    string after = await bat.GenerateAsync(q, new GenerationConfig { MaxNewTokens = 48, Strategy = "greedy" }, thinking: false);
                    string want = await gen.GenerateAsync(q, new GenerationConfig { MaxNewTokens = 48, Strategy = "greedy" }, thinking: false);
                    bool cancelOk = reason == StopReason.Cancelled && want.StartsWith(partial, StringComparison.Ordinal) && partial.Length > 0 && after == want;
                    Note(cancelOk, $"{file} cancel after 5 tokens: {reason}, partial '{Clip(partial)}', next request {(after == want ? "right" : "WRONG")}");
                }
                // stop strings, engine level (last: the second GgufGenerator shares the generator pipeline's session): a word from the greedy answer ends both engines' text right before it
                var tok = bat.Tokenizer;
                var (ids, stopIds) = ChatTemplates.BuildChatPrompt(bat.BatchDecoder!.Model, tok, q, thinking: false);
                var greedy = new GenerationConfig { MaxNewTokens = 48, Strategy = "greedy" };
                var engine = new GgufBatchTextEngine(bat.BatchDecoder, tok);
                var full = await engine.GenerateAsync(ids, greedy, stopTokenIds: stopIds);
                var words = full.Text.Split(' ', StringSplitOptions.RemoveEmptyEntries);
                string stopWord = words.Length > 6 ? " " + words[5] : " the";
                var deltas = new System.Text.StringBuilder();
                var cutB = await engine.GenerateAsync(ids, greedy, new[] { stopWord }, stopIds, d => { deltas.Append(d); return Task.CompletedTask; });
                var genEngine = new GgufGenerator(gen.Session, accelerator, gen.Session.GgufModel!, 1024);
                GenerationResult cutG;
                try { cutG = await genEngine.GenerateAsync(ids, greedy, new[] { stopWord }, stopIds); }
                finally { genEngine.Dispose(); }
                int idx = full.Text.IndexOf(stopWord, StringComparison.Ordinal);
                bool cutOk = idx >= 0 && cutB.Text == full.Text[..idx] && cutB.Stop == StopReason.StopString && cutG.Text == cutB.Text && cutG.Stop == StopReason.StopString
                    && deltas.ToString() == cutB.Text;
                Note(cutOk, $"{file} stop string '{stopWord.Trim()}': batch '{Clip(cutB.Text)}' ({cutB.Stop}), generator {(cutG.Text == cutB.Text ? "the same" : $"'{Clip(cutG.Text)}'")} ({cutG.Stop})");

            }
        }
        var report = $"{accelerator.AcceleratorType}: " + string.Join(" || ", notes);
        Console.WriteLine("[PipelineEngine] " + report);
        if (!ok) throw new Exception(report);
        return report;

        static string Clip(string s) { s = s.Replace('\n', ' '); return s[..Math.Min(60, s.Length)]; }
    });
}
