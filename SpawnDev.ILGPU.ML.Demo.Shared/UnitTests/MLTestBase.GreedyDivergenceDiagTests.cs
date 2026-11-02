using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnDev.ILGPU.ML.Tensors;
using SpawnDev.UnitTesting;
using System.Text.Json;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    /// <summary>
    /// A measurement for Data's report (2026-11-02): Ollama smollm2:360m greedy answers diverge between the session engine and the batch decoder ~90 tokens in,
    /// identically on CUDA and OpenCL. Both engines' greedy token ids for the same prompt (300 tokens), the first divergence, and at that step each engine's top
    /// two logits (the margin) and how far the engines' logits are apart: a margin under that gap is a near-tie (two kernel sets one ulp-chain apart), a margin
    /// well above it is a real difference.
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task Bench_OllamaGreedyDivergence_Margin() => await RunTest(async accelerator =>
    {
        if (OperatingSystem.IsBrowser() || accelerator.AcceleratorType is not (AcceleratorType.Cuda or AcceleratorType.OpenCL))
            throw new UnsupportedTestException("reads the local Ollama cache (CUDA / OpenCL lanes)");
        string root = Environment.GetEnvironmentVariable("OLLAMA_MODELS") is { Length: > 0 } env ? env
            : Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".ollama", "models");
        var manifest = Path.Combine(root, "manifests", "registry.ollama.ai", "library", "smollm2", "360m");
        if (!File.Exists(manifest)) throw new UnsupportedTestException("smollm2:360m not in the Ollama cache");
        string? path = null;
        using (var doc = JsonDocument.Parse(File.ReadAllText(manifest)))
            foreach (var layer in doc.RootElement.GetProperty("layers").EnumerateArray())
                if (layer.GetProperty("mediaType").GetString()!.EndsWith(".model")) path = Path.Combine(root, "blobs", layer.GetProperty("digest").GetString()!.Replace(':', '-'));
        var q = new List<(string, string)> { ("user", "Write a short paragraph about why GPUs are good at running neural networks.") };
        const int Steps = 300;
        static int ArgMax(float[] a) { int b = 0; for (int i = 1; i < a.Length; i++) if (a[i] > a[b]) b = i; return b; }
        static (int, float, int, float) Top2(float[] a)
        {
            int b = ArgMax(a), c = b == 0 ? 1 : 0;
            for (int i = 0; i < a.Length; i++) if (i != b && a[i] > a[c]) c = i;
            return (b, a[b], c, a[c]);
        }

        // the session engine, greedy step by step (its logits kept per step)
        int[] prompt; var sessIds = new List<int>(); var sessRows = new List<float[]>(); SentencePieceTokenizer tok;
        using (var gen = await GgufTextGenerationPipeline.CreateFromFileAsync(accelerator, path!, maxSeqLen: 1024, engine: GgufEngine.Generator))
        {
            var session = gen.Session; tok = gen.Tokenizer;
            (prompt, _) = ChatTemplates.BuildChatPrompt(session.GgufModel!, tok, q, thinking: false);
            using var inBuf = accelerator.Allocate1D<float>(1024);
            async Task<float[]> Step(int[] tokens)
            {
                inBuf.View.SubView(0, tokens.Length).CopyFromCPU(tokens.Select(t => (float)t).ToArray());
                var o = await session.RunDecodeStepAsync(new Dictionary<string, Tensor> { ["input_ids"] = new Tensor(inBuf.View.SubView(0, tokens.Length), new[] { 1, tokens.Length }, "input_ids") });
                var t = o.TryGetValue("logits", out var l) ? l : o.Values.First();
                int vocab = t.Shape[^1], rows = t.ElementCount / vocab;
                return await t.Data.SubView((long)(rows - 1) * vocab, vocab).CopyToHostAsync();
            }
            session.ResetGGUFDecode();
            var row = await Step(prompt);
            for (int s = 0; s < Steps; s++)
            {
                sessRows.Add(row);
                int next = ArgMax(row);
                sessIds.Add(next);
                if (next == tok.EosId) break;
                row = await Step(new[] { next });
            }
        }
        // the batch decoder, greedy
        int[] batchIds;
        using (var fs = new FileStream(path!, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 20, useAsync: true))
        using (var dec = await GgufBatchDecoder.CreateAsync(accelerator, fs, maxSeq: 1024, maxBatch: 1))
        {
            batchIds = (await dec.GenerateAsync(prompt, 1, Steps, 0f, 1f, 1, new HashSet<int> { tok.EosId }))[0];
            int d = 0; while (d < Math.Min(batchIds.Length, sessIds.Count) && batchIds[d] == sessIds[d]) d++;
            string report;
            if (d == Math.Min(batchIds.Length, sessIds.Count)) report = $"{accelerator.AcceleratorType}: no divergence in {d} tokens";
            else
            {
                // the batch decoder's logits at the divergence step: the prompt and the common tokens prefilled, the last row read
                await dec.PrefillAsync(prompt.Concat(batchIds.Take(d)).ToArray(), 0f);
                var b = await dec.ReadLogitsAsync();
                var s = sessRows[d];
                var (sa, sav, sb, sbv) = Top2(s);
                var (ba, bav, bb, bbv) = Top2(b);
                float gap = 0; for (int i = 0; i < s.Length; i++) gap = Math.Max(gap, Math.Abs(s[i] - b[i]));
                string T(int id) => tok.Decode(new[] { id }).Replace("\n", "\\n");
                report = $"{accelerator.AcceleratorType}: first divergence at token {d} | session top-2: {sa} '{T(sa)}' {sav:F4}, {sb} '{T(sb)}' {sbv:F4} (margin {sav - sbv:F4}) | " +
                    $"batch top-2: {ba} '{T(ba)}' {bav:F4}, {bb} '{T(bb)}' {bbv:F4} (margin {bav - bbv:F4}) | the two engines' logits at this step: max |diff| {gap:F4}, " +
                    $"on the two candidates {Math.Abs(s[sa] - b[sa]):F4} / {Math.Abs(s[sb] - b[sb]):F4} | text so far: ...{tok.Decode(batchIds.Take(d).ToArray())[^Math.Min(60, tok.Decode(batchIds.Take(d).ToArray()).Length)..].Replace("\n", " ")}";
            }
            Console.WriteLine("[GreedyDivergence] " + report);
            return report;    // a measurement (run the DemoConsole directly to see it on a desktop lane)
        }
    });
}
