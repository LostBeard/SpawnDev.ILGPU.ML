using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    /// <summary>
    /// <see cref="GgufBatchDecoder"/>'s prefix reuse (<c>reusePrefix</c>: a chat's next turn re-sends the conversation, only the new part is prefilled) against a
    /// full prefill of the same prompt, Qwen2.5-0.5B: turn 2 of a conversation (greedy, and seeded top-p + penalty on device draws: a draw depends on the seed,
    /// the sequence and the position, so both must give the same tokens), the same prompt asked again (the reuse stops one token short: the last token's logits
    /// start the answer), a turn after a generation ended early by <c>shouldContinue</c> (the cache record of a short answer), and two sequences (sequence 0's
    /// region is the one reused). Token for token equal; the reused row counts must be what the conversation implies.
    /// </summary>
    [TestMethod(Timeout = 900000, Category = "HeavyModel")]
    public async Task GgufBatchDecoder_PrefixReuse_MatchesFullPrefill() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is not (AcceleratorType.Cuda or AcceleratorType.OpenCL or AcceleratorType.WebGPU))
            throw new UnsupportedTestException("the fast lanes (CUDA, OpenCL, WebGPU)");
        var http = GetHttpClient() ?? throw new UnsupportedTestException("HttpClient not available");
        var hub = new ModelsHostStreamLoader(http);
        var src = await hub.OpenAsync("Qwen/Qwen2.5-0.5B-Instruct-GGUF", "qwen2.5-0.5b-instruct-q8_0.gguf", deselect: false);
        using var dec = await GgufBatchDecoder.CreateAsync(accelerator, src.Stream, maxSeq: 1024, maxBatch: 2);
        if (accelerator.AcceleratorType == AcceleratorType.WebGPU) dec.CaptureWebGPU = true;
        var tok = SentencePieceTokenizer.FromGGUF(dec.Model)!;
        string Text(int[] t) { var d = tok.CreateStreamingDecoder(); var sb = new System.Text.StringBuilder(); foreach (var x in t) sb.Append(d.Push(x)); sb.Append(d.Finish()); return sb.ToString(); }
        (int[] Ids, HashSet<int> Stops) Prompt(List<(string, string)> m)
        {
            var (ids, stops) = ChatTemplates.BuildChatPrompt(dec.Model, tok, m, thinking: false);
            var set = stops.ToHashSet(); if (tok.EosId >= 0) set.Add(tok.EosId);
            return (ids, set);
        }
        var notes = new List<string>();
        bool ok = true;
        void Check(string name, int[] reused, int[] full, int keptRows, Func<int, bool> keptOk)
        {
            bool same = reused.SequenceEqual(full), kept = keptOk(keptRows);
            ok &= same && kept;
            int at = 0; while (at < Math.Min(reused.Length, full.Length) && reused[at] == full[at]) at++;
            notes.Add($"{name}: {(same ? "identical" : $"DIFFERS at token {at}")} ({reused.Length} tokens), kept {keptRows} rows{(kept ? "" : " (WRONG)")}" +
                (same ? "" : $" | reused: {Text(reused)[..Math.Min(80, Text(reused).Length)]} | full: {Text(full)[..Math.Min(80, Text(full).Length)]}"));
        }

        foreach (var (mode, T, P, pen) in new[] { ("greedy", 0f, 1f, 1f), ("top-p 0.9 T 0.7 + penalty 1.15", 0.7f, 0.9f, 1.15f) })
        {
            // turn 1, then turn 2 with the answer and a new question appended
            var turn1 = new List<(string, string)> { ("user", "Name a large mammal that lives in the ocean.") };
            var (p1, stops) = Prompt(turn1);
            var a1 = (await dec.GenerateAsync(p1, 1, 40, T, P, 7, stops, repetitionPenalty: pen, reusePrefix: true))[0];
            var turn2 = new List<(string, string)>(turn1) { ("assistant", Text(a1)), ("user", "How much does it weigh?") };
            var (p2, _) = Prompt(turn2);
            var b = (await dec.GenerateAsync(p2, 1, 40, T, P, 7, stops, repetitionPenalty: pen, reusePrefix: true))[0];
            int kept2 = dec.LastReusedPrefix;
            var bFull = (await dec.GenerateAsync(p2, 1, 40, T, P, 7, stops, repetitionPenalty: pen, reusePrefix: false))[0];
            // turn 2 shares turn 1's prompt at least (the answer's re-tokenised text may differ from its tokens past some point)
            Check($"{mode}, turn 2", b, bFull, kept2, k => k >= p1.Length && k < p2.Length);

            // the same prompt again: everything but the last token is reused
            var again = (await dec.GenerateAsync(p2, 1, 40, T, P, 7, stops, repetitionPenalty: pen, reusePrefix: true))[0];
            Check($"{mode}, same prompt again", again, bFull, dec.LastReusedPrefix, k => k == p2.Length - 1);
        }

        // a generation ended early by shouldContinue, then a turn that reuses its rows
        {
            var t1 = new List<(string, string)> { ("user", "Count from one to twenty in words.") };
            var (p1, stops) = Prompt(t1);
            var shortA = (await dec.GenerateAsync(p1, 1, 60, 0f, 1f, 1, stops, reusePrefix: true, shouldContinue: (_, _) => ValueTask.FromResult(false)))[0];
            bool stoppedAtOne = shortA.Length == 1;
            int taken = 0;
            var five = (await dec.GenerateAsync(p1, 1, 60, 0f, 1f, 1, stops, reusePrefix: true, shouldContinue: (_, _) => ValueTask.FromResult(++taken < 5)))[0];
            bool stoppedAtFive = five.Length == 5;
            ok &= stoppedAtOne && stoppedAtFive;
            notes.Add($"shouldContinue: {(stoppedAtOne && stoppedAtFive ? "ended the generations at 1 and 5 tokens" : $"WRONG ({shortA.Length}, {five.Length} tokens)")}");
            var t2 = new List<(string, string)>(t1) { ("assistant", Text(five)), ("user", "Now in French.") };
            var (p2, _) = Prompt(t2);
            var r = (await dec.GenerateAsync(p2, 1, 30, 0f, 1f, 1, stops, reusePrefix: true))[0];
            int kept = dec.LastReusedPrefix;
            var f = (await dec.GenerateAsync(p2, 1, 30, 0f, 1f, 1, stops, reusePrefix: false))[0];
            Check("after a short answer", r, f, kept, k => k >= p1.Length && k < p2.Length);
        }

        // a generation ended by shouldContinue (as a stop string ends one) on host draws: its last token was never fed (no step is queued ahead), so its
        // row is unwritten; a next prompt that is the last prompt plus the answer's exact tokens plus more reuses right up to that token and must NOT claim it
        {
            bool devDraws = dec.DeviceSampling;
            dec.DeviceSampling = false;
            try
            {
                var t1 = new List<(string, string)> { ("user", "List five fruits.") };
                var (p1, stops) = Prompt(t1);
                int n = 0;
                var a = (await dec.GenerateAsync(p1, 1, 24, 0f, 1f, 1, stops, reusePrefix: true, shouldContinue: (_, _) => ValueTask.FromResult(++n < 6)))[0];
                var p2 = p1.Concat(a).Concat(tok.Encode(" And five vegetables?")).ToArray();
                var r = (await dec.GenerateAsync(p2, 1, 24, 0f, 1f, 1, stops, reusePrefix: true))[0];
                int kept = dec.LastReusedPrefix;
                var f = (await dec.GenerateAsync(p2, 1, 24, 0f, 1f, 1, stops, reusePrefix: false))[0];
                Check("host draws, a stopped answer's exact tokens continued", r, f, kept, k => a.Length == 6 && k == p1.Length + a.Length - 1);
            }
            finally { dec.DeviceSampling = devDraws; }
        }

        // two sequences: sequence 0's region holds the conversation that is reused
        {
            var t1 = new List<(string, string)> { ("user", "Give me one word for happy.") };
            var (p1, stops) = Prompt(t1);
            var two = await dec.GenerateAsync(p1, 2, 20, 0f, 1f, 1, stops, reusePrefix: true);
            var t2 = new List<(string, string)>(t1) { ("assistant", Text(two[0])), ("user", "And one for sad?") };
            var (p2, _) = Prompt(t2);
            var r = (await dec.GenerateAsync(p2, 1, 20, 0f, 1f, 1, stops, reusePrefix: true))[0];
            int kept = dec.LastReusedPrefix;
            var f = (await dec.GenerateAsync(p2, 1, 20, 0f, 1f, 1, stops, reusePrefix: false))[0];
            Check("after two sequences", r, f, kept, k => k >= p1.Length && k < p2.Length);
        }

        var report = $"{accelerator.AcceleratorType}: " + string.Join(" | ", notes);
        Console.WriteLine("[BatchPrefix] " + report);
        if (!ok) throw new Exception(report);
        return report;
    });

    /// <summary>
    /// <see cref="GgufBatchDecoder"/>'s top-k draw (<c>SampleAsync(row, topP, rng, topK)</c>: from the candidates up to <see cref="GgufBatchDecoder.Candidates"/>,
    /// the whole row past them) against <see cref="GgufBatchDecoder.SampleLogits(float[], int, int, float, float, double, int)"/> on the same logits with the same
    /// uniform: Qwen2.5-0.5B's prefill logits for a chat prompt, k 1, 2, 3, 5, 40, 64 and 100 (past the candidates), T 0.7 and 1.5, 200 uniforms each. Exactly equal
    /// (the candidates are the row's own values).
    /// </summary>
    [TestMethod(Timeout = 600000, Category = "HeavyModel")]
    public async Task GgufBatchDecoder_TopK_MatchesHostSampler() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is not (AcceleratorType.Cuda or AcceleratorType.OpenCL or AcceleratorType.WebGPU))
            throw new UnsupportedTestException("the fast lanes (CUDA, OpenCL, WebGPU)");
        var http = GetHttpClient() ?? throw new UnsupportedTestException("HttpClient not available");
        var hub = new ModelsHostStreamLoader(http);
        var src = await hub.OpenAsync("Qwen/Qwen2.5-0.5B-Instruct-GGUF", "qwen2.5-0.5b-instruct-q8_0.gguf", deselect: false);
        using var dec = await GgufBatchDecoder.CreateAsync(accelerator, src.Stream, maxSeq: 512, maxBatch: 1);
        var tok = SentencePieceTokenizer.FromGGUF(dec.Model)!;
        var (ids, _) = ChatTemplates.BuildChatPrompt(dec.Model, tok, new List<(string, string)> { ("user", "Name three primary colors and say why they are called primary.") }, thinking: false);
        var notes = new List<string>();
        bool ok = true;
        foreach (float T in new[] { 0.7f, 1.5f })
        {
            await dec.PrefillAsync(ids, T);
            var row = await dec.ReadLogitsAsync();
            foreach (int k in new[] { 1, 2, 3, 5, 40, 64, 100 })
            {
                int bad = 0, distinct = 0; var seen = new HashSet<int>();
                for (int i = 0; i < 200; i++)
                {
                    int got = await dec.SampleAsync(0, 1f, new Random(1000 + i), k);
                    int want = GgufBatchDecoder.SampleLogits(row, 0, row.Length, T, 1f, new Random(1000 + i).NextDouble(), k);
                    if (got != want) bad++;
                    if (seen.Add(got)) distinct++;
                }
                ok &= bad == 0;
                notes.Add($"T {T} k {k}: {(bad == 0 ? "equal" : $"{bad} of 200 WRONG")} ({distinct} distinct tokens)");
            }
        }
        var report = $"{accelerator.AcceleratorType}: " + string.Join("; ", notes);
        Console.WriteLine("[BatchTopK] " + report);
        if (!ok) throw new Exception(report);
        return report;
    });

    /// <summary>
    /// <see cref="GgufBatchDecoder.PrefillChunk"/>: a ~700-token prompt prefilled as one forward, in 256-row chunks and in 64-row chunks (each chunk its own
    /// WebGPU submission), then 32 greedy tokens: the same tokens, and the prefill's last-row logits within 1e-3. Then a turn whose new part (~700 tokens) runs
    /// in chunks over a reused prefix: the same tokens as a full one-forward prefill. Qwen3-0.6B. Added after SpawnDev.AI lost the WebGPU device on a ~600-token
    /// turn over a ~3k history prefilled as one forward (2026-11-01).
    /// </summary>
    [TestMethod(Timeout = 900000, Category = "HeavyModel")]
    public async Task GgufBatchDecoder_ChunkedPrefill_MatchesOneForward() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is not (AcceleratorType.Cuda or AcceleratorType.OpenCL or AcceleratorType.WebGPU))
            throw new UnsupportedTestException("the fast lanes (CUDA, OpenCL, WebGPU)");
        var http = GetHttpClient() ?? throw new UnsupportedTestException("HttpClient not available");
        var hub = new ModelsHostStreamLoader(http);
        var src = await hub.OpenAsync("Qwen/Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q8_0.gguf", deselect: false);
        using var dec = await GgufBatchDecoder.CreateAsync(accelerator, src.Stream, maxSeq: 2048, maxBatch: 1);
        var tok = SentencePieceTokenizer.FromGGUF(dec.Model)!;
        var text = string.Concat(Enumerable.Repeat("The lighthouse keeper watched the storm roll in over the grey water, counting the waves and the seconds between the flashes. ", 30));
        var (ids, stops) = ChatTemplates.BuildChatPrompt(dec.Model, tok, new List<(string, string)> { ("user", text + "Summarise this in one sentence.") }, thinking: false);
        var stopSet = stops.ToHashSet(); if (tok.EosId >= 0) stopSet.Add(tok.EosId);
        var notes = new List<string> { $"prompt {ids.Length} tokens" };
        bool ok = true;
        async Task<(int[] Tokens, float[] Logits)> Run(int chunk)
        {
            dec.PrefillChunk = chunk;
            await dec.PrefillAsync(ids, 0f);
            var logits = await dec.ReadLogitsAsync();
            var t = (await dec.GenerateAsync(ids, 1, 32, 0f, 1f, 1, stopSet))[0];
            return (t, logits);
        }
        var one = await Run(100000);
        foreach (int chunk in new[] { 256, 64 })
        {
            var r = await Run(chunk);
            float d = 0; for (int i = 0; i < one.Logits.Length; i++) d = Math.Max(d, Math.Abs(one.Logits[i] - r.Logits[i]));
            bool same = r.Tokens.SequenceEqual(one.Tokens) && d <= 1e-3f;
            ok &= same;
            notes.Add($"chunks of {chunk}: {(same ? "same tokens" : "DIFFERS")}, last-row logits max |diff| {d:G3}");
        }
        // a turn whose long new part runs in chunks over a reused prefix
        dec.PrefillChunk = 256;
        var a1 = (await dec.GenerateAsync(ids, 1, 16, 0f, 1f, 1, stopSet, reusePrefix: true))[0];
        var (ids2, _) = ChatTemplates.BuildChatPrompt(dec.Model, tok, new List<(string, string)> { ("user", text + "Summarise this in one sentence."),
            ("assistant", tok.Decode(a1)), ("user", text + "Now count the sentences.") }, thinking: false);
        var reused = (await dec.GenerateAsync(ids2, 1, 24, 0f, 1f, 1, stopSet, reusePrefix: true))[0];
        int kept = dec.LastReusedPrefix;
        dec.PrefillChunk = 100000;
        var full = (await dec.GenerateAsync(ids2, 1, 24, 0f, 1f, 1, stopSet, reusePrefix: false))[0];
        bool turnOk = reused.SequenceEqual(full) && kept > 0 && ids2.Length - kept > 256;
        ok &= turnOk;
        notes.Add($"turn 2: {ids2.Length - kept} new over {kept} reused in chunks of 256: {(turnOk ? "same tokens as one full forward" : "DIFFERS")}");
        var report = $"{accelerator.AcceleratorType}: " + string.Join("; ", notes);
        Console.WriteLine("[ChunkedPrefill] " + report);
        if (!ok) throw new Exception(report);
        return report;
    });

    /// <summary>
    /// <see cref="GgufBatchDecoder.FlashPrefill"/> (the prefill's tiled attention) against the three-kernel path it replaced for prefills: Qwen3-0.6B (head width
    /// 128) and Qwen2.5-0.5B (64, two kv heads for fourteen query heads), a ~740-token prompt prefilled in one forward, then a second ~770-token part over it
    /// (rows that start past position 0); the last row's logits within 1e-3 and the same 24 greedy tokens.
    /// </summary>
    [TestMethod(Timeout = 900000, Category = "HeavyModel")]
    public async Task GgufBatchDecoder_FlashPrefill_MatchesThreeKernelPath() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is not (AcceleratorType.Cuda or AcceleratorType.OpenCL or AcceleratorType.WebGPU))
            throw new UnsupportedTestException("the fast lanes (CUDA, OpenCL, WebGPU)");
        var http = GetHttpClient() ?? throw new UnsupportedTestException("HttpClient not available");
        var hub = new ModelsHostStreamLoader(http);
        var notes = new List<string>();
        bool ok = true;
        var text = string.Concat(Enumerable.Repeat("The lighthouse keeper watched the storm roll in over the grey water, counting the waves and the seconds between the flashes. ", 30));
        // F32 caches: this compares two attention kernels, so their inputs must not carry 16-bit rounding. Each run computes its own prefill; with F16 caches a
        // layer's float-order difference (1e-7) moves the next layer's K / V across F16 rounding boundaries (a 1e-3 step), MEASURED 2026-11-02 at 4-7e-3 in the
        // logits. The F16 cache is gated on its own (GgufBatchDecoder_HalfKvCache_MatchesF32).
        bool halfSaved = GgufBatchDecoder.HalfKvCache;
        GgufBatchDecoder.HalfKvCache = false;
        try
        {
        foreach (var (repo, file) in new[] { ("Qwen/Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q8_0.gguf"), ("Qwen/Qwen2.5-0.5B-Instruct-GGUF", "qwen2.5-0.5b-instruct-q8_0.gguf") })
        {
            var src = await hub.OpenAsync(repo, file, deselect: false);
            using var dec = await GgufBatchDecoder.CreateAsync(accelerator, src.Stream, maxSeq: 2048, maxBatch: 1);
            dec.PrefillChunk = 100000;
            var tok = SentencePieceTokenizer.FromGGUF(dec.Model)!;
            var (ids, stops) = ChatTemplates.BuildChatPrompt(dec.Model, tok, new List<(string, string)> { ("user", text + "Summarise this in one sentence.") }, thinking: false);
            var stopSet = stops.ToHashSet(); if (tok.EosId >= 0) stopSet.Add(tok.EosId);
            var more = tok.Encode(text + " And again, briefly.");
            async Task<(float[] L1, float[] L2, int[] T)> Run(bool flash)
            {
                dec.FlashPrefill = flash;
                await dec.PrefillAsync(ids, 0f);
                var l1 = await dec.ReadLogitsAsync();
                var t = (await dec.GenerateAsync(ids, 1, 24, 0f, 1f, 1, stopSet, reusePrefix: false))[0];
                var all = ids.Concat(t).Concat(more).ToArray();
                await dec.GenerateAsync(ids, 1, 24, 0f, 1f, 1, stopSet, reusePrefix: false);
                await dec.PrefillAsync(all, 0f, reusePrefix: true);                      // the second part over the first's rows
                var l2 = await dec.ReadLogitsAsync();
                return (l1, l2, t);
            }
            var flash = await Run(true);
            var three = await Run(false);
            static float Diff(float[] a, float[] b) { float d = 0; for (int i = 0; i < a.Length; i++) d = Math.Max(d, Math.Abs(a[i] - b[i])); return d; }
            float d1 = Diff(flash.L1, three.L1), d2 = Diff(flash.L2, three.L2);
            bool good = d1 <= 1e-3f && d2 <= 1e-3f && flash.T.SequenceEqual(three.T);
            ok &= good;
            notes.Add($"{file}: prompt {ids.Length}, logits max |diff| {d1:G3}, second part ({more.Length} over the first) {d2:G3}, tokens {(flash.T.SequenceEqual(three.T) ? "same" : "DIFFER")}{(good ? "" : " (WRONG)")}");
        }
        }
        finally { GgufBatchDecoder.HalfKvCache = halfSaved; }
        var report = $"{accelerator.AcceleratorType}: " + string.Join("; ", notes);
        Console.WriteLine("[FlashPrefill] " + report);
        if (!ok) throw new Exception(report);
        return report;
    });
}
