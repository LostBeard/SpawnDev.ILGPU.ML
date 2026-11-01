using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    /// <summary>
    /// <see cref="GgufBatchDecoder"/> against the engine it batches (Qwen2.5-0.5B-Instruct Q8_0, the models host): its greedy answer must be the existing generator's
    /// (<see cref="GgufTextGenerationPipeline"/>, itself checked against Ollama) word for word; four greedy sequences decoded together must all equal it (the batched step, the
    /// forked caches and the multi-row GEMV change nothing); four sampled sequences (different seeds) must not all agree. Every draw from the device's candidates
    /// (<see cref="GgufBatchDecoder.SampleAsync"/>) over 24 sampled steps of four sequences must equal <see cref="GgufBatchDecoder.SampleLogits"/> on the full logits row
    /// with the same generator state, and at most a tenth of them may need the full row (the fallback). Device draws (<see cref="GgufBatchDecoder.Draw"/>, the default
    /// for <see cref="GgufBatchDecoder.GenerateAsync"/>): greedy must equal the host draws; over 24 steps of four sequences each draw must equal
    /// <see cref="GgufBatchDecoder.SampleLogits(float[], int, int, float, float, double)"/> with its <see cref="GgufBatchDecoder.DrawUniform"/> (candidates path, whole
    /// row forced, a flat T 1.5 row), a float tie allowed within <see cref="TieWindow"/> of a boundary; the pipelined GenerateAsync must equal an unpipelined loop
    /// riding all four sequences along. Times of one and four sequences, device and host draws, are reported.
    /// </summary>
    [TestMethod(Timeout = 900000, Category = "HeavyModel")]
    public async Task GgufBatchDecoder_GreedyMatchesGenerator() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is AcceleratorType.WebGL or AcceleratorType.Wasm or AcceleratorType.CPU or AcceleratorType.Velocity)
            throw new UnsupportedTestException("a GPU decode test");
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");
        var hub = new ModelsHostStreamLoader(http);
        var messages = new List<(string, string)> { ("user", "Name three primary colors and say why they are called primary.") };
        const int maxNew = 48;
        string want;
        {
            var m = await hub.OpenAsync("Qwen/Qwen2.5-0.5B-Instruct-GGUF", "qwen2.5-0.5b-instruct-q8_0.gguf", deselect: false);
            await using (m.Stream)
            using (var pipe = await GgufTextGenerationPipeline.CreateFromStreamAsync(accelerator, m.Stream, maxSeqLen: 512))
                want = await pipe.GenerateAsync(messages, new GenerationConfig { MaxNewTokens = maxNew, Strategy = "greedy" }, thinking: false);
        }
        var src = await hub.OpenAsync("Qwen/Qwen2.5-0.5B-Instruct-GGUF", "qwen2.5-0.5b-instruct-q8_0.gguf", deselect: false);
        using var dec = await GgufBatchDecoder.CreateAsync(accelerator, src.Stream, maxSeq: 1024, maxBatch: 4);
        var tok = SentencePieceTokenizer.FromGGUF(dec.Model)!;
        var (ids, stops) = ChatTemplates.BuildChatPrompt(dec.Model, tok, messages, thinking: false);
        var stopSet = stops.ToHashSet(); if (tok.EosId >= 0) stopSet.Add(tok.EosId);
        string Text(int[] t) { var d = tok.CreateStreamingDecoder(); var sb = new System.Text.StringBuilder(); foreach (var x in t) sb.Append(d.Push(x)); sb.Append(d.Finish()); return sb.ToString(); }
        var sw = System.Diagnostics.Stopwatch.StartNew();
        var one = await dec.GenerateAsync(ids, 1, maxNew, 0f, 1f, 1, stopSet);
        double t1 = sw.Elapsed.TotalSeconds;
        sw.Restart();
        var four = await dec.GenerateAsync(ids, 4, maxNew, 0f, 1f, 1, stopSet);
        double t4 = sw.Elapsed.TotalSeconds;
        sw.Restart();
        var oneWarm = await dec.GenerateAsync(ids, 1, maxNew, 0f, 1f, 1, stopSet);           // warm (kernels compiled, WebGPU plans captured): the single-sequence speed
        double t1w = sw.Elapsed.TotalSeconds;
        // the host-draw path (a round trip per step) for the A/B, and greedy must agree with the device draws
        dec.DeviceSampling = false;
        await dec.GenerateAsync(ids, 1, maxNew, 0f, 1f, 1, stopSet);
        sw.Restart();
        var oneHost = await dec.GenerateAsync(ids, 1, maxNew, 0f, 1f, 1, stopSet);
        double t1h = sw.Elapsed.TotalSeconds;
        var dt = dec.DecodeTimings;
        sw.Restart();
        var fourHost = await dec.GenerateAsync(ids, 4, maxNew, 0f, 1f, 1, stopSet);
        double t4h = sw.Elapsed.TotalSeconds;
        dec.DeviceSampling = true;
        sw.Restart();
        var fourWarm = await dec.GenerateAsync(ids, 4, maxNew, 0f, 1f, 1, stopSet);
        double t4w = sw.Elapsed.TotalSeconds;
        bool hostSame = oneHost[0].SequenceEqual(oneWarm[0]) && fourHost.All(r => r.SequenceEqual(oneWarm[0])) && fourWarm.All(r => r.SequenceEqual(oneWarm[0]));
        // device draws against the host sampler with the same uniform (DrawUniform), draw for draw, over 24 steps of four sequences: the candidates path (T 0.8,
        // top-p 0.95), the full-row path forced, and a flat distribution (T 1.5, top-p 0.99) whose nucleus often outgrows the candidates. A draw within 1e-4 of a
        // boundary (u or top-p moved by 1e-4 gives the device's token) is counted as a float tie, not a mismatch.
        var badNotes = new List<string>();
        double maxTie = 0;
        async Task<(int Draws, int Bad, int Ties, int Wide)> CheckDraws(float T, float topP, int seed, bool force)
        {
            dec.ForceFullRowDraws = force;
            await dec.PrefillAsync(ids, T);
            dec.Fork(4);
            dec.Draw(topP, seed);
            var all = new[] { 0, 1, 2, 3 };
            int draws = 0, bad = 0, ties = 0, wideRows = 0;
            for (int step = 0; step < 24; step++)
            {
                var full = await dec.ReadLogitsAsync();
                var got = await dec.ReadDrawnAsync();
                for (int s = 0; s < 4; s++)
                {
                    int row = step == 0 ? 0 : s;
                    double u = GgufBatchDecoder.DrawUniform(seed, s, dec.Position);
                    int wantTok = GgufBatchDecoder.SampleLogits(full, row, dec.Vocab, T, topP, u);
                    draws++;
                    if (NucleusSize(full, row, dec.Vocab, T, topP) > GgufBatchDecoder.Candidates) wideRows++;
                    if (got[s] == wantTok) continue;
                    // a float tie: u lies within TieWindow of the device token's interval (the device sums in float, the host in double; the token next to a
                    // boundary u sits on can go either way), and that token is in the nucleus (or in the one top-p + TieWindow makes)
                    var di = DrawInterval(full, row, dec.Vocab, T, Math.Min(1f, topP + (float)TieWindow), got[s]);
                    var dn = DrawInterval(full, row, dec.Vocab, T, topP, got[s]);
                    double dist = Math.Max(0, Math.Max(dn.Lo - u, u - dn.Hi));
                    if (di.Rank >= 0 && di.Rank < di.Nucleus && dist <= TieWindow) { ties++; maxTie = Math.Max(maxTie, dist); }
                    else
                    {
                        bad++;
                        badNotes.Add($"[T {T} top-p {topP}{(force ? " forced" : "")} step {step} seq {s}: u {u:R}, host {wantTok} {RankNote(full, row, dec.Vocab, T, topP, wantTok, u)}, device {got[s]} {RankNote(full, row, dec.Vocab, T, topP, got[s], u)}]");
                    }
                }
                await dec.StepDrawnAsync(all, T);
            }
            await dec.ReadDrawnAsync();
            dec.ForceFullRowDraws = false;
            return (draws, bad, ties, wideRows);
        }
        var drawCand = await CheckDraws(0.8f, 0.95f, 41, false);
        var drawForced = await CheckDraws(0.8f, 0.95f, 42, true);
        var drawFlat = await CheckDraws(1.5f, 0.99f, 43, false);
        bool drawsOk = new[] { drawCand, drawForced, drawFlat }.All(c => c.Bad == 0 && c.Ties * 20 <= c.Draws) && drawFlat.Wide > 0;
        string DrawNote((int Draws, int Bad, int Ties, int Wide) c) => $"{c.Draws - c.Bad - c.Ties}/{c.Draws} equal, {c.Ties} float ties, {c.Bad} wrong, {c.Wide} nuclei wider than the candidates";
        var sampled = await dec.GenerateAsync(ids, 4, maxNew, 0.8f, 0.95f, 11, stopSet);
        // the device candidates against the full-row sampler, draw for draw
        int draws = 0, mismatches = 0;
        await dec.PrefillAsync(ids, 0.8f);
        dec.Fork(4);
        var nextTok = new int[4];
        for (int step = 0; step < 24; step++)
        {
            var full = await dec.ReadLogitsAsync();
            for (int s = 0; s < 4; s++)
            {
                int row = step == 0 ? 0 : s, seed = 1000 * step + s;
                int a = await dec.SampleAsync(row, 0.95f, new Random(seed));
                int b = GgufBatchDecoder.SampleLogits(full, row, dec.Vocab, 0.8f, 0.95f, new Random(seed));
                draws++; if (a != b) mismatches++;
                nextTok[s] = b;
            }
            await dec.StepAsync(nextTok, 0.8f);
        }
        int wide = dec.RowReads;
        // finished sequences leave the batch: GenerateAsync must equal a loop that keeps stepping all four (each row's arithmetic does not depend on the others),
        // with lengths that differ so the batch really shrank
        const int longNew = 160;
        var compacted = await dec.GenerateAsync(ids, 4, longNew, 0.8f, 0.95f, 21, stopSet);
        // (device draws, unpipelined: a draw is a function of seed, sequence and position, so riding along draws the same tokens; GenerateAsync also runs a step ahead)
        var ridden = new List<int>[4];
        {
            await dec.PrefillAsync(ids, 0.8f);
            dec.Fork(4);
            dec.Draw(0.95f, 21);
            var fin = new bool[4];
            var all = new[] { 0, 1, 2, 3 };
            for (int s = 0; s < 4; s++) ridden[s] = new List<int>();
            while (true)
            {
                var nx = await dec.ReadDrawnAsync();
                for (int s = 0; s < 4; s++)
                {
                    if (fin[s]) continue;
                    if (stopSet.Contains(nx[s]) || ridden[s].Count >= longNew) { fin[s] = true; continue; }
                    ridden[s].Add(nx[s]);
                }
                if (fin.All(f => f)) break;
                await dec.StepDrawnAsync(all, 0.8f);
            }
        }
        bool compactSame = Enumerable.Range(0, 4).All(s => compacted[s].SequenceEqual(ridden[s]));
        // the split-K attention against the three-kernel path past one chunk of keys (a ~600-token prompt: 3 chunks): the prefill's logits and, after four
        // decode steps of four sequences, every row's logits
        var longMsgs = new List<(string, string)> { ("user", string.Concat(Enumerable.Repeat("Name three primary colors and say why they are called primary. ", 40))) };
        var (longIds, _) = ChatTemplates.BuildChatPrompt(dec.Model, tok, longMsgs, thinking: false);
        async Task<(float[] Prefill, float[] Steps)> LongRun(bool fused)
        {
            dec.FusedAttention = fused;
            await dec.PrefillAsync(longIds, 0f);
            var pre = await dec.ReadLogitsAsync();
            dec.Fork(4);
            var nx = new[] { 11, 22, 33, 44 };
            for (int s = 0; s < 4; s++) await dec.StepAsync(nx.Select(x => x + s).ToArray(), 0f);
            return (pre, await dec.ReadLogitsAsync());
        }
        bool fusedDefault = dec.FusedAttention;
        var fusedRun = await LongRun(true);
        var plainRun = await LongRun(false);
        dec.FusedAttention = fusedDefault;
        static float RelDiff(float[] a, float[] b) { float m = 0f, s = 0f; for (int i = 0; i < a.Length; i++) { m = MathF.Max(m, MathF.Abs(a[i] - b[i])); s = MathF.Max(s, MathF.Abs(b[i])); } return m / s; }
        float attPre = RelDiff(fusedRun.Prefill, plainRun.Prefill), attSteps = RelDiff(fusedRun.Steps, plainRun.Steps);
        bool attOk = attPre < 1e-3f && attSteps < 1e-3f;
        // joined projections (q|k|v and gate|up in one GEMV each, bias + rotary + store in one kernel) against separate ones: a second decoder loaded with
        // JoinProjections off; each output column is computed alike, so the logits must agree after the prefill and four decode steps
        async Task<(float[] Prefill, float[] Steps)> ShortRun(GgufBatchDecoder d)
        {
            await d.PrefillAsync(ids, 0f);
            var pre = await d.ReadLogitsAsync();
            d.Fork(4);
            var nx = new[] { 11, 22, 33, 44 };
            for (int s = 0; s < 4; s++) await d.StepAsync(nx.Select(x => x + s).ToArray(), 0f);
            return (pre, await d.ReadLogitsAsync());
        }
        var joinedRun = await ShortRun(dec);
        (float[] Prefill, float[] Steps) separateRun;
        bool joinSaved = GgufBatchDecoder.JoinProjections;
        GgufBatchDecoder.JoinProjections = false;
        try
        {
            var src2 = await hub.OpenAsync("Qwen/Qwen2.5-0.5B-Instruct-GGUF", "qwen2.5-0.5b-instruct-q8_0.gguf", deselect: false);
            using var sep = await GgufBatchDecoder.CreateAsync(accelerator, src2.Stream, maxSeq: 1024, maxBatch: 4);
            separateRun = await ShortRun(sep);
        }
        finally { GgufBatchDecoder.JoinProjections = joinSaved; }
        float joinPre = RelDiff(joinedRun.Prefill, separateRun.Prefill), joinSteps = RelDiff(joinedRun.Steps, separateRun.Steps);
        bool joinOk = joinPre < 1e-6f && joinSteps < 1e-6f;
        // WebGPU: decode steps replayed from captured plans (the position patched in) against steps encoded kernel by kernel; four sampled sequences, so the batch
        // shrinks and a plan per batch size is captured
        string captureNote = "capture: n/a (not WebGPU)";
        bool captureOk = true;
        if (accelerator.AcceleratorType == AcceleratorType.WebGPU)
        {
            var before = dec.CaptureStats;
            var captured = await dec.GenerateAsync(ids, 4, longNew, 0.8f, 0.95f, 31, stopSet);
            var after = dec.CaptureStats;
            dec.CaptureWebGPU = false;
            var encoded = await dec.GenerateAsync(ids, 4, longNew, 0.8f, 0.95f, 31, stopSet);
            dec.CaptureWebGPU = true;
            captureOk = Enumerable.Range(0, 4).All(s => captured[s].SequenceEqual(encoded[s])) && after.Replays > before.Replays;
            captureNote = $"captured decode {(Enumerable.Range(0, 4).All(s => captured[s].SequenceEqual(encoded[s])) ? "= encoded" : "DIFFERS from encoded")} ({after.Replays - before.Replays} steps replayed, {after.Captures - before.Captures} plans; lengths {string.Join("/", captured.Select(c => c.Length))})";
        }
        int lengths = compacted.Select(c => c.Length).Distinct().Count();
        string got = Text(one[0]);
        bool same = got == want, rowsEqual = four.All(r => r.SequenceEqual(one[0]));
        int distinct = sampled.Select(Text).Distinct().Count();
        var report = $"{accelerator.AcceleratorType}: greedy {(same ? "= the generator" : "DIFFERS from the generator")} ({one[0].Length} tokens); four greedy rows {(rowsEqual ? "identical" : "DIFFER")}; four samples {distinct} distinct; "
            + $"one sequence {one[0].Length / t1:F1} tok/s; warm, device draws one {oneWarm[0].Length / t1w:F1} / four {fourWarm.Sum(r => r.Length) / t4w:F1} tok/s, host draws one {oneHost[0].Length / t1h:F1} / four {fourHost.Sum(r => r.Length) / t4h:F1} tok/s (greedy {(hostSame ? "identical" : "DIFFERS")}); "
            + $"a host-draw step: ids up {dt.Embed / dt.Steps:F2} ms, queued {dt.Device / dt.Steps:F2} ms, wait + candidates read {dt.Readback / dt.Steps:F2} ms; host draws from the candidates {draws - mismatches}/{draws} = the full-row sampler ({wide} read the row); "
            + $"device draws vs the host sampler: candidates path {DrawNote(drawCand)}; full row forced {DrawNote(drawForced)}; T 1.5 top-p 0.99 {DrawNote(drawFlat)}; float ties at most {maxTie:E1} from u (window {TieWindow:E0}) {string.Join(" ", badNotes)}; "
            + $"finished sequences leaving the batch (pipelined): {(compactSame ? "same tokens as riding along" : "DIFFERENT from riding along")} (lengths {string.Join("/", compacted.Select(c => c.Length))}); split-K attention vs three kernels at {longIds.Length} tokens: logits differ by {attPre:E1} (prefill), {attSteps:E1} (4 decode steps); joined projections vs separate: {joinPre:E1} / {joinSteps:E1}; {captureNote}";
        Console.WriteLine("[BatchDecode] " + report);
        if (!same) throw new Exception(report + $"\nwant: {want}\ngot:  {got}");
        if (!rowsEqual || distinct < 2 || mismatches > 0 || wide * 10 > draws || !hostSame || !drawsOk || !compactSame || lengths < 2 || !attOk || !joinOk || !captureOk) throw new Exception(report);      // the row read is a safety net: a device path that keeps falling back is broken
        return report;
    });

    /// <summary>
    /// The decoder's device draw (<see cref="GgufBatchDecoder.DrawRowsAsync"/>: the candidate kernels and the draw kernel, no model) against
    /// <see cref="GgufBatchDecoder.SampleLogits(float[], int, int, float, float, double)"/> with the same uniform, 64 rows per case over a 50,003-token row (not a
    /// multiple of a chunk or of a thread's id range): random logits (the candidates path), the same with the whole-row path forced, a flat row (T 1.5, top-p 0.99:
    /// nuclei of thousands, the whole-row path on its own), 300 exactly tied logits spread over the row with the nucleus cut inside them (the tie scans at the
    /// cut and at the draw, across thread ranges), and ties at the maximum (greedy: the lowest id; T 0.3: drawn in id order). A mismatch within
    /// <see cref="TieWindow"/> of a boundary is a float tie (at most 5% of a case); anything else fails.
    /// </summary>
    [TestMethod(Timeout = 300000)]
    public async Task GgufBatchDecoder_DeviceDraws_MatchHostSampler() => await RunTest(async accelerator =>
    {
        if (!GgufBatchDecoder.SupportsDeviceDraws(accelerator)) throw new UnsupportedTestException("device draws need groups of 256 threads");
        const int V = 50003, R = 64, Pos = 1234;
        var rng = new Random(7);
        float Gauss() => (float)(Math.Sqrt(-2 * Math.Log(1 - rng.NextDouble())) * Math.Cos(2 * Math.PI * rng.NextDouble()));
        float[] RandomRows(float sigma) { var a = new float[R * V]; for (int i = 0; i < a.Length; i++) a[i] = sigma * Gauss(); return a; }
        var notes = new List<string>();
        bool ok = true;
        // returns how many rows' host draws satisfy `counted` (proof the case exercised what it is for)
        async Task<int> Case(string name, float[] lg, float T, float topP, bool force, int seed, Func<int, int, bool>? counted = null)
        {
            var got = await GgufBatchDecoder.DrawRowsAsync(accelerator, lg, R, V, T, topP, seed, Pos, force);
            int eq = 0, ties = 0, bad = 0, wide = 0, hits = 0; double maxD = 0;
            var badNotes = new List<string>();
            for (int r = 0; r < R; r++)
            {
                double u = GgufBatchDecoder.DrawUniform(seed, r, Pos);
                int want = GgufBatchDecoder.SampleLogits(lg, r, V, T, topP, u);
                if (T > 0f && NucleusSize(lg, r, V, T, topP) > GgufBatchDecoder.Candidates) wide++;
                if (counted != null && counted(r, want)) hits++;
                if (got[r] == want) { eq++; continue; }
                if (T > 0f)
                {
                    var di = DrawInterval(lg, r, V, T, Math.Min(1f, topP + (float)TieWindow), got[r]);
                    var dn = DrawInterval(lg, r, V, T, topP, got[r]);
                    double dist = Math.Max(0, Math.Max(dn.Lo - u, u - dn.Hi));
                    if (di.Rank >= 0 && di.Rank < di.Nucleus && dist <= TieWindow) { ties++; maxD = Math.Max(maxD, dist); continue; }
                }
                bad++;
                if (badNotes.Count < 3) badNotes.Add($"[row {r}: u {u:R}, host {want} {RankNote(lg, r, V, T, topP, want, u)}, device {got[r]} {RankNote(lg, r, V, T, topP, got[r], u)}]");
            }
            bool caseOk = bad == 0 && ties * 20 <= R;
            ok &= caseOk;
            notes.Add($"{name}: {eq}/{R} equal, {ties} float ties (max {maxD:E1}), {bad} wrong, {wide} nuclei wider than the candidates" + (counted != null ? $", {hits} counted" : "") + (badNotes.Count > 0 ? " " + string.Join(" ", badNotes) : ""));
            return hits;
        }
        var rnd = RandomRows(4f);
        await Case("random T 0.5 top-p 0.9 (candidates)", rnd, 0.5f, 0.9f, false, 11);
        await Case("same, whole row forced", rnd, 0.5f, 0.9f, true, 12);
        int flatWide = 0;
        {
            var flat = RandomRows(1f);
            await Case("flat T 1.5 top-p 0.99", flat, 1.5f, 0.99f, false, 13);
            for (int r = 0; r < R; r++) if (NucleusSize(flat, r, V, 1.5f, 0.99f) > GgufBatchDecoder.Candidates) flatWide++;
        }
        // ties at the cut: per row a best token at 6, 300 tokens tied at 2 (ids 167 apart: across many threads' ranges), the rest at -100 (past e^20). T 1,
        // top-p 0.5: the nucleus is the best and the first 123 of the ties by id, so the cut falls inside the block and most draws land in it
        var tie = new float[R * V];
        var tieIds = new HashSet<int>[R];
        for (int r = 0; r < R; r++)
        {
            int o = r * V, best = (r * 131 + 5) % V;
            for (int i = 0; i < V; i++) tie[o + i] = -100f;
            tieIds[r] = new HashSet<int>();
            for (int j = 0; tieIds[r].Count < 300; j++) { int id = (r * 7 + j * 167) % V; if (id != best) tieIds[r].Add(id); }
            foreach (int id in tieIds[r]) tie[o + id] = 2f;
            tie[o + best] = 6f;
        }
        int inTies = await Case("300 ties at the cut, T 1 top-p 0.5", tie, 1f, 0.5f, false, 14, (r, t) => tieIds[r].Contains(t));
        int inTiesForced = await Case("same, whole row forced", tie, 1f, 0.5f, true, 15, (r, t) => tieIds[r].Contains(t));
        // ties at the maximum: three tokens tied at 9 (ids apart, the lowest not first in any thread order), the rest random below 3
        var top = RandomRows(1f);
        var topIds = new int[R][];
        for (int r = 0; r < R; r++)
        {
            int o = r * V;
            for (int i = 0; i < V; i++) top[o + i] = Math.Min(top[o + i], 3f);
            topIds[r] = new[] { (r * 977 + 40000) % V, (r * 977 + 3) % V, (r * 977 + 20000) % V };
            foreach (int id in topIds[r]) top[o + id] = 9f;
        }
        await Case("3 ties at the maximum, greedy", top, 0f, 1f, false, 16);
        int topDrawn = await Case("3 ties at the maximum, T 0.3", top, 0.3f, 0.9f, false, 17, (r, t) => topIds[r].Contains(t));
        bool covered = flatWide >= R / 2 && inTies >= R / 4 && inTiesForced >= R / 4 && topDrawn == R;
        var report = $"{accelerator.AcceleratorType}: device draws vs the host sampler, {R} rows of {V}: " + string.Join("; ", notes) + (covered ? "" : $"; COVERAGE short (flat wide {flatWide}, draws in the ties {inTies}/{inTiesForced}, at the maximum {topDrawn})");
        Console.WriteLine("[BatchDecodeDraws] " + report);
        if (!ok || !covered) throw new Exception(report);
        return report;
    });

    // how far u may sit from the device token's interval for a mismatch to count as a float tie (the device's float sums against the host's doubles). Measured:
    // the largest distance seen is reported with the checks
    private const double TieWindow = 1e-5;

    // where a token sits in SampleLogits' order (best first, ties by id): its rank (-1: not within e^20 of the best), its logit, its interval of u (the running
    // mass before and after it as fractions of the nucleus mass: the draw takes the token whose interval holds u), the nucleus size and the candidate count
    private static (int Rank, float Logit, double Lo, double Hi, int Nucleus, int Count) DrawInterval(float[] logits, int row, int vocab, float T, float topP, int token)
    {
        int o = row * vocab;
        float max = float.MinValue;
        for (int i = 0; i < vocab; i++) max = MathF.Max(max, logits[o + i]);
        var c = new List<(int Id, float L, double P)>();
        for (int i = 0; i < vocab; i++) if (logits[o + i] >= max - 20f * T) c.Add((i, logits[o + i], Math.Exp((logits[o + i] - max) / T)));
        c.Sort((a, b) => a.L != b.L ? b.L.CompareTo(a.L) : a.Id.CompareTo(b.Id));
        double total = c.Sum(x => x.P), keep = 0; int m = 0;
        while (m < c.Count && keep < topP * total) keep += c[m++].P;
        double run = 0;
        for (int i = 0; i < c.Count; i++)
        {
            if (c[i].Id == token) return (i, c[i].L, run / keep, (run + c[i].P) / keep, m, c.Count);
            run += c[i].P;
        }
        return (-1, 0f, 0, 0, m, c.Count);
    }
    private static string RankNote(float[] logits, int row, int vocab, float T, float topP, int token, double u)
    {
        var d = DrawInterval(logits, row, vocab, T, topP, token);
        return d.Rank < 0 ? $"(not within e^20 of the best, nucleus {d.Nucleus})"
            : $"(rank {d.Rank}, logit {d.Logit:R}, interval {d.Lo:R}..{d.Hi:R}, nucleus {d.Nucleus} of {d.Count})";
    }

    // the nucleus size GgufBatchDecoder.SampleLogits draws from (tokens within e^20 of the best, best first, the smallest set reaching topP of the mass)
    private static int NucleusSize(float[] logits, int row, int vocab, float T, float topP)
    {
        int o = row * vocab;
        float max = float.MinValue;
        for (int i = 0; i < vocab; i++) max = MathF.Max(max, logits[o + i]);
        var p = new List<double>();
        for (int i = 0; i < vocab; i++) if (logits[o + i] >= max - 20f * T) p.Add(Math.Exp((logits[o + i] - max) / T));
        p.Sort((a, b) => b.CompareTo(a));
        double total = p.Sum(), keep = 0; int m = 0;
        while (m < p.Count && keep < topP * total) keep += p[m++];
        return m;
    }

    /// <summary>
    /// <see cref="GgufBatchDecoder"/> on the other architectures it reads: qwen3 (per-head q / k norms, no biases, head width from its own metadata; Qwen3-0.6B) and
    /// llama (rotary on adjacent pairs, no biases; SmolLM2-360M). Per model the greedy answer must be the existing generator's word for word and three greedy
    /// rows decoded together must be identical.
    /// </summary>
    [TestMethod(Timeout = 900000, Category = "HeavyModel")]
    public async Task GgufBatchDecoder_OtherArchs_GreedyMatchesGenerator() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is AcceleratorType.WebGL or AcceleratorType.Wasm or AcceleratorType.CPU or AcceleratorType.Velocity)
            throw new UnsupportedTestException("a GPU decode test");
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");
        var hub = new ModelsHostStreamLoader(http);
        var messages = new List<(string, string)> { ("user", "Name three primary colors and say why they are called primary.") };
        const int maxNew = 32;
        var reports = new List<string>();
        bool allOk = true;
        foreach (var (repo, file) in new[] { ("Qwen/Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q8_0.gguf"), ("HuggingFaceTB/SmolLM2-360M-Instruct-GGUF", "smollm2-360m-instruct-q8_0.gguf") })
        {
            string want;
            {
                var m = await hub.OpenAsync(repo, file, deselect: false);
                await using (m.Stream)
                using (var pipe = await GgufTextGenerationPipeline.CreateFromStreamAsync(accelerator, m.Stream, maxSeqLen: 512))
                    want = await pipe.GenerateAsync(messages, new GenerationConfig { MaxNewTokens = maxNew, Strategy = "greedy" }, thinking: false);
            }
            var src = await hub.OpenAsync(repo, file, deselect: false);
            using var dec = await GgufBatchDecoder.CreateAsync(accelerator, src.Stream, maxSeq: 512, maxBatch: 3);
            var tok = SentencePieceTokenizer.FromGGUF(dec.Model)!;
            var (ids, stops) = ChatTemplates.BuildChatPrompt(dec.Model, tok, messages, thinking: false);
            var stopSet = stops.ToHashSet(); if (tok.EosId >= 0) stopSet.Add(tok.EosId);
            string Text(int[] t) { var d = tok.CreateStreamingDecoder(); var sb = new System.Text.StringBuilder(); foreach (var x in t) sb.Append(d.Push(x)); sb.Append(d.Finish()); return sb.ToString(); }
            var one = await dec.GenerateAsync(ids, 1, maxNew, 0f, 1f, 1, stopSet);
            var three = await dec.GenerateAsync(ids, 3, maxNew, 0f, 1f, 1, stopSet);
            string got = Text(one[0]);
            bool same = got == want, rows = three.All(r => r.SequenceEqual(one[0]));
            allOk &= same && rows;
            reports.Add($"{dec.Model.Architecture} ({file}, head width {dec.HeadDim}, query {dec.QDim} of {dec.Hidden}): greedy {(same ? "= the generator" : "DIFFERS")} ({one[0].Length} tokens), three rows {(rows ? "identical" : "DIFFER")}"
                + (same ? "" : $" | want: {want} | got: {got}"));
        }
        var report = $"{accelerator.AcceleratorType}: " + string.Join("; ", reports);
        Console.WriteLine("[BatchDecodeArchs] " + report);
        if (!allOk) throw new Exception(report);
        return report;
    });
}
