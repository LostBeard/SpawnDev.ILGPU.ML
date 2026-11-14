using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    /// <summary>
    /// Min-p and the presence penalty in <see cref="GgufBatchDecoder"/>'s device draws (<see cref="GgufBatchDecoder.DrawRowsAsync"/>) against the host sampler with the same uniform
    /// (<see cref="TextGenerationSampler.ApplyRepetitionPenalty"/> + <see cref="TextGenerationSampler.ApplyPresencePenalty"/> on each row's window, then
    /// <see cref="GgufBatchDecoder.SampleLogits(float[], int, int, float, float, double, int, float)"/>), 64 rows of 50,003 tokens per case: min-p on the candidates path, the
    /// same with the whole-row path forced, a flat row whose nucleus is wider than the candidates (the whole-row path's own min-p cut), and penalties with min-p. Every case
    /// must also change some draw against the same case without min-p / presence (it is exercised). ComfyUI's TextGenerate sampling mode has both (SpawnGen's ask).
    /// </summary>
    [TestMethod(Timeout = 300000)]
    public async Task GgufBatchDecoder_MinPAndPresence_MatchHostSampler() => await RunTest(async accelerator =>
    {
        if (!GgufBatchDecoder.SupportsDeviceDraws(accelerator)) throw new UnsupportedTestException("device draws need groups of 256 threads");
        const int V = 50003, R = 64, Pos = 777;
        var rng = new Random(23);
        float Gauss() => (float)(Math.Sqrt(-2 * Math.Log(1 - rng.NextDouble())) * Math.Cos(2 * Math.PI * rng.NextDouble()));
        float[] Rows(float sigma) { var a = new float[R * V]; for (int i = 0; i < a.Length; i++) a[i] = sigma * Gauss(); return a; }
        var notes = new List<string>();
        bool ok = true;
        async Task Case(string name, float[] lg, float T, float topP, float minP, bool force, int seed, int[][]? windows = null, float penalty = 1f, float presence = 0f)
        {
            var got = await GgufBatchDecoder.DrawRowsAsync(accelerator, lg, R, V, T, topP, seed, Pos, force, windows, penalty, 0, 0, 0, minP, presence);
            var plain = await GgufBatchDecoder.DrawRowsAsync(accelerator, lg, R, V, T, topP, seed, Pos, force, windows, 1f, 0, 0, 0, 0f, 0f);
            var hl = (float[])lg.Clone();
            if (windows != null)
            {
                var row = new float[V];
                for (int r = 0; r < R; r++)
                {
                    Array.Copy(hl, (long)r * V, row, 0, V);
                    TextGenerationSampler.ApplyRepetitionPenalty(row, windows[r], penalty);
                    TextGenerationSampler.ApplyPresencePenalty(row, windows[r], presence);
                    Array.Copy(row, 0, hl, (long)r * V, V);
                }
            }
            int eq = 0, changed = 0; var bad = new List<string>();
            for (int r = 0; r < R; r++)
            {
                double u = GgufBatchDecoder.DrawUniform(seed, r, Pos);
                int want = GgufBatchDecoder.SampleLogits(hl, r, V, T, topP, u, 0, minP);
                if (got[r] == want) eq++; else if (bad.Count < 3) bad.Add($"[row {r}: host {want}, device {got[r]}]");
                if (got[r] != plain[r]) changed++;
            }
            bool caseOk = eq == R && changed > 0;
            ok &= caseOk;
            notes.Add($"{name}: {eq}/{R} equal, {changed} draws changed by min-p / presence" + (bad.Count > 0 ? " " + string.Join(" ", bad) : ""));
        }
        var rnd = Rows(4f);
        await Case("T 1, min-p 0.05 (candidates)", rnd, 1f, 1f, 0.05f, false, 31);
        await Case("same, whole row forced", rnd, 1f, 1f, 0.05f, true, 32);
        await Case("T 1, top-p 0.9 then min-p 0.3", rnd, 1f, 0.9f, 0.3f, false, 33);
        var flat = Rows(1f);
        await Case("flat T 1.5, top-p 0.99 then min-p 0.5 (nucleus past the candidates)", flat, 1.5f, 0.99f, 0.5f, false, 34);
        // penalties: each row's window holds the row's 8 best tokens (so the penalties move the draw), repeated
        var windows = new int[R][];
        for (int r = 0; r < R; r++)
        {
            var best = Enumerable.Range(0, V).OrderByDescending(i => rnd[(long)r * V + i]).Take(8).ToArray();
            windows[r] = best.Concat(best).ToArray();
        }
        await Case("repetition 1.1 + presence 1.5, min-p 0.05", rnd, 1f, 1f, 0.05f, false, 35, windows, 1.1f, 1.5f);
        await Case("presence 2.0 alone, whole row forced", rnd, 1f, 0.95f, 0f, true, 36, windows, 1f, 2f);
        var report = $"{accelerator.AcceleratorType}: " + string.Join("; ", notes);
        Console.WriteLine("[BatchMinP] " + report);
        if (!ok) throw new Exception(report);
        return report;
    });
}
