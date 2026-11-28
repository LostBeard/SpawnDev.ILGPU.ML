using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Anima;
using SpawnDev.ILGPU.ML.Trellis2;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    // Anima fixtures: ANIMA_WEIGHTS (default D:/Projects/anima/anima-preview3-base.safetensors, sha256 14fffe8a...);
    // ANIMA_DUMPS (default the _DevComms anima-run1-dumps-2026-11-27 folder: SpawnGen's ComfyUI dumps, fp32 math on the bf16 weights).
    // Spec and float64 reference: tools/anima-ref (ANIMA-SPEC.md; the reference is within 2.4e-6 of these dumps end to end).
    private static string Anima_Weights() => Environment.GetEnvironmentVariable("ANIMA_WEIGHTS") ?? "D:/Projects/anima/anima-preview3-base.safetensors";
    private static string Anima_Dumps() => Environment.GetEnvironmentVariable("ANIMA_DUMPS") ?? "D:/users/tj/Projects/_DevComms/anima-run1-dumps-2026-11-27";
    private static float[] Anima_Dump(string prefix)
    {
        var hits = Directory.GetFiles(Anima_Dumps(), prefix + "_f32_*.raw");
        if (hits.Length != 1) throw new FileNotFoundException($"{hits.Length} dumps named {prefix}_f32_* in {Anima_Dumps()}");
        var shape = Path.GetFileNameWithoutExtension(hits[0])[(prefix.Length + 5)..].Split('x').Select(long.Parse).Aggregate(1L, (a, b) => a * b);
        return F2_Raw(hits[0], shape);
    }
    private static void Anima_Require(Accelerator accelerator)
    {
        if (accelerator.AcceleratorType is not (AcceleratorType.Cuda or AcceleratorType.OpenCL)) throw new UnsupportedTestException("reads a 4 GB weight file from disk: a desktop GPU");
        if (!File.Exists(Anima_Weights())) throw new UnsupportedTestException($"{Anima_Weights()} not found");
        if (!Directory.Exists(Anima_Dumps())) throw new UnsupportedTestException($"SpawnGen's Anima dumps not found in {Anima_Dumps()}");
    }

    /// <summary>
    /// <see cref="AnimaLlmAdapter"/> against ComfyUI: the positive and the negative prompt (Qwen3-0.6B's final states as ComfyUI encoded them, the T5 ids)
    /// against the rows of call_0's c_crossattn (row 0 = negative, 27 rows; row 1 = positive, 26 rows; the rest of the 512 rows are zero padding).
    /// </summary>
    [TestMethod(Timeout = 600000, Category = "HeavyModel")]
    public async Task Anima_LlmAdapter_MatchesComfyUi() => await RunTest(async accelerator =>
    {
        Anima_Require(accelerator);
        using var w = GpuWeights.LoadFile(accelerator, Anima_Weights(), n => n.StartsWith(AnimaLlmAdapter.Prefix, StringComparison.Ordinal));
        using var adapter = new AnimaLlmAdapter(accelerator, new DiTKernels(accelerator), w) { PreciseActivations = true };
        var ca = Anima_Dump("call_0_c_c_crossattn");
        var report = new List<string>(); double worst = 0;
        foreach (var (label, row, cond, idName) in new[] { ("positive", 1, "pos_0_cond", "pos_0_t5xxl_ids"), ("negative", 0, "neg_0_cond", "neg_0_t5xxl_ids") })
        {
            var q = Anima_Dump(cond);
            int N = q.Length / AnimaLlmAdapter.Dim;
            var ids = Anima_Dump(idName).Select(v => (int)MathF.Round(v)).ToArray();
            using var qb = accelerator.Allocate1D(q);
            var sw = System.Diagnostics.Stopwatch.StartNew();
            var outView = await adapter.ForwardAsync(qb.View, N, ids);
            var got = await outView.CopyToHostAsync();
            double ms = sw.Elapsed.TotalMilliseconds;
            int M = ids.Length;
            var gold = ca.AsSpan(row * AnimaDiT.ContextRows * AnimaLlmAdapter.Dim, M * AnimaLlmAdapter.Dim).ToArray();
            double e = F2_RelRms(got, gold);
            worst = Math.Max(worst, e);
            // the rows after the prompt must be the zero padding (ComfyUI pads the context, it does not trim it)
            Ur_Expect(ca.AsSpan((row * AnimaDiT.ContextRows + M) * AnimaLlmAdapter.Dim, (AnimaDiT.ContextRows - M) * AnimaLlmAdapter.Dim).IndexOfAnyExcept(0f) < 0, $"{label}: c_crossattn row {row} is not zero after {M} rows");
            report.Add($"{label} ({N} Qwen3 states, {M} T5 ids) relRMS {e:E2} in {ms:F0} ms");
        }
        string line = string.Join("; ", report);
        Console.WriteLine("[Anima] adapter vs ComfyUI: " + line);
        Ur_Expect(worst < 2e-5, line);   // PRE-REGISTERED 2026-11-28: the float64 reference is 1.2e-6 / 1.3e-6 off these rows; fp32 device math may add a few e-6
        return line;
    });

    /// <summary>
    /// <see cref="AnimaDiT"/>'s stages against SpawnGen's per-module ComfyUI taps (ANIMA_TAPS, default _DevComms anima-run3-taps-2026-11-27; that run's call_0,
    /// both CFG rows): the timestep sinusoid, the adaLN-LoRA vector, the RMSNorm'd embedding, the patch embedding, blocks 0 / 13 / 27, the final layer. Localizes a
    /// DiT difference to one stage (the end-to-end test only says that the output differs).
    /// </summary>
    [TestMethod(Timeout = 1200000, Category = "HeavyModel")]
    public async Task Anima_DiT_StagesMatchComfyUiTaps() => await RunTest(async accelerator =>
    {
        Anima_Require(accelerator);
        string taps = Environment.GetEnvironmentVariable("ANIMA_TAPS") ?? "D:/users/tj/Projects/_DevComms/anima-run3-taps-2026-11-27";
        if (!Directory.Exists(taps)) throw new UnsupportedTestException($"SpawnGen's Anima taps not found in {taps}");
        float[] Tap(string name)
        {
            var hits = Directory.GetFiles(taps, name + "_f32_*.raw");
            if (hits.Length != 1) throw new FileNotFoundException($"{hits.Length} taps named {name}_f32_* in {taps}");
            var n = Path.GetFileNameWithoutExtension(hits[0])[(name.Length + 5)..].Split('x').Select(long.Parse).Aggregate(1L, (a, b) => a * b);
            return F2_Raw(hits[0], n);
        }
        using var w = GpuWeights.LoadFile(accelerator, Anima_Weights(), n => !n.StartsWith(AnimaLlmAdapter.Prefix, StringComparison.Ordinal));
        using var dit = new AnimaDiT(accelerator, new DiTKernels(accelerator), w) { PreciseActivations = true };
        const int C = AnimaKernels.LatentChannels, T = 1, H = 64, W = 64, per = C * T * H * W, D = 2048, N = 1024;
        dit.SetGrid(T, H, W);
        var stages = new (string Mine, string Tap, int Count)[]
        {
            ("t_proj", "tap_diffusion_model_t_embedder_0_0_out0", D), ("adaln", "tap_diffusion_model_t_embedder_1_0_out1", 3 * D),
            ("emb", "tap_diffusion_model_t_embedding_norm_0_out0", D), ("x_embedder", "tap_diffusion_model_x_embedder_0_out0", N * D),
            ("block_0", "tap_diffusion_model_blocks_0_0_out0", N * D), ("block_13", "tap_diffusion_model_blocks_13_0_out0", N * D),
            ("block_27", "tap_diffusion_model_blocks_27_0_out0", N * D), ("final_layer", "tap_diffusion_model_final_layer_0_out0", N * 64),
        };
        var probes = stages.ToDictionary(s => s.Mine, s => accelerator.Allocate1D<float>(s.Count));
        // block 0's intermediates (no taps of their own): NaN / Inf counts and the largest value, to say which op fails first
        var extra = new List<(string, MemoryBuffer1D<float, Stride1D.Dense>)>();
        try
        {
            dit.Tap = (name, view, count) =>
            {
                if (probes.TryGetValue(name, out var b)) { b.View.CopyFrom(view.SubView(0, b.Length)); return; }
                if (!name.StartsWith("b0_", StringComparison.Ordinal)) return;
                var e = extra.FirstOrDefault(t => t.Item1 == name).Item2;
                if (e == null) { e = accelerator.Allocate1D<float>(count); extra.Add((name, e)); }
                e.View.CopyFrom(view.SubView(0, count));
            };
            var x = F2_Raw(Directory.GetFiles(taps, "call_0_input_f32_*.raw").Single(), 2L * per);
            var sig = F2_Raw(Directory.GetFiles(taps, "call_0_timestep_f32_*.raw").Single(), 2);
            var ctx = F2_Raw(Directory.GetFiles(taps, "call_0_c_c_crossattn_f32_*.raw").Single(), 2L * AnimaDiT.ContextRows * AnimaDiT.ContextDim);
            using var lat = accelerator.Allocate1D<float>(per);
            using var v = accelerator.Allocate1D<float>(per);
            var report = new List<string>(); double worst = 0;
            for (int row = 0; row < 2; row++)
            {
                dit.SetContext(row, ctx.AsSpan(row * AnimaDiT.ContextRows * AnimaDiT.ContextDim, AnimaDiT.ContextRows * AnimaDiT.ContextDim).ToArray(), AnimaDiT.ContextRows);
                lat.View.CopyFromCPU(x.AsSpan(row * per, per).ToArray());
                dit.Forward(lat.View, sig[row], row, v.View);
                await accelerator.SynchronizeAsync();
                var parts = new List<string>();
                foreach (var (mine, tapName, count) in stages)
                {
                    var got = await probes[mine].View.CopyToHostAsync();
                    var gold = Tap(tapName).AsSpan(row * count, count).ToArray();
                    double e = F2_RelRms(got, gold);
                    int nan = got.Count(float.IsNaN);
                    worst = Math.Max(worst, double.IsNaN(e) ? double.PositiveInfinity : e);
                    parts.Add($"{mine} {e:E2}" + (nan > 0 ? $" ({nan} NaN)" : ""));
                }
                foreach (var (name, buf) in extra)
                {
                    var got = await buf.View.CopyToHostAsync();
                    int nan = got.Count(float.IsNaN), inf = got.Count(float.IsInfinity);
                    double mx = got.Where(float.IsFinite).Select(f => Math.Abs((double)f)).DefaultIfEmpty(0).Max();
                    parts.Add($"{name} nan {nan} inf {inf} max {mx:G3}");
                }
                report.Add($"row {row}: " + string.Join(", ", parts));
            }
            string line = string.Join(" | ", report);
            Console.WriteLine("[Anima] DiT stages vs ComfyUI taps: " + line);
            Ur_Expect(worst < 1e-4, line);   // PRE-REGISTERED 2026-11-28 with the end-to-end gate: the float64 reference is 6.4e-6 worst over these stages
            return line;
        }
        finally { dit.Tap = null; await accelerator.SynchronizeAsync(); foreach (var b in probes.Values) b.Dispose(); foreach (var (_, b) in extra) b.Dispose(); }
    });

    /// <summary>
    /// <see cref="AnimaDiT"/> against ComfyUI's first two sampler calls (512 x 512, CFG batch [negative, positive], sigma 0.9999667 then 0.9): every row's
    /// velocity v, compared as ComfyUI records it (the denoised x - sigma * v), from that call's own input latent and c_crossattn.
    /// </summary>
    [TestMethod(Timeout = 1200000, Category = "HeavyModel")]
    public async Task Anima_DiT_MatchesComfyUi() => await RunTest(async accelerator =>
    {
        Anima_Require(accelerator);
        var sw = System.Diagnostics.Stopwatch.StartNew();
        using var w = GpuWeights.LoadFile(accelerator, Anima_Weights(), n => !n.StartsWith(AnimaLlmAdapter.Prefix, StringComparison.Ordinal));
        double load = sw.Elapsed.TotalSeconds;
        using var dit = new AnimaDiT(accelerator, new DiTKernels(accelerator), w) { PreciseActivations = true };
        const int C = AnimaKernels.LatentChannels, T = 1, H = 64, W = 64, per = C * T * H * W;
        dit.SetGrid(T, H, W);
        Ur_Expect(dit.Layers == 28 && dit.TokenCount == 1024, $"{dit.Layers} blocks, {dit.TokenCount} tokens");
        using var lat = accelerator.Allocate1D<float>(per);
        using var v = accelerator.Allocate1D<float>(per);
        var report = new List<string>(); double worst = 0; var times = new List<double>();
        foreach (int call in new[] { 0, 1 })
        {
            var x = Anima_Dump($"call_{call}_input");
            var sig = Anima_Dump($"call_{call}_timestep");
            var ctx = Anima_Dump($"call_{call}_c_c_crossattn");
            var gold = Anima_Dump($"call_{call}_output");
            for (int row = 0; row < 2; row++)
            {
                // the context rows ComfyUI fed (the adapter output, zero-padded to 512): pass the real rows only, SetContext pads
                int rows = AnimaDiT.ContextRows;
                while (rows > 0 && ctx.AsSpan((row * AnimaDiT.ContextRows + rows - 1) * AnimaDiT.ContextDim, AnimaDiT.ContextDim).IndexOfAnyExcept(0f) < 0) rows--;
                dit.SetContext(row, ctx.AsSpan(row * AnimaDiT.ContextRows * AnimaDiT.ContextDim, rows * AnimaDiT.ContextDim).ToArray(), rows);
                lat.View.CopyFromCPU(x.AsSpan(row * per, per).ToArray());
                sw.Restart();
                dit.Forward(lat.View, sig[row], row, v.View);
                var vv = await v.View.CopyToHostAsync();
                times.Add(sw.Elapsed.TotalSeconds);
                var den = new float[per];
                for (int i = 0; i < per; i++) den[i] = x[row * per + i] - sig[row] * vv[i];
                double e = F2_RelRms(den, gold.AsSpan(row * per, per).ToArray());
                worst = Math.Max(worst, e);
                report.Add($"call {call} row {row} (sigma {sig[row]:G7}, {rows} context rows) {e:E2}");
            }
        }
        string line = $"{string.Join("; ", report)} | weights {load:F1} s, forward {string.Join(" / ", times.Select(t => t.ToString("F2")))} s";
        Console.WriteLine("[Anima] DiT vs ComfyUI (denoised x - sigma v): " + line);
        Ur_Expect(worst < 1e-4, line);   // PRE-REGISTERED 2026-11-28: the float64 reference is 2.7e-6 worst over these four rows; fp32 device math through a residual of 2.5e5
        return line;
    });
}
