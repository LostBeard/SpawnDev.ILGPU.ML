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
    /// <see cref="AnimaLLLite"/> (anima-lllite-inpainting-v2 on anima-base-v1.0) against ComfyUI's AnimaLLLiteApply (SpawnGen's black-box run4: the Painter
    /// rectangle mask on compositor-a.png, strength 1, cfg 1, 2 steps; run5 = the same graph without the LLLite). Both sampler calls' denoised outputs with the
    /// LLLite, run5's first call without it (the same DiT on the same base), and the red check: run4's call without the LLLite must miss by far.
    /// ANIMA_BASE / ANIMA_LLLITE / ANIMA_LLLITE_RUN override the paths.
    /// </summary>
    [TestMethod(Timeout = 1200000, Category = "HeavyModel")]
    public async Task Anima_LLLite_MatchesComfyUi() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is not (AcceleratorType.Cuda or AcceleratorType.OpenCL)) throw new UnsupportedTestException("reads a 4 GB weight file from disk: a desktop GPU");
        string basePath = Environment.GetEnvironmentVariable("ANIMA_BASE") ?? "D:/Projects/anima/anima-base-v1.0.safetensors";
        string llPath = Environment.GetEnvironmentVariable("ANIMA_LLLITE") ?? "D:/Projects/anima/anima-lllite-inpainting-v2.safetensors";
        string run = Environment.GetEnvironmentVariable("ANIMA_LLLITE_RUN") ?? "D:/users/tj/Projects/_DevComms/anima-run4-lllite-2026-11-28";
        foreach (var f in new[] { basePath, llPath, Path.Combine(run, "compositor-a.png") })
            if (!File.Exists(f)) throw new UnsupportedTestException($"{f} not found");
        float[] Raw(string dir, string prefix)
        {
            var hit = Directory.GetFiles(Path.Combine(run, dir), prefix + "_f32_*.raw").Single();
            var n = Path.GetFileNameWithoutExtension(hit)[(prefix.Length + 5)..].Split('x').Select(long.Parse).Aggregate(1L, (a, b) => a * b);
            return F2_Raw(hit, n);
        }
        // the picture (ComfyUI IMAGE, [0, 1]) and the Painter's mask (its alpha: 1 = repaint), channel-first
        var rgba = Preprocessing.PngDecoder.DecodePixels(File.ReadAllBytes(Path.Combine(run, "compositor-a.png"))) ?? throw new InvalidDataException("compositor-a.png");
        var mrgba = Preprocessing.PngDecoder.DecodePixels(File.ReadAllBytes(Path.Combine(run, "painter-lllite-rect.png"))) ?? throw new InvalidDataException("painter-lllite-rect.png");
        const int PH = 512, PW = 512, HW = PH * PW;
        var rgb = new float[3 * HW]; var mask = new float[HW];
        for (int p = 0; p < HW; p++) { for (int c = 0; c < 3; c++) rgb[c * HW + p] = rgba[p * 4 + c] / 255f; mask[p] = mrgba[p * 4 + 3] / 255f; }

        using var bw = GpuWeights.LoadFile(accelerator, basePath, n => !n.StartsWith(AnimaLlmAdapter.Prefix, StringComparison.Ordinal));
        using var lw = GpuWeights.LoadFile(accelerator, llPath);
        var kernels = new DiTKernels(accelerator) { PreciseActivations = true };
        using var dit = new AnimaDiT(accelerator, kernels, bw);
        using var ll = new AnimaLLLite(accelerator, kernels, lw);
        Ur_Expect(ll.PictureChannels == 4 && ll.Modules == 112 && ll.ResBlocks == 4 && ll.CondEmbDim == 64 && ll.MlpDim == 64,
            $"inpainting-v2 should read as 4 channels, 112 modules, 4 res blocks, 64 / 64: {ll.PictureChannels}, {ll.Modules}, {ll.ResBlocks}, {ll.CondEmbDim} / {ll.MlpDim}");
        var ak = new AnimaKernels(accelerator);
        using var rgbB = accelerator.Allocate1D(rgb); using var maskB = accelerator.Allocate1D(mask); using var pic = accelerator.Allocate1D<float>(4 * HW);
        ak.InpaintPicture(rgbB.View, maskB.View, pic.View, PH, PW);
        var sw = System.Diagnostics.Stopwatch.StartNew();
        await ll.SetPictureAsync(pic.View, PH, PW);
        double condMs = sw.Elapsed.TotalMilliseconds;

        const int C = AnimaKernels.LatentChannels, per = C * 64 * 64;
        dit.SetGrid(1, 64, 64);
        using var lat = accelerator.Allocate1D<float>(per); using var v = accelerator.Allocate1D<float>(per);
        async Task<double> Call(string dir, int call, bool withLLLite)
        {
            var x = Raw(dir, $"call_{call}_input"); var sig = Raw(dir, $"call_{call}_timestep"); var ctx = Raw(dir, $"call_{call}_c_c_crossattn"); var gold = Raw(dir, $"call_{call}_output");
            int rows = AnimaDiT.ContextRows;
            while (rows > 0 && ctx.AsSpan((rows - 1) * AnimaDiT.ContextDim, AnimaDiT.ContextDim).IndexOfAnyExcept(0f) < 0) rows--;
            dit.SetContext(0, ctx.AsSpan(0, rows * AnimaDiT.ContextDim).ToArray(), rows);
            lat.View.CopyFromCPU(x);
            dit.LLLite = withLLLite ? ll : null;
            dit.Forward(lat.View, sig[0], 0, v.View);
            var vv = await v.View.CopyToHostAsync();
            var den = new float[per];
            for (int i = 0; i < per; i++) den[i] = x[i] - sig[0] * vv[i];
            return F2_RelRms(den, gold);
        }
        var times = new List<double>();
        async Task<double> Timed(string dir, int call, bool withLLLite) { sw.Restart(); double e = await Call(dir, call, withLLLite); times.Add(sw.Elapsed.TotalSeconds); return e; }
        double l0 = await Timed("run4-lllite", 0, true);         // the first call compiles the kernels
        double l1 = await Timed("run4-lllite", 1, true);
        double c0 = await Timed("run5-control", 0, false);
        double off = await Timed("run4-lllite", 0, false);       // red check: the same call without the LLLite
        string line = $"with the LLLite: call 0 {l0:E2}, call 1 {l1:E2}; control (no LLLite) call 0 {c0:E2}; run4 call 0 WITHOUT the LLLite {off:E2} (must be large)"
            + $" | picture -> rows {condMs:F0} ms; calls {string.Join(" / ", times.Select(t => t.ToString("F2")))} s (with, with, without, without; incl. the host reads)";
        Console.WriteLine("[Anima] LLLite vs ComfyUI (denoised): " + line);
        Ur_Expect(Math.Max(Math.Max(l0, l1), c0) < 1e-4 && off > 0.1, line);   // PRE-REGISTERED 2026-11-28 (as the DiT's): the float64 reference is 1.3e-6 / 1.0e-6 / 6.6e-6; without the LLLite 1.2
        return line;
    });

    /// <summary>
    /// <see cref="AnimaLora"/> (anima-turbo-lora-v0.2 on anima-base-v1.0) against ComfyUI's LoraLoaderModelOnly (SpawnGen's run 10: strength 1.0 and 0.5 and a
    /// no-LoRA control, cfg 1, seed 42): the denoised output of the calls with the LoRA at each strength (the same LoRA object, its Strength changed between calls),
    /// the control without it, and two red checks (the 1.0 call at 0.5; the 1.0 call without the LoRA). ANIMA_BASE / ANIMA_LORA / ANIMA_LORA_RUN override the paths.
    /// </summary>
    [TestMethod(Timeout = 1200000, Category = "HeavyModel")]
    public async Task Anima_Lora_MatchesComfyUi() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is not (AcceleratorType.Cuda or AcceleratorType.OpenCL)) throw new UnsupportedTestException("reads a 4 GB weight file from disk: a desktop GPU");
        string basePath = Environment.GetEnvironmentVariable("ANIMA_BASE") ?? "D:/Projects/anima/anima-base-v1.0.safetensors";
        string loraPath = Environment.GetEnvironmentVariable("ANIMA_LORA") ?? "D:/Projects/anima/anima-turbo-lora-v0.2.safetensors";
        string run = Environment.GetEnvironmentVariable("ANIMA_LORA_RUN") ?? "D:/users/tj/Projects/_DevComms/anima-run10-turbo-lora-2026-11-28";
        foreach (var f in new[] { basePath, loraPath }) if (!File.Exists(f)) throw new UnsupportedTestException($"{f} not found");
        if (!Directory.Exists(run)) throw new UnsupportedTestException($"SpawnGen's run 10 not found in {run}");
        float[] Raw(string prefix)
        {
            var hit = Directory.GetFiles(run, prefix + "_f32_*.raw").Single();
            var n = Path.GetFileNameWithoutExtension(hit)[(prefix.Length + 5)..].Split('x').Select(long.Parse).Aggregate(1L, (a, b) => a * b);
            return F2_Raw(hit, n);
        }
        using var bw = GpuWeights.LoadFile(accelerator, basePath, n => !n.StartsWith(AnimaLlmAdapter.Prefix, StringComparison.Ordinal));
        using var lora = new AnimaLora(accelerator, GpuWeights.LoadFile(accelerator, loraPath));
        Ur_Expect(lora.Modules == 448 && lora.Rank == 32, $"the turbo LoRA should patch 448 DiT Linears at rank 32: {lora.Modules}, rank {lora.Rank}");
        using var dit = new AnimaDiT(accelerator, new DiTKernels(accelerator) { PreciseActivations = true }, bw);
        const int per = AnimaKernels.LatentChannels * 64 * 64;
        dit.SetGrid(1, 64, 64);
        using var lat = accelerator.Allocate1D<float>(per); using var v = accelerator.Allocate1D<float>(per);
        var sw = System.Diagnostics.Stopwatch.StartNew(); var times = new List<double>();
        async Task<double> Call(string tag, int call, float? strength)
        {
            var x = Raw($"{tag}_call_{call}_input"); var sig = Raw($"{tag}_call_{call}_timestep"); var ctx = Raw($"{tag}_call_{call}_c_c_crossattn"); var gold = Raw($"{tag}_call_{call}_output");
            int rows = AnimaDiT.ContextRows;
            while (rows > 0 && ctx.AsSpan((rows - 1) * AnimaDiT.ContextDim, AnimaDiT.ContextDim).IndexOfAnyExcept(0f) < 0) rows--;
            dit.SetContext(0, ctx.AsSpan(0, rows * AnimaDiT.ContextDim).ToArray(), rows);
            lat.View.CopyFromCPU(x);
            dit.RemoveLora(lora);
            if (strength is float s) { lora.Strength = s; dit.AddLora(lora); }
            sw.Restart();
            dit.Forward(lat.View, sig[0], 0, v.View);
            var vv = await v.View.CopyToHostAsync();
            times.Add(sw.Elapsed.TotalSeconds);
            var den = new float[per];
            for (int i = 0; i < per; i++) den[i] = x[i] - sig[0] * vv[i];
            return F2_RelRms(den, gold);
        }
        double l100 = await Call("lora100", 0, 1f), l100b = await Call("lora100", 1, 1f), l050 = await Call("lora050", 0, 0.5f), ctrl = await Call("nolora", 0, null);
        double wrongStrength = await Call("lora100", 0, 0.5f), without = await Call("lora100", 0, null);
        string line = $"strength 1: call 0 {l100:E2}, call 1 {l100b:E2}; strength 0.5: call 0 {l050:E2}; no LoRA (control) {ctrl:E2}; red checks: the 1.0 call at 0.5 {wrongStrength:E2}, without the LoRA {without:E2}"
            + $" | calls {string.Join(" / ", times.Select(t => t.ToString("F2")))} s (with, with, with, without, with, without)";
        Console.WriteLine("[Anima] LoRA vs ComfyUI (denoised): " + line);
        Ur_Expect(new[] { l100, l100b, l050, ctrl }.Max() < 1e-4 && wrongStrength > 0.1 && without > 0.1, line);   // PRE-REGISTERED 2026-11-28 (as the DiT's): float64 8.2e-6 / 8.6e-6 / 2.2e-6 / 2.4e-6; 0.60 / 0.73
        return line;
    });

    /// <summary>
    /// Run 10's whole 8-step euler sampling (cfg 1) on the GPU with and without the turbo LoRA, against ComfyUI's final latent (out_samples is
    /// latent_format.process_out(x) = x * std_c + mean_c, the Wan 2.1 constants; compared here in the model's space). The final latents are written to
    /// _scratch/anima-ref/gpu_replay_{tag}_final_f32.raw for tools/anima-ref/compare_replay_final.py's float64 replay. MEASURED first in float64
    /// (replay_lora_steps.py): exact arithmetic lands 3.8e-4 from ComfyUI with the LoRA and 5.8e-6 without: the LoRA's trajectory amplifies rounding.
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task Anima_Lora_EightStepsMatchComfyUi() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is not (AcceleratorType.Cuda or AcceleratorType.OpenCL)) throw new UnsupportedTestException("reads a 4 GB weight file from disk: a desktop GPU");
        string basePath = Environment.GetEnvironmentVariable("ANIMA_BASE") ?? "D:/Projects/anima/anima-base-v1.0.safetensors";
        string loraPath = Environment.GetEnvironmentVariable("ANIMA_LORA") ?? "D:/Projects/anima/anima-turbo-lora-v0.2.safetensors";
        string run = Environment.GetEnvironmentVariable("ANIMA_LORA_RUN") ?? "D:/users/tj/Projects/_DevComms/anima-run10-turbo-lora-2026-11-28";
        foreach (var f in new[] { basePath, loraPath }) if (!File.Exists(f)) throw new UnsupportedTestException($"{f} not found");
        if (!Directory.Exists(run)) throw new UnsupportedTestException($"SpawnGen's run 10 not found in {run}");
        float[] Raw(string prefix)
        {
            var hit = Directory.GetFiles(run, prefix + "_f32_*.raw").Single();
            var n = Path.GetFileNameWithoutExtension(hit)[(prefix.Length + 5)..].Split('x').Select(long.Parse).Aggregate(1L, (a, b) => a * b);
            return F2_Raw(hit, n);
        }
        float[] mean = { -0.7571f, -0.7089f, -0.9113f, 0.1075f, -0.1745f, 0.9653f, -0.1517f, 1.5508f, 0.4134f, -0.0715f, 0.5517f, -0.3632f, -0.1922f, -0.9497f, 0.2503f, -0.2921f };
        float[] std = { 2.8184f, 1.4541f, 2.3275f, 2.6558f, 1.2196f, 1.7708f, 2.6052f, 2.0743f, 3.2687f, 2.1526f, 2.8652f, 1.5579f, 1.6382f, 1.1253f, 2.8251f, 1.9160f };
        using var bw = GpuWeights.LoadFile(accelerator, basePath, n => !n.StartsWith(AnimaLlmAdapter.Prefix, StringComparison.Ordinal));
        using var lora = new AnimaLora(accelerator, GpuWeights.LoadFile(accelerator, loraPath));
        using var dit = new AnimaDiT(accelerator, new DiTKernels(accelerator) { PreciseActivations = true }, bw);
        const int HW = 64 * 64, per = AnimaKernels.LatentChannels * HW;
        dit.SetGrid(1, 64, 64);
        using var lat = accelerator.Allocate1D<float>(per); using var v = accelerator.Allocate1D<float>(per);
        string outDir = Environment.GetEnvironmentVariable("ANIMA_REPLAY_OUT") ?? "D:/users/tj/Projects/SpawnDev.ILGPU.ML/SpawnDev.ILGPU.ML/_scratch/anima-ref";
        var report = new List<string>(); double worstLora = 0, worstPlain = 0;
        foreach (var (tag, withLora) in new[] { ("lora100", true), ("nolora", false) })
        {
            var s = Raw($"{tag}_call_0_c_transformer_options_sample_sigmas");
            var x = Raw($"{tag}_call_0_input"); var ctx = Raw($"{tag}_call_0_c_c_crossattn");
            int rows = AnimaDiT.ContextRows;
            while (rows > 0 && ctx.AsSpan((rows - 1) * AnimaDiT.ContextDim, AnimaDiT.ContextDim).IndexOfAnyExcept(0f) < 0) rows--;
            dit.SetContext(0, ctx.AsSpan(0, rows * AnimaDiT.ContextDim).ToArray(), rows);
            dit.RemoveLora(lora);
            if (withLora) { lora.Strength = 1f; dit.AddLora(lora); }
            for (int i = 0; i + 1 < s.Length; i++)
            {
                lat.View.CopyFromCPU(x);
                dit.Forward(lat.View, s[i], 0, v.View);
                var vv = await v.View.CopyToHostAsync();
                // euler on the flow model: D = x - s v; x' = x + (s' - s)(x - D) / s = x + (s' - s) v; the last step (s' = 0) leaves D
                for (int j = 0; j < per; j++) x[j] = s[i + 1] == 0f ? x[j] - s[i] * vv[j] : x[j] + (s[i + 1] - s[i]) * vv[j];
            }
            var gold = Raw($"{tag}_out_samples");
            var goldModel = new float[per];
            for (int j = 0; j < per; j++) { int c = j / HW; goldModel[j] = (gold[j] - mean[c]) / std[c]; }
            double e = F2_RelRms(x, goldModel);
            if (withLora) worstLora = e; else worstPlain = e;
            report.Add($"{tag} {e:E2}");
            Directory.CreateDirectory(outDir);
            var bytes = new byte[per * 4]; Buffer.BlockCopy(x, 0, bytes, 0, bytes.Length);
            File.WriteAllBytes(Path.Combine(outDir, $"gpu_replay_{tag}_final_f32.raw"), bytes);
        }
        string line = $"{accelerator.AcceleratorType}: 8-step final latent vs ComfyUI (model space): {string.Join(", ", report)} (float64: lora100 3.8e-4, nolora 5.8e-6)";
        Console.WriteLine("[Anima] LoRA 8 steps: " + line);
        Ur_Expect(worstLora < 1e-2 && worstPlain < 1e-3, line);   // PRE-REGISTERED 2026-11-28: Lore's bound 1e-2 for the LoRA trajectory (float64 itself is 3.8e-4 off); 1e-3 without
        return line;
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
