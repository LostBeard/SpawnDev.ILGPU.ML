using System.IO.Compression;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Ltx;
using SpawnDev.ILGPU.ML.Trellis2;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// LTX-Video image-to-video against diffusers' LTXConditionPipeline (tools/ltx-harness/reference/ref_condition_dump.py: the REAL pipeline, float32 math on the bf16 weights, every DiT
/// call recorded, the random draws re-drawn in the pipeline's order). Fixture <c>ref\cond_i2v.npz</c>: a 480x320 picture as frame 0 of 25 frames (latent 4 x 10 x 15 = 600 tokens, the
/// first 150 hard-conditioned), strength 1, image-conditioning noise 0.15, frame rate 25, the distilled first-pass sigmas, the prompt of <c>t5_0.npz</c>.
/// </summary>
public abstract partial class MLTestBase
{
    private static float[] LtxTokens(float[] coords3xN)
    {
        int n = coords3xN.Length / 3;
        var c = new float[coords3xN.Length];
        for (int k = 0; k < n; k++) for (int a = 0; a < 3; a++) c[3 * k + a] = coords3xN[a * n + k];
        return c;
    }

    private static List<LtxTimestepSegment> LtxRuns(float[] t)
    {
        var s = new List<LtxTimestepSegment>();
        for (int k = 0; k < t.Length; k++)
            if (s.Count > 0 && s[^1].Timestep == t[k]) s[^1] = s[^1] with { Count = s[^1].Count + 1 };
            else s.Add(new LtxTimestepSegment(k, 1, t[k]));
        return s;
    }

    /// <summary>
    /// The DiT with per-token timesteps and explicit token positions (the conditioned forward: 150 tokens at timestep 0, 450 at t; the causal time positions max(8 f - 7, 0) / 25 s):
    /// our prediction at steps 0 and 5 on the reference's own inputs against its float32 prediction, with the bf16 tensor cores (bound 0.05: the reference's own bf16 run is 5.2e-2 off exact math; ours 1.1e-2 at 600 tokens, 2.6e-2 at 216) and, on CUDA,
    /// without them (fp32 GEMMs: the exact math, bound 2e-3). Desktop GPU or WebGPU.
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task Ltx_ConditionedDiT_MatchesReference() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is AcceleratorType.WebGL or AcceleratorType.Wasm or AcceleratorType.CPU or AcceleratorType.Velocity)
            throw new UnsupportedTestException("the 3.9 GB DiT runs on a desktop GPU or WebGPU");
        var weightsPath = Path.Combine(LtxDir, "ltx_dit_bf16.safetensors"); var fxPath = Path.Combine(LtxDir, "ref", "cond_i2v.npz");
        if (!File.Exists(weightsPath) || !File.Exists(fxPath)) throw new UnsupportedTestException($"LTX fixtures not found in {LtxDir} (set LTX_DIR)");
        using var t5 = ZipFile.OpenRead(Path.Combine(LtxDir, "ref", "t5_0.npz"));
        var (ctx, _) = WanNpz(t5, "context"); var (mask, _) = WanNpz(t5, "mask"); int rows = (int)mask.Sum();
        using var weights = GpuWeights.LoadFile(accelerator, weightsPath);
        // 480x320 (600 tokens) and 288x192 (216 tokens, a 6 x 9 latent: the multi-scale recipe's first pass)
        var fixtures = new[] { "i2v", "i2v288" }.Where(f => File.Exists(Path.Combine(LtxDir, "ref", $"cond_{f}.npz"))).ToArray();
        var report = new List<string>();
        double worstTc = 0, worstExact = 0;
        bool exactPass = accelerator.AcceleratorType == AcceleratorType.Cuda;
        foreach (bool tensorCores in exactPass ? new[] { true, false } : new[] { true })
        {
            bool saved = DiTKernels.EnableTensorCores;
            try
            {
                DiTKernels.EnableTensorCores = tensorCores;
                bool mma = tensorCores && accelerator.AcceleratorType == AcceleratorType.Cuda;      // only CUDA has the bf16 tensor cores: elsewhere the math is float32
                foreach (var tag in fixtures)
                foreach (int step in new[] { 0, 5 })
                {
                    using var fx = ZipFile.OpenRead(Path.Combine(LtxDir, "ref", $"cond_{tag}.npz"));
                    using var dit = new LtxDiT(accelerator, new DiTKernels(accelerator), weights);
                    dit.SetPositions(LtxTokens(WanNpz(fx, "coords").Data));
                    dit.SetContext(0, ctx.AsSpan(0, rows * 4096).ToArray(), rows);
                    var (lat, _) = WanNpz(fx, $"latent_in{step}"); var (t, _) = WanNpz(fx, $"t{step}");
                    var segs = LtxRuns(t);
                    using var lIn = accelerator.Allocate1D(lat); using var lOut = accelerator.Allocate1D<float>(lat.Length);
                    dit.Forward(lIn.View, segs, 0, lOut.View, tokenMajor: true);
                    var got = await lOut.View.CopyToHostAsync();
                    var m = WanCompare(got, WanNpz(fx, $"pred{step}").Data);
                    report.Add($"{tag} step {step} ({string.Join(" + ", segs.Select(s => $"{s.Count} tokens at t {s.Timestep:F1}"))}) {(mma ? "tensor cores" : "fp32 GEMMs")}: relRMS {m.RelRms:E2}, maxAbs {m.MaxAbs:F3} of {m.RefMaxAbs:F2}");
                    if (mma) worstTc = Math.Max(worstTc, m.RelRms); else worstExact = Math.Max(worstExact, m.RelRms);
                }
            }
            finally { DiTKernels.EnableTensorCores = saved; }
        }
        LtxLog("ConditionedDiT", report);
        if (!(worstTc < 0.05 && worstExact < 2e-3)) throw new Exception("the conditioned DiT deviates from the reference: " + string.Join("; ", report));
        return string.Join("; ", report);
    });

    /// <summary>
    /// Image-to-video end to end up to the denoised latent: <see cref="LtxPipeline.DenoiseAsync"/> with the picture as the frame-0 condition, the reference's prompt embedding and its random
    /// draws (starting noise, posterior sample, the per-step image-conditioning noise), against the reference's final latent: the VAE encoder, the blend into the first latent frame, the
    /// per-token timesteps, the causal positions, the masked Euler steps and the re-noising of the conditioned frame all have to agree. The condition latent is also checked on its own.
    /// On CUDA with the tensor cores off (the exact math): condition latent bound 1e-4, final latent 1e-3 (measured 5.6e-6 and 1.4e-4; the reference's encoder rounded to fp16, the
    /// hosted precision, moves it only 3e-5: <c>--encoder-f16</c>). With the bf16 tensor cores (each step 1.1e-2 off the exact math, Ltx_ConditionedDiT_MatchesReference; the
    /// reference's own bf16 run is 5e-2 off on ONE step, Ltx_DiT_MatchesReference) the final bound is 0.08. Desktop GPU or WebGPU.
    /// The reference must run with TF32 OFF: cuDNN's default TF32 convolutions put 8e-4 of error into ITS encoding of the picture, which grew to 8e-3 over the steps and looked like a port error.
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task Ltx_ImageToVideo_MatchesReference() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is AcceleratorType.WebGL or AcceleratorType.Wasm or AcceleratorType.CPU or AcceleratorType.Velocity)
            throw new UnsupportedTestException("the 3.9 GB DiT runs on a desktop GPU or WebGPU");
        var fxPath = Path.Combine(LtxDir, "ref", "cond_i2v.npz");
        if (!File.Exists(Path.Combine(LtxDir, "ltx_dit_bf16.safetensors")) || !File.Exists(Path.Combine(LtxDir, "ltx_vae_encoder_f16.safetensors")) || !File.Exists(fxPath))
            throw new UnsupportedTestException($"LTX fixtures not found in {LtxDir} (set LTX_DIR)");
        using var t5 = ZipFile.OpenRead(Path.Combine(LtxDir, "ref", "t5_0.npz")); using var fx = ZipFile.OpenRead(fxPath);
        var (ctx, _) = WanNpz(t5, "context"); var (mask, _) = WanNpz(t5, "mask"); int rows = (int)mask.Sum();
        var (image, _) = WanNpz(fx, "image");                                                      // [3, 1, 320, 480]
        var steps = LtxSampler.DistilledFirstPass.Length;
        var req = new LtxRequest
        {
            Prompt = "", PromptEmbedding = ctx.AsSpan(0, rows * 4096).ToArray(), Width = 480, Height = 320, Frames = 25, FrameRate = LtxFrameRate,
            Conditions = new[] { new LtxCondition { Pixels = image } },
            InitialNoise = WanNpz(fx, "noise").Data, PosteriorNoise = new[] { WanNpz(fx, "posterior").Data },
            StepNoise = Enumerable.Range(0, steps).Select(i => WanNpz(fx, $"cond_noise{i}").Data).ToArray(),
        };
        var (want, ws) = WanNpz(fx, "final");                                                      // [128, 4, 10, 15]
        var report = new List<string>();
        var pipe = new LtxPipeline(accelerator, LtxPaths.InFolder(LtxDir));
        double worstTc = 0, worstExact = 0, condExact = 0;
        foreach (bool tensorCores in accelerator.AcceleratorType == AcceleratorType.Cuda ? new[] { true, false } : new[] { true })
        {
            bool saved = DiTKernels.EnableTensorCores;
            try
            {
                DiTKernels.EnableTensorCores = tensorCores;
                bool mma = tensorCores && accelerator.AcceleratorType == AcceleratorType.Cuda;      // only CUDA has the bf16 tensor cores: elsewhere the math is float32
                string mode = mma ? "tensor cores" : "fp32 GEMMs";
                // the condition latent on its own: our encoder's posterior sample against the reference's clean conditioned tokens (step-0 input minus its conditioning noise)
                {
                    using var ew = GpuWeights.LoadFile(accelerator, Path.Combine(LtxDir, "ltx_vae_encoder_f16.safetensors"));
                    using var enc = new LtxVaeEncoder(accelerator, ew);
                    var cl = await enc.EncodeAsync(image, 1, 320, 480, req.PosteriorNoise![0]);              // [128, 1, 10, 15]
                    int hw0 = cl.Length / 128;
                    var (l0, _) = WanNpz(fx, "latent_in0"); var (n0, _) = WanNpz(fx, "cond_noise0");            // [N, 128]
                    var clean = new float[cl.Length];
                    for (int n = 0; n < hw0; n++) for (int c = 0; c < 128; c++) clean[(long)c * hw0 + n] = l0[(long)n * 128 + c] - 0.15f * n0[(long)n * 128 + c];
                    var mc = WanCompare(cl, clean);
                    report.Add($"{mode}: condition latent (posterior sample) relRMS {mc.RelRms:E2}, maxAbs {mc.MaxAbs:F4}");
                    if (!mma) condExact = mc.RelRms;
                }
                var sw = System.Diagnostics.Stopwatch.StartNew();
                var got = await pipe.DenoiseAsync(req);
                var m = WanCompare(got, want);
                int hw = ws[2] * ws[3], N = ws[1] * hw;
                // the conditioned frame on its own (the picture's latent plus the last step's conditioning noise) and the generated frames
                var f0 = new List<float>(); var f0w = new List<float>(); var rest = new List<float>(); var restw = new List<float>();
                for (int c = 0; c < 128; c++) for (int n = 0; n < N; n++) { long i = (long)c * N + n; if (n < hw) { f0.Add(got[i]); f0w.Add(want[i]); } else { rest.Add(got[i]); restw.Add(want[i]); } }
                var m0 = WanCompare(f0.ToArray(), f0w.ToArray()); var mr = WanCompare(rest.ToArray(), restw.ToArray());
                report.Add($"{mode}: final latent [128, {ws[1]}, {ws[2]}, {ws[3]}] vs the reference relRMS {m.RelRms:E2} (conditioned frame {m0.RelRms:E2}, generated frames {mr.RelRms:E2}), maxAbs {m.MaxAbs:F3} of {m.RefMaxAbs:F2}; {sw.Elapsed.TotalSeconds:F1} s");
                if (mma) worstTc = m.RelRms; else worstExact = m.RelRms;
            }
            finally { DiTKernels.EnableTensorCores = saved; }
        }
        // the hosted 8-bit DiT (the demo's), exact math otherwise: the quantisation's cost on record and bounded (the browser gate holds its 8-bit run to the same bound)
        double q8 = 0;
        if (File.Exists(Path.Combine(LtxDir, "ltx_dit_q8.safetensors")))
        {
            bool saved = DiTKernels.EnableTensorCores;
            try
            {
                DiTKernels.EnableTensorCores = false;
                var pq = new LtxPipeline(accelerator, LtxPaths.InFolder(LtxDir, LtxPaths.DitFile("q8")));
                var mq = WanCompare(await pq.DenoiseAsync(req), want);
                report.Add($"8-bit DiT, fp32 GEMMs: final latent relRMS {mq.RelRms:E2}, maxAbs {mq.MaxAbs:F3}");
                q8 = mq.RelRms;
            }
            finally { DiTKernels.EnableTensorCores = saved; }
        }
        LtxLog("ImageToVideo", report);
        if (!(worstTc < 0.08 && worstExact < 1e-3 && condExact < 1e-4 && q8 < 0.2)) throw new Exception("image-to-video deviates from the reference: " + string.Join("; ", report));
        return string.Join("; ", report);
    });

    /// <summary>
    /// The other conditioning modes against the real LTXConditionPipeline (ref_condition_dump.py, 480x320 x 25, exact math: CUDA with the tensor cores off, or OpenCL / WebGPU):
    /// <c>cond_kf.npz</c> = a picture at frame 0 (strength 1) plus a KEYFRAME picture at frame 16 with strength 0.7 (150 extra tokens placed at frame 16, blended with noise, timestep
    /// 300 throughout); <c>cond_clip.npz</c> = a 17-frame clip at frame 8 (its first two latent frames become 300 extra tokens at frames 8 / 9, its third replaces latent frame 3)
    /// plus a 9-frame clip at frame 0 with strength 0.6 (latent frames 0 and 1 blended, timestep 400). Every latent frame of the final latent must match to 1e-3 (image to video: 1.4e-4).
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task Ltx_KeyframeAndClipConditions_MatchReference() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is AcceleratorType.WebGL or AcceleratorType.Wasm or AcceleratorType.CPU or AcceleratorType.Velocity)
            throw new UnsupportedTestException("the 3.9 GB DiT runs on a desktop GPU or WebGPU");
        if (!File.Exists(Path.Combine(LtxDir, "ltx_dit_bf16.safetensors")) || !File.Exists(Path.Combine(LtxDir, "ltx_vae_encoder_f16.safetensors")) || !File.Exists(Path.Combine(LtxDir, "ref", "cond_kf.npz")))
            throw new UnsupportedTestException($"LTX fixtures not found in {LtxDir} (set LTX_DIR)");
        using var t5 = ZipFile.OpenRead(Path.Combine(LtxDir, "ref", "t5_0.npz"));
        var (ctx, _) = WanNpz(t5, "context"); var (mask, _) = WanNpz(t5, "mask"); int rows = (int)mask.Sum();
        var report = new List<string>();
        double worst = 0;
        bool saved = DiTKernels.EnableTensorCores;
        try
        {
            DiTKernels.EnableTensorCores = false;
            var pipe = new LtxPipeline(accelerator, LtxPaths.InFolder(LtxDir));
            foreach (var (tag, specs) in new (string, (int Frames, int FrameIndex, float Strength)[])[]
            {
                ("kf", new[] { (1, 0, 1f), (1, 16, 0.7f) }),
                ("clip", new[] { (17, 8, 1f), (9, 0, 0.6f) }),
            })
            {
                using var fx = ZipFile.OpenRead(Path.Combine(LtxDir, "ref", $"cond_{tag}.npz"));
                var conditions = specs.Select((s, i) => new LtxCondition { Pixels = WanNpz(fx, $"cond{i}_pixels").Data, Frames = s.Frames, FrameIndex = s.FrameIndex, Strength = s.Strength }).ToArray();
                var req = new LtxRequest
                {
                    Prompt = "", PromptEmbedding = ctx.AsSpan(0, rows * 4096).ToArray(), Width = 480, Height = 320, Frames = 25, FrameRate = LtxFrameRate, Conditions = conditions,
                    InitialNoise = WanNpz(fx, "noise").Data,
                    PosteriorNoise = specs.Select((_, i) => WanNpz(fx, $"posterior{i}").Data).ToArray(),
                    ConditionNoise = specs.Select((s, i) => (s, i)).Where(p => p.s.FrameIndex != 0).Select(p => WanNpz(fx, $"condnoise{p.i}").Data).ToArray(),   // in the order of the extra tokens
                    StepNoise = Enumerable.Range(0, LtxSampler.DistilledFirstPass.Length).Select(s => WanNpz(fx, $"cond_noise{s}").Data).ToArray(),
                };
                var sw = System.Diagnostics.Stopwatch.StartNew();
                var got = await pipe.DenoiseAsync(req);
                var (want, ws) = WanNpz(fx, "final");                                               // [128, Fz, h, w]
                var m = WanCompare(got, want);
                // each latent frame on its own: the conditioned ones barely move, so the whole-latent figure alone would hide the generated frames
                int Fz = ws[1], hw = ws[2] * ws[3];
                var frames = new List<string>();
                for (int f = 0; f < Fz; f++)
                {
                    var gf = new float[128 * hw]; var wf = new float[128 * hw];
                    for (int c = 0; c < 128; c++) { Array.Copy(got, ((long)c * Fz + f) * hw, gf, (long)c * hw, hw); Array.Copy(want, ((long)c * Fz + f) * hw, wf, (long)c * hw, hw); }
                    double r = WanCompare(gf, wf).RelRms;
                    frames.Add($"{r:E1}");
                    worst = Math.Max(worst, r);
                }
                report.Add($"{tag} ({string.Join(" + ", specs.Select(s => $"{s.Frames} frame(s) at {s.FrameIndex} strength {s.Strength}"))}): final latent relRMS {m.RelRms:E2} (per latent frame {string.Join(" / ", frames)}), maxAbs {m.MaxAbs:F4} of {m.RefMaxAbs:F2}; {sw.Elapsed.TotalSeconds:F1} s");
                worst = Math.Max(worst, m.RelRms);
            }
        }
        finally { DiTKernels.EnableTensorCores = saved; }
        LtxLog("Conditions", report);
        if (!(worst < 1e-3)) throw new Exception("LTX conditioning deviates from the reference: " + string.Join("; ", report));
        return string.Join("; ", report);
    });
    /// <summary>
    /// The distilled model's multi-scale recipe against diffusers' real pipelines chained (ref_multiscale_dump.py: LTXConditionPipeline at 288x192 -> LTXLatentUpsamplePipeline with
    /// AdaIN -> LTXConditionPipeline at 576x384 from the upsampled latent re-noised to sigma 0.9094, the second-pass sigmas; a picture at frame 0 in both passes; the reference's draws),
    /// stage by stage with <see cref="LtxPipeline.DenoiseAsync"/> and <see cref="LtxPipeline.UpsampleLatentAsync"/> as <see cref="LtxPipeline.GenerateAsync"/> chains them. Exact math
    /// (tensor cores off; the fixture's upsampler has the export's fp16 weights). Each stage is checked on the REFERENCE's input as well as chained, because the model is
    /// sensitive at these small sizes (Ltx_ImageToVideo_Steps_MatchReference: the reference perturbed by its own fp16 encoder moves 9.7e-4): the upscaler on the reference's first-pass latent (bound 1e-4), the second pass on the
    /// reference's upsampled latent (5e-3), the first pass (5e-3) and the whole chain (1e-2). Also checks <see cref="LtxPipeline.FirstPassSize"/> against the reference's sizes.
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task Ltx_MultiScale_MatchesReference() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is AcceleratorType.WebGL or AcceleratorType.Wasm or AcceleratorType.CPU or AcceleratorType.Velocity)
            throw new UnsupportedTestException("the 3.9 GB DiT runs on a desktop GPU or WebGPU");
        var fxPath = Path.Combine(LtxDir, "ref", "ms_ms.npz");
        if (!File.Exists(Path.Combine(LtxDir, "ltx_dit_bf16.safetensors")) || !File.Exists(Path.Combine(LtxDir, "ltx_upsampler_f16.safetensors")) || !File.Exists(fxPath))
            throw new UnsupportedTestException($"LTX fixtures not found in {LtxDir} (set LTX_DIR)");
        using var t5 = ZipFile.OpenRead(Path.Combine(LtxDir, "ref", "t5_0.npz")); using var fx = ZipFile.OpenRead(fxPath);
        var (ctx, _) = WanNpz(t5, "context"); var (mask, _) = WanNpz(t5, "mask"); int rows = (int)mask.Sum();
        var sizes = WanNpz(fx, "sizes").Data;
        int w1 = (int)sizes[0], h1 = (int)sizes[1], w2 = (int)sizes[2], h2 = (int)sizes[3];
        var report = new List<string>();
        var fp = LtxPipeline.FirstPassSize(480, 320);
        report.Add($"first-pass size for 480x320: {fp.Width}x{fp.Height} (reference {w1}x{h1})");
        bool ok = fp.Width == w1 && fp.Height == h1;
        var emb = ctx.AsSpan(0, rows * 4096).ToArray();
        LtxRequest Req(string p, int w, int h, float[]? sigmas, float[]? latents, int steps) => new()
        {
            Prompt = "", PromptEmbedding = emb, Width = w, Height = h, Frames = 25, FrameRate = LtxFrameRate, Sigmas = sigmas, Latents = latents,
            Conditions = new[] { new LtxCondition { Pixels = WanNpz(fx, p + "_pixels").Data } },
            InitialNoise = WanNpz(fx, p + "_noise").Data, PosteriorNoise = new[] { WanNpz(fx, p + "_posterior").Data },
            StepNoise = Enumerable.Range(0, steps).Select(s => WanNpz(fx, $"{p}_cond_noise{s}").Data).ToArray(),
        };
        double worst = 0;
        bool saved = DiTKernels.EnableTensorCores;
        try
        {
            DiTKernels.EnableTensorCores = false;
            var pipe = new LtxPipeline(accelerator, LtxPaths.InFolder(LtxDir));
            var sw = System.Diagnostics.Stopwatch.StartNew();
            var p1 = await pipe.DenoiseAsync(Req("p1", w1, h1, null, null, LtxSampler.DistilledFirstPass.Length));
            var m1 = WanCompare(p1, WanNpz(fx, "p1_final").Data);
            var refUp = WanNpz(fx, "up").Data;
            var muRef = WanCompare(await pipe.UpsampleLatentAsync(WanNpz(fx, "p1_final").Data, 4, h1 / 32, w1 / 32, 1f), refUp);
            var m2Ref = WanCompare(await pipe.DenoiseAsync(Req("p2", w2, h2, LtxSampler.DistilledSecondPass, refUp, LtxSampler.DistilledSecondPass.Length)), WanNpz(fx, "p2_final").Data);
            var up = await pipe.UpsampleLatentAsync(p1, 4, h1 / 32, w1 / 32, 1f);
            var mu = WanCompare(up, refUp);
            var p2 = await pipe.DenoiseAsync(Req("p2", w2, h2, LtxSampler.DistilledSecondPass, up, LtxSampler.DistilledSecondPass.Length));
            var m2 = WanCompare(p2, WanNpz(fx, "p2_final").Data);
            report.Add($"on the reference's inputs: upscaler {muRef.RelRms:E2}, second pass {w2}x{h2} {m2Ref.RelRms:E2}");
            report.Add($"chained: first pass {w1}x{h1} {m1.RelRms:E2}, upsampled {mu.RelRms:E2}, second pass {m2.RelRms:E2} (maxAbs {m2.MaxAbs:F4} of {m2.RefMaxAbs:F2}); {sw.Elapsed.TotalSeconds:F1} s");
            ok &= muRef.RelRms < 1e-4 && m2Ref.RelRms < 5e-3 && m1.RelRms < 5e-3;
            worst = m2.RelRms;
        }
        finally { DiTKernels.EnableTensorCores = saved; }
        LtxLog("MultiScale", report);
        if (!(ok && worst < 1e-2)) throw new Exception("the multi-scale recipe deviates from the reference: " + string.Join("; ", report));
        return string.Join("; ", report);
    });
    /// <summary>
    /// The conditioned sampler step by step on the 288x192 image-to-video fixture (cond_i2v288.npz: 216 tokens, the multi-scale recipe's first-pass size), exact math: before every
    /// step the DiT input (after the image-conditioning noise) and the prediction against the reference's, a standalone DiT forward on the pipeline's own input (must equal the
    /// pipeline's prediction), and the final latent. At this size the trajectory amplifies tiny differences: the REFERENCE itself, with only its encoder rounded to fp16
    /// (cond_i2v288_encf16.npz: the condition latent moved 2.5e-6), is 1.0e-3 off its own step-5 prediction and 9.7e-4 off its final latent - the same as ours (1.3e-3 / 7.2e-4),
    /// while random 1e-5 input noise moves our step-5 prediction only 4e-5 (the amplified direction is along the trajectory). So the inputs must track the reference to 1e-3 and the
    /// final latent is bounded by 3e-3. Desktop GPU or WebGPU.
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task Ltx_ImageToVideo_Steps_MatchReference() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is AcceleratorType.WebGL or AcceleratorType.Wasm or AcceleratorType.CPU or AcceleratorType.Velocity)
            throw new UnsupportedTestException("the 3.9 GB DiT runs on a desktop GPU or WebGPU");
        var fxPath = Path.Combine(LtxDir, "ref", "cond_i2v288.npz");
        if (!File.Exists(Path.Combine(LtxDir, "ltx_dit_bf16.safetensors")) || !File.Exists(fxPath)) throw new UnsupportedTestException($"LTX fixtures not found in {LtxDir} (set LTX_DIR)");
        using var t5 = ZipFile.OpenRead(Path.Combine(LtxDir, "ref", "t5_0.npz")); using var fx = ZipFile.OpenRead(fxPath);
        var (ctx, _) = WanNpz(t5, "context"); var (mask, _) = WanNpz(t5, "mask"); int rows = (int)mask.Sum();
        int steps = LtxSampler.DistilledFirstPass.Length;
        var req = new LtxRequest
        {
            Prompt = "", PromptEmbedding = ctx.AsSpan(0, rows * 4096).ToArray(), Width = 288, Height = 192, Frames = 25, FrameRate = LtxFrameRate,
            Conditions = new[] { new LtxCondition { Pixels = WanNpz(fx, "image").Data } },
            InitialNoise = WanNpz(fx, "noise").Data, PosteriorNoise = new[] { WanNpz(fx, "posterior").Data },
            StepNoise = Enumerable.Range(0, steps).Select(i => WanNpz(fx, $"cond_noise{i}").Data).ToArray(),
        };
        var report = new List<string>();
        double worst = 0, worstInput = 0;
        bool saved = DiTKernels.EnableTensorCores;
        try
        {
            DiTKernels.EnableTensorCores = false;
            // the DiT's own sensitivity at step 5: the reference's input, and the same plus 1e-5 relative Gaussian noise
            using var weights = GpuWeights.LoadFile(accelerator, Path.Combine(LtxDir, "ltx_dit_bf16.safetensors"));
            using var dit = new LtxDiT(accelerator, new DiTKernels(accelerator), weights);
            {
                dit.SetPositions(LtxTokens(WanNpz(fx, "coords").Data));
                dit.SetContext(0, ctx.AsSpan(0, rows * 4096).ToArray(), rows);
                var (lat5, _) = WanNpz(fx, "latent_in5"); var segs = LtxRuns(WanNpz(fx, "t5").Data); var want5 = WanNpz(fx, "pred5").Data;
                double rms = Math.Sqrt(lat5.Select(v => (double)v * v).Average());
                var eps = LtxPipeline.Gaussian(lat5.Length, 5);
                var pert = lat5.Select((v, i) => v + (float)(1e-5 * rms) * eps[i]).ToArray();
                async Task<float[]> Pred(float[] x) { using var a = accelerator.Allocate1D(x); using var o = accelerator.Allocate1D<float>(x.Length); dit.Forward(a.View, segs, 0, o.View, tokenMajor: true); return await o.View.CopyToHostAsync(); }
                var p0 = await Pred(lat5); var p1 = await Pred(pert);
                report.Add($"step-5 sensitivity: prediction on the reference input {WanCompare(p0, want5).RelRms:E2}; with 1e-5 input noise {WanCompare(p1, want5).RelRms:E2} (vs our own clean prediction {WanCompare(p1, p0).RelRms:E2})");
            }
            var pipe = new LtxPipeline(accelerator, LtxPaths.InFolder(LtxDir));
            pipe.StepTap = (s, lat, v) =>
            {
                accelerator.Synchronize();
                var a = new float[lat.Length]; lat.CopyToCPU(a); var b = new float[v.Length]; v.CopyToCPU(b);
                var mi = WanCompare(a, WanNpz(fx, $"latent_in{s}").Data); var mp = WanCompare(b, WanNpz(fx, $"pred{s}").Data);
                // a fresh forward of the standalone DiT on the pipeline's own input and the reference's timesteps: the pipeline's forward must equal it
                using var xa = accelerator.Allocate1D(a); using var xo = accelerator.Allocate1D<float>(a.Length);
                dit.Forward(xa.View, LtxRuns(WanNpz(fx, $"t{s}").Data), 0, xo.View, tokenMajor: true);
                accelerator.Synchronize();
                var c = new float[a.Length]; xo.View.CopyToCPU(c);
                report.Add($"step {s}: input {mi.RelRms:E2}, prediction {mp.RelRms:E2}; standalone DiT on this input vs the pipeline's prediction {WanCompare(b, c).RelRms:E2}, vs the reference {WanCompare(c, WanNpz(fx, $"pred{s}").Data).RelRms:E2}");
                worstInput = Math.Max(worstInput, mi.RelRms);
            };
            var m = WanCompare(await pipe.DenoiseAsync(req), WanNpz(fx, "final").Data);
            pipe.StepTap = null;
            report.Add($"final latent relRMS {m.RelRms:E2}, maxAbs {m.MaxAbs:F4}");
            worst = m.RelRms;
        }
        finally { DiTKernels.EnableTensorCores = saved; }
        LtxLog("ImageToVideoSteps", report);
        if (!(worst < 3e-3 && worstInput < 1e-3)) throw new Exception("the conditioned sampler deviates from the reference: " + string.Join("; ", report));
        return string.Join("; ", report);
    });
}
