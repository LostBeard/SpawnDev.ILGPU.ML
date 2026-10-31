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
    /// our prediction at steps 0 and 5 on the reference's own inputs against its float32 prediction, with the bf16 tensor cores (bound 0.02, as the text-to-video DiT gate) and, on CUDA,
    /// without them (fp32 GEMMs: the exact math, bound 2e-3). Desktop GPU or WebGPU.
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task Ltx_ConditionedDiT_MatchesReference() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is AcceleratorType.WebGL or AcceleratorType.Wasm or AcceleratorType.CPU or AcceleratorType.Velocity)
            throw new UnsupportedTestException("the 3.9 GB DiT runs on a desktop GPU or WebGPU");
        var weightsPath = Path.Combine(LtxDir, "ltx_dit_bf16.safetensors"); var fxPath = Path.Combine(LtxDir, "ref", "cond_i2v.npz");
        if (!File.Exists(weightsPath) || !File.Exists(fxPath)) throw new UnsupportedTestException($"LTX fixtures not found in {LtxDir} (set LTX_DIR)");
        using var t5 = ZipFile.OpenRead(Path.Combine(LtxDir, "ref", "t5_0.npz")); using var fx = ZipFile.OpenRead(fxPath);
        var (ctx, _) = WanNpz(t5, "context"); var (mask, _) = WanNpz(t5, "mask"); int rows = (int)mask.Sum();
        var coords = LtxTokens(WanNpz(fx, "coords").Data);
        using var weights = GpuWeights.LoadFile(accelerator, weightsPath);
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
                using var dit = new LtxDiT(accelerator, new DiTKernels(accelerator), weights);
                dit.SetPositions(coords);
                dit.SetContext(0, ctx.AsSpan(0, rows * 4096).ToArray(), rows);
                foreach (int step in new[] { 0, 5 })
                {
                    var (lat, _) = WanNpz(fx, $"latent_in{step}"); var (t, _) = WanNpz(fx, $"t{step}");
                    var segs = LtxRuns(t);
                    using var lIn = accelerator.Allocate1D(lat); using var lOut = accelerator.Allocate1D<float>(lat.Length);
                    dit.Forward(lIn.View, segs, 0, lOut.View, tokenMajor: true);
                    var got = await lOut.View.CopyToHostAsync();
                    var m = WanCompare(got, WanNpz(fx, $"pred{step}").Data);
                    report.Add($"step {step} ({string.Join(" + ", segs.Select(s => $"{s.Count} tokens at t {s.Timestep:F1}"))}) {(mma ? "tensor cores" : "fp32 GEMMs")}: relRMS {m.RelRms:E2}, maxAbs {m.MaxAbs:F3} of {m.RefMaxAbs:F2}");
                    if (mma) worstTc = Math.Max(worstTc, m.RelRms); else worstExact = Math.Max(worstExact, m.RelRms);
                }
            }
            finally { DiTKernels.EnableTensorCores = saved; }
        }
        LtxLog("ConditionedDiT", report);
        if (!(worstTc < 0.02 && worstExact < 2e-3)) throw new Exception("the conditioned DiT deviates from the reference: " + string.Join("; ", report));
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
}
