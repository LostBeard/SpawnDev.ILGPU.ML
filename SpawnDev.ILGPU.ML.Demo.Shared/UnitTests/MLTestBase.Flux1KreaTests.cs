using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// FLUX.1 Krea's fp8_scaled transformer against ComfyUI (SpawnGen's dumps, F1_DIR\comfy\comfyui-dumps\split): its F16 biases are held exactly as bf16 hi + lo, which needs a float ->
/// BFloat16 -> float round trip that rounds in registers. SpawnDev.ILGPU before 5.4.33 does not (identity on CUDA / OpenCL / WebGPU / WebGL / Wasm: Bf16HiLo_SplitsFp16Exactly),
/// so the remainder is zero and one layer misses ComfyUI by 1.5e-4 (the model by 1.3e-3). With 5.4.33: 8e-6 per layer, 2.5e-5 / 4.8e-5 for the model (CUDA).
/// </summary>
public abstract partial class MLTestBase
{
    /// <summary>
    /// The Krea transformer from its fp8_scaled file (no prefix, scale_weight per matrix) against ComfyUI's two model calls (guidance 3.5: ComfyUI's value with no FluxGuidance node):
    /// denoised x - sigma v as for the checkpoint. The scales must be read (<see cref="Flux1.Flux1Weights.HasScales"/>); without them every matrix decodes at the wrong magnitude.
    /// </summary>
    [TestMethod(Timeout = 3600000, Category = "HeavyModel")]
    public async Task Flux1Split_Fp8ScaledTransformer_MatchesComfyUiCalls() => await RunTest(async accelerator =>
    {
        const string Rel = "split_files/diffusion_models/flux1-krea-dev_fp8_scaled.safetensors";
        var (split, d) = F1s_Require(accelerator, Rel);
        double residentGb = double.TryParse(Environment.GetEnvironmentVariable("F1_RESIDENT_GB"), System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out var g) ? g : 7.0;
        var sw = System.Diagnostics.Stopwatch.StartNew();
        using var w = await Flux1.Flux1Weights.OpenAsync(accelerator, new LocalFolderModelSource(Path.Combine(split, "split_files", "diffusion_models")), "", Path.GetFileName(Rel), (long)(residentGb * 1e9));
        Ur_Expect(w.HasScales, "the fp8_scaled file's scale_weight tensors must be read");
        const int T = 256, rows = 16, cols = 16, lat = 16 * 32 * 32;
        using var tr = new Flux1.Flux1Transformer(accelerator, w, T, rows * cols);
        tr.SetPositions(rows, cols);
        var k = new Flux1.Flux1Kernels(accelerator);
        var report = new List<string> { $"weights: {w.ResidentBlocks} of 57 blocks resident, {w.ResidentBytes / 1e9:F2} GB on the device, {w.StreamedBytesPerForward / 1e9:F2} GB streamed per call; opened in {sw.Elapsed.TotalSeconds:F1} s" };
        using var x = accelerator.Allocate1D<float>(lat); using var tokens = accelerator.Allocate1D<float>(rows * cols * 64L); using var den = accelerator.Allocate1D<float>(lat);
        using var txt = accelerator.Allocate1D<float>(256L * 4096); using var pooled = accelerator.Allocate1D<float>(768);
        double worst = 0;
        for (int c = 0; c < 2; c++)
        {
            float sigma = F2_Raw(Path.Combine(d, $"f1s_call_{c}_timestep_f32_1.raw"), 1)[0], guidance = F2_Raw(Path.Combine(d, $"f1s_call_{c}_c_guidance_f32_1.raw"), 1)[0];
            txt.View.CopyFromCPU(F2_Raw(Path.Combine(d, $"f1s_call_{c}_c_c_crossattn_f32_1x256x4096.raw"), 256L * 4096));
            pooled.View.CopyFromCPU(F2_Raw(Path.Combine(d, $"f1s_call_{c}_c_y_f32_1x768.raw"), 768));
            x.View.CopyFromCPU(F2_Raw(Path.Combine(d, $"f1s_call_{c}_input_f32_1x16x32x32.raw"), lat));
            k.Patchify(x.View, tokens.View, rows, cols);
            sw.Restart();
            await tr.ForwardAsync(tokens.View, txt.View, pooled.View, sigma, guidance);
            k.Denoised(x.View, tr.Output, den.View, sigma, rows, cols);
            double rel = F2_RelRms(await den.View.CopyToHostAsync(), F2_Raw(Path.Combine(d, $"f1s_call_{c}_output_f32_1x16x32x32.raw"), lat));
            worst = Math.Max(worst, rel);
            report.Add($"call {c} (sigma {sigma:G6}, guidance {guidance:G3}): denoised vs ComfyUI {rel:E2}, {sw.Elapsed.TotalSeconds:F1} s");
        }
        foreach (var line in report) Console.WriteLine("[Flux1Split] transformer: " + line);
        // MEASURED (RTX 4070, SpawnDev.ILGPU 5.4.33): CUDA 2.50e-5 / 4.81e-5, OpenCL 2.27e-6 / 2.77e-5 (5.4.32: 1.3e-3 / 2.6e-3, the fp16 bias remainders zero)
        Ur_Expect(worst < 1.5e-4, string.Join("; ", report));
        return string.Join("; ", report);
    });

    /// <summary>
    /// One scaled 8-bit Linear of the Krea file (double_blocks.0.img_attn.qkv) against ComfyUI's own input / output of that layer (SpawnGen hooked it: ComfyUI computes x (f8 * scale)^T
    /// + f16 bias in fp32, reproduced to 4.9e-8). Our building blocks: the exact unscaled decode with the input times the scale, the fp16 bias as bf16 hi + lo, precise activations.
    /// The folded decode (bf16(f8 * scale)) and bf16 activations are reported beside it.
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task Flux1Split_ScaledLinear_MatchesComfyUiLayer() => await RunTest(async accelerator =>
    {
        const string Rel = "split_files/diffusion_models/flux1-krea-dev_fp8_scaled.safetensors";
        var (split, d) = F1s_Require(accelerator, Rel);
        string one = Path.Combine(d, "one-linear");
        if (!Directory.Exists(one)) throw new UnsupportedTestException($"needs SpawnGen's one-layer dump in {one}");
        const int M = 256, K = 3072, N = 9216;
        const string Name = "double_blocks.0.img_attn.qkv";
        var xin = F2_Raw(Path.Combine(one, "krea_diffusion_model_double_blocks_0_img_attn_qkv_in_f32_1x256x3072.raw"), (long)M * K);
        var want = F2_Raw(Path.Combine(one, "krea_diffusion_model_double_blocks_0_img_attn_qkv_out_f32_1x256x9216.raw"), (long)M * N);
        using var w = await Flux1.Flux1Weights.OpenAsync(accelerator, new LocalFolderModelSource(Path.Combine(split, "split_files", "diffusion_models")), "", Path.GetFileName(Rel), 0);   // every block streamed
        var (gw, streamed) = await w.BlockAsync(0);
        try
        {
            float s = w.ActivationScale(gw, Name + ".weight");
            bool hasLo = gw.Has(Name + ".bias.lo");
            var k1 = new Flux1.Flux1Kernels(accelerator);
            using var o = accelerator.Allocate1D<float>((long)M * N);
            async Task<double> Run(bool precise, bool exact)
            {
                var k = new Trellis2.DiTKernels(accelerator) { PreciseActivations = precise };
                var xs = new float[xin.Length];
                for (int i = 0; i < xs.Length; i++) xs[i] = exact ? xin[i] * s : xin[i];
                using var xb = accelerator.Allocate1D(xs);
                var W = exact ? w.MatrixExact(gw, Name + ".weight") : w.Matrix(gw, Name + ".weight");
                k.Linear(xb.View, W, gw.BF16(Name + ".bias"), o.View, M, K, N);
                if (hasLo) k1.AddRowVector(o.View, gw.BF16(Name + ".bias.lo"), M, N, 0);
                return F2_RelRms(await o.View.CopyToHostAsync(), want);
            }
            double exactPrecise = await Run(true, true), exactBf16 = await Run(false, true), foldedPrecise = await Run(true, false);
            double tiled;
            bool tc = Trellis2.DiTKernels.EnableTensorCores;
            try { Trellis2.DiTKernels.EnableTensorCores = false; tiled = await Run(false, true); }       // the fp32 tiled GEMM (every non-CUDA backend's path)
            finally { Trellis2.DiTKernels.EnableTensorCores = tc; }
            // host reference for the first 8 rows: the file's f8 bytes decoded in C#, x (f8 * scale)^T + f16 bias in double
            var words = await gw.F8Words(Name + ".weight").CopyToHostAsync();
            var biasHi = await gw.BF16(Name + ".bias").CopyToHostAsync(); var biasLo = hasLo ? await gw.BF16(Name + ".bias.lo").CopyToHostAsync() : null;
            static double F8(uint b) { uint e = (b >> 3) & 15u, m = b & 7u; double mag = e == 0 ? m * Math.Pow(2, -9) : (1 + m / 8.0) * Math.Pow(2, (int)e - 7); return (b & 0x80u) != 0 ? -mag : mag; }
            const int R = 8;
            var host = new float[R * N]; var wantR = new float[R * N];
            for (int n = 0; n < N; n++)
            {
                double bias = (float)biasHi[n] + (biasLo != null ? (float)biasLo[n] : 0f);
                for (int r = 0; r < R; r++)
                {
                    double acc = bias;
                    for (int kk = 0; kk < K; kk++) { long idx = (long)n * K + kk; acc += xin[r * K + kk] * F8((words[idx >> 2] >> (int)((idx & 3) * 8)) & 0xFFu) * s; }
                    host[r * N + n] = (float)acc; wantR[r * N + n] = want[r * N + n];
                }
            }
            double hostVsComfy = F2_RelRms(host, wantR);
            var report = $"{Name}: scale {s:G6}, bias hi + lo {hasLo}; host reference (C# f8 decode, double) vs ComfyUI {hostVsComfy:E2} (8 rows); device: exact decode + precise activations {exactPrecise:E2}, "
                + $"exact + bf16 activations {exactBf16:E2}, folded decode + precise {foldedPrecise:E2}, exact + fp32 tiled GEMM (no tensor cores) {tiled:E2}";
            Console.WriteLine("[Flux1Split] " + report);
            // MEASURED (RTX 4070, SpawnDev.ILGPU 5.4.33): host reference 1.49e-7; CUDA exact + precise 8.25e-6 (split bf16 activations), OpenCL 8.55e-7 (fp32); bf16 activations 7.2e-4,
            // folded decode 8.8e-4. 5.4.32: 1.5e-4 (the fp16 bias remainder zero on every GPU backend)
            Ur_Expect(hostVsComfy < 1e-6 && exactPrecise < 2e-5 && tiled < 2e-6 && foldedPrecise > 1e-4, report);
            return report;
        }
        finally { if (streamed) { await accelerator.SynchronizeAsync(); gw.Dispose(); } }
    });
}
