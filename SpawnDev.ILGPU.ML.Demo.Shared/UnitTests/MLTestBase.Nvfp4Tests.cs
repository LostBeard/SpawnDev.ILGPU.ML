using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Kernels;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    /// <summary>
    /// ComfyUI's mixed fp4 checkpoints (gemma_3_12B_it_fp4_mixed) against ComfyUI's own dequantisation (SpawnGen's dump of Gemma 3 layer 24, _DevComms 2026-11-17-1700;
    /// LTX2_DIR, default F:/spawndev-ltx2/comfyui-dumps/gemma_l24): q_proj is NVFP4, k_proj fp8. For each: the weight ComfyUI computes with (read out through an identity
    /// matrix) must equal ours EXACTLY (host decode, and the GPU decode <see cref="Nvfp4.DecodeExact"/> times weight_scale_2), and one real call (x [1024, 3840] from the
    /// encode) through our GEMM on the exact bf16 matrix with the scale moved to the input must match ComfyUI's output (float32 activations, as ComfyUI multiplies).
    /// </summary>
    [TestMethod(Timeout = 600000, Category = "HeavyModel")]
    public async Task Nvfp4_GemmaLayer_MatchesComfyUi() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is not (AcceleratorType.Cuda or AcceleratorType.OpenCL)) throw new UnsupportedTestException("reads the dump from disk: a desktop GPU");
        var d = Environment.GetEnvironmentVariable("LTX2_DIR") ?? "F:/spawndev-ltx2/comfyui-dumps/gemma_l24";
        string F(string s) => Path.Combine(d, "ltx2_gemma_lin_gemma3_12b_transformer_model_layers_24_self_attn_" + s);
        string C(string s) => Path.Combine(d, "ltx2_gemma_l24_gemma3_12b_transformer_model_layers_24_self_attn_" + s);
        if (!File.Exists(F("q_proj_weight_raw_uint8.bin"))) throw new UnsupportedTestException($"SpawnGen's Gemma layer-24 dump not found in {d}");
        const int K = 3840, Rows = 1024;
        var report = new List<string>(); bool exact = true; double worstCall = 0;
        var k = new Trellis2.DiTKernels(accelerator) { PreciseActivations = true };
        var f2 = new Flux2.Flux2Kernels(accelerator);
        static MemoryBuffer1D<uint, Stride1D.Dense> Words(Accelerator a, byte[] b) { var w = new uint[(b.Length + 3) / 4]; Buffer.BlockCopy(b, 0, w, 0, b.Length); return a.Allocate1D(w); }

        async Task Check(string name, int N, float[] wHost, MemoryBuffer1D<BFloat16, Stride1D.Dense> wExact, float inputScale)
        {
            var weffT = F2_Raw(F($"{name}_weff_t_in_float32_f32_{K}x{N}.raw"), (long)K * N);               // W_eff^T [in, out]
            double hostMax = 0, gpuMax = 0;
            var gpu = await wExact.View.CopyToHostAsync();
            for (int n = 0; n < N; n++)
                for (int i = 0; i < K; i++)
                {
                    float c = weffT[(long)i * N + n];
                    hostMax = Math.Max(hostMax, Math.Abs(wHost[(long)n * K + i] - c));
                    gpuMax = Math.Max(gpuMax, Math.Abs((float)gpu[(long)n * K + i] * inputScale - c));
                }
            exact &= hostMax == 0 && gpuMax == 0;
            // one real call: y = (s x) W^T on the exact bf16 matrix
            var x = F2_Raw(C($"{name}_0_in0_f32_1x{Rows}x{K}.raw"), (long)Rows * K);
            var expected = F2_Raw(C($"{name}_0_out0_f32_1x{Rows}x{N}.raw"), (long)Rows * N);
            using var dx = accelerator.Allocate1D(x); using var xs = accelerator.Allocate1D<float>(x.Length); using var y = accelerator.Allocate1D<float>((long)Rows * N);
            f2.ScaleInto(dx.View, xs.View, inputScale, x.Length);
            k.Linear(xs.View, wExact.View, null, y.View, Rows, K, N);
            double e = F2_RelRms(await y.View.CopyToHostAsync(), expected); worstCall = Math.Max(worstCall, e);
            report.Add($"{name}: weight host {(hostMax == 0 ? "exact" : $"max {hostMax:E2}")}, GPU {(gpuMax == 0 ? "exact" : $"max {gpuMax:E2}")}; call [1024, {N}] {e:E2}");
        }

        // q_proj: NVFP4 [4096, 3840]
        {
            const int N = 4096;
            var packed = File.ReadAllBytes(F("q_proj_weight_raw_uint8.bin"));
            var scales = File.ReadAllBytes(F("q_proj_weight_scale_raw_float8_e4m3fn.bin"));
            float scale2 = BitConverter.ToSingle(File.ReadAllBytes(F("q_proj_weight_scale_2_raw_float32.bin")), 0);
            Ur_Expect(packed.Length == N * K / 2 && scales.Length == Nvfp4.ScaleBytes(N, K), $"sizes {packed.Length} / {scales.Length}");
            using var wp = Words(accelerator, packed); using var ws = Words(accelerator, scales);
            using var w = accelerator.Allocate1D<BFloat16>((long)N * K);
            Nvfp4.DecodeExact(accelerator, wp.View, ws.View, w.View, N, K);
            await Check("q_proj", N, Nvfp4.DecodeHost(packed, scales, scale2, N, K), w, scale2);
        }
        // k_proj: fp8 e4m3 [2048, 3840], one scale
        {
            const int N = 2048;
            var raw = File.ReadAllBytes(F("k_proj_weight_raw_float8_e4m3fn.bin"));
            float scale = BitConverter.ToSingle(File.ReadAllBytes(F("k_proj_weight_scale_raw_float32.bin")), 0);
            var host = new float[raw.Length];
            for (int i = 0; i < raw.Length; i++)
            {
                int b = raw[i], e = (b >> 3) & 15, m = b & 7;
                float v = e == 0 ? m * 0.001953125f : (1f + m * 0.125f) * MathF.Pow(2, e - 7);
                host[i] = ((b & 0x80) != 0 ? -v : v) * scale;
            }
            using var wr = Words(accelerator, raw);
            using var w = accelerator.Allocate1D<BFloat16>((long)N * K);
            Wan.WanComfyFormat.DecodeF8E4M3(accelerator, wr.View, w.View, 1f);                     // exact: every e4m3 value is a bf16 value
            await Check("k_proj", N, host, w, scale);
        }
        foreach (var line in report) Console.WriteLine("[Nvfp4] Gemma layer 24 vs ComfyUI: " + line);
        Ur_Expect(exact && worstCall < 3e-5, string.Join("; ", report));                      // MEASURED 2026-11-17 CUDA: weights bit-exact; calls 3.0e-6 (nvfp4) / 1.3e-6 (fp8) (6.8e-6 / 4.0e-6 before the MMA accumulation fix)
        return string.Join("; ", report);
    });
}
