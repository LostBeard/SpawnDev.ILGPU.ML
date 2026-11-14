using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Flux2;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.Trellis2;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// The FLUX.2 klein 4B transformer from the files ComfyUI's templates load, in Black Forest Labs' layout (<see cref="Flux2ComfyFormat"/>): Comfy-Org/flux2-klein's bf16
/// <c>flux-2-klein-4b.safetensors</c> and black-forest-labs/FLUX.2-klein-4b-fp8's <c>flux-2-klein-4b-fp8.safetensors</c>, in F2_DIR\comfy, against our export
/// (weights\transformer.safetensors, read from disk one tensor at a time so the two never share the card). Desktop GPU lanes.
/// </summary>
public abstract partial class MLTestBase
{
    private static string F2_RequireComfyTransformer(Accelerator accelerator, string file)
    {
        var dir = F2_Require(accelerator);
        if (!File.Exists(Path.Combine(dir, "comfy", file)) || !File.Exists(Path.Combine(dir, "weights", "transformer.safetensors")))
            throw new UnsupportedTestException($"{file} (ComfyUI's template file) not found in {dir}\\comfy, or our weights\\transformer.safetensors");
        return dir;
    }

    /// <summary>Every tensor of a loaded transformer (as <see cref="Flux2Weights.Matrix"/> gives it: the 8-bit ones decoded) against our bf16 export: (bit-identical count, decoded count, worst relRMS of the rest).</summary>
    private static async Task<(int Exact, int Near, double Worst, string WorstName)> F2_CompareWithExport(Flux2Weights w, string oursPath)
    {
        using var ours = SpawnDev.ILGPU.ML.SafeTensors.SafeTensorsFileReader.Open(oursPath);
        int exact = 0, near = 0; double worst = 0; string worstName = "";
        foreach (var t in ours.Tensors)
        {
            if (t.DType != "BF16") throw new Exception($"{t.Name}: our export is {t.DType}, expected BF16");
            var raw = new byte[t.DataLength];
            ours.ReadBytes(t, 0, raw);
            var mine = await w.Matrix(t.Name).CopyToHostAsync();
            if (mine.Length * 2L != raw.Length) throw new Exception($"{t.Name}: {mine.Length} values vs {raw.Length / 2}");
            bool same = true; double se = 0, sr = 0;
            for (int i = 0; i < mine.Length; i++)
            {
                ushort a = (ushort)(raw[2 * i] | (raw[2 * i + 1] << 8));
                if (mine[i].RawValue != a) same = false;
                double x = BitConverter.Int32BitsToSingle(a << 16), y = (float)mine[i];
                se += (y - x) * (y - x); sr += x * x;
            }
            if (same) { exact++; continue; }
            near++;
            double rel = Math.Sqrt(se / Math.Max(sr, 1e-30));
            if (rel > worst) { worst = rel; worstName = t.Name; }
        }
        return (exact, near, worst, worstName);
    }

    /// <summary>ComfyUI's bf16 klein 4B transformer loads to our export bit for bit (149 tensors, the final modulation's halves exchanged by the adapter).</summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task Flux2_ComfyTransformer_Bf16_AsOurExport() => await RunTest(async accelerator =>
    {
        const string File_ = "flux-2-klein-4b.safetensors";
        var dir = F2_RequireComfyTransformer(accelerator, File_);
        using var w = await Flux2Weights.LoadAsync(accelerator, new LocalFolderModelSource(Path.Combine(dir, "comfy")), "", File_);
        var (exact, near, worst, worstName) = await F2_CompareWithExport(w, Path.Combine(dir, "weights", "transformer.safetensors"));
        var report = $"{exact} tensors bit-identical to our export, {near} differ (worst {worstName} {worst:E2}); {w.Bytes / 1e9:F2} GB on the device";
        Console.WriteLine("[Flux2] ComfyUI bf16 transformer: " + report);
        Ur_Expect(near == 0 && exact == 149, report);
        return report;
    });

    /// <summary>
    /// Black Forest Labs' 8-bit float klein 4B transformer: the 69 bf16 tensors bit-identical to our export, the 80 F8_E4M3 matrices (decoded with their weight_scale just before use)
    /// within the format's own rounding of our bf16 weights (2.7e-2 relative RMS measured on the host), held as 8-bit on the device.
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task Flux2_ComfyTransformer_Fp8_NearOurExport() => await RunTest(async accelerator =>
    {
        const string File_ = "flux-2-klein-4b-fp8.safetensors";
        var dir = F2_RequireComfyTransformer(accelerator, File_);
        using var w = await Flux2Weights.LoadAsync(accelerator, new LocalFolderModelSource(Path.Combine(dir, "comfy")), "", File_);
        var (exact, near, worst, worstName) = await F2_CompareWithExport(w, Path.Combine(dir, "weights", "transformer.safetensors"));
        var report = $"{exact} tensors bit-identical, {near} 8-bit matrices worst {worstName} {worst:E2} relative RMS; {w.Bytes / 1e9:F2} GB on the device";
        Console.WriteLine("[Flux2] BFL fp8 transformer: " + report);
        Ur_Expect(exact == 69 && near == 80, $"69 bf16 + 80 8-bit tensors: {report}");
        Ur_Expect(worst < 0.03, $"the 8-bit matrices within the format's rounding: {report}");
        Ur_Expect(w.Bytes < 5_000_000_000L, $"held as 8 bits on the device: {report}");
        return report;
    });

    /// <summary>
    /// ComfyUI's qwen_3_4b.safetensors (Hugging Face Qwen3-4B names, all 36 layers) loads, through the layer filter and the q|k|v / gate|up fusing, to our text encoder export
    /// (embedding + 27 layers) bit for bit, and only those 27 layers are uploaded.
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task Flux2_ComfyTextEncoder_AsOurExport() => await RunTest(async accelerator =>
    {
        const string File_ = "qwen_3_4b.safetensors";
        var dir = F2_Require(accelerator);
        if (!File.Exists(Path.Combine(dir, "comfy", File_)) || !File.Exists(Path.Combine(dir, "weights", "text_encoder.safetensors")))
            throw new UnsupportedTestException($"{File_} not found in {dir}\\comfy, or our weights\\text_encoder.safetensors");
        using var w = await Flux2Weights.LoadAsync(accelerator, new LocalFolderModelSource(Path.Combine(dir, "comfy")), "", File_);
        var (exact, near, worst, worstName) = await F2_CompareWithExport(w, Path.Combine(dir, "weights", "text_encoder.safetensors"));
        var report = $"{exact} tensors bit-identical to our export, {near} differ (worst {worstName} {worst:E2}); {w.Bytes / 1e9:F2} GB on the device";
        Console.WriteLine("[Flux2] ComfyUI qwen_3_4b: " + report);
        Ur_Expect(near == 0 && exact == 1 + 8 * Flux2TextEncoder.Layers, report);
        Ur_Expect(w.Bytes < 6_500_000_000L, $"only the first {Flux2TextEncoder.Layers} layers uploaded: {report}");
        return report;
    });

    /// <summary>The whole klein pipeline on the 8-bit float transformer from the reference's prompt and noise, against the reference picture (the bf16 file is our export bit for bit).</summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task Flux2_ComfyTransformer_Fp8_Pipeline() => await RunTest(async accelerator =>
    {
        const string File_ = "flux-2-klein-4b-fp8.safetensors";
        var dir = F2_RequireComfyTransformer(accelerator, File_); var g = Path.Combine(dir, "golden", "cat");
        var (noise, _) = Ur_Npy<float>(Path.Combine(g, "noise.npy")); var (image, _) = Ur_Npy<float>(Path.Combine(g, "image.npy"));
        var tok = new Qwen2Tokenizer(File.ReadAllText(Path.Combine(dir, "klein4b", "tokenizer", "tokenizer.json")));
        using var vae = InferenceSession.CreateFromFile(accelerator, File.ReadAllBytes(Path.Combine(dir, "weights", "vae_decoder.onnx")), inputShapes: new Dictionary<string, int[]> { ["latents"] = new[] { 1, 32, 64, 64 } });
        var pipe = new Flux2Pipeline(accelerator, tok,
            () => Task.FromResult(new Flux2Weights(accelerator, GpuWeights.LoadFile(accelerator, Path.Combine(dir, "weights", "text_encoder.safetensors")))),
            () => Flux2Weights.LoadAsync(accelerator, new LocalFolderModelSource(Path.Combine(dir, "comfy")), "", File_), vae, F2_Norm(dir));
        var sw = System.Diagnostics.Stopwatch.StartNew();
        var result = await pipe.RunAsync(new Flux2Request { Prompt = "a photo of a cat sitting on a red sofa", Size = 512, Steps = 4, Noise = noise });
        double sq = 0; int n = 512 * 512;
        for (int p = 0; p < n; p++)
            for (int c = 0; c < 3; c++)
            {
                double want = Math.Clamp(Math.Round((image[c * n + p] / 2 + 0.5) * 255), 0, 255), have = result.Rgba[p * 4 + c];
                sq += (have - want) * (have - want);
            }
        double psnr = 10 * Math.Log10(255.0 * 255.0 / Math.Max(sq / (3.0 * n), 1e-9));
        var report = $"8-bit float transformer pipeline: {sw.Elapsed.TotalSeconds:F1} s, PSNR {psnr:F1} dB vs the reference picture";
        Console.WriteLine("[Flux2] " + report);
        // MEASURED (RTX 4070, CUDA, 2026-11-13): 24.4 dB. For scale, same reference picture: our bf16 export 29.3 dB, our 8-bit block export 29.2 dB (0.5% per-matrix error).
        // BFL's 8-bit floats err 2.7% per matrix (one scale per matrix), which moves the 4-step trajectory: the format's cost, not the decode's (every matrix equals an
        // independent decode within its rounding: Flux2_ComfyTransformer_Fp8_NearOurExport). The bound guards that cost from growing.
        Ur_Expect(psnr > 24.0, report);
        return report;
    });
}
