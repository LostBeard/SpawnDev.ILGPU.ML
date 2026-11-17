using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.QwenImage;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    /// <summary>
    /// Streaming cost of a 20B transformer that does not fit the card (Qwen-Image Edit 2511, 17 of 60 blocks resident at 6 GB): the same model call (q11_1 call 0) three
    /// times on one open transformer: (1) no read-ahead, no host cache; (2) read-ahead on, a host cache (QI_BENCH_CACHE_GB, default 6) filling; (3) the same with the cache warm.
    /// Reports each pass's wall time and the part the caller waited on streaming; every pass must still match ComfyUI (the order of reads must not change the math).
    /// MEASURED 2026-11-17 CUDA, the file on a USB 3.0 SATA drive (about 0.32 GB/s), 14.6 GB streamed per call, compute about 7 s: no read-ahead 52.1 s (45.1 s waiting),
    /// read-ahead + cache filling 44.8 s (39.2), cache warm (5.8 GB) 31.7 s (26.1); all 1.01e-3 vs ComfyUI. The disk dominates: read-ahead hides one block behind ~7 s of
    /// compute, the RAM tier removes its share of the reads.
    /// </summary>
    [TestMethod(Timeout = 3600000, Category = "HeavyModel")]
    public async Task QwenEdit2511_Streaming_Benchmark() => await RunTest(async accelerator =>
    {
        var path = Qe_Transformer(accelerator, "qwen_image_edit_2511_fp8mixed.safetensors"); var d = Qe_Dumps();
        double residentGb = double.TryParse(Environment.GetEnvironmentVariable("QI_RESIDENT_GB"), System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out var g) ? g : 6.0;
        double cacheGb = double.TryParse(Environment.GetEnvironmentVariable("QI_BENCH_CACHE_GB"), System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out var c) ? c : 6.0;
        var k = new Flux1.Flux1Kernels(accelerator);
        using var tr = await QwenImageTransformer.OpenAsync(accelerator, new LocalFolderModelSource(Path.GetDirectoryName(path)!), "", Path.GetFileName(path), 212, 12, 16, new[] { (56, 74) }, (long)(residentGb * 1e9));
        var lines = new List<string> { $"{tr.ResidentBlocks} of 60 blocks resident, {tr.StreamedBytesPerForward / 1e9:F1} GB streamed per call, host cache {cacheGb} GB" };
        double worst = 0;
        foreach (var (label, prefetch, cache) in new[] { ("no read-ahead, no cache", false, 0.0), ("read-ahead + cache filling", true, cacheGb), ("read-ahead + cache warm", true, cacheGb) })
        {
            tr.Weights.Prefetch = prefetch; tr.Weights.HostCacheBytes = (long)(cache * 1e9);
            var report = new List<string>();
            double s0 = tr.Weights.StreamSeconds; var sw = System.Diagnostics.Stopwatch.StartNew();
            double e = await Qe_CallAsync(accelerator, tr, k, d, "q11_1", 0, 212, 1, report);
            worst = Math.Max(worst, e);
            lines.Add($"{label}: {sw.Elapsed.TotalSeconds:F1} s (waited on streaming {tr.Weights.StreamSeconds - s0:F1} s), host cache {tr.Weights.HostCacheUsed / 1e9:F1} GB, vs ComfyUI {e:E2}");
        }
        foreach (var line in lines) Console.WriteLine("[QwenEdit] streaming benchmark: " + line);
        Ur_Expect(worst < 1.5e-3, string.Join("; ", lines));                                  // the 2511 gate's bound: streaming changes the timing, never the result
        return string.Join("; ", lines);
    });

    /// <summary>
    /// Where the 2511 transformer's residual difference to ComfyUI builds up: SpawnGen's taps of q11_1 call 0 (the positive at sigma 1, float32 CPU) at block 0's attention,
    /// blocks 0 / 29 / 59 (both streams), norm_out and proj_out, against <see cref="QwenImageTransformer.Probe"/> at the same points. Relative RMS per stage.
    /// MEASURED 2026-11-16, CUDA: block 0 attention img 1.83e-3 / txt 1.5e-5; block 0 img 1.6e-4 / txt 1.8e-5; block 29 2.6e-4 / 2.4e-4; block 59 2.7e-4 / 6.9e-4;
    /// norm_out 6.4e-4; proj_out 4.2e-4. The image attention's 1.8e-3 is float32 conditioning, not a difference in the math: a float64 rebuild of block 0's attention from
    /// the file's weights and the call's taps gives ComfyUI's text output to 7.7e-7 but its image output only to 1.1e-3, and the same rebuild in float32 moves the image rows
    /// by 9.9e-4 (text rows 4.8e-7): the image queries' softmax is so sharp (bf16-rounding the image keys alone moves the output 0.28) that float32 itself carries ~1e-3 there.
    /// That is the floor of the 1-picture sigma-1 calls' 1e-3 in <see cref="QwenEdit2511_Transformer_MatchesComfyUiCalls"/>.
    /// </summary>
    [TestMethod(Timeout = 3600000, Category = "HeavyModel")]
    public async Task QwenEdit2511_Stages_MatchComfyUiTaps() => await RunTest(async accelerator =>
    {
        var path = Qe_Transformer(accelerator, "qwen_image_edit_2511_fp8mixed.safetensors"); var d = Qe_Dumps();
        string T(string module, string rest) => Path.Combine(d, $"q11_1_taps2_diffusion_model_{module}_0_{rest}.raw");
        if (!File.Exists(T("proj_out", "out0_f32_1x4336x64"))) throw new UnsupportedTestException($"SpawnGen's 2511 stage taps (q11_1_taps2_*) not found in {d}");
        double residentGb = double.TryParse(Environment.GetEnvironmentVariable("QI_RESIDENT_GB"), System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out var g) ? g : 6.0;
        const int NoiseRows = 12 * 16;                                                      // the noise latent [16, 24, 32] packs 2 x 2 into 12 x 16 tokens; the reference adds 56 x 74 = 4144
        // our probe name -> (ComfyUI tap file, rows, columns)
        var stages = new (string Probe, string File, int Rows, int Cols)[]
        {
            ("transformer_blocks.0.attn_img", T("transformer_blocks_0_attn", "out0_f32_1x4336x3072"), 4336, 3072),
            ("transformer_blocks.0.attn_txt", T("transformer_blocks_0_attn", "out1_f32_1x212x3072"), 212, 3072),
            ("block0_txt", T("transformer_blocks_0", "out0_f32_1x212x3072"), 212, 3072),
            ("block0_img", T("transformer_blocks_0", "out1_f32_1x4336x3072"), 4336, 3072),
            ("block29_txt", T("transformer_blocks_29", "out0_f32_1x212x3072"), 212, 3072),
            ("block29_img", T("transformer_blocks_29", "out1_f32_1x4336x3072"), 4336, 3072),
            ("block59_txt", T("transformer_blocks_59", "out0_f32_1x212x3072"), 212, 3072),
            ("block59_img", T("transformer_blocks_59", "out1_f32_1x4336x3072"), 4336, 3072),
            ("norm_out", T("norm_out", "out0_f32_1x4336x3072"), NoiseRows, 3072),         // the head runs on the noise latent's rows only
            ("proj_out", T("proj_out", "out0_f32_1x4336x64"), NoiseRows, 64),
        };
        // captured to the host as the forward runs (desktop only: CopyToCPU), so the probes take no device memory next to a model that nearly fills the card
        var captured = new Dictionary<string, float[]>();
        var wanted = stages.ToDictionary(s => s.Probe, s => s.Rows * s.Cols);
        {
            var k = new Flux1.Flux1Kernels(accelerator);
            var report = new List<string>();
            using var tr = await QwenImageTransformer.OpenAsync(accelerator, new LocalFolderModelSource(Path.GetDirectoryName(path)!), "", Path.GetFileName(path), 212, 12, 16, new[] { (56, 74) }, (long)(residentGb * 1e9));
            tr.Probe = (name, view, rows, cols) => { if (wanted.TryGetValue(name, out int n)) { var host = new float[n]; accelerator.Synchronize(); view.SubView(0, n).CopyToCPU(host); captured[name] = host; } };
            double call = await Qe_CallAsync(accelerator, tr, k, d, "q11_1", 0, 212, 1, report);
            tr.Probe = null;
            double worst = 0;
            foreach (var s in stages)
            {
                var ours = captured[s.Probe];
                long fileRows = new FileInfo(s.File).Length / 4 / s.Cols;                         // norm_out / proj_out taps hold all 4336 rows: ours are the leading (noise) rows
                var theirs = F2_Raw(s.File, fileRows * s.Cols)[..(s.Rows * s.Cols)];
                double e = F2_RelRms(ours, theirs); worst = Math.Max(worst, e);
                string split = "";
                if (s.Rows == 4336)                                                               // noise rows vs reference rows
                    split = $" (noise rows {F2_RelRms(ours[..(NoiseRows * s.Cols)], theirs[..(NoiseRows * s.Cols)]):E2}, reference rows {F2_RelRms(ours[(NoiseRows * s.Cols)..], theirs[(NoiseRows * s.Cols)..]):E2})";
                report.Add($"{s.Probe}: {e:E2}{split}");
            }
            foreach (var line in report) Console.WriteLine("[QwenEdit] 2511 stages vs ComfyUI: " + line);
            Ur_Expect(worst < 3e-3, string.Join("; ", report));                                 // measured worst 1.83e-3 (block 0 image attention, float32 conditioning)
            return string.Join("; ", report);
        }
    });
}
