using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.Ltx2;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    // LTX-2 fixtures: LTX2_MODELS (default D:/spawndev-ltx2: comfy/split_files/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors, lightricks/ltx-2-19b-dev-fp8.safetensors);
    // LTX2_DUMPS (default F:/spawndev-ltx2/comfyui-dumps: SpawnGen's ComfyUI taps, _DevComms 2026-11-17-1900)
    private static string Ltx2_Models() => Environment.GetEnvironmentVariable("LTX2_MODELS") ?? "D:/spawndev-ltx2";
    private static string Ltx2_Dumps() => Environment.GetEnvironmentVariable("LTX2_DUMPS") ?? "F:/spawndev-ltx2/comfyui-dumps";
    private static string Ltx2_Tap(string dir, string prefix)
    {
        var hits = Directory.GetFiles(dir, prefix + "_f32_*.raw");
        return hits.Length == 1 ? hits[0] : throw new FileNotFoundException($"{hits.Length} taps named {prefix}_f32_* in {dir}");
    }
    private static int[] Ltx2_PromptIds(string encDir)
    {
        using var doc = System.Text.Json.JsonDocument.Parse(File.ReadAllText(Path.Combine(encDir, "ltx2_te_ids.json")));
        return doc.RootElement.GetProperty("calls")[0].GetProperty("ids").EnumerateArray().Select(e => e.GetInt32()).ToArray();
    }

    /// <summary>
    /// LTX-2's Gemma 3 12B text encoder (ComfyUI's gemma_3_12B_it_fp4_mixed: NVFP4 + fp8 layers, streamed) against ComfyUI's float32 run (SpawnGen's taps): the embedding,
    /// layers 0 / 23 / 47, the final norm, and the normalised 49-slot features [116, 188160] the text projection reads. The prompt is the template's (1024 ids left-padded
    /// with 0; the 116 real ones are encoded).
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task Ltx2_Gemma3Encode_MatchesComfyUi() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is not (AcceleratorType.Cuda or AcceleratorType.OpenCL)) throw new UnsupportedTestException("reads a 9.4 GB file from disk: a desktop GPU");
        var path = Path.Combine(Ltx2_Models(), "comfy", "split_files", "text_encoders", "gemma_3_12B_it_fp4_mixed.safetensors");
        var enc = Path.Combine(Ltx2_Dumps(), "enc");
        if (!File.Exists(path)) throw new UnsupportedTestException($"{path} not found");
        if (!Directory.Exists(enc)) throw new UnsupportedTestException($"SpawnGen's LTX-2 encode taps not found in {enc}");
        var all = Ltx2_PromptIds(enc);
        int first = Array.FindIndex(all, id => id != 0), n = all.Length - first;
        Ur_Expect(all.Length == Gemma3TextEncoder.MaxPositions && all[first] == 2, $"expected 1024 left-padded ids starting at BOS, got {all.Length}, first real {all[first]}");
        var ids = all[first..];
        double residentGb = double.TryParse(Environment.GetEnvironmentVariable("LTX2_GEMMA_RESIDENT_GB"), System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out var g) ? g : 4.0;
        var sw = System.Diagnostics.Stopwatch.StartNew();
        using var te = await Gemma3TextEncoder.OpenAsync(accelerator, new LocalFolderModelSource(Path.GetDirectoryName(path)!), "", Path.GetFileName(path), (long)(residentGb * 1e9));
        double open = sw.Elapsed.TotalSeconds;
        const int H = Gemma3TextEncoder.Hidden;
        var probes = new Dictionary<string, MemoryBuffer1D<float, Stride1D.Dense>>();
        try
        {
            foreach (var name in new[] { "embed", "layer0", "layer23", "layer47", "norm" }) probes[name] = accelerator.Allocate1D<float>((long)n * H);
            te.Probe = (name, v, rows, cols) => { if (probes.TryGetValue(name, out var b)) b.View.CopyFrom(v.SubView(0, (long)rows * cols)); };
            sw.Restart();
            var features = await te.EncodeAsync(ids);
            var got = await features.CopyToHostAsync();
            double encode = sw.Elapsed.TotalSeconds;
            var report = new List<string>(); double worst = 0;
            void Cmp(string label, float[] mine, float[] gold) { double e = F2_RelRms(mine, gold); worst = Math.Max(worst, e); report.Add($"{label} {e:E2}"); }
            // [1, 1024, 3840] taps: the real rows
            float[] Rows(string tap) { var a = F2_Raw(Ltx2_Tap(enc, tap), 1024L * H); return a[(first * H)..]; }
            Cmp("embed", await probes["embed"].View.CopyToHostAsync(), Rows("ltx2_te_gemma3_12b_transformer_model_embed_tokens_0_out0"));
            foreach (int l in new[] { 0, 23, 47 }) Cmp($"layer {l}", await probes[$"layer{l}"].View.CopyToHostAsync(), Rows($"ltx2_te_gemma3_12b_transformer_model_layers_{l}_0_out0"));
            Cmp("final norm", await probes["norm"].View.CopyToHostAsync(), Rows("ltx2_te_gemma3_12b_transformer_model_norm_0_out0"));
            Cmp("features [116, 188160]", got, F2_Raw(Ltx2_Tap(enc, "ltx2_te_text_embedding_projection_0_in0"), (long)n * Gemma3TextEncoder.FeatureWidth));
            var wts = te.Weights;
            string line = $"{string.Join("; ", report)} | open {open:F1} s, encode {encode:F1} s ({wts.ResidentBlocks} of 48 layers resident, {wts.StreamedBytesPerForward / 1e9:F1} GB streamed: read {wts.ReadSeconds:F1} s, upload {wts.UploadSeconds:F1} s)";
            Console.WriteLine("[Ltx2] Gemma 3 encode vs ComfyUI: " + line);
            Ur_Expect(worst < 1e-4, line);   // MEASURED 2026-11-17 CUDA (tensor cores): layers 7e-6..1.3e-5, final norm 2.8e-5, features 1.1e-5 (bf16 1 + w; fp32 1 + w was 2e-3)
            return line;
        }
        finally { await accelerator.SynchronizeAsync(); foreach (var b in probes.Values) b.Dispose(); }
    });

    /// <summary>
    /// The exact bf16 gathered convolution the LTX-2 decoders use (<see cref="Trellis2.SparseKernels.ConvTapsBf16Exact"/> after <see cref="Ltx2Kernels.GatherLayoutBf16"/>, on
    /// <see cref="Ltx2Kernels.ReflectTable"/>'s neighbours) against a host loop: random activations, random bf16 weights in torch's [cout, cin, 3, 3, 3] layout, a bias.
    /// </summary>
    [TestMethod(Timeout = 120000)]
    public async Task Ltx2_ConvBf16Exact_MatchesHost() => await RunTest(async accelerator =>
    {
        const int T = 3, H = 4, W = 5, cin = 16, cout = 24, M = T * H * W;
        var rng = new Random(7);
        var x = new float[M * cin]; for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 2 - 1);
        var wRaw = new ushort[cout * cin * 27]; for (int i = 0; i < wRaw.Length; i++) wRaw[i] = (ushort)(BitConverter.SingleToInt32Bits((float)(rng.NextDouble() * 2 - 1)) >> 16);
        var bRaw = new ushort[cout]; for (int i = 0; i < cout; i++) bRaw[i] = (ushort)(BitConverter.SingleToInt32Bits((float)(rng.NextDouble() - 0.5)) >> 16);
        static float Bf(ushort u) => BitConverter.Int32BitsToSingle(u << 16);
        var want = new float[M * cout];
        for (int t = 0; t < T; t++) for (int h = 0; h < H; h++) for (int w = 0; w < W; w++)
            for (int o = 0; o < cout; o++)
            {
                double acc = Bf(bRaw[o]);
                for (int a = 0; a < 3; a++) for (int b = 0; b < 3; b++) for (int c = 0; c < 3; c++)
                {
                    int tt = Math.Clamp(t + a - 1, 0, T - 1), hh = h + b - 1, ww = w + c - 1;
                    hh = hh < 0 ? -hh : hh >= H ? 2 * H - 2 - hh : hh; ww = ww < 0 ? -ww : ww >= W ? 2 * W - 2 - ww : ww;
                    for (int ci = 0; ci < cin; ci++) acc += x[((tt * H + hh) * W + ww) * cin + ci] * Bf(wRaw[(o * cin + ci) * 27 + a * 9 + b * 3 + c]);
                }
                want[((t * H + h) * W + w) * cout + o] = (float)acc;
            }
        var l = new Ltx2Kernels(accelerator); var sk = new Trellis2.SparseKernels(accelerator);
        using var xb = accelerator.Allocate1D(x);
        using var wt = accelerator.Allocate1D<BFloat16>(wRaw.Length);
        wt.View.Cast<BFloat16, ushort>().CopyFromCPU(wRaw);
        using var bias = accelerator.Allocate1D<BFloat16>(cout); bias.View.Cast<BFloat16, ushort>().CopyFromCPU(bRaw);
        using var g = accelerator.Allocate1D<BFloat16>(wRaw.Length);
        using var nbr = accelerator.Allocate1D<int>(M * 27);
        using var o2 = accelerator.Allocate1D<float>(M * cout);
        l.GatherLayoutBf16(wt.View, g.View, cout, cin, 27);
        l.ReflectTable(nbr.View, T, H, W);
        sk.ConvTapsBf16Exact(xb.View, nbr.View, g.View, bias.View, o2.View, M, cin, cout, 27);
        var got = await o2.View.CopyToHostAsync();
        double e = F2_RelRms(got, want);
        // bisect: the neighbour table, the permuted weights, and the naive fp32 conv on the same table
        var nb = await nbr.View.CopyToHostAsync(); int badNbr = 0;
        for (int t = 0; t < T; t++) for (int h = 0; h < H; h++) for (int w = 0; w < W; w++) for (int kk = 0; kk < 27; kk++)
        {
            int tt = Math.Clamp(t + kk / 9 - 1, 0, T - 1), hh = h + (kk / 3) % 3 - 1, ww = w + kk % 3 - 1;
            hh = hh < 0 ? -hh : hh >= H ? 2 * H - 2 - hh : hh; ww = ww < 0 ? -ww : ww >= W ? 2 * W - 2 - ww : ww;
            if (nb[((t * H + h) * W + w) * 27 + kk] != (tt * H + hh) * W + ww) badNbr++;
        }
        var gHost = await g.View.Cast<BFloat16, ushort>().CopyToHostAsync(); int badW = 0;
        for (int o = 0; o < cout; o++) for (int k = 0; k < 27; k++) for (int c = 0; c < cin; c++) if (gHost[(o * 27 + k) * cin + c] != wRaw[(o * cin + c) * 27 + k]) badW++;
        var wf = gHost.Select(Bf).ToArray(); var bfl = bRaw.Select(Bf).ToArray();
        using var wfb = accelerator.Allocate1D(wf); using var bfb = accelerator.Allocate1D(bfl); using var o3 = accelerator.Allocate1D<float>(M * cout);
        sk.Conv3F32(xb.View, nbr.View, wfb.View, bfb.View, o3.View, M, cin, cout);
        double eNaive = F2_RelRms(await o3.View.CopyToHostAsync(), want);
        string line = $"bf16 GEMM {e:E2} (MEASURED 2026-11-17 CUDA: 3.6e-7; Gemm's f16-only gathered loop gave 2.1); naive fp32 on the same table + weights {eNaive:E2}; bad table entries {badNbr}; bad permuted weights {badW}";
        Console.WriteLine("[Ltx2] conv bisect: " + line);
        Ur_Expect(e < 1e-6, line);
        return line;
    });

    /// <summary>
    /// LTX-2's video VAE decoder (the checkpoint's BF16 vae.decoder) on ComfyUI's final latent [128, 2, 2, 2], against ComfyUI's VAEDecode frames [9, 64, 64, 3] (images in
    /// [0, 1]: clamp((v + 1) / 2) of the decoder's output).
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task Ltx2_VideoDecode_MatchesComfyUi() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is not (AcceleratorType.Cuda or AcceleratorType.OpenCL)) throw new UnsupportedTestException("reads a 27 GB checkpoint from disk: a desktop GPU");
        var path = Path.Combine(Ltx2_Models(), "lightricks", "ltx-2-19b-dev-fp8.safetensors");
        var dec = Path.Combine(Ltx2_Dumps(), "dec");
        if (!File.Exists(path)) throw new UnsupportedTestException($"{path} not found");
        if (!Directory.Exists(dec)) throw new UnsupportedTestException($"SpawnGen's LTX-2 decode taps not found in {dec}");
        const int Fz = 2, h = 2, w = 2, F = 9, Hp = 64, Wp = 64;
        var sw = System.Diagnostics.Stopwatch.StartNew();
        using var vae = await Ltx2VideoDecoder.OpenAsync(accelerator, new LocalFolderModelSource(Path.GetDirectoryName(path)!), "", Path.GetFileName(path));
        double open = sw.Elapsed.TotalSeconds;
        sw.Restart();
        var video = await vae.DecodeAsync(F2_Raw(Ltx2_Tap(dec, "ltx2_dec_video_latent_samples"), 128L * Fz * h * w), Fz, h, w);
        double run = sw.Elapsed.TotalSeconds;
        Ur_Expect(video.Length == 3L * F * Hp * Wp, $"decoded {video.Length} values, expected [3, {F}, {Hp}, {Wp}]");
        var gold = F2_Raw(Ltx2_Tap(dec, "ltx2_dec_frames_images"), (long)F * Hp * Wp * 3);       // [F, H, W, 3] in [0, 1]
        var mine = new float[gold.Length];
        for (int c = 0; c < 3; c++) for (int f = 0; f < F; f++) for (int y = 0; y < Hp; y++) for (int x = 0; x < Wp; x++)
            mine[(((long)f * Hp + y) * Wp + x) * 3 + c] = Math.Clamp((video[(((long)c * F + f) * Hp + y) * Wp + x] + 1f) / 2f, 0f, 1f);
        double e = F2_RelRms(mine, gold), maxAbs = 0;
        for (int i = 0; i < gold.Length; i++) maxAbs = Math.Max(maxAbs, Math.Abs(mine[i] - gold[i]));
        string line = $"frames [9, 64, 64, 3] {e:E2} (max |diff| {maxAbs:E2}) | open {open:F1} s, decode {run:F2} s";
        Console.WriteLine("[Ltx2] video decode vs ComfyUI: " + line);
        Ur_Expect(e < 1e-4, line);   // MEASURED 2026-11-17 CUDA: 9.4e-6 (max |diff| 2.8e-5 on [0, 1]); reflect padding, exact bf16 convolutions
        return line;
    });

    /// <summary>
    /// LTX-2's audio decode (the checkpoint's BF16 audio_vae.decoder + vocoder) on ComfyUI's final audio latent [8, 9, 16], against ComfyUI's taps: the decoder's spectrogram
    /// [2, 33, 64] and the vocoder's waveform [2, 7920] at 24 kHz.
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task Ltx2_AudioDecode_MatchesComfyUi() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is not (AcceleratorType.Cuda or AcceleratorType.OpenCL)) throw new UnsupportedTestException("reads a 27 GB checkpoint from disk: a desktop GPU");
        var path = Path.Combine(Ltx2_Models(), "lightricks", "ltx-2-19b-dev-fp8.safetensors");
        var dec = Path.Combine(Ltx2_Dumps(), "dec");
        if (!File.Exists(path)) throw new UnsupportedTestException($"{path} not found");
        if (!Directory.Exists(dec)) throw new UnsupportedTestException($"SpawnGen's LTX-2 decode taps not found in {dec}");
        const int Ta = 9, T = 4 * Ta - 3, M = 64, S = 240 * T;
        var sw = System.Diagnostics.Stopwatch.StartNew();
        using var ad = await Ltx2AudioDecoder.OpenAsync(accelerator, new LocalFolderModelSource(Path.GetDirectoryName(path)!), "", Path.GetFileName(path));
        double open = sw.Elapsed.TotalSeconds;
        using var specBuf = accelerator.Allocate1D<float>((long)T * M * 2);
        ad.Tap = (name, v, rows, cols) => { if (name == "spectrogram") specBuf.View.CopyFrom(v.SubView(0, (long)rows * cols)); };
        sw.Restart();
        var wav = await ad.DecodeAsync(F2_Raw(Ltx2_Tap(dec, "ltx2_dec_audio_latent_samples"), 8L * Ta * 16), Ta);
        double run = sw.Elapsed.TotalSeconds;
        var spec = await specBuf.View.CopyToHostAsync();
        var specCf = new float[2 * T * M];                                                       // channel-last [T M, 2] -> [2, T, M]
        for (int s = 0; s < 2; s++) for (int p = 0; p < T * M; p++) specCf[s * T * M + p] = spec[p * 2 + s];
        double eSpec = F2_RelRms(specCf, F2_Raw(Ltx2_Tap(dec, "ltx2_dec_avae_decoder_0"), 2L * T * M));
        Ur_Expect(wav.Length == 2 * S, $"waveform of {wav.Length} samples, expected 2 x {S}");
        double eVoc = F2_RelRms(wav, F2_Raw(Ltx2_Tap(dec, "ltx2_dec_avae_vocoder_0"), 2L * S));
        string line = $"spectrogram [2, 33, 64] {eSpec:E2}; waveform [2, 7920] {eVoc:E2} | open {open:F1} s, decode {run:F2} s";
        Console.WriteLine("[Ltx2] audio decode vs ComfyUI: " + line);
        Ur_Expect(eSpec < 1e-5 && eVoc < 3e-4, line);   // MEASURED 2026-11-17 CUDA: spectrogram 5.9e-7, waveform 3.6e-5
        return line;
    });

    /// <summary>
    /// LTX-2's 19B transformer, call 0 of the template (sigma 1, batch [cond, uncond]), on ComfyUI's own inputs (the packed latents into patchify_proj, the contexts into the
    /// caption projections), against ComfyUI's float32 taps: the six timestep embedders, the caption projections, patchify, blocks 0 and 47 (both streams), the output heads.
    /// LTX2_DIT_RESIDENT_GB (default 6): blocks kept on the device; the rest stream from the checkpoint.
    /// </summary>
    [TestMethod(Timeout = 3600000, Category = "HeavyModel")]
    public async Task Ltx2_Transformer_Call0_MatchesComfyUi() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is not (AcceleratorType.Cuda or AcceleratorType.OpenCL)) throw new UnsupportedTestException("streams a 27 GB checkpoint from disk: a desktop GPU");
        var path = Path.Combine(Ltx2_Models(), "lightricks", "ltx-2-19b-dev-fp8.safetensors");
        string call = Path.Combine(Ltx2_Dumps(), "call");
        if (!File.Exists(path)) throw new UnsupportedTestException($"{path} not found");
        if (!Directory.Exists(call)) throw new UnsupportedTestException($"SpawnGen's LTX-2 call taps not found in {call}");
        const int B = 2, nv = 8, na = 9, L = 1024, Dv = Ltx2Transformer.Dv, Da = Ltx2Transformer.Da;
        double residentGb = double.TryParse(Environment.GetEnvironmentVariable("LTX2_DIT_RESIDENT_GB"), System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out var g) ? g : 6.0;
        float[] Tap(string dir, string name, long count) => F2_Raw(Ltx2_Tap(dir, "ltx2_tr_diffusion_model_" + name), count);
        var t0 = Tap(call, "adaln_single_0_in0", B);
        Ur_Expect(Math.Abs(t0[0] - 1000f) < 1e-3 && Math.Abs(t0[1] - 1000f) < 1e-3, $"call 0's adaLN input is {t0[0]} / {t0[1]}, expected sigma 1 x 1000");
        var sw = System.Diagnostics.Stopwatch.StartNew();
        using var dit = await Ltx2Transformer.OpenAsync(accelerator, new LocalFolderModelSource(Path.GetDirectoryName(path)!), "", Path.GetFileName(path), B, nv, na, L, (long)(residentGb * 1e9));
        double open = sw.Elapsed.TotalSeconds;
        var want = new (string Probe, long Count, Func<float[]> Gold)[]
        {
            ("adaln_video", B * 6L * Dv, () => Tap(call, "adaln_single_0_out0", B * 6L * Dv)), ("adaln_video_emb", B * (long)Dv, () => Tap(call, "adaln_single_0_out1", B * (long)Dv)),
            ("adaln_audio", B * 6L * Da, () => Tap(call, "audio_adaln_single_0_out0", B * 6L * Da)), ("adaln_audio_emb", B * (long)Da, () => Tap(call, "audio_adaln_single_0_out1", B * (long)Da)),
            ("caption_video", (long)B * L * Dv, () => Tap(call, "caption_projection_0_out0", (long)B * L * Dv)), ("caption_audio", (long)B * L * Da, () => Tap(call, "audio_caption_projection_0_out0", (long)B * L * Da)),
            ("patchify_video", (long)B * nv * Dv, () => Tap(call, "patchify_proj_0_out0", (long)B * nv * Dv)), ("patchify_audio", (long)B * na * Da, () => Tap(call, "audio_patchify_proj_0_out0", (long)B * na * Da)),
            ("block0_video", (long)B * nv * Dv, () => Tap(call, "transformer_blocks_0_0_out0", (long)B * nv * Dv)), ("block0_audio", (long)B * na * Da, () => Tap(call, "transformer_blocks_0_0_out1", (long)B * na * Da)),
            ("block47_video", (long)B * nv * Dv, () => Tap(call, "transformer_blocks_47_0_out0", (long)B * nv * Dv)), ("block47_audio", (long)B * na * Da, () => Tap(call, "transformer_blocks_47_0_out1", (long)B * na * Da)),
        };
        var probes = want.ToDictionary(x => x.Probe, x => accelerator.Allocate1D<float>(x.Count));
        try
        {
            dit.Probe = (name, v, rows, cols) => { if (probes.TryGetValue(name, out var b)) b.View.CopyFrom(v.SubView(0, (long)rows * cols)); };
            using var video = accelerator.Allocate1D(Tap(call, "patchify_proj_0_in0", (long)B * nv * 128));
            using var audio = accelerator.Allocate1D(Tap(call, "audio_patchify_proj_0_in0", (long)B * na * 128));
            using var vctx = accelerator.Allocate1D(Tap(call, "caption_projection_0_in0", (long)B * L * 3840));
            using var actx = accelerator.Allocate1D(Tap(call, "audio_caption_projection_0_in0", (long)B * L * 3840));
            sw.Restart();
            var (vOut, aOut) = await dit.CallAsync(video.View, audio.View, vctx.View, actx.View, new[] { 1000f, 1000f }, 2, 2, 2, na, 25f);
            var gv = await vOut.CopyToHostAsync(); var ga = await aOut.CopyToHostAsync();
            double run = sw.Elapsed.TotalSeconds;
            var report = new List<string>(); double worst = 0;
            void Cmp(string label, float[] mine, float[] gold) { double e = F2_RelRms(mine, gold); worst = Math.Max(worst, e); report.Add($"{label} {e:E2}"); }
            foreach (var (name, _, gold) in want) Cmp(name, await probes[name].View.CopyToHostAsync(), gold());
            Cmp("proj_out", gv, Tap(call, "proj_out_0_out0", (long)B * nv * 128));
            Cmp("audio_proj_out", ga, Tap(call, "audio_proj_out_0_out0", (long)B * na * 128));
            var wts = dit.Weights;
            string line = $"{string.Join("; ", report)} | open {open:F1} s, call {run:F1} s ({wts.ResidentBlocks} of 48 blocks resident, {wts.StreamedBytesPerForward / 1e9:F1} GB streamed: read {wts.ReadSeconds:F1} s, upload {wts.UploadSeconds:F1} s)";
            Console.WriteLine("[Ltx2] transformer call 0 vs ComfyUI: " + line);
            Ur_Expect(worst < 2e-4, line);   // MEASURED 2026-11-17 CUDA (tensor cores): embedders 8e-7, blocks 2e-5..6.3e-5, outputs 4.6e-5 / 1.3e-5 (q/k eps 1e-5, v2a after a2v)
            return line;
        }
        finally { await accelerator.SynchronizeAsync(); foreach (var b in probes.Values) b.Dispose(); }
    });

    /// <summary>
    /// LTX-2's text projection and connectors (the checkpoint's BF16 weights) on ComfyUI's own features [116, 188160] (so Gemma's error is not in it), against ComfyUI's
    /// float32 taps: the projection [116, 3840], each connector's output [1024, 3840] (registers, two blocks, final norm), and the conditioning [1024, 7680].
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task Ltx2_TextConnectors_MatchComfyUi() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is not (AcceleratorType.Cuda or AcceleratorType.OpenCL)) throw new UnsupportedTestException("reads a 27 GB checkpoint from disk: a desktop GPU");
        var path = Path.Combine(Ltx2_Models(), "lightricks", "ltx-2-19b-dev-fp8.safetensors");
        var enc = Path.Combine(Ltx2_Dumps(), "enc");
        if (!File.Exists(path)) throw new UnsupportedTestException($"{path} not found");
        if (!Directory.Exists(enc)) throw new UnsupportedTestException($"SpawnGen's LTX-2 encode taps not found in {enc}");
        const int n = 116, C = Ltx2TextConnectors.Width, T = Ltx2TextConnectors.Tokens;
        var sw = System.Diagnostics.Stopwatch.StartNew();
        using var tc = await Ltx2TextConnectors.OpenAsync(accelerator, new LocalFolderModelSource(Path.GetDirectoryName(path)!), "", Path.GetFileName(path));
        double open = sw.Elapsed.TotalSeconds;
        var probes = new Dictionary<string, MemoryBuffer1D<float, Stride1D.Dense>> { ["projection"] = accelerator.Allocate1D<float>((long)n * C), ["video"] = accelerator.Allocate1D<float>((long)T * C), ["audio"] = accelerator.Allocate1D<float>((long)T * C) };
        try
        {
            tc.Probe = (name, v, rows, cols) => { if (probes.TryGetValue(name, out var b)) b.View.CopyFrom(v.SubView(0, (long)rows * cols)); };
            using var feat = accelerator.Allocate1D(F2_Raw(Ltx2_Tap(enc, "ltx2_te_text_embedding_projection_0_in0"), (long)n * Gemma3TextEncoder.FeatureWidth));
            sw.Restart();
            var cond = await (await tc.RunAsync(feat.View, n)).CopyToHostAsync();
            double run = sw.Elapsed.TotalSeconds;
            var report = new List<string>(); double worst = 0;
            void Cmp(string label, float[] mine, float[] gold) { double e = F2_RelRms(mine, gold); worst = Math.Max(worst, e); report.Add($"{label} {e:E2}"); }
            Cmp("projection", await probes["projection"].View.CopyToHostAsync(), F2_Raw(Ltx2_Tap(enc, "ltx2_te_text_embedding_projection_0_out0"), (long)n * C));
            Cmp("video connector", await probes["video"].View.CopyToHostAsync(), F2_Raw(Ltx2_Tap(enc, "ltx2_te_video_embeddings_connector_0_out0"), (long)T * C));
            Cmp("audio connector", await probes["audio"].View.CopyToHostAsync(), F2_Raw(Ltx2_Tap(enc, "ltx2_te_audio_embeddings_connector_0_out0"), (long)T * C));
            Cmp("conditioning [1024, 7680]", cond, F2_Raw(Ltx2_Tap(enc, "ltx2_te_pos_0_cond"), (long)T * 2 * C));
            string line = $"{string.Join("; ", report)} | open {open:F1} s, run {run:F2} s";
            Console.WriteLine("[Ltx2] text connectors vs ComfyUI: " + line);
            Ur_Expect(worst < 1e-3, line);   // MEASURED 2026-11-17 CUDA: projection 4.6e-4 with tensor cores (fp32 GEMM 6.9e-6: open library issue), conditioning 3.2e-5
            return line;
        }
        finally { await accelerator.SynchronizeAsync(); foreach (var b in probes.Values) b.Dispose(); }
    });
}
