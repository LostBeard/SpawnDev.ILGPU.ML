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
            Ur_Expect(worst < 3e-5, line);   // MEASURED 2026-11-17 CUDA (tensor cores, rounded accumulation): layers 8e-7..1.9e-6, final norm 5.8e-6, features 2.3e-6 (bf16 1 + w; fp32 1 + w was 2e-3)
            return line;
        }
        finally { await accelerator.SynchronizeAsync(); foreach (var b in probes.Values) b.Dispose(); }
    });

    /// <summary>
    /// Diagnostic for the tensor-core "precise" GEMM (<see cref="Trellis2.DiTKernels.PreciseActivations"/>: activations split into bf16 hi + lo) against the fp32 tiled GEMM and a
    /// float64 host reference, over K from 1k to 188k (LTX-2's text projection), with random-sign and all-positive data. MEASURED 2026-11-17 RTX 4070: with the MMA accumulating
    /// in place the error grew linearly with K (5.7e-4 signed / 2.2e-3 positive at 188160: the tensor cores truncate); with each k step's MMA from zero and an ordinary fp32
    /// add, 2.9e-6 / 1.5e-6 (the tiled fp32 GEMM: 7.5e-6 / 8.7e-6). Both paths must stay below 2e-5.
    /// </summary>
    [TestMethod(Timeout = 600000)]
    public async Task Ltx2_PreciseGemm_KSweep() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType != AcceleratorType.Cuda) throw new UnsupportedTestException("the tensor-core path is CUDA's");
        const int M = 16, N = 64;
        var lines = new List<string>(); double worstTiled = 0, worstMma = 0;
        var k = new Trellis2.DiTKernels(accelerator) { PreciseActivations = true };
        foreach (bool positive in new[] { false, true })
            foreach (int K in new[] { 1024, 4096, 16384, 65536, 188160 })
            {
                var rng = new Random(K + (positive ? 1 : 0));
                var x = new float[(long)M * K]; for (long i = 0; i < x.Length; i++) x[i] = positive ? (float)rng.NextDouble() : (float)(rng.NextDouble() * 2 - 1);
                var wRaw = new ushort[(long)N * K];
                for (long i = 0; i < wRaw.Length; i++) wRaw[i] = (ushort)(BitConverter.SingleToInt32Bits(positive ? (float)rng.NextDouble() * 0.01f : (float)(rng.NextDouble() * 2 - 1) * 0.01f) >> 16);
                var want = new float[M * N];
                for (int m = 0; m < M; m++)
                    for (int n = 0; n < N; n++)
                    {
                        double acc = 0;
                        for (int kk = 0; kk < K; kk++) acc += (double)x[(long)m * K + kk] * BitConverter.Int32BitsToSingle(wRaw[(long)n * K + kk] << 16);
                        want[m * N + n] = (float)acc;
                    }
                using var xb = accelerator.Allocate1D(x); using var wb = accelerator.Allocate1D<BFloat16>(wRaw.Length); wb.View.Cast<BFloat16, ushort>().CopyFromCPU(wRaw);
                using var o = accelerator.Allocate1D<float>(M * N);
                bool saved = Trellis2.DiTKernels.EnableTensorCores;
                try
                {
                    Trellis2.DiTKernels.EnableTensorCores = true;
                    k.Linear(xb.View, wb.View, null, o.View, M, K, N);
                    double eMma = F2_RelRms(await o.View.CopyToHostAsync(), want);
                    Trellis2.DiTKernels.EnableTensorCores = false;
                    k.Linear(xb.View, wb.View, null, o.View, M, K, N);
                    double eTiled = F2_RelRms(await o.View.CopyToHostAsync(), want);
                    worstTiled = Math.Max(worstTiled, eTiled); worstMma = Math.Max(worstMma, eMma);
                    lines.Add($"{(positive ? "positive" : "signed")} K {K}: tensor cores x2 {eMma:E2}, tiled fp32 {eTiled:E2}");
                }
                finally { Trellis2.DiTKernels.EnableTensorCores = saved; }
            }
        // speed: the x2 (precise) kernel against the single-pass bf16 one on a DiT-sized GEMM [1024, 3072] x [3072, 3072]^T (two MMA passes: ~2x is its floor)
        {
            const int Mb = 1024, Kb = 3072, Nb = 3072, reps = 20;
            using var xb = accelerator.Allocate1D<float>((long)Mb * Kb); using var wb = accelerator.Allocate1D<BFloat16>((long)Nb * Kb); using var o = accelerator.Allocate1D<float>((long)Mb * Nb);
            xb.MemSetToZero(); wb.MemSetToZero();
            double Time(bool precise)
            {
                var kk = new Trellis2.DiTKernels(accelerator) { PreciseActivations = precise };
                kk.Linear(xb.View, wb.View, null, o.View, Mb, Kb, Nb); accelerator.Synchronize();
                var sw = System.Diagnostics.Stopwatch.StartNew();
                for (int r = 0; r < reps; r++) kk.Linear(xb.View, wb.View, null, o.View, Mb, Kb, Nb);
                accelerator.Synchronize();
                return sw.Elapsed.TotalMilliseconds / reps;
            }
            double single = Time(false), x2 = Time(true);
            lines.Add($"speed [1024, 3072] x [3072, 3072]: bf16 {single:F3} ms ({2.0 * Mb * Kb * Nb / single / 1e9:F1} TFLOPS), x2 {x2:F3} ms ({x2 / single:F2}x)");
        }
        foreach (var l in lines) Console.WriteLine("[Ltx2] precise GEMM sweep: " + l);
        Ur_Expect(worstTiled < 2e-5 && worstMma < 2e-5, string.Join("; ", lines));
        return string.Join("; ", lines);
    });

    /// <summary>
    /// SentencePiece BPE (<see cref="Preprocessing.SentencePieceTokenizer.FromSentencePieceModel"/>, the Gemma file's own <c>spiece_model</c>) against ComfyUI's ids for LTX-2's
    /// template prompt (SpawnGen's encode tap): the text is the reference ids decoded (identity normalizer: exact), and encoding it must give the same ids. The greedy longest
    /// match this replaced split "sways" as "▁sway" + "s" (SentencePiece: "▁s" + "ways"). Host only.
    /// </summary>
    [TestMethod(Timeout = 300000)]
    public async Task Ltx2_GemmaTokenizer_MatchesComfyUi() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is not (AcceleratorType.Cuda or AcceleratorType.OpenCL or AcceleratorType.CPU)) throw new UnsupportedTestException("reads the model file from disk");
        var path = Path.Combine(Ltx2_Models(), "comfy", "split_files", "text_encoders", "gemma_3_12B_it_fp4_mixed.safetensors");
        var enc = Path.Combine(Ltx2_Dumps(), "enc");
        if (!File.Exists(path) || !Directory.Exists(enc)) throw new UnsupportedTestException($"needs {path} and SpawnGen's taps in {enc}");
        byte[] spm;
        await using (var fs = File.OpenRead(path))
        {
            var lenBytes = new byte[8]; await fs.ReadExactlyAsync(lenBytes);
            var header = new byte[BitConverter.ToInt64(lenBytes)]; await fs.ReadExactlyAsync(header);
            using var doc = System.Text.Json.JsonDocument.Parse(header);
            var offs = doc.RootElement.GetProperty("spiece_model").GetProperty("data_offsets");
            long a = offs[0].GetInt64(), b = offs[1].GetInt64();
            fs.Seek(8 + header.Length + a, SeekOrigin.Begin);
            spm = new byte[b - a]; await fs.ReadExactlyAsync(spm);
        }
        var tok = Preprocessing.SentencePieceTokenizer.FromSentencePieceModel(spm);
        var all = Ltx2_PromptIds(enc);
        var ids = all[Array.FindIndex(all, id => id != 0)..];                                   // BOS 2, then the prompt
        Ur_Expect(ids[0] == tok.BosId, $"BOS {tok.BosId}, ComfyUI's first id {ids[0]}");
        var text = tok.Decode(ids[1..]);
        var mine = tok.Encode(text);
        int diff = Enumerable.Range(0, Math.Min(mine.Length, ids.Length - 1)).FirstOrDefault(i => mine[i] != ids[i + 1], -1);
        string line = $"{ids.Length - 1} prompt ids, ours {mine.Length}, first difference at {diff} | add_dummy_prefix {tok.AddSpacePrefix}, text \"{(text.Length > 60 ? text[..60] + "..." : text)}\"";
        Ur_Expect(mine.Length == ids.Length - 1 && diff < 0, line);
        // the same vocabulary through the GGUF path (FromGGUF: Ollama's gemma3 270m blob, GEMMA3_GGUF to override; skipped when absent)
        var gguf = Environment.GetEnvironmentVariable("GEMMA3_GGUF")
            ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".ollama", "models", "blobs", "sha256-735af2139dc652bf01112746474883d79a52fa1c19038265d363e3d42556f7a2");
        if (File.Exists(gguf))
        {
            await using var gs = File.OpenRead(gguf);
            var gtok = Preprocessing.SentencePieceTokenizer.FromGGUF(await GGUF.GGUFParser.ParseHeaderAsync(gs))!;
            var gids = gtok.Encode(text);
            bool same = gids.SequenceEqual(ids[1..]);
            line += $" | GGUF vocab (gemma3 270m, add_space_prefix {gtok.AddSpacePrefix}): {(same ? "identical ids" : $"{gids.Length} ids, DIFFERENT")}";
            Ur_Expect(same, line);
        }
        else line += " | GGUF cross-check skipped (no gemma3 blob)";
        Console.WriteLine("[Ltx2] Gemma tokenizer vs ComfyUI: " + line);
        await Task.CompletedTask;
        return line;
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
    /// The distilled LoRA (ltx-2-19b-distilled-lora-384, strength 1) on the LTX-2 transformer, restricted to the subset SpawnGen measured in ComfyUI (block 0 and every
    /// top-level layer: bf16 base layers, where ComfyUI's patched weight is float32 W + B A exactly; _DevComms 2026-11-18-0000): the template's two calls at cfg 1 (batch 1,
    /// the conditioned prompt) and the final latent, against ComfyUI's.
    /// </summary>
    [TestMethod(Timeout = 3600000, Category = "HeavyModel")]
    public async Task Ltx2_Lora_Bf16Subset_MatchesComfyUi() => await RunTest(async accelerator =>
    {
        var (line, worst, _) = await Ltx2_LoraSubset(accelerator, "bf16_block0_top", "ltx2_lora", n => !n.StartsWith("transformer_blocks.", StringComparison.Ordinal) || n.StartsWith("transformer_blocks.0.", StringComparison.Ordinal));
        Console.WriteLine("[Ltx2] LoRA (bf16 subset) vs ComfyUI: " + line);
        Ur_Expect(worst < 5e-5, line);   // MEASURED 2026-11-17 CUDA: calls 5.5e-6 / 5.5e-7, final latent 5.3e-6 (the LoRA's own effect 7.9e-2)
        return line;
    });

    /// <summary>
    /// The same LoRA on block 1 only (fp8 base layers): ComfyUI re-rounds W + B A into fp8 STOCHASTICALLY (seeded: byte-identical run to run, SpawnGen 11-18), which
    /// nothing deterministic reproduces; this implementation keeps W + B A exact in float. Reports the distance from ComfyUI's calls / final latent next to the LoRA's own
    /// effect (ComfyUI with the LoRA vs ComfyUI without it, same input). MEASURED 2026-11-17 CUDA: call 0 7.4e-3, call 1 7.6e-4, final 7.0e-3 against an effect of 2.2e-2:
    /// ComfyUI's fp8 re-rounding noise (~4% per weight, 3 mantissa bits) is a third of the LoRA's effect from one block. Diagnostic until parity is decided (TJ).
    /// </summary>
    [TestMethod(Timeout = 3600000, Category = "HeavyModel")]
    public async Task Ltx2_Lora_Fp8Subset_DistanceFromComfyUi() => await RunTest(async accelerator =>
    {
        var (line, worst, effect) = await Ltx2_LoraSubset(accelerator, "fp8_block1_repeat", "ltx2_lora8", n => n.StartsWith("transformer_blocks.1.", StringComparison.Ordinal));
        Console.WriteLine("[Ltx2] LoRA (fp8 subset) vs ComfyUI: " + line);
        Ur_Expect(worst < effect, line);
        return line;
    });

    // one LoRA subset against SpawnGen's taps in lora/<folder> (calls 0 / 1 at cfg 1 and the final latent); effect = ComfyUI's call 0 with the LoRA vs its call 0 without (the
    // base run's conditioned row, same input and prompt)
    private async Task<(string Line, double Worst, double Effect)> Ltx2_LoraSubset(Accelerator accelerator, string folder, string tap, Func<string, bool> include)
    {
        if (accelerator.AcceleratorType is not (AcceleratorType.Cuda or AcceleratorType.OpenCL)) throw new UnsupportedTestException("streams a 27 GB checkpoint from disk: a desktop GPU");
        var dir = Path.Combine(Ltx2_Models(), "lightricks");
        var d = Path.Combine(Ltx2_Dumps(), "lora", folder);
        if (!File.Exists(Path.Combine(dir, "ltx-2-19b-distilled-lora-384.safetensors")) || !Directory.Exists(d)) throw new UnsupportedTestException($"needs the distilled LoRA in {dir} and SpawnGen's taps in {d}");
        const int P = 128 * 8 + 8 * 9 * 16, L = 1024;
        double residentGb = double.TryParse(Environment.GetEnvironmentVariable("LTX2_DIT_RESIDENT_GB"), System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out var g) ? g : 5.0;
        float[] T(string prefix, long n) => F2_Raw(Ltx2_Tap(d, prefix), n);
        var src = new LocalFolderModelSource(dir);
        using var model = await Ltx2AvModel.OpenAsync(accelerator, src, "", "ltx-2-19b-dev-fp8.safetensors", 1, 2, 2, 2, 9, L, (long)(residentGb * 1e9));
        using var lora = await Ltx2Lora.OpenAsync(accelerator, src, "", "ltx-2-19b-distilled-lora-384.safetensors", 1f, 1L << 30, include);
        model.Transformer.AddLora(lora);
        using var ctx = accelerator.Allocate1D(T($"{tap}_call_0_c_c_crossattn", (long)L * Ltx2AvModel.ContextWidth));
        var sigmas = T($"{tap}_call_0_c_transformer_options_sample_sigmas", 3);
        var report = new List<string>(); double worst = 0;
        void Cmp(string label, float[] mine, float[] gold) { double e = F2_RelRms(mine, gold); worst = Math.Max(worst, e); report.Add($"{label} {e:E2}"); }
        var sw = System.Diagnostics.Stopwatch.StartNew();
        var out0 = T($"{tap}_call_0_output", P);
        Cmp("call 0", await model.DenoiseAsync(T($"{tap}_call_0_input", P), new[] { sigmas[0] }, ctx.View, 25f), out0);
        Cmp("call 1", await model.DenoiseAsync(T($"{tap}_call_1_input", P), new[] { sigmas[1] }, ctx.View, 25f), T($"{tap}_call_1_output", P));
        var x = await model.SampleAsync(T($"{tap}_call_0_input", P), ctx.View, sigmas, 1f, 25f);
        Cmp("final latent", x, T($"{tap}_out_samples_part0", 128L * 8).Concat(T($"{tap}_out_samples_part1", 8L * 9 * 16)).ToArray());
        double effect = double.NaN;
        var baseCall = Path.Combine(Ltx2_Dumps(), "call");
        if (Directory.Exists(baseCall))
        {
            var baseOut = F2_Raw(Ltx2_Tap(baseCall, "ltx2_tcall_0_output"), 2L * P)[P..];              // the base run's row 1: the conditioned prompt, same noise
            effect = F2_RelRms(out0, baseOut);
        }
        string line = $"{string.Join("; ", report)} | the LoRA's own effect on call 0 (ComfyUI with vs without) {effect:E2} | rank {lora.Rank}, {sw.Elapsed.TotalSeconds:F1} s";
        return (line, worst, effect);
    }

    /// <summary>
    /// LTX-2's x2 spatial latent upscaler (ltx-2-spatial-upscaler-x2-1.0, the checkpoint VAE's latent statistics) against ComfyUI's LTXVLatentUpsampler on the template run's
    /// video latent [128, 2, 2, 2] -> [128, 2, 4, 4] (SpawnGen's taps, _DevComms 2026-11-17-2300). Reports the result with and without AdaIN; the closer one must match.
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task Ltx2_LatentUpsample_MatchesComfyUi() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is not (AcceleratorType.Cuda or AcceleratorType.OpenCL)) throw new UnsupportedTestException("reads model files from disk: a desktop GPU");
        var dir = Path.Combine(Ltx2_Models(), "lightricks");
        var up = Path.Combine(Ltx2_Dumps(), "upenc");
        if (!File.Exists(Path.Combine(dir, "ltx-2-spatial-upscaler-x2-1.0.safetensors")) || !Directory.Exists(up)) throw new UnsupportedTestException($"needs the upscaler in {dir} and SpawnGen's taps in {up}");
        using var u = await Ltx2LatentUpsampler.OpenAsync(accelerator, new LocalFolderModelSource(dir), "", "ltx-2-spatial-upscaler-x2-1.0.safetensors", "ltx-2-19b-dev-fp8.safetensors");
        var z = F2_Raw(Ltx2_Tap(up, "ltx2_up_in_samples"), 128L * 2 * 2 * 2);
        var gold = F2_Raw(Ltx2_Tap(up, "ltx2_up_out_samples"), 128L * 2 * 4 * 4);
        var sw = System.Diagnostics.Stopwatch.StartNew();
        double e0 = F2_RelRms(await u.UpsampleAsync(z, 2, 2, 2, 0f), gold);
        double run = sw.Elapsed.TotalSeconds;
        double e1 = F2_RelRms(await u.UpsampleAsync(z, 2, 2, 2, 1f), gold);
        string line = $"[128, 2, 4, 4]: without AdaIN {e0:E2}, with AdaIN {e1:E2} | {run:F2} s";
        Console.WriteLine("[Ltx2] latent upsample vs ComfyUI: " + line);
        Ur_Expect(e0 < 3e-5, line);   // MEASURED 2026-11-17 CUDA: ComfyUI's LTXVLatentUpsampler applies NO AdaIN: without 4.8e-6, with 0.44
        return line;
    });

    /// <summary>
    /// LTX-2's video VAE encoder (the checkpoint's BF16 vae.encoder) on a 192 x 256 picture (ComfyUI's IMAGE, NHWC in [0, 1]) against ComfyUI's VAEEncode latent [128, 1, 6, 8]
    /// (SpawnGen's taps, _DevComms 2026-11-17-2300): the posterior mean, normalised with the checkpoint's latent statistics.
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task Ltx2_VideoEncode_MatchesComfyUi() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is not (AcceleratorType.Cuda or AcceleratorType.OpenCL)) throw new UnsupportedTestException("reads a 27 GB checkpoint from disk: a desktop GPU");
        var path = Path.Combine(Ltx2_Models(), "lightricks", "ltx-2-19b-dev-fp8.safetensors");
        var up = Path.Combine(Ltx2_Dumps(), "upenc");
        if (!File.Exists(path) || !Directory.Exists(up)) throw new UnsupportedTestException($"needs {path} and SpawnGen's taps in {up}");
        const int Hp = 192, Wp = 256;
        var img = F2_Raw(Ltx2_Tap(up, "venc_input_image"), (long)Hp * Wp * 3);                  // [H, W, 3] in [0, 1]
        var video = new float[3L * Hp * Wp];
        for (int c = 0; c < 3; c++) for (int y = 0; y < Hp; y++) for (int x = 0; x < Wp; x++) video[((long)c * Hp + y) * Wp + x] = img[((long)y * Wp + x) * 3 + c] * 2f - 1f;
        using var enc = await Ltx2VideoEncoder.OpenAsync(accelerator, new LocalFolderModelSource(Path.GetDirectoryName(path)!), "", Path.GetFileName(path));
        var sw = System.Diagnostics.Stopwatch.StartNew();
        var z = await enc.EncodeAsync(video, 1, Hp, Wp);
        double run = sw.Elapsed.TotalSeconds;
        double e = F2_RelRms(z, F2_Raw(Ltx2_Tap(up, "ltx2_venc_samples"), 128L * 6 * 8));
        string line = $"latent [128, 1, 6, 8] {e:E2} | encode {run:F2} s";
        Console.WriteLine("[Ltx2] video encode vs ComfyUI: " + line);
        Ur_Expect(e < 1e-4, line);
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
            Ur_Expect(worst < 6e-5, line);   // MEASURED 2026-11-17 CUDA (tensor cores, rounded accumulation): embedders 8e-7, blocks 1.5e-6..8.3e-6, outputs 1.2e-5 / 3.4e-6 (q/k eps 1e-5, v2a after a2v)
            return line;
        }
        finally { await accelerator.SynchronizeAsync(); foreach (var b in probes.Values) b.Dispose(); }
    });

    /// <summary>
    /// The template's whole sampling run through <see cref="Ltx2AvModel"/> (ComfyUI's model-call form: packed latents, sigma, context [2, 1024, 7680], 25 fps): from ComfyUI's
    /// noise and conditioning, CFG 4, Euler over sigmas [1, 0.1, 0], against ComfyUI's call-0 output (denoised, both rows), the latent entering call 1, and the final latents it decoded.
    /// </summary>
    [TestMethod(Timeout = 3600000, Category = "HeavyModel")]
    public async Task Ltx2_Sample_MatchesComfyUi() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is not (AcceleratorType.Cuda or AcceleratorType.OpenCL)) throw new UnsupportedTestException("streams a 27 GB checkpoint from disk: a desktop GPU");
        var path = Path.Combine(Ltx2_Models(), "lightricks", "ltx-2-19b-dev-fp8.safetensors");
        string call = Path.Combine(Ltx2_Dumps(), "call"), dec = Path.Combine(Ltx2_Dumps(), "dec");
        if (!File.Exists(path)) throw new UnsupportedTestException($"{path} not found");
        if (!Directory.Exists(call) || !Directory.Exists(dec)) throw new UnsupportedTestException($"SpawnGen's LTX-2 call / decode taps not found in {Ltx2_Dumps()}");
        const int P = 128 * 8 + 8 * 9 * 16, L = 1024;
        double residentGb = double.TryParse(Environment.GetEnvironmentVariable("LTX2_DIT_RESIDENT_GB"), System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out var g) ? g : 6.0;
        float[] T(string dir, string prefix, long n) => F2_Raw(Ltx2_Tap(dir, prefix), n);
        var in0 = T(call, "ltx2_tcall_0_input", 2L * P); var out0 = T(call, "ltx2_tcall_0_output", 2L * P); var in1 = T(call, "ltx2_tcall_1_input", 2L * P);
        var sigmas = T(call, "ltx2_tcall_0_c_transformer_options_sample_sigmas", 3);
        var final = T(dec, "ltx2_dec_video_latent_samples", 128L * 8).Concat(T(dec, "ltx2_dec_audio_latent_samples", 8L * 9 * 16)).ToArray();
        using var ctx = accelerator.Allocate1D(T(call, "ltx2_tcall_0_c_c_crossattn", 2L * L * Ltx2AvModel.ContextWidth));   // [uncond, cond]
        var sw = System.Diagnostics.Stopwatch.StartNew();
        using var model = await Ltx2AvModel.OpenAsync(accelerator, new LocalFolderModelSource(Path.GetDirectoryName(path)!), "", Path.GetFileName(path), 2, 2, 2, 2, 9, L, (long)(residentGb * 1e9));
        double open = sw.Elapsed.TotalSeconds;
        var report = new List<string>(); double worst = 0;
        void Cmp(string label, float[] mine, float[] gold) { double e = F2_RelRms(mine, gold); worst = Math.Max(worst, e); report.Add($"{label} {e:E2}"); }
        sw.Restart();
        Cmp("call 0 denoised [2, 2176]", await model.DenoiseAsync(in0, new[] { sigmas[0], sigmas[0] }, ctx.View, 25f), out0);
        double call0 = sw.Elapsed.TotalSeconds;
        float[]? afterStep0 = null;
        sw.Restart();
        var x = await model.SampleAsync(in0[P..], ctx.View, sigmas, 4f, 25f, (i, xi) => { if (i == 0) afterStep0 = (float[])xi.Clone(); });
        double sample = sw.Elapsed.TotalSeconds;
        Cmp("after step 0 (call 1's input)", afterStep0!, in1[P..]);
        Cmp("final latent [2176]", x, final);
        string line = $"{string.Join("; ", report)} | sigmas [{string.Join(", ", sigmas)}] | open {open:F1} s, one call {call0:F1} s, sampling {sample:F1} s";
        Console.WriteLine("[Ltx2] sampling vs ComfyUI: " + line);
        Ur_Expect(worst < 5e-4, line);   // MEASURED 2026-11-17 CUDA: call 0 4.6e-5, after step 0 1.1e-4, final 1.1e-4 (CFG 4 amplifies the call's error)
        return line;
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
            Ur_Expect(worst < 2e-5, line);   // MEASURED 2026-11-17 CUDA: projection 2.2e-6, conditioning 1.9e-6 (tensor cores with rounded accumulation; truncating accumulation gave 4.6e-4)
            return line;
        }
        finally { await accelerator.SynchronizeAsync(); foreach (var b in probes.Values) b.Dispose(); }
    });
}
