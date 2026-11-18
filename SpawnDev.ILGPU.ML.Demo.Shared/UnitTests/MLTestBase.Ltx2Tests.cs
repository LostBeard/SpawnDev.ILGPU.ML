using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.Ltx2;
using SpawnDev.UnitTesting;
using System.Runtime.Intrinsics;

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
    /// <see cref="Ltx2Schedule.Sigmas"/> against ComfyUI's own LTXVScheduler outputs (SpawnGen's dumps, _DevComms 2026-11-18-0800; values inline so the test runs
    /// everywhere): a video latent of 768 x 512 x 97 (4992 tokens) at 20 steps, no latent (4096 tokens) at 8 steps, an audio latent (1552 tokens) at 20 steps, stretch to
    /// terminal 0.1. Host arithmetic only.
    /// </summary>
    [TestMethod(Timeout = 30000)]
    public async Task Ltx2_Schedule_MatchesComfyUi() => await RunTest(accelerator =>
    {
        var cases = new (string Name, int Steps, int Tokens, float[] Gold)[]
        {
            ("768x512x97, 20 steps", 20, 13 * 16 * 24, new[] { 1.0f, 0.993116557598114f, 0.9855465888977051f, 0.9771823883056641f, 0.9678919911384583f, 0.9575125575065613f, 0.9458404183387756f,
                0.9326183795928955f, 0.9175154566764832f, 0.9000999927520752f, 0.8797963857650757f, 0.8558213710784912f, 0.8270803093910217f, 0.7919948101043701f, 0.7482041120529175f,
                0.6920087337493896f, 0.6172688007354736f, 0.5129902958869934f, 0.3573495149612427f, 0.10000002384185791f, 0.0f }),
            ("no latent, 8 steps", 8, 4096, new[] { 0.9999998807907104f, 0.9657117128372192f, 0.9218747615814209f, 0.8638560771942139f, 0.7834454774856567f, 0.6645792722702026f, 0.4710026979446411f,
                0.10000002384185791f, 0.0f }),
            ("audio 97 x 16, 20 steps", 20, 97 * 16, new[] { 1.0f, 0.9826359748840332f, 0.964005172252655f, 0.9439640045166016f, 0.9223458170890808f, 0.8989567756652832f, 0.8735702037811279f,
                0.845918595790863f, 0.8156849145889282f, 0.7824894189834595f, 0.7458747625350952f, 0.7052839398384094f, 0.6600320339202881f, 0.6092672944068909f, 0.5519169569015503f,
                0.4866105914115906f, 0.4115692973136902f, 0.324441134929657f, 0.22204995155334473f, 0.10000002384185791f, 0.0f }),
        };
        var report = new List<string>(); double worst = 0;
        foreach (var (name, steps, tokens, gold) in cases)
        {
            var s = Ltx2Schedule.Sigmas(steps, tokens);
            Ur_Expect(s.Length == gold.Length, $"{name}: {s.Length} sigmas, ComfyUI {gold.Length}");
            double e = 0; for (int i = 0; i < s.Length; i++) e = Math.Max(e, Math.Abs(s[i] - gold[i]));
            worst = Math.Max(worst, e); report.Add($"{name} max |diff| {e:E1}");
        }
        var one = Ltx2Schedule.Sigmas(1, 8);
        string line = string.Join("; ", report) + $"; one step [{one[0]}, {one[1]}]";
        Ur_Expect(worst < 5e-7 && one.Length == 2 && Math.Abs(one[0] - 0.1f) < 1e-7 && one[1] == 0f, line);   // MEASURED (numpy, the same float32 arithmetic): these three 1.2e-7; 8.9e-7 worst over 37 ComfyUI schedules
        // the schedule must depend on the token count (a constant-shift bug would still pass one case)
        Ur_Expect(Math.Abs(Ltx2Schedule.Sigmas(20, 8)[10] - Ltx2Schedule.Sigmas(20, 4992)[10]) > 0.1, "the shift does not move with the token count");
        Console.WriteLine("[Ltx2] schedule vs ComfyUI: " + line);
        return Task.FromResult(line);
    });

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
        // the same convolution over more rows than one row tile (T 9: 180 rows), whole and split into one-tile dispatches (what a real clip's decode does so no dispatch
        // runs for seconds): bit for bit the same
        {
            const int T2 = 9, M2 = T2 * H * W;
            var x2 = new float[M2 * cin]; for (int i = 0; i < x2.Length; i++) x2[i] = (float)(rng.NextDouble() * 2 - 1);
            using var xb2 = accelerator.Allocate1D(x2); using var nbr2 = accelerator.Allocate1D<int>(M2 * 27);
            using var whole = accelerator.Allocate1D<float>(M2 * cout); using var split = accelerator.Allocate1D<float>(M2 * cout);
            l.ReflectTable(nbr2.View, T2, H, W);
            sk.ConvTapsBf16Exact(xb2.View, nbr2.View, g.View, bias.View, whole.View, M2, cin, cout, 27);
            var a = await whole.View.CopyToHostAsync();
            long was = Trellis2.SparseKernels.MaxConvWorkPerDispatch;
            Trellis2.SparseKernels.MaxConvWorkPerDispatch = 1;
            float[] b2;
            try { sk.ConvTapsBf16Exact(xb2.View, nbr2.View, g.View, bias.View, split.View, M2, cin, cout, 27); b2 = await split.View.CopyToHostAsync(); }
            finally { Trellis2.SparseKernels.MaxConvWorkPerDispatch = was; }
            double moved = 0;
            for (int i = 0; i < a.Length; i++) { if (a[i] != b2[i]) throw new Exception($"the split convolution differs at {i} (row {i / cout}): {b2[i]} vs {a[i]} on {accelerator.AcceleratorType}"); moved += Math.Abs(a[i]); }
            Ur_Expect(moved / a.Length > 0.1, $"the {M2}-row convolution's output is near zero ({moved / a.Length:E2}): nothing was compared");
            line += $"; {M2} rows split into the smallest dispatches (a row tile on the tiled GEMM, a row on the plain kernel) == whole";
            Console.WriteLine("[Ltx2] conv split: " + line);
        }
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

    /// <summary>
    /// The same block-1 subset in the ComfyUI-compatible mode (<see cref="Ltx2Transformer.ComfyCompatibleFp8Lora"/>): the patched 8-bit weights are re-rounded by
    /// ComfyUI's seeded rule (<see cref="Ltx2Fp8Rounding"/>), so the calls and the final latent EQUAL ComfyUI's instead of sitting a third of the LoRA's effect away.
    /// </summary>
    [TestMethod(Timeout = 3600000, Category = "HeavyModel")]
    public async Task Ltx2_Lora_Fp8Subset_CompatMode_MatchesComfyUi() => await RunTest(async accelerator =>
    {
        var (line, worst, _) = await Ltx2_LoraSubset(accelerator, "fp8_block1_repeat", "ltx2_lora8", n => n.StartsWith("transformer_blocks.1.", StringComparison.Ordinal), compat: true);
        Console.WriteLine("[Ltx2] LoRA (fp8 subset, ComfyUI-compatible rounding) vs ComfyUI: " + line);
        Ur_Expect(worst < 1e-4, line);
        return line;
    });

    /// <summary>
    /// The parts of the ComfyUI-compatible fp8 LoRA mode (<see cref="Ltx2Fp8Rounding"/>). Host: the stream's seed and draws against values produced OUTSIDE this code
    /// (zlib's crc32 of the key; numpy's MT19937 seeded with it, the low 8 bits of words 0..7, 622..627 and 1246..1251: either side of the generator's first two state
    /// refills). Device: the whole chain (W x scale + two scaled LoRA products, max / 448, the seeded rounding) against the host rule on every element, bitwise. The
    /// matrix carries what a device can get wrong: quotients one and two float32 steps either side of half-precision rounding boundaries (normal and subnormal halves)
    /// and of F8_E4M3 cell edges, zeros of both signs, tiny values; a second pass at a power-of-two scale makes the ties exact. [16384, 4096] (the largest patched
    /// matrix) on CUDA / OpenCL, [1024, 4096] elsewhere.
    /// </summary>
    [TestMethod(Timeout = 900000)]
    public async Task Ltx2_Fp8CompatRounding_MatchesHostRule() => await RunTest(async accelerator =>
    {
        const string name = "transformer_blocks.1.attn1.to_q";
        uint seed = Ltx2Fp8Rounding.Seed(name);
        Ur_Expect(seed == 1881967891u, $"crc32 of {Ltx2Fp8Rounding.Key(name)} is {seed}, zlib's 1881967891");
        static int Draw(uint[] p, long i) => (int)((p[i >> 2] >> (int)((i & 3) << 3)) & 0xFFu);
        var head = new uint[313]; Ltx2Fp8Rounding.Low8(seed, head, 1252);
        void Draws(int at, int[] gold) { for (int i = 0; i < gold.Length; i++) Ur_Expect(Draw(head, at + i) == gold[i], $"draw {at + i} is {Draw(head, at + i)}, numpy's {gold[i]}"); }
        Draws(0, new[] { 200, 214, 181, 81, 235, 10, 189, 229 }); Draws(622, new[] { 181, 194, 47, 136, 106, 114 }); Draws(1246, new[] { 98, 92, 34, 74, 108, 139 });

        bool big = accelerator.AcceleratorType is AcceleratorType.Cuda or AcceleratorType.OpenCL;
        int N = big ? 16384 : 1024; const int K = 4096; int cnt = N * K, words = cnt / 4, C = 1 << 20;
        var l = new Ltx2Kernels(accelerator);
        using var decB = accelerator.Allocate1D<BFloat16>(cnt); using var wB = accelerator.Allocate1D<float>(cnt); using var ba1B = accelerator.Allocate1D<float>(cnt);
        using var ba2B = accelerator.Allocate1D<float>(cnt); using var rndB = accelerator.Allocate1D<uint>(words); using var rowMaxB = accelerator.Allocate1D<float>(N);
        using var scaleB = accelerator.Allocate1D<float>(1); using var qB = accelerator.Allocate1D<BFloat16>(cnt);
        var dec = new ushort[cnt]; var ba1 = new float[cnt]; var ba2 = new float[cnt]; var rnd = new uint[words]; var want = new float[cnt];
        Ltx2Fp8Rounding.Low8(seed, rnd, cnt);
        static float Bf(ushort u) => BitConverter.Int32BitsToSingle(u << 16);
        static float Steps(float x, int n) => BitConverter.Int32BitsToSingle(Math.Max(0, BitConverter.SingleToInt32Bits(x) + n));   // x >= 0 moved n float32 steps
        string[] kinds = { "midpoint - 2", "midpoint - 1", "midpoint", "midpoint + 1", "midpoint + 2", "half value", "cell edge", "special", "random" };
        var report = new List<string>(); long worstBad = 0;
        foreach (float wmax in new[] { 0.37f, 0.4375f })   // 0.4375 = 448 x 2^-10: that pass's scale is a power of two, every quotient exact, the ties real
        {
            float s = wmax / 448f, ws = wmax * 0.9f / 448f;
            var rng = new Random(11);
            Array.Clear(ba2);
            // the body: random F8_E4M3 values (subnormal ones too) x a scale + 0.7 x a random product
            for (int i = C; i < cnt; i++)
            {
                int m = rng.Next(8), e = rng.Next(-9, 9);
                if (e == 8 && m == 7) m = 6;                                                                    // 448 is the largest value
                float v = e < -6 ? m * 0.001953125f : (1f + m * 0.125f) * MathF.ScaleB(1f, e);
                if ((rng.Next() & 1) != 0) v = -v;
                dec[i] = (ushort)(BitConverter.SingleToUInt32Bits(v) >> 16);
                ba1[i] = (float)((rng.NextDouble() * 2 - 1) * 0.05 * wmax);
            }
            // the head: W = 0 and the first product 0, so the patched weight IS the second product: element 0 the maximum, then the crafted cases
            Array.Clear(dec, 0, C); Array.Clear(ba1, 0, C);
            ba2[0] = wmax;
            float[] specials = { 0f, -0f, float.Epsilon, 1e-30f, (float)(Math.ScaleB(1.0, -25) * s), Steps((float)(Math.ScaleB(1.0, -25) * s), 1), (float)(Math.ScaleB(1.0, -24) * s), (float)(Math.ScaleB(3.0, -25) * s) };
            for (int j = 1; j < C; j++)
            {
                int kind = j % 8; float v;
                // a half-precision value hv below 448 and the half grid's step there (subnormal halves below 2^-14: multiples of 2^-24)
                int e = rng.Next(-18, 9); double hv, step;
                if (e < -14) { step = Math.ScaleB(1.0, -24); hv = rng.Next(1, 1024) * step; }
                else { step = Math.ScaleB(1.0, e - 10); hv = (1024 + rng.Next(e == 8 ? 767 : 1024)) * step; }
                if (kind < 5) v = Steps((float)((hv + 0.5 * step) * s), kind - 2);
                else if (kind == 5) v = Steps((float)(hv * s), rng.Next(-1, 2));
                else if (kind == 6) { int e2 = rng.Next(-6, 9), m2 = rng.Next(e2 == 8 ? 6 : 8); v = Steps((float)((1.0 + m2 * 0.125) * Math.ScaleB(1.0, e2) * s), rng.Next(-1, 2)); }
                else v = specials[(j >> 3) % specials.Length];
                ba2[j] = (rng.Next() & 1) != 0 ? -v : v;
            }
            float max = 0f;
            for (int i = 0; i < cnt; i++)
            {
                float w0 = Bf(dec[i]) * ws; float t = 0.7f * ba1[i]; w0 += t; w0 += ba2[i];
                max = MathF.Max(max, MathF.Abs(w0));
                want[i] = Ltx2Fp8Rounding.Round(w0, s, Draw(rnd, i));
            }
            Ur_Expect(max == wmax, $"the matrix's maximum is {max:R}, built for {wmax:R}");
            decB.View.Cast<BFloat16, ushort>().CopyFromCPU(dec); ba1B.CopyFromCPU(ba1); ba2B.CopyFromCPU(ba2); rndB.CopyFromCPU(rnd);
            l.PatchedWeightInit(decB.View, wB.View, ws, cnt);
            l.AddScaled(wB.View, ba1B.View, 0.7f, cnt);
            l.AddScaled(wB.View, ba2B.View, 1f, cnt);
            l.Fp8Scale(wB.View, rowMaxB.View, scaleB.View, N, K);
            l.Fp8StochasticRound(wB.View, rndB.View, scaleB.View, qB.View, cnt);
            l.Bf16ToF32(qB.View, ba1B.View, cnt);                                                              // the rounded matrix as float32 for the readback
            var sc = await scaleB.View.CopyToHostAsync();
            var got = await ba1B.View.CopyToHostAsync();
            long bad = 0, zeroSign = 0, up = 0; var byKind = new long[9]; string first = "";
            for (int i = 0; i < cnt; i++)
            {
                if (MathF.Abs(want[i]) * s > MathF.Abs(Bf(dec[i]) * ws + 0.7f * ba1[i] + ba2[i])) up++;
                if (BitConverter.SingleToInt32Bits(got[i]) == BitConverter.SingleToInt32Bits(want[i])) continue;
                if (got[i] == 0f && want[i] == 0f) { zeroSign++; continue; }                                    // +0 against -0: the same weight
                int kind = i == 0 || i >= C ? 8 : i % 8;
                byKind[kind]++;
                if (bad++ == 0) first = $"; first at {i} ({kinds[kind]}): device {got[i]:R}, host {want[i]:R}, draw {Draw(rnd, i)}";
            }
            string by = string.Join(", ", Enumerable.Range(0, 9).Where(k => byKind[k] > 0).Select(k => $"{kinds[k]} {byKind[k]}"));
            report.Add($"max {wmax}: scale device {sc[0]:R} host {s:R}, {bad} of {cnt} differ{(bad > 0 ? " (" + by + ")" + first : "")}, {zeroSign} zeros of the other sign, {100.0 * up / cnt:F1}% rounded up");
            worstBad = Math.Max(worstBad, bad);
            Ur_Expect(sc[0] == s, report[^1]);
            Ur_Expect(up > cnt / 4 && up < cnt - cnt / 4, "the rounding does not go both ways: " + report[^1]);
        }
        string line = $"[{N}, {K}] on {accelerator.AcceleratorType}: " + string.Join(" | ", report);
        Console.WriteLine("[Ltx2] fp8 compat rounding vs the host rule: " + line);
        Ur_Expect(worstBad == 0, line);
        return line;
    });

    /// <summary>
    /// The compatible mode's random stream at speed: the vectorized <see cref="Ltx2Fp8Rounding.Low8"/> against the plain <see cref="Ltx2Fp8Rounding.Low8Scalar"/>, word
    /// for word, at counts either side of the generator's 624-word refills and of the 16-word output step, then the largest matrix's stream timed both ways. And the
    /// contract the mode's pooled host arrays lean on: an upload has TAKEN the words when CopyFromCPU returns (the host array is overwritten right after each upload,
    /// the device must still hold what was uploaded).
    /// </summary>
    [TestMethod(Timeout = 600000)]
    public async Task Ltx2_Fp8CompatStream_MatchesPlainFormAndUploads() => await RunTest(async accelerator =>
    {
        bool big = accelerator.AcceleratorType is AcceleratorType.Cuda or AcceleratorType.OpenCL;
        var report = new List<string>();
        // what this runtime's Vector128.Narrow does with values above the narrow type's range (reported, not asserted: Low8 masks before it narrows)
        var narIn = new uint[8]; for (int i = 0; i < 8; i++) narIn[i] = i < 4 ? 0x12345678u + (uint)report.Count : 0x9ABCDEF0u;   // run-time values: nothing to fold
        var nar = System.Runtime.Intrinsics.Vector128.Narrow(System.Runtime.Intrinsics.Vector128.Create(narIn, 0), System.Runtime.Intrinsics.Vector128.Create(narIn, 4));
        var narIn16 = new ushort[16]; for (int i = 0; i < 16; i++) narIn16[i] = (ushort)((i < 8 ? 0x12ACu : 0xFE37u) + (uint)report.Count);
        var nar8 = System.Runtime.Intrinsics.Vector128.Narrow(System.Runtime.Intrinsics.Vector128.Create(narIn16, 0), System.Runtime.Intrinsics.Vector128.Create(narIn16, 8));
        report.Add($"runtime: Vector128 accelerated {System.Runtime.Intrinsics.Vector128.IsHardwareAccelerated}, Narrow 32 to 16 bits (12345678, 9ABCDEF0) = {nar[0]:X4}, {nar[4]:X4} ({(nar[0] == 0x5678 && nar[4] == 0xDEF0 ? "truncates" : "DOES NOT truncate")}), "
            + $"16 to 8 bits (12AC, FE37) = {nar8[0]:X2}, {nar8[8]:X2} ({(nar8[0] == 0xAC && nar8[8] == 0x37 ? "truncates" : "DOES NOT truncate")})");
        // the form Low8 had before it masked: two 256-bit vectors, each narrowed half by half to 16 bits, the pair narrowed to bytes
        var n256a = System.Runtime.Intrinsics.Vector256.Create(narIn, 0); var n256b = System.Runtime.Intrinsics.Vector256.Create(0x0BADF00Du + (uint)report.Count - 1u);
        var nested = System.Runtime.Intrinsics.Vector128.Narrow(System.Runtime.Intrinsics.Vector128.Narrow(System.Runtime.Intrinsics.Vector256.GetLower(n256a), System.Runtime.Intrinsics.Vector256.GetUpper(n256a)), System.Runtime.Intrinsics.Vector128.Narrow(System.Runtime.Intrinsics.Vector256.GetLower(n256b), System.Runtime.Intrinsics.Vector256.GetUpper(n256b)));
        report.Add($"nested 256-bit narrow to bytes (12345678 x 4, 9ABCDEF0 x 4, 0BADF00D x 8) = {nested[0]:X2}, {nested[4]:X2}, {nested[8]:X2} ({(nested[0] == 0x78 && nested[4] == 0xF0 && nested[8] == 0x0D ? "truncates" : "DOES NOT truncate")})");
        // the generator's output mix exactly as Low8 runs it but WITHOUT the mask before the narrowing, against the plain mix (reported, not asserted)
        {
            // run HOT (300,000 rounds): the browser runtime compiles hot loops separately, and the library's loop was hot when it returned wrong bytes
            var mixIn = new uint[16];
            static uint Mix(uint v) { v ^= v >> 11; v ^= (v << 7) & 0x9D2C5680u; v ^= (v << 15) & 0xEFC60000u; v ^= v >> 18; return v; }
            var mb = System.Runtime.Intrinsics.Vector256.Create(0x9D2C5680u); var mc = System.Runtime.Intrinsics.Vector256.Create(0xEFC60000u);
            long mixBad = 0; int firstBadRound = -1; byte gotByte = 0, wantByte = 0; var mixOut = new byte[16];
            for (int round = 0; round < 300000; round++)
            {
                for (int i = 0; i < 16; i++) mixIn[i] = 2567483615u * (uint)(i + 1 + round) + 12345u;
                var m0 = System.Runtime.Intrinsics.Vector256.Create(mixIn, 0); var m1 = System.Runtime.Intrinsics.Vector256.Create(mixIn, 8);
                m0 ^= m0 >>> 11; m0 ^= (m0 << 7) & mb; m0 ^= (m0 << 15) & mc; m0 ^= m0 >>> 18;
                m1 ^= m1 >>> 11; m1 ^= (m1 << 7) & mb; m1 ^= (m1 << 15) & mc; m1 ^= m1 >>> 18;
                System.Runtime.Intrinsics.Vector128.Narrow(
                    System.Runtime.Intrinsics.Vector128.Narrow(System.Runtime.Intrinsics.Vector256.GetLower(m0), System.Runtime.Intrinsics.Vector256.GetUpper(m0)),
                    System.Runtime.Intrinsics.Vector128.Narrow(System.Runtime.Intrinsics.Vector256.GetLower(m1), System.Runtime.Intrinsics.Vector256.GetUpper(m1))).CopyTo(mixOut);
                for (int i = 0; i < 16; i++) if (mixOut[i] != (byte)Mix(mixIn[i])) { if (mixBad++ == 0) { firstBadRound = round; gotByte = mixOut[i]; wantByte = (byte)Mix(mixIn[i]); } }
            }
            report.Add($"unmasked output mix, 300000 hot rounds of 16 words: {(mixBad == 0 ? "equals the plain mix" : $"{mixBad} bytes WRONG from round {firstBadRound} on (first {gotByte:X2}, plain {wantByte:X2})")}");
        }
        foreach (int count in new[] { 1, 3, 4, 15, 16, 17, 623, 624, 625, 640, 1247, 1248, 1249, 2000003 })
            foreach (uint seed in new[] { 0u, 5489u, 1881967891u, 0xFFFFFFFFu })
            {
                int words = (count + 3) / 4;
                var a = new uint[words + 2]; var b = new uint[words + 2];
                a[words] = b[words] = 0xDEADBEEFu;                                                             // a guard word: neither form may write past its words
                Ltx2Fp8Rounding.Low8(seed, a, count); Ltx2Fp8Rounding.Low8Scalar(seed, b, count);
                for (int i = 0; i <= words; i++) Ur_Expect(a[i] == b[i], $"seed {seed}, {count} draws: word {i} is {a[i]:X8}, the plain form's {b[i]:X8}");
            }
        int n = big ? 16384 * 4096 : 1024 * 4096, nw = n / 4;
        var fast = new uint[nw]; var plain = new uint[nw];
        double tf = double.MaxValue, tp = double.MaxValue;
        for (int rep = 0; rep < 3; rep++)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew(); Ltx2Fp8Rounding.Low8(77u + (uint)rep, fast, n); tf = Math.Min(tf, sw.Elapsed.TotalSeconds);
            sw.Restart(); Ltx2Fp8Rounding.Low8Scalar(77u + (uint)rep, plain, n); tp = Math.Min(tp, sw.Elapsed.TotalSeconds);
            for (int i = 0; i < nw; i++) if (fast[i] != plain[i]) throw new Exception($"seed {77 + rep}, {n} draws: word {i} is {fast[i]:X8}, the plain form's {plain[i]:X8}");
        }
        report.Add($"{n} draws: vectorized {tf * 1e9 / n:F2} ns a draw ({tf * 1e3:F0} ms), plain {tp * 1e9 / n:F2} ns ({tp * 1e3:F0} ms), {tp / tf:F1}x");

        // the upload contract: three rounds, the host array overwritten the moment CopyFromCPU returns
        using var dev = accelerator.Allocate1D<uint>(nw);
        for (int round = 0; round < 3; round++)
        {
            Ltx2Fp8Rounding.Low8(1000u + (uint)round, fast, n);
            dev.CopyFromCPU(fast);
            Array.Fill(fast, 0xA5A5A5A5u);
            var back = await dev.View.CopyToHostAsync();
            Ltx2Fp8Rounding.Low8(1000u + (uint)round, plain, n);
            long wrong = 0; int firstWrong = -1;
            for (int i = 0; i < nw; i++) if (back[i] != plain[i]) { if (wrong++ == 0) firstWrong = i; }
            Ur_Expect(wrong == 0, $"upload round {round} on {accelerator.AcceleratorType}: {wrong} of {nw} words on the device are not what was uploaded (first at {firstWrong}: {back[Math.Max(0, firstWrong)]:X8}): CopyFromCPU returned before it had the data");
        }
        report.Add($"3 uploads of {nw} words each survived the host array's overwrite");
        string line = $"{accelerator.AcceleratorType}: " + string.Join("; ", report);
        Console.WriteLine("[Benchmark] Ltx2 fp8 compat stream: " + line);
        if (big && System.Runtime.Intrinsics.Vector128.IsHardwareAccelerated) Ur_Expect(tp / tf > 1.5, "the vectorized stream is not faster than the plain one: " + line);
        return line;
    });

    /// <summary>
    /// A LoRA that covers only some blocks (here block 1 alone, cut from the distilled LoRA: test-lora-block1-only.safetensors next to it) opens: the anchor is any lora_A
    /// tensor, the rank comes from the first covered block, and only that block carries tensors (SpawnGen's request, 11-17).
    /// </summary>
    [TestMethod(Timeout = 600000, Category = "HeavyModel")]
    public async Task Ltx2_Lora_OpensWithoutBlock0() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is not (AcceleratorType.Cuda or AcceleratorType.OpenCL)) throw new UnsupportedTestException("reads a LoRA file from disk: a desktop GPU");
        var dir = Path.Combine(Ltx2_Models(), "lightricks");
        if (!File.Exists(Path.Combine(dir, "test-lora-block1-only.safetensors"))) throw new UnsupportedTestException($"needs test-lora-block1-only.safetensors in {dir}");
        using var lora = await Ltx2Lora.OpenAsync(accelerator, new LocalFolderModelSource(dir), "", "test-lora-block1-only.safetensors", 1f, 0);
        var blocks = lora.Weights.HeaderNames.Select(n => n.Split('.')[1]).Distinct().ToArray();
        string line = $"rank {lora.Rank}, blocks [{string.Join(", ", blocks)}], {lora.Weights.HeaderNames.Count} tensors";
        Console.WriteLine("[Ltx2] LoRA without block 0: " + line);
        Ur_Expect(lora.Rank == 384 && blocks.SequenceEqual(new[] { "1" }) && lora.Weights.HeaderNames.Count == 56, line);
        return line;
    });

    // one LoRA subset against SpawnGen's taps in lora/<folder> (calls 0 / 1 at cfg 1 and the final latent); effect = ComfyUI's call 0 with the LoRA vs its call 0 without (the
    // base run's conditioned row, same input and prompt)
    private async Task<(string Line, double Worst, double Effect)> Ltx2_LoraSubset(Accelerator accelerator, string folder, string tap, Func<string, bool> include, bool compat = false)
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
        model.Transformer.ComfyCompatibleFp8Lora = compat;
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
        string line = $"{string.Join("; ", report)} | the LoRA's own effect on call 0 (ComfyUI with vs without) {effect:E2} | rank {lora.Rank}, {sw.Elapsed.TotalSeconds:F1} s{(compat ? $", ComfyUI-compatible fp8 rounding: {model.Transformer.CompatStreamSeconds:F2} s of host stream in the last call, {model.Transformer.CompatHostBytes / 1048576.0:F0} MiB of host memory" : "")}";
        return (line, worst, effect);
    }

    /// <summary>
    /// Keyframe-guided LTX-2 calls (ComfyUI's LTXVAddGuide, 128 x 96 x 9 + audio, cfg 4; SpawnGen's taps _DevComms 2026-11-18-0600 and -1000): a picture guide at the last
    /// frame at strengths 1.0 and 0.6, the same guide with a pixel attention mask (left half kept, right half blocked), and two chained guides (frame 0 at 0.6, the last
    /// frame at 0.3). Per-frame timesteps from denoise_mask, the guide tokens' RoPE boxes from keyframe_idxs, ln(strength x mask) per guide token on the noisy / guide
    /// self-attention. Against ComfyUI's call-0 output (denoised, both rows).
    /// </summary>
    [TestMethod(Timeout = 3600000, Category = "HeavyModel")]
    public async Task Ltx2_GuidedCall_MatchesComfyUi() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is not (AcceleratorType.Cuda or AcceleratorType.OpenCL)) throw new UnsupportedTestException("streams a 27 GB checkpoint from disk: a desktop GPU");
        var dir = Path.Combine(Ltx2_Models(), "lightricks");
        var root = Path.Combine(Ltx2_Dumps(), "guided");
        if (!Directory.Exists(root)) throw new UnsupportedTestException($"SpawnGen's guided-call taps not found in {root}");
        const int B = 2, H = 3, W = 4, Ta = 9, L = 1024, hw = H * W;
        double residentGb = double.TryParse(Environment.GetEnvironmentVariable("LTX2_DIT_RESIDENT_GB"), System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out var gb) ? gb : 6.0;
        // (folder, guide frames, the guide's strength, each guide token's attention strength or null = the strength)
        var half = Enumerable.Range(0, hw).Select(j => j % W < W / 2 ? 0.6f : 0f).ToArray();                       // the pixel mask at latent resolution: left half 1, right half 0
        var two = Enumerable.Range(0, 2 * hw).Select(j => j < hw ? 0.6f : 0.3f).ToArray();                         // guide A's tokens, then guide B's
        var cases = new (string Folder, int GuideFrames, float Strength, float[]? Tokens)[]
        {
            ("guidecall_s10", 1, 1.0f, null), ("guidecall_s06", 1, 0.6f, null), ("guidecall_mask", 1, 0.6f, half), ("guidecall_two", 2, 0.6f, two),
        };
        var report = new List<string>(); double worst = 0; int ran = 0;
        foreach (var group in cases.Where(c => Directory.Exists(Path.Combine(root, c.Folder))).GroupBy(c => c.GuideFrames))
        {
            int F = 2 + group.Key, P = 128 * F * hw + 8 * Ta * 16;
            using var model = await Ltx2AvModel.OpenAsync(accelerator, new LocalFolderModelSource(dir), "", "ltx-2-19b-dev-fp8.safetensors", B, F, H, W, Ta, L, (long)(residentGb * 1e9));
            foreach (var (folder, gf, strength, tokens) in group)
            {
                var d = Path.Combine(root, folder);
                float[] T(string prefix, long n) => F2_Raw(Ltx2_Tap(d, prefix), n);
                var sigma = T("guide_call_0_timestep", B);
                var mask = T("guide_call_0_c_denoise_mask", (long)B * 128 * F * hw);
                var frameMask = new float[B * F];
                for (int b = 0; b < B; b++) for (int f = 0; f < F; f++) frameMask[b * F + f] = mask[(((long)b * 128) * F + f) * hw];
                var kf = T("guide_call_0_c_keyframe_idxs", 2L * 3 * gf * hw * 2)[..(3 * gf * hw * 2)];              // row 0 (both rows carry the same guides)
                using var ctx = accelerator.Allocate1D(T("guide_call_0_c_c_crossattn", (long)B * L * Ltx2AvModel.ContextWidth));
                var den = await model.DenoiseAsync(T("guide_call_0_input", (long)B * P), new[] { sigma[0], sigma[1] }, ctx.View, 25f, new Ltx2Guide(gf, kf, strength, frameMask, tokens));
                double e = F2_RelRms(den, T("guide_call_0_output", (long)B * P)); worst = Math.Max(worst, e); ran++;
                report.Add($"{folder}: call 0 {e:E2} (frame mask [{string.Join(", ", frameMask[..F])}])");
                await accelerator.SynchronizeAsync();
            }
        }
        string line = string.Join("; ", report);
        Console.WriteLine("[Ltx2] guided call vs ComfyUI: " + line);
        Ur_Expect(ran >= 2, $"only {ran} guided-call tap folders found in {root}");
        Ur_Expect(worst < 1e-4, line);
        return line;
    });

    /// <summary>
    /// A guided call at a size where the kernels leave their one-group case (SpawnGen's tap _DevComms 2026-11-18-1200: 384 x 256 x 25 frames + 25 audio frames, a picture
    /// guide at the last frame, strength 0.6, cfg 4): 480 video tokens (4 latent frames of 8 x 12 + the 96-token guide frame: four query groups of the flash attention's
    /// 128, 960-row GEMMs over the batch), both of ComfyUI's calls (sigma 1 and 0.1, each on ComfyUI's own input), against its denoised outputs. Every other LTX-2 tap is 8 to 48 tokens.
    /// </summary>
    [TestMethod(Timeout = 3600000, Category = "HeavyModel")]
    public async Task Ltx2_GuidedCallMidSize_MatchesComfyUi() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is not (AcceleratorType.Cuda or AcceleratorType.OpenCL)) throw new UnsupportedTestException("streams a 27 GB checkpoint from disk: a desktop GPU");
        var dir = Path.Combine(Ltx2_Models(), "lightricks");
        var d = Path.Combine(Ltx2_Dumps(), "guided", "guidecall_mid");
        if (!Directory.Exists(d)) throw new UnsupportedTestException($"SpawnGen's mid-size guided-call taps not found in {d}");
        const int B = 2, F = 5, H = 8, W = 12, Ta = 25, L = 1024, hw = H * W, nv = F * hw, P = 128 * nv + 8 * Ta * 16;
        double residentGb = double.TryParse(Environment.GetEnvironmentVariable("LTX2_DIT_RESIDENT_GB"), System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out var gb) ? gb : 6.0;
        float[] T(string prefix, long n) => F2_Raw(Ltx2_Tap(d, prefix), n);
        var sw = System.Diagnostics.Stopwatch.StartNew();
        using var model = await Ltx2AvModel.OpenAsync(accelerator, new LocalFolderModelSource(dir), "", "ltx-2-19b-dev-fp8.safetensors", B, F, H, W, Ta, L, (long)(residentGb * 1e9));
        double open = sw.Elapsed.TotalSeconds;
        var mask = T("guide_call_0_c_denoise_mask", (long)B * 128 * nv);
        var frameMask = new float[F];
        for (int f = 0; f < F; f++) frameMask[f] = mask[(long)f * hw];                                           // the guide frame's mask is 0.4 to one float32 step across its tokens (0.4 / 0.40000004): the first
        var kf = T("guide_call_0_c_keyframe_idxs", 2L * 3 * hw * 2)[..(3 * hw * 2)];
        var guide = new Ltx2Guide(1, kf, 0.6f, frameMask);
        var report = new List<string>(); double worst = 0; var secs = new List<double>();
        for (int call = 0; call < 2; call++)
        {
            var sigma = T($"guide_call_{call}_timestep", B);
            using var ctx = accelerator.Allocate1D(T($"guide_call_{call}_c_c_crossattn", (long)B * L * Ltx2AvModel.ContextWidth));
            sw.Restart();
            var den = await model.DenoiseAsync(T($"guide_call_{call}_input", (long)B * P), new[] { sigma[0], sigma[1] }, ctx.View, 25f, guide);
            secs.Add(sw.Elapsed.TotalSeconds);
            var gold = T($"guide_call_{call}_output", (long)B * P);
            double e = F2_RelRms(den, gold); worst = Math.Max(worst, e);
            // the video part and the guide frame alone (the audio and the free frames must not hide them)
            double ev = F2_RelRms(den[..(128 * nv)], gold[..(128 * nv)]), num = 0, dn = 0;
            for (int c = 0; c < 128; c++) for (int t = nv - hw; t < nv; t++) { double a = den[c * nv + t], g = gold[c * nv + t]; num += (a - g) * (a - g); dn += g * g; }
            double eg = Math.Sqrt(num / dn); worst = Math.Max(worst, Math.Max(ev, eg));
            report.Add($"call {call} (sigma {sigma[0]}) {e:E2}, video {ev:E2}, guide frame {eg:E2}");
            await accelerator.SynchronizeAsync();
        }
        var wts = model.Transformer.Weights;
        string line = $"{string.Join("; ", report)} | {nv} video + {Ta} audio tokens, open {open:F1} s, calls {string.Join(" / ", secs.Select(s => s.ToString("F1")))} s ({wts.ResidentBlocks} of 48 blocks resident, read {wts.ReadSeconds:F1} s, upload {wts.UploadSeconds:F1} s)";
        Console.WriteLine("[Ltx2] mid-size guided call vs ComfyUI: " + line);
        Ur_Expect(worst < 1e-4, line);
        return line;
    });

    /// <summary>
    /// LTX-2 at a real clip's size (LTX2_REAL_SIZE = "width x height x frames", default 768x512x121: a [128, 16, 16, 24] video latent = 6144 tokens, 121 audio frames at 25 fps):
    /// one model call (batch 2, sigma 1, seeded noise, the template's conditioning), then the video and audio decodes of its denoised latent. No reference exists at
    /// this size (ComfyUI on a CPU takes hours), so this is a MEASUREMENT with sanity bounds: it must not throw, every output is finite and of ordinary size, and the
    /// stage seconds are printed ("[Benchmark] Ltx2 real size ..."). Parity at 480 tokens: <see cref="Ltx2_GuidedCallMidSize_MatchesComfyUi"/>.
    /// </summary>
    [TestMethod(Timeout = 7200000, Category = "HeavyModel")]
    public async Task Ltx2_RealSize_Benchmark() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is not (AcceleratorType.Cuda or AcceleratorType.OpenCL)) throw new UnsupportedTestException("streams a 27 GB checkpoint from disk: a desktop GPU");
        var dir = Path.Combine(Ltx2_Models(), "lightricks");
        const string file = "ltx-2-19b-dev-fp8.safetensors";
        if (!File.Exists(Path.Combine(dir, file))) throw new UnsupportedTestException($"{file} not found in {dir}");
        var size = (Environment.GetEnvironmentVariable("LTX2_REAL_SIZE") ?? "768x512x121").Split('x').Select(int.Parse).ToArray();
        int width = size[0], height = size[1], frames = size[2];
        Ur_Expect(size.Length == 3 && width % 32 == 0 && height % 32 == 0 && (frames - 1) % 8 == 0, $"LTX2_REAL_SIZE {width}x{height}x{frames}: multiples of 32 and 8 n + 1 frames");
        const int B = 2, L = 1024;
        int F = (frames - 1) / 8 + 1, H = height / 32, W = width / 32, nv = F * H * W, Ta = frames, V = 128 * nv, P = V + 8 * Ta * 16;   // 25 audio latent frames per second at 25 fps: one per video frame (the template: 9 / 9, the mid-size tap: 25 / 25)
        double residentGb = double.TryParse(Environment.GetEnvironmentVariable("LTX2_DIT_RESIDENT_GB"), System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out var gb) ? gb : 5.0;
        // seeded standard normal noise (Box-Muller); the same for both rows, as a sampler's first call
        var rng = new Random(20261117);
        var noise = new float[P];
        for (int i = 0; i < P; i += 2)
        {
            double u1 = 1.0 - rng.NextDouble(), u2 = rng.NextDouble(), r = Math.Sqrt(-2.0 * Math.Log(u1));
            noise[i] = (float)(r * Math.Cos(2 * Math.PI * u2)); if (i + 1 < P) noise[i + 1] = (float)(r * Math.Sin(2 * Math.PI * u2));
        }
        var packed = new float[2L * P]; Array.Copy(noise, 0, packed, 0, P); Array.Copy(noise, 0, packed, P, P);
        // the template's conditioning [uncond, cond] when SpawnGen's taps are here, else small seeded values (the timing does not depend on it)
        string call = Path.Combine(Ltx2_Dumps(), "call");
        float[] ctxHost;
        if (Directory.Exists(call)) ctxHost = F2_Raw(Ltx2_Tap(call, "ltx2_tcall_0_c_c_crossattn"), 2L * L * Ltx2AvModel.ContextWidth);
        else { ctxHost = new float[2L * L * Ltx2AvModel.ContextWidth]; for (int i = 0; i < ctxHost.Length; i++) ctxHost[i] = (float)(rng.NextDouble() - 0.5); }
        static (double Rms, int Bad) Stat(ReadOnlySpan<float> a) { double s = 0; int bad = 0; foreach (float v in a) { if (!float.IsFinite(v)) bad++; else s += (double)v * v; } return (Math.Sqrt(s / Math.Max(1, a.Length - bad)), bad); }
        var sw = System.Diagnostics.Stopwatch.StartNew();
        float[] den; double open, callS, read, upload; int resident;
        using (var ctx = accelerator.Allocate1D(ctxHost))
        using (var model = await Ltx2AvModel.OpenAsync(accelerator, new LocalFolderModelSource(dir), "", file, B, F, H, W, Ta, L, (long)(residentGb * 1e9)))
        {
            open = sw.Elapsed.TotalSeconds; sw.Restart();
            den = await model.DenoiseAsync(packed, new[] { 1f, 1f }, ctx.View, 25f);
            callS = sw.Elapsed.TotalSeconds;
            var wts = model.Transformer.Weights; read = wts.ReadSeconds; upload = wts.UploadSeconds; resident = wts.ResidentBlocks;
            await accelerator.SynchronizeAsync();
        }
        var (rmsV, badV) = Stat(den.AsSpan(P, V)); var (rmsA, badA) = Stat(den.AsSpan(P + V, P - V));           // the conditioned row
        string line = $"{width}x{height}x{frames}: {nv} video + {Ta} audio tokens | open {open:F1} s, one call (batch 2) {callS:F1} s ({resident} of 48 blocks resident, read {read:F1} s, upload {upload:F1} s); denoised rms video {rmsV:F3} / audio {rmsA:F3}";
        Console.WriteLine("[Benchmark] Ltx2 real size call: " + line);
        Ur_Expect(badV == 0 && badA == 0 && rmsV > 0.05 && rmsV < 20 && rmsA > 0.01 && rmsA < 20, "the denoised latent is not finite / ordinary: " + line + $" (non-finite {badV} / {badA})");
        // decodes of the conditioned row's denoised latent
        sw.Restart();
        float[] video; double vOpen, vDec;
        using (var vae = await Ltx2VideoDecoder.OpenAsync(accelerator, new LocalFolderModelSource(dir), "", file))
        {
            vOpen = sw.Elapsed.TotalSeconds; sw.Restart();
            video = await vae.DecodeAsync(den.AsSpan(P, V).ToArray(), F, H, W);
            vDec = sw.Elapsed.TotalSeconds;
            await accelerator.SynchronizeAsync();
        }
        var (rmsF, badF) = Stat(video);
        Ur_Expect(video.Length == 3L * frames * height * width && badF == 0 && rmsF > 0.01 && rmsF < 5, $"decoded video: {video.Length} values (want [3, {frames}, {height}, {width}]), rms {rmsF:F3}, non-finite {badF}");
        video = Array.Empty<float>();
        sw.Restart();
        float[] wav; double aDec;
        using (var ad = await Ltx2AudioDecoder.OpenAsync(accelerator, new LocalFolderModelSource(dir), "", file))
        {
            sw.Restart();
            wav = await ad.DecodeAsync(den.AsSpan(P + V, P - V).ToArray(), Ta);
            aDec = sw.Elapsed.TotalSeconds;
            await accelerator.SynchronizeAsync();
        }
        var (rmsW, badW) = Stat(wav);
        Ur_Expect(badW == 0 && wav.Length == 2 * 240 * (4 * Ta - 3), $"decoded audio: {wav.Length} samples, non-finite {badW}");
        line += $" | video decode open {vOpen:F1} s, decode {vDec:F1} s (frames rms {rmsF:F3}); audio decode {aDec:F1} s ({wav.Length / 2 / 24000.0:F2} s of stereo, rms {rmsW:F4})";
        Console.WriteLine("[Benchmark] Ltx2 real size: " + line);
        return line;
    });

    /// <summary>
    /// <see cref="Ltx2Pipeline"/> end to end: a prompt to a clip with sound (Gemma 3 encode, connectors, the sampler, video and audio decode), written as PNG frames and a
    /// WAV file to LTX2_PIPE_OUT (default: a folder in the temp directory) so a person can look at it. LTX2_PIPE_SIZE "width x height x frames" (default 768x512x121),
    /// LTX2_PIPE_DISTILLED (default 1: the distilled LoRA, 8 steps, no guidance; 0: LTX2_PIPE_STEPS steps at cfg 4), LTX2_PIPE_PROMPT, LTX2_PIPE_SEED.
    /// No reference for the whole run (every stage has its own ComfyUI gate): the checks are shape, finiteness, and that the frames are a moving picture
    /// (not one colour, not one frame repeated). Prints "[Benchmark] Ltx2 pipeline ...".
    /// </summary>
    [TestMethod(Timeout = 10800000, Category = "HeavyModel")]
    public async Task Ltx2_Pipeline_GeneratesClip() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is not (AcceleratorType.Cuda or AcceleratorType.OpenCL)) throw new UnsupportedTestException("streams a 27 GB checkpoint from disk: a desktop GPU");
        var paths = Ltx2Paths.InFolder();
        var src = new LocalFolderModelSource(Ltx2_Models());
        if (!src.Exists(paths.Checkpoint) || !src.Exists(paths.TextEncoder)) throw new UnsupportedTestException($"needs {paths.Checkpoint} and {paths.TextEncoder} in {Ltx2_Models()}");
        string Env(string name, string fallback) => Environment.GetEnvironmentVariable(name) is { Length: > 0 } v ? v : fallback;
        var size = Env("LTX2_PIPE_SIZE", "768x512x121").Split('x').Select(int.Parse).ToArray();
        bool distilled = Env("LTX2_PIPE_DISTILLED", "1") != "0";
        if (distilled && !src.Exists(paths.DistilledLora!)) throw new UnsupportedTestException($"the distilled run needs {paths.DistilledLora}");
        string prompt = Env("LTX2_PIPE_PROMPT", "A golden retriever puppy runs toward the camera through a sunlit meadow of tall grass and wildflowers, ears flapping, shallow depth of field, "
            + "warm evening light, cinematic slow motion. Birds sing and a gentle breeze rustles the grass.");
        string outDir = Env("LTX2_PIPE_OUT", Path.Combine(Path.GetTempPath(), "spawndev-ltx2-clip"));
        Directory.CreateDirectory(outDir);
        // LTX2_PIPE_KEYFRAME: a PNG at the clip's size to guide by (image-to-video), at pixel frame LTX2_PIPE_KEYFRAME_INDEX (default 0), strength LTX2_PIPE_KEYFRAME_STRENGTH (default 1)
        string keyPath = Env("LTX2_PIPE_KEYFRAME", "");
        int keyIndex = int.Parse(Env("LTX2_PIPE_KEYFRAME_INDEX", "0"));
        float keyStrength = float.Parse(Env("LTX2_PIPE_KEYFRAME_STRENGTH", "1"), System.Globalization.CultureInfo.InvariantCulture);
        byte[]? keyRgba = null; List<Ltx2Keyframe>? keyframes = null;
        if (keyPath != "")
        {
            var png = await File.ReadAllBytesAsync(keyPath);
            var (kw, kh) = SpawnDev.ILGPU.ML.Preprocessing.PngDecoder.GetDimensions(png);
            Ur_Expect(kw == size[0] && kh == size[1], $"the keyframe is {kw} x {kh}, the clip {size[0]} x {size[1]}");
            keyRgba = SpawnDev.ILGPU.ML.Preprocessing.PngDecoder.DecodePixels(png) ?? throw new Exception($"{keyPath}: not an 8-bit non-interlaced PNG");
            var rgb = new byte[3 * kw * kh];
            for (int i = 0; i < kw * kh; i++) { rgb[3 * i] = keyRgba[4 * i]; rgb[3 * i + 1] = keyRgba[4 * i + 1]; rgb[3 * i + 2] = keyRgba[4 * i + 2]; }
            keyframes = new List<Ltx2Keyframe> { Ltx2Keyframe.FromRgb(rgb, kw, kh, keyIndex, keyStrength) };
        }
        var req = new Ltx2Request
        {
            Keyframes = keyframes,
            ComfyCompatibleFp8Lora = Env("LTX2_PIPE_COMPAT", "0") == "1",                        // LTX2_PIPE_COMPAT=1: the LoRA's 8-bit weights re-rounded ComfyUI's way
            Prompt = prompt, NegativePrompt = "worst quality, inconsistent motion, blurry, jittery, distorted", Width = size[0], Height = size[1], Frames = size[2],
            Distilled = distilled, Steps = int.Parse(Env("LTX2_PIPE_STEPS", "20")), Cfg = 4f, Seed = long.Parse(Env("LTX2_PIPE_SEED", "42")),
        };
        var pipe = new Ltx2Pipeline(accelerator, src, paths);
        var stages = new List<string>();
        var progress = new Progress<Wan.WanProgress>(p => { if (p.Fraction is 0.0 or 1.0) lock (stages) stages.Add($"{p.Stage}: {p.Message}"); });
        var sw = System.Diagnostics.Stopwatch.StartNew();
        var clip = await pipe.GenerateAsync(req, progress);
        double total = sw.Elapsed.TotalSeconds;
        var frames = clip.Video.FramesRgba ?? throw new Exception("no RGBA frames");
        int w = clip.Video.Width, h = clip.Video.Height, n = frames.Count;
        Ur_Expect(n == req.Frames && w == req.Width && h == req.Height && frames.All(f => f.Length == 4 * w * h), $"{n} frames of {w} x {h}");
        // a moving picture: the first frame has contrast, and the last differs from the first
        static (double Mean, double Std) Luma(byte[] f) { double s = 0, s2 = 0; int c = f.Length / 4; for (int i = 0; i < f.Length; i += 4) { double y = 0.299 * f[i] + 0.587 * f[i + 1] + 0.114 * f[i + 2]; s += y; s2 += y * y; } double m = s / c; return (m, Math.Sqrt(Math.Max(0, s2 / c - m * m))); }
        var (m0, sd0) = Luma(frames[0]); var (m1, sd1) = Luma(frames[n - 1]);
        double moved = 0; for (int i = 0; i < frames[0].Length; i += 4) moved += Math.Abs(frames[0][i + 1] - frames[n - 1][i + 1]); moved /= w * h;
        string keyNote = "";
        if (keyRgba != null)
        {
            // the guided frame against the keyframe: luma correlation and mean absolute difference (0..255)
            var gf = frames[keyIndex >= 0 ? keyIndex : n + keyIndex];
            double sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0, mad = 0; int px = w * h;
            for (int i = 0; i < 4 * px; i += 4)
            {
                double a = 0.299 * gf[i] + 0.587 * gf[i + 1] + 0.114 * gf[i + 2], b = 0.299 * keyRgba[i] + 0.587 * keyRgba[i + 1] + 0.114 * keyRgba[i + 2];
                sa += a; sb += b; saa += a * a; sbb += b * b; sab += a * b; mad += Math.Abs(a - b);
            }
            double corr = (sab / px - sa / px * sb / px) / Math.Sqrt(Math.Max(1e-9, (saa / px - sa / px * sa / px) * (sbb / px - sb / px * sb / px)));
            keyNote = $" | keyframe at {keyIndex} strength {keyStrength}: guided frame vs the picture luma correlation {corr:F3}, mean |diff| {mad / px:F1}";
            Ur_Expect(corr > 0.5, "the guided frame does not resemble the keyframe:" + keyNote);
        }
        for (int i = 0; i < n; i++) await File.WriteAllBytesAsync(Path.Combine(outDir, $"frame_{i:D4}.png"), SpawnDev.ILGPU.ML.Preprocessing.PngEncoder.Encode(frames[i], w, h));
        double audioRms = 0; string wavNote = "no audio";
        if (clip.Audio is { } wav)
        {
            int samples = wav.Length / 2;
            for (int i = 0; i < wav.Length; i++) { if (!float.IsFinite(wav[i])) throw new Exception($"non-finite audio sample at {i}"); audioRms += (double)wav[i] * wav[i]; }
            audioRms = Math.Sqrt(audioRms / Math.Max(1, wav.Length));
            // 16-bit PCM stereo WAV (interleaved from the channel-first waveform)
            using var fs = File.Create(Path.Combine(outDir, "audio.wav")); using var bw = new BinaryWriter(fs);
            int dataBytes = samples * 4, rate = clip.AudioSampleRate;
            bw.Write("RIFF"u8.ToArray()); bw.Write(36 + dataBytes); bw.Write("WAVEfmt "u8.ToArray()); bw.Write(16); bw.Write((short)1); bw.Write((short)2); bw.Write(rate); bw.Write(rate * 4); bw.Write((short)4); bw.Write((short)16);
            bw.Write("data"u8.ToArray()); bw.Write(dataBytes);
            for (int i = 0; i < samples; i++) for (int c = 0; c < 2; c++) bw.Write((short)Math.Clamp((int)MathF.Round(wav[c * samples + i] * 32767f), -32768, 32767));
            wavNote = $"{samples / (double)rate:F2} s of stereo at {rate} Hz, rms {audioRms:F4}";
        }
        string secs = string.Join(", ", clip.Seconds.Select(kv => $"{kv.Key} {kv.Value:F1} s"));
        string line = $"{w}x{h}x{n} {(distilled ? "distilled 8 steps" : $"{req.Steps} steps cfg {req.Cfg}")}{(req.ComfyCompatibleFp8Lora ? ", ComfyUI-compatible fp8 LoRA" : "")}, seed {req.Seed}: total {total:F1} s ({secs}) | first frame luma {m0:F1} +- {sd0:F1}, last {m1:F1} +- {sd1:F1}, mean |first - last| {moved:F1}{keyNote} | {wavNote} | {outDir}";
        Console.WriteLine("[Benchmark] Ltx2 pipeline: " + line);
        lock (stages) Console.WriteLine("[Ltx2] pipeline stages: " + string.Join(" | ", stages));
        Ur_Expect(sd0 > 4 && sd1 > 4 && moved > 1, "the frames are flat or do not move: " + line);
        return line;
    });

    /// <summary>
    /// ComfyUI's whole guided sampling run (the same graph as <see cref="Ltx2_GuidedCall_MatchesComfyUi"/>: CFG 4, Euler over sigmas [1, 0.1, 0], strengths 1.0 and 0.6)
    /// through <see cref="Ltx2AvModel"/>'s guided SampleAsync:
    /// masked sampling (the model sees m x + (1 - m) latent, the step uses m D + (1 - m) latent). Against the latent entering call 1 and the final latent (guide frame
    /// included). The start state is ComfyUI's call-0 input with the guide frame's blend undone (strength 1: the frame is the clean latent and its state is never read).
    /// </summary>
    [TestMethod(Timeout = 3600000, Category = "HeavyModel")]
    public async Task Ltx2_GuidedSample_MatchesComfyUi() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is not (AcceleratorType.Cuda or AcceleratorType.OpenCL)) throw new UnsupportedTestException("streams a 27 GB checkpoint from disk: a desktop GPU");
        var dir = Path.Combine(Ltx2_Models(), "lightricks");
        var root = Path.Combine(Ltx2_Dumps(), "guided");
        if (!Directory.Exists(root)) throw new UnsupportedTestException($"SpawnGen's guided-call taps not found in {root}");
        const int B = 2, F = 3, H = 3, W = 4, Ta = 9, L = 1024, nv = F * H * W, V = 128 * nv, P = V + 8 * Ta * 16;
        double residentGb = double.TryParse(Environment.GetEnvironmentVariable("LTX2_DIT_RESIDENT_GB"), System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out var gb) ? gb : 6.0;
        using var model = await Ltx2AvModel.OpenAsync(accelerator, new LocalFolderModelSource(dir), "", "ltx-2-19b-dev-fp8.safetensors", B, F, H, W, Ta, L, (long)(residentGb * 1e9));
        var report = new List<string>(); double worst = 0;
        foreach (var (folder, strength) in new[] { ("guidecall_s10", 1.0f), ("guidecall_s06", 0.6f) })
        {
            var d = Path.Combine(root, folder);
            float[] T(string prefix, long n) => F2_Raw(Ltx2_Tap(d, prefix), n);
            var sigmas = T("guide_call_0_c_transformer_options_sample_sigmas", 3);
            var mask = T("guide_call_0_c_denoise_mask", (long)B * 128 * F * H * W);
            var frameMask = new float[F];
            for (int f = 0; f < F; f++) frameMask[f] = mask[(long)f * H * W];
            var kf = T("guide_call_0_c_keyframe_idxs", 2L * 3 * H * W * 2)[..(3 * H * W * 2)];
            var latent = T("guide_av_in_samples_part0", V).Concat(T("guide_av_in_samples_part1", 8L * Ta * 16)).ToArray();     // the clean latent: the guide frame's encoding, zeros elsewhere
            var in0 = T("guide_call_0_input", (long)B * P)[..P];
            float M(int k) => k < V ? frameMask[(k % nv) / (H * W)] : 1f;
            // the sampler's own start state: ComfyUI's call input is m x + (1 - m) latent
            var start = new float[P];
            for (int k = 0; k < P; k++) { float m = M(k); start[k] = m > 0f ? (in0[k] - (1f - m) * latent[k]) / m : in0[k]; }
            using var ctx = accelerator.Allocate1D(T("guide_call_0_c_c_crossattn", (long)B * L * Ltx2AvModel.ContextWidth));
            float[]? afterStep0 = null;
            var x = await model.SampleAsync(start, ctx.View, sigmas, 4f, 25f, new Ltx2Guide(1, kf, strength, frameMask), latent, (i, xi) => { if (i == 0) afterStep0 = xi; });
            // call 1's input is the blend of the state after step 0
            var in1 = new float[P];
            for (int k = 0; k < P; k++) { float m = M(k); in1[k] = m * afterStep0![k] + (1f - m) * latent[k]; }
            double e1 = F2_RelRms(in1, T("guide_call_1_input", (long)B * P)[..P]);
            var final = T("guide_out_samples_part0", V).Concat(T("guide_out_samples_part1", 8L * Ta * 16)).ToArray();
            double e2 = F2_RelRms(x, final);
            // the guide frame alone (a third of the video tokens: an error there must not hide in the whole-latent figure)
            int g0 = 2 * H * W; double num = 0, den = 0;
            for (int c = 0; c < 128; c++) for (int t = g0; t < nv; t++) { double a = x[c * nv + t], b = final[c * nv + t]; num += (a - b) * (a - b); den += b * b; }
            double eg = Math.Sqrt(num / den);
            worst = Math.Max(worst, Math.Max(e1, Math.Max(e2, eg)));
            report.Add($"strength {strength}: call 1's input {e1:E2}, final latent {e2:E2} (guide frame {eg:E2})");
        }
        string line = string.Join("; ", report);
        Console.WriteLine("[Ltx2] guided sampling vs ComfyUI: " + line);
        Ur_Expect(worst < 5e-4, line);
        return line;
    });

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
