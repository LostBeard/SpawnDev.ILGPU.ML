// ─────────────────────────────────────────────────────────────────────────────────────────────────
//  SpawnDev.ILGPU.ML — Example 04: GGUF text-gen runner (desktop)
//
//  STREAM-loads a local .gguf (any size — gemma4:12b is 7 GB, past the ~2 GB byte[] cap) and runs a
//  forward pass on the GPU, printing the per-position argmax. This is the desktop validation vehicle for
//  the streaming GGUF loader + the gemma4 attention path; the argmax can be diffed against a llama.cpp /
//  ollama reference for the E2E correctness check.
//
//    dotnet run -- <path/to/model.gguf>                 # default tokens (load + finite-logits probe)
//    dotnet run -- <model.gguf> 2,651,6037,576,9881     # explicit input_ids (comma-separated)
// ─────────────────────────────────────────────────────────────────────────────────────────────────

using System.Diagnostics;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using ILGPU.Runtime.OpenCL;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.GGUF;
using SpawnDev.ILGPU.ML.Kernels;
using SpawnDev.ILGPU.ML.Tensors;

string modelPath = args.FirstOrDefault(a => !a.Contains(',') && (a.EndsWith(".gguf") || File.Exists(a)))
    ?? @"D:\users\tj\Projects\gemma4-12b-Q4_K_M.gguf";

// input_ids: explicit comma-separated arg, else a default probe sequence (bos=2 + arbitrary valid ids).
int[] tokenIds = args.FirstOrDefault(a => a.Contains(','))?.Split(',')
        .Select(s => int.Parse(s.Trim())).ToArray()
    ?? new[] { 2, 1000, 2000, 3000, 4000 };

// GGUF_GEMM_BENCH=1 → localize the prefill MatMul GFLOPS ceiling: f32 register-blocked GEMM (no dequant)
// vs the Q4_K dequant register-blocked GEMM at the SAME shapes. If f32 >> dequant the per-element B-tile
// dequant is the overhead (ML fix); if ~equal, the GEMM codegen is the ceiling. Random weight bytes (timing
// only). No model needed.
if (Environment.GetEnvironmentVariable("GGUF_GEMM_BENCH") == "1")
{
    using var bctx = MLContext.Create().ToContext();
    var bcuda = bctx.GetCudaDevices(); var bcl = bctx.GetCLDevices();
    Device bdev = bcuda.Count > 0 ? (Device)bcuda[0] : bcl.Count > 0 ? (Device)bcl[0] : bctx.GetPreferredDevice(preferCPU: false);
    using var bacc = bdev.CreateAccelerator(bctx);
    Console.WriteLine($"Accelerator: {bacc.Name} ({bacc.AcceleratorType})");
    SpawnDev.ILGPU.ML.Kernels.FusedDequantMatMul.EnableMultiRowGemm = true;
    var brng = new Random(1);
    double Gf(int M, int K, int N, double ms) => 2.0 * M * K * N / (ms / 1000.0) / 1e9;
    void Bench(int M, int K, int N)
    {
        Console.WriteLine($"\n--- M={M} K={K} N={N} (2MKN={2.0 * M * K * N / 1e9:F2} GFLOP) ---");
        int iters = 20;
        var A = new float[M * K]; for (int i = 0; i < A.Length; i++) A[i] = (float)(brng.NextDouble() - 0.5);
        var Bf = new float[K * N]; for (int i = 0; i < Bf.Length; i++) Bf[i] = (float)(brng.NextDouble() - 0.5);
        if (M > 1)
        using (var aBuf = bacc.Allocate1D(A))
        using (var bBuf = bacc.Allocate1D(Bf))
        using (var cBuf = bacc.Allocate1D<float>(M * N))
        {
            var rb = new SpawnDev.ILGPU.ML.Kernels.RegisterBlockedMatMul(bacc);
            rb.MatMul(aBuf.View, bBuf.View, cBuf.View, M, K, N); bacc.Synchronize();
            var sw = System.Diagnostics.Stopwatch.StartNew();
            for (int i = 0; i < iters; i++) rb.MatMul(aBuf.View, bBuf.View, cBuf.View, M, K, N);
            bacc.Synchronize(); sw.Stop();
            double ms = sw.Elapsed.TotalMilliseconds / iters;
            Console.WriteLine($"  f32 RegBlocked : {ms,8:F3} ms  {Gf(M, K, N, ms),7:F1} GFLOPS");
        }
        void DequantBench(SpawnDev.ILGPU.ML.GGUF.GGMLType type, int rowBytesPerBlock, string label)
        {
            int bytesPerRow = K / 256 * rowBytesPerBlock; int wBytes = ((N * bytesPerRow) + 3) / 4 * 4;
            var wq = new byte[wBytes]; brng.NextBytes(wq);
            using var aBuf = bacc.Allocate1D(A);
            using var wBuf = bacc.Allocate1D(wq);
            using var cBuf = bacc.Allocate1D<float>(M * N);
            var fq = new SpawnDev.ILGPU.ML.Kernels.FusedDequantMatMul(bacc);
            fq.Forward(aBuf.View, wBuf.View, cBuf.View, M, K, N, type); bacc.Synchronize();
            var sw = System.Diagnostics.Stopwatch.StartNew();
            for (int i = 0; i < iters; i++) fq.Forward(aBuf.View, wBuf.View, cBuf.View, M, K, N, type);
            bacc.Synchronize(); sw.Stop();
            double ms = sw.Elapsed.TotalMilliseconds / iters;
            // At M=1 (decode GEMV) the kernel is BANDWIDTH-bound: it reads the whole quant weight (wBytes) once.
            // GB/s vs the card's peak (RTX 4070 ~504 GB/s) is the right metric; GFLOPS is for the M>1 GEMM.
            double gbs = wBytes / (ms / 1000.0) / 1e9;
            Console.WriteLine($"  {label}: {ms,8:F3} ms  {Gf(M, K, N, ms),7:F1} GFLOPS  {gbs,7:F1} GB/s  (weight {wBytes / 1e6:F1} MB)");
        }
        DequantBench(SpawnDev.ILGPU.ML.GGUF.GGMLType.Q4_K, 144, "Q4_K dequant   ");
        DequantBench(SpawnDev.ILGPU.ML.GGUF.GGMLType.Q6_K, 210, "Q6_K dequant   ");
        // Q8_0 (32-elem blocks, 34B = 272B/256-elem) has a TRIVIAL dequant (d × int8, no sub-block scale) — the
        // ALU-vs-memory diagnostic for the M=1 GEMV: if Q8_0 GB/s >> Q4_K's, the Q4_K 6-bit scale extraction is
        // the per-element ALU bottleneck (→ cache the sub-block scales), not memory bandwidth.
        DequantBench(SpawnDev.ILGPU.ML.GGUF.GGMLType.Q8_0, 272, "Q8_0 dequant   ");
    }
    Console.WriteLine("\n######## DECODE GEMV (M=1, bandwidth-bound — the Ollama decode gap) ########");
    Bench(1, 3584, 18944);    // MLP gate/up   (Q4_K)
    Bench(1, 18944, 3584);    // MLP down      (Q6_K in qwen)
    Bench(1, 3584, 3584);     // attn q-proj   (Q4_K)
    Console.WriteLine("\n######## PREFILL GEMM (M=1081) ########");
    Bench(1081, 3584, 18944); // MLP gate/up
    Bench(1081, 18944, 3584); // MLP down
    Bench(1081, 3584, 3584);  // attn q-proj
    return 0;
}

if (!File.Exists(modelPath))
{
    Console.Error.WriteLine($"Model not found: {modelPath}");
    return 1;
}

// GGUF_PROBE=1 → header-only diagnostic: dump norm-weight stats per layer + decode named token ids.
// No GPU, no full load - just reads the small F32 norm tensors on demand from the stream.
if (Environment.GetEnvironmentVariable("GGUF_PROBE") == "1")
{
    await using var ps = File.OpenRead(modelPath);
    var pm = await SpawnDev.ILGPU.ML.GGUF.GGUFParser.ParseHeaderAsync(ps);
    pm.SourceStream = ps;
    (double rms, double min, double max, double mean) St(float[]? v)
    {
        if (v == null || v.Length == 0) return (0, 0, 0, 0);
        double s2 = 0, mn = v[0], mx = v[0], sum = 0;
        foreach (var x in v) { s2 += (double)x * x; mn = Math.Min(mn, x); mx = Math.Max(mx, x); sum += x; }
        return (Math.Sqrt(s2 / v.Length), mn, mx, sum / v.Length);
    }
    void Dump(string name)
    {
        var t = pm.Tensors.FirstOrDefault(x => x.Name == name);
        if (t == null) { Console.WriteLine($"  {name,-34} (absent)"); return; }
        var v = pm.GetTensorFloat32(t);
        var (r, mn, mx, me) = St(v);
        string head = v == null ? "" : string.Join(" ", v.Take(6).Select(x => x.ToString("F3")));
        Console.WriteLine($"  {name,-34} {t.Type,-5} shape=[{string.Join(",", t.Shape)}] n={v?.Length} rms={r,7:F3} min={mn,8:F3} max={mx,8:F3}  first=[{head}]");
    }
    Console.WriteLine("=== NORM WEIGHT STATS (raw stored values; gemma applies (1+w)) ===");
    foreach (int L in new[] { 0, 1, 5, 11, 24, 47 })
        foreach (var sub in new[] { "attn_norm", "attn_q_norm", "ffn_norm", "post_attention_norm", "post_ffw_norm" })
            Dump($"blk.{L}.{sub}.weight");
    Dump("output_norm.weight");
    Console.WriteLine("\n=== TOKEN DECODE ===");
    var tok = SpawnDev.ILGPU.ML.Preprocessing.SentencePieceTokenizer.FromGGUF(pm);
    if (tok != null)
        foreach (int id in new[] { 1748, 215625, 2, 651, 6037 })
            Console.WriteLine($"  token {id,7} = '{tok.Decode(new[] { id })}'");
    else Console.WriteLine("  (tokenizer FromGGUF returned null)");
    return 0;
}

// GGUF_GEN="<prompt>" → greedy autoregressive decode (the real production correctness test:
// coherent text out = a correct forward; compare to ollama greedy). GGUF_GEN_RAW=1 skips the
// gemma chat template (raw completion). GGUF_GEN_N sets the token budget (default 24).
string? genPrompt = Environment.GetEnvironmentVariable("GGUF_GEN");
if (!string.IsNullOrEmpty(genPrompt))
{
    try { return await GenerateAsync(modelPath, genPrompt,
        raw: Environment.GetEnvironmentVariable("GGUF_GEN_RAW") == "1",
        maxNew: int.TryParse(Environment.GetEnvironmentVariable("GGUF_GEN_N"), out var nn) ? nn : 24); }
    catch (Exception ex) { Console.Error.WriteLine($"GEN FAILED: {ex.GetType().Name}: {ex.Message}\n{ex.StackTrace}"); return 1; }
}

try
{
    return await RunAsync(modelPath, tokenIds);
}
catch (Exception ex)
{
    Console.Error.WriteLine($"FAILED: {ex.GetType().Name}: {ex.Message}\n{ex.StackTrace}");
    return 1;
}

async Task<int> GenerateAsync(string path, string prompt, bool raw, int maxNew)
{
    // Tokenizer + control-token ids from the GGUF header (fast; no weight load).
    await using var hs = File.OpenRead(path);
    var gm = await SpawnDev.ILGPU.ML.GGUF.GGUFParser.ParseHeaderAsync(hs);
    gm.SourceStream = hs;
    var tokenArr = gm.GetMetadataStringArray("tokenizer.ggml.tokens") ?? throw new Exception("no tokens");
    var idOf = new Dictionary<string, int>();
    for (int i = 0; i < tokenArr.Length; i++) idOf[tokenArr[i]] = i;
    var tok = SpawnDev.ILGPU.ML.Preprocessing.SentencePieceTokenizer.FromGGUF(gm)!;
    int Id(string s) => idOf.TryGetValue(s, out var v) ? v : -1;
    // gemma4 control tokens (NOT gemma2/3's <start_of_turn>): turn-open <|turn> / turn-close <turn|>,
    // thinking toggle <|think|>, and the eot/eos. Verified by scanning the GGUF vocab.
    int bos = Id("<bos>"), turnO = Id("<|turn>"), turnC = Id("<turn|>"), think = Id("<|think|>"), eos = Id("<eos>");

    var ids = new List<int>();
    if (raw) { if (bos >= 0) ids.Add(bos); ids.AddRange(tok.Encode(prompt)); }
    // gemma4 chat template, now via the library helper (dogfooding ChatTemplates.BuildGemma4PromptTokens).
    else ids.AddRange(SpawnDev.ILGPU.ML.Preprocessing.ChatTemplates.BuildGemma4PromptTokens(tok, systemPrompt: null, userMessage: prompt, thinking: true));
    Console.WriteLine($"Prompt: \"{prompt}\"  (raw={raw})\nPrompt token ids ({ids.Count}): [{string.Join(",", ids)}]");
    Console.WriteLine($"control: bos={bos} turn_open={turnO} turn_close={turnC} think={think} eos={eos}\n");

    using var context = MLContext.Create().ToContext();
    var cuda = context.GetCudaDevices();
    Device device = cuda.Count > 0 ? (Device)cuda[0] : context.GetPreferredDevice(preferCPU: false);
    using var accelerator = device.CreateAccelerator(context);
    Console.WriteLine($"Accelerator: {accelerator.Name}");

    using var session = await InferenceSession.CreateFromGGUFFileAsync(accelerator, path);
    Console.WriteLine($"Loaded: {session}\n");

    // GGUF_GEN_GEN2=1 → drive the general GgufGenerator (architecture-agnostic core: incremental
    // streaming detokenizer + stop sequences). Same gemma4 prompt + turn-close stop → must reproduce the
    // pipeline output. Also self-checks that the streamed deltas equal the returned full text.
    if (Environment.GetEnvironmentVariable("GGUF_GEN_GEN2") == "1")
    {
        var tok2 = SpawnDev.ILGPU.ML.Preprocessing.SentencePieceTokenizer.FromGGUF(gm)!;
        var promptIds = SpawnDev.ILGPU.ML.Preprocessing.ChatTemplates.BuildGemma4PromptTokens(tok2, null, prompt, thinking: true);
        int turnClose = SpawnDev.ILGPU.ML.Preprocessing.ChatTemplates.Gemma4TurnCloseId(tok2);
        using var gen2 = new SpawnDev.ILGPU.ML.Pipelines.GgufGenerator(session, accelerator, gm, maxSeqLen: promptIds.Length + maxNew + 8);
        var streamed = new System.Text.StringBuilder();
        var gSw = Stopwatch.StartNew();
        var stopEnv = Environment.GetEnvironmentVariable("GGUF_GEN_STOP");
        var stopStrings = string.IsNullOrEmpty(stopEnv) ? null : stopEnv.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        var res = await gen2.GenerateAsync(promptIds,
            config: new SpawnDev.ILGPU.ML.Preprocessing.GenerationConfig { MaxNewTokens = maxNew },
            stopStrings: stopStrings,
            stopTokenIds: new[] { turnClose },
            onDelta: d => { streamed.Append(d); return Task.CompletedTask; });
        gSw.Stop();
        bool streamMatches = streamed.ToString() == res.Text;
        Console.WriteLine($"\n=== GENERATOR OUTPUT ({gSw.Elapsed.TotalSeconds:F1}s, stop={res.Stop}, gen={res.GeneratedTokens}, streamed==full:{streamMatches}) ===\n{res.Text}");
        return 0;
    }

    // GGUF_GEN_PIPE=1 → use the first-class GgufTextGenerationPipeline (chat template + KV-cache decode +
    // sampler in one call). Dogfoods the library API end-to-end.
    if (Environment.GetEnvironmentVariable("GGUF_GEN_PIPE") == "1")
    {
        using var pipe = new SpawnDev.ILGPU.ML.Pipelines.GgufTextGenerationPipeline(session, accelerator, gm, maxSeqLen: ids.Count + maxNew + 8);
        var pSw = Stopwatch.StartNew();
        var text = await pipe.GenerateAsync(prompt, maxNewTokens: maxNew,
            onToken: (n, _) => { Console.Write($"\r  [pipeline] {n} tokens..."); return Task.CompletedTask; });
        pSw.Stop();
        Console.WriteLine($"\n=== PIPELINE OUTPUT ({pSw.Elapsed.TotalSeconds:F1}s) ===\n{text}");
        return 0;
    }

    // GGUF_GEN_KV=1 → incremental KV-cache decode (O(n)); else full-recompute (O(n^2)). The two MUST
    // produce identical greedy tokens — that's the decode-equivalence validation gate for the cache.
    bool useKV = Environment.GetEnvironmentVariable("GGUF_GEN_KV") == "1";
    GGUFDecodeKVCache? kvCache = null;
    if (useKV)
    {
        int nLayers = (int)gm.BlockCount, nH = (int)gm.AttentionHeadCount;
        int defNKV = (int)gm.AttentionHeadCountKV; if (defNKV == 0) defNKV = nH;
        int embd = (int)gm.EmbeddingLength, defHd = embd / nH;
        var kvHeadsArr = new int[nLayers]; var hdArr = new int[nLayers];
        for (int L = 0; L < nLayers; L++)
        { var cfg = GGUFGraphBuilder.GetLayerAttnConfig(gm, L, nH, defNKV, defHd); kvHeadsArr[L] = cfg.NKVHeads; hdArr[L] = cfg.HeadDim; }
        // GGUF_GEN_MAXSEQ overrides the cache size — the SERVER runs at 16384 (for agentic clients' big
        // prompts), which exposes a gemma4 large-context decode slowdown that the tight default hides.
        int defMaxSeq = ids.Count + maxNew + 8;
        int maxSeq = int.TryParse(Environment.GetEnvironmentVariable("GGUF_GEN_MAXSEQ"), out var ms) && ms > defMaxSeq ? ms : defMaxSeq;
        kvCache = new GGUFDecodeKVCache(accelerator, kvHeadsArr, hdArr, maxSeqLen: maxSeq);
        session.EnableGGUFDecode(kvCache);
        Console.WriteLine($"[KV-cache decode] {nLayers} layers, maxSeq={maxSeq}");
    }

    // GGUF_NODE_TIMING=1 → per-node Execute timing (decomposes the residual: which ops eat the CPU
    // dispatch time). Dumped after the last (steady-state seq=1) step, aggregated by op type.
    bool nodeTiming = Environment.GetEnvironmentVariable("GGUF_NODE_TIMING") == "1";
    if (nodeTiming) SpawnDev.ILGPU.ML.Graph.GraphExecutor.CapturedNodeTimingsMs = new();
    // GGUF_PEROP_SYNC=1 with node timing → each node's captured time includes its OWN GPU drain,
    // giving TRUE per-op GPU compute attribution (vs the sync-blocking attribution of the default
    // path, where async kernel work surfaces at the next sync point, not its real producer).
    if (nodeTiming && Environment.GetEnvironmentVariable("GGUF_PEROP_SYNC") == "1")
        SpawnDev.ILGPU.ML.Graph.GraphExecutor.PerOpSync = true;

    var gen = new List<int>();
    var sw = Stopwatch.StartNew();
    int[] stepIds = ids.ToArray();  // KV path: prefill = whole prompt, then 1 token/step
    for (int step = 0; step < maxNew; step++)
    {
        var cur = useKV ? stepIds : ids.Concat(gen).ToArray();
        var idf = cur.Select(i => (float)i).ToArray();
        using var inBuf = accelerator.Allocate1D(idf);
        var input = new Tensor(inBuf.View, new[] { 1, cur.Length }, "input_ids");
        var stepSw = Stopwatch.StartNew();
        var outputs = useKV
            ? await session.RunDecodeStepAsync(new Dictionary<string, Tensor> { ["input_ids"] = input })
            : await session.RunAsync(new Dictionary<string, Tensor> { ["input_ids"] = input });
        await accelerator.SynchronizeAsync();
        stepSw.Stop();
        var logits = outputs.TryGetValue("logits", out var l) ? l : outputs.Values.First();
        int vocab = logits.Shape[^1];
        int seqOut = logits.ElementCount / vocab;  // KV decode step => 1; prefill/full => cur.Length
        var host = new float[logits.ElementCount];
        logits.Data.CopyToCPU(host);
        int last = (seqOut - 1) * vocab, arg = 0; float best = host[last];
        for (int v = 1; v < vocab; v++) if (host[last + v] > best) { best = host[last + v]; arg = v; }
        gen.Add(arg);
        // PERF BREAKDOWN (Rule 4 measurement): partition the step into executor-total / readback round-trips /
        // GPU sync-drains / recompile, so we KNOW where decode time goes instead of guessing. The residual
        // (execMs - readbackMs - drainMs) is pure per-node dispatch + CPU + buffer-alloc cost.
        double execMs = SpawnDev.ILGPU.ML.Graph.GraphExecutor.LastRunTotalMs;
        double rbMs = SpawnDev.ILGPU.ML.Graph.GraphExecutor.LastRunReadbackMs;
        int rbCount = SpawnDev.ILGPU.ML.Graph.GraphExecutor.LastRunReadbackCount;
        double drainMs = SpawnDev.ILGPU.ML.Graph.GraphExecutor.LastRunSyncDrainMs;
        int drainCount = SpawnDev.ILGPU.ML.Graph.GraphExecutor.LastRunSyncDrainCount;
        double recompMs = session.LastRecompileMs;
        double residMs = execMs - rbMs - drainMs;
        Console.WriteLine($"  step {step,2}: token {arg,7} '{tok.Decode(new[] { arg })}' (logit {best:F3})  "
            + $"| seq={cur.Length} wall={stepSw.Elapsed.TotalMilliseconds,7:F1}ms exec={execMs,7:F1} "
            + $"readback={rbMs,6:F1}({rbCount}) drain={drainMs,6:F1}({drainCount}) recompile={recompMs,6:F1} "
            + $"residual={residMs,7:F1} bufs={session.LastExecutorBufferCount}");
        if (arg == turnC || arg == eos) { Console.WriteLine("  [stop token]"); break; }
        stepIds = new[] { arg };  // decode: feed only the new token (KV path)
    }
    sw.Stop();

    if (nodeTiming && SpawnDev.ILGPU.ML.Graph.GraphExecutor.CapturedNodeTimingsMs is { } nt && nt.Count > 0)
    {
        // Key format: "{idx:D3}_{OpType}_{outName}". Aggregate by op type (the lever-relevant axis).
        var byOp = new Dictionary<string, (double ms, int n)>();
        foreach (var (k, ms) in nt)
        {
            var parts = k.Split('_', 3);
            string op = parts.Length >= 2 ? parts[1] : k;
            var cur = byOp.GetValueOrDefault(op, (0, 0));
            byOp[op] = (cur.ms + ms, cur.n + 1);
        }
        double totalMs = nt.Values.Sum();
        Console.WriteLine($"\n=== PER-NODE EXECUTE TIMING (last steady-state step; {nt.Count} nodes, sum {totalMs:F1}ms) ===");
        Console.WriteLine("  by op type (sum ms desc):");
        foreach (var (op, v) in byOp.OrderByDescending(x => x.Value.ms))
            Console.WriteLine($"    {op,-22} sum={v.ms,8:F1}ms  n={v.n,4}  avg={v.ms / v.n,6:F3}ms  ({100 * v.ms / totalMs,5:F1}%)");
        Console.WriteLine("  top 15 individual nodes:");
        foreach (var (k, ms) in nt.OrderByDescending(x => x.Value).Take(15))
            Console.WriteLine($"    {ms,8:F3}ms  {k}");
    }

    kvCache?.Dispose();
    Console.WriteLine($"\n=== GENERATED ({gen.Count} tokens, {sw.Elapsed.TotalSeconds:F1}s) ===");
    Console.WriteLine(tok.Decode(gen.ToArray()));
    return 0;
}

async Task<int> RunAsync(string path, int[] ids)
{
    // The application owns the accelerator (library code never disposes it). Prefer CUDA.
    using var context = MLContext.Create().ToContext();
    var cuda = context.GetCudaDevices();
    var opencl = context.GetCLDevices();
    Device device = cuda.Count > 0 ? (Device)cuda[0]
                  : opencl.Count > 0 ? (Device)opencl[0]
                  : context.GetPreferredDevice(preferCPU: false);
    using var accelerator = device.CreateAccelerator(context);
    Console.WriteLine($"Accelerator : {accelerator.Name} ({accelerator.AcceleratorType})");

    var fi = new FileInfo(path);
    Console.WriteLine($"Model       : {path} ({fi.Length / 1024.0 / 1024.0 / 1024.0:F2} GB)");
    Console.WriteLine($"input_ids   : [{string.Join(", ", ids)}]  (seq={ids.Length})");
    Console.WriteLine();

    // ── STREAM-load (never materializes the 7 GB as a byte[]) ──
    var sw = Stopwatch.StartNew();
    InferenceSession.VerboseLogging = true;
    using var session = await InferenceSession.CreateFromGGUFFileAsync(accelerator, path,
        onProgress: (stage, pct) => { if (pct == 0 || pct == 100) Console.WriteLine($"  [load] {stage} {pct}%"); });
    sw.Stop();
    Console.WriteLine($"\nLoaded in {sw.Elapsed.TotalSeconds:F1}s — {session}\n");

    // ── Forward pass ──
    var idf = ids.Select(i => (float)i).ToArray();
    using var inBuf = accelerator.Allocate1D(idf);
    var input = new Tensor(inBuf.View, new[] { 1, ids.Length }, "input_ids");

    // Kernel-bisection: sync after every node so a GPU trap fires on the EXACT faulting node (the verbose
    // log shows which). Toggle with env GGUF_PEROP_SYNC=1 (slow — one flush per op on a 1437-node graph).
    if (Environment.GetEnvironmentVariable("GGUF_PEROP_SYNC") == "1")
    {
        SpawnDev.ILGPU.ML.Graph.GraphExecutor.PerOpSync = true;
        SpawnDev.ILGPU.ML.Graph.GraphExecutor.VerboseLogging = true;
    }

    // GGUF_CAPTURE=1 → dump per-node RMS/absMax of the residual trajectory (our CapturedOutputs gear) so
    // we can SEE where magnitude blows up and whether positions differentiate. Captures up to 5 positions
    // of the 3840-d hidden (5*3840 = 19200) for the residual stream.
    bool capture = Environment.GetEnvironmentVariable("GGUF_CAPTURE") == "1";
    if (capture)
    {
        SpawnDev.ILGPU.ML.Graph.GraphExecutor.CapturedOutputs = new();
        SpawnDev.ILGPU.ML.Graph.GraphExecutor.CapturedNodeInfo = new();
        SpawnDev.ILGPU.ML.Graph.GraphExecutor.CaptureMaxElements = 20000;
    }

    sw.Restart();
    var outputs = await session.RunAsync(new Dictionary<string, Tensor> { ["input_ids"] = input });
    await accelerator.SynchronizeAsync();
    sw.Stop();

    if (capture)
    {
        var caps = SpawnDev.ILGPU.ML.Graph.GraphExecutor.CapturedOutputs!;
        int hdim = 3840, sq = ids.Length;
        // RMS/absMax of a slice; and per-position RMS to test differentiation.
        (double rms, double amax) Stat(float[] v, int off, int n)
        {
            double s2 = 0, am = 0; int c = 0;
            for (int i = off; i < off + n && i < v.Length; i++) { s2 += (double)v[i] * v[i]; am = Math.Max(am, Math.Abs(v[i])); c++; }
            return (c > 0 ? Math.Sqrt(s2 / c) : 0, am);
        }
        Console.WriteLine("\n=== RESIDUAL-STREAM TRAJECTORY (per-node stats; pos-RMS = RMS of each position's 3840-d hidden) ===");
        // Show: embed_out, every 6th layer's residual output, final_norm_out, logits_presoftcap.
        var keys = caps.Keys.ToList();
        foreach (var key in keys)
        {
            string name = key.Substring(key.IndexOf('_', 4) + 1);
            bool interesting = name == "embed_out" || name == "final_norm_out" || name.Contains("logits")
                || name.EndsWith("_attn_merged")
                || System.Text.RegularExpressions.Regex.IsMatch(name, @"^scaled_out|^blk\.(0|1|5|6|11|23|24|47)_");
            if (!interesting) continue;
            var v = caps[key];
            var (rms, amax) = Stat(v, 0, v.Length);
            // per-position RMS (first min(sq, v.Length/hdim) positions)
            var posRms = new List<string>();
            for (int p = 0; p < sq && (p + 1) * hdim <= v.Length; p++)
            { var (r, _) = Stat(v, p * hdim, hdim); posRms.Add($"{r:F3}"); }
            // Cross-position COSINE similarity pos0-vs-posLast: if it climbs to ~1 at deep layers the
            // positions have COLLAPSED (attention is averaging everything / not differentiating context).
            int nPos = 0; for (int p = 0; p < sq && (p + 1) * hdim <= v.Length; p++) nPos++;
            string cos = "";
            if (nPos >= 2)
            {
                int last = nPos - 1; double dot = 0, n0 = 0, nl = 0;
                for (int i = 0; i < hdim; i++) { float a = v[i], b = v[last * hdim + i]; dot += (double)a * b; n0 += (double)a * a; nl += (double)b * b; }
                double c = (n0 > 0 && nl > 0) ? dot / (Math.Sqrt(n0) * Math.Sqrt(nl)) : 0;
                cos = $" cos(p0,p{last})={c:F3}";
            }
            Console.WriteLine($"  {key,-44} rms={rms,8:F3} absMax={amax,10:F3}{cos}");
        }
    }

    var logits = outputs.TryGetValue("logits", out var l) ? l : outputs.Values.First();
    int vocab = logits.Shape[^1];
    int seq = logits.ElementCount / vocab;
    Console.WriteLine($"Forward in {sw.Elapsed.TotalSeconds:F2}s — logits {string.Join("x", logits.Shape)} (vocab={vocab})");

    var host = new float[logits.ElementCount];
    logits.Data.CopyToCPU(host);

    // Finite check + per-position argmax (the last position is the next-token prediction).
    int nonFinite = host.Count(v => float.IsNaN(v) || float.IsInfinity(v));
    Console.WriteLine(nonFinite == 0 ? "Logits      : all finite ✓" : $"Logits      : {nonFinite} non-finite ✗");
    // TEACHER-FORCING: when seq>1, the input IS a known-good sequence (prompt + a llama.cpp reference
    // continuation). At position s the argmax should predict input[s+1]. We print argmax, the EXPECTED
    // next token, match?, and (on a miss) the rank + logit of the expected token vs the argmax — that
    // localizes the FIRST position my forward diverges from llama.cpp and by how much.
    int matches = 0, checks = 0;
    for (int s = 0; s < seq; s++)
    {
        int baseIdx = s * vocab, arg = 0; float best = host[baseIdx];
        for (int v = 1; v < vocab; v++) if (host[baseIdx + v] > best) { best = host[baseIdx + v]; arg = v; }
        string tf = "";
        if (s < seq - 1)
        {
            int expected = ids[s + 1];
            float expLogit = host[baseIdx + expected];
            int rank = 1; for (int v = 0; v < vocab; v++) if (host[baseIdx + v] > expLogit) rank++;
            bool ok = arg == expected; checks++; if (ok) matches++;
            tf = ok ? $"  ✓ predicts next {expected}"
                    : $"  ✗ next={expected} rank={rank} (its logit {expLogit:F3} vs argmax {best:F3}, gap {best - expLogit:F3})";
        }
        Console.WriteLine($"  pos {s,2}: argmax = {arg,7} (logit {best:F4}){tf}");
    }
    if (checks > 0) Console.WriteLine($"\nTEACHER-FORCING vs reference: {matches}/{checks} positions predict the reference next token.");
    Console.WriteLine($"NEXT-TOKEN argmax (pos {seq - 1}) is the value to diff against a llama.cpp/ollama reference.");
    return nonFinite == 0 ? 0 : 2;
}
