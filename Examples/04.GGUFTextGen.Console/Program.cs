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

// GGUF_PTX_PROBE=1 → compile the dp4a Q4_K decode GEMV via the CUDA backend and dump its PTX, then count
// 128-bit vectorized weight loads (ld.v4.b32 = LDG.E.128) vs scalar (ld.b32). PROVES the AsAligned16/W16
// weight load lit up — must NOT guess (CLAUDE.md 4b). No model needed. (Tuvok 2026-06-23 GEMV bandwidth lever.)
if (Environment.GetEnvironmentVariable("GGUF_PTX_PROBE") == "1")
{
    using var pctx = MLContext.Create().ToContext();
    var pcuda = pctx.GetCudaDevices();
    if (pcuda.Count == 0) { Console.WriteLine("[ptx-probe] no CUDA device"); return 1; }
    using var pacc = (CudaAccelerator)pcuda[0].CreateAccelerator(pctx);
    Console.WriteLine($"[ptx-probe] {pacc.Name}");
    foreach (var name in new[] { "GemvDp4aQ4_KImpl", "GemvDp4aQ6_KImpl" })
    {
        var m = typeof(FusedDequantMatMul).GetMethod(name,
            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static);
        if (m == null) { Console.WriteLine($"[ptx-probe] {name}: NOT FOUND"); continue; }
        var compiled = pacc.Backend.Compile(
            ILGPU.Backends.EntryPoints.EntryPointDescription.FromExplicitlyGroupedKernel(m),
            new ILGPU.Runtime.KernelSpecialization(128, null));
        var ptx = (compiled as ILGPU.Backends.PTX.PTXCompiledKernel)?.PTXAssembly ?? "(not PTX)";
        int Count(string s) { int n = 0, i = 0; while ((i = ptx.IndexOf(s, i, StringComparison.Ordinal)) >= 0) { n++; i += s.Length; } return n; }
        int v4 = Count("ld.v4.b32"), v2 = Count("ld.v2.b32");
        int scalar = Count("ld.b32") - Count("ld.param.b32");
        Console.WriteLine($"[ptx-probe] {name}: ld.v4.b32={v4}  ld.v2.b32={v2}  scalar ld.b32={scalar}");
        File.WriteAllText(Path.Combine(Path.GetTempPath(), name + ".ptx"), ptx);
    }
    return 0;
}

// GGUF_ATTN_BENCH=1 → register vs shared-slice per-query attention A/B on the desktop accelerator (same
// 8h×256×256×128 shape as the PMT WebGPU benchmark, for a CUDA-vs-WebGPU side-by-side). No model needed.
if (Environment.GetEnvironmentVariable("GGUF_ATTN_BENCH") == "1")
{
    using var actx = MLContext.Create().ToContext();
    var acuda = actx.GetCudaDevices();
    Device adev = acuda.Count > 0 ? (Device)acuda[0] : actx.GetPreferredDevice(preferCPU: false);
    using var aacc = adev.CreateAccelerator(actx);
    Console.WriteLine($"[attn-bench] {aacc.Name} ({aacc.AcceleratorType}, warp={aacc.WarpSize})");
    int nHeads = 8, SQ = 256, SKV = 256, D = 128;
    var arng = new Random(42);
    var aq = new float[nHeads * SQ * D]; var ak = new float[nHeads * SKV * D]; var av = new float[nHeads * SKV * D];
    for (int i = 0; i < aq.Length; i++) aq[i] = (float)(arng.NextDouble() * 2 - 1);
    for (int i = 0; i < ak.Length; i++) ak[i] = (float)(arng.NextDouble() * 2 - 1);
    for (int i = 0; i < av.Length; i++) av[i] = (float)(arng.NextDouble() * 2 - 1);
    using var aqB = aacc.Allocate1D(aq); using var akB = aacc.Allocate1D(ak); using var avB = aacc.Allocate1D(av);
    using var aoB = aacc.Allocate1D<float>(nHeads * SQ * D);
    var fa = new FusedAttentionKernel(aacc);
    double TimeAttn(bool register)
    {
        FusedAttentionKernel.EnableRegisterAttention = register;
        fa.ForwardStrided<float>(aqB.View, akB.View, avB.View, aoB.View, nHeads, nHeads, SQ, SKV, D, true, int.MaxValue, 0, 0f, SKV * D);
        aacc.Synchronize();
        double best = double.MaxValue;
        for (int it = 0; it < 8; it++)
        {
            var sw = Stopwatch.StartNew();
            fa.ForwardStrided<float>(aqB.View, akB.View, avB.View, aoB.View, nHeads, nHeads, SQ, SKV, D, true, int.MaxValue, 0, 0f, SKV * D);
            aacc.Synchronize(); sw.Stop();
            best = Math.Min(best, sw.Elapsed.TotalMilliseconds);
        }
        return best;
    }
    double aShared = TimeAttn(false), aReg = TimeAttn(true);
    Console.WriteLine($"[attn-bench] Register attention {nHeads}h×{SQ}×{SKV}×{D} [{aacc.AcceleratorType}]: shared-slice {aShared:F3}ms → register {aReg:F3}ms = {aShared / aReg:F2}× faster");
    return 0;
}

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

// GGUF_TENSORS=<substr> → header-only: dump every tensor whose name contains <substr> with its GGUF
// Dimensions (ne order, ne[0] contiguous) + Type. Decisive for layout questions (e.g. is the LFM2
// shortconv.conv.weight [d_conv, d_inner] or [d_inner, d_conv]?). No GPU, no weight load.
if (Environment.GetEnvironmentVariable("GGUF_TENSORS") is { Length: > 0 } tsub)
{
    await using var ps = File.OpenRead(modelPath);
    var pm = await SpawnDev.ILGPU.ML.GGUF.GGUFParser.ParseHeaderAsync(ps);
    Console.WriteLine($"arch={pm.Architecture} embd={pm.EmbeddingLength} blocks={pm.BlockCount} heads={pm.AttentionHeadCount} kv={pm.AttentionHeadCountKV}");
    foreach (var t in pm.Tensors.Where(t => t.Name.Contains(tsub, StringComparison.OrdinalIgnoreCase)).OrderBy(t => t.Name))
        Console.WriteLine($"  {t.Name,-42} {t.Type,-8} ne=[{string.Join(",", t.Dimensions)}]");
    return 0;
}

// GGUF_ATTNCFG=1 → header-only: print GetLayerAttnConfig (isGlobal/window/ropeBase/rotaryDim/nkv/headDim)
// for EVERY layer. Confirms per-layer KV-head resolution (LFM2 stores head_count_kv as a per-layer array;
// a scalar-fallback bug would silently give attention layers the wrong KV-head count). No GPU.
if (Environment.GetEnvironmentVariable("GGUF_ATTNCFG") == "1")
{
    await using var ps = File.OpenRead(modelPath);
    var pm = await SpawnDev.ILGPU.ML.GGUF.GGUFParser.ParseHeaderAsync(ps);
    int nH = (int)pm.AttentionHeadCount, defKV = (int)pm.AttentionHeadCountKV; if (defKV == 0) defKV = nH;
    int embd = (int)pm.EmbeddingLength, defHd = nH > 0 ? embd / nH : 0;
    Console.WriteLine($"arch={pm.Architecture} embd={embd} blocks={pm.BlockCount} nH={nH} defKV={defKV} defHd={defHd}");
    bool hasKvArr = pm.Metadata.TryGetValue($"{pm.Architecture}.attention.head_count_kv", out var kvRaw);
    Console.WriteLine($"  head_count_kv metadata: present={hasKvArr} type={(kvRaw?.GetType().Name ?? "null")} value={(kvRaw is Array ar ? "[" + string.Join(",", ar.Cast<object>()) + "]" : kvRaw?.ToString() ?? "null")}");
    for (int L = 0; L < (int)pm.BlockCount; L++)
    {
        var c = SpawnDev.ILGPU.ML.GGUF.GGUFGraphBuilder.GetLayerAttnConfig(pm, L, nH, defKV, defHd);
        bool isConv = pm.Tensors.Any(t => t.Name == $"blk.{L}.shortconv.in_proj.weight");
        Console.WriteLine($"  L{L,2} {(isConv ? "CONV" : "attn")}  nkv={c.NKVHeads,2} headDim={c.HeadDim,3} rotaryDim={c.RotaryDim,3} ropeBase={c.RopeBase,10:F0} global={c.IsGlobal} window={c.Window}");
    }
    return 0;
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

// GGUF_DECODE_EQUIV="<prompt>" → decode-equivalence gate: greedy full-recompute (O(n^2), zero-pad conv)
// MUST produce token-identical output to KV-cache decode (O(n), conv-STATE cache). This is the LFM2
// short-conv regression guard - a broken/absent conv-state cache diverges after the prefill. GGUF_BACKEND
// selects the backend. Returns 0 on identity, 3 on divergence. (Tuvok 2026-07-14.)
string? equivPrompt = Environment.GetEnvironmentVariable("GGUF_DECODE_EQUIV");
if (!string.IsNullOrEmpty(equivPrompt))
{
    try { return await DecodeEquivTestAsync(modelPath, equivPrompt,
        maxNew: int.TryParse(Environment.GetEnvironmentVariable("GGUF_GEN_N"), out var en) ? en : 24); }
    catch (Exception ex) { Console.Error.WriteLine($"EQUIV TEST FAILED: {ex.GetType().Name}: {ex.Message}\n{ex.StackTrace}"); return 1; }
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

// GGUF_PREFIX_CACHE_TEST=1 → prove KV-prefix-cache reuse is TOKEN-IDENTICAL to a fresh full prefill, and
// measure the prefill-TTFT win. Builds prompt A (200 raw tokens) and B = A + 50 tokens; decodes 10 greedy
// tokens from B with the cache OFF (fresh) and with the cache ON (after first decoding 1 token from A to
// populate the prefix), then asserts the two 10-token id sequences match.
if (Environment.GetEnvironmentVariable("GGUF_PREFIX_CACHE_TEST") == "1")
{
    try { return await PrefixCacheTestAsync(modelPath); }
    catch (Exception ex) { Console.Error.WriteLine($"PREFIX-CACHE TEST FAILED: {ex.GetType().Name}: {ex.Message}\n{ex.StackTrace}"); return 1; }
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

async Task<int> PrefixCacheTestAsync(string path)
{
    using var context = MLContext.Create().ToContext();
    var cuda = context.GetCudaDevices();
    Device device = cuda.Count > 0 ? (Device)cuda[0] : context.GetPreferredDevice(preferCPU: false);
    using var accelerator = device.CreateAccelerator(context);
    Console.WriteLine($"Accelerator: {accelerator.Name} ({accelerator.AcceleratorType})");

    await using var hs = File.OpenRead(path);
    var gm = await SpawnDev.ILGPU.ML.GGUF.GGUFParser.ParseHeaderAsync(hs);
    gm.SourceStream = hs;
    var tok = SpawnDev.ILGPU.ML.Preprocessing.SentencePieceTokenizer.FromGGUF(gm)!;

    using var session = await InferenceSession.CreateFromGGUFFileAsync(accelerator, path);
    Console.WriteLine($"Loaded: {session}\n");

    // Build a deterministic 200-token raw prompt A, and B = A + 50 more tokens. BOS-prefixed so position 0
    // is a real model start. We encode some natural text and pad/extend to the target lengths from the vocab.
    int bosId = tok.BosId >= 0 ? tok.BosId : 1;
    var baseText = "The quick brown fox jumps over the lazy dog. In a distant land, a curious scientist "
        + "studied the patterns of the stars and the rhythms of the tides, recording every observation in a "
        + "weathered leather journal that smelled faintly of ink and rain.";
    var seed = tok.Encode(baseText).ToList();
    if (seed.Count == 0) seed.Add(100);
    var aList = new List<int> { bosId };
    // Repeat the seed deterministically until we have >= 200 tokens, then trim to exactly 200.
    int si = 0;
    while (aList.Count < 200) { aList.Add(seed[si % seed.Count]); si++; }
    var A = aList.Take(200).ToArray();
    var bList = new List<int>(A);
    while (bList.Count < 250) { bList.Add(seed[si % seed.Count]); si++; }
    var B = bList.Take(250).ToArray();

    Console.WriteLine($"Prompt A: {A.Length} tokens, Prompt B: {B.Length} tokens (B = A + {B.Length - A.Length} suffix)\n");

    const int decodeN = 10;
    var cfg = new SpawnDev.ILGPU.ML.Preprocessing.GenerationConfig { MaxNewTokens = decodeN };
    // maxSeqLen must hold B + decode with margin; keep it well above so no tail-truncation occurs (which
    // would disable reuse by design).
    int maxSeq = B.Length + decodeN + 64;

    // WARMUP — make the TTFT numbers TRUSTWORTHY. ILGPU JIT-caches kernels per-kernel-method on the shared
    // `session` (sizes are runtime args, so the FIRST prefill pays ALL the JIT, not per-M). Without a warmup,
    // the first timed leg would be JIT+compute and the second pure compute — not comparable. Prime the whole
    // prefill+decode graph once here (throwaway generator, result discarded) so BOTH timed legs below measure
    // pure PREFILL COMPUTE. On qwen2 the M>1 prefill GEMM dominates (re-dequantizes the weight per output
    // row), so warm prefill scales ~linearly with token count: 250 (fresh) vs 49 (reuse) ≈ the prefix-cache win.
    SpawnDev.ILGPU.ML.Pipelines.GgufGenerator.EnablePrefixCache = false;
    using (var warmGen = new SpawnDev.ILGPU.ML.Pipelines.GgufGenerator(session, accelerator, gm, maxSeqLen: maxSeq))
    {
        var w = await warmGen.GenerateFirstTokenIdsAsync(B, cfg); // primes M=250 prefill + decode kernels
        Console.WriteLine($"[warmup] JIT primed (discarded {w.ids.Length} ids, {w.ttftMs:F1}ms incl. one-time JIT)\n");
    }

    // ── Generator 1: prefix cache OFF, fresh full prefill of B, decode 10 (now WARM → pure compute) ──
    SpawnDev.ILGPU.ML.Pipelines.GgufGenerator.EnablePrefixCache = false;
    int[] freshIds;
    double freshTtftMs;
    using (var gen1 = new SpawnDev.ILGPU.ML.Pipelines.GgufGenerator(session, accelerator, gm, maxSeqLen: maxSeq))
    {
        var sw = Stopwatch.StartNew();
        var r1 = await gen1.GenerateFirstTokenIdsAsync(B, cfg);
        sw.Stop();
        freshIds = r1.ids;
        freshTtftMs = r1.ttftMs;
        Console.WriteLine($"[fresh, cache OFF] reusedPrefix={gen1.LastReusedPrefix} TTFT(prefill+1st)={freshTtftMs:F1}ms  ids=[{string.Join(",", freshIds)}]");
    }

    // ── Generator 2: prefix cache ON. First decode 1 token from A (populates the cache with A's prefix),
    //    THEN decode 10 from B (must reuse the ~200-token A prefix, prefill only the 50-token suffix). ──
    SpawnDev.ILGPU.ML.Pipelines.GgufGenerator.EnablePrefixCache = true;
    int[] reuseIds;
    double reuseTtftMs;
    int reusedPrefix;
    using (var gen2 = new SpawnDev.ILGPU.ML.Pipelines.GgufGenerator(session, accelerator, gm, maxSeqLen: maxSeq))
    {
        // Turn 1: decode 1 token from A → cache now holds A (200) + 1 generated token.
        var warm = await gen2.GenerateFirstTokenIdsAsync(A, new SpawnDev.ILGPU.ML.Preprocessing.GenerationConfig { MaxNewTokens = 1 });
        Console.WriteLine($"[warm from A] reusedPrefix={gen2.LastReusedPrefix} cachedAfter={A.Length + warm.ids.Length}");
        // Turn 2: decode 10 from B → should reuse the 200-token A prefix (B[0..200) == A).
        var sw = Stopwatch.StartNew();
        var r2 = await gen2.GenerateFirstTokenIdsAsync(B, cfg);
        sw.Stop();
        reuseIds = r2.ids;
        reuseTtftMs = r2.ttftMs;
        reusedPrefix = gen2.LastReusedPrefix;
        Console.WriteLine($"[reuse, cache ON] reusedPrefix={reusedPrefix} (expected ~{A.Length}) TTFT(prefill+1st)={reuseTtftMs:F1}ms  ids=[{string.Join(",", reuseIds)}]");
    }

    // ── Assert token-identity ──
    int firstDiv = -1;
    int n = Math.Min(freshIds.Length, reuseIds.Length);
    for (int i = 0; i < n; i++) if (freshIds[i] != reuseIds[i]) { firstDiv = i; break; }
    bool pass = firstDiv < 0 && freshIds.Length == reuseIds.Length && freshIds.Length == decodeN;

    Console.WriteLine();
    if (pass)
    {
        Console.WriteLine("PREFIX-CACHE TOKEN-IDENTICAL: PASS");
        Console.WriteLine($"  reusedPrefix={reusedPrefix} tokens (of B's {B.Length}); prefilled only {B.Length - reusedPrefix} tokens on reuse.");
        Console.WriteLine($"  TTFT fresh={freshTtftMs:F1}ms  reuse={reuseTtftMs:F1}ms  speedup={(reuseTtftMs > 0 ? freshTtftMs / reuseTtftMs : 0):F2}x");
        return 0;
    }
    else
    {
        Console.WriteLine("PREFIX-CACHE TOKEN-IDENTICAL: FAIL");
        if (firstDiv >= 0)
            Console.WriteLine($"  first divergent index {firstDiv}: fresh={freshIds[firstDiv]} reuse={reuseIds[firstDiv]}");
        else
            Console.WriteLine($"  length mismatch: fresh={freshIds.Length} reuse={reuseIds.Length} (expected {decodeN})");
        Console.WriteLine($"  fresh=[{string.Join(",", freshIds)}]");
        Console.WriteLine($"  reuse=[{string.Join(",", reuseIds)}]");
        return 3;
    }
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
    // BOS: prefer the tokenizer's bos_token_id (from GGUF metadata) over a NAME lookup — LFM2's BOS is
    // "<|startoftext|>" (id 1), not "<bos>", so Id("<bos>")=-1 would silently skip it. Many models
    // (LFM2, llama) set add_bos_token=true and produce degenerate output WITHOUT the leading BOS.
    int bos = tok.BosId >= 0 ? tok.BosId : Id("<bos>");
    int turnO = Id("<|turn>"), turnC = Id("<turn|>"), think = Id("<|think|>"), eos = Id("<eos>");

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
        // Match the production decode path (GgufGenerator): recycle fixed-shape decode output buffers across
        // steps instead of renting a fresh one every token. Without this the bare harness leaks +1 pool buffer
        // per step (unbounded growth + per-step alloc/GC) and reports a PESSIMISTIC decode time vs the server.
        session.CacheShapeReadbacks = true;
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
        // DIAGNOSTIC: GGUF_GEN_TOPK=N dumps the top-N next tokens at THIS step; GGUF_FIND=<substr>
        // reports the logit + rank of every token whose decode contains <substr> (e.g. "Paris").
        if (step == 0)
        {
            var topkEnv = Environment.GetEnvironmentVariable("GGUF_GEN_TOPK");
            var findEnv = Environment.GetEnvironmentVariable("GGUF_FIND");
            if (topkEnv != null && int.TryParse(topkEnv, out var topk))
            {
                var idx = Enumerable.Range(0, vocab).OrderByDescending(v => host[last + v]).Take(topk).ToArray();
                Console.WriteLine($"  TOP-{topk}: " + string.Join("  ", idx.Select(v => $"[{v}]'{tok.Decode(new[] { v })}'={host[last + v]:F2}")));
            }
            if (findEnv != null)
            {
                var ranked = Enumerable.Range(0, vocab).OrderByDescending(v => host[last + v]).ToArray();
                for (int r = 0; r < ranked.Length; r++)
                {
                    var s = tok.Decode(new[] { ranked[r] });
                    if (s.Contains(findEnv, StringComparison.OrdinalIgnoreCase))
                    { Console.WriteLine($"  FIND '{findEnv}': token [{ranked[r]}]'{s}' logit={host[last + ranked[r]]:F2} RANK={r}"); if (r > 0) break; }
                }
            }
        }
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

    // ── Stage-1 CUDA-graph decode capture/replay PROBE (GGUF_DECODE_GRAPH_PROBE=1) ──
    // De-risks the dispatch-collapse arc on the REAL model: the decode loop above left the executor
    // WARM (pools + readback cache primed, CacheShapeReadbacks finalized). We capture ONE more decode
    // forward at the current state into a CUDA graph and replay it. SAME-STATE replay (we do NOT advance
    // the KV cursor), so every replay reproduces the SAME logits — the correctness gate is
    // replay-argmax == a fresh non-graph forward at this exact state. Then we time direct forwards vs
    // graph replays to MEASURE the ~26ms CPU-dispatch residual collapse (the lever to beat Ollama).
    if (useKV && Environment.GetEnvironmentVariable("GGUF_DECODE_GRAPH_PROBE") == "1")
    {
        if (accelerator is not CudaAccelerator)
            Console.WriteLine("\n[graph-probe] SKIP: not a CUDA accelerator (graph capture is CUDA-only).");
        else if (!CudaStream.SupportsGraphCapture)
            Console.WriteLine("\n[graph-probe] SKIP: driver does not expose the CUDA graph API.");
        else if (gen.Count < 3)
            Console.WriteLine("\n[graph-probe] SKIP: need >=3 warm decode steps first (run with GGUF_GEN_N>=8).");
        else
        {
            Console.WriteLine("\n=== Stage-1 CUDA-graph decode capture/replay probe ===");
            int nextTok = stepIds[0];                         // the next single-token decode input
            int statePast = session.DecodePastLen;            // freeze this decode state
            const int R = 50;                                 // replays / direct steps to time

            // Stable single-token input buffer — a captured embedding-gather bakes THIS device pointer,
            // so it must not move between replays (a fresh Allocate1D per step would break replay).
            using var capIn = accelerator.Allocate1D(new[] { (float)nextTok });
            var capInput = new Tensor(capIn.View, new[] { 1, 1 }, "input_ids");

            float[] ReadLogits(Tensor lg)
            {
                int vc = lg.Shape[^1]; int so = lg.ElementCount / vc;
                var h = new float[lg.ElementCount]; lg.Data.CopyToCPU(h);
                var last = new float[vc]; Array.Copy(h, (so - 1) * vc, last, 0, vc);
                return last;
            }
            int ArgMax(float[] v) { int a = 0; float b = v[0]; for (int i = 1; i < v.Length; i++) if (v[i] > b) { b = v[i]; a = i; } return a; }

            // (1) BASELINE: a fresh non-graph forward at this exact state → reference token.
            session.SetGGUFDecodePastLen(statePast);
            var refOut = await session.RunDecodeStepAsync(new Dictionary<string, Tensor> { ["input_ids"] = capInput });
            await accelerator.SynchronizeAsync();
            var refLogits = ReadLogits(refOut.TryGetValue("logits", out var rl) ? rl : refOut.Values.First());
            int refArg = ArgMax(refLogits);

            var capStream = (CudaStream)accelerator.CreateStream();
            int replayArg; double directMs, graphMs; float maxAbsDiff;
            try
            {
                // Stable per-forward attention params slots (vs the ring) so captured nodes read a fixed
                // device pointer the warm pass populated — and the capture pass skips the H2D (sync-illegal).
                FusedAttentionKernel.UseStableCaptureSlots = true;
                using (accelerator.WithDefaultStream(capStream))   // reroute all *StreamKernel launches → capStream
                {
                    // Warm pass A (drains ON): populate the stable attention params slots + prime JIT/modules.
                    session.SetGGUFDecodePastLen(statePast);
                    await session.RunDecodeStepAsync(new Dictionary<string, Tensor> { ["input_ids"] = capInput });
                    await accelerator.SynchronizeAsync();

                    // Warm pass B (drains SUPPRESSED, immediate buffer-return = same footprint as capture): grows
                    // the pool to the no-drain working set so the capture pass allocates NOTHING (a cuMemAlloc
                    // mid-capture is illegal and crashes). Not captured.
                    session.SetGGUFDecodePastLen(statePast);
                    SpawnDev.ILGPU.ML.Graph.GraphExecutor.SuppressDrains = true;
                    await session.RunDecodeStepAsync(new Dictionary<string, Tensor> { ["input_ids"] = capInput });
                    SpawnDev.ILGPU.ML.Graph.GraphExecutor.SuppressDrains = false;
                    await accelerator.SynchronizeAsync();

                    // CAPTURE one forward at the same state (drains suppressed = capture-clean).
                    session.SetGGUFDecodePastLen(statePast);
                    SpawnDev.ILGPU.ML.Graph.GraphExecutor.SuppressDrains = true;
                    // Global (not ThreadLocal): RunDecodeStepAsync's awaits (Task.Yield etc.) resume on other
                    // thread-pool threads, and ThreadLocal capture forbids ending from a different thread.
                    capStream.BeginCapture(CudaStreamCaptureMode.Global);
                    var capOut = await session.RunDecodeStepAsync(new Dictionary<string, Tensor> { ["input_ids"] = capInput });
                    using var graph = capStream.EndCapture();
                    SpawnDev.ILGPU.ML.Graph.GraphExecutor.SuppressDrains = false;
                    var capLogitsT = capOut.TryGetValue("logits", out var cl) ? cl : capOut.Values.First();

                    using var gexec = graph.Instantiate();
                    gexec.Upload(capStream);

                    // REPLAY once → compare logits to the non-graph baseline (must match: same kernels, same buffers).
                    gexec.Launch(capStream);
                    await capStream.SynchronizeAsync();
                    var replayLogits = ReadLogits(capLogitsT);
                    replayArg = ArgMax(replayLogits);
                    maxAbsDiff = 0f;
                    for (int i = 0; i < refLogits.Length; i++) { float d = MathF.Abs(replayLogits[i] - refLogits[i]); if (d > maxAbsDiff) maxAbsDiff = d; }

                    // TIME graph replays (fixed state — pure 1×cuGraphLaunch + GPU compute, ~zero host dispatch).
                    var gsw = Stopwatch.StartNew();
                    for (int r = 0; r < R; r++) { gexec.Launch(capStream); await capStream.SynchronizeAsync(); }
                    gsw.Stop(); graphMs = gsw.Elapsed.TotalMilliseconds / R;

                    // BATCHED: R launches back-to-back, ONE sync → isolates pure GPU compute/step from any
                    // per-step sync overhead. If this ≈ per-step graphMs, decode is GPU-bound (graphs can't help
                    // more); if much smaller, the per-step sync was the cost.
                    var bsw = Stopwatch.StartNew();
                    for (int r = 0; r < R; r++) gexec.Launch(capStream);
                    await capStream.SynchronizeAsync();
                    bsw.Stop();
                    Console.WriteLine($"  graph batched   : {bsw.Elapsed.TotalMilliseconds / R,7:F2} ms/step (R launches, 1 sync = pure GPU)");
                }

                // Direct timing uses the production ring path (apples-to-apples with today's decode).
                FusedAttentionKernel.UseStableCaptureSlots = false;
                // TIME direct non-graph forwards at the same state (full ~703-launch host dispatch each).
                var dsw = Stopwatch.StartNew();
                for (int r = 0; r < R; r++)
                {
                    session.SetGGUFDecodePastLen(statePast);
                    await session.RunDecodeStepAsync(new Dictionary<string, Tensor> { ["input_ids"] = capInput });
                    await accelerator.SynchronizeAsync();
                }
                dsw.Stop(); directMs = dsw.Elapsed.TotalMilliseconds / R;
            }
            finally { SpawnDev.ILGPU.ML.Graph.GraphExecutor.SuppressDrains = false; FusedAttentionKernel.UseStableCaptureSlots = false; capStream.Dispose(); }

            Console.WriteLine($"  state: pastLen={statePast}, input token={nextTok} '{tok.Decode(new[] { nextTok })}'");
            Console.WriteLine($"  correctness: replay token={replayArg} '{tok.Decode(new[] { replayArg })}' vs baseline={refArg} '{tok.Decode(new[] { refArg })}'  -> {(replayArg == refArg ? "MATCH" : "MISMATCH")} (max|Δlogit|={maxAbsDiff:E3})");
            Console.WriteLine($"  direct  forward : {directMs,7:F2} ms/step");
            Console.WriteLine($"  graph   replay  : {graphMs,7:F2} ms/step");
            Console.WriteLine($"  decode speedup  : {directMs / graphMs,7:F2}x   (Ollama decode ~12 ms/tok reference)");
        }
    }

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

// Decode-equivalence gate: full-recompute greedy MUST equal KV-cache greedy, token-for-token. The LFM2
// short-conv regression guard (conv-state cache correctness). Runs on the GGUF_BACKEND-selected device.
async Task<int> DecodeEquivTestAsync(string path, string prompt, int maxNew)
{
    using var context = MLContext.Create().ToContext();
    var cuda = context.GetCudaDevices(); var opencl = context.GetCLDevices();
    string? be = Environment.GetEnvironmentVariable("GGUF_BACKEND")?.ToLowerInvariant();
    Device device = be == "cpu" ? context.GetPreferredDevice(preferCPU: true)
                  : be == "opencl" && opencl.Count > 0 ? (Device)opencl[0]
                  : cuda.Count > 0 ? (Device)cuda[0]
                  : opencl.Count > 0 ? (Device)opencl[0]
                  : context.GetPreferredDevice(preferCPU: false);
    using var accelerator = device.CreateAccelerator(context);

    await using var hs = File.OpenRead(path);
    var gm = await SpawnDev.ILGPU.ML.GGUF.GGUFParser.ParseHeaderAsync(hs); gm.SourceStream = hs;
    var tok = SpawnDev.ILGPU.ML.Preprocessing.SentencePieceTokenizer.FromGGUF(gm)!;
    using var session = await InferenceSession.CreateFromGGUFFileAsync(accelerator, path);

    var promptIds = new List<int>();
    if (tok.BosId >= 0) promptIds.Add(tok.BosId);
    promptIds.AddRange(tok.Encode(prompt));
    Console.WriteLine($"Accelerator: {accelerator.Name} ({accelerator.AcceleratorType})");
    Console.WriteLine($"Prompt ids ({promptIds.Count}): [{string.Join(",", promptIds)}]  budget={maxNew}\n");

    int Argmax(float[] h, int off, int vocab) { int a = 0; float b = h[off]; for (int v = 1; v < vocab; v++) if (h[off + v] > b) { b = h[off + v]; a = v; } return a; }

    // ── Path A: full-recompute greedy (zero-pad conv; the O(n^2) reference) ──
    var full = new List<int>();
    for (int step = 0; step < maxNew; step++)
    {
        var cur = promptIds.Concat(full).Select(i => (float)i).ToArray();
        using var inBuf = accelerator.Allocate1D(cur);
        var outs = await session.RunAsync(new Dictionary<string, Tensor> { ["input_ids"] = new Tensor(inBuf.View, new[] { 1, cur.Length }, "input_ids") });
        await accelerator.SynchronizeAsync();
        var lg = outs.TryGetValue("logits", out var l) ? l : outs.Values.First();
        int vocab = lg.Shape[^1]; int seqOut = lg.ElementCount / vocab;
        var host = new float[lg.ElementCount]; lg.Data.CopyToCPU(host);
        full.Add(Argmax(host, (seqOut - 1) * vocab, vocab));
    }

    // ── Path B: KV-cache greedy (conv-STATE cache; the O(n) production path) ──
    int nLayers = (int)gm.BlockCount, nH = (int)gm.AttentionHeadCount;
    int defNKV = (int)gm.AttentionHeadCountKV; if (defNKV == 0) defNKV = nH;
    int embd = (int)gm.EmbeddingLength, defHd = embd / nH;
    var kvHeadsArr = new int[nLayers]; var hdArr = new int[nLayers];
    for (int L = 0; L < nLayers; L++) { var c = GGUFGraphBuilder.GetLayerAttnConfig(gm, L, nH, defNKV, defHd); kvHeadsArr[L] = c.NKVHeads; hdArr[L] = c.HeadDim; }
    using var kv = new GGUFDecodeKVCache(accelerator, kvHeadsArr, hdArr, maxSeqLen: promptIds.Count + maxNew + 8);
    session.EnableGGUFDecode(kv);
    session.CacheShapeReadbacks = true;
    var kvGen = new List<int>();
    int[] stepIds = promptIds.ToArray();
    for (int step = 0; step < maxNew; step++)
    {
        var idf = stepIds.Select(i => (float)i).ToArray();
        using var inBuf = accelerator.Allocate1D(idf);
        var outs = await session.RunDecodeStepAsync(new Dictionary<string, Tensor> { ["input_ids"] = new Tensor(inBuf.View, new[] { 1, stepIds.Length }, "input_ids") });
        await accelerator.SynchronizeAsync();
        var lg = outs.TryGetValue("logits", out var l) ? l : outs.Values.First();
        int vocab = lg.Shape[^1]; int seqOut = lg.ElementCount / vocab;
        var host = new float[lg.ElementCount]; lg.Data.CopyToCPU(host);
        int arg = Argmax(host, (seqOut - 1) * vocab, vocab);
        kvGen.Add(arg); stepIds = new[] { arg };
    }

    // ── Compare ──
    int firstDiv = -1; int n = Math.Min(full.Count, kvGen.Count);
    for (int i = 0; i < n; i++) if (full[i] != kvGen[i]) { firstDiv = i; break; }
    Console.WriteLine($"full-recompute: {tok.Decode(full.ToArray())}");
    Console.WriteLine($"kv-decode     : {tok.Decode(kvGen.ToArray())}");
    if (firstDiv < 0 && full.Count == kvGen.Count)
    {
        Console.WriteLine($"\nDECODE-EQUIVALENCE: PASS ({full.Count} tokens identical on {accelerator.AcceleratorType})");
        return 0;
    }
    Console.WriteLine($"\nDECODE-EQUIVALENCE: FAIL (first divergence at index {firstDiv})");
    Console.WriteLine($"  full=[{string.Join(",", full)}]");
    Console.WriteLine($"  kv  =[{string.Join(",", kvGen)}]");
    return 3;
}

async Task<int> RunAsync(string path, int[] ids)
{
    // The application owns the accelerator (library code never disposes it). Prefer CUDA.
    // GGUF_BACKEND=cpu|opencl|cuda forces a backend (for CPU-vs-GPU kernel-codegen bisection).
    using var context = MLContext.Create().ToContext();
    var cuda = context.GetCudaDevices();
    var opencl = context.GetCLDevices();
    string? beSel = Environment.GetEnvironmentVariable("GGUF_BACKEND")?.ToLowerInvariant();
    Device device = beSel == "cpu" ? context.GetPreferredDevice(preferCPU: true)
                  : beSel == "opencl" && opencl.Count > 0 ? (Device)opencl[0]
                  : beSel == "cuda" && cuda.Count > 0 ? (Device)cuda[0]
                  : cuda.Count > 0 ? (Device)cuda[0]
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
        // 40000 covers the widest layer-0/1 tensor (sc_bcx = 5 tokens * 3*2048 = 30720) so the LAST
        // position is captured in full for numerical diff against an external (numpy) reference.
        SpawnDev.ILGPU.ML.Graph.GraphExecutor.CaptureMaxElements = 40000;
    }

    sw.Restart();
    var outputs = await session.RunAsync(new Dictionary<string, Tensor> { ["input_ids"] = input });
    await accelerator.SynchronizeAsync();
    sw.Stop();

    if (capture)
    {
        var caps = SpawnDev.ILGPU.ML.Graph.GraphExecutor.CapturedOutputs!;
        int sq = ids.Length;
        // hdim = embed_out width (arch-agnostic: derive from the embedding output, not a hard-coded gemma dim).
        int hdim = 0;
        foreach (var kv in caps) if (kv.Key.EndsWith("embed_out") && sq > 0) { hdim = kv.Value.Length / sq; break; }
        if (hdim <= 0) hdim = sq > 0 ? caps.Values.First().Length / sq : caps.Values.First().Length;
        // GGUF_CAPTURE_ALL=1 → dump EVERY captured node (full trajectory, incl. shortconv sc_bcx/sc_y/sc_out).
        bool capAll = Environment.GetEnvironmentVariable("GGUF_CAPTURE_ALL") == "1";
        // RMS/absMax of a slice; and per-position RMS to test differentiation.
        (double rms, double amax) Stat(float[] v, int off, int n)
        {
            double s2 = 0, am = 0; int c = 0;
            for (int i = off; i < off + n && i < v.Length; i++) { s2 += (double)v[i] * v[i]; am = Math.Max(am, Math.Abs(v[i])); c++; }
            return (c > 0 ? Math.Sqrt(s2 / c) : 0, am);
        }
        Console.WriteLine($"\n=== RESIDUAL-STREAM TRAJECTORY (hdim={hdim}, seq={sq}; pos-RMS = RMS of each position's hidden) ===");
        var keys = caps.Keys.ToList();
        foreach (var key in keys)
        {
            string name = key.Substring(key.IndexOf('_', 4) + 1);
            bool interesting = capAll || name == "embed_out" || name == "final_norm_out" || name.Contains("logits")
                || name.EndsWith("_attn_merged")
                || System.Text.RegularExpressions.Regex.IsMatch(name, @"^scaled_out|^blk[\._](0|1|5|6|11|23|24|47)[\._]");
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
            // GGUF_CAPTURE_VALS=1 → also print RMS + first-8 values of the LAST position (matches an
            // external numpy reference dump that reports position sq-1). width = per-position stride.
            string vals = "";
            if (Environment.GetEnvironmentVariable("GGUF_CAPTURE_VALS") == "1" && nPos >= 1)
            {
                int width = v.Length / nPos;                 // hidden=hdim; sc_bcx=3*hdim; etc.
                int off = (nPos - 1) * width;                // last captured position
                var (lrms, _) = Stat(v, off, width);
                var head = new List<string>();
                for (int i = off; i < off + 8 && i < v.Length; i++) head.Add(v[i].ToString("F4"));
                vals = $"\n        last-pos(w={width}) rms={lrms:F4} first8=[{string.Join(", ", head)}]";
            }
            Console.WriteLine($"  {key,-44} rms={rms,8:F3} absMax={amax,10:F3}{cos}{vals}");
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
