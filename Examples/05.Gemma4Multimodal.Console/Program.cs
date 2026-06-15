// ─────────────────────────────────────────────────────────────────────────────────────────────────
//  SpawnDev.ILGPU.ML — Example 05: Gemma 4 12B multimodal (text + image/audio/video → text)
//
//  Gemma 4 12B is the ENCODER-FREE "Unified" model: raw image patches and raw audio frames are
//  projected DIRECTLY into the LLM embedding space by the lightweight linear layers in the companion
//  "mmproj" GGUF — no SigLIP / Conformer towers. This example loads the text decoder GGUF + the mmproj
//  and runs full multimodal generation via the decoder's `inputs_embeds` entry (text rows gathered+scaled
//  host-side; media rows = RAW projected embeddings).
//
//    dotnet run -- <text.gguf> <mmproj.gguf>            # load + projector smoke
//    GEMMA4_GEN="your prompt" dotnet run                # text generation via inputs_embeds (equivalence test)
//    GEMMA4_GEN="describe this" GEMMA4_IMAGE=pic.png dotnet run   # (image path — once preprocessing lands)
//
//  Build order (Plans/gemma4-multimodal-bringup.md): mmproj load ✔ → projector forward ✔ →
//  inputs_embeds gen (this step) → image/audio preprocessing → splice → E2E.
// ─────────────────────────────────────────────────────────────────────────────────────────────────

using System.Diagnostics;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.GGUF;
using SpawnDev.ILGPU.ML.Kernels;
using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnDev.ILGPU.ML.Tensors;

string textModel = ArgPath(0) ?? @"D:\users\tj\Projects\gemma4-12b-Q4_K_M.gguf";
string mmprojModel = ArgPath(1) ?? @"D:\users\tj\Projects\mmproj-gemma-4-12B-it-bf16.gguf";

// GEMMA4_PROBE=1 → fast diagnostic: gather a few token_embd rows host-side, print stats + cross-token
// distinctness (no model load — header + stream reads only). Validates GetTensorRowFloat32.
if (Environment.GetEnvironmentVariable("GEMMA4_PROBE") == "1")
{
    await using var phs = File.OpenRead(textModel);
    var pgm = await GGUFParser.ParseHeaderAsync(phs); pgm.SourceStream = phs;
    var te = pgm.Tensors.First(t => t.Name == "token_embd.weight");
    Console.WriteLine($"token_embd: {te.Type} dims=[{string.Join(",", te.Dimensions)}] dataOff={pgm.GetTensorDataOffset(te)}");
    int nE = (int)pgm.EmbeddingLength;
    float[]? prev = null;
    // Each token must gather a DISTINCT, nonzero embedding row (regression guard for the subnormal-fp16
    // bug: GGUFModel.HalfToFloat once flushed subnormal Q6_K scales to zero → all-zero rows for many tokens).
    foreach (int t in new[] { 2, 105, 1458, 9079, 7001 })
    {
        var row = pgm.GetTensorRowFloat32(te, t)!;
        double s2 = 0, mn = row[0], mx = row[0]; foreach (var v in row) { s2 += (double)v * v; mn = Math.Min(mn, v); mx = Math.Max(mx, v); }
        double diff = prev == null ? -1 : Enumerable.Range(0, nE).Sum(i => Math.Abs((double)row[i] - prev[i])) / nE;
        Console.WriteLine($"  tok {t,6}: rms={Math.Sqrt(s2 / nE):F5} min={mn:F4} max={mx:F4} distinctFromPrev={diff:F5}");
        prev = row;
    }
    return 0;
}

// GEMMA4_GEN="<prompt>" → run generation (text via inputs_embeds; media once preprocessing lands).
string? genPrompt = Environment.GetEnvironmentVariable("GEMMA4_GEN");
if (!string.IsNullOrEmpty(genPrompt))
{
    int maxNew = int.TryParse(Environment.GetEnvironmentVariable("GEMMA4_GEN_N"), out var nn) ? nn : 32;
    try { return await GenerateAsync(textModel, genPrompt, maxNew); }
    catch (Exception ex) { Console.Error.WriteLine($"GEN FAILED: {ex.GetType().Name}: {ex.Message}\n{ex.StackTrace}"); return 1; }
}

// Default: load + inspect the mmproj + projector smoke (verifies the loader + bf16 dequant + forward).
if (!File.Exists(mmprojModel)) { Console.Error.WriteLine($"mmproj not found: {mmprojModel}"); return 1; }
Console.WriteLine($"Loading mmproj: {mmprojModel}");
var mm = MmprojModel.Load(mmprojModel);
Console.WriteLine($"\n=== gemma4 mmproj ===");
Console.WriteLine($"  vision encoder : {mm.HasVisionEncoder}  ({mm.VisionProjectorType})");
Console.WriteLine($"  audio  encoder : {mm.HasAudioEncoder}  ({mm.AudioProjectorType})");
Console.WriteLine($"  image size     : {mm.VisionImageSize}   patch {mm.VisionPatchSize} (effective 48 via n_merge=3)");
Console.WriteLine($"  audio frame/proj : {mm.AudioFrameLength} / {mm.AudioProjectionDim}");

var proj = new SpawnDev.ILGPU.ML.Multimodal.Gemma4MultimodalProjector(mm);
Console.WriteLine($"\n=== projector smoke (EmbedDim={proj.EmbedDim} PatchLen={proj.PatchLen} AudioFrameLen={proj.AudioFrameLen} PosTableLen={proj.PosTableLen}) ===");
var rng = new Random(1);
int gridC = 2, gridR = 2, nPatches = gridC * gridR;
var dummyPatches = new float[nPatches * proj.PatchLen];
for (int i = 0; i < dummyPatches.Length; i++) dummyPatches[i] = (float)rng.NextDouble();
ReportEmb("vision", proj.EncodeImage(dummyPatches, nPatches, gridC, gridR), nPatches, proj.EmbedDim);
int nFrames = 3;
var dummyAudio = new float[nFrames * proj.AudioFrameLen];
for (int i = 0; i < dummyAudio.Length; i++) dummyAudio[i] = (float)(rng.NextDouble() * 2 - 1);
ReportEmb("audio", proj.EncodeAudio(dummyAudio, nFrames), nFrames, proj.EmbedDim);
Console.WriteLine("\n[OK] mmproj loaded + projector forward runs. (Run with GEMMA4_GEN=\"...\" to generate.)");
return 0;

// ── text generation through the decoder's inputs_embeds entry ─────────────────────────────────────
// Text-only here = the EQUIVALENCE TEST: gathering token_embd rows × sqrt(n_embd) host-side and feeding
// them as inputs_embeds must reproduce the exact input_ids-path output ("...Paris."). That isolates the
// inputs_embeds plumbing + row gather before media rows get spliced in.
async Task<int> GenerateAsync(string textPath, string prompt, int maxNew)
{
    // Own stream for host-side token-row gather (independent of the session, which closes its stream).
    await using var hs = File.OpenRead(textPath);
    var gm = await GGUFParser.ParseHeaderAsync(hs);
    gm.SourceStream = hs;
    var tok = SentencePieceTokenizer.FromGGUF(gm)!;
    var tokenArr = gm.GetMetadataStringArray("tokenizer.ggml.tokens")!;
    var idOf = new Dictionary<string, int>();
    for (int i = 0; i < tokenArr.Length; i++) idOf[tokenArr[i]] = i;
    int Id(string s) => idOf.TryGetValue(s, out var v) ? v : -1;
    int turnC = Id("<turn|>"), eos = Id("<eos>");

    var ids = ChatTemplates.BuildGemma4PromptTokens(tok, systemPrompt: null, userMessage: prompt, thinking: true).ToList();
    int nEmbd = (int)gm.EmbeddingLength;
    float embScale = MathF.Sqrt(nEmbd);
    var tokenEmbd = gm.Tensors.First(t => t.Name == "token_embd.weight");
    Console.WriteLine($"Prompt: \"{prompt}\"\nPrompt token ids ({ids.Count}): [{string.Join(",", ids)}]");
    Console.WriteLine($"n_embd={nEmbd} embScale={embScale:F3} token_embd={tokenEmbd.Type}\n");

    using var context = MLContext.Create().ToContext();
    var cuda = context.GetCudaDevices();
    Device device = cuda.Count > 0 ? (Device)cuda[0] : context.GetPreferredDevice(preferCPU: false);
    using var accelerator = device.CreateAccelerator(context);
    Console.WriteLine($"Accelerator: {accelerator.Name}");

    // acceptInputsEmbeds: the graph skips the token Gather + gemma scale; we supply the embedding sequence.
    using var session = await InferenceSession.CreateFromGGUFFileAsync(accelerator, textPath, acceptInputsEmbeds: true);
    Console.WriteLine($"Loaded: {session}\n");

    // KV-cache decode (per-layer head config like the text example).
    int nLayers = (int)gm.BlockCount, nH = (int)gm.AttentionHeadCount;
    int defNKV = (int)gm.AttentionHeadCountKV; if (defNKV == 0) defNKV = nH;
    int defHd = nEmbd / nH;
    var kvHeads = new int[nLayers]; var hdArr = new int[nLayers];
    for (int L = 0; L < nLayers; L++)
    { var cfg = GGUFGraphBuilder.GetLayerAttnConfig(gm, L, nH, defNKV, defHd); kvHeads[L] = cfg.NKVHeads; hdArr[L] = cfg.HeadDim; }
    using var kv = new GGUFDecodeKVCache(accelerator, kvHeads, hdArr, maxSeqLen: ids.Count + maxNew + 8);
    session.EnableGGUFDecode(kv);
    Console.WriteLine($"[KV-cache decode] {nLayers} layers, maxSeq={ids.Count + maxNew + 8}");

    // Build the [seq, n_embd] embedding block for a run of token ids: gather each row, scale by sqrt(n_embd).
    float[] EmbedTokens(IReadOnlyList<int> toks)
    {
        var buf = new float[(long)toks.Count * nEmbd];
        for (int i = 0; i < toks.Count; i++)
        {
            var row = gm.GetTensorRowFloat32(tokenEmbd, toks[i]) ?? throw new Exception($"no embed row for token {toks[i]}");
            int b = i * nEmbd;
            for (int e = 0; e < nEmbd; e++) buf[b + e] = row[e] * embScale;
        }
        return buf;
    }

    var gen = new List<int>();
    var sw = Stopwatch.StartNew();
    int[] stepIds = ids.ToArray();   // prefill = whole prompt, then 1 token/step
    for (int step = 0; step < maxNew; step++)
    {
        var emb = EmbedTokens(stepIds);
        int seq = stepIds.Length;
        using var inBuf = accelerator.Allocate1D(emb);
        var input = new Tensor(inBuf.View, new[] { 1, seq, nEmbd }, "inputs_embeds");
        var stepSw = Stopwatch.StartNew();
        var outputs = await session.RunDecodeStepAsync(new Dictionary<string, Tensor> { ["inputs_embeds"] = input });
        await accelerator.SynchronizeAsync();
        stepSw.Stop();
        var logits = outputs["logits"];
        int vocab = logits.Shape[^1];
        int seqOut = logits.ElementCount / vocab;
        var host = new float[logits.ElementCount];
        logits.Data.CopyToCPU(host);
        int last = (seqOut - 1) * vocab, arg = 0; float best = host[last];
        for (int v = 1; v < vocab; v++) if (host[last + v] > best) { best = host[last + v]; arg = v; }
        gen.Add(arg);
        Console.WriteLine($"  step {step,2}: token {arg,7} '{tok.Decode(new[] { arg })}' (logit {best:F3})  seq={seq} wall={stepSw.Elapsed.TotalMilliseconds:F0}ms");
        if (arg == turnC || arg == eos) { Console.WriteLine("  [stop token]"); break; }
        stepIds = new[] { arg };
    }
    sw.Stop();
    Console.WriteLine($"\n=== GENERATED via inputs_embeds ({gen.Count} tokens, {sw.Elapsed.TotalSeconds:F1}s) ===");
    Console.WriteLine(tok.Decode(gen.ToArray()));
    return 0;
}

static string? ArgPath(int idx)
{
    var paths = Environment.GetCommandLineArgs().Skip(1).Where(a => a.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase)).ToArray();
    return idx < paths.Length ? paths[idx] : null;
}

static void ReportEmb(string tag, float[] emb, int n, int d)
{
    double mn = emb[0], mx = emb[0], sum = 0; int bad = 0;
    foreach (var v in emb) { if (float.IsNaN(v) || float.IsInfinity(v)) bad++; else { mn = Math.Min(mn, v); mx = Math.Max(mx, v); sum += v; } }
    Console.WriteLine($"  {tag}: [{n},{d}] min={mn:F4} max={mx:F4} mean={sum / emb.Length:F6} nan/inf={bad}");
}
