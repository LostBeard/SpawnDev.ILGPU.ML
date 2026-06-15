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
using SpawnDev.ILGPU.ML.Multimodal;
using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnDev.ILGPU.ML.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

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
    string? imagePath = Environment.GetEnvironmentVariable("GEMMA4_IMAGE");
    string? audioPath = Environment.GetEnvironmentVariable("GEMMA4_AUDIO");
    try { return await GenerateAsync(textModel, mmprojModel, genPrompt, imagePath, audioPath, maxNew); }
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

// ── multimodal generation through the decoder's inputs_embeds entry ───────────────────────────────
// Assemble the prompt embedding sequence host-side: TEXT rows gathered from token_embd × sqrt(n_embd);
// IMAGE rows = the RAW projected patch embeddings (gemma4 splices media unscaled), placed between the
// <|image> / <image|> marker tokens at the start of the user content (mtmd layout). Then greedy KV decode.
// (Text-only, no image: this is also the equivalence test — it reproduces the exact input_ids "...Paris.")
async Task<int> GenerateAsync(string textPath, string mmprojPath, string prompt, string? imagePath, string? audioPath, int maxNew)
{
    await using var hs = File.OpenRead(textPath);
    var gm = await GGUFParser.ParseHeaderAsync(hs);
    gm.SourceStream = hs;
    var tok = SentencePieceTokenizer.FromGGUF(gm)!;
    int Id(string s) => tok.TryGetId(s, out var v) ? v : -1;
    int turnO = Id("<|turn>"), turnC = Id("<turn|>"), eos = Id("<eos>"), bos = Id("<bos>");
    int nEmbd = (int)gm.EmbeddingLength;
    float embScale = MathF.Sqrt(nEmbd);
    var tokenEmbd = gm.Tensors.First(t => t.Name == "token_embd.weight");

    float[] ScaledRow(int t)
    {
        var r = gm.GetTensorRowFloat32(tokenEmbd, t) ?? throw new Exception($"no embed row for token {t}");
        var o = new float[nEmbd];
        for (int e = 0; e < nEmbd; e++) o[e] = r[e] * embScale;
        return o;
    }

    // ── image → raw projected embeddings (if provided) ──
    float[]? imageEmb = null; int nImg = 0, imgCols = 0, imgRows = 0;
    if (!string.IsNullOrEmpty(imagePath))
    {
        if (!File.Exists(imagePath)) { Console.Error.WriteLine($"image not found: {imagePath}"); return 1; }
        using var img = Image.Load<Rgb24>(imagePath);
        int iw = img.Width, ih = img.Height;
        var rgb = new byte[(long)iw * ih * 3];
        img.CopyPixelDataTo(rgb);
        var (patches, nCols, nRows) = Gemma4ImagePreprocessor.Preprocess(rgb, iw, ih);
        imgCols = nCols; imgRows = nRows; nImg = nCols * nRows;
        var mmProj = MmprojModel.Load(mmprojPath);
        var projector = new Gemma4MultimodalProjector(mmProj);
        imageEmb = projector.EncodeImage(patches, nImg, nCols, nRows);   // raw [nImg, nEmbd]
        Console.WriteLine($"Image: {imagePath} {iw}x{ih} -> grid {nCols}x{nRows} = {nImg} tokens (raw embeddings)");
    }

    // ── audio → raw projected embeddings (if provided) ──
    float[]? audioEmb = null; int nAud = 0;
    if (!string.IsNullOrEmpty(audioPath))
    {
        if (!File.Exists(audioPath)) { Console.Error.WriteLine($"audio not found: {audioPath}"); return 1; }
        var (samples, sr) = ReadWavMono16k(audioPath);
        var (frames, nFrames) = Gemma4AudioPreprocessor.Frame(samples);
        nAud = nFrames;
        var mmProj = MmprojModel.Load(mmprojPath);
        var projector = new Gemma4MultimodalProjector(mmProj);
        audioEmb = projector.EncodeAudio(frames, nFrames);   // raw [nFrames, nEmbd]
        Console.WriteLine($"Audio: {audioPath} {samples.Length} samples @ {sr}Hz -> {nFrames} frames/tokens (raw embeddings)");
    }

    // ── assemble the prefill embedding sequence (mirrors ChatTemplates.BuildGemma4PromptTokens + media block) ──
    var rows = new List<float[]>();
    void Text(string s) { foreach (var t in tok.Encode(s)) rows.Add(ScaledRow(t)); }
    void Control(int t) => rows.Add(ScaledRow(t));
    if (bos >= 0) Control(bos);
    Control(turnO); Text("system\n"); Control(Id("<|think|>")); Text("\n"); Control(turnC); Text("\n");
    Control(turnO); Text("user\n");
    if (imageEmb != null)
    {
        int imgBegin = Id("<|image>"), imgEnd = Id("<image|>");
        if (imgBegin < 0 || imgEnd < 0) throw new Exception("gemma4 image marker tokens <|image>/<image|> not in vocab.");
        Control(imgBegin);
        for (int p = 0; p < nImg; p++) { var o = new float[nEmbd]; Array.Copy(imageEmb, (long)p * nEmbd, o, 0, nEmbd); rows.Add(o); }
        Control(imgEnd);
    }
    if (audioEmb != null)
    {
        int audBegin = Id("<|audio>"), audEnd = Id("<audio|>");
        if (audBegin < 0 || audEnd < 0) throw new Exception("gemma4 audio marker tokens <|audio>/<audio|> not in vocab.");
        Control(audBegin);
        for (int p = 0; p < nAud; p++) { var o = new float[nEmbd]; Array.Copy(audioEmb, (long)p * nEmbd, o, 0, nEmbd); rows.Add(o); }
        Control(audEnd);
    }
    Text(prompt); Control(turnC); Text("\n");
    Control(turnO); Text("model\n");

    int prefillSeq = rows.Count;
    var prefill = new float[(long)prefillSeq * nEmbd];
    for (int i = 0; i < prefillSeq; i++) Array.Copy(rows[i], 0, prefill, (long)i * nEmbd, nEmbd);
    Console.WriteLine($"Prompt: \"{prompt}\"  prefill={prefillSeq} tokens ({nImg} image, {nAud} audio)  n_embd={nEmbd} embScale={embScale:F3}\n");

    using var context = MLContext.Create().ToContext();
    var cuda = context.GetCudaDevices();
    Device device = cuda.Count > 0 ? (Device)cuda[0] : context.GetPreferredDevice(preferCPU: false);
    using var accelerator = device.CreateAccelerator(context);
    Console.WriteLine($"Accelerator: {accelerator.Name}");

    using var session = await InferenceSession.CreateFromGGUFFileAsync(accelerator, textPath, acceptInputsEmbeds: true);
    Console.WriteLine($"Loaded: {session}\n");

    int nLayers = (int)gm.BlockCount, nH = (int)gm.AttentionHeadCount;
    int defNKV = (int)gm.AttentionHeadCountKV; if (defNKV == 0) defNKV = nH;
    int defHd = nEmbd / nH;
    var kvHeads = new int[nLayers]; var hdArr = new int[nLayers];
    for (int L = 0; L < nLayers; L++)
    { var cfg = GGUFGraphBuilder.GetLayerAttnConfig(gm, L, nH, defNKV, defHd); kvHeads[L] = cfg.NKVHeads; hdArr[L] = cfg.HeadDim; }
    int maxSeq = prefillSeq + maxNew + 8;
    using var kv = new GGUFDecodeKVCache(accelerator, kvHeads, hdArr, maxSeqLen: maxSeq);
    session.EnableGGUFDecode(kv);
    Console.WriteLine($"[KV-cache decode] {nLayers} layers, maxSeq={maxSeq}");

    var gen = new List<int>();
    var sw = Stopwatch.StartNew();
    float[] stepEmb = prefill; int stepSeq = prefillSeq;   // prefill, then 1 token/step
    for (int step = 0; step < maxNew; step++)
    {
        using var inBuf = accelerator.Allocate1D(stepEmb);
        var input = new Tensor(inBuf.View, new[] { 1, stepSeq, nEmbd }, "inputs_embeds");
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
        Console.WriteLine($"  step {step,2}: token {arg,7} '{tok.Decode(new[] { arg })}' (logit {best:F3})  seq={stepSeq} wall={stepSw.Elapsed.TotalMilliseconds:F0}ms");
        if (arg == turnC || arg == eos) { Console.WriteLine("  [stop token]"); break; }
        stepEmb = ScaledRow(arg); stepSeq = 1;
    }
    sw.Stop();
    Console.WriteLine($"\n=== GENERATED via inputs_embeds ({gen.Count} tokens, {sw.Elapsed.TotalSeconds:F1}s) ===");
    Console.WriteLine(tok.Decode(gen.ToArray()));
    return 0;
}

// Minimal WAV reader: PCM16 (or float32) mono @ 16 kHz → float[] in [-1,1]. Scans RIFF chunks for fmt/data
// (robust to extra chunks). gemma4ua needs 16 kHz mono raw waveform; errors clearly on other formats.
static (float[] samples, int sampleRate) ReadWavMono16k(string path)
{
    var b = File.ReadAllBytes(path);
    if (b.Length < 12 || b[0] != 'R' || b[1] != 'I' || b[2] != 'F' || b[3] != 'F' || b[8] != 'W' || b[9] != 'A' || b[10] != 'V' || b[11] != 'E')
        throw new Exception($"{path} is not a RIFF/WAVE file.");
    int fmt = 0, channels = 0, sampleRate = 0, bits = 0; int dataOff = -1, dataLen = 0;
    int pos = 12;
    while (pos + 8 <= b.Length)
    {
        string id = System.Text.Encoding.ASCII.GetString(b, pos, 4);
        int sz = BitConverter.ToInt32(b, pos + 4);
        int body = pos + 8;
        if (id == "fmt ")
        {
            fmt = BitConverter.ToInt16(b, body); channels = BitConverter.ToInt16(b, body + 2);
            sampleRate = BitConverter.ToInt32(b, body + 4); bits = BitConverter.ToInt16(b, body + 14);
        }
        else if (id == "data") { dataOff = body; dataLen = Math.Min(sz, b.Length - body); }
        pos = body + sz + (sz & 1); // chunks are word-aligned
    }
    if (dataOff < 0) throw new Exception($"{path}: no data chunk.");
    if (channels != 1 || sampleRate != Gemma4AudioPreprocessor.SampleRate)
        throw new Exception($"{path}: need mono 16 kHz (got {channels}ch {sampleRate}Hz). Resample first.");
    if (fmt == 1 && bits == 16)
    {
        int n = dataLen / 2; var s = new float[n];
        for (int i = 0; i < n; i++) s[i] = BitConverter.ToInt16(b, dataOff + i * 2) / 32768f;
        return (s, sampleRate);
    }
    if (fmt == 3 && bits == 32)
    {
        int n = dataLen / 4; var s = new float[n];
        for (int i = 0; i < n; i++) s[i] = BitConverter.ToSingle(b, dataOff + i * 4);
        return (s, sampleRate);
    }
    throw new Exception($"{path}: unsupported WAV format fmt={fmt} bits={bits} (need PCM16 or float32).");
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
