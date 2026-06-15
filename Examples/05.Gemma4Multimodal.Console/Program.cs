// ─────────────────────────────────────────────────────────────────────────────────────────────────
//  SpawnDev.ILGPU.ML — Example 05: Gemma 4 12B multimodal (text + image/audio/video → text)
//
//  Gemma 4 12B is the ENCODER-FREE "Unified" model: raw image patches and raw audio frames are
//  projected DIRECTLY into the LLM embedding space by the lightweight linear layers in the companion
//  "mmproj" GGUF — no SigLIP / Conformer towers. This example loads the text decoder GGUF + the mmproj
//  and runs full multimodal generation.
//
//    dotnet run -- <text.gguf> <mmproj.gguf> [--image path] [--audio path] "your prompt"
//    dotnet run                                  # uses the default local model paths
//
//  Build order (see Plans/gemma4-multimodal-bringup.md): [1] mmproj load (this step, verified) →
//  vision projector → audio projector → splice + generate.
// ─────────────────────────────────────────────────────────────────────────────────────────────────

using SpawnDev.ILGPU.ML.GGUF;

string textModel = ArgPath(0) ?? @"D:\users\tj\Projects\gemma4-12b-Q4_K_M.gguf";
string mmprojModel = ArgPath(1) ?? @"D:\users\tj\Projects\mmproj-gemma-4-12B-it-bf16.gguf";

if (!File.Exists(mmprojModel)) { Console.Error.WriteLine($"mmproj not found: {mmprojModel}"); return 1; }

// ── Step 1: load + inspect the multimodal projector (verifies the loader + bf16 dequant) ──────────
Console.WriteLine($"Loading mmproj: {mmprojModel}");
var mm = MmprojModel.Load(mmprojModel);

Console.WriteLine($"\n=== gemma4 mmproj ===");
Console.WriteLine($"  vision encoder : {mm.HasVisionEncoder}  ({mm.VisionProjectorType})");
Console.WriteLine($"  audio  encoder : {mm.HasAudioEncoder}  ({mm.AudioProjectorType})");
Console.WriteLine($"  image size     : {mm.VisionImageSize}   patch {mm.VisionPatchSize} (effective 48 via n_merge=3)");
Console.WriteLine($"  vision embed/proj : {mm.VisionEmbeddingLength} / {mm.VisionProjectionDim}");
Console.WriteLine($"  image mean/std : [{string.Join(",", mm.VisionImageMean)}] / [{string.Join(",", mm.VisionImageStd)}]");
Console.WriteLine($"  audio frame/proj : {mm.AudioFrameLength} / {mm.AudioProjectionDim}");

Console.WriteLine($"\n=== tensors ({mm.Gguf.Tensors.Length}) ===");
foreach (var t in mm.Gguf.Tensors.OrderBy(t => t.Name, StringComparer.Ordinal))
    Console.WriteLine($"  {t.Name,-32} [{string.Join(", ", t.Shape)}]  {t.Type}");

// Dequant the two bf16 projection weights and sanity-check the values are finite + reasonable.
// A wrong bf16 widening would show NaN/Inf or absurd magnitudes here.
foreach (var name in new[] { "mm.input_projection.weight", "mm.a.input_projection.weight" })
{
    var w = mm.GetTensorF32(name);
    if (w == null) { Console.WriteLine($"\n  {name}: ABSENT"); continue; }
    double mn = w[0], mx = w[0], sum = 0; int nan = 0;
    foreach (var x in w) { if (float.IsNaN(x) || float.IsInfinity(x)) nan++; else { mn = Math.Min(mn, x); mx = Math.Max(mx, x); sum += x; } }
    Console.WriteLine($"\n  {name} (bf16→f32): n={w.Length}  min={mn:F4} max={mx:F4} mean={sum / w.Length:F6}  nan/inf={nan}  first=[{string.Join(", ", w.Take(5).Select(v => v.ToString("F4")))}]");
}

// ── Step 2: smoke-test the projector forward on the real weights (dummy inputs) ───────────────────
// Confirms the production EncodeImage/EncodeAudio paths run end-to-end and emit finite [N,3840] vectors.
// Correctness vs the llama.cpp mtmd oracle comes once the image/audio preprocessing lands.
var proj = new SpawnDev.ILGPU.ML.Multimodal.Gemma4MultimodalProjector(mm);
Console.WriteLine($"\n=== projector smoke (EmbedDim={proj.EmbedDim} PatchLen={proj.PatchLen} AudioFrameLen={proj.AudioFrameLen} PosTableLen={proj.PosTableLen}) ===");

var rng = new Random(1);
int gridC = 2, gridR = 2, nPatches = gridC * gridR;
var dummyPatches = new float[nPatches * proj.PatchLen];
for (int i = 0; i < dummyPatches.Length; i++) dummyPatches[i] = (float)rng.NextDouble();   // /255-range stand-in
var imgEmb = proj.EncodeImage(dummyPatches, nPatches, gridC, gridR);
ReportEmb("vision", imgEmb, nPatches, proj.EmbedDim);

int nFrames = 3;
var dummyAudio = new float[nFrames * proj.AudioFrameLen];
for (int i = 0; i < dummyAudio.Length; i++) dummyAudio[i] = (float)(rng.NextDouble() * 2 - 1);  // [-1,1] PCM stand-in
var audEmb = proj.EncodeAudio(dummyAudio, nFrames);
ReportEmb("audio", audEmb, nFrames, proj.EmbedDim);

Console.WriteLine("\n[OK] mmproj loaded + projector forward runs (correctness vs oracle pending preprocessing).");
return 0;

static void ReportEmb(string tag, float[] emb, int n, int d)
{
    double mn = emb[0], mx = emb[0], sum = 0; int bad = 0;
    foreach (var v in emb) { if (float.IsNaN(v) || float.IsInfinity(v)) bad++; else { mn = Math.Min(mn, v); mx = Math.Max(mx, v); sum += v; } }
    Console.WriteLine($"  {tag}: [{n},{d}] min={mn:F4} max={mx:F4} mean={sum / emb.Length:F6} nan/inf={bad}  row0=[{string.Join(", ", emb.Take(5).Select(v => v.ToString("F4")))}]");
}

static string? ArgPath(int idx)
{
    var paths = Environment.GetCommandLineArgs().Skip(1).Where(a => a.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase)).ToArray();
    return idx < paths.Length ? paths[idx] : null;
}
