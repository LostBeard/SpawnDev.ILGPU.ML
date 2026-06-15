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
Console.WriteLine($"  audio frame/embed/proj : {mm.AudioFrameLength} / {mm.AudioEmbeddingLength} / {mm.AudioProjectionDim}");

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

Console.WriteLine("\n[OK] mmproj loaded + bf16 projections dequantized.");
return 0;

static string? ArgPath(int idx)
{
    var paths = Environment.GetCommandLineArgs().Skip(1).Where(a => a.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase)).ToArray();
    return idx < paths.Length ? paths[idx] : null;
}
