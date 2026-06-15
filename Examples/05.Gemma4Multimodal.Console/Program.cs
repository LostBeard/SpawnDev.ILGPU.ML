// ─────────────────────────────────────────────────────────────────────────────────────────────────
//  SpawnDev.ILGPU.ML — Example 05: Gemma 4 12B multimodal (text + image/audio/video → text)
//
//  Gemma 4 12B is the ENCODER-FREE "Unified" model: raw image patches and audio frames project DIRECTLY
//  into the LLM embedding space via the companion "mmproj" GGUF (no SigLIP / Conformer towers). All of the
//  multimodal plumbing — preprocessing, the vision/audio projection, the RAW media-embedding splice, and the
//  inputs_embeds KV-cache decode — lives in the library's Gemma4MultimodalPipeline. This example just decodes
//  the media (ImageSharp / a WAV reader — the consumer's job) and calls the pipeline.
//
//    GEMMA4_GEN="Describe this." GEMMA4_IMAGE=cat.jpg dotnet run
//    GEMMA4_GEN="What do you hear?" GEMMA4_AUDIO=tone.wav dotnet run
//    GEMMA4_GEN="..." GEMMA4_IMAGE=a.jpg,b.jpg dotnet run        # multiple images = video frames
//    dotnet run                                                  # no env: mmproj load + projector smoke
// ─────────────────────────────────────────────────────────────────────────────────────────────────

using System.Diagnostics;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.GGUF;
using SpawnDev.ILGPU.ML.Pipelines;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

string textModel = ArgPath(0) ?? @"D:\users\tj\Projects\gemma4-12b-Q4_K_M.gguf";
string mmprojModel = ArgPath(1) ?? @"D:\users\tj\Projects\mmproj-gemma-4-12B-it-bf16.gguf";

string? prompt = Environment.GetEnvironmentVariable("GEMMA4_GEN");
if (!string.IsNullOrEmpty(prompt))
{
    int maxNew = int.TryParse(Environment.GetEnvironmentVariable("GEMMA4_GEN_N"), out var nn) ? nn : 256;
    try { return await Generate(prompt, maxNew); }
    catch (Exception ex) { Console.Error.WriteLine($"GEN FAILED: {ex.GetType().Name}: {ex.Message}\n{ex.StackTrace}"); return 1; }
}

// No env → mmproj load + projector smoke (verifies the loader + bf16 dequant + projector forward).
if (!File.Exists(mmprojModel)) { Console.Error.WriteLine($"mmproj not found: {mmprojModel}"); return 1; }
var mm = MmprojModel.Load(mmprojModel);
var proj = new SpawnDev.ILGPU.ML.Multimodal.Gemma4MultimodalProjector(mm);
Console.WriteLine($"gemma4 mmproj: vision={mm.HasVisionEncoder}({mm.VisionProjectorType}) audio={mm.HasAudioEncoder}({mm.AudioProjectorType}) " +
    $"embed={proj.EmbedDim} patchLen={proj.PatchLen} audioFrame={proj.AudioFrameLen}");
var rng = new Random(1);
var dummy = new float[4 * proj.PatchLen]; for (int i = 0; i < dummy.Length; i++) dummy[i] = (float)rng.NextDouble();
var ie = proj.EncodeImage(dummy, 4, 2, 2);
Console.WriteLine($"projector smoke: image[4,{proj.EmbedDim}] finite={ie.All(v => !float.IsNaN(v) && !float.IsInfinity(v))}");
Console.WriteLine("[OK] Run with GEMMA4_GEN=\"...\" [GEMMA4_IMAGE=x.jpg] [GEMMA4_AUDIO=x.wav] to generate.");
return 0;

// ── generation: decode the media, then one pipeline call ──────────────────────────────────────────
async Task<int> Generate(string userPrompt, int maxNew)
{
    // Decode media (the CONSUMER's job — desktop uses ImageSharp / a WAV reader; browser would use canvas).
    var images = new List<ImageInput>();
    foreach (var ip in SplitPaths("GEMMA4_IMAGE"))
    {
        using var img = Image.Load<Rgb24>(ip);
        var rgb = new byte[(long)img.Width * img.Height * 3];
        img.CopyPixelDataTo(rgb);
        images.Add(new ImageInput(rgb, img.Width, img.Height));
        Console.WriteLine($"image: {ip} ({img.Width}x{img.Height})");
    }
    var audio = new List<float[]>();
    foreach (var ap in SplitPaths("GEMMA4_AUDIO"))
    {
        audio.Add(ReadWavMono16k(ap));
        Console.WriteLine($"audio: {ap}");
    }

    using var context = MLContext.Create().ToContext();
    var cuda = context.GetCudaDevices();
    Device device = cuda.Count > 0 ? (Device)cuda[0] : context.GetPreferredDevice(preferCPU: false);
    using var accelerator = device.CreateAccelerator(context);
    Console.WriteLine($"Accelerator: {accelerator.Name}");

    var sw = Stopwatch.StartNew();
    using var pipe = await Gemma4MultimodalPipeline.CreateAsync(accelerator, textModel, mmprojModel);
    Console.WriteLine($"Loaded in {sw.Elapsed.TotalSeconds:F1}s. Generating (prompt + {images.Count} image, {audio.Count} audio)...\n");

    string answer = await pipe.GenerateAsync(userPrompt, images: images, audio: audio, maxNewTokens: maxNew,
        onToken: (n, _) => { Console.Write($"\r  {n} tokens..."); return Task.CompletedTask; });

    Console.WriteLine($"\n\n=== gemma4 ({sw.Elapsed.TotalSeconds:F1}s) ===\n{answer}");
    return 0;
}

static IEnumerable<string> SplitPaths(string envVar)
{
    var v = Environment.GetEnvironmentVariable(envVar);
    if (string.IsNullOrEmpty(v)) yield break;
    foreach (var p in v.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
    {
        if (!File.Exists(p)) throw new FileNotFoundException($"media not found: {p}");
        yield return p;
    }
}

static string? ArgPath(int idx)
{
    var paths = Environment.GetCommandLineArgs().Skip(1).Where(a => a.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase)).ToArray();
    return idx < paths.Length ? paths[idx] : null;
}

// Minimal WAV reader: PCM16 / float32, mono, 16 kHz → float[] in [-1,1]. Scans RIFF chunks (robust to extras).
static float[] ReadWavMono16k(string path)
{
    var b = File.ReadAllBytes(path);
    if (b.Length < 12 || b[0] != 'R' || b[1] != 'I' || b[2] != 'F' || b[3] != 'F' || b[8] != 'W' || b[9] != 'A' || b[10] != 'V' || b[11] != 'E')
        throw new Exception($"{path} is not a RIFF/WAVE file.");
    int fmt = 0, channels = 0, sampleRate = 0, bits = 0, dataOff = -1, dataLen = 0, pos = 12;
    while (pos + 8 <= b.Length)
    {
        string id = System.Text.Encoding.ASCII.GetString(b, pos, 4);
        int sz = BitConverter.ToInt32(b, pos + 4), body = pos + 8;
        if (id == "fmt ") { fmt = BitConverter.ToInt16(b, body); channels = BitConverter.ToInt16(b, body + 2); sampleRate = BitConverter.ToInt32(b, body + 4); bits = BitConverter.ToInt16(b, body + 14); }
        else if (id == "data") { dataOff = body; dataLen = Math.Min(sz, b.Length - body); }
        pos = body + sz + (sz & 1);
    }
    if (dataOff < 0) throw new Exception($"{path}: no data chunk.");
    if (channels != 1 || sampleRate != 16000) throw new Exception($"{path}: need mono 16 kHz (got {channels}ch {sampleRate}Hz).");
    if (fmt == 1 && bits == 16)
    { int n = dataLen / 2; var s = new float[n]; for (int i = 0; i < n; i++) s[i] = BitConverter.ToInt16(b, dataOff + i * 2) / 32768f; return s; }
    if (fmt == 3 && bits == 32)
    { int n = dataLen / 4; var s = new float[n]; for (int i = 0; i < n; i++) s[i] = BitConverter.ToSingle(b, dataOff + i * 4); return s; }
    throw new Exception($"{path}: unsupported WAV fmt={fmt} bits={bits} (need PCM16 or float32).");
}
