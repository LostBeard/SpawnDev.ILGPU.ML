// ─────────────────────────────────────────────────────────────────────────────────────────────────
//  SpawnDev.ILGPU.ML — Example 05: Gemma 4 12B multimodal CHAT (text + image/audio/video → text)
//
//  Gemma 4 12B is the ENCODER-FREE "Unified" model: raw image patches and audio frames project DIRECTLY
//  into the LLM embedding space via the companion "mmproj" GGUF (no SigLIP / Conformer towers). All of the
//  multimodal plumbing — preprocessing, the vision/audio projection, the RAW media-embedding splice, and the
//  inputs_embeds KV-cache decode — lives in the library's Gemma4MultimodalPipeline / Gemma4Chat. This example
//  decodes the media (ImageSharp / a WAV reader — the consumer's job) and drives an interactive chat.
//
//  Run (loads the model, then drops into a chat session until /exit):
//    dotnet run                                         # interactive multimodal chat
//    dotnet run -- C:\path\text.gguf C:\path\mmproj.gguf
//
//  In-chat commands:
//    /image <path>[,<path2>]   attach image(s) to your NEXT message (multiple = video frames)
//    /audio <path>             attach 16 kHz mono WAV to your next message
//    /reset                    start a fresh conversation (clears KV cache)
//    /help                     show commands
//    /exit  (or /quit)         leave
//
//  One-shot (scripts/CI), bypasses the chat loop:
//    GEMMA4_GEN="Describe this." GEMMA4_IMAGE=cat.jpg dotnet run
// ─────────────────────────────────────────────────────────────────────────────────────────────────

using System.Diagnostics;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using ILGPU.Runtime.OpenCL;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.Kernels;
using SpawnDev.ILGPU.ML.Pipelines;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

string textModel = ArgPath(0) ?? @"D:\users\tj\Projects\gemma4-12b-Q4_K_M.gguf";
string mmprojModel = ArgPath(1) ?? @"D:\users\tj\Projects\mmproj-gemma-4-12B-it-bf16.gguf";

if (!File.Exists(textModel)) { Console.Error.WriteLine($"text model not found: {textModel}"); return 1; }
if (!File.Exists(mmprojModel)) { Console.Error.WriteLine($"mmproj not found: {mmprojModel}"); return 1; }

bool thinking = (Environment.GetEnvironmentVariable("GEMMA4_THINK") ?? "1") != "0";
int maxNew = int.TryParse(Environment.GetEnvironmentVariable("GEMMA4_GEN_N"), out var nn) ? nn : 256;

// ── Accelerator: prefer a FAST desktop backend (CUDA, then OpenCL), per the fast-backends-first rule ──
using var context = MLContext.Create().ToContext();
Device device = PickDevice(context);
using var accelerator = device.CreateAccelerator(context);
Console.WriteLine($"Accelerator: {accelerator.Name} ({accelerator.AcceleratorType})");

var sw = Stopwatch.StartNew();
using var pipe = await Gemma4MultimodalPipeline.CreateAsync(accelerator, textModel, mmprojModel);
Console.WriteLine($"Loaded gemma4 in {sw.Elapsed.TotalSeconds:F1}s — vision={pipe.SupportsImages} audio={pipe.SupportsAudio}, thinking={thinking}.");

// ── One-shot mode (GEMMA4_GEN): kept for scripts/CI — generate once and exit ──
string? oneShot = Environment.GetEnvironmentVariable("GEMMA4_GEN");
if (!string.IsNullOrEmpty(oneShot))
{
    try
    {
        var imgs = LoadImages(SplitPaths("GEMMA4_IMAGE"));
        var auds = LoadAudio(SplitPaths("GEMMA4_AUDIO"));
        Console.WriteLine($"\nyou> {oneShot}  (+{imgs.Count} image, {auds.Count} audio)\ngemma4> ");
        var shown1 = "";
        string ans = await pipe.GenerateAsync(oneShot, images: imgs.Count > 0 ? imgs : null, audio: auds.Count > 0 ? auds : null,
            maxNewTokens: maxNew, thinking: thinking, onToken: (n, t) => { Stream(ref shown1, t); return Task.CompletedTask; });
        Console.WriteLine();
        return 0;
    }
    catch (Exception ex) { Console.Error.WriteLine($"GEN FAILED: {ex.GetType().Name}: {ex.Message}\n{ex.StackTrace}"); return 1; }
}

// ── Interactive chat loop ──
PrintHelp();
var chat = pipe.StartChat(thinking);
var pendingImages = new List<ImageInput>();
var pendingAudio = new List<float[]>();

while (true)
{
    Console.Write("\nyou> ");
    string? line = Console.ReadLine();
    if (line == null) break;                 // EOF (Ctrl+Z / piped input ended)
    line = line.Trim();
    if (line.Length == 0) continue;

    if (line is "/exit" or "/quit") break;
    if (line is "/help" or "/?") { PrintHelp(); continue; }
    if (line is "/reset")
    {
        chat = pipe.StartChat(thinking);
        pendingImages.Clear(); pendingAudio.Clear();
        Console.WriteLine("[new conversation — KV cache cleared]");
        continue;
    }
    if (line.StartsWith("/image ", StringComparison.OrdinalIgnoreCase))
    {
        try { var imgs = LoadImages(line[7..].Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)); pendingImages.AddRange(imgs); Console.WriteLine($"[attached {imgs.Count} image(s); {pendingImages.Count} queued for next message]"); }
        catch (Exception ex) { Console.Error.WriteLine($"[image error: {ex.Message}]"); }
        continue;
    }
    if (line.StartsWith("/audio ", StringComparison.OrdinalIgnoreCase))
    {
        try { var auds = LoadAudio(line[7..].Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)); pendingAudio.AddRange(auds); Console.WriteLine($"[attached {auds.Count} audio; {pendingAudio.Count} queued for next message]"); }
        catch (Exception ex) { Console.Error.WriteLine($"[audio error: {ex.Message}]"); }
        continue;
    }
    if (line.StartsWith("/")) { Console.WriteLine($"[unknown command {line.Split(' ')[0]} — /help for the list]"); continue; }

    // Send the message (with any queued media) and stream the reply.
    var imAttach = pendingImages.Count > 0 ? new List<ImageInput>(pendingImages) : null;
    var auAttach = pendingAudio.Count > 0 ? new List<float[]>(pendingAudio) : null;
    pendingImages.Clear(); pendingAudio.Clear();

    Console.Write("gemma4> ");
    var turnSw = Stopwatch.StartNew();
    string shown = "";
    try
    {
        await chat.SendAsync(line, images: imAttach, audio: auAttach, maxNewTokens: maxNew,
            onToken: (n, textSoFar) => { Stream(ref shown, textSoFar); return Task.CompletedTask; });
        Console.WriteLine($"\n  [{turnSw.Elapsed.TotalSeconds:F1}s]");
    }
    catch (Exception ex) { Console.Error.WriteLine($"\n[generation error: {ex.GetType().Name}: {ex.Message}]"); }
}
Console.WriteLine("bye 🖖");
return 0;

// ── helpers ───────────────────────────────────────────────────────────────────────────────────────

// Stream the incremental decoded text (textSoFar is cumulative each token). Prints the new suffix in the
// common case; control markers (<turn|>/<eos>) are dropped so the chat reads clean.
static void Stream(ref string shown, string textSoFar)
{
    if (textSoFar.Length <= shown.Length || !textSoFar.StartsWith(shown, StringComparison.Ordinal)) return;
    var delta = textSoFar.Substring(shown.Length);
    shown = textSoFar;
    // Drop gemma4 control markers; turn the thinking-channel open/close into readable labels.
    delta = delta.Replace("<|channel>thought", "[thinking] ").Replace("<|channel>", "[thinking] ")
                 .Replace("<channel|>", "\n[answer] ")
                 .Replace("<turn|>", "").Replace("<|turn>", "").Replace("<eos>", "").Replace("<|think|>", "");
    if (delta.Length > 0) Console.Write(delta);
}

static void PrintHelp() => Console.WriteLine(
    "\nGemma 4 multimodal chat. Type a message and press Enter.\n" +
    "  /image <path>[,<path2>]   attach image(s) to your next message (multiple = video frames)\n" +
    "  /audio <path>             attach a 16 kHz mono WAV to your next message\n" +
    "  /reset                    start a fresh conversation\n" +
    "  /help                     show this\n" +
    "  /exit                     quit");

static Device PickDevice(Context context)
{
    var cuda = context.GetCudaDevices();
    if (cuda.Count > 0) return cuda[0];
    var cl = context.GetCLDevices();
    if (cl.Count > 0) return cl[0];
    return context.GetPreferredDevice(preferCPU: false);
}

static List<ImageInput> LoadImages(IEnumerable<string> paths)
{
    var list = new List<ImageInput>();
    foreach (var ip in paths)
    {
        if (!File.Exists(ip)) throw new FileNotFoundException($"image not found: {ip}");
        using var img = Image.Load<Rgb24>(ip);
        var rgb = new byte[(long)img.Width * img.Height * 3];
        img.CopyPixelDataTo(rgb);
        list.Add(new ImageInput(rgb, img.Width, img.Height));
        Console.WriteLine($"  image: {ip} ({img.Width}x{img.Height})");
    }
    return list;
}

static List<float[]> LoadAudio(IEnumerable<string> paths)
{
    var list = new List<float[]>();
    foreach (var ap in paths)
    {
        if (!File.Exists(ap)) throw new FileNotFoundException($"audio not found: {ap}");
        list.Add(ReadWavMono16k(ap));
        Console.WriteLine($"  audio: {ap}");
    }
    return list;
}

static IEnumerable<string> SplitPaths(string envVar)
{
    var v = Environment.GetEnvironmentVariable(envVar);
    if (string.IsNullOrEmpty(v)) yield break;
    foreach (var p in v.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
        yield return p;
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
