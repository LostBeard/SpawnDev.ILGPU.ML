// Face detection (YuNet) and recognition (SFace) on OUR engine against OpenCV's reference, on the six public-domain portraits in
// SpawnDev.ILGPU.ML.Demo/wwwroot/test-images/faces: agreement with OpenCV, the same-person / different-person cosine similarities, and the time per call.
//
//   dotnet run --project tools/face-bench -c Release -- <facesDir> <modelDir> [--cuda]
//     <facesDir>  wwwroot/test-images/faces (the *.bgra pictures and faces-fixture.json)
//     <modelDir>  a folder with face_detection_yunet_2023mar.onnx and face_recognition_sface_2021dec.onnx
//     --cuda      use the preferred accelerator (the GPU); the default is the CPU accelerator, so a bench never lands on a GPU somebody else is using
using System.Diagnostics;
using System.Text.Json;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.Pipelines;

var positional = args.Where(a => !a.StartsWith("--")).ToArray();
if (positional.Length < 2) { Console.Error.WriteLine("usage: <facesDir> <modelDir> [--cuda]"); return 2; }
var facesDir = positional[0]; var modelDir = positional[1];

var builder = MLContext.Create();
await builder.AllAcceleratorsAsync();
var context = builder.ToContext();
Accelerator accelerator = args.Contains("--cuda")
    ? await context.CreatePreferredAcceleratorAsync() ?? throw new Exception("no accelerator")
    : MLContext.CreateCpuAccelerator(context);
Console.WriteLine($"device : {accelerator.AcceleratorType} {accelerator.Name}");

async Task<InferenceSession> Load(string file, Dictionary<string, int[]> shapes)
{
    var stream = new FileStream(Path.Combine(modelDir, file), FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 20, FileOptions.Asynchronous);
    return await InferenceSession.CreateFromOnnxStreamAsync(accelerator, stream, inputShapes: shapes);
}
using var detector = new YuNetFaceDetector(await Load("face_detection_yunet_2023mar.onnx", YuNetFaceDetector.InputShapes()), accelerator);
using var recognizer = new SFaceRecognizer(await Load("face_recognition_sface_2021dec.onnx", new() { ["data"] = new[] { 1, 3, 112, 112 } }), accelerator);

using var doc = JsonDocument.Parse(File.ReadAllText(Path.Combine(facesDir, "faces-fixture.json")));
float[] Floats(JsonElement e) => e.EnumerateArray().Select(v => v.GetSingle()).ToArray();
var ours = new Dictionary<string, float[]>();
var detectMs = new List<double>(); var embedMs = new List<double>();
foreach (var p in doc.RootElement.GetProperty("images").EnumerateObject())
{
    var v = p.Value;
    int w = v.GetProperty("width").GetInt32(), h = v.GetProperty("height").GetInt32();
    var bgra = File.ReadAllBytes(Path.Combine(facesDir, p.Name + ".bgra"));
    var box = Floats(v.GetProperty("box")); var landmarks = Floats(v.GetProperty("landmarks"));
    var aligned = Convert.FromBase64String(v.GetProperty("aligned_bgr_b64").GetString()!);
    var reference = Floats(v.GetProperty("embedding"));

    var mine = SFaceRecognizer.AlignCrop(bgra, w, h, landmarks);
    double diff = 0; for (var i = 0; i < mine.Length; i++) diff += Math.Abs(mine[i] - aligned[i]);
    var embOnTheirCrop = await recognizer.EmbedAlignedAsync(aligned);

    await detector.DetectAsync(bgra, w, h);   // warm
    var clock = Stopwatch.StartNew();
    var faces = await detector.DetectAsync(bgra, w, h);
    detectMs.Add(clock.Elapsed.TotalMilliseconds);
    var f = faces[0];
    var lm = 0f; for (var i = 0; i < 10; i++) lm = Math.Max(lm, Math.Abs(f.Landmarks[i] - landmarks[i]) / box[2]);
    clock.Restart();
    var emb = await recognizer.EmbedAsync(bgra, w, h, f);
    embedMs.Add(clock.Elapsed.TotalMilliseconds);
    ours[p.Name] = emb;
    Console.WriteLine($"{p.Name,-12} align |d| {diff / mine.Length:0.00} | embedding vs OpenCV's on its crop {SFaceRecognizer.Cosine(embOnTheirCrop, reference):0.00000} | {faces.Count} face, score {f.Score:0.00}, IoU {IouOf(f, box):0.00}, landmark {lm:P1} of box | whole chain vs OpenCV's embedding {SFaceRecognizer.Cosine(emb, reference):0.000}");
}
static float IouOf(FaceDetection a, float[] b)
{
    var x1 = Math.Max(a.X, b[0]); var y1 = Math.Max(a.Y, b[1]); var x2 = Math.Min(a.X + a.Width, b[0] + b[2]); var y2 = Math.Min(a.Y + a.Height, b[1] + b[3]);
    var inter = Math.Max(0, x2 - x1) * Math.Max(0, y2 - y1);
    return inter / (a.Width * a.Height + b[2] * b[3] - inter);
}

string Person(string n) => n.Split('_')[0];
var names = ours.Keys.ToList();
var same = new List<(string, float)>(); var different = new List<(string, float)>();
for (var i = 0; i < names.Count; i++)
    for (var j = i + 1; j < names.Count; j++)
    {
        var c = SFaceRecognizer.Cosine(ours[names[i]], ours[names[j]]);
        (Person(names[i]) == Person(names[j]) ? same : different).Add(($"{names[i]} / {names[j]}", c));
    }
Console.WriteLine();
foreach (var (pair, c) in same) Console.WriteLine($"same person     {c:0.000}  {pair}");
foreach (var (pair, c) in different.OrderByDescending(x => x.Item2).Take(4)) Console.WriteLine($"different people {c:0.000}  {pair}");
Console.WriteLine($"OpenCV threshold {SFaceRecognizer.OpenCvCosineThreshold}; lowest same {same.Min(x => x.Item2):0.000}, highest different {different.Max(x => x.Item2):0.000}");
Console.WriteLine($"time: detect median {detectMs.Order().ElementAt(detectMs.Count / 2):0.0} ms, embed (align + network) median {embedMs.Order().ElementAt(embedMs.Count / 2):0.0} ms");
return 0;
