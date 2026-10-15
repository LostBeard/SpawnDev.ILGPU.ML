using System.Text.Json;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// Face detection (YuNet) and recognition (SFace), held to OpenCV's own FaceDetectorYN / FaceRecognizerSF on six public-domain official portraits stored at 360 px
/// (test-images/faces; PROVENANCE: Wikimedia Commons, tools in _vision-research/oracle_faces.py): the aligned crop is OpenCV's to the pixel, the embedding of an identical crop is OpenCV's,
/// the boxes agree, and the identity separation holds - two people photographed years apart score high with themselves and low with each other.
/// </summary>
public abstract partial class MLTestBase
{
    private const string YuNetRepo = "opencv/face_detection_yunet";
    private const string YuNetFile = "face_detection_yunet_2023mar.onnx";
    private const string SFaceRepo = "opencv/face_recognition_sface";
    private const string SFaceFile = "face_recognition_sface_2021dec.onnx";

    private sealed record FaceFixtureImage(string Name, int Width, int Height, byte[] Bgra, float[] Box, float[] Landmarks, float Score, byte[] AlignedBgr, float[] Embedding);

    private static async Task<(List<FaceFixtureImage> Images, Dictionary<string, float> Cosine)> LoadFaceFixture(System.Net.Http.HttpClient http)
    {
        using var doc = JsonDocument.Parse(await http.GetStringAsync("test-images/faces/faces-fixture.json"));
        var images = new List<FaceFixtureImage>();
        foreach (var p in doc.RootElement.GetProperty("images").EnumerateObject())
        {
            var v = p.Value;
            images.Add(new FaceFixtureImage(p.Name, v.GetProperty("width").GetInt32(), v.GetProperty("height").GetInt32(),
                await http.GetByteArrayAsync($"test-images/faces/{p.Name}.bgra"),
                v.GetProperty("box").EnumerateArray().Select(e => e.GetSingle()).ToArray(),
                v.GetProperty("landmarks").EnumerateArray().Select(e => e.GetSingle()).ToArray(),
                v.GetProperty("score").GetSingle(),
                Convert.FromBase64String(v.GetProperty("aligned_bgr_b64").GetString()!),
                v.GetProperty("embedding").EnumerateArray().Select(e => e.GetSingle()).ToArray()));
        }
        var cosine = doc.RootElement.GetProperty("cosine").EnumerateObject().ToDictionary(p => p.Name, p => p.Value.GetSingle());
        return (images, cosine);
    }

    /// <summary>
    /// The alignment that feeds the recogniser is OpenCV's <c>alignCrop</c>: from OpenCV's own five landmarks our 112 x 112 crop differs from its crop by under 2.5 grey levels on
    /// average, on every portrait. (Host code only: the warp is a similarity transform and a bilinear sample.)
    /// </summary>
    [TestMethod(Timeout = 120000)]
    public async Task FaceAlign_MatchesOpenCvAlignCrop()
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");
        var (images, _) = await LoadFaceFixture(http);
        if (images.Count != 6) throw new Exception($"{images.Count} fixture images, expected 6");
        foreach (var img in images)
        {
            var ours = SFaceRecognizer.AlignCrop(img.Bgra, img.Width, img.Height, img.Landmarks);
            double sum = 0; int worst = 0;
            for (var i = 0; i < ours.Length; i++) { var d = Math.Abs(ours[i] - img.AlignedBgr[i]); sum += d; worst = Math.Max(worst, d); }
            var mean = sum / ours.Length;
            Console.WriteLine($"[FaceAlign] {img.Name}: mean |d| {mean:0.00}, worst {worst}");
            if (mean > 2.5) throw new Exception($"{img.Name}: aligned crop differs from OpenCV's by {mean:0.00} levels on average");
        }
    }

    /// <summary>
    /// SFace on OpenCV's own aligned crops gives OpenCV's embeddings (cosine >= 0.999), and YuNet finds the same face OpenCV does (box IoU >= 0.8, landmarks within 6% of the box)
    /// - then the whole chain, detect -> align -> embed, separates identities: the two photographs of a person are more alike than any two different people by a clear margin, and
    /// the OpenCV threshold of 0.363 sits between them. Desktop CPU / CUDA / OpenCL and WebGPU (two small networks: a CPU run is seconds); the 38 MB recogniser comes from the opencv Hugging Face repo.
    /// </summary>
    [TestMethod(Timeout = 600000, Category = "HeavyModel")]
    public async Task Pipeline_FaceRecognition_MatchesOpenCvAndSeparatesPeople() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is AcceleratorType.WebGL or AcceleratorType.Wasm)
            throw new UnsupportedTestException($"{accelerator.AcceleratorType}: face recognition is gated on the fast backends");
        var assets = GetHttpClient();
        if (assets == null) throw new UnsupportedTestException("HttpClient not available");
        var (images, cosine) = await LoadFaceFixture(assets);

        using var modelHttp = new System.Net.Http.HttpClient { Timeout = System.Threading.Timeout.InfiniteTimeSpan };
        var source = new HttpClientModelSource(modelHttp);
        InferenceSession detSession, recSession;
        await using (var s = await source.OpenAsync(YuNetRepo, YuNetFile))
            detSession = await InferenceSession.CreateFromOnnxStreamAsync(accelerator, s, inputShapes: YuNetFaceDetector.InputShapes());
        await using (var s = await source.OpenAsync(SFaceRepo, SFaceFile))
            recSession = await InferenceSession.CreateFromOnnxStreamAsync(accelerator, s, inputShapes: new Dictionary<string, int[]> { ["data"] = new[] { 1, 3, 112, 112 } });
        using var detector = new YuNetFaceDetector(detSession, accelerator);
        using var recognizer = new SFaceRecognizer(recSession, accelerator);

        // 1. The recogniser on OpenCV's own crops.
        foreach (var img in images)
        {
            var emb = await recognizer.EmbedAlignedAsync(img.AlignedBgr);
            var c = SFaceRecognizer.Cosine(emb, img.Embedding);
            Console.WriteLine($"[FaceRecognition] {img.Name}: embedding cosine to OpenCV's {c:0.00000}");
            if (c < 0.999f) throw new Exception($"{img.Name}: embedding cosine to OpenCV's is {c:0.00000} (expected >= 0.999) on {BackendName}");
        }

        // 2. The detector against OpenCV's boxes, then the whole chain.
        var ours = new Dictionary<string, float[]>();
        foreach (var img in images)
        {
            var faces = await detector.DetectAsync(img.Bgra, img.Width, img.Height);
            if (faces.Count == 0) throw new Exception($"{img.Name}: no face found on {BackendName}");
            var f = faces[0];
            var iou = Iou(f.X, f.Y, f.Width, f.Height, img.Box[0], img.Box[1], img.Box[2], img.Box[3]);
            var lm = 0f;
            for (var i = 0; i < 10; i++) lm = Math.Max(lm, Math.Abs(f.Landmarks[i] - img.Landmarks[i]) / img.Box[2]);
            Console.WriteLine($"[FaceDetect] {img.Name}: {faces.Count} face(s), score {f.Score:0.00}, box IoU {iou:0.00}, worst landmark {lm:P1} of the box width");
            if (iou < 0.8f) throw new Exception($"{img.Name}: box IoU {iou:0.00} with OpenCV's on {BackendName}");
            if (lm > 0.06f) throw new Exception($"{img.Name}: a landmark is {lm:P1} of the box width from OpenCV's on {BackendName}");
            ours[img.Name] = await recognizer.EmbedAsync(img.Bgra, img.Width, img.Height, f);
        }

        float Sim(string a, string b) => SFaceRecognizer.Cosine(ours[a], ours[b]);
        var same = new[] { Sim("biden_2013", "biden_2021"), Sim("trump_2017", "trump_2025") };
        var names = images.Select(i => i.Name).ToList();
        var different = new List<float>();
        for (var i = 0; i < names.Count; i++)
            for (var j = i + 1; j < names.Count; j++)
            {
                var person = (string n) => n.Split('_')[0];
                if (person(names[i]) != person(names[j])) different.Add(Sim(names[i], names[j]));
            }
        Console.WriteLine($"[FaceRecognition] same person: {string.Join(", ", same.Select(s => s.ToString("0.000")))}; different people: max {different.Max():0.000}, min {different.Min():0.000}; OpenCV threshold {SFaceRecognizer.OpenCvCosineThreshold}");
        if (same.Min() < 0.45f) throw new Exception($"the same person scored {same.Min():0.000} (expected >= 0.45) on {BackendName}");
        if (different.Max() > 0.35f) throw new Exception($"two different people scored {different.Max():0.000} (expected <= 0.35) on {BackendName}");
        if (same.Min() - different.Max() < 0.15f) throw new Exception($"identity margin {same.Min() - different.Max():0.000} is under 0.15 on {BackendName}");
    });

    private static float Iou(float ax, float ay, float aw, float ah, float bx, float by, float bw, float bh)
    {
        var x1 = Math.Max(ax, bx); var y1 = Math.Max(ay, by);
        var x2 = Math.Min(ax + aw, bx + bw); var y2 = Math.Min(ay + ah, by + bh);
        var inter = Math.Max(0, x2 - x1) * Math.Max(0, y2 - y1);
        var union = aw * ah + bw * bh - inter;
        return union <= 0 ? 0 : inter / union;
    }
}
