using System.IO;
using System.Net;
using System.Net.Http;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    /// <summary>
    /// Browser WASM streaming can stop early (read==0 before Content-Length bytes arrive).
    /// Returning a PREFIX used to ship a truncated .tflite that then exploded in FlatBufferReader
    /// as IndexOutOfRangeException on https://lostbeard.github.io/SpawnDev.ILGPU.ML/face (2026-09-27).
    /// Incomplete stream MUST fall through to ReadAsByteArrayAsync and return the FULL body.
    /// </summary>
    [TestMethod(Timeout = 30000)]
    public async Task DownloadBytesChunked_IncompleteStream_FallsBackToFullBody() => await RunTest(async accelerator =>
    {
        _ = accelerator;
        var full = new byte[8192];
        for (int i = 0; i < full.Length; i++) full[i] = (byte)(i & 0xFF);

        var handler = new IncompleteThenFullHandler(full);
        using var http = new HttpClient(handler) { BaseAddress = new Uri("http://test.local/") };
        var got = await InferenceSession.DownloadBytesChunkedAsync(http, "http://test.local/model.bin");

        if (got.Length != full.Length)
            throw new Exception($"expected full {full.Length} bytes, got {got.Length}");
        for (int i = 0; i < full.Length; i++)
            if (got[i] != full[i])
                throw new Exception($"byte mismatch at {i}: got {got[i]}, expected {full[i]}");
        // RED-CHECK: if the truncated prefix is returned, CallCount stays 1 (no fallback GET).
        if (handler.CallCount < 2)
            throw new Exception(
                $"fallback never ran (GETs={handler.CallCount}) - truncated Content-Length prefix was accepted");
        Console.WriteLine(
            $"[DownloadChunked] incomplete stream recovered via fallback ({got.Length} bytes, {handler.CallCount} GETs)");
    });

    /// <summary>
    /// Fallback that is ALSO short of Content-Length must THROW — not hand a truncated byte[] to TFLite parse.
    /// </summary>
    [TestMethod(Timeout = 30000)]
    public async Task DownloadBytesChunked_FallbackAlsoShort_Throws() => await RunTest(async accelerator =>
    {
        _ = accelerator;
        var full = new byte[8192];
        for (int i = 0; i < full.Length; i++) full[i] = (byte)(i & 0xFF);

        var handler = new AlwaysShortHandler(full);
        using var http = new HttpClient(handler) { BaseAddress = new Uri("http://test.local/") };
        try
        {
            await InferenceSession.DownloadBytesChunkedAsync(http, "http://test.local/model.bin");
            throw new Exception("truncated download did not throw");
        }
        catch (InvalidDataException ex) when (ex.Message.Contains("truncated", StringComparison.OrdinalIgnoreCase))
        {
            Console.WriteLine($"[DownloadChunked] short fallback refused: {ex.Message}");
        }
    });

    /// <summary>
    /// Content-Length is a hint, never where the body ends: with Content-Encoding it is the COMPRESSED size while the stream holds decoded bytes (GitHub Pages gzips .onnx and .tflite:
    /// Bones 2026-11-25, MobileNetV2 14,246,826 bytes with Content-Length 13,120,039 loaded as a truncated model; BlazeFace 229,746 / 199,050 in September). Cross-origin the
    /// Content-Encoding header is hidden, so both cases: header visible and not. One GET, every byte, progress never past 100 and ending there.
    /// </summary>
    [TestMethod(Timeout = 60000)]
    public async Task DownloadBytesChunked_CompressedContentLength_ReturnsTheWholeBody() => await RunTest(async accelerator =>
    {
        _ = accelerator;
        var body = new byte[3_500_000];
        for (int i = 0; i < body.Length; i++) body[i] = (byte)((i * 31 + (i >> 8)) & 0xFF);
        // shorter (the usual gzip case), far shorter (the array grows many times), exact, and longer with the header (a body that compressed to MORE than it was)
        foreach (var (length, encoded) in new[] { (2_000_000L, true), (2_000_000L, false), (3_220_000L, false), (1_000L, false), (3_500_000L, false), (3_500_000L, true), (3_600_000L, true) })
        {
            var handler = new LyingLengthHandler(body, length, encoded);
            using var http = new HttpClient(handler) { BaseAddress = new Uri("http://test.local/") };
            var pcts = new List<int>();
            var got = await InferenceSession.DownloadBytesChunkedAsync(http, "http://test.local/model.bin", (_, p) => pcts.Add(p));
            string what = $"Content-Length {length}, Content-Encoding {(encoded ? "gzip" : "not visible")}";
            if (!got.AsSpan().SequenceEqual(body)) throw new Exception($"{what}: got {got.Length} bytes of {body.Length} (or they differ)");
            if (handler.CallCount != 1) throw new Exception($"{what}: {handler.CallCount} GETs, expected 1");
            if (pcts.Count == 0 || pcts[^1] != 100 || pcts.Any(p => p is < 0 or > 100)) throw new Exception($"{what}: progress {string.Join(",", pcts.TakeLast(5))}");
        }
        Console.WriteLine("[DownloadChunked] compressed Content-Length: the whole body, one GET: PASS");
    });

    /// <summary>
    /// <see cref="Hub.HttpClientModelSource"/> stores a compressed response whole (it rejected it as "truncated" because the stored bytes were not the compressed Content-Length), and still
    /// rejects a body that is really short. <see cref="Hub.ModelsHost.ProbeSizeAsync"/> does not report a compressed length as the file's size. Desktop store.
    /// </summary>
    [TestMethod(Timeout = 60000)]
    public async Task HttpClientModelSource_CompressedContentLength_StoresTheWholeFile() => await RunTest(async accelerator =>
    {
        _ = accelerator;
        if (OperatingSystem.IsBrowser()) throw new UnsupportedTestException("file store: desktop");
        var body = new byte[2_500_000];
        for (int i = 0; i < body.Length; i++) body[i] = (byte)((i * 17 + (i >> 10)) & 0xFF);
        var dir = Path.Combine(Path.GetTempPath(), "ilgpu-ml-test-encoded-" + Guid.NewGuid().ToString("N"));
        try
        {
            foreach (var (length, encoded) in new[] { (1_700_000L, true), (1_700_000L, false), (2_500_000L, false) })
            {
                var store = new SpawnDev.ILGPU.ML.Hub.FileModelStore(Path.Combine(dir, $"{length}{encoded}"));
                using var http = new HttpClient(new LyingLengthHandler(body, length, encoded));
                var source = new SpawnDev.ILGPU.ML.Hub.HttpClientModelSource(http, store);
                await using var s = await source.OpenAsync("acme/model", "weights.bin");
                var got = new MemoryStream();
                await s.CopyToAsync(got);
                if (!got.ToArray().AsSpan().SequenceEqual(body)) throw new Exception($"Content-Length {length}, encoded {encoded}: stored {got.Length} of {body.Length} bytes");
            }
            // really short: no Content-Encoding and fewer bytes than Content-Length
            {
                var store = new SpawnDev.ILGPU.ML.Hub.FileModelStore(Path.Combine(dir, "short"));
                using var http = new HttpClient(new LyingLengthHandler(body, 4_000_000, false));
                bool threw = false;
                try { await using var s = await new SpawnDev.ILGPU.ML.Hub.HttpClientModelSource(http, store).OpenAsync("acme/model", "weights.bin"); }
                catch (IOException) { threw = true; }
                if (!threw) throw new Exception("a body shorter than an unencoded Content-Length was accepted");
            }
            // a size probe must not take the compressed length
            using (var http = new HttpClient(new LyingLengthHandler(body, 1_700_000, true)))
            {
                long size = await SpawnDev.ILGPU.ML.Hub.ModelsHost.ProbeSizeAsync(http, "http://test.local/acme/model.bin");
                if (size == 1_700_000) throw new Exception("ProbeSizeAsync reported the compressed length as the file's size");
            }
            Console.WriteLine("[HttpClientModelSource] compressed Content-Length stored whole, a short body rejected, probe not fooled: PASS");
        }
        finally { try { Directory.Delete(dir, true); } catch { } }
    });

    /// <summary>
    /// The live case: GitHub Pages serves these files gzip-encoded with the compressed Content-Length (199,050 for a 229,746-byte one), refuses a CORS preflight, and hides Content-Range
    /// cross-origin. Through every browser path: the HttpClient download, the size probe, the OPFS downloader with and without ranges (one file inside the first 512 KB segment, one past it), FetchRangeStream
    /// and HttpRangeStream. The plain download runs FIRST, so Chrome's cache holds the encoded entry that broke every later range of the same URL. Real network.
    /// </summary>
    [TestMethod(Timeout = 180000)]
    public async Task Download_GzippedByGitHubPages_IsWhole() => await RunTest(async accelerator =>
    {
        if (!OperatingSystem.IsBrowser()) throw new UnsupportedTestException("browser network test: desktop HttpClient does not ask for gzip");
        if (accelerator.AcceleratorType != AcceleratorType.WebGPU) throw new UnsupportedTestException("network test: runs once, on the WebGPU lane");
        var js = RequireBrowserRuntime();
        var http = GetHttpClient() ?? throw new UnsupportedTestException("HttpClient not available");
        using var cache = new SpawnDev.ILGPU.ML.Hub.OpfsModelCache(js) { CacheDirectoryName = "ilgpu-ml-test-gzip" };
        if (!await cache.IsAvailableAsync()) throw new UnsupportedTestException("OPFS unavailable in this context");
        try
        {
            foreach (var (path, size) in new[] { ("models/blaze-face/model.tflite", 229_746), ("references/vad/silero_vad.onnx", 643_854) })
            {
                string url = "https://lostbeard.github.io/SpawnDev.ILGPU.ML/" + path;
                var local = await http.GetByteArrayAsync(path);        // the same file, from this page
                if (local.Length != size) throw new UnsupportedTestException($"the bundled copy of {path} is {local.Length} bytes, expected {size}");
                void Same(byte[] got, string what) { if (!got.AsSpan().SequenceEqual(local)) throw new Exception($"{path}, {what}: {got.Length} of {size} bytes (or they differ)"); }

                Same(await InferenceSession.DownloadBytesChunkedAsync(http, url), "DownloadBytesChunkedAsync");
                long probed = await SpawnDev.ILGPU.ML.Hub.ModelsHost.ProbeSizeAsync(http, url);
                if (probed != size) throw new Exception($"{path}: ProbeSizeAsync said {probed}, the file is {size}");
                foreach (bool segmented in new[] { true, false })
                {
                    cache.Downloader.UseSegmentedDownload = segmented;
                    string key = $"gzip-{segmented}-{size}";
                    await cache.RemoveAsync(key);
                    await using var s = await cache.Downloader.GetOrDownloadAsync(url, key);
                    var buf = new MemoryStream();
                    await s.CopyToAsync(buf);
                    Same(buf.ToArray(), $"OPFS download (segmented {segmented})");
                }
                await using (var fs = new SpawnDev.ILGPU.ML.Hub.FetchRangeStream(url, size) { ReadAhead = 100_000, Depth = 3 })
                {
                    var got = new byte[size];
                    for (int at = 0; at < size;) { int n = await fs.ReadAsync(got.AsMemory(at, Math.Min(70_000, size - at))); if (n == 0) break; at += n; }
                    Same(got, "FetchRangeStream");
                }
                {
                    var hs = new SpawnDev.ILGPU.ML.Hub.HttpRangeStream(http, url, size);
                    var got = new byte[size];
                    for (int at = 0; at < size;) { int n = await hs.ReadAsync(got.AsMemory(at, Math.Min(90_000, size - at))); if (n == 0) break; at += n; }
                    Same(got, "HttpRangeStream");
                }
            }
        }
        finally { await cache.ClearAllAsync(); }
        Console.WriteLine("[Download] files GitHub Pages gzips arrive whole on every browser path: PASS");
    });

    /// <summary>Every GET / HEAD: the whole body in the stream, a Content-Length that is not its length, optionally Content-Encoding: gzip (the body is not compressed: what a browser hands back is decoded).</summary>
    private sealed class LyingLengthHandler(byte[] body, long contentLength, bool encoded) : HttpMessageHandler
    {
        public int CallCount;
        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            CallCount++;
            var content = new StreamContent(new MemoryStream(body));
            content.Headers.ContentLength = contentLength;
            if (encoded) content.Headers.ContentEncoding.Add("gzip");
            return Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK) { Content = content });
        }
    }

    /// <summary>Every GET: Content-Length = full, body = first half then EOF.</summary>
    private sealed class AlwaysShortHandler : HttpMessageHandler
    {
        private readonly byte[] _full;
        public AlwaysShortHandler(byte[] full) => _full = full;

        protected override Task<HttpResponseMessage> SendAsync(
            HttpRequestMessage request, CancellationToken cancellationToken)
        {
            var half = new byte[_full.Length / 2];
            Buffer.BlockCopy(_full, 0, half, 0, half.Length);
            var content = new StreamContent(new MemoryStream(half));
            content.Headers.ContentLength = _full.Length;
            return Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK) { Content = content });
        }
    }

    /// <summary>First GET: Content-Length = full, body = first half then EOF. Second GET: full body.</summary>
    private sealed class IncompleteThenFullHandler : HttpMessageHandler
    {
        private readonly byte[] _full;
        public int CallCount;
        public IncompleteThenFullHandler(byte[] full) => _full = full;

        protected override Task<HttpResponseMessage> SendAsync(
            HttpRequestMessage request, CancellationToken cancellationToken)
        {
            CallCount++;
            if (CallCount == 1)
            {
                var half = new byte[_full.Length / 2];
                Buffer.BlockCopy(_full, 0, half, 0, half.Length);
                var content = new StreamContent(new MemoryStream(half));
                content.Headers.ContentLength = _full.Length;
                return Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK) { Content = content });
            }

            var fullContent = new ByteArrayContent(_full);
            fullContent.Headers.ContentLength = _full.Length;
            return Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK) { Content = fullContent });
        }
    }

    /// <summary>
    /// Test TFLite model loading and compilation via InferenceSession.
    /// Uses the BlazeFace model if available.
    /// </summary>
    [TestMethod(Timeout = 60000)]
    public async Task TFLite_CreateSession_BlazeFace() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient();
        if (http == null)
            throw new UnsupportedTestException("HttpClient not available for this backend");

        // Try to load BlazeFace TFLite model
        try
        {
            using var session = await InferenceSession.CreateFromTFLiteAsync(
                accelerator, http, "models/blaze-face/model.tflite");

            Console.WriteLine($"[TFLite] Session: {session}");
            Console.WriteLine($"[TFLite] Inputs: {string.Join(", ", session.InputNames)}");
            Console.WriteLine($"[TFLite] Outputs: {string.Join(", ", session.OutputNames)}");
            Console.WriteLine($"[TFLite] Nodes: {session.NodeCount}");
            Console.WriteLine($"[TFLite] Weights: {session.WeightCount}");
            Console.WriteLine($"[TFLite] Operators: {string.Join(", ", session.OperatorTypes)}");

            if (session.NodeCount == 0)
                throw new Exception("TFLite session has 0 nodes");

            Console.WriteLine("[TFLite] PASS — session created from .tflite");
        }
        catch (HttpRequestException)
        {
            throw new UnsupportedTestException("BlazeFace model not available at models/blaze-face/model.tflite");
        }
    });

    /// <summary>
    /// Test auto-format detection — the same API loads both ONNX and TFLite.
    /// </summary>
    /// <remarks>
    /// ⚠️ This test used to wrap BOTH loads in <c>catch (HttpRequestException)</c> that only logged, then
    /// print "PASS — format auto-detection works" unconditionally. It asserted NOTHING: if both fetches
    /// 404'd it passed having loaded no model at all, and even on success it never checked that a session
    /// came back usable. Both files ARE served from the demo's wwwroot (verified 2026-08-30:
    /// models/squeezenet/model.onnx and models/blaze-face/model.tflite), so a fetch failure is a real
    /// regression in serving or paths - exactly what this test should catch - not a reason to pass.
    /// A genuinely absent model now reports as a visible Skip via <see cref="UnsupportedTestException"/>,
    /// matching how <c>LoadTFLite_BlazeFace</c> above handles it.
    /// </remarks>
    [TestMethod(Timeout = 60000)]
    public async Task AutoDetect_LoadOnnxAndTFLite() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient();
        if (http == null)
            throw new UnsupportedTestException("HttpClient not available for this backend");

        // Load ONNX via auto-detect
        InferenceSession onnxSession;
        try
        {
            onnxSession = await InferenceSession.CreateFromFileAsync(
                accelerator, http, "models/squeezenet/model.onnx");
        }
        catch (HttpRequestException ex)
        {
            throw new UnsupportedTestException(
                $"models/squeezenet/model.onnx not served by this lane ({ex.Message})");
        }
        using (onnxSession)
        {
            Console.WriteLine($"[AutoDetect] ONNX: {onnxSession.ModelName}, {onnxSession.NodeCount} nodes");
            if (onnxSession.NodeCount <= 0)
                throw new Exception($"ONNX auto-detect produced an empty graph: NodeCount={onnxSession.NodeCount}");
        }

        // Load TFLite via auto-detect — the point of the test is that this is the SAME call.
        InferenceSession tfliteSession;
        try
        {
            tfliteSession = await InferenceSession.CreateFromFileAsync(
                accelerator, http, "models/blaze-face/model.tflite");
        }
        catch (HttpRequestException ex)
        {
            throw new UnsupportedTestException(
                $"models/blaze-face/model.tflite not served by this lane ({ex.Message})");
        }
        using (tfliteSession)
        {
            Console.WriteLine($"[AutoDetect] TFLite: {tfliteSession.ModelName}, {tfliteSession.NodeCount} nodes");
            if (tfliteSession.NodeCount <= 0)
                throw new Exception($"TFLite auto-detect produced an empty graph: NodeCount={tfliteSession.NodeCount}");
        }

        Console.WriteLine("[AutoDetect] PASS — both formats loaded through one API, each with a non-empty graph");
    });

    /// <summary>
    /// Test format detection from magic bytes.
    /// </summary>
    [TestMethod]
    public async Task FormatDetection_MagicBytes() => await RunTest(async accelerator =>
    {
        // ONNX: starts with protobuf field tag
        var onnxLike = new byte[] { 0x08, 0x07, 0x12, 0x04, 0x6F, 0x6E, 0x6E, 0x78 }; // "onnx" at offset 4
        if (InferenceSession.DetectModelFormat(onnxLike) != ModelFormat.ONNX)
            throw new Exception("Failed to detect ONNX");

        // GGUF: starts with "GGUF"
        var ggufLike = new byte[] { 0x47, 0x47, 0x55, 0x46, 0x03, 0x00, 0x00, 0x00 };
        if (InferenceSession.DetectModelFormat(ggufLike) != ModelFormat.GGUF)
            throw new Exception("Failed to detect GGUF");

        // TFLite: "TFL3" at offset 4
        var tfliteLike = new byte[] { 0x00, 0x00, 0x00, 0x00, 0x54, 0x46, 0x4C, 0x33 };
        if (InferenceSession.DetectModelFormat(tfliteLike) != ModelFormat.TFLite)
            throw new Exception("Failed to detect TFLite");

        Console.WriteLine("[FormatDetect] All 3 formats detected correctly");
        await Task.CompletedTask;
    });
}
