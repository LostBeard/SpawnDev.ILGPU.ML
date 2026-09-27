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
