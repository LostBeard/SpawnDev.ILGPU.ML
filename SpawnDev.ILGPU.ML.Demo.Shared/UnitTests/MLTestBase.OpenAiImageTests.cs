using System.Net;
using System.Text;
using System.Text.Json;
using SpawnDev.ILGPU.ML.Assets;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// <see cref="OpenAiImageSource"/> against a fake HTTP server: the request it sends (endpoint, bearer key, JSON body) and how it reads the answer and the errors. The real service is not called
/// (it needs a key and bills per picture), so the exact wire format is checked against the documented one, not against OpenAI.
/// </summary>
public abstract partial class MLTestBase
{
    private static void Oa_Expect(bool ok, string what) { if (!ok) throw new Exception(what); }

    private sealed class Oa_Handler(Func<HttpRequestMessage, string, (HttpStatusCode, string)> reply) : HttpMessageHandler
    {
        public readonly List<(string Url, string? Auth, string Body)> Requests = new();
        protected override async Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken ct)
        {
            var body = request.Content == null ? "" : await request.Content.ReadAsStringAsync(ct);
            Requests.Add((request.RequestUri!.ToString(), request.Headers.Authorization?.ToString(), body));
            var (code, json) = reply(request, body);
            return new HttpResponseMessage(code) { Content = new StringContent(json, Encoding.UTF8, "application/json") };
        }
    }

    [TestMethod]
    public async Task OpenAiImageSource_SendsTheDocumentedRequest_AndDecodesThePicture() => await RunPureTest(async () =>
    {
        // a 4 x 2 RGBA picture, red left half, blue right half
        var rgba = new byte[4 * 2 * 4];
        for (int y = 0; y < 2; y++) for (int x = 0; x < 4; x++) { int o = (y * 4 + x) * 4; rgba[o] = (byte)(x < 2 ? 255 : 0); rgba[o + 2] = (byte)(x < 2 ? 0 : 255); rgba[o + 3] = 255; }
        var png = AssetGenerator.EncodePng(rgba, 4, 2);
        var handler = new Oa_Handler((_, _) => (HttpStatusCode.OK, JsonSerializer.Serialize(new { data = new[] { new { b64_json = Convert.ToBase64String(png) } } })));
        using var http = new HttpClient(handler);
        using var src = new OpenAiImageSource("sk-test-key", "dall-e-3", "1024x1024", http);
        var pic = await src.DrawAsync("a knight", 7);
        Oa_Expect(pic.Seed == 7 && pic.Width == 4 && pic.Height == 2 && pic.Rgba.Length == rgba.Length, $"the picture decodes ({pic.Width}x{pic.Height}, seed {pic.Seed})");
        Oa_Expect(pic.Rgba[0] == 255 && pic.Rgba[2] == 0 && pic.Rgba[4 * 3 + 2] == 255, "its pixels survive the round trip");
        var (url, auth, body) = handler.Requests.Single();
        Oa_Expect(url == "https://api.openai.com/v1/images/generations", $"the images endpoint ({url})");
        Oa_Expect(auth == "Bearer sk-test-key", "the key travels only as the bearer token");
        using var doc = JsonDocument.Parse(body);
        var r = doc.RootElement;
        Oa_Expect(r.GetProperty("model").GetString() == "dall-e-3" && r.GetProperty("prompt").GetString() == "a knight" && r.GetProperty("n").GetInt32() == 1 && r.GetProperty("size").GetString() == "1024x1024", $"model, prompt, n = 1 and size ({body})");
        Oa_Expect(r.GetProperty("response_format").GetString() == "b64_json", "DALL-E 3 is asked for base64 (no second download)");

        // gpt-image-1 always answers base64 and rejects response_format
        var h2 = new Oa_Handler((_, _) => (HttpStatusCode.OK, JsonSerializer.Serialize(new { data = new[] { new { b64_json = Convert.ToBase64String(png) } } })));
        using var http2 = new HttpClient(h2);
        using var src2 = new OpenAiImageSource("k", "gpt-image-1", "1024x1024", http2);
        await src2.DrawAsync("x", 0);
        Oa_Expect(!h2.Requests[0].Body.Contains("response_format"), "gpt-image-1 is not sent response_format");
        await Task.CompletedTask;
    });

    [TestMethod]
    public async Task OpenAiImageSource_ReportsTheServicesError_WithoutTheKey() => await RunPureTest(async () =>
    {
        var handler = new Oa_Handler((_, _) => (HttpStatusCode.BadRequest, JsonSerializer.Serialize(new { error = new { message = "Your request was rejected by the safety system." } })));
        using var http = new HttpClient(handler);
        using var src = new OpenAiImageSource("sk-secret-123", "dall-e-3", "1024x1024", http);
        try { await src.DrawAsync("x", 0); throw new Exception("an error response must throw"); }
        catch (HttpRequestException e)
        {
            Oa_Expect(e.Message.Contains("400") && e.Message.Contains("safety system"), $"the status and the service's message are reported ({e.Message})");
            Oa_Expect(!e.Message.Contains("sk-secret-123"), "the key never appears in the error");
        }
        try { _ = new OpenAiImageSource(" "); throw new Exception("an empty key must be refused"); }
        catch (ArgumentException) { }
        await Task.CompletedTask;
    });
}
