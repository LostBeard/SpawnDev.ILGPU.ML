using System.Net;
using System.Net.Http;
using System.Net.Http.Headers;
using SpawnDev.ILGPU.ML.Hub;

namespace SpawnDev.ILGPU.ML.Onnx;

public static partial class ModelInspectorHelper
{
    /// <summary>
    /// Inspect a model at a URL, fetching only its structure: the file is read with HTTP byte ranges and every weight blob is
    /// seeked past, so a 623 MB ONNX inspects from a few MB over the wire.
    /// </summary>
    /// <remarks>
    /// Prefer this to <see cref="InspectAsync(Stream, CancellationToken)"/> with <c>HttpClient.GetStreamAsync</c>: a response
    /// stream cannot seek, so the stream overload has to READ every weight byte to get past it - in the browser that is the
    /// whole model pushed through the .NET heap in 64 KB chunks (GPT-2: over 120 s, versus 0.6 s for its structure).
    /// </remarks>
    public static async Task<InspectionResult> InspectAsync(HttpClient http, string url, CancellationToken ct = default)
    {
        await using var stream = await OpenUrlForInspectionAsync(http, url, ct).ConfigureAwait(false);
        return await InspectAsync(stream, ct).ConfigureAwait(false);
    }

    /// <summary>Operator compatibility of the ONNX model at a URL, fetching only its structure (see <see cref="InspectAsync(HttpClient, string, CancellationToken)"/>).</summary>
    public static async Task<CompatibilityResult> CheckCompatibilityAsync(
        HttpClient http, string url, Operators.OperatorRegistry? registry = null, CancellationToken ct = default)
    {
        await using var stream = await OpenUrlForInspectionAsync(http, url, ct).ConfigureAwait(false);
        return await CheckCompatibilityAsync(stream, registry, ct).ConfigureAwait(false);
    }

    /// <summary>Inspection and compatibility of the model at a URL from one structure-only pass (see <see cref="InspectAsync(HttpClient, string, CancellationToken)"/>).</summary>
    public static async Task<(InspectionResult Inspection, CompatibilityResult Compatibility)> InspectWithCompatibilityAsync(
        HttpClient http, string url, Operators.OperatorRegistry? registry = null, CancellationToken ct = default)
    {
        await using var stream = await OpenUrlForInspectionAsync(http, url, ct).ConfigureAwait(false);
        return await InspectWithCompatibilityAsync(stream, registry, ct).ConfigureAwait(false);
    }

    /// <summary>
    /// A stream over <paramref name="url"/> for inspection: a seekable <see cref="HttpRangeStream"/> when the server honours
    /// byte ranges, otherwise the plain response stream.
    /// </summary>
    /// <remarks>
    /// <see cref="HttpRangeStream"/> on every platform, browser included, as <see cref="HubModelSource.OpenForInspectionAsync"/>
    /// does: it requests exactly the bytes of each read. <see cref="FetchRangeStream"/> is built for LOADING (16 MB chunks, four
    /// in flight) and restarts that prefetch at every seek past a weight blob: GPT-2 fetched 470 MB of 653 MB through it. The
    /// structure bytes enter .NET either way, because the protobuf reader parses them there.
    /// <para>
    /// The probe is a 0-0 range GET and only a <c>206</c> with a <c>Content-Range</c> total counts: a server that ignores
    /// <c>Range</c> answers <c>200</c> with the whole body, and a ranged stream over it would download the file once per read.
    /// Without ranges the only way past a weight blob is to read it, so that case is still correct, only not structure-only.
    /// </para>
    /// </remarks>
    public static async Task<Stream> OpenUrlForInspectionAsync(HttpClient http, string url, CancellationToken ct = default)
    {
        if (http == null) throw new ArgumentNullException(nameof(http));
        if (url == null) throw new ArgumentNullException(nameof(url));
        using (var req = new HttpRequestMessage(HttpMethod.Get, url))
        {
            req.Headers.Range = new RangeHeaderValue(0, 0);
            using var res = await ModelsHost.SendAsync(http, req, HttpCompletionOption.ResponseHeadersRead, ct).ConfigureAwait(false);
            if (res.StatusCode == HttpStatusCode.PartialContent && res.Content.Headers.ContentRange?.Length is long length && length > 0)
                return new HttpRangeStream(http, url, length);
        }
        return await http.GetStreamAsync(url, ct).ConfigureAwait(false);
    }
}
