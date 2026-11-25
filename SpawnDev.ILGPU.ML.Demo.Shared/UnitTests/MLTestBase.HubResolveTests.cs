using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.UnitTesting;
using System.Net;
using System.Net.Sockets;
using System.Text;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    // Hugging Face answers /resolve/ with a 302 to a signed CDN URL, and every /resolve/ request is one of the Hub's "resolver" calls: 3,000 per 5 minutes per
    // client (MEASURED 2026-11-19 by Lore: ratelimit "resolvers";r=2999;t=118). The range readers sent every range there (a 27 GB checkpoint: ~1,700 a load);
    // they now resolve once and send the ranges to where the source landed. An LFS (Xet) file: small non-LFS files are served without the redirect.
    private const string ResolveRepo = "onnx-community/depth-anything-v2-small", ResolveFile = "onnx/model.onnx";
    private const string ResolveWholeFile = "onnx/model_q4f16.onnx";
    private const long ResolveWholeBytes = 19126267;

    /// <summary>
    /// <see cref="HttpRangeStream"/> against a local server shaped like Hugging Face: /resolve/ answers 302 to a signed /cdn/ URL (counted), the /cdn/ URL
    /// serves the ranges, and on demand refuses an expired signature with 403. Twenty reads must resolve ONCE (was: once per read), and a read after the
    /// signature expires must resolve again and still return the right bytes. Desktop only (the browser cannot listen); runs once, on the CPU lane.
    /// </summary>
    [TestMethod(Timeout = 120000)]
    public async Task HttpRangeStream_ResolvesARedirectOnce_AndAgainWhenItExpires() => await RunTest(async accelerator =>
    {
        if (OperatingSystem.IsBrowser()) throw new UnsupportedTestException("the browser cannot run a local server");
        if (accelerator.AcceleratorType != AcceleratorType.CPU) throw new UnsupportedTestException("host-only test: runs on the CPU lane");
        const int N = 100_000;
        var data = new byte[N];
        for (int i = 0; i < N; i++) data[i] = (byte)(i * 37 % 251);
        var listener = new TcpListener(IPAddress.Loopback, 0);
        listener.Start();
        int port = ((IPEndPoint)listener.LocalEndpoint).Port;
        int resolves = 0, cdnRequests = 0, signature = 0, expiredBelow = 0;
        using var stop = new CancellationTokenSource();
        var server = Task.Run(async () =>
        {
            while (!stop.IsCancellationRequested)
            {
                TcpClient client;
                try { client = await listener.AcceptTcpClientAsync(stop.Token); } catch { break; }
                _ = Task.Run(async () =>
                {
                    using var c = client;
                    var ns = c.GetStream();
                    var reader = new StreamReader(ns, Encoding.ASCII, false, 1024, leaveOpen: true);
                    string path = (await reader.ReadLineAsync())?.Split(' ')[1] ?? "";
                    long a = 0, b = N - 1;
                    string? line;
                    while (!string.IsNullOrEmpty(line = await reader.ReadLineAsync()))
                        if (line.StartsWith("Range: bytes=", StringComparison.OrdinalIgnoreCase))
                        {
                            var r = line["Range: bytes=".Length..].Split('-');
                            a = long.Parse(r[0]); b = r[1].Length > 0 ? long.Parse(r[1]) : N - 1;
                        }
                    string head;
                    byte[] body = Array.Empty<byte>();
                    if (path.StartsWith("/resolve/"))
                    {
                        Interlocked.Increment(ref resolves);
                        int sig = Interlocked.Increment(ref signature);
                        head = $"HTTP/1.1 302 Found\r\nLocation: http://127.0.0.1:{port}/cdn/model.bin?sig={sig}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
                    }
                    else if (path.StartsWith("/cdn/model.bin?sig=") && int.Parse(path[(path.IndexOf('=') + 1)..]) is int s)
                    {
                        Interlocked.Increment(ref cdnRequests);
                        if (s <= Volatile.Read(ref expiredBelow)) head = "HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
                        else
                        {
                            body = data.AsSpan((int)a, (int)(b - a + 1)).ToArray();
                            head = $"HTTP/1.1 206 Partial Content\r\nContent-Length: {body.Length}\r\nContent-Range: bytes {a}-{b}/{N}\r\nConnection: close\r\n\r\n";
                        }
                    }
                    else head = "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
                    await ns.WriteAsync(Encoding.ASCII.GetBytes(head));
                    await ns.WriteAsync(body);
                    await ns.FlushAsync();
                });
            }
        });
        try
        {
            using var http = new HttpClient { Timeout = TimeSpan.FromSeconds(30) };
            string url = $"http://127.0.0.1:{port}/resolve/model.bin";
            var s = new HttpRangeStream(http, url, N);
            const int reads = 20, size = 5_000;
            for (int r = 0; r < reads; r++)
            {
                var buf = new byte[size];
                int n = await s.ReadAsync(buf);
                Ur_Expect(n == size && buf.AsSpan().SequenceEqual(data.AsSpan(r * size, size)), $"read {r}: the right bytes ({n})");
            }
            Ur_Expect(resolves == 1 && s.ResolveRequests == 1, $"{reads} reads resolved once: the server saw {resolves} /resolve/ requests, the stream counted {s.ResolveRequests}");
            Ur_Expect(s.RangeUrl?.Contains("/cdn/") == true && cdnRequests == reads, $"the ranges went to the signed URL ({s.RangeUrl}; {cdnRequests} /cdn/ requests)");

            // the signature expires: the next read meets a 403, resolves again and gets the right bytes from the new signature
            Volatile.Write(ref expiredBelow, signature);
            s.Seek(12_345, SeekOrigin.Begin);
            var again = new byte[7_000];
            int m = await s.ReadAsync(again);
            Ur_Expect(m == again.Length && again.AsSpan().SequenceEqual(data.AsSpan(12_345, again.Length)), $"after the expiry: the right bytes ({m})");
            Ur_Expect(resolves == 2 && s.ResolveRequests == 2, $"the expired signature was resolved again once: {resolves} /resolve/ requests, {s.ResolveRequests} counted");
            Ur_Expect(s.RangeUrl?.EndsWith($"sig={signature}") == true, $"the stream now uses the new signature ({s.RangeUrl})");
            Console.WriteLine($"[Hub] HttpRangeStream on a redirecting source: {reads} reads, {resolves} resolves (1 + 1 after an expiry), {cdnRequests} ranges to the signed URL");
            return $"{reads} reads: 1 resolve; after an expiry: 1 more";
        }
        finally { stop.Cancel(); listener.Stop(); try { await server; } catch { } }
    });

    /// <summary>
    /// The three range readers against a real Hugging Face LFS file (anonymous): each resolves /resolve/ ONCE and sends its ranges to the signed CDN URL.
    /// Browser (the WebGPU lane, once): <see cref="FetchRangeStream"/> reading the first MiB in 64 KiB chunks (16 ranges, 4 in flight), and the OPFS
    /// downloader (<see cref="HttpModelDownloader"/>) taking a 19 MB file in 1 MiB segments; both against the same bytes fetched in one request.
    /// <see cref="HttpRangeStream"/> on the browser's HttpClient, which does not expose where a redirect landed (it asks fetch: 2 resolves). Desktop (the CPU lane):
    /// <see cref="HttpRangeStream"/> in 8 reads behind a request spy, which sees exactly one request to huggingface.co and the rest to the CDN.
    /// </summary>
    [TestMethod(Timeout = 300000)]
    public async Task HuggingFace_RangeReadersResolveOnce() => await RunTest(async accelerator =>
    {
        // Hugging Face itself, by name: GetDownloadUrl on "main" gives the models host
        string url = HuggingFaceClient.GetDownloadUrl(ResolveRepo, ResolveFile, "main", "https://huggingface.co");
        const int prefix = 1 << 20;
        if (!OperatingSystem.IsBrowser())
        {
            if (accelerator.AcceleratorType != AcceleratorType.CPU) throw new UnsupportedTestException("network test: runs on the CPU lane");
            var spy = new RequestSpy(new HttpClientHandler());
            using var http = new HttpClient(spy) { Timeout = TimeSpan.FromMinutes(2) };
            using var refReq = new HttpRequestMessage(HttpMethod.Get, url);
            refReq.Headers.Range = new System.Net.Http.Headers.RangeHeaderValue(0, prefix - 1);
            using var refResp = await http.SendAsync(refReq);
            var reference = await refResp.Content.ReadAsByteArrayAsync();
            Ur_Expect(reference.Length == prefix, $"the reference range ({reference.Length} bytes)");
            spy.Requests.Clear();
            var s = new HttpRangeStream(http, url, prefix);
            const int reads = 8;
            var got = new byte[prefix];
            for (int r = 0; r < reads; r++) Ur_Expect(await s.ReadAsync(got.AsMemory(r * (prefix / reads), prefix / reads)) == prefix / reads, $"read {r}");
            Ur_Expect(got.AsSpan().SequenceEqual(reference), "the ranges match the single request");
            int toHub = spy.Requests.Count(q => q.Host.EndsWith("huggingface.co"));
            var cdnHosts = spy.Requests.Where(q => !q.Host.EndsWith("huggingface.co")).Select(q => q.Host).Distinct().ToList();
            Ur_Expect(toHub == 1 && s.ResolveRequests == 1, $"{reads} reads resolved once: {toHub} requests to huggingface.co, {s.ResolveRequests} counted");
            Ur_Expect(spy.Requests.Count == reads && cdnHosts.Count == 1, $"the other {reads - 1} went to the CDN ({string.Join(", ", cdnHosts)}; {spy.Requests.Count} requests)");
            Console.WriteLine($"[Hub] HttpRangeStream vs Hugging Face: {reads} reads, {toHub} resolver request, {reads - toHub} to {cdnHosts[0]}");
            return $"desktop: {reads} reads, {toHub} resolve";
        }

        if (accelerator.AcceleratorType != AcceleratorType.WebGPU) throw new UnsupportedTestException("network test: runs once, on the WebGPU lane");
        var js = RequireBrowserRuntime();
        var report = new List<string>();
        byte[] refBytes;
        await using (var one = new FetchRangeStream(url, prefix) { ReadAhead = prefix, Depth = 1 })
        {
            using var all = await one.ReadUint8ArrayAsync(prefix);
            refBytes = all.ReadBytes();
        }
        Ur_Expect(refBytes.Length == prefix, $"the reference range ({refBytes.Length} bytes)");

        // FetchRangeStream: 16 chunks, 4 in flight; the ones scheduled before the first answer wait for it
        await using (var fs = new FetchRangeStream(url, prefix) { ReadAhead = 64 << 10, Depth = 4 })
        {
            var got = new byte[prefix];
            for (int off = 0; off < prefix; off += 40_000)
            {
                using var part = await fs.ReadUint8ArrayAsync(Math.Min(40_000, prefix - off));
                part.ReadBytes().CopyTo(got, off);
            }
            Ur_Expect(got.AsSpan().SequenceEqual(refBytes), "FetchRangeStream: the chunks match the single request");
            // a request that FAILED (a transient network or browser-cache error: Chrome's ERR_CACHE_OPERATION_NOT_SUPPORTED was seen once in four sweeps) resolves again by
            // design, so one resolve plus at most one per failure. No failure: exactly one.
            Ur_Expect(fs.ResolveRequests >= 1 && fs.ResolveRequests <= 1 + fs.FailedRequests, $"FetchRangeStream: 16 chunks resolved once ({fs.ResolveRequests} resolves, {fs.FailedRequests} failed requests)");
            Ur_Expect(fs.RangeUrl != null && !fs.RangeUrl.StartsWith("https://huggingface.co/"), $"FetchRangeStream: the chunks went to the CDN ({fs.RangeUrl?[..Math.Min(60, fs.RangeUrl.Length)]})");
            report.Add($"FetchRangeStream 16 chunks: {fs.ResolveRequests} resolve, to {new Uri(fs.RangeUrl!).Host}");
        }

        // the OPFS downloader: 1 MiB segments over a 19 MB file
        using var cache = new OpfsModelCache(js) { CacheDirectoryName = "ilgpu-ml-test-resolve" };
        if (!await cache.IsAvailableAsync()) throw new UnsupportedTestException("OPFS unavailable in this context");
        cache.Downloader.SegmentBytes = 1 << 20;
        cache.Downloader.MinSegmentBytes = 1 << 20;
        const string key = "resolve-once-fixture";
        await cache.RemoveAsync(key);
        try
        {
            await using (var stream = await cache.Downloader.GetOrDownloadAsync(HuggingFaceClient.GetDownloadUrl(ResolveRepo, ResolveWholeFile, "main", "https://huggingface.co"), key))
                Ur_Expect(stream.Length == ResolveWholeBytes, $"the downloader got the whole file ({stream.Length} bytes)");
            long chunks = HttpModelDownloader.LastFetchChunks, resolves = HttpModelDownloader.LastResolveRequests;
            Ur_Expect(chunks >= 10, $"the download was segmented ({chunks} fetches)");
            Ur_Expect(resolves == 1, $"the downloader resolved once for {chunks} segments ({resolves})");
            report.Add($"downloader {chunks} segments: {resolves} resolve");
        }
        finally { await cache.RemoveAsync(key); }

        // HttpRangeStream on the browser's HttpClient, which does not expose where a redirect landed: the stream asks fetch (one byte), then pins
        var bhttp = GetHttpClient() ?? throw new UnsupportedTestException("HttpClient not available");
        var hs = new HttpRangeStream(bhttp, url, prefix);
        var hgot = new byte[prefix];
        for (int r = 0; r < 8; r++) await hs.ReadAsync(hgot.AsMemory(r * (prefix / 8), prefix / 8));
        Ur_Expect(hgot.AsSpan().SequenceEqual(refBytes), "HttpRangeStream (browser): the ranges match the single request");
        Ur_Expect(hs.RangeUrl != null && !hs.RangeUrl.StartsWith("https://huggingface.co/"), $"HttpRangeStream (browser): the reads went to the CDN ({hs.RangeUrl})");
        Ur_Expect(hs.ResolveRequests == 2, $"HttpRangeStream (browser): 8 reads, 2 resolves (the first read + fetch's answer where it landed): {hs.ResolveRequests}");
        report.Add($"browser HttpRangeStream 8 reads: {hs.ResolveRequests} resolves, to {new Uri(hs.RangeUrl!).Host}");
        string line = string.Join("; ", report);
        Console.WriteLine("[Hub] range readers vs Hugging Face (browser): " + line);
        return line;
    });
}
