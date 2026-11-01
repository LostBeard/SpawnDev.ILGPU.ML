using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.UnitTesting;
using System.Diagnostics;
using System.Net;
using System.Net.Sockets;
using System.Text;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    /// <summary>
    /// <see cref="HttpRangeStream"/> against a local server that misbehaves on purpose: a body that stalls half way (the read must resume from the bytes it
    /// has, not hang: a test held an idle connection to the models host for 25 minutes, 2026-11-01), a connection dropped half way and a body that ends early without an error (resume, counted against the attempts), a server that
    /// ignores Range and sends the whole file with 200 (must throw: those are not the bytes asked for), and one that never sends a byte (must throw after
    /// its attempts, in bounded time). Desktop only (the browser cannot listen; <see cref="FetchRangeStream"/> has its own timeout); runs once, on the CPU lane.
    /// </summary>
    [TestMethod(Timeout = 120000)]
    public async Task HttpRangeStream_ResumesStalledAndDroppedBodies_RefusesIgnoredRange() => await RunTest(async accelerator =>
    {
        if (OperatingSystem.IsBrowser()) throw new UnsupportedTestException("the browser cannot run a local server");
        if (accelerator.AcceleratorType != AcceleratorType.CPU) throw new UnsupportedTestException("host-only test: runs on the CPU lane");
        const int N = 200_000;
        var data = new byte[N];
        for (int i = 0; i < N; i++) data[i] = (byte)(i * 31 % 251);
        var listener = new TcpListener(IPAddress.Loopback, 0);
        listener.Start();
        int port = ((IPEndPoint)listener.LocalEndpoint).Port;
        string mode = "good";
        int requests = 0;
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
                    long a = 0, b = N - 1;
                    string? line;
                    while (!string.IsNullOrEmpty(line = await reader.ReadLineAsync()))
                        if (line.StartsWith("Range: bytes=", StringComparison.OrdinalIgnoreCase))
                        {
                            var r = line["Range: bytes=".Length..].Split('-');
                            a = long.Parse(r[0]); b = r[1].Length > 0 ? long.Parse(r[1]) : N - 1;
                        }
                    int k = Interlocked.Increment(ref requests);
                    async Task Send(string head, byte[] body, int count)
                    {
                        var h = Encoding.ASCII.GetBytes(head);
                        await ns.WriteAsync(h);
                        await ns.WriteAsync(body.AsMemory(0, count));
                        await ns.FlushAsync();
                    }
                    var slice = data.AsSpan((int)a, (int)(b - a + 1)).ToArray();
                    string partial = $"HTTP/1.1 206 Partial Content\r\nContent-Length: {slice.Length}\r\nContent-Range: bytes {a}-{b}/{N}\r\nConnection: close\r\n\r\n";
                    switch (mode)
                    {
                        case "good": await Send(partial, slice, slice.Length); break;
                        case "stall" when k == 1: await Send(partial, slice, slice.Length / 2); await Task.Delay(Timeout.Infinite, stop.Token).ContinueWith(_ => { }); break;
                        case "drop" when k == 1: await Send(partial, slice, slice.Length / 2); break;   // the connection closes half way
                        // no Content-Length: the body just ends at the close, so the client sees an early end of stream, not an error
                        case "short" when k == 1: await Send($"HTTP/1.1 206 Partial Content\r\nContent-Range: bytes {a}-{b}/{N}\r\nConnection: close\r\n\r\n", slice, slice.Length / 2); break;
                        case "stall" or "drop" or "short": await Send(partial, slice, slice.Length); break;
                        case "ignore": await Send($"HTTP/1.1 200 OK\r\nContent-Length: {N}\r\nConnection: close\r\n\r\n", data, N); break;
                        case "silent": await Task.Delay(Timeout.Infinite, stop.Token).ContinueWith(_ => { }); break;
                    }
                });
            }
        });
        var notes = new List<string>();
        bool ok = true;
        using var http = new HttpClient { Timeout = Timeout.InfiniteTimeSpan };   // as the stall bug needs: the client's own timeout does not cover a body
        string url = $"http://127.0.0.1:{port}/model.bin";
        async Task<(byte[]? Got, Exception? Error, int Resumes, double Ms)> Read(string m, long offset, int count)
        {
            mode = m; requests = 0;
            var s = new HttpRangeStream(http, url, N) { StallTimeout = TimeSpan.FromSeconds(1), Attempts = 3 };
            s.Seek(offset, SeekOrigin.Begin);
            var buf = new byte[count];
            var sw = Stopwatch.StartNew();
            try
            {
                int n = await s.ReadAsync(buf);
                return (buf[..n], null, s.Resumes, sw.Elapsed.TotalMilliseconds);
            }
            catch (Exception ex) { return (null, ex, s.Resumes, sw.Elapsed.TotalMilliseconds); }
        }
        bool Same(byte[]? got, long offset, int count) => got != null && got.Length == count && got.AsSpan().SequenceEqual(data.AsSpan((int)offset, count));
        try
        {
            foreach (var m in new[] { "good", "stall", "drop", "short" })
            {
                var (got, err, resumes, ms) = await Read(m, 12345, 100_000);
                bool pass = err == null && Same(got, 12345, 100_000) && (m == "good" ? resumes == 0 : resumes >= 1);
                ok &= pass;
                notes.Add($"{m}: {(pass ? "right bytes" : "WRONG")} (resumes {resumes}, {ms:F0} ms{(err != null ? ", " + err.Message : "")})");
            }
            {
                var (got, err, resumes, ms) = await Read("ignore", 12345, 1000);
                bool pass = got == null && err is IOException && resumes == 0;
                ok &= pass;
                notes.Add($"range ignored: {(pass ? "refused" : "WRONG")} ({(err != null ? err.Message : got != null ? (Same(got, 12345, 1000) ? "read" : "read the WRONG bytes") : "?")})");
            }
            {
                var (got, err, resumes, ms) = await Read("silent", 0, 1000);
                bool pass = got == null && err is IOException && err.InnerException is TimeoutException && ms < 20000;
                ok &= pass;
                notes.Add($"silent server: {(pass ? "gave up" : "WRONG")} after {ms:F0} ms, {resumes} resumes ({err?.Message})");
            }
        }
        finally { stop.Cancel(); listener.Stop(); }
        var report = string.Join(" | ", notes);
        Console.WriteLine("[HttpRangeStream] " + report);
        if (!ok) throw new Exception(report);
        return report;
    });
}
