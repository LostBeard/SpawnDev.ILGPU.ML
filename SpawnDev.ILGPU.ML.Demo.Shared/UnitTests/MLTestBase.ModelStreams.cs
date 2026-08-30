using SpawnDev.UnitTesting;
using SpawnDev.ILGPU.ML.Hub;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    /// <summary>
    /// Open a remote ONNX model as a SEEKABLE stream WITHOUT ever holding the whole file in the managed
    /// heap - the source every heavy-model test should load from.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <c>InferenceSession.DownloadBytesChunkedAsync</c> returns the model as one <c>byte[]</c>. For
    /// DistilGPT-2 that is a single 329 MB managed allocation, and in a browser lane it lands in the
    /// single-threaded WASM heap. MEASURED 2026-08-30: that is why
    /// <c>TurboQuant_DistilGPT2_KVCacheCaptures</c> passed 8/8 in isolation and failed only under
    /// full-sweep memory pressure - the test was fine, its model DELIVERY was not. Wrapping those same
    /// bytes in a <c>MemoryStream</c> changes nothing; the array is still there.
    /// </para>
    /// <para>
    /// Two sources, picked by what the lane can actually do:
    /// <list type="bullet">
    /// <item><description><b>Browser</b> - <see cref="ModelHub.OpenStreamFromUrlAsync"/>, an OPFS-cached
    /// <c>BlobStream</c>. It is an <c>IJSReadStream</c>, so the bytes stay JS-side end to end, the weights
    /// upload zero-copy JS-&gt;GPU, and <c>InferenceSession</c> ARMS
    /// <c>BrowserBufferPolicy.StrictHostCopyMaxBytes</c> for the load - any weight that regresses onto a
    /// host copy fails the test loudly instead of silently costing main-thread time. OPFS also persists,
    /// so the second and third browser backends in a sweep do not re-download.</description></item>
    /// <item><description><b>Everywhere else</b> - a disk-cached copy under the temp dir, opened as a
    /// <see cref="FileStream"/>. The download streams response-&gt;file with
    /// <see cref="HttpCompletionOption.ResponseHeadersRead"/>, so peak managed memory is one 1 MB copy
    /// buffer and never the model. One download serves every desktop backend and every test in the sweep.
    /// (<see cref="HttpRangeStream"/> also works and holds no more memory - MEASURED 2026-08-30: a full
    /// 328,939,971-byte load pulled 330,610,779 bytes in 55.5s - but it re-fetches for every backend.)
    /// </description></item>
    /// </list>
    /// </para>
    /// <para>
    /// ⚠️ The caller owns the returned stream and must dispose it. It is NOT rewound between uses; the ONNX
    /// streaming parser seeks from 0 itself.
    /// </para>
    /// </remarks>
    /// <param name="url">Absolute URL of the model file.</param>
    /// <param name="ct">Cancellation token.</param>
    /// <returns>A seekable stream over the model.</returns>
    protected async Task<Stream> OpenSeekableModelStreamAsync(string url, CancellationToken ct = default)
    {
        var js = SpawnDev.SpawnJS.SpawnJSRuntime.Instance;
        if (js != null && js.IsBrowser)
        {
            // OPFS-cached, JS-side, and an IJSReadStream - the only source that keeps a 329 MB model off
            // the WASM heap entirely. Null means OPFS is unavailable in this context (some worker/private
            // modes), which is a reason to fall through, not to fail the test.
            var hub = new ModelHub(js);
            var blob = await hub.OpenStreamFromUrlAsync(url).ConfigureAwait(false);
            if (blob != null) return blob;
        }

        var http = GetHttpClient() ?? throw new UnsupportedTestException("HttpClient not available");
        var cachePath = Path.Combine(Path.GetTempPath(), "spawndev-ml-test-models", CacheFileName(url));
        Directory.CreateDirectory(Path.GetDirectoryName(cachePath)!);

        long expected = await ContentLengthAsync(http, url, ct).ConfigureAwait(false);
        if (!IsComplete(cachePath, expected))
            await DownloadToCacheAsync(http, url, cachePath, expected, ct).ConfigureAwait(false);

        return new FileStream(cachePath, FileMode.Open, FileAccess.Read, FileShare.Read,
                              bufferSize: 1 << 20, useAsync: true);
    }

    /// <summary>
    /// Download <paramref name="url"/> to <paramref name="cachePath"/> exactly once, safely under
    /// concurrent lanes.
    /// </summary>
    /// <remarks>
    /// ⚠️ PMT runs the desktop backends in SEPARATE PROCESSES (one DemoConsole per lane), and they start
    /// together. MEASURED 2026-08-30, both failure modes in turn:
    /// <list type="number">
    /// <item><description>A fixed <c>.part</c> name: CPU won, CUDA + OpenCL died with "The process cannot
    /// access the file ... .part because it is being used by another process".</description></item>
    /// <item><description>A per-attempt <c>.part</c> name plus an in-process <see cref="SemaphoreSlim"/>:
    /// still 2 failures, now <c>UnauthorizedAccessException</c> out of <c>File.Move</c> - a static gate
    /// cannot serialise anything when the contenders are different PROCESSES, and Windows denies an
    /// overwriting move onto a file another lane already holds open.</description></item>
    /// </list>
    /// So the coordination has to be cross-process, and the move has to tolerate losing the race:
    /// a named <see cref="Mutex"/> keeps the cold-cache case to ONE 329 MB download instead of three, and
    /// a non-overwriting move plus a "did someone else finish it" check makes losing harmless even if the
    /// mutex is unavailable on some platform.
    /// </remarks>
    private static async Task DownloadToCacheAsync(
        HttpClient http, string url, string cachePath, long expected, CancellationToken ct)
    {
        // Named mutexes are a Windows-first primitive; never let one being unavailable fail a test run.
        // Without it the worst case is a duplicated download, which the tolerant move below absorbs.
        Mutex? mutex = null;
        bool held = false;
        try { mutex = new Mutex(false, @"Local\spawndev-ml-model-" + Path.GetFileName(cachePath)); }
        catch { /* not supported here - fall through unsynchronised */ }

        try
        {
            if (mutex != null)
            {
                // AbandonedMutexException = a previous lane was killed holding it. That still grants us
                // ownership, and the completeness re-check below covers whatever it left behind.
                try { held = mutex.WaitOne(TimeSpan.FromMinutes(10)); }
                catch (AbandonedMutexException) { held = true; }
                catch { held = false; }
            }

            // Re-check under the mutex: the lane that just released it did the download for us.
            if (IsComplete(cachePath, expected)) return;

            var partPath = $"{cachePath}.{Environment.ProcessId}.{Guid.NewGuid():N}.part";
            try
            {
                using (var res = await http.GetAsync(
                    url, HttpCompletionOption.ResponseHeadersRead, ct).ConfigureAwait(false))
                {
                    res.EnsureSuccessStatusCode();
                    using var src = await res.Content.ReadAsStreamAsync(ct).ConfigureAwait(false);
                    using var dst = new FileStream(partPath, FileMode.CreateNew, FileAccess.Write,
                                                   FileShare.None, bufferSize: 1 << 20, useAsync: true);
                    await src.CopyToAsync(dst, 1 << 20, ct).ConfigureAwait(false);
                }

                long got = new FileInfo(partPath).Length;
                if (expected > 0 && got != expected)
                    throw new Exception($"downloaded {got:N0} bytes from {url}, expected {expected:N0}");

                try
                {
                    // overwrite:false on purpose - never clobber a copy another lane may have OPEN.
                    File.Move(partPath, cachePath);
                }
                catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
                {
                    // Lost the race. That is a success as long as the winner's file is complete.
                    if (!IsComplete(cachePath, expected)) throw;
                }
            }
            finally
            {
                // Never leave scratch behind - a half file that looks like a model is worse than no file,
                // and this one is 329 MB of the Captain's disk.
                try { if (File.Exists(partPath)) File.Delete(partPath); } catch { /* best effort */ }
            }
        }
        finally
        {
            if (held) { try { mutex!.ReleaseMutex(); } catch { /* best effort */ } }
            mutex?.Dispose();
        }
    }

    /// <summary>
    /// Whether <paramref name="path"/> holds a COMPLETE download. Length, not existence: a run killed
    /// mid-download leaves a short file, and a short ONNX file fails as a confusing parse error rather than
    /// as the truncated download it is. When the server reports no length, existence is all we have.
    /// </summary>
    private static bool IsComplete(string path, long expectedLength)
    {
        if (!File.Exists(path)) return false;
        return expectedLength <= 0 || new FileInfo(path).Length == expectedLength;
    }

    /// <summary>Content-Length of <paramref name="url"/>, or -1 when the server does not report one.</summary>
    private static async Task<long> ContentLengthAsync(HttpClient http, string url, CancellationToken ct)
    {
        using var req = new HttpRequestMessage(HttpMethod.Head, url);
        using var res = await http.SendAsync(req, ct).ConfigureAwait(false);
        return res.IsSuccessStatusCode ? res.Content.Headers.ContentLength ?? -1 : -1;
    }

    /// <summary>A filesystem-safe cache name for a URL: its last path segment prefixed by a hash of the
    /// whole URL, so two repos' <c>model.onnx</c> cannot collide while the name stays readable.</summary>
    private static string CacheFileName(string url)
    {
        var hash = Convert.ToHexString(
            System.Security.Cryptography.SHA256.HashData(System.Text.Encoding.UTF8.GetBytes(url)))[..16];
        var leaf = url.Split('/', '?', '#').LastOrDefault(p => !string.IsNullOrEmpty(p)) ?? "model.onnx";
        foreach (var bad in Path.GetInvalidFileNameChars()) leaf = leaf.Replace(bad, '_');
        return $"{hash}_{leaf}";
    }
}
