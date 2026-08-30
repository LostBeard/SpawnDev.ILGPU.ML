using SpawnDev.SpawnJS;
using SpawnDev.SpawnJS.JSObjects;
using SpawnDev.SpawnJS.Toolbox;

namespace SpawnDev.ILGPU.ML.Hub;

/// <summary>
/// Browser-side model cache using OPFS (Origin Private File System).
/// Models are downloaded once and cached locally — subsequent loads are instant.
/// Falls back to in-memory cache if OPFS is unavailable.
///
/// On desktop (.NET console/WPF), a separate implementation should use
/// the local filesystem (e.g., %APPDATA%/SpawnDev.ILGPU.ML/models/).
/// This class is for browser (Blazor WASM) only.
/// </summary>
public class ModelCache : IDisposable
{
    private readonly SpawnJSRuntime _js;
    private FileSystemDirectoryHandle? _cacheDir;
    private bool _initialized;

    /// <summary>Name of the OPFS subdirectory for cached models.</summary>
    public string CacheDirectoryName { get; set; } = "ilgpu-ml-models";

    /// <summary>
    /// Fired during model download with (bytesReceived, totalBytes).
    /// totalBytes may be -1 if the server doesn't provide Content-Length.
    /// </summary>
    public event Action<long, long>? OnDownloadProgress;

    public ModelCache(SpawnJSRuntime js)
    {
        _js = js;
    }

    /// <summary>
    /// Get a model file from cache, or download and cache it.
    /// Returns the raw bytes of the file.
    /// </summary>
    /// <param name="url">URL to download from if not cached</param>
    /// <param name="cacheKey">Cache key (filename in OPFS). If null, derived from URL.</param>
    public async Task<byte[]> GetOrFetchAsync(string url, string? cacheKey = null)
    {
        cacheKey ??= UrlToCacheKey(url);

        // Try cache first
        var cached = await TryReadFromCacheAsync(cacheKey);
        if (cached != null) return cached;

        // Download
        var bytes = await DownloadWithProgressAsync(url);

        // Cache for next time
        await WriteToCacheAsync(cacheKey, bytes);

        return bytes;
    }

    /// <summary>
    /// True if the browser OPFS cache is available (secure context + a modern browser). When false,
    /// nothing is cached and every model load re-downloads — a cache-management UI should say so.
    /// </summary>
    public async Task<bool> IsAvailableAsync()
    {
        await EnsureInitializedAsync();
        return _cacheDir != null;
    }

    /// <summary>
    /// Check if a model is already cached.
    /// </summary>
    public async Task<bool> IsCachedAsync(string url, string? cacheKey = null)
    {
        cacheKey ??= UrlToCacheKey(url);
        await EnsureInitializedAsync();
        if (_cacheDir == null) return false;

        try
        {
            using var fileHandle = await _cacheDir.GetFileHandle(cacheKey);
            return true;
        }
        catch
        {
            return false;
        }
    }

    /// <summary>
    /// Remove a cached model.
    /// </summary>
    public async Task RemoveAsync(string cacheKey)
    {
        await EnsureInitializedAsync();
        if (_cacheDir == null) return;

        try
        {
            await _cacheDir.RemoveEntry(cacheKey);
        }
        catch { /* File didn't exist */ }
    }

    /// <summary>
    /// Clear all cached models.
    /// </summary>
    public async Task ClearAllAsync()
    {
        await EnsureInitializedAsync();
        if (_cacheDir == null) return;

        var entries = await _cacheDir.ValuesList();
        foreach (var entry in entries)
        {
            try
            {
                using var handle = entry;
                await _cacheDir.RemoveEntry(handle.Name);
            }
            catch { }
        }
    }

    /// <summary>
    /// Get total size of all cached models in bytes.
    /// </summary>
    public async Task<long> GetCacheSizeAsync()
    {
        await EnsureInitializedAsync();
        if (_cacheDir == null) return 0;

        long total = 0;
        var entries = await _cacheDir.ValuesList();
        foreach (var entry in entries)
        {
            try
            {
                using var handle = entry;
                if (handle is FileSystemFileHandle fileHandle)
                {
                    using var file = await fileHandle.GetFile();
                    total += (long)file.Size;
                }
            }
            catch { }
        }
        return total;
    }

    /// <summary>
    /// List all cached model keys and their sizes.
    /// </summary>
    public async Task<List<(string Key, long SizeBytes)>> ListCachedAsync()
    {
        var result = new List<(string, long)>();
        await EnsureInitializedAsync();
        if (_cacheDir == null) return result;

        var entries = await _cacheDir.ValuesList();
        foreach (var entry in entries)
        {
            try
            {
                using var handle = entry;
                if (handle is FileSystemFileHandle fileHandle)
                {
                    using var file = await fileHandle.GetFile();
                    result.Add((handle.Name, (long)file.Size));
                }
            }
            catch { }
        }
        return result;
    }

    // ──────────────────────────────────────────────
    //  Internal
    // ──────────────────────────────────────────────

    private async Task EnsureInitializedAsync()
    {
        if (_initialized) return;
        _initialized = true;

        try
        {
            using var navigator = _js.Get<Navigator>("navigator");
            using var storage = navigator.Storage;
            using var root = await storage.GetDirectory();
            _cacheDir = await root.GetDirectoryHandle(CacheDirectoryName, create: true);
        }
        catch
        {
            // OPFS not available (older browser, non-secure context)
            _cacheDir = null;
        }
    }

    /// <summary>
    /// Open a CACHED entry as a seekable stream whose bytes stay JS-side, or <c>null</c> if it is not cached.
    /// </summary>
    /// <remarks>
    /// This is the streaming counterpart to <see cref="GetOrFetchAsync"/>, and the difference is the whole
    /// point: that method reaches the same OPFS <c>File</c> (which IS a Blob) and then calls
    /// <c>ArrayBuffer()</c> + <c>ReadBytes()</c>, materialising the ENTIRE model in JS and then copying all
    /// of it onto the .NET/WASM managed heap. For a 300 MB+ model that is the thing the standing rule
    /// forbids, and it is what makes a model load OOM under memory pressure.
    /// <para>
    /// A <see cref="BlobStream"/> is an <c>IJSReadStream</c>, so <c>InferenceSession.CreateFromOnnxStreamAsync</c>
    /// can parse structure from it and <c>BufferPool</c> can send each weight JS-&gt;GPU via <c>CopyFromJS</c>,
    /// never touching the managed heap. It is async-only (<c>CanReadSync == false</c>) and seekable - the
    /// contract the ONNX/GGUF parsers are already written against.
    /// </para>
    /// <para>
    /// ⚠️ The returned stream OWNS the underlying <c>File</c>: <see cref="BlobStream.Dispose(bool)"/> disposes
    /// it, so the handle must not be disposed here. Dispose the stream when done.
    /// </para>
    /// </remarks>
    /// <param name="cacheKey">Cache key, as produced by <see cref="UrlToCacheKey"/>.</param>
    /// <returns>A seekable JS-side stream, or null when the entry is not cached.</returns>
    public async Task<BlobStream?> OpenCachedStreamAsync(string cacheKey)
    {
        await EnsureInitializedAsync();
        if (_cacheDir == null) return null;

        try
        {
            using var fileHandle = await _cacheDir.GetFileHandle(cacheKey);
            var file = await fileHandle.GetFile();   // NOT disposed here - BlobStream takes ownership
            return new BlobStream(file);
        }
        catch
        {
            return null; // Not cached
        }
    }

    /// <summary>
    /// Cached-or-fetched, returned as a seekable JS-side stream instead of a <c>byte[]</c>.
    /// </summary>
    /// <remarks>
    /// ⚠️ Honest limitation: on a cache MISS this still downloads via the existing byte[] path and writes
    /// that to OPFS before re-opening it as a stream, so the first load of a model still materialises it
    /// once. The cached path - which is every subsequent load, and every test run after the first - no
    /// longer does. Closing the remaining gap means fetching JS-side straight into OPFS (a JS
    /// <c>fetch</c> -&gt; <c>Response.body</c> piped to the OPFS writable) so the bytes never enter .NET at
    /// all; that is a separate change and is deliberately not smuggled in here.
    /// </remarks>
    /// <param name="url">Source URL, used only on a cache miss.</param>
    /// <param name="cacheKey">Optional explicit cache key; derived from the URL when omitted.</param>
    /// <returns>A seekable JS-side stream, or null when OPFS is unavailable.</returns>
    public async Task<BlobStream?> GetOrFetchStreamAsync(string url, string? cacheKey = null)
    {
        cacheKey ??= UrlToCacheKey(url);

        var cached = await OpenCachedStreamAsync(cacheKey);
        if (cached != null) return cached;

        var bytes = await DownloadWithProgressAsync(url);
        await WriteToCacheAsync(cacheKey, bytes);
        return await OpenCachedStreamAsync(cacheKey);
    }

    private async Task<byte[]?> TryReadFromCacheAsync(string cacheKey)
    {
        await EnsureInitializedAsync();
        if (_cacheDir == null) return null;

        try
        {
            using var fileHandle = await _cacheDir.GetFileHandle(cacheKey);
            using var file = await fileHandle.GetFile();
            using var arrayBuffer = await file.ArrayBuffer();
            using var uint8 = new Uint8Array(arrayBuffer);
            return uint8.ReadBytes();
        }
        catch
        {
            return null; // Not cached
        }
    }

    private async Task WriteToCacheAsync(string cacheKey, byte[] data)
    {
        await EnsureInitializedAsync();
        if (_cacheDir == null) return;

        try
        {
            using var fileHandle = await _cacheDir.GetFileHandle(cacheKey, create: true);
            using var writable = await fileHandle.CreateWritable();
            using var uint8 = new Uint8Array(data);
            await writable.Write(uint8);
            await writable.Close();
        }
        catch { /* Cache write failed — not critical */ }
    }

    private async Task<byte[]> DownloadWithProgressAsync(string url)
    {
        // Use fetch for streaming progress
        using var response = await _js.Get<Window>("window").Fetch(url);

        // fetch() does NOT throw on 404/500 — it resolves with ok=false and an ERROR BODY. Without this
        // check that body was streamed, returned as if it were the model, and then WRITTEN TO THE OPFS
        // CACHE by GetOrFetchAsync. A 404 page is ~15 bytes, so the failure surfaced much later and
        // somewhere else entirely, as "EndOfStreamException: Expected 110 bytes, stream ended 97 short"
        // out of the ONNX proto reader — and because it was cached, it then failed IDENTICALLY on every
        // subsequent run with no hint that the download had failed. Only clearing the cache recovered it.
        // MEASURED 2026-08-29: KnownModels.SqueezeNet + KnownFiles.OnnxModel 404s, and this is how it
        // presented. Fail here, naming the status and URL, and never cache a failed response.
        if (!response.Ok)
            throw new HttpRequestException($"Model download failed: {(int)response.Status} {response.StatusText} for {url}");

        var contentLength = response.Headers.Get("content-length");
        long totalBytes = contentLength != null ? long.Parse(contentLength) : -1;

        using var body = response.Body!;
        using var reader = body.GetReader();

        var chunks = new List<byte[]>();
        long received = 0;

        while (true)
        {
            var result = await reader.Read();
            if (result.Done) break;

            using var chunk = result.Value!;
            var bytes = chunk.ReadBytes();
            chunks.Add(bytes);
            received += bytes.Length;

            OnDownloadProgress?.Invoke(received, totalBytes);
        }

        // Concatenate chunks
        var output = new byte[received];
        int offset = 0;
        foreach (var chunk in chunks)
        {
            System.Array.Copy(chunk, 0, output, offset, chunk.Length);
            offset += chunk.Length;
        }

        return output;
    }

    private static string UrlToCacheKey(string url)
    {
        // Use the last path segments as cache key, sanitized
        var uri = new Uri(url);
        var key = uri.AbsolutePath
            .TrimStart('/')
            .Replace('/', '_')
            .Replace('\\', '_');

        // Limit length
        if (key.Length > 200) key = key[^200..];

        return key;
    }

    public void Dispose()
    {
        _cacheDir?.Dispose();
    }
}
