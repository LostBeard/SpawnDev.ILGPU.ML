using SpawnDev.BlazorJS;
using SpawnDev.BlazorJS.JSObjects;

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
    private readonly BlazorJSRuntime _js;
    private FileSystemDirectoryHandle? _cacheDir;
    private bool _initialized;

    /// <summary>Name of the OPFS subdirectory for cached models.</summary>
    public string CacheDirectoryName { get; set; } = "ilgpu-ml-models";

    /// <summary>
    /// Fired during model download with (bytesReceived, totalBytes).
    /// totalBytes may be -1 if the server doesn't provide Content-Length.
    /// </summary>
    public event Action<long, long>? OnDownloadProgress;

    public ModelCache(BlazorJSRuntime js)
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
