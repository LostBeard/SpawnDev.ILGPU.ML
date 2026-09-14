using System.Text;

namespace SpawnDev.ILGPU.ML.Hub;

/// <summary>
/// An <see cref="IResumableModelStore"/> over the local filesystem - the desktop counterpart to
/// <see cref="OpfsModelCache"/>.
/// </summary>
/// <remarks>
/// <para>
/// Same contract, same sidecar format, different substrate. A console app, a WPF app or a test harness
/// gets resumable downloads, truncation refusal and cache management without OPFS and without a browser.
/// </para>
/// <para>
/// Bytes here necessarily pass through managed buffers - that is what a <see cref="FileStream"/> is. The
/// "bulk data stays in JS" rule is about the browser, where the WASM managed heap is small and the JS
/// crossing IS the cost; on desktop a 16 MiB copy buffer is simply how file I/O works.
/// </para>
/// </remarks>
public class FileModelStore : IResumableModelStore
{
    private const string MetaSuffix = ".meta";

    private readonly string _root;

    /// <summary>Copy buffer for <see cref="PutAsync"/>, and the write size a downloader should aim for.</summary>
    public int BufferSize { get; set; } = 16 * 1024 * 1024;

    /// <summary>Create a store rooted at <paramref name="directory"/>, creating it if needed.</summary>
    public FileModelStore(string directory)
    {
        _root = directory ?? throw new ArgumentNullException(nameof(directory));
        Directory.CreateDirectory(_root);
    }

    /// <summary>The default per-user location: <c>%LOCALAPPDATA%/SpawnDev.ILGPU.ML/models</c> (or the XDG
    /// equivalent), so several apps and repeated runs share one cache instead of re-downloading.</summary>
    public static FileModelStore Default() => new(Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
        "SpawnDev.ILGPU.ML", "models"));

    /// <summary>The directory this store writes to.</summary>
    public string RootDirectory => _root;

    private string PathFor(string key) => Path.Combine(_root, Sanitize(key));
    private string MetaPathFor(string key) => PathFor(key) + MetaSuffix;

    /// <summary>Keys come from URLs and repo paths, so they can contain characters a filesystem refuses.</summary>
    private static string Sanitize(string key)
    {
        var sb = new StringBuilder(key.Length);
        foreach (var c in key) sb.Append(Array.IndexOf(Path.GetInvalidFileNameChars(), c) >= 0 ? '_' : c);
        return sb.ToString();
    }

    /// <inheritdoc/>
    public async Task<bool> ExistsAsync(string key, CancellationToken cancellationToken = default)
        => (await GetStateAsync(key, cancellationToken).ConfigureAwait(false)).Complete;

    /// <inheritdoc/>
    public async Task<Stream?> OpenReadAsync(string key, CancellationToken cancellationToken = default)
    {
        if (!await ExistsAsync(key, cancellationToken).ConfigureAwait(false)) return null;
        return new FileStream(PathFor(key), FileMode.Open, FileAccess.Read, FileShare.Read,
            bufferSize: 1 << 20, useAsync: true);
    }

    /// <inheritdoc/>
    public async Task PutAsync(string key, Stream source, IProgress<ModelDownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(source);
        long expected = -1;
        try { if (source.CanSeek) expected = source.Length - source.Position; } catch { /* unknowable */ }

        // Clear prior state first: a stale sidecar saying "complete" must not survive a put that then fails
        // partway, or the truncated result would be served as a finished model.
        await RemoveAsync(key, cancellationToken).ConfigureAwait(false);

        long written;
        var path = PathFor(key);
        var dest = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None,
            bufferSize: 1 << 20, useAsync: true);
        await using (dest.ConfigureAwait(false))
        {
            await source.CopyToAsync(dest, BufferSize, cancellationToken).ConfigureAwait(false);
            await dest.FlushAsync(cancellationToken).ConfigureAwait(false);
            written = dest.Length;
        }

        if (expected >= 0 && written != expected)
            throw new IOException($"Storing '{key}' copied {written} bytes but the source reported {expected}.");

        await SetStateAsync(key, "", written, written, true, null, cancellationToken).ConfigureAwait(false);
        progress?.Report(new ModelDownloadProgress(written, written, false));
    }

    /// <inheritdoc/>
    public Task RemoveAsync(string key, CancellationToken cancellationToken = default)
    {
        try { File.Delete(PathFor(key)); } catch (DirectoryNotFoundException) { }
        try { File.Delete(MetaPathFor(key)); } catch (DirectoryNotFoundException) { }
        return Task.CompletedTask;
    }

    /// <inheritdoc/>
    public async Task<IReadOnlyList<ModelStoreEntry>> ListAsync(CancellationToken cancellationToken = default)
    {
        var result = new List<ModelStoreEntry>();
        if (!Directory.Exists(_root)) return result;
        foreach (var path in Directory.EnumerateFiles(_root))
        {
            if (path.EndsWith(MetaSuffix, StringComparison.Ordinal)) continue;
            var key = Path.GetFileName(path);
            long size;
            try { size = new FileInfo(path).Length; } catch { continue; }
            var state = await GetStateAsync(key, cancellationToken).ConfigureAwait(false);
            result.Add(new ModelStoreEntry(key, size, state.Complete));
        }
        return result;
    }

    /// <inheritdoc/>
    public Task<long> GetTotalSizeAsync(CancellationToken cancellationToken = default)
    {
        if (!Directory.Exists(_root)) return Task.FromResult(0L);
        long total = 0;
        foreach (var path in Directory.EnumerateFiles(_root))
        {
            try { total += new FileInfo(path).Length; } catch { /* raced with a delete */ }
        }
        return Task.FromResult(total);
    }

    /// <inheritdoc/>
    public Task ClearAsync(CancellationToken cancellationToken = default)
    {
        if (!Directory.Exists(_root)) return Task.CompletedTask;
        foreach (var path in Directory.EnumerateFiles(_root))
        {
            try { File.Delete(path); } catch { /* in use or raced */ }
        }
        return Task.CompletedTask;
    }

    /// <inheritdoc/>
    public async Task<ModelStoreState> GetStateAsync(string key, CancellationToken cancellationToken = default)
    {
        var path = PathFor(key);
        var onDisk = File.Exists(path) ? new FileInfo(path).Length : -1;
        var meta = await ReadMetaAsync(key, cancellationToken).ConfigureAwait(false);
        if (onDisk < 0 && meta == null) return default;

        // Confirmed bytes are the LESSER of what the sidecar acknowledged and what is actually on disk -
        // either can lead the other after a crash, and a resume must trust only what both agree on.
        var written = meta == null
            ? Math.Max(0, onDisk)
            : Math.Max(0, Math.Min(onDisk < 0 ? 0 : onDisk, meta.Value.Received));
        var complete = meta is { Complete: true } && onDisk >= 0
                       && (meta.Value.Total < 0 || meta.Value.Total == onDisk);
        return new ModelStoreState(onDisk >= 0, complete, complete ? onDisk : written,
            meta?.Total ?? -1, meta?.Url, meta?.ETag);
    }

    /// <inheritdoc/>
    public Task<Stream> OpenWriteAsync(string key, long startOffset, CancellationToken cancellationToken = default)
    {
        if (startOffset < 0) throw new ArgumentOutOfRangeException(nameof(startOffset));
        var stream = new FileStream(PathFor(key), FileMode.OpenOrCreate, FileAccess.Write, FileShare.None,
            bufferSize: 1 << 20, useAsync: true);
        // Truncate rather than merely seek: bytes past the confirmed resume point were never acknowledged,
        // and keeping them would bury unverified data inside a file that later reports itself complete.
        stream.SetLength(startOffset);
        stream.Seek(startOffset, SeekOrigin.Begin);
        return Task.FromResult<Stream>(stream);
    }

    /// <inheritdoc/>
    public async Task SetStateAsync(string key, string sourceRef, long totalBytes, long bytesWritten,
        bool complete, string? etag, CancellationToken cancellationToken = default)
    {
        var text = new StringBuilder()
            .Append("v=1\n")
            .Append("url=").Append(sourceRef ?? "").Append('\n')
            .Append("total=").Append(totalBytes).Append('\n')
            .Append("received=").Append(bytesWritten).Append('\n')
            .Append("complete=").Append(complete ? '1' : '0').Append('\n')
            .Append("etag=").Append(etag ?? "").Append('\n')
            .ToString();
        await File.WriteAllTextAsync(MetaPathFor(key), text, cancellationToken).ConfigureAwait(false);
    }

    private readonly record struct Meta(string Url, long Total, long Received, bool Complete, string? ETag);

    private async Task<Meta?> ReadMetaAsync(string key, CancellationToken ct)
    {
        var path = MetaPathFor(key);
        if (!File.Exists(path)) return null;
        try
        {
            var text = await File.ReadAllTextAsync(path, ct).ConfigureAwait(false);
            string url = ""; long total = -1, received = 0; bool complete = false; string? etag = null;
            foreach (var line in text.Split('\n'))
            {
                var eq = line.IndexOf('=');
                if (eq <= 0) continue;
                var name = line[..eq].Trim();
                var value = line[(eq + 1)..].Trim('\r', ' ');
                switch (name)
                {
                    case "url": url = value; break;
                    case "total": total = long.TryParse(value, out var t) ? t : -1; break;
                    case "received": received = long.TryParse(value, out var r) ? r : 0; break;
                    case "complete": complete = value == "1"; break;
                    case "etag": etag = string.IsNullOrEmpty(value) ? null : value; break;
                }
            }
            return new Meta(url, total, received, complete, etag);
        }
        catch { return null; }
    }
}
