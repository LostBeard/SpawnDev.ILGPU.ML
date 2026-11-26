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
        var path = PathFor(key);
        return await RetrySharingAsync<Stream?>(() => Task.FromResult<Stream?>(new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read,
            bufferSize: 1 << 20, useAsync: true)), cancellationToken).ConfigureAwait(false);
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
    public async Task RemoveAsync(string key, CancellationToken cancellationToken = default)
    {
        // retried like the sidecar: PutAsync removes first, and a scanner holding the old file must not fail the put
        foreach (var path in new[] { PathFor(key), MetaPathFor(key) })
            await RetrySharingAsync(() => { try { File.Delete(path); } catch (DirectoryNotFoundException) { } return Task.FromResult(0); }, cancellationToken).ConfigureAwait(false);
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
    /// <remarks>Written to a temporary file and moved over the sidecar, so a reader sees the old record or the new one, never a half-written one;
    /// the temporary name ends in the sidecar suffix, so a listing never takes it for a model.</remarks>
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
        var meta = MetaPathFor(key);
        var temp = PathFor(key) + "." + Guid.NewGuid().ToString("N") + ".tmp" + MetaSuffix;
        try
        {
            await File.WriteAllTextAsync(temp, text, cancellationToken).ConfigureAwait(false);
            await RetrySharingAsync(() => { ReplaceAtomically(temp, meta); return Task.FromResult(0); }, cancellationToken).ConfigureAwait(false);
        }
        catch
        {
            try { File.Delete(temp); } catch { /* best effort: the original failure is the one to report */ }
            throw;
        }
    }

    /// <summary>
    /// Moves <paramref name="source"/> over <paramref name="dest"/> in one step. On Windows File.Move cannot replace a file that is open, even by a reader that shares
    /// delete (measured: ERROR_ACCESS_DENIED for as long as the reader holds it; on-access scanners open files exactly that way). A POSIX-semantics rename
    /// (FileRenameInfoEx, NTFS since Windows 10 1709) replaces the name at once and leaves open handles on the old file; where the file system does not support it
    /// (FAT, some network shares) this falls back to File.Move, which the caller retries while the target is held. Elsewhere rename already replaces atomically.
    /// </summary>
    private static void ReplaceAtomically(string source, string dest)
    {
        if (OperatingSystem.IsWindows() && PosixRename.TryReplace(source, Path.GetFullPath(dest))) return;
        File.Move(source, dest, overwrite: true);
    }

    private static class PosixRename
    {
        private const uint DELETE = 0x00010000, FILE_ATTRIBUTE_NORMAL = 0x80, OPEN_EXISTING = 3;
        private const int FileRenameInfoEx = 22;
        private const uint FILE_RENAME_FLAG_REPLACE_IF_EXISTS = 0x1, FILE_RENAME_FLAG_POSIX_SEMANTICS = 0x2;
        private const int ERROR_INVALID_FUNCTION = 1, ERROR_NOT_SUPPORTED = 50, ERROR_INVALID_PARAMETER = 87;

        [System.Runtime.InteropServices.DllImport("kernel32.dll", EntryPoint = "CreateFileW", SetLastError = true, CharSet = System.Runtime.InteropServices.CharSet.Unicode)]
        private static extern Microsoft.Win32.SafeHandles.SafeFileHandle CreateFile(string name, uint access, FileShare share, IntPtr security, uint disposition, uint flags, IntPtr template);

        [System.Runtime.InteropServices.DllImport("kernel32.dll", SetLastError = true)]
        [return: System.Runtime.InteropServices.MarshalAs(System.Runtime.InteropServices.UnmanagedType.Bool)]
        private static extern bool SetFileInformationByHandle(Microsoft.Win32.SafeHandles.SafeFileHandle file, int infoClass, IntPtr info, uint size);

        /// <summary>True when replaced; false when the file system does not support it (the caller falls back). Any other failure throws, mapped like .NET's own
        /// (ERROR_ACCESS_DENIED -> UnauthorizedAccessException, a sharing violation -> IOException with its HResult) so the caller's retry sees it.</summary>
        public static bool TryReplace(string source, string fullDest)
        {
            using var handle = CreateFile(source, DELETE, FileShare.ReadWrite | FileShare.Delete, IntPtr.Zero, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, IntPtr.Zero);
            if (handle.IsInvalid) throw Error(System.Runtime.InteropServices.Marshal.GetLastPInvokeError(), source);
            // FILE_RENAME_INFO: { ULONG Flags (union with BOOLEAN ReplaceIfExists); HANDLE RootDirectory; DWORD FileNameLength (bytes); WCHAR FileName[] }
            int rootOffset = IntPtr.Size, lengthOffset = rootOffset + IntPtr.Size, nameOffset = lengthOffset + 4;
            int nameBytes = fullDest.Length * 2;
            int size = nameOffset + nameBytes + 2;
            IntPtr info = System.Runtime.InteropServices.Marshal.AllocHGlobal(size);
            try
            {
                for (int i = 0; i < size; i++) System.Runtime.InteropServices.Marshal.WriteByte(info, i, 0);
                System.Runtime.InteropServices.Marshal.WriteInt32(info, 0, (int)(FILE_RENAME_FLAG_REPLACE_IF_EXISTS | FILE_RENAME_FLAG_POSIX_SEMANTICS));
                System.Runtime.InteropServices.Marshal.WriteInt32(info, lengthOffset, nameBytes);
                System.Runtime.InteropServices.Marshal.Copy(fullDest.ToCharArray(), 0, info + nameOffset, fullDest.Length);
                if (SetFileInformationByHandle(handle, FileRenameInfoEx, info, (uint)size)) return true;
                int err = System.Runtime.InteropServices.Marshal.GetLastPInvokeError();
                if (err is ERROR_INVALID_FUNCTION or ERROR_NOT_SUPPORTED or ERROR_INVALID_PARAMETER) return false;
                throw Error(err, fullDest);
            }
            finally { System.Runtime.InteropServices.Marshal.FreeHGlobal(info); }
        }

        private static Exception Error(int err, string path)
        {
            var e = System.Runtime.InteropServices.Marshal.GetExceptionForHR(unchecked((int)0x80070000) | err) ?? new IOException($"Win32 error {err}");
            return e is UnauthorizedAccessException ? new UnauthorizedAccessException($"Access to '{path}' is denied.", e)
                : e is IOException io ? new IOException($"Replacing '{path}': {io.Message}", io.HResult) : e;
        }
    }

    private readonly record struct Meta(string Url, long Total, long Received, bool Complete, string? ETag);

    /// <summary>How long a sidecar held open by another process (an on-access scanner, an indexer, a backup tool) is waited for: attempts and the
    /// first delay, doubled per attempt up to <see cref="SharingRetryMaxDelayMs"/> (10 attempts, about 2.1 s of waiting in all by default).</summary>
    public int SharingRetryAttempts { get; set; } = 10;
    /// <summary>The first delay between attempts on a sharing violation, in milliseconds.</summary>
    public int SharingRetryDelayMs { get; set; } = 10;
    /// <summary>The longest delay between attempts on a sharing violation, in milliseconds.</summary>
    public int SharingRetryMaxDelayMs { get; set; } = 500;

    /// <summary>A file another handle holds: opening it reports ERROR_SHARING_VIOLATION (32) or ERROR_LOCK_VIOLATION (33); replacing or deleting it reports
    /// ERROR_ACCESS_DENIED (UnauthorizedAccessException) - measured: File.Move over a sidecar a reader holds without sharing. A real permission problem is also
    /// access denied; it costs the retries and then surfaces unchanged.</summary>
    private static bool IsHeldElsewhere(Exception e) => e is UnauthorizedAccessException
        || e is IOException io && io is not FileNotFoundException and not DirectoryNotFoundException && (io.HResult & 0xFFFF) is 32 or 33;

    private async Task<T> RetrySharingAsync<T>(Func<Task<T>> op, CancellationToken ct)
    {
        int delay = Math.Max(1, SharingRetryDelayMs);
        for (int attempt = 1; ; attempt++)
        {
            try { return await op().ConfigureAwait(false); }
            catch (Exception e) when (IsHeldElsewhere(e) && attempt < SharingRetryAttempts)
            {
                await Task.Delay(delay, ct).ConfigureAwait(false);
                delay = Math.Min(delay * 2, Math.Max(delay, SharingRetryMaxDelayMs));
            }
        }
    }

    /// <summary>
    /// Reads the sidecar. Null means there is none. A sidecar that exists but cannot be read THROWS (after the sharing retries): reading it as
    /// absent made a complete model look like an unrecorded one, and the downloader then started over from byte 0 - truncating a multi-GB file
    /// because a scanner held a 200-byte sidecar for a moment (Lore, 2026-11-26: SD 1.5's UNet and VAE sidecars, "used by another process").
    /// </summary>
    private async Task<Meta?> ReadMetaAsync(string key, CancellationToken ct)
    {
        var path = MetaPathFor(key);
        string? text = await RetrySharingAsync<string?>(async () =>
        {
            try
            {
                // Share everything: this read must never be the reason a writer's replace or another reader fails.
                var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete, bufferSize: 4096, useAsync: true);
                await using (fs.ConfigureAwait(false))
                using (var reader = new StreamReader(fs, Encoding.UTF8))
                    return await reader.ReadToEndAsync(ct).ConfigureAwait(false);
            }
            catch (FileNotFoundException) { return null; }
            catch (DirectoryNotFoundException) { return null; }
        }, ct).ConfigureAwait(false);
        if (text == null) return null;
        {
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
    }
}
