namespace SpawnDev.ILGPU.ML.Hub;

/// <summary>One entry in an <see cref="IModelStore"/>.</summary>
/// <param name="Key">The key the entry is stored under.</param>
/// <param name="SizeBytes">Bytes currently held for this entry (a partial entry is smaller than its total).</param>
/// <param name="Complete">False when the entry is a partial - resumable, but NOT usable.</param>
public readonly record struct ModelStoreEntry(string Key, long SizeBytes, bool Complete);

/// <summary>
/// Stores model files by key and hands them back as a <see cref="Stream"/>, wherever they came from.
/// </summary>
/// <remarks>
/// <para>
/// <b>Why the currency is <see cref="Stream"/>.</b> Everything moves as a <c>Stream</c> because a
/// JS-backed stream also implements <c>IJSReadStream</c> / <c>IJSWriteStream</c>, and
/// <c>JSReadStreamBase</c> OVERRIDES <see cref="Stream.CopyToAsync(Stream, int, CancellationToken)"/> to
/// pump <c>Uint8Array</c> chunks JS-side whenever the destination is an <c>IJSWriteStream</c>, falling back
/// to the managed path otherwise. So <c>await source.CopyToAsync(destination)</c> is zero-copy when both
/// ends are JS-side and still correct for a <c>FileStream</c> or a <c>MemoryStream</c>. A plain
/// <c>Stream</c> parameter therefore costs nothing and buys every source.
/// </para>
/// <para>
/// <b>Why this interface exists at all.</b> Model delivery was built directly on WebTorrent, so a consumer
/// could not get a model without a <c>WebTorrentClient</c> - the transport had been welded to the storage.
/// The store does not know or care whether the bytes arrived over HTTP from the hub, out of a torrent, off
/// local disk, or from a file the user picked. It takes a <c>Stream</c> and a key.
/// </para>
/// </remarks>
public interface IModelStore
{
    /// <summary>True when <paramref name="key"/> is present AND complete - i.e. usable without a network.</summary>
    /// <remarks>A partial entry reports false: it is resumable, not usable. Reporting a half-downloaded
    /// model as present is how a truncated file reaches a parser and fails as a confusing format error.</remarks>
    Task<bool> ExistsAsync(string key, CancellationToken cancellationToken = default);

    /// <summary>
    /// Open a complete entry for reading, or null when it is absent or incomplete. The caller disposes it.
    /// </summary>
    /// <remarks>In a browser store this is an <c>IJSReadStream</c>, which is what lets
    /// <c>InferenceSession.CreateFromOnnxStreamAsync</c> upload each weight JS-&gt;GPU without the model
    /// entering the .NET/WASM managed heap.</remarks>
    Task<Stream?> OpenReadAsync(string key, CancellationToken cancellationToken = default);

    /// <summary>
    /// Store <paramref name="source"/> under <paramref name="key"/>, replacing any existing entry, and mark
    /// it complete. The caller owns <paramref name="source"/> and disposes it.
    /// </summary>
    /// <remarks>
    /// The copy goes through <see cref="Stream.CopyToAsync(Stream, int, CancellationToken)"/>, so a JS-side
    /// source into a JS-side store never touches the managed heap. This is the "store it wherever it came
    /// from" entry point: a torrent stream, an HTTP response, a picked file and a <c>MemoryStream</c> all
    /// arrive the same way.
    /// </remarks>
    /// <param name="key">Key to store under.</param>
    /// <param name="source">The bytes. Read from its current position to its end.</param>
    /// <param name="progress">Optional progress; <c>TotalBytes</c> is -1 when the source length is unknown.</param>
    /// <param name="cancellationToken">Cancels the copy. A cancelled put leaves no complete entry.</param>
    Task PutAsync(string key, Stream source, IProgress<ModelDownloadProgress>? progress = null,
        CancellationToken cancellationToken = default);

    /// <summary>Delete an entry and any bookkeeping that belongs to it.</summary>
    Task RemoveAsync(string key, CancellationToken cancellationToken = default);

    /// <summary>Every entry, with its real size and whether it is complete.</summary>
    Task<IReadOnlyList<ModelStoreEntry>> ListAsync(CancellationToken cancellationToken = default);

    /// <summary>Total bytes held by the store.</summary>
    Task<long> GetTotalSizeAsync(CancellationToken cancellationToken = default);
}
