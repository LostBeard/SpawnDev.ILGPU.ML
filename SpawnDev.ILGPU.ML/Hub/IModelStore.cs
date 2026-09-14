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

/// <summary>What a store knows about an entry, including a partial one.</summary>
/// <param name="Exists">An entry (complete or partial) is present.</param>
/// <param name="Complete">The entry is whole and safe to serve.</param>
/// <param name="BytesWritten">Bytes the store has CONFIRMED on disk - the resume point, never a buffered count.</param>
/// <param name="TotalBytes">Expected final size, or -1 when unknown.</param>
/// <param name="SourceRef">Where the bytes came from (a URL), or null/empty when they were handed in directly.</param>
/// <param name="ETag">Origin validator recorded at download time, for <c>If-Range</c> on resume.</param>
public readonly record struct ModelStoreState(
    bool Exists, bool Complete, long BytesWritten, long TotalBytes, string? SourceRef, string? ETag);

/// <summary>
/// A store that can be written INCREMENTALLY, so an interrupted transfer resumes instead of restarting.
/// </summary>
/// <remarks>
/// <para>
/// This is what lets the transport and the storage be separate things. A downloader needs three
/// store-shaped capabilities and nothing more: ask what is already here, open a writer positioned at the
/// resume point, and record what landed. It does not need to know the store is OPFS, and the store does not
/// need to know the bytes arrived over HTTP.
/// </para>
/// <para>
/// 🔴 <see cref="ModelStoreState.BytesWritten"/> must be what is DURABLE, never what has merely been
/// received. A resume trusts it, so counting buffered-but-unwritten bytes there corrupts the file at the
/// seam.
/// </para>
/// </remarks>
public interface IResumableModelStore : IModelStore
{
    /// <summary>What the store holds for <paramref name="key"/>, including a partial entry.</summary>
    Task<ModelStoreState> GetStateAsync(string key, CancellationToken cancellationToken = default);

    /// <summary>
    /// Open <paramref name="key"/> for writing positioned at <paramref name="startOffset"/>, discarding
    /// anything already stored beyond it. Pass 0 to start over. The caller disposes the stream.
    /// </summary>
    /// <remarks>Truncating is deliberate rather than merely seeking: bytes past the confirmed resume point
    /// were never acknowledged, and keeping them would leave unverified data inside a file that later
    /// reports itself complete.</remarks>
    Task<Stream> OpenWriteAsync(string key, long startOffset, CancellationToken cancellationToken = default);

    /// <summary>
    /// Record an entry's state. Call with <paramref name="complete"/> false to checkpoint a resume point
    /// mid-transfer, and true exactly once the whole file is durable.
    /// </summary>
    Task SetStateAsync(string key, string sourceRef, long totalBytes, long bytesWritten, bool complete,
        string? etag, CancellationToken cancellationToken = default);
}
