using SpawnDev.ILGPU.ML.Hub;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// A seekable stream over one model file from the SpawnDev models host (https://models.spawndev.com, Hugging Face as the fallback), read with HTTP range
/// requests: what many tests used the WebTorrent <c>HubModelStream</c> for, without the torrent. A test that only needs the bytes of a model has no
/// business depending on a hub building a magnet for it; the tests that are ABOUT WebTorrent (<c>MLTestBase.WebTorrentTests</c>) still use the
/// torrent adapter.
/// </summary>
/// <remarks>
/// Browser: <see cref="FetchRangeStream"/> (JS-side <c>fetch</c> ranges, an <c>IJSReadStream</c>: the bytes never enter the .NET heap, as with the torrent
/// stream). Desktop: <see cref="HttpRangeStream"/>. The call sites keep the shape they had (<c>OpenAsync(repo, file, deselect, ct)</c>, <c>.Stream</c>,
/// <c>PrepareTimeout</c>, <c>await client.DisposeAsync()</c>) so a test changes by the line that built the loader.
/// </remarks>
public sealed class ModelsHostStreamLoader : IModelSource, IAsyncDisposable
{
    private readonly HttpClient _http;

    public ModelsHostStreamLoader(HttpClient http) => _http = http ?? throw new ArgumentNullException(nameof(http));

    /// <summary>Kept so call sites written for the torrent loader compile unchanged. Static files have nothing to prepare, so it is not used.</summary>
    public TimeSpan PrepareTimeout { get; set; }

    /// <summary>The opened model: the stream (the torrent loader's result also carried the torrent and file, which no test used).</summary>
    public sealed record LoadedModel(Stream Stream)
    {
        /// <summary>The file's size in bytes.</summary>
        public long Length => Stream.Length;
    }

    /// <param name="deselect">Kept for call-site compatibility with the torrent loader; ignored.</param>
    public async Task<LoadedModel> OpenAsync(string repoId, string filePath, bool deselect = false, CancellationToken cancellationToken = default)
    {
        var url = HuggingFaceClient.GetDownloadUrl(repoId, filePath);
        long length = await ModelsHost.ProbeSizeAsync(_http, url, cancellationToken).ConfigureAwait(false);
        if (length <= 0) throw new IOException($"Could not determine the size of {url}");
        Stream stream = OperatingSystem.IsBrowser()
            ? new FetchRangeStream(url, length)
            : new HttpRangeStream(_http, url, length);
        return new LoadedModel(stream);
    }

    /// <inheritdoc/>
    async Task<Stream> IModelSource.OpenAsync(string repoId, string filePath, CancellationToken cancellationToken)
        => (await OpenAsync(repoId, filePath, false, cancellationToken).ConfigureAwait(false)).Stream;

    /// <inheritdoc/>
    async Task<byte[]> IModelSource.FetchBytesAsync(string repoId, string filePath, CancellationToken cancellationToken)
    {
        var bytes = await new HuggingFaceClient(_http).DownloadFileAsync(repoId, filePath).ConfigureAwait(false);
        cancellationToken.ThrowIfCancellationRequested();
        return bytes;
    }

    public ValueTask DisposeAsync() => ValueTask.CompletedTask;
}
