namespace SpawnDev.ILGPU.ML.Hub;

/// <summary>
/// An <see cref="IModelSource"/> over a local folder (desktop: a model set that was exported or downloaded to disk). The repo id is ignored; a file path is read relative to the folder
/// (an absolute path is used as it is, so an empty folder lets the caller name whole paths). Streams are seekable file streams.
/// </summary>
public sealed class LocalFolderModelSource : IModelSource
{
    private readonly string _root;

    public LocalFolderModelSource(string root) => _root = root ?? "";

    private string Resolve(string filePath) => Path.Combine(_root, filePath.Replace('/', Path.DirectorySeparatorChar));

    public Task<Stream> OpenAsync(string repoId, string filePath, CancellationToken cancellationToken = default)
        => Task.FromResult<Stream>(new FileStream(Resolve(filePath), FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 16, FileOptions.Asynchronous | FileOptions.RandomAccess));

    public Task<byte[]> FetchBytesAsync(string repoId, string filePath, CancellationToken cancellationToken = default)
        => File.ReadAllBytesAsync(Resolve(filePath), cancellationToken);

    /// <summary>True when the file exists (a source for the tests and examples to skip cleanly when a model set is absent).</summary>
    public bool Exists(string filePath) => File.Exists(Resolve(filePath));
}
