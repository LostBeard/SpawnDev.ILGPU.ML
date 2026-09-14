namespace SpawnDev.ILGPU.ML.Hub;

/// <summary>
/// Resolves a model file, addressed by repo and path, into a readable <see cref="Stream"/>.
/// </summary>
/// <remarks>
/// <para>
/// This is the TRANSPORT half of model delivery; <see cref="IModelStore"/> is the storage half. A pipeline
/// depends on this interface and therefore on no transport in particular: <see cref="HubModelSource"/>
/// serves models over plain HTTP from the hub with an OPFS cache, and <c>HubModelStream</c> serves them as
/// lazy-hash torrents. Same pipeline call, different source.
/// </para>
/// <para>
/// ⚠️ We got here by welding the pipelines to <c>HubModelStream</c>, which needs a <c>WebTorrentClient</c> -
/// so depth estimation and image generation could not be used at all without WebTorrent, and every
/// WebTorrent problem became an ML problem. The transport is a detail; it belongs behind an interface.
/// </para>
/// <para>
/// Everything is a <see cref="Stream"/> because a JS-backed stream also implements <c>IJSReadStream</c>,
/// and <c>InferenceSession</c> detects that to upload weights JS-&gt;GPU without the model entering the
/// managed heap. The interface costs nothing and admits every source.
/// </para>
/// </remarks>
public interface IModelSource
{
    /// <summary>
    /// Open a model file as a readable stream. Seekable where the transport allows it, which is what lets
    /// the ONNX reader skip past weight blobs instead of downloading them.
    /// </summary>
    /// <param name="repoId">Repository id, e.g. <c>onnx-community/depth-anything-v2-small</c>.</param>
    /// <param name="filePath">Path within the repo, e.g. <c>onnx/model.onnx</c>.</param>
    /// <param name="cancellationToken">Cancellation.</param>
    /// <returns>The stream. The caller owns and disposes it.</returns>
    Task<Stream> OpenAsync(string repoId, string filePath, CancellationToken cancellationToken = default);

    /// <summary>
    /// Fetch a SMALL file whole. For a KB-scale structure or tokenizer file - an external-data model's
    /// <c>model.onnx</c> holds only the graph, with weights in a sibling <c>model.onnx_data</c>.
    /// </summary>
    /// <remarks>
    /// 🔴 Never use this for weights. It returns the file on the .NET/WASM managed heap; that is fine for a
    /// vocab or a graph, and it is the rule-breaking OOM path for a 1.7 GB U-Net. Weights go through
    /// <see cref="OpenAsync"/>.
    /// </remarks>
    Task<byte[]> FetchBytesAsync(string repoId, string filePath, CancellationToken cancellationToken = default);
}

/// <summary>
/// An <see cref="IModelSource"/> that caches into an <see cref="IModelStore"/> and can report what it
/// holds and what is downloading - enough to drive a model-management UI.
/// </summary>
/// <remarks>
/// Separate from <see cref="IModelSource"/> because not every source caches: a plain ranged-HTTP reader or
/// a local-file source delivers models without a store behind it, and forcing those to invent one would
/// make the interface a lie. A caller that wants cache state tests for this interface and degrades when it
/// is absent.
/// </remarks>
public interface ICachingModelSource : IModelSource
{
    /// <summary>The store being filled, for listing, sizing and eviction.</summary>
    IModelStore Store { get; }

    /// <summary>The store key this source uses for a repo file - so a caller can ask the store about it.</summary>
    string CacheKey(string repoId, string filePath);

    /// <summary>Downloads in flight through this source. Empty when the source does not track them.</summary>
    IReadOnlyList<ActiveModelDownload> ActiveDownloads { get; }
}
