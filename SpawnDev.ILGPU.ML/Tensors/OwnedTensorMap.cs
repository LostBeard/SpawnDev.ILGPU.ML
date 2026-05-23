namespace SpawnDev.ILGPU.ML.Tensors;

/// <summary>
/// Disposable collection of named <see cref="OwnedTensor{T}"/> outputs. Returned from
/// <c>InferenceSession.RunOwnedAsync</c> and similar pipeline methods that produce
/// multiple named results — the caller wraps the map in <c>using</c>, accesses outputs
/// by name through the indexer, and every contained tensor is disposed in one go when
/// the map goes out of scope.
///
/// <para>
/// Matches the ergonomics of Transformers.js / ONNX Runtime "outputs dictionary"
/// returns, with explicit lifetime management replacing JS garbage collection.
/// </para>
///
/// <para>
/// Construction takes ownership of the supplied dictionary; do not mutate it externally
/// after handing it to the map. Iteration order is insertion order (the underlying
/// <see cref="Dictionary{TKey, TValue}"/> contract).
/// </para>
/// </summary>
public sealed class OwnedTensorMap<T> : IDisposable where T : unmanaged
{
    private readonly Dictionary<string, OwnedTensor<T>> _tensors;
    private bool _disposed;

    /// <summary>Wrap an existing dictionary of named owned tensors. The map assumes
    /// ownership of every tensor in <paramref name="tensors"/>.</summary>
    public OwnedTensorMap(Dictionary<string, OwnedTensor<T>> tensors)
        => _tensors = tensors ?? throw new ArgumentNullException(nameof(tensors));

    /// <summary>Lookup by name.</summary>
    public OwnedTensor<T> this[string name] => _tensors[name];

    /// <summary>Output names in insertion order.</summary>
    public IEnumerable<string> Keys => _tensors.Keys;

    /// <summary>All contained tensors.</summary>
    public IEnumerable<OwnedTensor<T>> Values => _tensors.Values;

    /// <summary>Number of named outputs.</summary>
    public int Count => _tensors.Count;

    /// <summary>True if a tensor with the given name is present.</summary>
    public bool ContainsKey(string name) => _tensors.ContainsKey(name);

    /// <summary>Try-get with the standard dictionary contract.</summary>
    public bool TryGetValue(string name, out OwnedTensor<T>? tensor)
        => _tensors.TryGetValue(name, out tensor);

    /// <summary>Convenience for single-output models — returns the first (and only) tensor.</summary>
    public OwnedTensor<T> Single()
    {
        if (_tensors.Count != 1)
            throw new InvalidOperationException(
                $"OwnedTensorMap.Single() expects exactly one tensor, has {_tensors.Count}");
        foreach (var t in _tensors.Values) return t;
        throw new InvalidOperationException("unreachable");
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        foreach (var t in _tensors.Values) t.Dispose();
    }
}
