using ILGPU;
using ILGPU.Runtime;
using System.Runtime.InteropServices;

namespace SpawnDev.ILGPU.ML.Kernels;

/// <summary>
/// Content-addressed, WRITE-ONCE device copies of small kernel params arrays (strides, shapes, counts).
/// <para>
/// A params array is a pure function of the op's shapes, so one device buffer per distinct content can serve
/// every call with that content, forever: it is uploaded once at creation and never written again. That makes it
/// safe on every backend without a sync point - a dispatch still pending in an un-submitted WebGPU encoder, on the
/// Wasm worker pool, or recorded into a captured dispatch plan always reads exactly the params it was issued with,
/// because nothing ever overwrites or frees that buffer before <see cref="Dispose"/>.
/// </para>
/// <para>
/// Replaces the per-call "allocate a FRESH buffer, retire the previous one to a list freed in Dispose()" pattern,
/// which was race-free but grew without bound: every call leaked one buffer for the kernel instance's lifetime.
/// SpawnScene's TruckFull depth cascade (83 DAv3 passes on one pipeline) accumulated ~19,000 of them - ~234 per
/// pass from Broadcast/Slice/Gather alone - and the browser GPU process died out of memory (2026-09-27).
/// With content addressing a fixed-shape model stops allocating after its first pass; a dynamic-shape model holds
/// one buffer per DISTINCT params content, never more buffers than the old pattern did.
/// </para>
/// </summary>
internal sealed class ContentParamBuffers<T> : IDisposable where T : unmanaged
{
    readonly Accelerator _accelerator;
    readonly Dictionary<T[], MemoryBuffer1D<T, Stride1D.Dense>> _byContent = new(BitwiseContentComparer.Instance);
    readonly object _lock = new();

    public ContentParamBuffers(Accelerator accelerator) => _accelerator = accelerator;

    /// <summary>Distinct params contents held (one device buffer each).</summary>
    public int Count { get { lock (_lock) return _byContent.Count; } }

    /// <summary>
    /// A device view holding exactly <paramref name="data"/>. The first request for a content allocates and uploads it;
    /// later requests return the same buffer. The caller may reuse or mutate <paramref name="data"/> afterwards.
    /// </summary>
    public ArrayView1D<T, Stride1D.Dense> Get(T[] data)
    {
        lock (_lock)
        {
            if (!_byContent.TryGetValue(data, out var buf))
            {
                buf = _accelerator.Allocate1D(data);
                _byContent.Add((T[])data.Clone(), buf);   // own the key: a caller mutating its array must not corrupt the lookup
            }
            return buf.View;
        }
    }

    public void Dispose()
    {
        lock (_lock)
        {
            foreach (var buf in _byContent.Values) buf.Dispose();
            _byContent.Clear();
        }
    }

    /// <summary>Bitwise equality, so float params compare by representation (NaN == NaN, -0 != +0) like the device sees them.</summary>
    sealed class BitwiseContentComparer : IEqualityComparer<T[]>
    {
        public static readonly BitwiseContentComparer Instance = new();

        public bool Equals(T[]? x, T[]? y)
        {
            if (ReferenceEquals(x, y)) return true;
            if (x == null || y == null || x.Length != y.Length) return false;
            return MemoryMarshal.AsBytes(x.AsSpan()).SequenceEqual(MemoryMarshal.AsBytes(y.AsSpan()));
        }

        public int GetHashCode(T[] obj)
        {
            var h = new HashCode();
            h.AddBytes(MemoryMarshal.AsBytes(obj.AsSpan()));
            return h.ToHashCode();
        }
    }
}
