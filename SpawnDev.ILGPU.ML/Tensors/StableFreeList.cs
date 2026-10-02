using System.Collections;
using System.Runtime.CompilerServices;

namespace SpawnDev.ILGPU.ML.Tensors;

/// <summary>
/// A size bucket's free buffers, handed out LOWEST SEQUENCE NUMBER FIRST instead of last-returned-first.
/// </summary>
/// <remarks>
/// <para>A forward rents and returns in the same order every time, but with a LIFO stack the free list a forward
/// ENDS with is a permutation of the one it started with - so the next forward's nodes get different physical
/// buffers, cycling over many forwards. MEASURED 2026-10-02 (Anaglyphohol, DAv3 168x98 on WebGPU, Blazor WASM AOT):
/// one input size touched ~55.7k distinct (pipeline, buffer, offset) bind groups because of that rotation, and the
/// WebGPU bind-group cache hit only ~45%. Handing out the lowest-numbered free buffer makes a steady forward bind the
/// SAME buffers in the SAME places every time, on every backend - which buffer serves a tensor never changes a
/// result.</para>
/// <para>Same surface as the <see cref="Stack{T}"/> it replaces (Push, Pop, Count, enumeration). A buffer's number
/// is assigned the first time it is pushed and lives in a <see cref="ConditionalWeakTable{TKey, TValue}"/>, so a
/// disposed buffer is not kept alive by it. Buckets are small, so the sorted insert is cheap.</para>
/// </remarks>
internal sealed class StableFreeList<T> : IEnumerable<T> where T : class
{
    /// <summary>False = plain LIFO (the pre-2026-10-02 stack order) - the A/B arm of BufferPool.DeterministicReuse.</summary>
    internal static volatile bool Sorted = true;

    private static readonly ConditionalWeakTable<T, StrongBox<long>> s_sequence = new();
    private static long s_next;

    // Sorted DESCENDING by sequence number, so the lowest is at the end: Pop is O(1).
    private readonly List<(long Seq, T Item)> _items = new();

    public int Count => _items.Count;

    public void Push(T item)
    {
        if (!Sorted) { _items.Add((0, item)); return; }   // LIFO: Pop takes the last pushed
        long seq = s_sequence.GetValue(item, static _ => new StrongBox<long>(Interlocked.Increment(ref s_next))).Value;
        // First index whose seq is LOWER than this one (descending order) - insert there.
        int lo = 0, hi = _items.Count;
        while (lo < hi)
        {
            int mid = (lo + hi) >> 1;
            if (_items[mid].Seq > seq) lo = mid + 1; else hi = mid;
        }
        _items.Insert(lo, (seq, item));
    }

    public T Pop()
    {
        int last = _items.Count - 1;
        if (last < 0) throw new InvalidOperationException("StableFreeList is empty");
        var item = _items[last].Item;
        _items.RemoveAt(last);
        return item;
    }

    public IEnumerator<T> GetEnumerator()
    {
        for (int i = _items.Count - 1; i >= 0; i--) yield return _items[i].Item;
    }

    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
}
