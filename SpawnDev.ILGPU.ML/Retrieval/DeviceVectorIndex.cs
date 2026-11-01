using System.Runtime.InteropServices;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Kernels;

namespace SpawnDev.ILGPU.ML.Retrieval;

using FV = ArrayView1D<float, Stride1D.Dense>;
using IV = ArrayView1D<int, Stride1D.Dense>;

/// <summary>
/// Vectors kept on the device and searched there: each row a slot of <see cref="Dim"/> floats, a search scores every live slot as
/// <c>dotWeight * (row . query) + addend[slot]</c> (cosine when rows and query are unit length), drops the slots <c>allow</c> masks out, and ranks the
/// best <c>k</c> on the device (<see cref="TopKKernels"/>: exact, ties to the lower slot), so one search reads back k (slot, score) pairs, never a row.
/// </summary>
/// <remarks>
/// <para>
/// Slots: <see cref="Add(float[])"/> reuses the lowest freed slot, else takes the next one; <see cref="AddRange(ReadOnlySpan{float}, int)"/> takes
/// consecutive new slots in one upload. The index grows by doubling (<see cref="CapacityChanged"/> fires after it, with the new <see cref="Capacity"/>, so
/// per-slot arrays kept beside it can follow). <see cref="Remove"/> zeroes the row on the device (a forgotten vector does not linger in GPU memory) and frees
/// the slot.
/// </para>
/// <para>
/// WebGL (no atomic slots for the ranking): the scores are read and ranked on the host, the same order. One call at a time per index.
/// </para>
/// </remarks>
public sealed class DeviceVectorIndex : IDisposable
{
    private readonly Accelerator _acc;
    private MemoryBuffer1D<float, Stride1D.Dense> _rows;
    private MemoryBuffer1D<int, Stride1D.Dense> _valid;
    private readonly List<IDisposable> _retired = new();       // replaced buffers: a queued dispatch may still read them; freed with the index
    private readonly SortedSet<int> _free = new();
    private bool[] _live;
    private int _high;                                          // slots below it have been used (live or freed)
    private MemoryBuffer1D<float, Stride1D.Dense>? _scores, _maxV, _selV, _cand, _query, _noF;
    private MemoryBuffer1D<int, Stride1D.Dense>? _maxI, _selI, _selN, _noI;
    private Action<Index1D, FV, FV, IV, IV, FV, FV, int, int, float>? _score;
    private Action<Index1D, FV, FV, IV, int, int, int>? _chunkMax;
    private Action<Index1D, FV, IV, FV, IV, int, int>? _floor;
    private Action<Index1D, FV, FV, IV, FV, IV, int, int, int, float, int>? _select;
    private Action<Index1D, FV, IV, IV, FV, int>? _final;

    /// <summary>Most results one search returns.</summary>
    public const int MaxK = TopKKernels.K;

    /// <summary>Floats per row.</summary>
    public int Dim { get; }
    /// <summary>Slots the device buffers hold (grows by doubling).</summary>
    public int Capacity { get; private set; }
    /// <summary>Live rows.</summary>
    public int Count { get; private set; }
    /// <summary>Raised after the index grew, with the new <see cref="Capacity"/>, before the add that grew it returns.</summary>
    public event Action<int>? CapacityChanged;
    /// <summary>The rows on the device ([<see cref="Capacity"/>, <see cref="Dim"/>], a removed slot all zeros), for kernels of the caller's own.</summary>
    public FV Rows => _rows.View;

    public DeviceVectorIndex(Accelerator accelerator, int dim, int capacity = 1024)
    {
        if (dim <= 0) throw new ArgumentOutOfRangeException(nameof(dim));
        _acc = accelerator; Dim = dim; Capacity = Math.Max(capacity, 1);
        _rows = _acc.Allocate1D<float>((long)Capacity * dim); _rows.MemSetToZero();
        _valid = _acc.Allocate1D<int>(Capacity); _valid.MemSetToZero();
        _live = new bool[Capacity];
    }

    /// <summary>True when <paramref name="slot"/> holds a row.</summary>
    public bool IsLive(int slot) => (uint)slot < (uint)_high && _live[slot];

    // ---- adding and removing ----

    /// <summary>Add one row (<see cref="Dim"/> floats); returns its slot (the lowest freed one, else the next new one).</summary>
    public int Add(float[] row)
    {
        if (row.Length != Dim) throw new ArgumentException($"a row is {Dim} floats", nameof(row));
        int slot = TakeSlot();
        _rows.View.SubView((long)slot * Dim, Dim).CopyFromCPU(row);
        SetValid(slot, 1);
        return slot;
    }

    /// <summary>Add one row already on the device (<see cref="Dim"/> floats from <paramref name="row"/>'s start); returns its slot.</summary>
    public int Add(FV row)
    {
        if (row.Length < Dim) throw new ArgumentException($"a row is {Dim} floats", nameof(row));
        int slot = TakeSlot();
        _rows.View.SubView((long)slot * Dim, Dim).CopyFrom(row.SubView(0, Dim));
        SetValid(slot, 1);
        return slot;
    }

    /// <summary><paramref name="count"/> rows ([count, <see cref="Dim"/>] row-major) into consecutive new slots in one upload; returns the slots.</summary>
    public int[] AddRange(ReadOnlySpan<float> rows, int count)
    {
        if (count < 0 || rows.Length < (long)count * Dim) throw new ArgumentException($"{count} rows are {(long)count * Dim} floats", nameof(rows));
        if (count == 0) return Array.Empty<int>();
        int first = TakeRange(count);
        _rows.View.SubView((long)first * Dim, (long)count * Dim).CopyFromCPU(ref MemoryMarshal.GetReference(rows), (long)count * Dim);
        return FinishRange(first, count);
    }

    /// <summary><paramref name="count"/> rows already on the device ([count, <see cref="Dim"/>] row-major) into consecutive new slots; returns the slots.</summary>
    public int[] AddRange(FV rows, int count)
    {
        if (count < 0 || rows.Length < (long)count * Dim) throw new ArgumentException($"{count} rows are {(long)count * Dim} floats", nameof(rows));
        if (count == 0) return Array.Empty<int>();
        int first = TakeRange(count);
        _rows.View.SubView((long)first * Dim, (long)count * Dim).CopyFrom(rows.SubView(0, (long)count * Dim));
        return FinishRange(first, count);
    }

    /// <summary>Zero <paramref name="slot"/>'s row on the device and free the slot for reuse.</summary>
    public void Remove(int slot)
    {
        if (!IsLive(slot)) throw new ArgumentException($"slot {slot} holds no row", nameof(slot));
        _rows.View.SubView((long)slot * Dim, Dim).MemSetToZero();
        SetValid(slot, 0);
        _live[slot] = false; _free.Add(slot); Count--;
    }

    /// <summary>A row read back from the device (diagnostics: a removed slot reads all zeros).</summary>
    public Task<float[]> ReadRowAsync(int slot)
    {
        if ((uint)slot >= (uint)Capacity) throw new ArgumentOutOfRangeException(nameof(slot));
        return _rows.View.SubView((long)slot * Dim, Dim).CopyToHostAsync();
    }

    private int TakeSlot()
    {
        int slot;
        if (_free.Count > 0) { slot = _free.Min; _free.Remove(slot); }
        else { EnsureCapacity(_high + 1); slot = _high++; }
        _live[slot] = true; Count++;
        return slot;
    }

    private int TakeRange(int count)
    {
        EnsureCapacity(_high + count);
        int first = _high;
        _high += count;
        return first;
    }

    private int[] FinishRange(int first, int count)
    {
        var ones = new int[count];
        Array.Fill(ones, 1);
        _valid.View.SubView(first, count).CopyFromCPU(ones);
        var slots = new int[count];
        for (int i = 0; i < count; i++) { slots[i] = first + i; _live[first + i] = true; }
        Count += count;
        return slots;
    }

    private void SetValid(int slot, int v) => _valid.View.SubView(slot, 1).CopyFromCPU(new[] { v });

    private void EnsureCapacity(int needed)
    {
        if (needed <= Capacity) return;
        int cap = Capacity;
        while (cap < needed) cap = checked(cap * 2);
        var rows = _acc.Allocate1D<float>((long)cap * Dim); rows.MemSetToZero();
        var valid = _acc.Allocate1D<int>(cap); valid.MemSetToZero();
        rows.View.SubView(0, (long)Capacity * Dim).CopyFrom(_rows.View);
        valid.View.SubView(0, Capacity).CopyFrom(_valid.View);
        _retired.Add(_rows); _retired.Add(_valid);
        _rows = rows; _valid = valid;
        Array.Resize(ref _live, cap);
        Capacity = cap;
        CapacityChanged?.Invoke(cap);
    }

    // ---- searching ----

    /// <summary>
    /// The best <paramref name="k"/> (at most <see cref="MaxK"/>) live slots for <paramref name="query"/> (<see cref="Dim"/> floats on the device), best first
    /// (ties: the lower slot): score = <paramref name="dotWeight"/> x (row . query) + <paramref name="addend"/>[slot] (absent: 0), slots with
    /// <paramref name="allow"/>[slot] == 0 left out. <paramref name="allow"/> and <paramref name="addend"/> hold at least <see cref="Capacity"/> entries.
    /// Fewer than k are returned when fewer slots qualify.
    /// </summary>
    public async Task<(int Slot, float Score)[]> SearchAsync(FV query, int k, IV? allow = null, FV? addend = null, float dotWeight = 1f)
    {
        if (k < 1 || k > MaxK) throw new ArgumentOutOfRangeException(nameof(k), $"1..{MaxK}");
        if (query.Length < Dim) throw new ArgumentException($"a query is {Dim} floats", nameof(query));
        if (allow is { } a && a.Length < Capacity) throw new ArgumentException($"allow must cover the {Capacity} slots", nameof(allow));
        if (addend is { } d && d.Length < Capacity) throw new ArgumentException($"addend must cover the {Capacity} slots", nameof(addend));
        int n = _high;
        if (n == 0) return Array.Empty<(int, float)>();
        _score ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, IV, IV, FV, FV, int, int, float>(ScoreImpl);
        _noI ??= _acc.Allocate1D<int>(64);
        _noF ??= _acc.Allocate1D<float>(64);
        if (_scores == null || _scores.Length < n) { if (_scores != null) _retired.Add(_scores); _scores = _acc.Allocate1D<float>(Capacity); }
        int flags = (allow.HasValue ? 1 : 0) | (addend.HasValue ? 2 : 0);
        _score(n, _rows.View, query.SubView(0, Dim), _valid.View, allow ?? _noI.View, addend ?? _noF.View, _scores.View, Dim, flags, dotWeight);
        var scores = _scores.View.SubView(0, n);
        if (_acc.AcceleratorType == AcceleratorType.WebGL) return RankOnHost(await scores.CopyToHostAsync().ConfigureAwait(false), k);
        // the ranking: TopKKernels on the one row of scores, chunks small enough that there are at least K of them (or the floor is no floor)
        int chunk = Math.Clamp(n / TopKKernels.K, 1, 256), nch = (n + chunk - 1) / chunk, S = 2 * TopKKernels.K + 1 + nch;
        _chunkMax ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, IV, int, int, int>(TopKKernels.ChunkMaxImpl);
        _floor ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, IV, FV, IV, int, int>(TopKKernels.FloorImpl);
        _select ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, IV, FV, IV, int, int, int, float, int>(TopKKernels.SelectImpl);
        _final ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, IV, IV, FV, int>(TopKKernels.FinalImpl);
        Ensure(ref _maxV, nch); EnsureI(ref _maxI, nch); Ensure(ref _cand, S);
        Ensure(ref _selV, TopKKernels.SelectCap); EnsureI(ref _selI, TopKKernels.SelectCap); EnsureI(ref _selN, 64);
        _chunkMax(nch, scores, _maxV!.View, _maxI!.View, n, nch, chunk);
        _floor(1, _maxV.View, _maxI.View, _cand!.View, _selN!.View, nch, S);
        _select(nch, scores, _cand.View, _selN.View, _selV!.View, _selI!.View, n, nch, S, 0f, chunk);
        _final(1, _selV.View, _selI.View, _selN.View, _cand.View, S);
        // the read is ordered after the kernels (no SynchronizeAsync: a second round trip)
        var c = await _cand.View.SubView(0, 2 * TopKKernels.K + 1).CopyToHostAsync().ConfigureAwait(false);
        if (c[2 * TopKKernels.K] > TopKKernels.SelectCap)
            return RankOnHost(await scores.CopyToHostAsync().ConfigureAwait(false), k);      // more values tied at the floor than the gather holds
        var res = new List<(int, float)>(k);
        for (int j = 0; j < k; j++)
        {
            int id = (int)c[TopKKernels.K + j];
            float v = c[j];
            if (id < 0 || float.IsNegativeInfinity(v)) break;
            res.Add((id, v));
        }
        return res.ToArray();
    }

    /// <summary><see cref="SearchAsync(FV, int, IV?, FV?, float)"/> with the query from the host (uploaded into a buffer the index keeps).</summary>
    public Task<(int Slot, float Score)[]> SearchAsync(float[] query, int k, IV? allow = null, FV? addend = null, float dotWeight = 1f)
    {
        if (query.Length != Dim) throw new ArgumentException($"a query is {Dim} floats", nameof(query));
        _query ??= _acc.Allocate1D<float>(Dim);
        _query.View.CopyFromCPU(query);
        return SearchAsync(_query.View, k, allow, addend, dotWeight);
    }

    // best first, ties to the lower slot, -Infinity (left out) dropped
    private static (int Slot, float Score)[] RankOnHost(float[] s, int k)
    {
        var idx = new List<int>();
        for (int i = 0; i < s.Length; i++) if (!float.IsNegativeInfinity(s[i])) idx.Add(i);
        idx.Sort((a, b) => s[a] != s[b] ? s[b].CompareTo(s[a]) : a.CompareTo(b));
        return idx.Take(k).Select(i => (i, s[i])).ToArray();
    }

    private void Ensure(ref MemoryBuffer1D<float, Stride1D.Dense>? b, long n)
    {
        if (b != null && b.Length >= n) return;
        if (b != null) _retired.Add(b);
        b = _acc.Allocate1D<float>(Math.Max(n, 64));
    }
    private void EnsureI(ref MemoryBuffer1D<int, Stride1D.Dense>? b, long n)
    {
        if (b != null && b.Length >= n) return;
        if (b != null) _retired.Add(b);
        b = _acc.Allocate1D<int>(Math.Max(n, 64));
    }

    // scores[s] = w (row s . query) + addend[s] for a live, allowed slot, else -Infinity (never reaches a top-k floor). flags: 1 allow, 2 addend
    private static void ScoreImpl(Index1D i, FV rows, FV query, IV valid, IV allow, FV addend, FV scores, int dim, int flags, float w)
    {
        int s = i;
        bool ok = valid[s] != 0;
        bool useAllow = (flags & 1) != 0;
        if (useAllow) ok = ok & (allow[s] != 0);
        float v = float.NegativeInfinity;
        if (ok)
        {
            float d = 0f;
            long o = (long)s * dim;
            for (int k = 0; k < dim; k++) d += rows[o + k] * query[k];
            v = w * d;
            bool useAdd = (flags & 2) != 0;
            if (useAdd) v += addend[s];
        }
        scores[s] = v;
    }

    public void Dispose()
    {
        _rows.Dispose(); _valid.Dispose();
        foreach (var b in _retired) b.Dispose();
        _retired.Clear();
        _scores?.Dispose(); _maxV?.Dispose(); _selV?.Dispose(); _cand?.Dispose(); _query?.Dispose(); _noF?.Dispose();
        _maxI?.Dispose(); _selI?.Dispose(); _selN?.Dispose(); _noI?.Dispose();
    }
}
