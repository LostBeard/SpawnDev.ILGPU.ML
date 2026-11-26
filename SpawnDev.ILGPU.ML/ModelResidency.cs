using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML;

/// <summary>Something that knows how many bytes it holds on the device.</summary>
public interface IDeviceSized
{
    /// <summary>Bytes this object holds on the device NOW: its weights and the scratch it keeps between runs. Not a run's working set.</summary>
    long DeviceBytes { get; }
}

/// <summary>Why <see cref="ModelResidency"/> dropped a model.</summary>
public enum ModelEvictionReason
{
    /// <summary>Another model needed the room (least recently used first).</summary>
    MakeRoom,
    /// <summary>The app released its tag (<see cref="ModelResidency.ReleaseAsync"/>) or everything (<see cref="ModelResidency.ReleaseAllAsync"/>).</summary>
    Released,
    /// <summary><see cref="ModelResidency.TrimAsync"/> brought the models back under the budget.</summary>
    Trimmed,
}

/// <summary>One dropped model: its key, the bytes it held, why, and (for <see cref="ModelEvictionReason.MakeRoom"/>) the key it made room for.</summary>
public readonly record struct ModelEviction(string Key, long Bytes, ModelEvictionReason Reason, string? ForKey);

/// <summary>A kept model: its key, the bytes it holds, how many leases it has, and when it was last acquired (<see cref="System.Diagnostics.Stopwatch.GetTimestamp"/>).</summary>
public readonly record struct ResidentModel(string Key, long Bytes, int Leases, long LastUsedTimestamp);

/// <summary>
/// Keeps models on the device while there is room and drops the least recently used idle one when the next would not fit. One per accelerator, owned by the application.
/// <para>
/// A model is used through a lease (<see cref="AcquireAsync"/>): while any lease on it is alive it is never dropped. Dropping happens only inside an awaited call of this class, after the
/// device is drained, so no queued browser work reads freed weights; using the model again loads it again through its <c>load</c> delegate, and nothing in the model classes changes.
/// </para>
/// <para>
/// Design and the measurements behind it: _DevComms/2026-11-25-tuvok-DESIGN-ml-model-residency.md (GpuResidency in SpawnDev.AI.Server, which this generalizes: evicting everything cost 130 s of
/// re-uploads on a card with 10.6 GB free; evicting nothing froze a 12 GB card at 96 %).
/// </para>
/// </summary>
public sealed class ModelResidency : IAsyncDisposable
{
    /// <summary>What the manager needs from the device (a fake in tests).</summary>
    internal interface IDevice
    {
        /// <summary>The device's free bytes, when the backend can say (see <see cref="Accelerator.TryGetFreeMemory"/>).</summary>
        bool TryGetFree(out long free);
        /// <summary>Gives back idle scratch (buffer pools), at least <paramref name="bytes"/> when they hold that much; returns the bytes freed.</summary>
        long ReclaimIdle(long bytes);
        /// <summary>Waits until no queued work can touch memory about to be freed.</summary>
        Task DrainAsync();
        /// <summary>Sets the device's own budget (ILGPU <c>Accelerator.DeviceMemoryBudgetBytes</c>), so its free-memory answer and its idle-scratch reclaim use the same number.</summary>
        void SetBudget(long? bytes);
    }

    private sealed class AcceleratorDevice(Accelerator accelerator) : IDevice
    {
        public bool TryGetFree(out long free) => accelerator.TryGetFreeMemory(out free);
        public long ReclaimIdle(long bytes) => accelerator.ReclaimDeviceMemory(bytes);
        public Task DrainAsync() => accelerator.SynchronizeAsync();
        public void SetBudget(long? bytes) => accelerator.DeviceMemoryBudgetBytes = bytes;
    }

    private sealed class Entry(string key)
    {
        public readonly string Key = key;
        public readonly HashSet<string> Tags = new(StringComparer.Ordinal);
        public readonly TaskCompletionSource Ready = new(TaskCreationOptions.RunContinuationsAsynchronously);
        public IDisposable? Model;          // null while it loads
        public int Leases;
        public long Bytes;                  // measured after the load
        public long Reserved;               // counted while it loads
        public long LastUsed = System.Diagnostics.Stopwatch.GetTimestamp();
        public int Generation;
        public bool DropWhenIdle;           // released while leased: dropped when the last lease ends
    }

    private sealed class Holder(string key, Func<bool> isResident, Func<long> bytes, Func<Task<bool>> evict)
    {
        public readonly string Key = key;
        public readonly Func<bool> IsResident = isResident;
        public readonly Func<long> Bytes = bytes;
        public readonly Func<Task<bool>> Evict = evict;
        public long LastUsed = System.Diagnostics.Stopwatch.GetTimestamp();     // never "never": a holder registered and not yet touched must not look like the oldest
    }

    private readonly IDevice _device;
    private readonly object _sync = new();                  // the tables and counters: never held across an await
    private readonly SemaphoreSlim _room = new(1, 1);       // one make-room / drop at a time
    private readonly Dictionary<string, Entry> _entries = new(StringComparer.Ordinal);
    private readonly Dictionary<string, Holder> _holders = new(StringComparer.Ordinal);
    private readonly Dictionary<string, long> _knownBytes = new(StringComparer.Ordinal);
    private int _generation;
    private bool _disposed, _warnedNoBudget;

    /// <summary>A manager for <paramref name="accelerator"/>, which stays the application's (never disposed here).</summary>
    public ModelResidency(Accelerator accelerator) : this(new AcceleratorDevice(accelerator ?? throw new ArgumentNullException(nameof(accelerator)))) { }

    internal ModelResidency(IDevice device) => _device = device;

    /// <summary>
    /// The most bytes the kept models may hold; null (the default) = no limit of the app's. Where the backend reports free memory, that limits too. With neither, nothing is ever dropped to make
    /// room (today's behaviour) and <see cref="Log"/> says once what to set. In a browser it must come from the app: a page gets no memory figure.
    /// Lowering it takes effect at the next <see cref="AcquireAsync"/> or at <see cref="TrimAsync"/>.
    /// </summary>
    public long? BudgetBytes
    {
        get => _budget;
        set
        {
            if (value < 0) throw new ArgumentOutOfRangeException(nameof(value));
            _budget = value;
            _device.SetBudget(value);
        }
    }
    private long? _budget;

    /// <summary>Bytes kept free beyond a model's own, for a run's working set (default 0).</summary>
    public long ReserveBytes { get; set; }

    /// <summary>Raised after a model was dropped.</summary>
    public event Action<ModelEviction>? Evicted;

    /// <summary>One line per decision worth explaining later (a drop, a model loaded over the budget, an undisposed lease). Null for quiet.</summary>
    public Action<string>? Log { get; set; }

    /// <summary>The models on the device now, least recently used first.</summary>
    public IReadOnlyList<ResidentModel> Resident
    {
        get
        {
            lock (_sync)
                return _entries.Values.Where(e => e.Model != null).OrderBy(e => e.LastUsed).Select(e => new ResidentModel(e.Key, e.Bytes, e.Leases, e.LastUsed)).ToList();
        }
    }

    /// <summary>
    /// The model for <paramref name="key"/>: the kept one, or a new one from <paramref name="load"/> after making room for it. Dispose the lease (<c>await using</c>) when the run that uses the model ended.
    /// </summary>
    /// <param name="key">What identifies the model (a file, a pipeline): the same key is the same model.</param>
    /// <param name="load">Loads it. One load per key at a time; a failure throws in every waiting caller and leaves nothing behind.</param>
    /// <param name="expectedBytes">What it will hold on the device (0 = the size measured the last time, if any). With a budget and no size, every idle model is dropped before the load: on a small
    /// card the over-allocation is the load itself.</param>
    /// <param name="tag">Whose it is (a tab, a workflow): <see cref="ReleaseAsync"/> drops it.</param>
    public async Task<ModelLease<T>> AcquireAsync<T>(string key, Func<Task<T>> load, long expectedBytes = 0, string? tag = null) where T : class, IDisposable
    {
        ArgumentException.ThrowIfNullOrEmpty(key);
        ArgumentNullException.ThrowIfNull(load);
        ArgumentOutOfRangeException.ThrowIfNegative(expectedBytes);
        Entry entry;
        bool owner = false;
        await _room.WaitAsync().ConfigureAwait(false);
        try
        {
            lock (_sync)
            {
                ObjectDisposedException.ThrowIf(_disposed, this);
                if (_entries.TryGetValue(key, out var existing))
                {
                    entry = existing;
                }
                else
                {
                    long want = expectedBytes > 0 ? expectedBytes : _knownBytes.GetValueOrDefault(key);
                    entry = new Entry(key) { Reserved = want, Generation = _generation };
                    _entries[key] = entry;
                    owner = true;
                }
                entry.Leases++;
                entry.LastUsed = System.Diagnostics.Stopwatch.GetTimestamp();
                entry.DropWhenIdle = false;
                if (tag != null) entry.Tags.Add(tag);
            }
            if (owner)
            {
                try { await MakeRoomAsync(entry.Reserved, key).ConfigureAwait(false); }
                catch (Exception ex) { Fail(entry, ex); throw; }
            }
        }
        finally { _room.Release(); }

        if (!owner)
        {
            await entry.Ready.Task.ConfigureAwait(false);       // a failed load throws here too
        }
        else
        {
            T model;
            try { model = await load().ConfigureAwait(false) ?? throw new InvalidOperationException($"The load of '{key}' returned no model."); }
            catch (Exception ex) { Fail(entry, ex); throw; }
            bool lost, closed;
            lock (_sync)
            {
                lost = entry.Generation != _generation;         // the device went away while it loaded
                closed = _disposed;
                if (!lost && !closed)
                {
                    entry.Model = model;
                    entry.Bytes = model is IDeviceSized sized ? sized.DeviceBytes : entry.Reserved;
                    entry.Reserved = 0;
                    if (entry.Bytes > 0) _knownBytes[key] = entry.Bytes;
                }
            }
            if (lost || closed)
            {
                var ex = lost ? new InvalidOperationException($"The device was lost while '{key}' loaded.") : (Exception)new ObjectDisposedException(nameof(ModelResidency));
                if (closed && !lost) model.Dispose();           // a lost device's memory is gone: freeing into it is what must not happen
                Fail(entry, ex);
                throw ex;
            }
            entry.Ready.TrySetResult();
        }
        if (entry.Model is not T typed)
        {
            string actual = entry.Model?.GetType().Name ?? "nothing";
            Release(entry);
            throw new InvalidOperationException($"'{key}' holds a {actual}, not a {typeof(T).Name}: one key is one model.");
        }
        return new ModelLease<T>(this, entry, typed, key);
    }

    /// <summary>
    /// Joins an engine that loads and unloads its own model: counted in the budget while <paramref name="isResident"/>, and asked to unload (<paramref name="evict"/>, false = refused, e.g. it is in use)
    /// when room is needed, in least-recently-used order with the leased models. Call <see cref="EnsureRoomForAsync"/> before it loads and <see cref="Touch"/> when it is used.
    /// </summary>
    public void Register(string key, Func<bool> isResident, Func<long> bytes, Func<Task<bool>> evict)
    {
        ArgumentException.ThrowIfNullOrEmpty(key);
        ArgumentNullException.ThrowIfNull(isResident); ArgumentNullException.ThrowIfNull(bytes); ArgumentNullException.ThrowIfNull(evict);
        lock (_sync) _holders[key] = new Holder(key, isResident, bytes, evict);
    }

    /// <summary>Removes a holder joined with <see cref="Register"/>.</summary>
    public void Unregister(string key) { lock (_sync) _holders.Remove(key); }

    /// <summary>Notes that a registered holder was just used.</summary>
    public void Touch(string key) { lock (_sync) if (_holders.TryGetValue(key, out var h)) h.LastUsed = System.Diagnostics.Stopwatch.GetTimestamp(); }

    /// <summary>Makes room for a registered holder that is about to load (a no-op when it is resident already).</summary>
    public async Task EnsureRoomForAsync(string key)
    {
        Holder? h;
        lock (_sync) { _holders.TryGetValue(key, out h); if (h != null) h.LastUsed = System.Diagnostics.Stopwatch.GetTimestamp(); }
        if (h == null) throw new ArgumentException($"No holder '{key}' is registered.", nameof(key));
        await _room.WaitAsync().ConfigureAwait(false);
        try { if (!h.IsResident()) await MakeRoomAsync(h.Bytes(), key).ConfigureAwait(false); }
        finally { _room.Release(); }
    }

    /// <summary>
    /// Drops the idle models of <paramref name="tag"/> that no other tag holds (a closed tab). One in use is dropped when its last lease ends. Returns how many were dropped now.
    /// </summary>
    public async Task<int> ReleaseAsync(string tag)
    {
        ArgumentNullException.ThrowIfNull(tag);
        return await DropAsync(e => e.Tags.Remove(tag) && e.Tags.Count == 0, ModelEvictionReason.Released).ConfigureAwait(false);
    }

    /// <summary>Drops every idle model; one in use when its last lease ends. Returns how many were dropped now.</summary>
    public Task<int> ReleaseAllAsync() => DropAsync(_ => true, ModelEvictionReason.Released);

    /// <summary>Drops idle models, least recently used first, until <see cref="ReserveBytes"/> is free within <see cref="BudgetBytes"/> (after it was lowered), as the accelerator counts it. Returns how many were dropped.</summary>
    public async Task<int> TrimAsync()
    {
        await _room.WaitAsync().ConfigureAwait(false);
        try { return await EvictUntilAsync(() => BudgetBytes != null && TryFree(0, out long free) && free < ReserveBytes, ModelEvictionReason.Trimmed, null).ConfigureAwait(false); }
        finally { _room.Release(); }
    }

    /// <summary>
    /// The device is gone (a lost WebGPU device): forget every model WITHOUT disposing it. A load that is still running throws in its callers when it ends. The manager can be used again with models
    /// loaded on the new device only if that is the same accelerator; a new accelerator needs a new manager.
    /// </summary>
    public void Invalidate()
    {
        lock (_sync)
        {
            _generation++;
            _entries.Clear();
        }
    }

    /// <summary>Drops every idle model and refuses new ones. A model still leased is dropped by its last lease.</summary>
    public async ValueTask DisposeAsync()
    {
        lock (_sync)
        {
            if (_disposed) return;
            _disposed = true;
        }
        await ReleaseAllAsync().ConfigureAwait(false);
    }

    // ------------------------------------------------------------------------------------------------------------------------ leases

    private void Release(Entry entry)
    {
        lock (_sync) entry.Leases--;
    }

    internal async Task EndLeaseAsync(object entryObject)
    {
        var entry = (Entry)entryObject;
        bool drop;
        lock (_sync)
        {
            entry.Leases--;
            entry.LastUsed = System.Diagnostics.Stopwatch.GetTimestamp();
            drop = entry.Leases == 0 && entry.DropWhenIdle && _entries.TryGetValue(entry.Key, out var e) && ReferenceEquals(e, entry);
        }
        if (drop) await DropAsync(e => ReferenceEquals(e, entry) && e.DropWhenIdle, ModelEvictionReason.Released).ConfigureAwait(false);
    }

    // a sync Dispose cannot drain the device: the model stays, and the next drop takes it like any idle model
    internal void EndLease(object entryObject)
    {
        var entry = (Entry)entryObject;
        lock (_sync) { entry.Leases--; entry.LastUsed = System.Diagnostics.Stopwatch.GetTimestamp(); }
    }

    internal void ReportLeakedLease(string key) => Log?.Invoke($"[ModelResidency] a lease of '{key}' was never disposed: the model stays on the device for good.");

    private void Fail(Entry entry, Exception ex)
    {
        lock (_sync)
            if (_entries.TryGetValue(entry.Key, out var e) && ReferenceEquals(e, entry)) _entries.Remove(entry.Key);
        entry.Ready.TrySetException(ex);
        _ = entry.Ready.Task.Exception;         // observed: the owner throws it itself
    }

    // ------------------------------------------------------------------------------------------------------------------------ room

    private long ModelBytes()
    {
        Holder[] holders;
        long sum;
        lock (_sync)
        {
            sum = _entries.Values.Sum(e => e.Model != null ? e.Bytes : e.Reserved);
            holders = _holders.Values.ToArray();
        }
        foreach (var h in holders) if (h.IsResident()) sum += h.Bytes();     // app code: never under the lock
        return sum;
    }

    // free for a model of `forKey` (whose own reservation is in the table already, and must not count against itself); false = nobody can say
    private bool TryFree(long ownReserved, out long free)
    {
        bool known = _device.TryGetFree(out long measured);
        long loading;
        lock (_sync) loading = _entries.Values.Where(e => e.Model == null).Sum(e => e.Reserved) - ownReserved;
        free = known ? measured - loading : long.MaxValue;                    // what other loads are about to take
        if (BudgetBytes is long budget) { free = Math.Min(free, budget - (ModelBytes() - ownReserved)); known = true; }
        return known;
    }

    private async Task MakeRoomAsync(long want, string forKey)
    {
        long own;
        lock (_sync) own = _entries.TryGetValue(forKey, out var e) && e.Model == null ? e.Reserved : 0;
        if (!TryFree(own, out long free))
        {
            if (!_warnedNoBudget)
            {
                _warnedNoBudget = true;
                Log?.Invoke("[ModelResidency] this backend reports no free memory and no BudgetBytes is set: no model is ever dropped to make room. Set BudgetBytes.");
            }
            return;
        }
        if (want <= 0 && BudgetBytes != null)
        {
            // the size is unknown and the load itself is what over-allocates: everything idle goes first
            Log?.Invoke($"[ModelResidency] '{forKey}' has no known size: dropping every idle model before it loads.");
            await EvictUntilAsync(() => true, ModelEvictionReason.MakeRoom, forKey).ConfigureAwait(false);
            return;
        }
        long Short() => TryFree(own, out long f) ? want + ReserveBytes - f : 0;
        long shortfall = want + ReserveBytes - free;
        if (shortfall <= 0) return;
        _device.ReclaimIdle(shortfall);                                       // idle scratch before any model (7.3 GB idle in pools on a 12 GB card, Vertex 2026-10-12)
        if (Short() <= 0) return;
        await EvictUntilAsync(() => Short() > 0, ModelEvictionReason.MakeRoom, forKey).ConfigureAwait(false);
        long left = Short();
        if (left > 0)
            // a wrong budget must not make a workflow impossible (TJ, 2026-11-25): say so and load
            Log?.Invoke($"[ModelResidency] '{forKey}' ({want >> 20} MB) is {left >> 20} MB over what is free after dropping every idle model. Loading it anyway: if the device is lost or slow, this is why.");
    }

    private async Task<int> EvictUntilAsync(Func<bool> more, ModelEvictionReason reason, string? forKey)
    {
        int dropped = 0;
        var refused = new HashSet<string>(StringComparer.Ordinal);
        while (more())
        {
            Entry? victim = null; Holder? holder = null;
            Holder[] holders;
            lock (_sync) holders = _holders.Values.Where(h => h.Key != forKey && !refused.Contains(h.Key)).ToArray();
            holders = holders.Where(h => h.IsResident()).ToArray();
            lock (_sync)
            {
                victim = _entries.Values.Where(e => e.Model != null && e.Leases == 0).OrderBy(e => e.LastUsed).FirstOrDefault();
                holder = holders.OrderBy(h => h.LastUsed).FirstOrDefault();
                if (victim != null && (holder == null || victim.LastUsed <= holder.LastUsed)) { holder = null; _entries.Remove(victim.Key); }   // out of the table: nobody can lease it any more
                else victim = null;
            }
            if (victim != null)
            {
                if (dropped == 0) await _device.DrainAsync().ConfigureAwait(false);
                DisposeModel(victim, reason, forKey);
                dropped++;
            }
            else if (holder != null)
            {
                long bytes = holder.Bytes();
                if (dropped == 0) await _device.DrainAsync().ConfigureAwait(false);
                if (await holder.Evict().ConfigureAwait(false)) { dropped++; Report(new ModelEviction(holder.Key, bytes, reason, forKey)); }
                else { refused.Add(holder.Key); Log?.Invoke($"[ModelResidency] '{holder.Key}' refused to unload (in use)."); }
            }
            else break;
        }
        return dropped;
    }

    private async Task<int> DropAsync(Func<Entry, bool> which, ModelEvictionReason reason)
    {
        await _room.WaitAsync().ConfigureAwait(false);
        try
        {
            List<Entry> idle;
            lock (_sync)
            {
                var chosen = _entries.Values.Where(which).ToList();
                foreach (var e in chosen.Where(e => e.Leases > 0 || e.Model == null)) e.DropWhenIdle = true;       // in use or loading: when its last lease ends
                idle = chosen.Where(e => e.Leases == 0 && e.Model != null).ToList();
                foreach (var e in idle) _entries.Remove(e.Key);
            }
            if (idle.Count == 0) return 0;
            await _device.DrainAsync().ConfigureAwait(false);
            foreach (var e in idle) DisposeModel(e, reason, null);
            return idle.Count;
        }
        finally { _room.Release(); }
    }

    private void DisposeModel(Entry entry, ModelEvictionReason reason, string? forKey)
    {
        try { entry.Model!.Dispose(); }
        catch (Exception ex) { Log?.Invoke($"[ModelResidency] disposing '{entry.Key}' threw {ex.GetType().Name}: {ex.Message}"); }
        entry.Model = null;
        Report(new ModelEviction(entry.Key, entry.Bytes, reason, forKey));
    }

    private void Report(ModelEviction eviction)
    {
        Log?.Invoke($"[ModelResidency] dropped '{eviction.Key}' ({eviction.Bytes >> 20} MB, {eviction.Reason}{(eviction.ForKey != null ? $" for '{eviction.ForKey}'" : "")}).");
        Evicted?.Invoke(eviction);
    }
}

/// <summary>A use of a kept model. The model is never dropped while a lease on it is alive; end it with <c>await using</c> when the run that uses the model ended.</summary>
public sealed class ModelLease<T> : IAsyncDisposable, IDisposable where T : class
{
    private readonly ModelResidency _owner;
    private readonly object _entry;
    private int _ended;

    internal ModelLease(ModelResidency owner, object entry, T model, string key) { _owner = owner; _entry = entry; Model = model; Key = key; }

    /// <summary>The model.</summary>
    public T Model { get; }
    /// <summary>Its key.</summary>
    public string Key { get; }

    /// <summary>Ends the lease. A model released while it was leased (<see cref="ModelResidency.ReleaseAsync"/>) is dropped here when this was the last lease.</summary>
    public async ValueTask DisposeAsync()
    {
        if (Interlocked.Exchange(ref _ended, 1) != 0) return;
        GC.SuppressFinalize(this);
        await _owner.EndLeaseAsync(_entry).ConfigureAwait(false);
    }

    /// <summary>Ends the lease without waiting for the device: a model released meanwhile stays until the next drop. Prefer <see cref="DisposeAsync"/>.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _ended, 1) != 0) return;
        GC.SuppressFinalize(this);
        _owner.EndLease(_entry);
    }

    /// <summary>A lease collected without being ended is reported, not silently ended: the model may be in use by code that dropped its lease.</summary>
    ~ModelLease() { if (Volatile.Read(ref _ended) == 0) _owner.ReportLeakedLease(Key); }
}
