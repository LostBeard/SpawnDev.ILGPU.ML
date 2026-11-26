using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary><see cref="ModelResidency"/> on a fake device: the decisions (what is dropped, when, in what order), no GPU.</summary>
public abstract partial class MLTestBase
{
    private sealed class FakeModel(string name, long bytes, List<string> log) : IDisposable, IDeviceSized
    {
        public long DeviceBytes => bytes;
        public bool Disposed;
        public void Dispose() { Disposed = true; log.Add("dispose " + name); }
    }

    private sealed class FakeDevice(List<string> log) : ModelResidency.IDevice
    {
        public long? Free;                  // null = the backend cannot say
        public long Idle;                   // idle scratch ReclaimIdle can give back
        public bool TryGetFree(out long free) { free = Free ?? 0; return Free != null; }
        public long ReclaimIdle(long bytes) { log.Add("reclaim"); long got = Math.Min(bytes, Idle); Idle -= got; if (Free != null) Free += got; return got; }
        public async Task DrainAsync() { log.Add("drain"); await Task.Yield(); }
        public long? Budget;
        public void SetBudget(long? bytes) => Budget = bytes;
        public long Allocated;               // what the device has allocated (a model that does not report its size is measured from it)
        public long AllocatedBytes => Allocated;
    }

    private static void Mr(bool ok, string what) { if (!ok) throw new Exception(what); }
    private static string MrKeys(ModelResidency r) => string.Join(",", r.Resident.Select(m => m.Key));

    /// <summary>
    /// <see cref="InferenceSession.DeviceBytes"/> against the accelerator's own count of what it allocated (ILGPU <c>AllocatedDeviceBytes</c>, 5.4.50): never more than the session added,
    /// exactly equal right after loading; after a run, never more (kernel scratch is not in it: SqueezeNet 73.5 % on CUDA, OpenCL and WebGPU, 2026-11-25). After Dispose the accelerator gets its bytes back.
    /// </summary>
    [TestMethod(Timeout = 120000)]
    public async Task InferenceSession_DeviceBytes_MatchesTheAcceleratorsCount() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient() ?? throw new UnsupportedTestException("HttpClient not available for this backend");
        var (pixels, width, height) = await LoadCatImage(http);
        await accelerator.SynchronizeAsync();
        long before = accelerator.AllocatedDeviceBytes;
        long sessionBytes, added;
        {
            using var session = await InferenceSession.CreateFromFileAsync(accelerator, http, "models/squeezenet/model.onnx");
            long afterCreate = accelerator.AllocatedDeviceBytes - before, createCounted = session.DeviceBytes;
            var pipeline = new SpawnDev.ILGPU.ML.Pipelines.ClassificationPipeline(session, accelerator);
            await pipeline.ClassifyAsync(pixels, width, height, 5);
            await accelerator.SynchronizeAsync();
            added = accelerator.AllocatedDeviceBytes - before;
            sessionBytes = session.DeviceBytes;
            Console.WriteLine($"[DeviceBytes] {accelerator.AcceleratorType}: after create {createCounted} of {afterCreate} counted; after a run {sessionBytes} of {added} ({(added > 0 ? 100.0 * sessionBytes / added : 0):F1} %)");
            Mr(createCounted == afterCreate, $"right after loading, the session's own size ({createCounted}) must be exactly what the accelerator allocated ({afterCreate})");
            Mr(sessionBytes > 0, "the session reports no device bytes");
            Mr(sessionBytes <= added, $"the session claims {sessionBytes} bytes but the accelerator allocated only {added} since it was created");
        }
        await accelerator.SynchronizeAsync();
        long left = accelerator.AllocatedDeviceBytes - before;
        Console.WriteLine($"[DeviceBytes] {accelerator.AcceleratorType}: after Dispose {left} bytes still allocated of the {added}");
        // what stays is the accelerator's shared parameter caches (60-72 bytes measured on every backend), not the session's
        Mr(left <= 4096, $"Dispose gave back all but {left} of the session's {added} bytes");
    });

    /// <summary>
    /// The manager on a real accelerator with ILGPU's budget (5.4.50): two copies of SqueezeNet under a budget that holds one (plus what a run needs). Loading the second must drop the first, the
    /// accelerator's count must fall back to one model's worth, and both copies must classify the cat. A budget that the manager sets is the accelerator's own (one number for both).
    /// </summary>
    [TestMethod(Timeout = 180000)]
    public async Task ModelResidency_RealDevice_BudgetDropsTheIdleModel() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient() ?? throw new UnsupportedTestException("HttpClient not available for this backend");
        var (pixels, width, height) = await LoadCatImage(http);
        var onnx = await http.GetByteArrayAsync("models/squeezenet/model.onnx");
        var catClasses = new HashSet<int> { 281, 282, 283, 284, 285 };
        async Task Classify(InferenceSession s, string which)
        {
            var results = await new SpawnDev.ILGPU.ML.Pipelines.ClassificationPipeline(s, accelerator).ClassifyAsync(pixels, width, height, 5);
            Mr(results.Any(r => catClasses.Contains(r.ClassIndex)), $"{which}: no cat in the top 5 ({string.Join(", ", results.Select(r => r.ClassIndex))})");
        }
        await accelerator.SynchronizeAsync();
        long before = accelerator.AllocatedDeviceBytes;
        long oneModel, weights;
        {   // measure one copy: its weights (exact right after loading), then with what a run keeps
            using var probe = await InferenceSession.CreateFromStreamAsync(accelerator, new MemoryStream(onnx, writable: false));
            weights = probe.DeviceBytes;
            await Classify(probe, "probe");
            await accelerator.SynchronizeAsync();
            oneModel = accelerator.AllocatedDeviceBytes - before;
        }
        await accelerator.SynchronizeAsync();
        var log = new List<string>();
        var evicted = new List<string>();
        await using var r = new ModelResidency(accelerator) { Log = log.Add };
        r.Evicted += e => evicted.Add(e.Key);
        try
        {
            // A run's scratch is idle between runs and the budget reclaims it first, so "two do not fit" has to hold for the weights: one copy + a run's scratch (kept free as the reserve)
            // + half a copy. MEASURED without this: a budget of 1.5 x (weights + scratch) held both copies once their idle scratch was reclaimed, and nothing was dropped.
            long scratch = oneModel - weights;
            r.ReserveBytes = scratch;
            r.BudgetBytes = before + weights + scratch + weights / 2;
            Mr(accelerator.DeviceMemoryBudgetBytes == r.BudgetBytes, "the manager's budget is the accelerator's");
            await using (var a = await r.AcquireAsync("squeezenet-a", () => InferenceSession.CreateFromStreamAsync(accelerator, new MemoryStream(onnx, writable: false)), onnx.Length))
                await Classify(a.Model, "copy a");
            await using (var b = await r.AcquireAsync("squeezenet-b", () => InferenceSession.CreateFromStreamAsync(accelerator, new MemoryStream(onnx, writable: false)), onnx.Length))
            {
                await Classify(b.Model, "copy b");
                await accelerator.SynchronizeAsync();
                long held = accelerator.AllocatedDeviceBytes - before;
                Console.WriteLine($"[DeviceBytes] {accelerator.AcceleratorType}: weights {weights}, with a run {oneModel}, budget {r.BudgetBytes - before} over the start, held with copy b {held}; dropped [{string.Join(",", evicted)}]; log: {string.Join(" | ", log)}");
                Mr(string.Join(",", evicted) == "squeezenet-a", $"loading copy b drops copy a, and only it: [{string.Join(",", evicted)}]");
                Mr(held <= r.BudgetBytes - before, $"after the drop the device holds {held} bytes for the models, more than the budget allows ({r.BudgetBytes - before})");
            }
        }
        finally { await r.ReleaseAllAsync(); accelerator.DeviceMemoryBudgetBytes = null; }
    });

    /// <summary>The kept model is reused; one over the budget drops the least recently used idle model first, after a drain; leased models are never dropped.</summary>
    [TestMethod(Timeout = 30000)]
    public async Task ModelResidency_LruWithinTheBudget_LeasesPin() => await RunPureTest(async () =>
    {
        var log = new List<string>();
        await using var r = new ModelResidency(new FakeDevice(log)) { BudgetBytes = 100 };
        var evicted = new List<ModelEviction>();
        r.Evicted += evicted.Add;
        int loads = 0;
        Func<Task<FakeModel>> Load(string n, long b) => () => { loads++; return Task.FromResult(new FakeModel(n, b, log)); };
        async Task Use(string n, long b) { await using var l = await r.AcquireAsync(n, Load(n, b), b); await Task.Yield(); }

        await Use("a", 40); await Use("b", 40); await Use("a", 40);
        Mr(loads == 2, $"a and b loaded once each ({loads}): the second use of a is the kept one");
        Mr(MrKeys(r) == "b,a", $"least recently used first: {MrKeys(r)}");
        await Use("c", 40);
        Mr(MrKeys(r) == "a,c", $"b (least recently used) made room for c: {MrKeys(r)}");
        Mr(evicted.Count == 1 && evicted[0] is { Key: "b", Bytes: 40, Reason: ModelEvictionReason.MakeRoom, ForKey: "c" }, "the eviction event says what, how big, why and for whom");
        Mr(string.Join(",", log) == "reclaim,drain,dispose b", $"idle scratch first, then drained, then disposed: {string.Join(",", log)}");

        // a and c leased: loading d drops nothing and still loads (over the budget, with a line)
        var lines = new List<string>(); r.Log = lines.Add;
        await using (var la = await r.AcquireAsync("a", Load("a", 40), 40))
        await using (var lc = await r.AcquireAsync("c", Load("c", 40), 40))
        {
            await Use("d", 60);
            Mr(MrKeys(r) == "a,c,d", $"leased models stay, the over-budget one loads: {MrKeys(r)}");
            Mr(lines.Any(l => l.Contains("over what is free")), "loading over the budget says so");
            Mr(!la.Model.Disposed && !lc.Model.Disposed, "a leased model was disposed");
        }
        Mr(await r.TrimAsync() >= 1 && r.Resident.Sum(m => m.Bytes) <= 100, $"TrimAsync brings the idle models within the budget: {MrKeys(r)}");
        r.BudgetBytes = 1000;
        int kept = r.Resident.Count;
        int dropped = await r.ReleaseAllAsync();
        Mr(kept > 0 && dropped == kept && r.Resident.Count == 0, $"ReleaseAll dropped {dropped} of {kept} idle models, left: {MrKeys(r)}");
    });

    /// <summary>A budget with an unknown size drops every idle model before the load; a known (remembered) size drops only what is needed; without a budget or a free-memory number nothing is dropped.</summary>
    [TestMethod(Timeout = 30000)]
    public async Task ModelResidency_UnknownSizes_RememberedSizes_NoBudget() => await RunPureTest(async () =>
    {
        var log = new List<string>();
        await using var r = new ModelResidency(new FakeDevice(log)) { BudgetBytes = 1000 };
        Task<FakeModel> New(string n, long b) => Task.FromResult(new FakeModel(n, b, log));
        async Task Use(string n, long b, long expected) { await using var l = await r.AcquireAsync(n, () => New(n, b), expected); }
        await Use("a", 10, 10); await Use("b", 10, 10);
        await Use("x", 10, 0);                        // size unknown: assumed as large as the largest measured (10), and 1000 holds it: nothing goes
        Mr(MrKeys(r) == "a,b,x", $"an unknown size is assumed to be the largest measured one, which fits: {MrKeys(r)}");
        await Use("x", 10, 0);
        Mr(MrKeys(r) == "a,b,x", $"x is the kept one: nothing dropped: {MrKeys(r)}");
        r.BudgetBytes = 40;
        await Use("c", 10, 10);
        Mr(MrKeys(r) == "a,b,x,c", $"40 is enough for four 10s: {MrKeys(r)}");
        await Use("d", 10, 0);                        // never seen: assumed 10, so one goes, the least recently used
        Mr(MrKeys(r) == "b,x,c,d", $"d: room for one more 10, a dropped: {MrKeys(r)}");
        r.UnknownSizeBytes = 25;
        await Use("e", 10, 0);                        // the app's guess: 25 of 40 -> three of the four 10s go (30 free with only d left)
        Mr(MrKeys(r) == "d,e", $"UnknownSizeBytes = 25 makes room for 25: {MrKeys(r)}");

        // nothing measured (models that do not report a size, no expectedBytes): every idle model goes before an unknown load
        await using var blind = new ModelResidency(new FakeDevice(log)) { BudgetBytes = 1L << 40 };
        async Task UseBlind(string n) { await using var l = await blind.AcquireAsync(n, () => Task.FromResult(new MemoryStream())); }
        await UseBlind("p"); await UseBlind("q");
        Mr(string.Join(",", blind.Resident.Select(m => m.Key)) == "q", "with nothing measured, an unknown load drops every idle model first");
        blind.UnknownSizeBytes = 1L << 30;
        await UseBlind("s");
        Mr(string.Join(",", blind.Resident.Select(m => m.Key)) == "q,s", "with UnknownSizeBytes set, a 1 TB budget keeps them");

        await using var none = new ModelResidency(new FakeDevice(log));
        var lines = new List<string>(); none.Log = lines.Add;
        for (int i = 0; i < 20; i++) await using (var l = await none.AcquireAsync($"m{i}", () => New($"m{i}", 1L << 30), 1L << 30)) { }
        Mr(none.Resident.Count == 20, $"no budget, no free memory: nothing is dropped ({none.Resident.Count})");
        Mr(lines.Count(l => l.Contains("BudgetBytes")) == 1, "it says once what to set");
    });

    /// <summary>IModelSource.GetSizeAsync: a local file's length without reading it, null for a missing one, and null from a source that does not implement it (the default member).</summary>
    [TestMethod(Timeout = 30000)]
    public async Task ModelResidency_ModelSource_GetSizeAsync() => await RunPureTest(async () =>
    {
        if (OperatingSystem.IsBrowser()) throw new UnsupportedTestException("a local folder");
        var dir = Directory.CreateTempSubdirectory("ml-size-").FullName;
        try
        {
            await File.WriteAllBytesAsync(Path.Combine(dir, "w.bin"), new byte[12345]);
            Hub.IModelSource local = new Hub.LocalFolderModelSource(dir);
            Mr(await local.GetSizeAsync("", "w.bin") == 12345, "the file's length");
            Mr(await local.GetSizeAsync("", "missing.bin") == null, "a missing file: null");
            Hub.IModelSource plain = new Fp8MixedSource(new byte[1]);
            Mr(await plain.GetSizeAsync("", "x") == null, "a source without GetSizeAsync answers null");
        }
        finally { Directory.Delete(dir, true); }
    });

    /// <summary>
    /// A model that does not report its size (Lore's SpawnGen objects, any app's): measured from the device's own allocation count, over its load (eager) or from its lease's
    /// EnsureRoomAsync to the lease's end (lazy); a load that overlaps another load is not measured. The make-room log line carries the arithmetic.
    /// </summary>
    [TestMethod(Timeout = 30000)]
    public async Task ModelResidency_NotSized_MeasuredByTheDevice() => await RunPureTest(async () =>
    {
        var log = new List<string>();
        var dev = new FakeDevice(log);
        var lines = new List<string>();
        await using var r = new ModelResidency(dev) { BudgetBytes = 100, Log = lines.Add };
        await using (var l = await r.AcquireAsync("eager", () => { dev.Allocated += 30; return Task.FromResult(new MemoryStream()); })) { }   // no size given: only the device can tell
        Mr(r.Resident.Single().Bytes == 30 && r.KnownSizes["eager"] == 30, $"eager: the device's gain over the load ({r.Resident.Single().Bytes})");
        var lazyStream = new MemoryStream();
        await using (var l = await r.AcquireAsync("lazy", () => Task.FromResult(lazyStream), 1))
        {
            await l.EnsureRoomAsync(40);
            dev.Allocated += 40;                          // its weights load inside the lease
        }
        Mr(r.Resident.Single(m => m.Key == "lazy").Bytes == 40 && r.KnownSizes["lazy"] == 40, $"lazy: from EnsureRoomAsync to the lease's end ({r.Resident.Single(m => m.Key == "lazy").Bytes})");
        // two loads at once: neither can be told apart from the other, so neither is measured from the device (each keeps its expected size)
        var gate = new TaskCompletionSource();
        async Task<MemoryStream> Slow(long b) { dev.Allocated += b; await gate.Task; return new MemoryStream(); }
        var a = r.AcquireAsync("p", () => Slow(5), 7);
        var b = r.AcquireAsync("q", () => Slow(6), 8);
        await Task.Yield(); gate.SetResult();
        await using (var la = await a) await using (var lb = await b) { }
        Mr(r.Resident.Single(m => m.Key == "p").Bytes == 7 && r.Resident.Single(m => m.Key == "q").Bytes == 8, "overlapping loads keep their expected sizes");
        dev.Allocated = 95;                               // 95 of 100 allocated: 30 + 40 + 7 + 8 = 85 of it models, 10 other
        await using (var l = await r.AcquireAsync("big", () => Task.FromResult(new MemoryStream()), 20)) { }
        var room = lines.FirstOrDefault(s => s.Contains("room for 'big'")) ?? "";
        Mr(room.Contains("MB other") && room.Contains("dropping idle models"), "the make-room line says what is allocated and why it drops: " + string.Join(" | ", lines));
        await Task.CompletedTask;
    });

    private sealed class LazyModel : IDisposable, IDeviceSized
    {
        public long Bytes;                  // 0 until its weights load inside a lease
        public bool Disposed;
        public long DeviceBytes => Bytes;
        public void Dispose() => Disposed = true;
    }

    /// <summary>
    /// A lazy model (Lore's SpawnGen wrappers: nothing on the device at acquire, the weights load inside the lease): the lease makes room before that load (EnsureRoomAsync), the size is read
    /// again when the lease ends and remembered per key (the largest), and remembered sizes carry to another manager (KnownSizes / SeedKnownSizes).
    /// </summary>
    [TestMethod(Timeout = 30000)]
    public async Task ModelResidency_LazyModels_EnsureRoom_KnownSizes() => await RunPureTest(async () =>
    {
        var log = new List<string>();
        await using var r = new ModelResidency(new FakeDevice(log)) { BudgetBytes = 100 };
        Task<FakeModel> New(string n, long b) => Task.FromResult(new FakeModel(n, b, log));
        await using (var l = await r.AcquireAsync("a", () => New("a", 40), 40)) { }
        await using (var l = await r.AcquireAsync("b", () => New("b", 40), 40)) { }
        var lazy = new LazyModel();
        await using (var l = await r.AcquireAsync("lazy", () => Task.FromResult(lazy), 1))
        {
            Mr(MrKeys(r) == "a,b,lazy", $"acquiring the empty wrapper drops nothing: {MrKeys(r)}");
            await l.EnsureRoomAsync(50);                  // 100 - 81 = 19 free, 50 wanted: the least recently used idle model goes, not the leased one
            Mr(MrKeys(r) == "b,lazy" && log.Contains("dispose a"), $"room for the lazy load: a dropped: {MrKeys(r)} / {string.Join(",", log)}");
            lazy.Bytes = 50;                              // its weights load inside the lease
        }
        Mr(r.Resident.Single(m => m.Key == "lazy").Bytes == 50 && r.KnownSizes["lazy"] == 50, $"the size is read again when the lease ends: {r.Resident.Single(m => m.Key == "lazy").Bytes}");
        lazy.Bytes = 30;                                  // it shrank (scratch freed): the entry follows, the remembered size keeps the largest
        await using (var l = await r.AcquireAsync<LazyModel>("lazy", () => throw new Exception("kept: no load"), 0)) { }
        Mr(r.Resident.Single(m => m.Key == "lazy").Bytes == 30 && r.KnownSizes["lazy"] == 50, "current size 30, remembered 50");

        // a new session: the remembered sizes go in, so a first load is no longer an unknown size
        await using var next = new ModelResidency(new FakeDevice(log)) { BudgetBytes = 60, Log = log.Add };
        await using (var l = await next.AcquireAsync("c", () => New("c", 40), 40)) { }
        next.SeedKnownSizes(r.KnownSizes);
        log.Clear();
        await using (var l = await next.AcquireAsync("lazy", () => Task.FromResult(new LazyModel { Bytes = 50 }))) { }
        Mr(string.Join(",", next.Resident.Select(m => m.Key)) == "lazy" && log.Contains("dispose c") && !log.Any(s => s.Contains("no known size")),
            $"the seeded 50 made room (c dropped) without an unknown-size guess: {string.Join(",", next.Resident.Select(m => m.Key))} / {string.Join(",", log)}");
        await Task.CompletedTask;
    });

    /// <summary>The device's own free memory limits without a budget; idle scratch is reclaimed before any model is dropped; ReserveBytes is kept free.</summary>
    [TestMethod(Timeout = 30000)]
    public async Task ModelResidency_MeasuredFreeMemory_ReclaimsIdleScratchFirst() => await RunPureTest(async () =>
    {
        var log = new List<string>();
        var dev = new FakeDevice(log) { Free = 100, Idle = 30 };
        await using var r = new ModelResidency(dev) { ReserveBytes = 10 };
        // the fake device's free memory falls as models load and rises when they are disposed
        Task<FakeModel> New(string n, long b) { dev.Free -= b; return Task.FromResult(new FakeModel(n, b, log)); }
        r.Evicted += e => dev.Free += e.Bytes;
        async Task Use(string n, long b) { await using var l = await r.AcquireAsync(n, () => New(n, b), b); }
        await Use("a", 60);                           // free 40
        log.Clear();
        await Use("b", 50);                           // needs 60 of 40: 20 short -> idle scratch gives 20, no model goes
        Mr(MrKeys(r) == "a,b" && string.Join(",", log) == "reclaim", $"idle scratch first, no model dropped: {MrKeys(r)} / {string.Join(",", log)}");
        log.Clear();
        await Use("c", 50);                           // free 10: 50 short; 10 of idle scratch left is not enough -> a goes
        Mr(MrKeys(r) == "b,c" && log.Contains("dispose a"), $"then the least recently used: {MrKeys(r)} / {string.Join(",", log)}");
        Mr(log.IndexOf("reclaim") < log.IndexOf("drain") && log.IndexOf("drain") < log.IndexOf("dispose a"), $"reclaim, drain, dispose: {string.Join(",", log)}");
    });

    /// <summary>Two callers of one key share one load; a load that throws throws in both and leaves nothing; the next acquire loads again; one key is one model type.</summary>
    [TestMethod(Timeout = 30000)]
    public async Task ModelResidency_OneLoadPerKey_FailureReachesEveryCaller() => await RunPureTest(async () =>
    {
        var log = new List<string>();
        await using var r = new ModelResidency(new FakeDevice(log)) { BudgetBytes = 100 };
        int loads = 0;
        var gate = new TaskCompletionSource();
        async Task<FakeModel> Slow() { loads++; await gate.Task; return new FakeModel("s", 10, log); }
        var t1 = r.AcquireAsync("s", Slow, 10); var t2 = r.AcquireAsync("s", Slow, 10);
        await Task.Delay(50); gate.SetResult();
        await using (var l1 = await t1) await using (var l2 = await t2)
        {
            Mr(loads == 1, $"{loads} loads for one key");
            Mr(ReferenceEquals(l1.Model, l2.Model) && r.Resident[0].Leases == 2, "both leases are the same model");
        }
        Mr(r.Resident[0].Leases == 0, "both leases ended");

        var boom = new TaskCompletionSource();
        async Task<FakeModel> Bad() { loads++; await boom.Task; throw new IOException("boom"); }
        var f1 = r.AcquireAsync("bad", Bad, 10); var f2 = r.AcquireAsync("bad", Bad, 10);
        await Task.Delay(50); boom.SetResult();
        int threw = 0;
        foreach (var f in new[] { f1, f2 })
            try { await using var l = await f.WaitAsync(TimeSpan.FromSeconds(5)); } catch (IOException) { threw++; }
        Mr(threw == 2, $"the failure reached {threw} of 2 callers");
        Mr(!r.Resident.Any(m => m.Key == "bad"), "a failed load left something behind");
        await using (var ok = await r.AcquireAsync("bad", () => Task.FromResult(new FakeModel("bad", 10, log)), 10)) Mr(!ok.Model.Disposed, "the next acquire loads again");
        bool wrongType = false;
        try { await using var l = await r.AcquireAsync("s", () => Task.FromResult(new MemoryStream()), 10); } catch (InvalidOperationException) { wrongType = true; }
        Mr(wrongType && r.Resident.First(m => m.Key == "s").Leases == 0, "a key used as another type throws and keeps no lease");
    });

    /// <summary>Tags: a released tag drops the idle model nobody else holds; one another tag holds stays; one in use is dropped by its last lease, and re-acquiring it first keeps it.</summary>
    [TestMethod(Timeout = 30000)]
    public async Task ModelResidency_ReleaseByTag_AndWhileLeased() => await RunPureTest(async () =>
    {
        var log = new List<string>();
        await using var r = new ModelResidency(new FakeDevice(log)) { BudgetBytes = 1000 };
        Task<FakeModel> New(string n) => Task.FromResult(new FakeModel(n, 10, log));
        async Task Use(string n, string tag) { await using var l = await r.AcquireAsync(n, () => New(n), 10, tag); }
        await Use("one", "tab1"); await Use("both", "tab1"); await Use("both", "tab2");
        Mr(await r.ReleaseAsync("tab1") == 1 && MrKeys(r) == "both", $"tab1's own model goes, the shared one stays: {MrKeys(r)}");
        var lease = await r.AcquireAsync("busy", () => New("busy"), 10, "tab1");
        Mr(await r.ReleaseAsync("tab1") == 0 && r.Resident.Any(m => m.Key == "busy"), "a model in use is not dropped now");
        await lease.DisposeAsync();
        Mr(!r.Resident.Any(m => m.Key == "busy") && lease.Model.Disposed, "its last lease dropped it");
        var l1 = await r.AcquireAsync("keep", () => New("keep"), 10, "tab1");
        await r.ReleaseAsync("tab1");
        await using (var l2 = await r.AcquireAsync("keep", () => New("keep"), 10, "tab2")) await l1.DisposeAsync();
        Mr(r.Resident.Any(m => m.Key == "keep"), "re-acquired before its last lease ended: kept");
        var l3 = await r.AcquireAsync("twice", () => New("twice"), 10);
        await l3.DisposeAsync(); await l3.DisposeAsync(); l3.Dispose();
        Mr(r.Resident.First(m => m.Key == "twice").Leases == 0, "ending a lease twice counts once");
    });

    /// <summary>Holders that load themselves take part in LRU, can refuse, and are never the first victim only because they were never touched; a lost device forgets models without disposing them.</summary>
    [TestMethod(Timeout = 30000)]
    public async Task ModelResidency_RegisteredHolders_AndALostDevice() => await RunPureTest(async () =>
    {
        var log = new List<string>();
        await using var r = new ModelResidency(new FakeDevice(log)) { BudgetBytes = 100 };
        Task<FakeModel> New(string n) => Task.FromResult(new FakeModel(n, 40, log));
        async Task Use(string n) { await using var l = await r.AcquireAsync(n, () => New(n), 40); }
        bool vadResident = true, vadBusy = true; int vadAsked = 0;
        await Use("old");
        r.Register("vad", () => vadResident, () => 40, () => { vadAsked++; if (vadBusy) return Task.FromResult(false); vadResident = false; return Task.FromResult(true); });
        await Use("new");                                        // 40 + 40 + 40 > 100: "old" is older than the freshly registered holder
        Mr(MrKeys(r) == "new" && vadAsked == 0, $"a holder that was never touched is not the oldest: {MrKeys(r)}, asked {vadAsked}");
        await Use("new2");                                       // vad is older now, refuses; then "new" goes
        Mr(vadResident && vadAsked == 1 && MrKeys(r) == "new2", $"a refusing holder stays and the next victim goes: {MrKeys(r)}, asked {vadAsked}");
        vadBusy = false;
        await Use("new3");
        Mr(!vadResident, "an idle holder unloads when it is the oldest");
        r.Unregister("vad");
        await r.EnsureRoomForAsyncExpectsUnknown();

        var model = new FakeModel("gone", 10, log);
        await using (var l = await r.AcquireAsync("gone", () => Task.FromResult(model), 10)) r.Invalidate();
        Mr(!model.Disposed && r.Resident.Count == 0, "a lost device: forgotten, never disposed into the dead device");
        var gate = new TaskCompletionSource();
        var late = r.AcquireAsync("late", async () => { await gate.Task; return new FakeModel("late", 10, log); }, 10);
        await Task.Delay(30); r.Invalidate(); gate.SetResult();
        bool threw = false; try { await using var l = await late; } catch (InvalidOperationException) { threw = true; }
        Mr(threw && !r.Resident.Any(m => m.Key == "late"), "a load that outlived its device throws");
    });
}

internal static class ModelResidencyTestExtensions
{
    // EnsureRoomForAsync on a key nobody registered is a caller's mistake
    public static async Task EnsureRoomForAsyncExpectsUnknown(this ModelResidency r)
    {
        bool threw = false;
        try { await r.EnsureRoomForAsync("vad"); } catch (ArgumentException) { threw = true; }
        if (!threw) throw new Exception("EnsureRoomForAsync on an unregistered holder did not throw");
    }
}
