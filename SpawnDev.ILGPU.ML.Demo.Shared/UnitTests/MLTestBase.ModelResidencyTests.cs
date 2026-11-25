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
    }

    private static void Mr(bool ok, string what) { if (!ok) throw new Exception(what); }
    private static string MrKeys(ModelResidency r) => string.Join(",", r.Resident.Select(m => m.Key));

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
        await Use("x", 10, 0);                        // size unknown: a and b go first, the load itself would be what over-allocates
        Mr(MrKeys(r) == "x", $"an unknown size drops every idle model before it loads: {MrKeys(r)}");
        await Use("a", 10, 10); await Use("b", 10, 10);
        Mr(MrKeys(r) == "x,a,b", $"all fit: {MrKeys(r)}");
        await Use("x", 10, 0);
        Mr(MrKeys(r) == "a,b,x", $"x is the kept one: nothing dropped: {MrKeys(r)}");
        r.BudgetBytes = 40;
        await Use("c", 10, 10);
        Mr(MrKeys(r) == "a,b,x,c", $"40 is enough for four 10s: {MrKeys(r)}");
        await Use("d", 10, 0);                        // never seen: unknown
        Mr(MrKeys(r) == "d", $"d: unknown again: {MrKeys(r)}");

        await using var none = new ModelResidency(new FakeDevice(log));
        var lines = new List<string>(); none.Log = lines.Add;
        for (int i = 0; i < 20; i++) await using (var l = await none.AcquireAsync($"m{i}", () => New($"m{i}", 1L << 30), 1L << 30)) { }
        Mr(none.Resident.Count == 20, $"no budget, no free memory: nothing is dropped ({none.Resident.Count})");
        Mr(lines.Count(l => l.Contains("BudgetBytes")) == 1, "it says once what to set");
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
