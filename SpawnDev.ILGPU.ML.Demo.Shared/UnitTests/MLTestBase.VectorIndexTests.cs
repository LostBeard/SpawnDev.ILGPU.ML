using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Retrieval;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

using FV = global::ILGPU.Runtime.ArrayView1D<float, global::ILGPU.Stride1D.Dense>;
using IV = global::ILGPU.Runtime.ArrayView1D<int, global::ILGPU.Stride1D.Dense>;

public abstract partial class MLTestBase
{
    /// <summary>
    /// <see cref="DeviceVectorIndex"/> against a host brute force on the same rows, every backend (WebGL ranks on the host). 10,000 unit vectors of 384
    /// (half bulk-loaded, half added one by one from capacity 1,024: the growth events and the rows surviving every growth), searched for k = 1, 10, 64:
    /// plain cosine; with an allow mask and an addend at dotWeight 0.7; after removing 1,000 rows (never returned, their rows read back as zeros, the
    /// lowest freed slot reused); a tiny index (fewer qualifying rows than k); and 5,000 identical rows (more ties at the floor than the gather holds: the
    /// overflow path), which must return the lowest slots. A result differing from the host's only in the order of scores within 1e-4 (float sums in
    /// another order: FMA) counts as equal.
    /// </summary>
    [TestMethod(Timeout = 300000)]
    public async Task DeviceVectorIndex_MatchesHostReference() => await RunTest(async accelerator =>
    {
        const int Dim = 384, N = 10000;
        var rng = new Random(9);
        float[] Unit() { var v = new float[Dim]; double s = 0; for (int i = 0; i < Dim; i++) { v[i] = (float)(rng.NextDouble() * 2 - 1); s += v[i] * v[i]; } float inv = (float)(1 / Math.Sqrt(s)); for (int i = 0; i < Dim; i++) v[i] *= inv; return v; }
        var rows = new float[N][];
        for (int i = 0; i < N; i++) rows[i] = Unit();
        var notes = new List<string>();
        bool ok = true;
        using var index = new DeviceVectorIndex(accelerator, Dim, capacity: 1024);
        var grown = new List<int>();
        index.CapacityChanged += c => grown.Add(c);
        var flat = new float[(N / 2) * Dim];
        for (int i = 0; i < N / 2; i++) Array.Copy(rows[i], 0, flat, i * Dim, Dim);
        var bulk = index.AddRange(flat, N / 2);
        bool slotsOk = bulk.Length == N / 2 && bulk[0] == 0 && bulk[^1] == N / 2 - 1;
        for (int i = N / 2; i < N; i++) slotsOk &= index.Add(rows[i]) == i;
        bool growthOk = grown.SequenceEqual(new[] { 8192, 16384 }) && index.Capacity == 16384 && index.Count == N;
        notes.Add($"slots {(slotsOk ? "in order" : "WRONG")}, growth {string.Join("->", grown)} {(growthOk ? "" : "WRONG")}");
        ok &= slotsOk && growthOk;
        // sizes without growth: an index made at each capacity, half filled in one bulk load, its row 17 must find itself
        foreach (int cap in new[] { 1024, 4096, 8192, 16384 })
        {
            using var sized = new DeviceVectorIndex(accelerator, Dim, cap);
            int fill = cap / 2;
            var buf = new float[fill * Dim];
            for (int i = 0; i < fill; i++) Array.Copy(rows[i % N], 0, buf, i * Dim, Dim);
            sized.AddRange(buf, fill);
            var self = await sized.SearchAsync(rows[17], 1);
            bool selfOk = self.Length == 1 && self[0].Slot == 17 && MathF.Abs(self[0].Score - 1f) < 1e-3f;
            notes.Add($"capacity {cap} (no growth): row 17 {(selfOk ? "finds itself" : $"finds {(self.Length > 0 ? self[0].Slot + ":" + self[0].Score.ToString("F4") : "nothing")}")}");
            ok &= selfOk;
        }
        // how the rows got there: (1) bulk load growing from empty, (2) single adds that cross a growth, (3) single adds, no growth
        async Task<string> Variant(string name, Func<DeviceVectorIndex> make, Action<DeviceVectorIndex> fill, int probe)
        {
            using var v = make();
            fill(v);
            var self = await v.SearchAsync(rows[probe], 1);
            bool good = self.Length == 1 && self[0].Slot == probe && MathF.Abs(self[0].Score - 1f) < 1e-3f;
            ok &= good;
            return $"{name}: row {probe} {(good ? "finds itself" : $"finds {(self.Length > 0 ? self[0].Slot + ":" + self[0].Score.ToString("F4") : "nothing")}")}";
        }
        notes.Add(await Variant("bulk 5000 into capacity 1024 (grows from empty)", () => new DeviceVectorIndex(accelerator, Dim, 1024), v => v.AddRange(flat, N / 2), 17));
        notes.Add(await Variant("5000 single adds into capacity 4096 (grows with data)", () => new DeviceVectorIndex(accelerator, Dim, 4096), v => { for (int i = 0; i < 5000; i++) v.Add(rows[i]); }, 4999));
        notes.Add(await Variant("5000 single adds into capacity 8192 (no growth)", () => new DeviceVectorIndex(accelerator, Dim, 8192), v => { for (int i = 0; i < 5000; i++) v.Add(rows[i]); }, 4999));
        // the stored rows (one from the bulk load, one added singly after the last growth) read back as written, and each finds itself
        foreach (int probe in new[] { 17, N - 1 })
        {
            var back = await index.ReadRowAsync(probe);
            bool rowOk = back.AsSpan().SequenceEqual(rows[probe]);
            var self = await index.SearchAsync(rows[probe], 1);
            bool selfOk = self.Length == 1 && self[0].Slot == probe && MathF.Abs(self[0].Score - 1f) < 1e-3f;
            notes.Add($"row {probe} {(rowOk ? "stored" : $"STORED WRONG (first {back[0]} vs {rows[probe][0]})")}, finds {(selfOk ? "itself" : $"{(self.Length > 0 ? self[0].Slot + ":" + self[0].Score.ToString("F4") : "nothing")}")}");
            ok &= rowOk && selfOk;
        }

        var live = Enumerable.Range(0, N).ToHashSet();
        float[]? allowHost = null, addHost = null; float w = 1f;
        (int, float)[] Host(float[] q, int k)
        {
            var sc = new List<(int, float)>();
            foreach (int s in live)
            {
                if (allowHost != null && allowHost[s] == 0) continue;
                float d = 0f; var r = rows[s];
                for (int j = 0; j < Dim; j++) d += r[j] * q[j];
                sc.Add((s, w * d + (addHost?[s] ?? 0f)));
            }
            sc.Sort((a, b) => a.Item2 != b.Item2 ? b.Item2.CompareTo(a.Item2) : a.Item1.CompareTo(b.Item1));
            return sc.Take(k).ToArray();
        }
        // equal, or the same scores where only near-equal scores swap places
        bool Same((int Slot, float Score)[] dev, (int, float)[] host)
        {
            if (dev.Length != host.Length) return false;
            for (int j = 0; j < dev.Length; j++)
            {
                if (MathF.Abs(dev[j].Score - host[j].Item2) > 1e-4f) return false;
                if (dev[j].Slot == host[j].Item1) continue;
                bool swap = host.Any(h => h.Item1 == dev[j].Slot && MathF.Abs(h.Item2 - dev[j].Score) <= 1e-4f);
                if (!swap) return false;
            }
            return true;
        }
        async Task Check(string name, int queries, IV? allow = null, FV? addend = null)
        {
            int bad = 0;
            string? first = null;
            foreach (int k in new[] { 1, 10, 64 })
                for (int qi = 0; qi < queries; qi++)
                {
                    var q = Unit();
                    var dev = await index.SearchAsync(q, k, allow, addend, w);
                    var host = Host(q, k);
                    if (Same(dev, host)) continue;
                    bad++;
                    first ??= $" [k {k}: device {string.Join(", ", dev.Take(3).Select(d => $"{d.Slot}:{d.Score:F4}"))} | host {string.Join(", ", host.Take(3).Select(h => $"{h.Item1}:{h.Item2:F4}"))} | {dev.Length} vs {host.Length} results]";
                }
            notes.Add($"{name}: {3 * queries - bad}/{3 * queries} = host{first}");
            ok &= bad == 0;
        }
        await Check("cosine", 8);

        allowHost = new float[16384]; addHost = new float[16384];
        for (int s = 0; s < N; s++) { allowHost[s] = rng.NextDouble() < 0.5 ? 1 : 0; addHost[s] = (float)(rng.NextDouble() * 0.2); }
        using var allowBuf = accelerator.Allocate1D(allowHost.Select(a => (int)a).ToArray());
        using var addBuf = accelerator.Allocate1D(addHost);
        w = 0.7f;
        await Check("allow + addend, w 0.7", 6, allowBuf.View, addBuf.View);
        allowHost = null; addHost = null; w = 1f;
        // one search's floor must not carry into the next: a search whose scores all sit near 10 (an addend of 10 everywhere), then a plain one (scores
        // near 0). A stale floor of ~10 would gather nothing on the second search.
        using (var high = accelerator.Allocate1D(Enumerable.Repeat(10f, 16384).ToArray()))
        {
            await index.SearchAsync(Unit(), 64, null, high.View, 1f);
            var q1 = Unit();
            var dev = await index.SearchAsync(q1, 64);
            bool good = Same(dev, Host(q1, 64));
            notes.Add($"a plain search after one scoring near 10: {(good ? "= host" : $"WRONG ({dev.Length} results, best {dev.FirstOrDefault().Score:F4})")}");
            ok &= good;
        }

        var removed = Enumerable.Range(0, N).Where(_ => rng.NextDouble() < 0.1).Take(1000).ToList();
        foreach (int s in removed) { index.Remove(s); live.Remove(s); }
        await Check("after 1,000 removals", 6);
        bool zeroed = (await index.ReadRowAsync(removed[0])).All(v => v == 0f) && (await index.ReadRowAsync(removed[^1])).All(v => v == 0f);
        int reuse = index.Add(rows[removed.Min()]);
        live.Add(reuse);
        bool reused = reuse == removed.Min() && index.Count == N - removed.Count + 1;
        notes.Add($"removed rows {(zeroed ? "read back zero" : "NOT ZERO")}, freed slot {(reused ? "reused (lowest first)" : "NOT reused")}");
        ok &= zeroed && reused;

        // a tiny index: fewer rows than k
        using (var tiny = new DeviceVectorIndex(accelerator, Dim, 8))
        {
            for (int i = 0; i < 5; i++) tiny.Add(rows[i]);
            var r = await tiny.SearchAsync(rows[2], 64);
            bool tinyOk = r.Length == 5 && r[0].Slot == 2 && MathF.Abs(r[0].Score - 1f) < 1e-4f;
            notes.Add($"tiny index: {r.Length} results, best {r.FirstOrDefault().Slot} {(tinyOk ? "" : "WRONG")}");
            ok &= tinyOk;
        }
        // ties past the gather: 5,000 identical rows, the 64 best are the 64 lowest slots
        using (var ties = new DeviceVectorIndex(accelerator, Dim, 5000))
        {
            var same = new float[5000 * Dim];
            for (int i = 0; i < 5000; i++) Array.Copy(rows[0], 0, same, i * Dim, Dim);
            ties.AddRange(same, 5000);
            var r = await ties.SearchAsync(rows[0], 64);
            bool tiesOk = r.Length == 64 && r.Select(x => x.Slot).SequenceEqual(Enumerable.Range(0, 64));
            notes.Add($"5,000 tied rows: {(tiesOk ? "the 64 lowest slots" : "WRONG: " + string.Join(",", r.Take(5).Select(x => x.Slot)))}");
            ok &= tiesOk;
        }
        var report = $"{accelerator.AcceleratorType}: " + string.Join("; ", notes);
        Console.WriteLine("[VectorIndex] " + report);
        if (!ok) throw new Exception(report);
        return report;
    });
}
