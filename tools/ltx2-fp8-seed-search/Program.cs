// Exhaustive 32-bit seed search for a per-element uniform stream (black-box reproduction of a seeded stochastic rounding).
// The generator is torch's CPU default (MT19937, seeded from the low 32 bits of manual_seed's argument), as measured against torch 2.15:
//   float32 uniform = (word & 0xFFFFFF) * 2^-24; float64 uniform = (word64 & (2^53 - 1)) * 2^-53 with word64 from two words (order checked by --selftest).
// Input: a raw file of float64 pairs (lo, hi) per element (u in [lo, hi) for "round up iff u < frac"); variants also test 1 - u.
// usage: SeedHarness intervals.bin first last [K]  |  SeedHarness --selftest
using System.Diagnostics;

static class Mt
{
    const int N = 624, M = 397;
    // the first count outputs of MT19937 seeded with s (init_genrand), computed without the full twist (count <= 227)
    public static void Outputs(uint s, Span<uint> o, Span<uint> st)
    {
        int count = o.Length, need = M + count + 1;
        st[0] = s;
        for (int i = 1; i < need; i++) st[i] = 1812433253u * (st[i - 1] ^ (st[i - 1] >> 30)) + (uint)i;
        for (int i = 0; i < count; i++)
        {
            uint y = (st[i] & 0x80000000u) | (st[i + 1] & 0x7fffffffu);
            uint v = st[i + M] ^ (y >> 1) ^ ((y & 1u) != 0 ? 0x9908b0dfu : 0u);
            v ^= v >> 11; v ^= (v << 7) & 0x9d2c5680u; v ^= (v << 15) & 0xefc60000u; v ^= v >> 18;
            o[i] = v;
        }
    }
}

static class Program
{
    static int Main(string[] args)
    {
        if (args.Length > 0 && args[0] == "--selftest")
        {
            Span<uint> o = stackalloc uint[4]; var st = new uint[624];
            foreach (uint s in new uint[] { 0, 10, 123456789, 5 })
            {
                Mt.Outputs(s, o, st);
                double f0 = (o[0] & 0xFFFFFF) * Math.ScaleB(1.0, -24), f1 = (o[1] & 0xFFFFFF) * Math.ScaleB(1.0, -24);
                ulong a = ((ulong)o[0] << 32) | o[1], b = ((ulong)o[1] << 32) | o[0];
                Console.WriteLine($"seed {s}: f32 {f0:F9} {f1:F9} | f64 hi-first {(a & ((1UL << 53) - 1)) * Math.ScaleB(1.0, -53):F17} lo-first {(b & ((1UL << 53) - 1)) * Math.ScaleB(1.0, -53):F17}");
            }
            return 0;
        }
        var bytes = File.ReadAllBytes(args[0]);
        var iv = new double[bytes.Length / 8]; Buffer.BlockCopy(bytes, 0, iv, 0, bytes.Length);
        int n = iv.Length / 2;
        long first = long.Parse(args[1]), last = long.Parse(args[2]);
        int K = args.Length > 3 ? int.Parse(args[3]) : 48;
        K = Math.Min(K, Math.Min(n, 100));                      // float64 uses two words per element: 2 K <= 227
        double[] lo = new double[K], hi = new double[K];
        for (int i = 0; i < K; i++) { lo[i] = iv[i]; hi[i] = iv[n + i]; }
        // variants: 0 f32 u, 1 f32 1-u, 2 f64 hi-first u, 3 f64 hi-first 1-u, 4 f64 lo-first u, 5 f64 lo-first 1-u
        string[] names = { "f32 u<frac", "f32 1-u", "f64(hi,lo) u<frac", "f64(hi,lo) 1-u", "f64(lo,hi) u<frac", "f64(lo,hi) 1-u" };
        var sw = Stopwatch.StartNew();
        long tested = 0;
        var found = new System.Collections.Concurrent.ConcurrentBag<string>();
        Parallel.For(first >> 16, (last >> 16) + 1, chunk =>
        {
            Span<uint> o = stackalloc uint[2 * K + 1]; var st = new uint[624];
            long a = Math.Max(first, chunk << 16), b = Math.Min(last, (chunk << 16) | 0xFFFF);
            for (long s = a; s <= b; s++)
            {
                Mt.Outputs((uint)s, o, st);
                for (int v = 0; v < 6; v++)
                {
                    int i = 0;
                    for (; i < K; i++)
                    {
                        double u;
                        if (v < 2) u = (o[i] & 0xFFFFFF) * (1.0 / 16777216.0);
                        else
                        {
                            ulong w = v < 4 ? ((ulong)o[2 * i] << 32) | o[2 * i + 1] : ((ulong)o[2 * i + 1] << 32) | o[2 * i];
                            u = (w & ((1UL << 53) - 1)) * Math.ScaleB(1.0, -53);
                        }
                        if ((v & 1) == 1) u = 1 - u;
                        if (u < lo[i] || u >= hi[i]) break;
                    }
                    if (i == K) { string line = $"MATCH seed {s} variant {names[v]}"; found.Add(line); Console.WriteLine(line); }
                }
            }
            Interlocked.Add(ref tested, b - a + 1);
        });
        Console.WriteLine($"tested {tested} seeds [{first}, {last}], K {K}, {sw.Elapsed.TotalSeconds:F1} s, {found.Count} matches");
        return 0;
    }
}
