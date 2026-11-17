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
        // variants = (how a uniform is made from the 32-bit outputs) x (how many outputs were drawn before this tensor's first element) x (u or 1 - u).
        // One-word kinds: 0 float32 (low 24 bits: torch.rand float32), 1 the top 24 bits, 2 word / 2^32, 3 float16 (low 11 bits), 4 bfloat16 (low 8 bits).
        // Two-word kinds: 5 float64 hi word first (torch.rand float64), 6 lo word first, 7 numpy's ((a >> 5) 2^26 + (b >> 6)) / 2^53.
        string[] kinds = { "f32 low24", "top24", "word/2^32", "f16 low11", "bf16 low8", "f64(hi,lo)", "f64(lo,hi)", "numpy f64" };
        var variants = new List<(int Kind, int Offset, bool Flip)>();
        for (int kind = 0; kind < kinds.Length; kind++)
            for (int off = 0; off < (kind < 5 ? 4 : 2); off++)
                foreach (bool flip in new[] { false, true }) variants.Add((kind, off, flip));
        var vs = variants.ToArray();
        var sw = Stopwatch.StartNew();
        long tested = 0;
        var found = new System.Collections.Concurrent.ConcurrentBag<string>();
        Parallel.For(first >> 16, (last >> 16) + 1, chunk =>
        {
            Span<uint> o = stackalloc uint[2 * K + 4]; var st = new uint[624];
            long a = Math.Max(first, chunk << 16), b = Math.Min(last, (chunk << 16) | 0xFFFF);
            for (long s = a; s <= b; s++)
            {
                Mt.Outputs((uint)s, o, st);
                for (int v = 0; v < vs.Length; v++)
                {
                    var (kind, off, flip) = vs[v];
                    int i = 0;
                    for (; i < K; i++)
                    {
                        double u;
                        if (kind < 5)
                        {
                            uint w = o[off + i];
                            u = kind switch
                            {
                                0 => (w & 0xFFFFFF) * (1.0 / 16777216.0),
                                1 => (w >> 8) * (1.0 / 16777216.0),
                                2 => w * (1.0 / 4294967296.0),
                                3 => (w & 0x7FF) * (1.0 / 2048.0),
                                _ => (w & 0xFF) * (1.0 / 256.0),
                            };
                        }
                        else
                        {
                            uint w0 = o[off + 2 * i], w1 = o[off + 2 * i + 1];
                            if (kind == 7) u = ((w0 >> 5) * 67108864.0 + (w1 >> 6)) * (1.0 / 9007199254740992.0);
                            else
                            {
                                ulong w = kind == 5 ? ((ulong)w0 << 32) | w1 : ((ulong)w1 << 32) | w0;
                                u = (w & ((1UL << 53) - 1)) * (1.0 / 9007199254740992.0);
                            }
                        }
                        if (flip) u = 1 - u;
                        if (u < lo[i] || u >= hi[i]) break;
                    }
                    if (i == K) { string line = $"MATCH seed {s} variant {kinds[kind]} offset {off}{(flip ? " 1-u" : "")}"; found.Add(line); Console.WriteLine(line); }
                }
            }
            Interlocked.Add(ref tested, b - a + 1);
        });
        Console.WriteLine($"tested {tested} seeds [{first}, {last}], K {K}, {sw.Elapsed.TotalSeconds:F1} s, {found.Count} matches");
        return 0;
    }
}
