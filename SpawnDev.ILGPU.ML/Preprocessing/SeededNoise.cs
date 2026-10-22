namespace SpawnDev.ILGPU.ML.Preprocessing;

/// <summary>
/// Seeded standard-normal noise for the diffusion pipelines: xoshiro256** seeded through splitmix64, then Box-Muller. The same on every platform and runtime, and the streams of NEARBY seeds
/// are independent (<c>new Random(seed)</c>, the legacy subtractive generator this replaces, gives positively correlated streams for neighbouring seeds: measured about +0.025 between
/// the per-step seeds of SpawnGen's ancestral samplers, which inflated their output spread; reported by that project).
/// Not torch's generator: a seed here is not a seed there.
/// </summary>
public static class SeededNoise
{
    /// <summary><paramref name="count"/> standard normal samples for <paramref name="seed"/>.</summary>
    public static float[] Gaussian(long count, long seed)
    {
        ulong sm = unchecked((ulong)seed);
        ulong SplitMix() { sm += 0x9E3779B97F4A7C15UL; ulong z = sm; z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL; z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL; return z ^ (z >> 31); }
        ulong s0 = SplitMix(), s1 = SplitMix(), s2 = SplitMix(), s3 = SplitMix();
        static ulong Rotl(ulong x, int k) => (x << k) | (x >> (64 - k));
        double Next()                                                                  // uniform in (0, 1]
        {
            ulong r = Rotl(s1 * 5, 7) * 9, t = s1 << 17;
            s2 ^= s0; s3 ^= s1; s1 ^= s2; s0 ^= s3; s2 ^= t; s3 = Rotl(s3, 45);
            return ((r >> 11) + 1) * (1.0 / 9007199254740992.0);
        }
        var x = new float[count];
        for (long i = 0; i < count; i += 2)
        {
            double u1 = Next(), u2 = Next(), r = Math.Sqrt(-2.0 * Math.Log(u1));
            x[i] = (float)(r * Math.Cos(2 * Math.PI * u2));
            if (i + 1 < count) x[i + 1] = (float)(r * Math.Sin(2 * Math.PI * u2));
        }
        return x;
    }
}
