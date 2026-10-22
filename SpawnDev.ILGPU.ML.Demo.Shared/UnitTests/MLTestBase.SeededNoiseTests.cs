using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    /// <summary>
    /// The diffusion pipelines' seeded noise: deterministic, standard normal, and independent across NEARBY seeds. The mean correlation of the noise of 200 adjacent seed pairs (4096 samples each, so about
    /// 0.0016 standard error) must stay within 0.006; the legacy <c>new Random(seed)</c> + Box-Muller generator this replaced is measured alongside for the record (about +0.025 reported by SpawnGen).
    /// </summary>
    [TestMethod(Timeout = 120000)]
    public async Task SeededNoise_IsStandardNormalAndIndependentAcrossNearbySeeds() => await RunTest(async accelerator =>
    {
        var a = SeededNoise.Gaussian(10000, 42); var b = SeededNoise.Gaussian(10000, 42);
        if (!a.SequenceEqual(b)) throw new Exception("the same seed gave different noise");
        double mean = a.Average(), sd = Math.Sqrt(a.Select(v => (v - mean) * (v - mean)).Average());
        if (Math.Abs(mean) > 0.04 || Math.Abs(sd - 1) > 0.04) throw new Exception($"not standard normal: mean {mean:F3}, std {sd:F3}");
        double Corr(float[] x, float[] y) { double mx = x.Average(), my = y.Average(), sxy = 0, sxx = 0, syy = 0; for (int i = 0; i < x.Length; i++) { sxy += (x[i] - mx) * (y[i] - my); sxx += (x[i] - mx) * (x[i] - mx); syy += (y[i] - my) * (y[i] - my); } return sxy / Math.Sqrt(sxx * syy); }
        float[] Legacy(int seed) { var r = new Random(seed); var o = new float[4096]; for (int i = 0; i < o.Length; i++) { double u1 = 1.0 - r.NextDouble(), u2 = r.NextDouble(); o[i] = (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2)); } return o; }
        double sumNew = 0, sumOld = 0; int n = 200;
        for (int s = 0; s < n; s++)
        {
            sumNew += Corr(SeededNoise.Gaussian(4096, s), SeededNoise.Gaussian(4096, s + 1));
            sumOld += Corr(Legacy(s), Legacy(s + 1));
        }
        Console.WriteLine($"[SeededNoise] mean correlation of adjacent seeds over {n} pairs: new {sumNew / n:F4}, legacy Random {sumOld / n:F4}");
        if (Math.Abs(sumNew / n) > 0.006) throw new Exception($"adjacent seeds are correlated: mean {sumNew / n:F4}");
        await Task.CompletedTask;
        return $"adjacent-seed correlation {sumNew / n:F4} (legacy {sumOld / n:F4})";
    });
}
