using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Kernels;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    /// <summary>
    /// <see cref="MmaFlashPrefill"/> (causal grouped-query flash attention on f16 tensor cores, CUDA sm_80+, head dim 128, K / V from F16 caches) against a CPU
    /// double-precision oracle on the same f16 K / V: rows at positions p .. p + R - 1 attend keys 0 .. their position. Cases: a reused prefix (p = 77) with
    /// R = 200 rows (two row tiles, the second partial) and G = 4; p = 0 with R = 37 (one partial tile) and G = 1; p = 300 with R = 129 and G = 2 (a row tile of
    /// one row). Q and the probabilities are rounded to f16 inside the kernel, so each output is held to 4e-3 x sum_j p_j |v_jd| + 1e-5 (f16 keeps 11 significant
    /// bits). Added 2026-11-02 with the kernel.
    /// </summary>
    [TestMethod]
    public async Task MmaFlashPrefill_MatchesCausalGqaOracle() => await RunTest(async accelerator =>
    {
        if (!CudaCaps.TensorCores(accelerator)) throw new UnsupportedTestException("the tensor-core flash prefill runs on CUDA sm_80+");
        const int D = 128;
        var kernel = accelerator.LoadStreamKernel<ArrayView1D<float, Stride1D.Dense>, ArrayView<uint>, ArrayView<uint>,
            ArrayView1D<float, Stride1D.Dense>, int, int, int, int>(MmaFlashPrefill.Kernel);
        var notes = new List<string>();
        foreach (var (p, R, H, kvH) in new[] { (77, 200, 8, 2), (0, 37, 4, 4), (300, 129, 4, 2) })
        {
            var rng = new Random(5 + p + R + H);
            int keys = p + R, G = H / kvH;
            var q = new float[R * H * D];
            for (int i = 0; i < q.Length; i++) q[i] = (float)(rng.NextDouble() * 2 - 1);
            // the caches: F16 values (scores of O(1..10) so the softmax is neither flat nor one-hot)
            var kh = new System.Half[keys * kvH * D]; var vh = new System.Half[keys * kvH * D];
            for (int i = 0; i < kh.Length; i++) { kh[i] = (System.Half)(float)((rng.NextDouble() * 2 - 1) * 0.6); vh[i] = (System.Half)(float)(rng.NextDouble() * 2 - 1); }
            uint[] Words(System.Half[] a) { var u = new uint[a.Length / 2]; for (int i = 0; i < u.Length; i++) u[i] = BitConverter.HalfToUInt16Bits(a[2 * i]) | ((uint)BitConverter.HalfToUInt16Bits(a[2 * i + 1]) << 16); return u; }
            using var qBuf = accelerator.Allocate1D(q);
            using var kBuf = accelerator.Allocate1D(Words(kh));
            using var vBuf = accelerator.Allocate1D(Words(vh));
            using var oBuf = accelerator.Allocate1D<float>(q.Length);
            kernel(new KernelConfig(((R + MmaFlashPrefill.QTile - 1) / MmaFlashPrefill.QTile) * H, MmaFlashPrefill.Threads), qBuf.View, kBuf.View.BaseView, vBuf.View.BaseView, oBuf.View, R, H, kvH, p);
            await accelerator.SynchronizeAsync();
            var got = await oBuf.CopyToHostAsync<float>(0, q.Length);
            int bad = 0; double worst = 0;
            Parallel.For(0, R * H, () => (bad: 0, worst: 0.0), (rh, _, acc) =>
            {
                int r = rh / H, h = rh % H, hk = h / G, pos = p + r;
                var s = new double[pos + 1];
                double mx = double.NegativeInfinity;
                for (int j = 0; j <= pos; j++)
                {
                    double dot = 0;
                    for (int d = 0; d < D; d++) dot += q[(r * H + h) * D + d] * (double)(float)kh[(j * kvH + hk) * D + d];
                    s[j] = dot / Math.Sqrt(D); mx = Math.Max(mx, s[j]);
                }
                double sum = 0;
                for (int j = 0; j <= pos; j++) { s[j] = Math.Exp(s[j] - mx); sum += s[j]; }
                for (int d = 0; d < D; d++)
                {
                    double o = 0, a = 0;
                    for (int j = 0; j <= pos; j++) { double v = (float)vh[(j * kvH + hk) * D + d]; o += s[j] * v; a += s[j] * Math.Abs(v); }
                    o /= sum; a /= sum;
                    double e = Math.Abs(got[(r * H + h) * D + d] - o), bound = 4e-3 * a + 1e-5;
                    if (!(e <= bound)) acc.bad++;
                    acc.worst = Math.Max(acc.worst, e / bound);
                }
                return acc;
            }, acc => { lock (notes) { bad += acc.bad; worst = Math.Max(worst, acc.worst); } });
            if (bad > 0) throw new Exception($"p={p} R={R} H={H} kvH={kvH}: {bad} of {got.Length} outputs past the bound (worst {worst:F2}x)");
            notes.Add($"p={p} R={R} H={H} kvH={kvH}: worst {worst:F3}x the bound");
        }
        return string.Join("; ", notes);
    });
}
