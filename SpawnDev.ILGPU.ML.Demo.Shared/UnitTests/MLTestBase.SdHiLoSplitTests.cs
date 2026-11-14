using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.StableDiffusion;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    /// <summary>
    /// The VAE loader's fp32 -> (hi, lo) fp16 split (SdWeights.KeepLowHalves: the second conv pass that brings fp32 VAEs to ComfyUI's numbers) on every backend:
    /// hi and lo must be the host's IEEE round-to-nearest-even halves bit for bit (hi = (Half)v, lo = (Half)(v - (float)hi)), so hi + lo carries ~22 bits of v. RED when a backend's
    /// float -> Half conversion does not round (SpawnDev.ILGPU before 5.4.32 on OpenCL, Wasm and WebGL: lo all zero, the hi/lo VAE silently back to fp16). Synthetic values: no
    /// model files, every lane.
    /// </summary>
    [TestMethod]
    public async Task Sd_HiLoSplit_RoundsOnEveryBackend() => await RunTest(async accelerator =>
    {
        const int N = 4096;
        var rng = new Random(11);
        async Task<(int HiWrong, int LoWrong, int LoZero, int WantLoZero, double Recon)> Split(float[] v, int layout = 0, int la = 0, int lb = 0)
        {
            using var src = accelerator.Allocate1D(v);
            using var hi = accelerator.Allocate1D<global::ILGPU.Half>(v.Length); using var lo = accelerator.Allocate1D<global::ILGPU.Half>(v.Length);
            SdTensorStreamer.SplitHiLo(accelerator, src.View, hi.View, lo.View, layout, la, lb);
            var gotHi = await hi.View.CopyToHostAsync(); var gotLo = await lo.View.CopyToHostAsync();
            int hiWrong = 0, loWrong = 0, loZero = 0, wantLoZero = 0; double recon = 0;
            for (int i = 0; i < v.Length; i++)
            {
                float vi = v[SdTensorStreamer.SourceIndexForTests(i, layout, la, lb)];                  // the loader's source order (a conv weight is permuted)
                var h = (System.Half)vi;
                var l = (System.Half)(vi - (float)h);
                if (BitConverter.HalfToUInt16Bits(h) != gotHi[i].RawValue) hiWrong++;
                if (BitConverter.HalfToUInt16Bits(l) != gotLo[i].RawValue) loWrong++;
                if ((float)gotLo[i] == 0f) loZero++;
                if ((float)l == 0f) wantLoZero++;
                recon = Math.Max(recon, Math.Abs((float)gotHi[i] + (float)gotLo[i] - vi) / Math.Abs(vi));
            }
            return (hiWrong, loWrong, loZero, wantLoZero, recon);
        }
        // (a) |v| in [0.5, 64), both signs: every lo is a normal fp16 (>= ~2^-12): rounding alone
        var a = new float[N];
        for (int i = 0; i < N; i++) a[i] = (float)((rng.Next(2) == 0 ? -1 : 1) * (1 + rng.NextDouble()) * Math.Pow(2, rng.Next(-1, 6)));
        // (b) |v| in [2^-12, 2^-2): a VAE's weights, whose lo halves are mostly fp16 SUBNORMALS (below 2^-14): a backend that flushes them loses the second pass for most weights
        var b = new float[N];
        for (int i = 0; i < N; i++) b[i] = (float)((rng.Next(2) == 0 ? -1 : 1) * (1 + rng.NextDouble()) * Math.Pow(2, rng.Next(-12, -2)));
        var ra = await Split(a); var rb = await Split(b);
        var rc = await Split(b, 1, 64, 4 * 4);                                               // (c) the weight-sized values through the conv permutation (layout 1: cin 64, 16 taps), as a VAE load runs it
        var report = $"normal lo ({N}): hi wrong {ra.HiWrong}, lo wrong {ra.LoWrong}, lo zero {ra.LoZero} (host {ra.WantLoZero}), worst |hi + lo - v| / |v| {ra.Recon:E2}; "
            + $"weight-sized ({N}, lo mostly subnormal): hi wrong {rb.HiWrong}, lo wrong {rb.LoWrong}, lo zero {rb.LoZero} (host {rb.WantLoZero}), worst {rb.Recon:E2}; "
            + $"conv-permuted: hi wrong {rc.HiWrong}, lo wrong {rc.LoWrong}, lo zero {rc.LoZero} (host {rc.WantLoZero})";
        Console.WriteLine("[Sd] hi/lo split: " + report);
        Ur_Expect(ra.HiWrong == 0 && ra.LoWrong == 0 && ra.Recon < 1e-6 && rb.HiWrong == 0 && rb.LoWrong == 0 && rc.HiWrong == 0 && rc.LoWrong == 0, report);   // both bit for bit against the host's IEEE halves
        return report;
    });
}
