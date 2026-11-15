using ILGPU;
using ILGPU.Runtime;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    /// <summary>
    /// fp16 -> bf16 hi + bf16 lo on the device (<see cref="Wan.WanComfyFormat.ToBf16HiLo"/>: how StreamedModelWeights keeps an fp16 bias or norm scale exactly): hi = bf16(v) round to nearest
    /// even, lo = bf16(v - hi), so hi + lo == v exactly. Compared bit for bit with the host. RED when a backend's float -> bf16 -> float round trip does not round (lo all zero: the
    /// fp16 vector silently back to bf16 precision).
    /// </summary>
    [TestMethod]
    public async Task Bf16HiLo_SplitsFp16Exactly() => await RunTest(async accelerator =>
    {
        const int N = 4096;
        var rng = new Random(5);
        var v = new global::ILGPU.Half[N];
        for (int i = 0; i < N; i++) v[i] = (global::ILGPU.Half)(float)((rng.Next(2) == 0 ? -1 : 1) * (1 + rng.NextDouble()) * Math.Pow(2, rng.Next(-8, 8)));
        using var src = accelerator.Allocate1D(v);
        using var hi = accelerator.Allocate1D<BFloat16>(N); using var lo = accelerator.Allocate1D<BFloat16>(N);
        Wan.WanComfyFormat.ToBf16HiLo(accelerator, src.View, hi.View, lo.View);
        var gh = await hi.View.CopyToHostAsync(); var gl = await lo.View.CopyToHostAsync();
        static ushort Bf16Bits(float f) { uint b = BitConverter.SingleToUInt32Bits(f); return (ushort)((b + 0x7FFFu + ((b >> 16) & 1u)) >> 16); }
        static float FromBf16(ushort u) => BitConverter.UInt32BitsToSingle((uint)u << 16);
        int hiWrong = 0, loWrong = 0, loZero = 0, wantLoZero = 0, notExact = 0;
        for (int i = 0; i < N; i++)
        {
            float f = (float)v[i];
            ushort h = Bf16Bits(f), l = Bf16Bits(f - FromBf16(h));
            if (BitConverter.SingleToUInt32Bits((float)gh[i]) >> 16 != h) hiWrong++;
            if (BitConverter.SingleToUInt32Bits((float)gl[i]) >> 16 != l) loWrong++;
            if ((float)gl[i] == 0f) loZero++;
            if (FromBf16(l) == 0f) wantLoZero++;
            if ((float)gh[i] + (float)gl[i] != f) notExact++;
        }
        var report = $"{N} fp16 values: hi wrong {hiWrong}, lo wrong {loWrong}, lo zero {loZero} (host {wantLoZero}), hi + lo != v {notExact}";
        Console.WriteLine("[Bf16] hi/lo: " + report);
        Ur_Expect(hiWrong == 0 && loWrong == 0 && notExact == 0, report);
        return report;
    });
}
