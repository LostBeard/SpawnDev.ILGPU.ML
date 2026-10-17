using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using SpawnDev.ILGPU;

namespace Trellis2Harness;

/// <summary>
/// Probe: can inline PTX issue a bf16 tensor-core MMA from SpawnDev.ILGPU on this machine, and is the
/// fragment layout understood? One warp computes D[16x8] = A[16x16] * B[16x8] (+0) with
/// mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 and the result is compared with a CPU product of the
/// same bf16 values. A[row][k] row-major; B is passed as Bt[n][k] (the nn.Linear weight layout).
/// </summary>
internal static class MmaProbe
{
    private static uint Pack(ushort lo, ushort hi) => (uint)lo | ((uint)hi << 16);

    private static void Kernel(Index1D i, ArrayView1D<ushort, Stride1D.Dense> a, ArrayView1D<ushort, Stride1D.Dense> bt, ArrayView1D<float, Stride1D.Dense> d)
    {
        int lane = Group.IdxX;
        int g = lane >> 2, t = lane & 3;
        // A fragment (row-major 16x16): a0 (g, 2t..), a1 (g+8, 2t..), a2 (g, 2t+8..), a3 (g+8, 2t+8..)
        Input<uint> a0 = (uint)a[g * 16 + 2 * t] | ((uint)a[g * 16 + 2 * t + 1] << 16);
        Input<uint> a1 = (uint)a[(g + 8) * 16 + 2 * t] | ((uint)a[(g + 8) * 16 + 2 * t + 1] << 16);
        Input<uint> a2 = (uint)a[g * 16 + 2 * t + 8] | ((uint)a[g * 16 + 2 * t + 9] << 16);
        Input<uint> a3 = (uint)a[(g + 8) * 16 + 2 * t + 8] | ((uint)a[(g + 8) * 16 + 2 * t + 9] << 16);
        // B fragment: b0 (k = 2t.., n = g), b1 (k = 2t+8.., n = g); Bt is [n][k]
        Input<uint> b0 = (uint)bt[g * 16 + 2 * t] | ((uint)bt[g * 16 + 2 * t + 1] << 16);
        Input<uint> b1 = (uint)bt[g * 16 + 2 * t + 8] | ((uint)bt[g * 16 + 2 * t + 9] << 16);
        Ref<float> c0 = default, c1 = default, c2 = default, c3 = default;
        CudaAsm.EmitRef(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};",
            ref c0, ref c1, ref c2, ref c3, ref a0, ref a1, ref a2, ref a3, ref b0, ref b1);
        // D fragment: c0 (g, 2t), c1 (g, 2t+1), c2 (g+8, 2t), c3 (g+8, 2t+1)
        d[g * 8 + 2 * t] = c0.Value;
        d[g * 8 + 2 * t + 1] = c1.Value;
        d[(g + 8) * 8 + 2 * t] = c2.Value;
        d[(g + 8) * 8 + 2 * t + 1] = c3.Value;
    }

    private static ushort ToBf16(float f)   // round to nearest even
    {
        uint bits = BitConverter.SingleToUInt32Bits(f);
        uint rounded = bits + 0x7FFFu + ((bits >> 16) & 1u);
        return (ushort)(rounded >> 16);
    }
    private static float FromBf16(ushort b) => BitConverter.UInt32BitsToSingle((uint)b << 16);

    public static int Run()
    {
        var (ctx, accel) = Gpu.Create();
        using var _c = ctx; using var _a = accel;
        if (accel is not CudaAccelerator) { Console.WriteLine("mma probe needs the CUDA backend"); return 2; }
        var rng = new Random(7);
        var a = new ushort[16 * 16]; var bt = new ushort[8 * 16];
        for (int i = 0; i < a.Length; i++) a[i] = ToBf16((float)(rng.NextDouble() * 2 - 1));
        for (int i = 0; i < bt.Length; i++) bt[i] = ToBf16((float)(rng.NextDouble() * 2 - 1));
        var want = new float[16 * 8];
        for (int m = 0; m < 16; m++)
            for (int n = 0; n < 8; n++)
            {
                double s = 0;
                for (int k = 0; k < 16; k++) s += (double)FromBf16(a[m * 16 + k]) * FromBf16(bt[n * 16 + k]);
                want[m * 8 + n] = (float)s;
            }
        using var ab = accel.Allocate1D(a); using var bb = accel.Allocate1D(bt); using var db = accel.Allocate1D<float>(16 * 8);
        var k2 = accel.LoadStreamKernel<ArrayView1D<ushort, Stride1D.Dense>, ArrayView1D<ushort, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(
            (av, bv, dv) => Kernel(Group.IdxX, av, bv, dv));
        k2(new KernelConfig(1, 32), ab.View, bb.View, db.View);
        accel.Synchronize();
        var got = db.GetAsArray1D();
        return Golden.Compare("mma m16n8k16 bf16", want, got, 1e-6) ? 0 : 1;
    }
}
