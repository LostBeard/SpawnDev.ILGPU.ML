using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Kernels;
using SpawnDev.ILGPU.ML.Tensors;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// <see cref="BufferPool.RentHalf"/> — pooled fp16 (Half) ACTIVATION buffers, the foundational allocation
/// primitive for mixed-precision activations (the executor keeps heavy intermediates fp16). Verifies a
/// rented Half buffer is a usable GPU tensor (write via the convert kernels, read back correct) AND that
/// Return→Rent of the same size REUSES the buffer (pool bounded, no growth) — on every backend.
/// </summary>
public abstract partial class MLTestBase
{
    [TestMethod]
    public async Task BufferPool_RentHalf_UsableAndReused_AllBackends() => await RunTest(async accelerator =>
    {
        var pool = new BufferPool(accelerator);
        var conv = new PrecisionConvertKernels(accelerator);
        try
        {
            int[] shape = { 64, 16 }; // 1024 elements
            const int n = 1024;
            var rng = new Random(61);
            var x = new float[n];
            for (int i = 0; i < n; i++) x[i] = (float)(rng.NextDouble() * 6 - 3);

            // Rent a Half activation buffer, fill it (fp32 -> fp16) and read it back (fp16 -> fp32).
            var ht = pool.RentHalf(shape, "x");
            if (ht.ElementCount != n) throw new Exception($"RentHalf shape wrong: {ht.ElementCount} != {n}");
            using var fIn = accelerator.Allocate1D(x);
            using var fOut = accelerator.Allocate1D<float>(n);
            conv.FloatToHalf(fIn.View, ht.Data, n);
            conv.HalfToFloat(ht.Data, fOut.View, n);
            await accelerator.SynchronizeAsync();
            var got = await fOut.CopyToHostAsync<float>(0, n);
            for (int i = 0; i < n; i++)
            {
                float want = (float)(global::ILGPU.Half)x[i];
                if (MathF.Abs(got[i] - want) > MathF.Max(1e-3f, MathF.Abs(want) * 1e-3f))
                    throw new Exception($"RentHalf buffer round-trip @{i}: got {got[i]}, want {want}");
            }

            // Return it, then rent the same size again — must REUSE (pool count must not grow).
            int countBefore = pool.AllocatedHalfBufferCount;
            pool.ReturnHalf(ht);
            var ht2 = pool.RentHalf(shape, "y");
            int countAfter = pool.AllocatedHalfBufferCount;
            if (countAfter != countBefore)
                throw new Exception($"RentHalf did not reuse a returned buffer: pool grew {countBefore} -> {countAfter}");
            if (ht2.ElementCount != n) throw new Exception("reused Half buffer wrong size");

            Console.WriteLine($"[BufferPool] RentHalf usable + reused (pool bounded at {countAfter}) on {BackendName}");
        }
        finally { pool.Dispose(); }
    });
}
