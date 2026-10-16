using System.Diagnostics;
using SpawnDev.ILGPU.ML.GGUF;
using SpawnDev.ILGPU.ML.Kernels;
using Xunit.Abstractions;

namespace SpawnDev.ILGPU.ML.Tests;

/// <summary>
/// The 128 x 128 tile Q4_K prefill GEMM against the 64 x 64 register-blocked kernel it replaces, on CUDA: random but valid Q4_K weights, same activations, outputs equal to float
/// rounding, and the time and the TFLOPS of each at the shapes of a llama3.1:8b prefill. Skipped where there is no CUDA device.
/// </summary>
[Collection("GemmFlag")]
public class FastGemmCudaTests(ITestOutputHelper output)
{
    /// <summary>Random Q4_K rows: [n rows][k/256 blocks of 144 bytes]: d and dmin small valid halves, random 6-bit scales and mins, random nibbles.</summary>
    private static byte[] RandomQ4K(int n, int k, int seed)
    {
        var r = new Random(seed);
        var blocks = k / 256;
        var bytes = new byte[n * blocks * 144];
        r.NextBytes(bytes);
        for (var i = 0; i < n * blocks; i++)
        {
            var o = i * 144;
            var d = BitConverter.HalfToUInt16Bits((System.Half)(0.002f + (float)r.NextDouble() * 0.004f));
            var dmin = BitConverter.HalfToUInt16Bits((System.Half)(0.001f + (float)r.NextDouble() * 0.003f));
            bytes[o] = (byte)d; bytes[o + 1] = (byte)(d >> 8);
            bytes[o + 2] = (byte)dmin; bytes[o + 3] = (byte)(dmin >> 8);
        }
        return bytes;
    }

    private static (float[] Out, double Ms) Run(Accelerator acc, bool fast, float[] a, byte[] w, int m, int k, int n, int repeats)
    {
        FusedDequantMatMul.EnableFastGemm = fast;
        var kernel = new FusedDequantMatMul(acc);
        using var ab = acc.Allocate1D(a);
        using var wb = acc.Allocate1D(w);
        using var ob = acc.Allocate1D<float>(m * n);
        kernel.Forward(ab.View, wb.View, ob.View, m, k, n, GGMLType.Q4_K);
        acc.Synchronize();
        var times = new List<double>();
        for (var i = 0; i < repeats; i++)
        {
            var sw = Stopwatch.StartNew();
            kernel.Forward(ab.View, wb.View, ob.View, m, k, n, GGMLType.Q4_K);
            acc.Synchronize();
            times.Add(sw.Elapsed.TotalMilliseconds);
        }
        times.Sort();
        return (ob.GetAsArray1D(), times[times.Count / 2]);
    }

    [Theory]
    [InlineData(64, 4096, 4096)]       // the smallest M the register-blocked kernels take
    [InlineData(165, 4096, 14336)]     // a suffix after a cached prompt: the FFN gate/up shape of llama3.1:8b
    [InlineData(165, 14336, 4096)]     // the FFN down shape
    [InlineData(300, 4096, 4096)]      // attention projections
    [InlineData(333, 4096, 1024)]      // a ragged M and a narrow N (the k/v projections)
    [InlineData(100, 2048, 200)]       // N not a multiple of the tile
    public void MatchesTheRegisterBlockedKernel_AndTheSpeedIsShown(int m, int k, int n)
    {
        var context = MLContext.CreateContext();
        var device = context.Devices.FirstOrDefault(d => d.AcceleratorType == AcceleratorType.Cuda);
        if (device == null) { output.WriteLine("no CUDA device: skipped"); context.Dispose(); return; }
        using var acc = device.CreateAccelerator(context);
        var rnd = new Random(7);
        var a = new float[m * k];
        for (var i = 0; i < a.Length; i++) a[i] = ((float)rnd.NextDouble() * 2f - 1f) * 0.5f;
        var w = RandomQ4K(n, k, 11);
        var reference = Run(acc, false, a, w, m, k, n, 5);
        var fast = Run(acc, true, a, w, m, k, n, 5);
        FusedDequantMatMul.EnableFastGemm = true;
        var scale = reference.Out.Max(MathF.Abs);
        var worst = 0f;
        for (var i = 0; i < reference.Out.Length; i++) worst = MathF.Max(worst, MathF.Abs(reference.Out[i] - fast.Out[i]));
        double flops = 2.0 * m * k * n;
        output.WriteLine($"M{m} K{k} N{n}: 64x64 {reference.Ms:0.00} ms ({flops / reference.Ms / 1e9:0.00} TFLOPS), 128x128 {fast.Ms:0.00} ms ({flops / fast.Ms / 1e9:0.00} TFLOPS), {reference.Ms / fast.Ms:0.0}x; max |diff| {worst:E2} (output scale {scale:0.00})");
        Assert.True(worst <= 1e-4f * Math.Max(scale, 1f), $"max |diff| {worst:E3} against output scale {scale:0.00}");
        context.Dispose();
    }
}

[CollectionDefinition("GemmFlag", DisableParallelization = true)]
public class GemmFlagCollection { }
