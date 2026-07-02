using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Kernels;
using SpawnDev.UnitTesting;
using System.Diagnostics;
using System.Text;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// Vec4LoadMatMul (128-bit F4/AsAligned16 GEMM loads) - correctness vs CPU reference and vs the
/// production scalar RegisterBlockedMatMul, plus a THREE-WAY throughput A/B at real DAv3 shapes:
///   scalar-float GEMM  vs  F4-struct-load GEMM (no AsAligned16)  vs  F4-AsAligned16 GEMM.
/// The middle variant isolates struct-packing cost from the 128-bit-load win: it stays a scalar
/// struct load on every backend even after the WGSL AsAligned16-trigger lands in SpawnDev.ILGPU,
/// while the third becomes a single vec4 load - so one post-trigger re-run of the SAME binary
/// attributes any delta to load width alone. The bench RETURNS its numbers (ResultText -> the
/// PMT results JSON) so measurements persist per backend per run.
///
/// WebGL is gated out: struct-of-4 element loads emit invalid GLSL (tracked:
/// geordi-webgl-struct-of-4-load-glsl-bug-tracked-2026-07-01) and WebGL has no 128-bit loads.
/// DAv3-shape tests run on the fast lanes (CUDA/OpenCL/WebGPU) per the backend-priority rule;
/// CPU/Wasm cover the full-CPU-reference correctness test.
/// </summary>
public abstract partial class MLTestBase
{
    private static Vec4LoadMatMul.F4[] PackF4(float[] src)
    {
        var packed = new Vec4LoadMatMul.F4[src.Length / 4];
        for (int i = 0; i < packed.Length; i++)
            packed[i] = new Vec4LoadMatMul.F4 { A = src[4 * i], B = src[4 * i + 1], C = src[4 * i + 2], D = src[4 * i + 3] };
        return packed;
    }

    private static void GateGemmVec4Capability(Accelerator accelerator)
    {
        if (accelerator.AcceleratorType == AcceleratorType.WebGL)
            throw new UnsupportedTestException("WebGL: struct-of-4 GLSL load bug (tracked geordi-webgl-struct-of-4-load-glsl-bug-tracked-2026-07-01); WebGL has no 128-bit loads");
        if (accelerator.MaxNumThreadsPerGroup < 256)
            throw new UnsupportedTestException($"Device max group size {accelerator.MaxNumThreadsPerGroup} < 256 required by the 64x64 tile kernel (ILGPU CPU accelerator caps at 64)");
    }

    private static void GateGemmVec4FastLane(Accelerator accelerator)
    {
        GateGemmVec4Capability(accelerator);
        if (accelerator.AcceleratorType is not (AcceleratorType.Cuda or AcceleratorType.OpenCL or AcceleratorType.WebGPU))
            throw new UnsupportedTestException($"{accelerator.AcceleratorType}: DAv3-shape lane runs on the fast backends (CUDA/OpenCL/WebGPU); CPU-reference coverage is GemmVec4_AlignedShapes_MatchCPU");
    }

    /// <summary>RunTest wrapper whose body returns a string; the runner stores the returned string
    /// as the test's ResultText, which lands in the PMT results JSON (measurement persistence).</summary>
    protected async Task<string> RunTestWithResult(Func<Accelerator, Task<string>> testBody)
    {
        string result = "";
        await RunTest(async accelerator => { result = await testBody(accelerator); });
        return result;
    }

    [TestMethod]
    public async Task GemmVec4_AlignedShapes_MatchCPU() => await RunTest(async accelerator =>
    {
        GateGemmVec4Capability(accelerator);

        var mm = new Vec4LoadMatMul(accelerator);
        // Minimum single tile, square multi-tile, and a non-square multi-K-tile shape.
        foreach (var (M, K, N) in new[] { (64, 16, 64), (128, 128, 128), (192, 96, 128) })
        {
            var A = RandomFloats(M * K, seed: 42);
            var B = RandomFloats(K * N, seed: 123);
            var expected = CpuMatMul(A, B, M, K, N);

            using var aBuf = accelerator.Allocate1D(PackF4(A));
            using var bBuf = accelerator.Allocate1D(PackF4(B));
            using var cBuf = accelerator.Allocate1D<float>(M * N);

            mm.MatMul(aBuf.View, bBuf.View, cBuf.View, M, K, N);
            await accelerator.SynchronizeAsync();
            var gpuC = await cBuf.CopyToHostAsync<float>(0, M * N);
            AssertClose(expected, gpuC, 0.05f, $"GemmVec4 vec4 {M}x{K}x{N}: ");

            mm.MatMulStructLoad(aBuf.View, bBuf.View, cBuf.View, M, K, N);
            await accelerator.SynchronizeAsync();
            var gpuC2 = await cBuf.CopyToHostAsync<float>(0, M * N);
            AssertClose(expected, gpuC2, 0.05f, $"GemmVec4 struct-load {M}x{K}x{N}: ");

            Console.WriteLine($"[GemmVec4] {M}x{K}x{N} vec4 + struct-load vs CPU reference - correct");
        }
    });

    [TestMethod]
    public async Task GemmVec4_DAv3Shapes_MatchScalarGemm() => await RunTest(async accelerator =>
    {
        GateGemmVec4FastLane(accelerator);

        var vec4 = new Vec4LoadMatMul(accelerator);
        var scalar = new RegisterBlockedMatMul(accelerator);
        // DAv3-Small production GEMM shapes (patch tokens M=1344, width 384): QKV, MLP fc1, MLP fc2.
        foreach (var (label, M, K, N) in new[] { ("qkv", 1344, 384, 1152), ("fc1", 1344, 384, 1536), ("fc2", 1344, 1536, 384) })
        {
            var A = RandomFloats(M * K, seed: 42);
            var B = RandomFloats(K * N, seed: 123);

            using var aBuf = accelerator.Allocate1D(A);
            using var bBuf = accelerator.Allocate1D(B);
            using var a4Buf = accelerator.Allocate1D(PackF4(A));
            using var b4Buf = accelerator.Allocate1D(PackF4(B));
            using var cScalar = accelerator.Allocate1D<float>(M * N);
            using var cOther = accelerator.Allocate1D<float>(M * N);

            scalar.MatMul(aBuf.View, bBuf.View, cScalar.View, M, K, N);
            await accelerator.SynchronizeAsync();
            var ew = GetOrCreateEW(accelerator);

            // Identical tile structure and MAC order in all three kernels -> float-noise agreement.
            // GPU-side full-output compares; only 2 floats read back per check.
            vec4.MatMul(a4Buf.View, b4Buf.View, cOther.View, M, K, N);
            await accelerator.SynchronizeAsync();
            var (_, maxErrV) = await ew.CompareOnGpuAsync(cOther.View, cScalar.View, M * N);
            if (maxErrV > 1e-5f)
                throw new Exception($"GemmVec4 {label} {M}x{K}x{N}: vec4 vs scalar maxErr={maxErrV:E3} (identical MAC order should be bit-level)");

            vec4.MatMulStructLoad(a4Buf.View, b4Buf.View, cOther.View, M, K, N);
            await accelerator.SynchronizeAsync();
            var (_, maxErrS) = await ew.CompareOnGpuAsync(cOther.View, cScalar.View, M * N);
            if (maxErrS > 1e-5f)
                throw new Exception($"GemmVec4 {label} {M}x{K}x{N}: struct-load vs scalar maxErr={maxErrS:E3}");

            Console.WriteLine($"[GemmVec4] {label} {M}x{K}x{N} vec4 maxErr={maxErrV:E3}, struct-load maxErr={maxErrS:E3} vs RegisterBlockedMatMul - correct");
        }
    });

    // 5-min timeout: on WebGPU every warmup sync is a ~345ms mapAsync round-trip - the 9 Time()
    // calls (3 shapes x 3 kernels) blow the 30s NUnit default long before the kernels are the cost.
    [TestMethod(Timeout = 300000)]
    public async Task<string> GemmVec4_Bench_DAv3Shapes() => await RunTestWithResult(async accelerator =>
    {
        GateGemmVec4FastLane(accelerator);

        var vec4 = new Vec4LoadMatMul(accelerator);
        var scalar = new RegisterBlockedMatMul(accelerator);
        var report = new StringBuilder();
        report.Append($"{accelerator.AcceleratorType} {accelerator.Name}");
        foreach (var (label, M, K, N) in new[] { ("qkv", 1344, 384, 1152), ("fc1", 1344, 384, 1536), ("fc2", 1344, 1536, 384) })
        {
            var A = RandomFloats(M * K, seed: 42);
            var B = RandomFloats(K * N, seed: 123);

            using var aBuf = accelerator.Allocate1D(A);
            using var bBuf = accelerator.Allocate1D(B);
            using var a4Buf = accelerator.Allocate1D(PackF4(A));
            using var b4Buf = accelerator.Allocate1D(PackF4(B));
            using var cScalar = accelerator.Allocate1D<float>(M * N);
            using var cStruct = accelerator.Allocate1D<float>(M * N);
            using var cVec4 = accelerator.Allocate1D<float>(M * N);

            async Task<double> Time(Action run)
            {
                for (int w = 0; w < 3; w++) { run(); await accelerator.SynchronizeAsync(); }
                const int runs = 10;
                var sw = Stopwatch.StartNew();
                for (int r = 0; r < runs; r++) run();
                await accelerator.SynchronizeAsync();
                sw.Stop();
                return sw.Elapsed.TotalMilliseconds / runs;
            }

            double msScalar = await Time(() => scalar.MatMul(aBuf.View, bBuf.View, cScalar.View, M, K, N));
            double msStruct = await Time(() => vec4.MatMulStructLoad(a4Buf.View, b4Buf.View, cStruct.View, M, K, N));
            double msVec4 = await Time(() => vec4.MatMul(a4Buf.View, b4Buf.View, cVec4.View, M, K, N));
            double gf = 2.0 * M * K * N * 1e-6;
            var line = $"{label} M={M} K={K} N={N}: scalar {msScalar:F3}ms {gf / msScalar:F0}GF | struct {msStruct:F3}ms {gf / msStruct:F0}GF | vec4 {msVec4:F3}ms {gf / msVec4:F0}GF | vec4/scalar {msScalar / msVec4:F2}x";
            Console.WriteLine($"[GemmVec4] BENCH [{accelerator.AcceleratorType}] {line}");
            report.Append(" || ").Append(line);
        }
        return report.ToString();
    });
}
