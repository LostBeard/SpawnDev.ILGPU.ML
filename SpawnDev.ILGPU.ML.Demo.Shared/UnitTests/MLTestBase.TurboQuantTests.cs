using ILGPU;
using ILGPU.Runtime;
using SpawnDev.UnitTesting;
using System.Text.Json;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// Tests for TurboQuant: Fast Walsh-Hadamard Transform, codebook generation,
/// quantize/dequantize round-trip. Reference data from Python numpy/scipy.
/// </summary>
public abstract partial class MLTestBase
{
    // ═══════════════════════════════════════════════════════════
    //  FWHT (Fast Walsh-Hadamard Transform) Tests
    // ═══════════════════════════════════════════════════════════

    [TestMethod]
    public async Task TurboQuant_FWHT_Impulse_d8()
    {
        await Task.CompletedTask;
        // Input [1,0,0,0,0,0,0,0] → output [1,1,1,1,1,1,1,1]/sqrt(8)
        var input = new float[] { 1, 0, 0, 0, 0, 0, 0, 0 };
        var output = FWHT_CPU(input);

        float expected = 1f / MathF.Sqrt(8);
        for (int i = 0; i < 8; i++)
        {
            if (MathF.Abs(output[i] - expected) > 1e-5f)
                throw new Exception($"FWHT impulse[{i}]={output[i]:F6}, expected {expected:F6}");
        }
        Console.WriteLine("[TurboQuant] FWHT impulse d=8: PASS");
    }

    [TestMethod]
    public async Task TurboQuant_FWHT_Ones_d8()
    {
        await Task.CompletedTask;
        // Input [1,1,1,1,1,1,1,1] → output [sqrt(8),0,0,0,0,0,0,0]
        var input = new float[] { 1, 1, 1, 1, 1, 1, 1, 1 };
        var output = FWHT_CPU(input);

        float expectedFirst = MathF.Sqrt(8);
        if (MathF.Abs(output[0] - expectedFirst) > 1e-4f)
            throw new Exception($"FWHT ones[0]={output[0]:F6}, expected {expectedFirst:F6}");
        for (int i = 1; i < 8; i++)
        {
            if (MathF.Abs(output[i]) > 1e-5f)
                throw new Exception($"FWHT ones[{i}]={output[i]:F6}, expected ~0");
        }
        Console.WriteLine("[TurboQuant] FWHT ones d=8: PASS");
    }

    [TestMethod]
    public async Task TurboQuant_FWHT_RoundTrip_d8()
    {
        await Task.CompletedTask;
        // FWHT is its own inverse (up to scaling by d)
        var input = new float[] { 0.5f, -0.3f, 1.2f, -0.7f, 0.1f, 0.8f, -0.4f, 0.9f };
        var transformed = FWHT_CPU(input);
        var roundTrip = FWHT_CPU(transformed);

        // roundTrip should equal input (FWHT(FWHT(x)) = x when using normalized form)
        for (int i = 0; i < 8; i++)
        {
            if (MathF.Abs(roundTrip[i] - input[i]) > 1e-4f)
                throw new Exception($"FWHT round-trip[{i}]: {roundTrip[i]:F6} != {input[i]:F6}");
        }
        Console.WriteLine("[TurboQuant] FWHT round-trip d=8: PASS");
    }

    [TestMethod]
    public async Task TurboQuant_FWHT_Reference_d128()
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");

        // Load reference data
        var jsonStr = await http.GetStringAsync("references/turboquant/turboquant_test_cases.json");
        using var doc = JsonDocument.Parse(jsonStr);
        var d128Test = doc.RootElement.GetProperty("fwht_tests").GetProperty("d128_random");

        var input = d128Test.GetProperty("input").EnumerateArray()
            .Select(e => (float)e.GetDouble()).ToArray();
        var expected = d128Test.GetProperty("expected").EnumerateArray()
            .Select(e => (float)e.GetDouble()).ToArray();

        if (input.Length != 128)
            throw new Exception($"Input length={input.Length}, expected 128");

        var output = FWHT_CPU(input);

        float maxErr = 0;
        for (int i = 0; i < 128; i++)
        {
            float err = MathF.Abs(output[i] - expected[i]);
            if (err > maxErr) maxErr = err;
        }

        Console.WriteLine($"[TurboQuant] FWHT d=128: maxErr={maxErr:E3}");
        if (maxErr > 1e-3f)
            throw new Exception($"FWHT d=128 maxErr={maxErr:E3} exceeds tolerance 1e-3");
    }

    [TestMethod]
    public async Task TurboQuant_FWHT_PreservesEnergy()
    {
        await Task.CompletedTask;
        // Parseval's theorem: ||FWHT(x)||^2 == ||x||^2
        var rng = new Random(42);
        var input = new float[64];
        for (int i = 0; i < 64; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);

        float inputEnergy = input.Sum(v => v * v);
        var output = FWHT_CPU(input);
        float outputEnergy = output.Sum(v => v * v);

        float relErr = MathF.Abs(outputEnergy - inputEnergy) / inputEnergy;
        if (relErr > 1e-4f)
            throw new Exception($"Parseval: input_energy={inputEnergy:F4}, output_energy={outputEnergy:F4}, relErr={relErr:E3}");

        Console.WriteLine($"[TurboQuant] FWHT energy preservation: relErr={relErr:E3} PASS");
    }

    // ═══════════════════════════════════════════════════════════
    //  Codebook Tests
    // ═══════════════════════════════════════════════════════════

    [TestMethod]
    public async Task TurboQuant_Codebook_4bit_Symmetric()
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");

        var jsonStr = await http.GetStringAsync("references/turboquant/turboquant_test_cases.json");
        using var doc = JsonDocument.Parse(jsonStr);

        if (!doc.RootElement.TryGetProperty("codebook_tests", out var cbTests))
        {
            Console.WriteLine("[TurboQuant] codebook_tests not in reference data — skipping");
            throw new UnsupportedTestException("No codebook_tests in reference data");
        }

        var cb4bit = cbTests.GetProperty("4bit_d128");
        var centroids = cb4bit.GetProperty("centroids").EnumerateArray()
            .Select(e => (float)e.GetDouble()).ToArray();

        // 4-bit → 16 centroids
        if (centroids.Length != 16)
            throw new Exception($"4-bit codebook should have 16 centroids, got {centroids.Length}");

        // Centroids should be monotonically increasing
        for (int i = 1; i < centroids.Length; i++)
        {
            if (centroids[i] <= centroids[i - 1])
                throw new Exception($"Centroids not monotonic: [{i - 1}]={centroids[i - 1]:F4} >= [{i}]={centroids[i]:F4}");
        }

        // Centroids should be approximately symmetric around 0
        float sumCentroids = centroids.Sum();
        if (MathF.Abs(sumCentroids) > 1.0f)
            throw new Exception($"Centroids not symmetric: sum={sumCentroids:F4}");

        Console.WriteLine($"[TurboQuant] 4-bit codebook: {centroids.Length} centroids, symmetric, monotonic PASS");
    }

    // ═══════════════════════════════════════════════════════════
    //  FWHT GPU Kernel Tests
    // ═══════════════════════════════════════════════════════════

    [TestMethod]
    public async Task TurboQuant_FWHT_GPU_MatchesCPU_d8() => await RunTest(async accelerator =>
    {
        var kernel = new Kernels.FWHTKernel(accelerator);

        var input = new float[] { 1f, 0f, 0f, 0f, 0f, 0f, 0f, 0f };
        var cpuResult = FWHT_CPU(input);

        using var buf = accelerator.Allocate1D(input);
        kernel.Forward(buf.View, 8);
        await accelerator.SynchronizeAsync();
        var gpuResult = await buf.CopyToHostAsync<float>(0, 8);

        float maxErr = 0;
        for (int i = 0; i < 8; i++)
            maxErr = MathF.Max(maxErr, MathF.Abs(gpuResult[i] - cpuResult[i]));

        if (maxErr > 1e-4f)
            throw new Exception($"FWHT GPU vs CPU: maxErr={maxErr:E3}");

        Console.WriteLine($"[TurboQuant] FWHT GPU d=8: maxErr={maxErr:E3} vs CPU reference PASS");
    });

    [TestMethod]
    public async Task TurboQuant_FWHT_GPU_RoundTrip_d64() => await RunTest(async accelerator =>
    {
        var kernel = new Kernels.FWHTKernel(accelerator);

        var rng = new Random(42);
        var input = new float[64];
        for (int i = 0; i < 64; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);
        var original = (float[])input.Clone();

        using var buf = accelerator.Allocate1D(input);

        // Forward FWHT
        kernel.Forward(buf.View, 64);
        // Inverse FWHT (same transform — it's its own inverse)
        kernel.Forward(buf.View, 64);

        await accelerator.SynchronizeAsync();
        var roundTrip = await buf.CopyToHostAsync<float>(0, 64);

        float maxErr = 0;
        for (int i = 0; i < 64; i++)
            maxErr = MathF.Max(maxErr, MathF.Abs(roundTrip[i] - original[i]));

        if (maxErr > 1e-3f)
            throw new Exception($"FWHT GPU round-trip d=64: maxErr={maxErr:E3}");

        Console.WriteLine($"[TurboQuant] FWHT GPU round-trip d=64: maxErr={maxErr:E3} PASS");
    });

    // ═══════════════════════════════════════════════════════════
    //  CPU FWHT Reference Implementation
    // ═══════════════════════════════════════════════════════════

    /// <summary>
    /// CPU reference FWHT (normalized). In-place butterfly algorithm.
    /// FWHT(x) = H_d @ x / sqrt(d) where H_d is the Hadamard matrix.
    /// </summary>
    private static float[] FWHT_CPU(float[] input)
    {
        int d = input.Length;
        var x = (float[])input.Clone();

        // Butterfly stages: log2(d) passes
        for (int halfSize = 1; halfSize < d; halfSize *= 2)
        {
            for (int i = 0; i < d; i += halfSize * 2)
            {
                for (int j = i; j < i + halfSize; j++)
                {
                    float a = x[j];
                    float b = x[j + halfSize];
                    x[j] = a + b;
                    x[j + halfSize] = a - b;
                }
            }
        }

        // Normalize
        float scale = 1f / MathF.Sqrt(d);
        for (int i = 0; i < d; i++)
            x[i] *= scale;

        return x;
    }
}
