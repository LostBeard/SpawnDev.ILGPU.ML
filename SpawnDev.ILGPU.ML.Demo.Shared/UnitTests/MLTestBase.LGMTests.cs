using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Kernels;
using SpawnDev.ILGPU.ML.Formats;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// Tests for LGM (Large Gaussian Model) components:
/// GroupNorm with custom weights, SPZ/PLY round-trips, Gaussian validation.
/// </summary>
public abstract partial class MLTestBase
{
    [TestMethod]
    public async Task LGM_GroupNorm_CustomWeightBias() => await RunTest(async accelerator =>
    {
        // GroupNorm with weight=2, bias=1 should double and shift the normalized output
        int B = 1, C = 4, H = 2, W = 2, G = 2;
        int total = B * C * H * W;

        // All values constant (5.0) → within each group, mean=5, var=0
        // normalized = (5-5)/sqrt(0+eps) = 0, output = weight*0 + bias = bias = 1
        var input = new float[total];
        Array.Fill(input, 5f);

        var weight = new float[C];
        var bias = new float[C];
        Array.Fill(weight, 2f);
        Array.Fill(bias, 1f);

        using var inputBuf = accelerator.Allocate1D(input);
        using var outputBuf = accelerator.Allocate1D<float>(total);
        using var wBuf = accelerator.Allocate1D(weight);
        using var bBuf = accelerator.Allocate1D(bias);

        var gn = new GroupNormKernel(accelerator);
        gn.Forward(inputBuf.View, outputBuf.View, wBuf.View, bBuf.View, B, C, H * W, G);
        await accelerator.SynchronizeAsync();
        var output = await outputBuf.CopyToHostAsync<float>(0, total);

        // Within each group, constant values normalize to 0 (mean=value, var=0)
        // So output = weight * 0 + bias = bias = 1 for all elements
        for (int i = 0; i < total; i++)
        {
            if (MathF.Abs(output[i] - 1f) > 0.01f)
                throw new Exception($"GroupNorm custom: output[{i}]={output[i]:F4}, expected ~1.0 (const input → normalized to 0 → bias)");
        }

        Console.WriteLine("[LGM] GroupNorm custom weight=2 bias=1: constant input → all bias values");
    });

    [TestMethod]
    public async Task LGM_SPZ_PLY_RoundTrip()
    {
        await Task.CompletedTask;

        // Create cloud, export to SPZ, reimport, export to PLY, reimport
        var cloud = CreateTestGaussianCloud(20);

        // SPZ round-trip
        var spzBytes = SPZExporter.ExportSPZ(cloud);
        var spzCloud = SPZParser.Parse(spzBytes);

        if (spzCloud.NumPoints != 20)
            throw new Exception($"SPZ round-trip: {spzCloud.NumPoints} points, expected 20");

        // PLY round-trip from the SPZ-loaded cloud
        var plyBytes = SPZExporter.ExportPLY(spzCloud);
        var plyData = PLYParser.Parse(plyBytes);

        if (plyData.VertexCount != 20)
            throw new Exception($"SPZ→PLY round-trip: {plyData.VertexCount} vertices, expected 20");

        Console.WriteLine($"[LGM] SPZ→PLY round-trip: 20 Gaussians, SPZ={spzBytes.Length}B, PLY={plyBytes.Length}B");
    }

    [TestMethod]
    public async Task LGM_GaussianValidation_OpacityRange()
    {
        await Task.CompletedTask;

        // Generate cloud and verify all values are in valid ranges
        var cloud = CreateTestGaussianCloud(50);

        // Verify all opacities in [0, 1]
        for (int i = 0; i < cloud.NumPoints; i++)
        {
            if (cloud.Alphas[i] < 0 || cloud.Alphas[i] > 1)
                throw new Exception($"Alpha[{i}]={cloud.Alphas[i]:F4} outside [0,1]");
        }

        // Verify all colors in [0, 1]
        for (int i = 0; i < cloud.NumPoints * 3; i++)
        {
            if (cloud.Colors[i] < 0 || cloud.Colors[i] > 1)
                throw new Exception($"Color[{i}]={cloud.Colors[i]:F4} outside [0,1]");
        }

        // Verify quaternion rotations are unit length
        for (int i = 0; i < cloud.NumPoints; i++)
        {
            float w = cloud.Rotations[i * 4], x = cloud.Rotations[i * 4 + 1];
            float y = cloud.Rotations[i * 4 + 2], z = cloud.Rotations[i * 4 + 3];
            float len = MathF.Sqrt(w * w + x * x + y * y + z * z);
            if (MathF.Abs(len - 1f) > 0.01f)
                throw new Exception($"Rotation[{i}] length={len:F4}, expected ~1.0");
        }

        Console.WriteLine("[LGM] Gaussian validation: opacity [0,1], colors [0,1], rotations unit length");
    }

    private static GaussianCloud CreateTestGaussianCloud(int n)
    {
        var rng = new Random(42);
        var cloud = new GaussianCloud
        {
            NumPoints = n,
            Positions = new float[n * 3],
            Alphas = new float[n],
            Colors = new float[n * 3],
            Scales = new float[n * 3],
            Rotations = new float[n * 4],
        };
        for (int i = 0; i < n; i++)
        {
            cloud.Positions[i * 3] = (float)(rng.NextDouble() * 10 - 5);
            cloud.Positions[i * 3 + 1] = (float)(rng.NextDouble() * 10 - 5);
            cloud.Positions[i * 3 + 2] = (float)(rng.NextDouble() * 10 - 5);
            cloud.Alphas[i] = (float)(rng.NextDouble() * 0.8 + 0.1);
            cloud.Colors[i * 3] = (float)rng.NextDouble();
            cloud.Colors[i * 3 + 1] = (float)rng.NextDouble();
            cloud.Colors[i * 3 + 2] = (float)rng.NextDouble();
            cloud.Scales[i * 3] = -2f + (float)rng.NextDouble() * 2;
            cloud.Scales[i * 3 + 1] = -2f + (float)rng.NextDouble() * 2;
            cloud.Scales[i * 3 + 2] = -2f + (float)rng.NextDouble() * 2;
            cloud.Rotations[i * 4] = 1; cloud.Rotations[i * 4 + 1] = 0;
            cloud.Rotations[i * 4 + 2] = 0; cloud.Rotations[i * 4 + 3] = 0;
        }
        return cloud;
    }
}
