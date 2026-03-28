using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    /// <summary>
    /// Diagnostic: dispatch N trivial Scale kernels on a small buffer,
    /// then CopyToHostAsync. Tests if MapAsync hangs based on dispatch count.
    /// Safe test — tiny buffer, trivial kernels, short timeout.
    /// </summary>
    [TestMethod(Timeout = 15000)]
    public async Task Diagnostic_MapAsync_After_ManyDispatches() => await RunTest(async accelerator =>
    {
        var ew = new ElementWiseKernels(accelerator);
        // Use LARGE buffers (1.6M floats = 6.4 MB) to match style transfer Conv output size
        int bigSize = 1605632; // [1,32,224,224]
        using var bufA = accelerator.Allocate1D<float>(bigSize);
        using var bufB = accelerator.Allocate1D<float>(bigSize);

        // Dispatch Scale kernels on large buffers
        for (int i = 0; i < 4; i++)
        {
            if (i % 2 == 0)
                ew.Scale(bufA.View, bufB.View, bigSize, 1.0f);
            else
                ew.Scale(bufB.View, bufA.View, bigSize, 1.0f);
            // Flush every 16 to match the inference pattern
            if (i > 0 && i % 16 == 0)
                accelerator.Synchronize();
        }
        accelerator.Synchronize(); // Final flush

        Console.WriteLine($"[DiagMapAsync] 4 dispatches on {bigSize}-element buffers done, attempting CopyToHostAsync...");

        // Now try CopyToHostAsync on a SMALL fresh buffer
        var tinyData = new float[] { 1f, 2f, 3f, 4f };
        using var tinyBuf = accelerator.Allocate1D(tinyData);
        accelerator.Synchronize();
        var result = await tinyBuf.CopyToHostAsync<float>(0, 4);
        Console.WriteLine($"[DiagMapAsync] CopyToHostAsync succeeded: {result[0]}, {result[1]}, {result[2]}, {result[3]}");

        Console.WriteLine("[DiagMapAsync] PASS — large buffer dispatches + CopyToHostAsync works");

        // Test Conv2D specifically — does it poison MapAsync?
        Console.WriteLine("[DiagMapAsync] Testing Conv2D dispatch...");
        var conv = new Conv2DKernel(accelerator);
        int inC = 3, inH = 232, inW = 232, outC = 32, kH = 9, kW = 9, pad = 4;
        int outH2 = (inH + 2 * pad - kH) + 1; // 224
        int outW2 = (inW + 2 * pad - kW) + 1; // 224
        using var convIn = accelerator.Allocate1D<float>(inC * inH * inW);
        using var convWeight = accelerator.Allocate1D<float>(outC * inC * kH * kW);
        using var convBias = accelerator.Allocate1D<float>(outC);
        using var convOut = accelerator.Allocate1D<float>(outC * outH2 * outW2);
        conv.Forward(convIn.View, convWeight.View, convBias.View, convOut.View,
            inC, inH, inW, outC, kH, kW, 1, pad);
        accelerator.Synchronize();
        Console.WriteLine("[DiagMapAsync] Conv2D dispatched, testing MapAsync...");
        var convTestData = new float[] { 99f, 98f, 97f, 96f };
        using var convTestBuf = accelerator.Allocate1D(convTestData);
        accelerator.Synchronize();
        var convResult = await convTestBuf.CopyToHostAsync<float>(0, 4);
        Console.WriteLine($"[DiagMapAsync] Conv2D + MapAsync: {convResult[0]}, {convResult[1]}, {convResult[2]}, {convResult[3]}");

        // Test Pad(reflect) → Conv → MapAsync (exactly what style transfer does)
        Console.WriteLine("[DiagMapAsync] Testing Pad(reflect) → Conv → MapAsync...");
        int padInC = 3, padInH = 224, padInW = 224;
        int[] padPads = new int[] { 0, 0, 4, 4, 0, 0, 4, 4 }; // reflect pad
        int padOutH = padInH + 4 + 4; // 232
        int padOutW = padInW + 4 + 4; // 232
        int[] padInShape = { 1, padInC, padInH, padInW };
        using var padInput = accelerator.Allocate1D<float>(padInC * padInH * padInW);
        using var padOutput = accelerator.Allocate1D<float>(padInC * padOutH * padOutW);
        var padKernel = new Kernels.PadKernel(accelerator);
        padKernel.Forward(padInput.View, padOutput.View, padInShape, padPads, 2, 0f); // mode=2=reflect
        accelerator.Synchronize();
        Console.WriteLine("[DiagMapAsync] Pad done, running Conv on padded output...");
        conv.Forward(padOutput.View, convWeight.View, convBias.View, convOut.View,
            padInC, padOutH, padOutW, outC, kH, kW, 1, 0); // no padding in conv, pad already applied
        accelerator.Synchronize();
        Console.WriteLine("[DiagMapAsync] Pad→Conv done, testing MapAsync...");
        using var padConvTestBuf = accelerator.Allocate1D(new float[] { 77f, 78f, 79f, 80f });
        accelerator.Synchronize();
        var padConvResult = await padConvTestBuf.CopyToHostAsync<float>(0, 4);
        Console.WriteLine($"[DiagMapAsync] Pad→Conv + MapAsync: {padConvResult[0]}, {padConvResult[1]}");
    });

    /// <summary>
    /// Diagnostic: run actual SqueezeNet inference (66 nodes) then immediately
    /// dispatch 60 more Scale kernels to reach 126 total dispatches.
    /// Tests if MapAsync hangs with real inference + extra dispatches.
    /// </summary>
    [TestMethod(Timeout = 30000)]
    public async Task Diagnostic_MapAsync_After_SqueezeNet_PlusExtra() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient();
        if (http == null)
            throw new UnsupportedTestException("HttpClient not available");

        // Run SqueezeNet (66 nodes) — this works
        var session = await InferenceSession.CreateAsync(accelerator, http, "models/squeezenet");

        int w = 32; int h = 32;
        var pixels = new int[(int)(w * h)];
        for (int i = 0; i < pixels.Length; i++)
            pixels[i] = unchecked((int)0xFF808080);

        var pipeline = new ClassificationPipeline(session, accelerator);
        var results = await pipeline.ClassifyAsync(pixels, w, h, 5);
        Console.WriteLine($"[DiagMapAsync2] SqueezeNet done: {results[0].Label} ({results[0].Confidence:P1})");

        // Now dispatch 60 more trivial kernels
        var ew = new ElementWiseKernels(accelerator);
        using var bufA = accelerator.Allocate1D<float>(1024);
        using var bufB = accelerator.Allocate1D<float>(1024);
        for (int i = 0; i < 60; i++)
        {
            if (i % 2 == 0) ew.Scale(bufA.View, bufB.View, 1024, 1.0f);
            else ew.Scale(bufB.View, bufA.View, 1024, 1.0f);
            if (i % 16 == 0) accelerator.Synchronize();
        }
        accelerator.Synchronize();

        // Try CopyToHostAsync after 66 + 60 = 126 total dispatches
        Console.WriteLine("[DiagMapAsync2] 126 total dispatches, attempting CopyToHostAsync...");
        var result = await bufA.CopyToHostAsync<float>(0, 10);
        Console.WriteLine($"[DiagMapAsync2] PASS — CopyToHostAsync after 126 dispatches works");

        pipeline.Dispose();
    });
}
