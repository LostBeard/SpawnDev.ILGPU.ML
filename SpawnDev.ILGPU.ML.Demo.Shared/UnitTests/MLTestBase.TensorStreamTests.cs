using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Tensors;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// GPU→Stream SAVE primitive (<see cref="Tensor.CopyToStreamAsync"/>, added 2026-07-01 on SpawnDev.ILGPU 4.17.0's
/// <c>ArrayView&lt;T&gt;.CopyToStreamAsync</c>). The save-side mirror of the streaming model-load path — lets a
/// large tensor / GPU buffer be exported to OPFS one bounded chunk at a time (browser: GPU→JS Uint8Array→stream,
/// no managed-heap copy), instead of reading the whole thing into a .NET array and OOMing (SpawnScene's need).
/// </summary>
public abstract partial class MLTestBase
{
    [TestMethod]
    public async Task Tensor_CopyToStreamAsync_RoundTrip() => await RunTest(async accelerator =>
    {
        // Known fp32 payload on the GPU.
        const int n = 5000;
        var data = new float[n];
        var rng = new Random(1234);
        for (int i = 0; i < n; i++) data[i] = (float)(rng.NextDouble() * 200 - 100);

        using var buf = accelerator.Allocate1D(data);
        var tensor = new Tensor(buf.View, new[] { n });

        // SAVE: stream the tensor's raw fp32 bytes out. Small 4 KiB chunk to exercise the multi-chunk loop
        // (20,000 bytes → ~5 chunks) without a 16 MiB allocation; 4-byte aligned for the WebGPU copy rule.
        using var ms = new System.IO.MemoryStream();
        await tensor.CopyToStreamAsync(ms, chunkSizeInBytes: 4096);
        await accelerator.SynchronizeAsync();

        var bytes = ms.ToArray();
        int expectedBytes = n * sizeof(float);
        if (bytes.Length < expectedBytes)
            throw new Exception($"CopyToStreamAsync wrote {bytes.Length} bytes, expected >= {expectedBytes}");

        // Verify the fp32 payload is byte-exact (allow trailing 4-byte-alignment padding on browser backends).
        var got = new float[n];
        Buffer.BlockCopy(bytes, 0, got, 0, expectedBytes);
        for (int i = 0; i < n; i++)
            if (got[i] != data[i])
                throw new Exception($"CopyToStreamAsync byte mismatch at [{i}]: got {got[i]}, expected {data[i]}");

        Console.WriteLine($"[TensorStream] CopyToStreamAsync round-trip byte-exact: {n} floats, {bytes.Length} bytes (~{(bytes.Length + 4095) / 4096} chunks)");
    });
}
