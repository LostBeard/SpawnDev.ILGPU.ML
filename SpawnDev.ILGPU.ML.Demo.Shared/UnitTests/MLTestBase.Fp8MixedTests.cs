using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.Trellis2;
using SpawnDev.UnitTesting;
using System.Text;
using System.Text.Json;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    /// <summary>
    /// ComfyUI's mixed-precision ("fp8mixed") safetensors layout, as qwen_image_edit_2511_fp8mixed stores it, built in memory (every backend, no download): an 8-bit matrix with
    /// <c>weight_scale</c> / <c>input_scale</c> (fp32 scalars), a U8 <c>comfy_quant</c> json blob per layer, <c>_quantization_metadata</c> in the header and the empty
    /// <c>__index_timestep_zero__</c> marker. <see cref="StreamedModelWeights"/> must take weight_scale as the matrix's scale (exact decode + activation scale, and the folded
    /// decode = f8 * scale), drop input_scale and the blobs, keep the marker in <see cref="StreamedModelWeights.HeaderNames"/>, and refuse a layer format it does not compute
    /// exactly (full_precision_matrix_mult false).
    /// </summary>
    [TestMethod]
    public async Task StreamedWeights_Fp8Mixed_ScalesBlobsAndMarker() => await RunTest(async accelerator =>
    {
        // e4m3fn bytes with exact values: 1, 2, -1, 1.5, 0.5, 0
        byte[] codes = { 0x38, 0x40, 0xB8, 0x3C, 0x30, 0x00 };
        float[] values = { 1f, 2f, -1f, 1.5f, 0.5f, 0f };
        var weight = Enumerable.Range(0, 32).Select(i => codes[i % codes.Length]).ToArray();     // [4, 8]
        const float Scale = 0.375f;                                                              // f8 * 0.375 is exact in bf16 for these values

        var source = new Fp8MixedSource(Fp8MixedFile(weight, Scale, fullPrecision: true));
        using (var w = await StreamedModelWeights.OpenAsync(accelerator, source, "", "m.safetensors", "blocks.0.lin.weight", n => n.StartsWith("blocks.0.", StringComparison.Ordinal) ? 0 : -1, 1, long.MaxValue))
        {
            Ur_Expect(w.HeaderNames.Contains("__index_timestep_zero__"), "the empty marker must be listed in HeaderNames");
            Ur_Expect(w.HasScales, "weight_scale must be read as the 8-bit matrix's scale");
            var (block, streamed) = await w.BlockAsync(0);
            Ur_Expect(!streamed, "one tiny block stays resident");
            Ur_Expect(block.IsF8("blocks.0.lin.weight"), "the matrix stays 8-bit");
            foreach (var gone in new[] { "blocks.0.lin.weight_scale", "blocks.0.lin.input_scale", "blocks.0.lin.comfy_quant" })
                Ur_Expect(!block.Has(gone), $"{gone} must not be kept as a weight");
            Ur_Expect(w.ActivationScale(block, "blocks.0.lin.weight") == Scale, $"activation scale {w.ActivationScale(block, "blocks.0.lin.weight")}, expected {Scale}");
            var exact = await w.MatrixExact(block, "blocks.0.lin.weight").CopyToHostAsync();
            var folded = await w.Matrix(block, "blocks.0.lin.weight").CopyToHostAsync();
            for (int i = 0; i < 32; i++)
            {
                float v = values[i % values.Length];
                Ur_Expect((float)exact[i] == v, $"exact decode [{i}] = {(float)exact[i]}, expected {v}");
                Ur_Expect((float)folded[i] == v * Scale, $"folded decode [{i}] = {(float)folded[i]}, expected {v * Scale}");
            }
        }

        // a layer whose matmul is not full precision (ComfyUI would quantize the activations too): refused, never run differently
        bool refused = false;
        try { using var bad = await StreamedModelWeights.OpenAsync(accelerator, new Fp8MixedSource(Fp8MixedFile(weight, Scale, fullPrecision: false)), "", "m.safetensors", "blocks.0.lin.weight", n => n.StartsWith("blocks.0.", StringComparison.Ordinal) ? 0 : -1, 1, long.MaxValue); }
        catch (NotSupportedException) { refused = true; }
        Ur_Expect(refused, "full_precision_matrix_mult false must be refused");
        return "scales, blobs, marker, exact + folded decode, refusal: as specified";
    });

    /// <summary>
    /// The read-ahead queue of <see cref="StreamedModelWeights"/> (<see cref="StreamedModelWeights.ReadAheadBytes"/>) and the host cache hand out the SAME blocks as a
    /// plain read, in every combination: a 6-block file built in memory (block 0 resident, 1..5 streamed), three forward passes each, then blocks asked out of order
    /// (the queue must drop what it prepared and still return the right block). Every block's tensor is read back and compared with what the file holds.
    /// </summary>
    [TestMethod]
    public async Task StreamedWeights_ReadAhead_HandsOutTheSameBlocks() => await RunTest(async accelerator =>
    {
        const int Blocks = 6, N = 64;
        static float Value(int k, int i) => k * 8 + (i % 8) + 1;                                   // exact in bf16, different per block
        var blobs = new List<(string Name, byte[] Data)>();
        for (int k = 0; k < Blocks; k++)
        {
            var d = new byte[2 * N];
            for (int i = 0; i < N; i++) BitConverter.TryWriteBytes(d.AsSpan(2 * i, 2), (ushort)(BitConverter.SingleToUInt32Bits(Value(k, i)) >> 16));
            blobs.Add(($"blocks.{k}.w", d));
        }
        blobs.Add(("head.bias", new byte[] { 0x80, 0x3F, 0x80, 0x3F }));
        var header = new Dictionary<string, object>(); long off = 0;
        foreach (var b in blobs) { header[b.Name] = new { dtype = "BF16", shape = new[] { b.Data.Length / 2 }, data_offsets = new[] { off, off + b.Data.Length } }; off += b.Data.Length; }
        var json = Encoding.UTF8.GetBytes(JsonSerializer.Serialize(header));
        int pad = (8 - json.Length % 8) % 8;
        using var ms = new MemoryStream();
        ms.Write(BitConverter.GetBytes((long)(json.Length + pad))); ms.Write(json); ms.Write(Enumerable.Repeat((byte)' ', pad).ToArray());
        foreach (var b in blobs) ms.Write(b.Data);
        var file = ms.ToArray();

        var report = new List<string>();
        // (label, prefetch, read-ahead bytes, host cache bytes): a block's image is its 128 bytes plus a small header
        foreach (var (label, prefetch, ahead, cache) in new[] { ("no prefetch", false, 0L, 0L), ("one block ahead", true, 0L, 0L), ("everything ahead", true, 1L << 20, 0L),
            ("about two blocks ahead", true, 700L, 0L), ("everything ahead + two blocks cached", true, 1L << 20, 700L), ("one ahead + all cached", true, 0L, 1L << 20) })
        {
            using var w = await StreamedModelWeights.OpenAsync(accelerator, new Fp8MixedSource(file), "", "m.safetensors", "blocks.0.w",
                n => n.StartsWith("blocks.", StringComparison.Ordinal) ? int.Parse(n.Split('.')[1]) : -1, Blocks, 2 * N);
            w.Prefetch = prefetch; w.ReadAheadBytes = ahead; w.HostCacheBytes = cache;
            Ur_Expect(w.ResidentBlocks == 1, $"{label}: {w.ResidentBlocks} resident blocks, built for 1");
            int loads = 0;
            async Task Check(int k, string when)
            {
                var (block, streamed) = await w.BlockAsync(k);
                var got = await block.BF16($"blocks.{k}.w").CopyToHostAsync();
                await accelerator.SynchronizeAsync();
                if (streamed) { block.Dispose(); loads++; }
                Ur_Expect(streamed == (k != 0), $"{label}, {when}: block {k} streamed = {streamed}");
                for (int i = 0; i < N; i++) Ur_Expect((float)got[i] == Value(k, i), $"{label}, {when}: block {k}[{i}] = {(float)got[i]}, the file holds {Value(k, i)}");
            }
            for (int pass = 0; pass < 3; pass++) for (int k = 0; k < Blocks; k++) await Check(k, $"pass {pass}");
            foreach (int k in new[] { 3, 1, 5, 2, 2, 4 }) await Check(k, "out of order");
            for (int k = 0; k < Blocks; k++) await Check(k, "back in order");
            Ur_Expect(w.HostCacheUsed <= cache, $"{label}: the host cache holds {w.HostCacheUsed} bytes of a {cache} budget");
            if (cache >= (1L << 20)) Ur_Expect(w.HostCacheUsed > 5 * 2 * N, $"{label}: only {w.HostCacheUsed} bytes cached, five streamed blocks expected");
            report.Add($"{label}: {loads} streamed loads right, cache {w.HostCacheUsed} bytes");
        }
        return string.Join("; ", report);
    });

    private static byte[] Fp8MixedFile(byte[] weight, float scale, bool fullPrecision)
    {
        string quant = JsonSerializer.Serialize(new { format = "float8_e4m3fn", full_precision_matrix_mult = fullPrecision });
        var blobs = new List<(string Name, string Dtype, int[] Shape, byte[] Data)>
        {
            ("__index_timestep_zero__", "F32", new[] { 0 }, Array.Empty<byte>()),
            ("blocks.0.lin.weight", "F8_E4M3", new[] { 4, 8 }, weight),
            ("blocks.0.lin.weight_scale", "F32", Array.Empty<int>(), BitConverter.GetBytes(scale)),
            ("blocks.0.lin.input_scale", "F32", Array.Empty<int>(), BitConverter.GetBytes(1f)),
            ("blocks.0.lin.comfy_quant", "U8", new[] { Encoding.UTF8.GetByteCount(quant) }, Encoding.UTF8.GetBytes(quant)),
            ("blocks.0.lin.bias", "BF16", new[] { 4 }, new byte[] { 0x80, 0x3F, 0x80, 0x3F, 0x80, 0x3F, 0x80, 0x3F }),
            ("head.bias", "BF16", new[] { 4 }, new byte[] { 0x80, 0x3F, 0x80, 0x3F, 0x80, 0x3F, 0x80, 0x3F }),
        };
        var header = new Dictionary<string, object>
        {
            ["__metadata__"] = new Dictionary<string, string>
            {
                ["_quantization_metadata"] = JsonSerializer.Serialize(new { format_version = "1.0", layers = new Dictionary<string, object> { ["blocks.0.lin"] = new { format = "float8_e4m3fn", full_precision_matrix_mult = fullPrecision } } }),
            },
        };
        long off = 0;
        foreach (var b in blobs) { header[b.Name] = new { dtype = b.Dtype, shape = b.Shape, data_offsets = new[] { off, off + b.Data.Length } }; off += b.Data.Length; }
        var json = Encoding.UTF8.GetBytes(JsonSerializer.Serialize(header));
        int pad = (8 - json.Length % 8) % 8;
        using var ms = new MemoryStream();
        ms.Write(BitConverter.GetBytes((long)(json.Length + pad)));
        ms.Write(json); ms.Write(Enumerable.Repeat((byte)' ', pad).ToArray());
        foreach (var b in blobs) ms.Write(b.Data);
        return ms.ToArray();
    }

    private sealed class Fp8MixedSource(byte[] file) : IModelSource
    {
        public Task<Stream> OpenAsync(string repoId, string filePath, CancellationToken cancellationToken = default) => Task.FromResult<Stream>(new MemoryStream(file, writable: false));
        public Task<byte[]> FetchBytesAsync(string repoId, string filePath, CancellationToken cancellationToken = default) => Task.FromResult(file);
    }
}
