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
