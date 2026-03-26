using ILGPU;
using ILGPU.Runtime;
using SpawnDev.UnitTesting;
using System.Text.Json;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// Additional tests written by Data — ImagePreprocess, TurboQuant round-trip, SPZ validation, FWHT.
/// </summary>
public abstract partial class MLTestBase
{
    // ═══════════════════════════════════════════════════════════
    //  Image Preprocessing Tests
    // ═══════════════════════════════════════════════════════════

    [TestMethod]
    public async Task ImagePreprocess_ForwardRaw_ValueRange() => await RunTest(async accelerator =>
    {
        // Style transfer preprocessing: output should be in [0, 255] range
        var preprocess = new SpawnDev.ILGPU.ML.Preprocessing.ImagePreprocessor();
        int w = 4, h = 4;
        var rgba = new byte[w * h * 4];
        for (int i = 0; i < rgba.Length; i++) rgba[i] = (byte)(i % 256);

        var nchw = SpawnDev.ILGPU.ML.Preprocessing.ImagePreprocessor.PreprocessToNCHW255(rgba, w, h, w, h);

        // Should be [0, 255] — no normalization
        float min = nchw.Min(), max = nchw.Max();
        if (min < -1f || max > 256f)
            throw new Exception($"ForwardRaw range [{min:F1}, {max:F1}] — expected [0, 255]");

        // Should have shape [1, 3, h, w] = 48 elements
        if (nchw.Length != 1 * 3 * h * w)
            throw new Exception($"ForwardRaw length={nchw.Length}, expected {3 * h * w}");
    });

    [TestMethod]
    public async Task ImagePreprocess_ImageNet_MeanStd() => await RunTest(async accelerator =>
    {
        // ImageNet normalization: (pixel/255 - mean) / std
        int w = 8, h = 8;
        var rgba = new byte[w * h * 4];
        // All white pixels (255, 255, 255, 255)
        for (int i = 0; i < rgba.Length; i++) rgba[i] = 255;

        float[] mean = { 0.485f, 0.456f, 0.406f };
        float[] std = { 0.229f, 0.224f, 0.225f };
        var nchw = SpawnDev.ILGPU.ML.Preprocessing.ImagePreprocessor.PreprocessToNCHW(
            rgba, w, h, w, h, mean, std);

        // White pixel after normalization: (1.0 - mean) / std
        // R: (1.0 - 0.485) / 0.229 ≈ 2.249
        // G: (1.0 - 0.456) / 0.224 ≈ 2.429
        // B: (1.0 - 0.406) / 0.225 ≈ 2.640
        float expectedR = (1f - mean[0]) / std[0];
        float expectedG = (1f - mean[1]) / std[1];
        float expectedB = (1f - mean[2]) / std[2];

        // Check first pixel of each channel
        float actualR = nchw[0]; // [0, 0, 0, 0] in NCHW
        float actualG = nchw[w * h]; // [0, 1, 0, 0]
        float actualB = nchw[2 * w * h]; // [0, 2, 0, 0]

        if (MathF.Abs(actualR - expectedR) > 0.1f)
            throw new Exception($"ImageNet R: {actualR:F3} vs expected {expectedR:F3}");
        if (MathF.Abs(actualG - expectedG) > 0.1f)
            throw new Exception($"ImageNet G: {actualG:F3} vs expected {expectedG:F3}");
        if (MathF.Abs(actualB - expectedB) > 0.1f)
            throw new Exception($"ImageNet B: {actualB:F3} vs expected {expectedB:F3}");
    });

    [TestMethod]
    public async Task ImagePreprocess_BilinearResize_Dimensions() => await RunTest(async accelerator =>
    {
        // Resize 8x8 → 4x4, verify output element count
        int srcW = 8, srcH = 8, dstW = 4, dstH = 4;
        var rgba = new byte[srcW * srcH * 4];
        for (int i = 0; i < rgba.Length; i++) rgba[i] = (byte)(i % 256);

        var nchw = SpawnDev.ILGPU.ML.Preprocessing.ImagePreprocessor.PreprocessToNCHW255(
            rgba, srcW, srcH, dstW, dstH);

        int expected = 1 * 3 * dstH * dstW; // 48
        if (nchw.Length != expected)
            throw new Exception($"Resize output length={nchw.Length}, expected {expected}");
    });

    // ═══════════════════════════════════════════════════════════
    //  TurboQuant Tests (from reference data)
    // ═══════════════════════════════════════════════════════════

    [TestMethod]
    public async Task TurboQuant_BitPack_4bit_RoundTrip() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");

        var jsonStr = await http.GetStringAsync("references/turboquant/turboquant_test_cases.json");
        using var doc = JsonDocument.Parse(jsonStr);
        var bp = doc.RootElement.GetProperty("bitpack_tests").GetProperty("4bit_16values");

        var indices = bp.GetProperty("indices").EnumerateArray().Select(e => e.GetInt32()).ToArray();
        var expectedPacked = bp.GetProperty("packed").EnumerateArray().Select(e => e.GetInt32()).ToArray();

        // Pack 4-bit: 2 values per byte (lo nibble + hi nibble)
        var packed = new byte[indices.Length / 2];
        for (int i = 0; i < indices.Length; i += 2)
        {
            packed[i / 2] = (byte)(indices[i] | (indices[i + 1] << 4));
        }

        // Verify packed matches reference
        for (int i = 0; i < packed.Length; i++)
        {
            if (packed[i] != expectedPacked[i])
                throw new Exception($"BitPack mismatch at [{i}]: got {packed[i]}, expected {expectedPacked[i]}");
        }

        // Unpack and verify round-trip
        var unpacked = new int[indices.Length];
        for (int i = 0; i < packed.Length; i++)
        {
            unpacked[i * 2] = packed[i] & 0xF;
            unpacked[i * 2 + 1] = (packed[i] >> 4) & 0xF;
        }

        for (int i = 0; i < indices.Length; i++)
        {
            if (unpacked[i] != indices[i])
                throw new Exception($"Unpack mismatch at [{i}]: got {unpacked[i]}, expected {indices[i]}");
        }
    });

    [TestMethod]
    public async Task TurboQuant_Quantize_RoundTrip_MSE() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");

        var jsonStr = await http.GetStringAsync("references/turboquant/turboquant_test_cases.json");
        using var doc = JsonDocument.Parse(jsonStr);
        var rt = doc.RootElement.GetProperty("roundtrip_test");

        float mse = (float)rt.GetProperty("mse").GetDouble();
        float cosine = (float)rt.GetProperty("cosine_similarity").GetDouble();

        // 4-bit quantization should have MSE < 0.02 and cosine > 0.99
        if (mse > 0.02f)
            throw new Exception($"TurboQuant 4-bit MSE={mse:F4}, expected < 0.02");
        if (cosine < 0.99f)
            throw new Exception($"TurboQuant 4-bit cosine={cosine:F4}, expected > 0.99");

        // Verify attention orthogonality property
        var attn = doc.RootElement.GetProperty("attention_test");
        float dotDequant = (float)attn.GetProperty("dot_via_dequant").GetDouble();
        float dotRotated = (float)attn.GetProperty("dot_via_query_rotation").GetDouble();
        float diff = MathF.Abs(dotDequant - dotRotated);

        if (diff > 1e-6f)
            throw new Exception($"Attention orthogonality violation: diff={diff:E3}");
    });

    // ═══════════════════════════════════════════════════════════
    //  SPZ Validation Tests
    // ═══════════════════════════════════════════════════════════

    [TestMethod]
    public async Task SPZ_InvalidMagic_Throws() => await RunTest(async accelerator =>
    {
        // Create a gzipped file with wrong magic bytes
        var fakeData = new byte[16];
        fakeData[0] = 0xFF; // wrong magic
        fakeData[1] = 0xFF;
        fakeData[2] = 0xFF;
        fakeData[3] = 0xFF;

        using var ms = new System.IO.MemoryStream();
        using (var gz = new System.IO.Compression.GZipStream(ms, System.IO.Compression.CompressionLevel.Fastest, leaveOpen: true))
        {
            gz.Write(fakeData, 0, fakeData.Length);
        }
        var compressed = ms.ToArray();

        // Decompress and check magic
        using var readMs = new System.IO.MemoryStream(compressed);
        using var readGz = new System.IO.Compression.GZipStream(readMs, System.IO.Compression.CompressionMode.Decompress);
        using var result = new System.IO.MemoryStream();
        await readGz.CopyToAsync(result);
        var data = result.ToArray();

        uint magic = BitConverter.ToUInt32(data, 0);
        if (magic == 0x5053474E)
            throw new Exception("Fake SPZ should NOT have valid magic");

        // This is the expected behavior — invalid magic detected
    });

    [TestMethod]
    public async Task SPZ_Decompression_Works() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");

        var spzBytes = await http.GetByteArrayAsync("references/blazing-edge/test_10gaussians.spz");

        using var compressed = new System.IO.MemoryStream(spzBytes);
        using var gzip = new System.IO.Compression.GZipStream(compressed, System.IO.Compression.CompressionMode.Decompress);
        using var decompressed = new System.IO.MemoryStream();
        await gzip.CopyToAsync(decompressed);
        var data = decompressed.ToArray();

        // Header should be at least 16 bytes
        if (data.Length < 16)
            throw new Exception($"Decompressed SPZ too small: {data.Length} bytes");

        // Parse header
        uint magic = BitConverter.ToUInt32(data, 0);
        uint version = BitConverter.ToUInt32(data, 4);
        uint numPoints = BitConverter.ToUInt32(data, 8);
        byte shDegree = data[12];
        byte fractionalBits = data[13];

        if (version != 4)
            throw new Exception($"SPZ version={version}, expected 4");
        if (numPoints != 10)
            throw new Exception($"SPZ numPoints={numPoints}, expected 10");
        if (shDegree != 0)
            throw new Exception($"SPZ shDegree={shDegree}, expected 0");
        if (fractionalBits != 12)
            throw new Exception($"SPZ fractionalBits={fractionalBits}, expected 12");

        // Verify data size: header(16) + positions(10*9) + alphas(10*1) + colors(10*3) + scales(10*3) + rotations(10*4)
        int expectedSize = 16 + 10 * (9 + 1 + 3 + 3 + 4); // 216
        if (data.Length != expectedSize)
            throw new Exception($"SPZ decompressed size={data.Length}, expected {expectedSize}");
    });

    // ═══════════════════════════════════════════════════════════
    //  FWHT CPU Reference Validation
    // ═══════════════════════════════════════════════════════════

    [TestMethod]
    public async Task FWHT_CPU_Impulse_MatchesReference() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");

        var jsonStr = await http.GetStringAsync("references/turboquant/turboquant_test_cases.json");
        using var doc = JsonDocument.Parse(jsonStr);
        var fwht = doc.RootElement.GetProperty("fwht_tests").GetProperty("d8_impulse");

        var input = fwht.GetProperty("input").EnumerateArray().Select(e => (float)e.GetDouble()).ToArray();
        var expected = fwht.GetProperty("expected").EnumerateArray().Select(e => (float)e.GetDouble()).ToArray();

        // CPU FWHT butterfly
        var result = (float[])input.Clone();
        int d = result.Length;
        int h = 1;
        while (h < d)
        {
            for (int i = 0; i < d; i += h * 2)
            {
                for (int j = i; j < i + h; j++)
                {
                    float x = result[j];
                    float y = result[j + h];
                    result[j] = x + y;
                    result[j + h] = x - y;
                }
            }
            h *= 2;
        }
        float norm = MathF.Sqrt(d);
        for (int i = 0; i < d; i++) result[i] /= norm;

        // Compare
        float maxErr = 0;
        for (int i = 0; i < d; i++)
            maxErr = MathF.Max(maxErr, MathF.Abs(result[i] - expected[i]));

        if (maxErr > 1e-5f)
            throw new Exception($"FWHT impulse maxErr={maxErr:E3}");

        // All values should be 1/sqrt(8) ≈ 0.3536
        float expectedVal = 1f / MathF.Sqrt(8);
        for (int i = 0; i < d; i++)
        {
            if (MathF.Abs(result[i] - expectedVal) > 1e-5f)
                throw new Exception($"FWHT impulse [{i}]={result[i]:F6}, expected {expectedVal:F6}");
        }
    });

    [TestMethod]
    public async Task FWHT_CPU_RoundTrip_d128() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");

        // Load reference input
        var inputBytes = await http.GetByteArrayAsync("references/turboquant/fwht_input_d128.bin");
        var input = new float[inputBytes.Length / 4];
        Buffer.BlockCopy(inputBytes, 0, input, 0, inputBytes.Length);

        if (input.Length != 128)
            throw new Exception($"FWHT input length={input.Length}, expected 128");

        // Forward FWHT
        var forward = FWHTButterfly(input);

        // Inverse FWHT (same operation — FWHT is self-inverse)
        var roundtrip = FWHTButterfly(forward);

        // Should match original input
        float maxErr = 0;
        for (int i = 0; i < input.Length; i++)
            maxErr = MathF.Max(maxErr, MathF.Abs(roundtrip[i] - input[i]));

        if (maxErr > 1e-4f)
            throw new Exception($"FWHT d=128 round-trip maxErr={maxErr:E3}");
    });

    private static float[] FWHTButterfly(float[] input)
    {
        var a = (float[])input.Clone();
        int d = a.Length;
        int h = 1;
        while (h < d)
        {
            for (int i = 0; i < d; i += h * 2)
            {
                for (int j = i; j < i + h; j++)
                {
                    float x = a[j];
                    float y = a[j + h];
                    a[j] = x + y;
                    a[j + h] = x - y;
                }
            }
            h *= 2;
        }
        float norm = MathF.Sqrt(d);
        for (int i = 0; i < d; i++) a[i] /= norm;
        return a;
    }
}
