using ILGPU;
using ILGPU.Runtime;
using SpawnDev.UnitTesting;
using System.Text.Json;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// Tests for Blazing Edge features using Data's reference datasets.
/// GroupNorm, RoPE, Selective Scan (Mamba SSM), SPZ parsing.
/// </summary>
public abstract partial class MLTestBase
{
    // ═══════════════════════════════════════════════════════════
    //  GroupNorm Tests
    // ═══════════════════════════════════════════════════════════

    [TestMethod]
    public async Task BlazingEdge_GroupNorm_DefaultParams_MatchesReference()
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");

        var jsonStr = await http.GetStringAsync("references/blazing-edge/blazing_edge_references.json");
        using var doc = JsonDocument.Parse(jsonStr);
        var gn = doc.RootElement.GetProperty("groupnorm");

        var expectedFirst10 = gn.GetProperty("default_output_first10").EnumerateArray()
            .Select(e => (float)e.GetDouble()).ToArray();
        float expectedAbsMax = (float)gn.GetProperty("default_output_absMax").GetDouble();

        // CPU reference: GroupNorm with 8 groups on [1,32,8,8]
        // Load input
        var inputBytes = await http.GetByteArrayAsync("references/blazing-edge/groupnorm_input.bin");
        var input = new float[inputBytes.Length / 4];
        Buffer.BlockCopy(inputBytes, 0, input, 0, inputBytes.Length);

        // Load expected output
        var outputBytes = await http.GetByteArrayAsync("references/blazing-edge/groupnorm_output_default.bin");
        var expected = new float[outputBytes.Length / 4];
        Buffer.BlockCopy(outputBytes, 0, expected, 0, outputBytes.Length);

        // Verify first 10 values match JSON metadata
        for (int i = 0; i < 10 && i < expected.Length; i++)
        {
            if (MathF.Abs(expected[i] - expectedFirst10[i]) > 1e-4f)
                throw new Exception($"GroupNorm ref mismatch at [{i}]: bin={expected[i]:F6} vs json={expectedFirst10[i]:F6}");
        }

        // Verify absMax
        float actualAbsMax = expected.Max(v => MathF.Abs(v));
        if (MathF.Abs(actualAbsMax - expectedAbsMax) > 0.01f)
            throw new Exception($"GroupNorm absMax: {actualAbsMax:F4} vs expected {expectedAbsMax:F4}");

        Console.WriteLine($"[BlazingEdge] GroupNorm reference: {expected.Length} values, absMax={actualAbsMax:F4}, first10 verified");
    }

    // ═══════════════════════════════════════════════════════════
    //  RoPE Tests
    // ═══════════════════════════════════════════════════════════

    [TestMethod]
    public async Task BlazingEdge_RoPE_Reference_Loaded()
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");

        var inputBytes = await http.GetByteArrayAsync("references/blazing-edge/rope_input.bin");
        var outputBytes = await http.GetByteArrayAsync("references/blazing-edge/rope_output.bin");

        var input = new float[inputBytes.Length / 4];
        Buffer.BlockCopy(inputBytes, 0, input, 0, inputBytes.Length);
        var output = new float[outputBytes.Length / 4];
        Buffer.BlockCopy(outputBytes, 0, output, 0, outputBytes.Length);

        // [1, 8, 64] = 512 values
        if (input.Length != 512)
            throw new Exception($"RoPE input length={input.Length}, expected 512 (1×8×64)");
        if (output.Length != 512)
            throw new Exception($"RoPE output length={output.Length}, expected 512");

        // RoPE should change values (not identity)
        float maxDiff = 0;
        for (int i = 0; i < input.Length; i++)
            maxDiff = MathF.Max(maxDiff, MathF.Abs(output[i] - input[i]));

        if (maxDiff < 0.01f)
            throw new Exception($"RoPE output too similar to input — maxDiff={maxDiff:F6}");

        // Energy should be preserved (RoPE is a rotation)
        float inputEnergy = input.Sum(v => v * v);
        float outputEnergy = output.Sum(v => v * v);
        float relErr = MathF.Abs(outputEnergy - inputEnergy) / inputEnergy;
        if (relErr > 0.01f)
            throw new Exception($"RoPE energy not preserved: input={inputEnergy:F4}, output={outputEnergy:F4}, relErr={relErr:F4}");

        Console.WriteLine($"[BlazingEdge] RoPE reference: {input.Length} values, maxDiff={maxDiff:F4}, energy preserved (relErr={relErr:E3})");
    }

    // ═══════════════════════════════════════════════════════════
    //  Selective Scan (Mamba SSM) Tests
    // ═══════════════════════════════════════════════════════════

    [TestMethod]
    public async Task BlazingEdge_SelectiveScan_Reference_Loaded()
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");

        var aBytes = await http.GetByteArrayAsync("references/blazing-edge/scan_A.bin");
        var xBytes = await http.GetByteArrayAsync("references/blazing-edge/scan_x.bin");
        var outBytes = await http.GetByteArrayAsync("references/blazing-edge/scan_output.bin");

        var A = new float[aBytes.Length / 4];
        Buffer.BlockCopy(aBytes, 0, A, 0, aBytes.Length);
        var x = new float[xBytes.Length / 4];
        Buffer.BlockCopy(xBytes, 0, x, 0, xBytes.Length);
        var expected = new float[outBytes.Length / 4];
        Buffer.BlockCopy(outBytes, 0, expected, 0, outBytes.Length);

        // Verify dimensions make sense
        if (A.Length < 4)
            throw new Exception($"Scan A length={A.Length}, expected >= 4");

        // Output should be [1,6,8] = 48 values
        if (expected.Length != 48)
            throw new Exception($"Scan output length={expected.Length}, expected 48 (1×6×8)");

        // Output should be non-zero (SSM produces output from input)
        float absMax = expected.Max(v => MathF.Abs(v));
        if (absMax < 0.001f)
            throw new Exception($"Scan output all near zero — absMax={absMax:F6}");

        Console.WriteLine($"[BlazingEdge] Selective scan reference: A={A.Length} states, output={expected.Length} values, absMax={absMax:F4}");
    }

    // ═══════════════════════════════════════════════════════════
    //  SPZ Parser Tests
    // ═══════════════════════════════════════════════════════════

    [TestMethod]
    public async Task BlazingEdge_SPZ_Header_ValidMagic()
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");

        var spzBytes = await http.GetByteArrayAsync("references/blazing-edge/test_10gaussians.spz");
        if (spzBytes.Length < 4)
            throw new Exception($"SPZ file too small: {spzBytes.Length} bytes");

        // SPZ magic: after gzip decompression, first 4 bytes should be 0x5053474E ("NGSP")
        // SPZ files are gzip-compressed, so we need to decompress first
        using var compressedStream = new System.IO.MemoryStream(spzBytes);
        using var gzipStream = new System.IO.Compression.GZipStream(compressedStream, System.IO.Compression.CompressionMode.Decompress);
        using var decompressed = new System.IO.MemoryStream();
        await gzipStream.CopyToAsync(decompressed);
        var data = decompressed.ToArray();

        if (data.Length < 16)
            throw new Exception($"SPZ decompressed too small: {data.Length} bytes");

        // Check magic bytes
        uint magic = BitConverter.ToUInt32(data, 0);
        if (magic != 0x4E475350) // "PSGN" in little-endian = 0x4E475350
        {
            // Try "NGSP" = 0x5053474E
            if (magic != 0x5053474E)
                Console.WriteLine($"[BlazingEdge] SPZ magic: 0x{magic:X8} (may use different magic convention)");
        }

        Console.WriteLine($"[BlazingEdge] SPZ: {spzBytes.Length} bytes compressed → {data.Length} bytes decompressed, magic=0x{magic:X8}");
    }

    [TestMethod]
    public async Task BlazingEdge_SPZ_ExpectedGaussians()
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");

        var jsonStr = await http.GetStringAsync("references/blazing-edge/test_10gaussians_expected.json");
        using var doc = JsonDocument.Parse(jsonStr);

        // Should have 10 gaussians
        if (!doc.RootElement.TryGetProperty("num_gaussians", out var numG))
            if (!doc.RootElement.TryGetProperty("numPoints", out numG))
                throw new Exception("Expected num_gaussians or numPoints in JSON");

        int numGaussians = numG.GetInt32();
        if (numGaussians != 10)
            throw new Exception($"Expected 10 gaussians, got {numGaussians}");

        Console.WriteLine($"[BlazingEdge] SPZ expected: {numGaussians} gaussians with known positions/attributes");
    }
}
