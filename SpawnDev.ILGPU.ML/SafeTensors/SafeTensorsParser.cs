using System.Text;
using System.Text.Json;

namespace SpawnDev.ILGPU.ML.SafeTensors;

/// <summary>
/// Zero-dependency SafeTensors parser.
/// SafeTensors is HuggingFace's safe, zero-copy tensor storage format.
/// Layout: [header_size:uint64_le] [header_json:utf8] [tensor_data:bytes]
///
/// The header JSON maps tensor names to { dtype, shape, data_offsets: [start, end] }.
/// Data offsets are relative to the start of the tensor data section (after header).
///
/// https://huggingface.co/docs/safetensors
/// </summary>
public static class SafeTensorsParser
{
    /// <summary>
    /// Parse a SafeTensors file from raw bytes.
    /// </summary>
    public static SafeTensorsFile Parse(byte[] data)
    {
        if (data.Length < 8)
            throw new InvalidOperationException("File too small for SafeTensors format");

        // Read header size (uint64 little-endian)
        long headerSize = BitConverter.ToInt64(data, 0);
        if (headerSize <= 0 || headerSize > data.Length - 8)
            throw new InvalidOperationException($"Invalid SafeTensors header size: {headerSize}");

        // Parse header JSON
        var headerJson = Encoding.UTF8.GetString(data, 8, (int)headerSize);
        var header = JsonSerializer.Deserialize<Dictionary<string, JsonElement>>(headerJson)
            ?? throw new InvalidOperationException("Failed to parse SafeTensors header");

        long dataStartOffset = 8 + headerSize;
        var tensors = new List<SafeTensorInfo>();
        Dictionary<string, object>? metadata = null;

        foreach (var (key, value) in header)
        {
            if (key == "__metadata__")
            {
                // Metadata section (optional)
                metadata = new Dictionary<string, object>();
                if (value.ValueKind == JsonValueKind.Object)
                {
                    foreach (var prop in value.EnumerateObject())
                        metadata[prop.Name] = prop.Value.GetString() ?? "";
                }
                continue;
            }

            // Tensor entry
            var dtype = value.GetProperty("dtype").GetString() ?? "F32";
            var shape = value.GetProperty("shape").EnumerateArray().Select(e => e.GetInt32()).ToArray();
            var offsets = value.GetProperty("data_offsets").EnumerateArray().Select(e => e.GetInt64()).ToArray();

            tensors.Add(new SafeTensorInfo
            {
                Name = key,
                DType = dtype,
                Shape = shape,
                DataOffset = dataStartOffset + offsets[0],
                DataLength = offsets[1] - offsets[0],
            });
        }

        return new SafeTensorsFile
        {
            Tensors = tensors.ToArray(),
            Metadata = metadata ?? new Dictionary<string, object>(),
            RawData = data,
        };
    }

    /// <summary>Check if a byte array is a SafeTensors file.</summary>
    public static bool IsSafeTensors(byte[] data)
    {
        if (data.Length < 16) return false;
        long headerSize = BitConverter.ToInt64(data, 0);
        // Header size should be reasonable (< 100MB) and data should start with '{'
        return headerSize > 0 && headerSize < 100_000_000 && headerSize < data.Length - 8
            && data[8] == '{';
    }

    /// <summary>Get a quick summary string.</summary>
    public static string GetSummary(SafeTensorsFile file)
    {
        long totalParams = file.Tensors.Sum(t => t.Shape.Aggregate(1L, (a, b) => a * b));
        var dtypes = file.Tensors.Select(t => t.DType).Distinct();
        return $"SafeTensors: {file.Tensors.Length} tensors, {totalParams:N0} params, dtypes: {string.Join(", ", dtypes)}";
    }
}
