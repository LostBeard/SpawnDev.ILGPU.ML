using System.IO.Compression;

namespace SpawnDev.ILGPU.ML.PyTorch;

/// <summary>
/// Minimal PyTorch checkpoint loader.
/// PyTorch .pt/.pth files are ZIP archives containing:
///   - archive/data.pkl (pickle-serialized model structure — we parse minimally)
///   - archive/data/0, archive/data/1, ... (raw tensor data files)
///
/// Full pickle parsing is impractical without a Python interpreter.
/// Instead, we extract tensor data from the ZIP entries and pair with
/// metadata from the pickle header (tensor names, shapes, dtypes, storage keys).
///
/// For most use cases, users should export to ONNX or SafeTensors instead.
/// This loader provides basic weight extraction for inspection and compatibility checking.
/// </summary>
public static class PyTorchLoader
{
    /// <summary>Check if a byte array is a ZIP (PyTorch checkpoint) file.</summary>
    public static bool IsPyTorchCheckpoint(byte[] data) =>
        data.Length >= 4 && data[0] == 'P' && data[1] == 'K' && data[2] == 0x03 && data[3] == 0x04;

    /// <summary>
    /// Extract tensor information from a PyTorch checkpoint.
    /// Returns tensor names and data file references.
    /// Does NOT parse the pickle — reads ZIP entry names to infer structure.
    /// </summary>
    public static PyTorchCheckpoint Parse(byte[] data)
    {
        var checkpoint = new PyTorchCheckpoint { RawData = data };

        using var ms = new MemoryStream(data);
        using var zip = new ZipArchive(ms, ZipArchiveMode.Read);

        foreach (var entry in zip.Entries)
        {
            var path = entry.FullName;

            if (path.EndsWith(".pkl"))
            {
                // Read pickle to extract tensor metadata
                // Minimal parsing: look for tensor names + storage keys in the binary
                using var stream = entry.Open();
                using var pklMs = new MemoryStream();
                stream.CopyTo(pklMs);
                checkpoint.PickleData = pklMs.ToArray();
                ExtractTensorMetadataFromPickle(checkpoint);
            }
            else if (path.Contains("/data/") && !path.EndsWith("/"))
            {
                // Raw tensor data file (e.g., "archive/data/0")
                var key = Path.GetFileName(path);
                using var stream = entry.Open();
                using var dataMs = new MemoryStream();
                stream.CopyTo(dataMs);
                checkpoint.DataFiles[key] = dataMs.ToArray();
            }
        }

        return checkpoint;
    }

    /// <summary>Get a summary string.</summary>
    public static string GetSummary(PyTorchCheckpoint checkpoint)
    {
        long totalBytes = checkpoint.DataFiles.Values.Sum(d => d.Length);
        return $"PyTorch checkpoint: {checkpoint.DataFiles.Count} data files, " +
               $"{checkpoint.TensorNames.Count} tensors, " +
               $"{totalBytes / 1024.0 / 1024.0:F1} MB data";
    }

    /// <summary>
    /// Minimal pickle parser — extracts tensor names from the pickle stream.
    /// Doesn't fully parse pickle opcodes, just scans for string patterns
    /// that look like tensor names (e.g., "model.layers.0.weight").
    /// </summary>
    private static void ExtractTensorMetadataFromPickle(PyTorchCheckpoint checkpoint)
    {
        if (checkpoint.PickleData == null) return;
        var data = checkpoint.PickleData;

        // Scan for short strings (pickle BINUNICODE opcode = 0x8C, followed by length byte, then UTF8)
        for (int i = 0; i < data.Length - 3; i++)
        {
            if (data[i] == 0x8C) // SHORT_BINUNICODE
            {
                int len = data[i + 1];
                if (len > 2 && len < 200 && i + 2 + len <= data.Length)
                {
                    var str = System.Text.Encoding.UTF8.GetString(data, i + 2, len);
                    // Tensor names typically contain dots and end with .weight or .bias
                    if (str.Contains('.') && (str.EndsWith(".weight") || str.EndsWith(".bias")
                        || str.EndsWith(".running_mean") || str.EndsWith(".running_var")
                        || str.EndsWith(".num_batches_tracked") || str.EndsWith(".scale")
                        || str.Contains("embed") || str.Contains("norm")))
                    {
                        if (!checkpoint.TensorNames.Contains(str))
                            checkpoint.TensorNames.Add(str);
                    }
                }
            }
        }
    }
}

/// <summary>
/// Parsed PyTorch checkpoint.
/// </summary>
public class PyTorchCheckpoint
{
    public byte[] RawData { get; set; } = Array.Empty<byte>();
    public byte[]? PickleData { get; set; }
    public Dictionary<string, byte[]> DataFiles { get; set; } = new();
    public List<string> TensorNames { get; set; } = new();
}
