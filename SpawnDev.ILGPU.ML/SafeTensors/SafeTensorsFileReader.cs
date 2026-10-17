using System.Text;
using System.Text.Json;
using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.SafeTensors;

/// <summary>
/// File-backed SafeTensors reader for checkpoints that do NOT fit in one <c>byte[]</c> (a 2.6 GB DiT is over the
/// 2 GB array limit, and holding a 16 GB model set in managed memory is the thing this class exists to avoid).
/// Only the header is read at <see cref="Open"/>; tensor bytes are streamed from the file to the device in
/// bounded chunks (<see cref="Upload{T}"/>), so peak managed memory is one chunk, never a tensor.
/// <see cref="SafeTensorsParser"/> remains the in-memory path for small files and the browser.
/// </summary>
public sealed class SafeTensorsFileReader : IDisposable
{
    private readonly FileStream _file;
    private readonly Dictionary<string, SafeTensorInfo> _byName;

    /// <summary>Absolute path of the opened file.</summary>
    public string Path { get; }

    /// <summary>All tensors in header order. <see cref="SafeTensorInfo.DataOffset"/> is ABSOLUTE in the file.</summary>
    public IReadOnlyList<SafeTensorInfo> Tensors { get; }

    /// <summary>The optional <c>__metadata__</c> string map (empty when absent).</summary>
    public IReadOnlyDictionary<string, string> Metadata { get; }

    private SafeTensorsFileReader(string path, FileStream file, List<SafeTensorInfo> tensors, Dictionary<string, string> metadata)
    {
        Path = path;
        _file = file;
        Tensors = tensors;
        Metadata = metadata;
        _byName = new Dictionary<string, SafeTensorInfo>(tensors.Count, StringComparer.Ordinal);
        foreach (var t in tensors) _byName[t.Name] = t;
    }

    /// <summary>Open <paramref name="path"/> and parse ONLY the header. Throws on a malformed or truncated file.</summary>
    public static SafeTensorsFileReader Open(string path)
    {
        var file = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 16, FileOptions.RandomAccess);
        try
        {
            Span<byte> lenBytes = stackalloc byte[8];
            ReadExactly(file, 0, lenBytes);
            long headerSize = BitConverter.ToInt64(lenBytes);
            if (headerSize <= 0 || headerSize > 100_000_000 || 8 + headerSize > file.Length)
                throw new InvalidDataException($"'{path}': invalid SafeTensors header size {headerSize} (file is {file.Length} bytes)");

            var headerBytes = new byte[headerSize];
            ReadExactly(file, 8, headerBytes);
            var header = MLJson.Deserialize<Dictionary<string, JsonElement>>(Encoding.UTF8.GetString(headerBytes))
                ?? throw new InvalidDataException($"'{path}': SafeTensors header is not a JSON object");

            long dataStart = 8 + headerSize;
            long dataLength = file.Length - dataStart;
            var tensors = new List<SafeTensorInfo>(header.Count);
            var metadata = new Dictionary<string, string>();
            foreach (var (key, value) in header)
            {
                if (key == "__metadata__")
                {
                    if (value.ValueKind == JsonValueKind.Object)
                        foreach (var prop in value.EnumerateObject())
                            metadata[prop.Name] = prop.Value.GetString() ?? "";
                    continue;
                }
                var shape = value.GetProperty("shape").EnumerateArray().Select(e => e.GetInt32()).ToArray();
                var offsets = value.GetProperty("data_offsets").EnumerateArray().Select(e => e.GetInt64()).ToArray();
                if (offsets[0] < 0 || offsets[1] < offsets[0] || offsets[1] > dataLength)
                    throw new InvalidDataException($"'{path}': tensor '{key}' data_offsets [{offsets[0]}, {offsets[1]}) fall outside the {dataLength} byte data section (truncated download?)");
                tensors.Add(new SafeTensorInfo
                {
                    Name = key,
                    DType = value.GetProperty("dtype").GetString() ?? "F32",
                    Shape = shape,
                    DataOffset = dataStart + offsets[0],
                    DataLength = offsets[1] - offsets[0],
                });
            }
            // header order is a dictionary order; sort by file offset so sequential reads stay sequential
            tensors.Sort((a, b) => a.DataOffset.CompareTo(b.DataOffset));
            return new SafeTensorsFileReader(path, file, tensors, metadata);
        }
        catch
        {
            file.Dispose();
            throw;
        }
    }

    /// <summary>True when the file holds a tensor named <paramref name="name"/>.</summary>
    public bool Has(string name) => _byName.ContainsKey(name);

    /// <summary>The header entry for <paramref name="name"/>; throws <see cref="KeyNotFoundException"/> naming the file.</summary>
    public SafeTensorInfo Get(string name)
        => _byName.TryGetValue(name, out var t) ? t : throw new KeyNotFoundException($"Tensor '{name}' not in '{Path}'");

    /// <summary>Element count of <paramref name="t"/> (product of shape; 1 for a scalar).</summary>
    public static long ElementCount(SafeTensorInfo t) => t.Shape.Aggregate(1L, (a, b) => a * b);

    /// <summary>Bytes per element of a SafeTensors dtype string, or throws for dtypes this reader does not map.</summary>
    public static int BytesPerElement(string dtype) => dtype switch
    {
        "F64" or "I64" or "U64" => 8,
        "F32" or "I32" or "U32" => 4,
        "F16" or "BF16" or "I16" or "U16" => 2,
        "I8" or "U8" or "BOOL" or "F8_E4M3" or "F8_E5M2" => 1,
        _ => throw new NotSupportedException($"SafeTensors dtype '{dtype}' has no known element size"),
    };

    /// <summary>
    /// Read the raw bytes of <paramref name="t"/> from <paramref name="byteOffset"/> (relative to the tensor start)
    /// into <paramref name="destination"/>. Used by the chunked upload; callable directly for small tensors.
    /// </summary>
    public void ReadBytes(SafeTensorInfo t, long byteOffset, Span<byte> destination)
    {
        if (byteOffset < 0 || byteOffset + destination.Length > t.DataLength)
            throw new ArgumentOutOfRangeException(nameof(byteOffset), $"read [{byteOffset}, {byteOffset + destination.Length}) is outside tensor '{t.Name}' ({t.DataLength} bytes)");
        lock (_file) ReadExactly(_file, t.DataOffset + byteOffset, destination);
    }

    /// <summary>
    /// Stream tensor <paramref name="t"/> into <paramref name="destination"/> (a view of at least ElementCount
    /// elements of <typeparamref name="T"/>) in chunks of at most <paramref name="chunkBytes"/>. The element
    /// type must have the same byte width as the stored dtype; this is a byte-exact upload (bf16 stays bf16),
    /// never a conversion, so a bf16 weight lands as <c>ILGPU.BFloat16</c> and an fp16 one as <c>ILGPU.Half</c>.
    /// </summary>
    public void Upload<T>(SafeTensorInfo t, ArrayView1D<T, Stride1D.Dense> destination,
        int chunkBytes = 32 * 1024 * 1024, CancellationToken cancellationToken = default)
        where T : unmanaged
    {
        int elemBytes = System.Runtime.CompilerServices.Unsafe.SizeOf<T>();
        if (BytesPerElement(t.DType) != elemBytes)
            throw new ArgumentException($"Tensor '{t.Name}' is {t.DType} ({BytesPerElement(t.DType)} B/element) but the destination element type {typeof(T).Name} is {elemBytes} B");
        long count = ElementCount(t);
        if (destination.Length < count)
            throw new ArgumentException($"Destination holds {destination.Length} elements, tensor '{t.Name}' needs {count}");
        int chunkElems = Math.Max(1, chunkBytes / elemBytes);
        var staging = new T[Math.Min(count, chunkElems)];
        for (long done = 0; done < count; done += staging.Length)
        {
            cancellationToken.ThrowIfCancellationRequested();
            int n = (int)Math.Min(staging.Length, count - done);
            ReadBytes(t, done * elemBytes, System.Runtime.InteropServices.MemoryMarshal.AsBytes(staging.AsSpan(0, n)));
            // CopyFromCPU is an immediate write (no pending-dispatch dependency), see CLAUDE.md batching rules.
            destination.SubView(done, n).CopyFromCPU(n == staging.Length ? staging : staging[..n]);
        }
    }

    /// <summary>Allocate a device buffer for <paramref name="name"/> and stream it in. Caller owns the buffer.</summary>
    public MemoryBuffer1D<T, Stride1D.Dense> UploadNew<T>(Accelerator accelerator, string name,
        int chunkBytes = 32 * 1024 * 1024, CancellationToken cancellationToken = default)
        where T : unmanaged
    {
        var t = Get(name);
        var buffer = accelerator.Allocate1D<T>(ElementCount(t));
        try
        {
            Upload(t, buffer.View, chunkBytes, cancellationToken);
            return buffer;
        }
        catch
        {
            buffer.Dispose();
            throw;
        }
    }

    private static void ReadExactly(FileStream file, long position, Span<byte> destination)
    {
        file.Position = position;
        file.ReadExactly(destination);
    }

    public void Dispose() => _file.Dispose();
}
