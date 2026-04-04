using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Kernels;

namespace SpawnDev.ILGPU.ML.Tensors;

/// <summary>
/// GPU buffer pool for reusing allocations. Buffers are bucketed by size
/// (rounded up to next power of 2) to maximize reuse.
///
/// Usage:
///   var tensor = pool.Rent(new[] { 1370, 384 });
///   // ... use tensor ...
///   pool.Return(tensor);
/// </summary>
public class BufferPool : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly Dictionary<int, Stack<MemoryBuffer1D<float, Stride1D.Dense>>> _buckets = new();
    private readonly List<MemoryBuffer1D<float, Stride1D.Dense>> _allBuffers = new();

    /// <summary>Total number of GPU buffers allocated by this pool.</summary>
    public int AllocatedBufferCount => _allBuffers.Count;
    /// <summary>Number of buffers available for reuse.</summary>
    public int AvailableBufferCount => _buckets.Values.Sum(s => s.Count);

    public BufferPool(Accelerator accelerator) => _accelerator = accelerator;

    /// <summary>Rent a tensor with the given shape. May reuse a pooled buffer.</summary>
    private readonly Dictionary<string, MemoryBuffer1D<float, Stride1D.Dense>> _namedBuffers = new();

    public Tensor Rent(int[] shape, string? name = null)
    {
        int count = TensorHelpers.ElementCount(shape);
        int bucketSize = NextPowerOf2(count);

        if (_buckets.TryGetValue(bucketSize, out var stack) && stack.Count > 0)
        {
            var buffer = stack.Pop();
            var tensor = new Tensor(buffer.View, shape, name);
            if (name != null) _namedBuffers[name] = buffer;
            return tensor;
        }

        var newBuffer = _accelerator.Allocate1D<float>(bucketSize);
        _allBuffers.Add(newBuffer);
        var newTensor = new Tensor(newBuffer.View, shape, name);
        if (name != null) _namedBuffers[name] = newBuffer;
        return newTensor;
    }

    /// <summary>Return a tensor's buffer to the pool for reuse by name.</summary>
    public void Return(Tensor tensor)
    {
        var name = tensor.Name;
        if (name != null && _namedBuffers.TryGetValue(name, out var buffer))
        {
            _namedBuffers.Remove(name);
            int bucketSize = (int)buffer.Length;
            if (!_buckets.TryGetValue(bucketSize, out var stack))
            {
                stack = new Stack<MemoryBuffer1D<float, Stride1D.Dense>>();
                _buckets[bucketSize] = stack;
            }
            stack.Push(buffer);
        }
    }

    /// <summary>Allocate a permanent tensor (not pooled). For weights.</summary>
    public Tensor AllocatePermanent(float[] data, int[] shape, string? name = null)
    {
        var buffer = _accelerator.Allocate1D(data);
        _allBuffers.Add(buffer);
        return new Tensor(buffer.View, shape, name);
    }

    /// <summary>
    /// Allocate a permanent tensor from an ONNX tensor proto, uploading in chunks.
    /// Avoids allocating the full float[] for large tensors (GPT-2 = 154MB).
    /// Uses a reusable chunk buffer — peak CPU: chunk size (~1MB), not full tensor.
    /// </summary>
    public Tensor AllocatePermanentChunked(Onnx.OnnxTensorProto tensor, int[] shape, string? name = null)
    {
        int count = shape.Length > 0 ? shape.Aggregate(1, (a, b) => a * b) : 1;

        // For small tensors, use the standard path (no chunking overhead)
        if (count <= 262144) // 1MB
            return AllocatePermanent(tensor.ToFloatArray(), shape, name);

        // Large tensor: allocate empty GPU buffer, then fill in chunks
        var buffer = _accelerator.Allocate1D<float>(count);
        _allBuffers.Add(buffer);

        // Use a fixed-size chunk buffer (reusable across calls)
        const int CHUNK = 262144; // 256K floats = 1MB
        var chunk = new float[Math.Min(CHUNK, count)];

        // Determine raw data source — either RawData copy or zero-copy reference
        byte[]? rawBytes = tensor.RawData;
        int rawOffset = 0;
        if (rawBytes == null && tensor.RawDataSource != null)
        {
            rawBytes = tensor.RawDataSource;
            rawOffset = tensor.RawDataOffset;
        }

        // Convert and upload chunk by chunk
        // For FLOAT raw data (most common large tensor type): direct BlockCopy
        // Uses Scale(1.0f) for GPU→GPU copy — CopyTo is not supported on WebGPU.
        if (rawBytes != null && rawBytes.Length > 0 && tensor.DataType == 1)
        {
            // Upload chunks directly via CopyFromCPU (queue.writeBuffer on WebGPU).
            // Do NOT use Scale kernel + temp buffer: on WebGPU, the temp buffer is
            // destroyed before the batched command encoder submits, causing use-after-free
            // (all weights read as zeros). CopyFromCPU is immediate — no temp buffer needed.
            int offset = 0;
            while (offset < count)
            {
                int n = Math.Min(CHUNK, count - offset);
                var chunkSlice = new float[n];
                Buffer.BlockCopy(rawBytes, rawOffset + offset * 4, chunkSlice, 0, n * 4);
                buffer.View.SubView(offset, n).CopyFromCPU(chunkSlice);
                offset += n;
            }
        }
        else
        {
            // Other formats: convert full tensor (fallback — rare for large tensors)
            var data = tensor.ToFloatArray();
            buffer.CopyFromCPU(data);
        }

        return new Tensor(buffer.View, shape, name);
    }

    /// <summary>Allocate a permanent zero-initialized tensor.</summary>
    public Tensor AllocatePermanent(int[] shape, string? name = null)
    {
        int count = TensorHelpers.ElementCount(shape);
        var buffer = _accelerator.Allocate1D<float>(count);
        _allBuffers.Add(buffer);
        return new Tensor(buffer.View, shape, name);
    }

    public void Dispose()
    {
        foreach (var buffer in _allBuffers)
        {
            try { buffer.Dispose(); }
            catch { /* Buffer may already be disposed by executor ref-counting or external code */ }
        }
        _allBuffers.Clear();
        _buckets.Clear();
    }

    private static int NextPowerOf2(int v)
    {
        if (v <= 0) return 1;
        v--;
        v |= v >> 1;
        v |= v >> 2;
        v |= v >> 4;
        v |= v >> 8;
        v |= v >> 16;
        return v + 1;
    }
}
