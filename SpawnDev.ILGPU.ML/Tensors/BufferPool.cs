using ILGPU;
using ILGPU.Runtime;

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
    // Maps tensor data pointer → parent buffer for proper return-to-pool
    private readonly Dictionary<long, MemoryBuffer1D<float, Stride1D.Dense>> _tensorBufferMap = new();

    /// <summary>Total number of GPU buffers allocated by this pool.</summary>
    public int AllocatedBufferCount => _allBuffers.Count;
    /// <summary>Number of buffers available for reuse.</summary>
    public int AvailableBufferCount => _buckets.Values.Sum(s => s.Count);

    public BufferPool(Accelerator accelerator) => _accelerator = accelerator;

    /// <summary>Rent a tensor with the given shape. May reuse a pooled buffer.</summary>
    // Track latest buffer for each bucket size for Return
    private MemoryBuffer1D<float, Stride1D.Dense>? _lastRentedBuffer;
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
            _lastRentedBuffer = buffer;
            return tensor;
        }

        var newBuffer = _accelerator.Allocate1D<float>(bucketSize);
        _allBuffers.Add(newBuffer);
        var newTensor = new Tensor(newBuffer.View, shape, name);
        if (name != null) _namedBuffers[name] = newBuffer;
        _lastRentedBuffer = newBuffer;
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

    /// <summary>Allocate a permanent zero-initialized tensor.</summary>
    public Tensor AllocatePermanent(int[] shape, string? name = null)
    {
        int count = TensorHelpers.ElementCount(shape);
        var buffer = _accelerator.Allocate1D<float>(count);
        _allBuffers.Add(buffer);
        return new Tensor(buffer.View, shape, name);
    }

    /// <summary>Force-dispose the GPU buffer backing a tensor. Removes from tracking.</summary>
    public void ForceDispose(Tensor tensor)
    {
        // Find and dispose the buffer that contains this tensor's view
        for (int i = _allBuffers.Count - 1; i >= 0; i--)
        {
            var buf = _allBuffers[i];
            // Match by checking if the tensor's element count fits in this buffer's size
            // and the buffer isn't already disposed
            if (buf.Length >= tensor.ElementCount)
            {
                try
                {
                    buf.Dispose();
                    _allBuffers.RemoveAt(i);
                    return;
                }
                catch { }
            }
        }
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
