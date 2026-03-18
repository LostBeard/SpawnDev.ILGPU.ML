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

    public BufferPool(Accelerator accelerator) => _accelerator = accelerator;

    /// <summary>Rent a tensor with the given shape. May reuse a pooled buffer.</summary>
    public Tensor Rent(int[] shape, string? name = null)
    {
        int count = TensorHelpers.ElementCount(shape);
        int bucketSize = NextPowerOf2(count);

        if (_buckets.TryGetValue(bucketSize, out var stack) && stack.Count > 0)
        {
            var buffer = stack.Pop();
            return new Tensor(buffer.View, shape, name);
        }

        var newBuffer = _accelerator.Allocate1D<float>(bucketSize);
        _allBuffers.Add(newBuffer);
        return new Tensor(newBuffer.View, shape, name);
    }

    /// <summary>Return a tensor to the pool for reuse.</summary>
    public void Return(Tensor tensor)
    {
        int bucketSize = NextPowerOf2(tensor.ElementCount);
        if (!_buckets.TryGetValue(bucketSize, out var stack))
        {
            stack = new Stack<MemoryBuffer1D<float, Stride1D.Dense>>();
            _buckets[bucketSize] = stack;
        }
        // We need to find the parent buffer — for now, we can't easily do this
        // since Tensor holds a View, not a Buffer. This is a simplified pool.
        // TODO: Track buffer→tensor mapping for proper pooling.
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

    public void Dispose()
    {
        foreach (var buffer in _allBuffers)
            buffer.Dispose();
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
