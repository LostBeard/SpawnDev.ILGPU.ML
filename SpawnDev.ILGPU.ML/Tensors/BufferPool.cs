using System.IO;
using System.Threading;
using System.Threading.Tasks;
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
    // fp16 weight buffers (ILGPU.Half) — half the bytes of the fp32 _allBuffers. Tracked separately for
    // disposal (different element type). See AllocateHalfWeightFromStreamAsync.
    private readonly List<MemoryBuffer1D<global::ILGPU.Half, Stride1D.Dense>> _allHalfBuffers = new();

    /// <summary>Total number of GPU buffers allocated by this pool.</summary>
    public int AllocatedBufferCount => _allBuffers.Count;
    /// <summary>Number of buffers available for reuse.</summary>
    public int AvailableBufferCount => _buckets.Values.Sum(s => s.Count);

    /// <summary>Test/diagnostic switch: when true, the browser JS zero-copy weight-upload path is skipped and
    /// the .NET byte[] chunked path is used instead. Lets a measurement A/B the two upload paths from the same
    /// cached source to isolate the JS&lt;-&gt;.NET copy cost. Default false (zero-copy on where applicable).</summary>
    public static bool DisableJsZeroCopyWeights = false;

    /// <summary>Total weight bytes uploaded straight from JS to the GPU (zero-copy: never entered the .NET heap)
    /// by the browser streaming-load path. Stays 0 on desktop / non-JS streams. Lets a load measurement confirm
    /// the zero-copy path actually fired instead of the .NET byte[] fallback.</summary>
    public long ZeroCopyWeightBytes { get; private set; }

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

        // Memory-bounded execution: the pool retains every Returned buffer in size-buckets for reuse, so a
        // 1-pass large model (e.g. a 512x512 VAE decode = ~227 distinct-size feature maps, each used once)
        // grows the pool to the SUM of its intermediates (multi-GB), not the live working set. Under GPU memory
        // pressure, ILGPU's AllocateWithReclaim flushes pending GPU work, runs our reclaim (dispose the
        // AVAILABLE Returned-not-live bucketed buffers), retries once, and throws our working-set message if
        // still OOM. Models that fit never hit this; models that don't are bounded to their live set.
        MemoryBuffer1D<float, Stride1D.Dense> newBuffer = _accelerator.AllocateWithReclaim(
            () => _accelerator.Allocate1D<float>(bucketSize),   // allocate
            DisposeBucketedBuffers,                             // reclaim (dispose Returned-not-live), returns bytes freed
            reclaimed =>
            {
                long live = 0; foreach (var b in _allBuffers) live += b.LengthInBytes;
                long half = 0; foreach (var b in _allHalfBuffers) half += b.LengthInBytes;
                return $"GPU out of memory renting '{name}' (+{bucketSize * 4L / 1048576}MB) after reclaiming " +
                       $"{reclaimed / 1048576}MB of pooled buffers; live working set = {_allBuffers.Count} fp32 " +
                       $"({live / 1048576}MB) + {_allHalfBuffers.Count} fp16 ({half / 1048576}MB). Exceeds VRAM — " +
                       "needs tiled execution or fp16 activations.";
            });
        _allBuffers.Add(newBuffer);
        var newTensor = new Tensor(newBuffer.View, shape, name);
        if (name != null) _namedBuffers[name] = newBuffer;
        return newTensor;
    }

    /// <summary>
    /// Dispose all AVAILABLE (Returned, not-live) bucketed buffers, reclaiming their GPU memory. Live (rented)
    /// buffers — held in <c>_namedBuffers</c> until Returned — and permanent weights are untouched. Called under
    /// GPU memory pressure (see Rent's catch). The pool re-allocates these sizes on demand afterward, so the
    /// only cost is lost reuse, never correctness. Returns bytes freed.
    /// </summary>
    private long DisposeBucketedBuffers()
    {
        long freed = 0;
        foreach (var stack in _buckets.Values)
            while (stack.Count > 0)
            {
                var buf = stack.Pop();
                freed += buf.LengthInBytes;
                _allBuffers.Remove(buf);
                buf.Dispose();
            }
        _buckets.Clear();
        return freed;
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

    /// <summary>
    /// Allocate a permanent GPU tensor whose weight data lives at <paramref name="byteOffset"/> in a
    /// SEEKABLE <paramref name="stream"/> (the streaming-load path). Seeks once, then reads the raw_data in
    /// 1 MB chunks straight to the GPU via <c>CopyFromCPU</c> (queue.writeBuffer on WebGPU) — peak CPU is one
    /// chunk, never the whole tensor, and the bytes are never held as a managed array. Supports FLOAT32
    /// (dtype 1) and FLOAT16 (dtype 10) raw_data, converting FP16 → FP32 per chunk.
    /// </summary>
    public async Task<Tensor> AllocatePermanentFromStreamAsync(
        Stream stream, long byteOffset, int byteLength, int dataType, int[] shape,
        string? name = null, CancellationToken ct = default)
    {
        int count = shape.Length > 0 ? shape.Aggregate(1, (a, b) => a * b) : 1;
        var buffer = _accelerator.Allocate1D<float>(count);
        _allBuffers.Add(buffer);
        if (count == 0) return new Tensor(buffer.View, shape, name);

        int srcElemBytes = dataType switch { 1 => 4, 10 => 2, _ => 0 };
        if (srcElemBytes == 0)
            throw new NotSupportedException(
                $"Streaming load supports FLOAT32 (1) and FLOAT16 (10) raw_data; got dtype {dataType} for '{name}'. " +
                "Load this model via CreateFromOnnx(byte[]).");

        stream.Seek(byteOffset, SeekOrigin.Begin);

        // RAW FLOAT32: a fp32 weight's raw little-endian bytes ARE the GPU float buffer's bytes (no conversion),
        // so delegate to ILGPU's CopyFromStreamAsync. It streams the bytes in 16 MiB chunks and, on a browser
        // IJSReadStream source + browser buffer, goes JS->GPU via CopyFromJS without the bytes ever entering the
        // .NET/WASM managed heap (the whole point in the browser); on desktop it awaits ReadExactlyAsync ->
        // CopyFromCPU (genuinely async, never blocking). FP16 source (dtype 10) needs a per-element upcast, so
        // it stays on the byte[] path below. DisableJsZeroCopyWeights forces the managed loop (A/B diagnostic).
        if (dataType == 1 && byteLength == count * 4 && !DisableJsZeroCopyWeights)
        {
            await buffer.View.CopyFromStreamAsync(stream, cancellationToken: ct).ConfigureAwait(false);
            if (stream is SpawnDev.BlazorJS.Toolbox.IJSReadStream && buffer.Buffer is SpawnDev.ILGPU.IBrowserMemoryBuffer)
                ZeroCopyWeightBytes += byteLength; // count only the true JS->GPU zero-copy path
            return new Tensor(buffer.View, shape, name);
        }

        const int CHUNK = 262144; // 256K floats = 1 MB float buffer
        var byteBuf = new byte[CHUNK * srcElemBytes];
        var floatChunk = new float[CHUNK];
        int uploaded = 0;
        while (uploaded < count)
        {
            int n = Math.Min(CHUNK, count - uploaded);
            int wantBytes = n * srcElemBytes;
            await ReadExactAsync(stream, byteBuf, wantBytes, ct).ConfigureAwait(false);

            if (dataType == 1) // FLOAT32 — direct little-endian copy
                Buffer.BlockCopy(byteBuf, 0, floatChunk, 0, wantBytes);
            else // FLOAT16 → FLOAT32 per element
                for (int i = 0; i < n; i++)
                    floatChunk[i] = (float)BitConverter.ToHalf(byteBuf, i * 2);

            // Upload exactly n floats. CopyFromCPU is immediate (no temp buffer / command-encoder hazard).
            buffer.View.SubView(uploaded, n).CopyFromCPU(n == floatChunk.Length ? floatChunk : floatChunk[..n]);
            uploaded += n;
        }
        return new Tensor(buffer.View, shape, name);
    }

    /// <summary>
    /// Like <see cref="AllocatePermanentFromStreamAsync"/> but stores the weight as fp16
    /// (<see cref="ILGPU.Half"/>) — HALF the GPU bytes of the fp32 path. FLOAT16 (dtype 10) source is the
    /// common case (e.g. SD-Turbo); FLOAT32 (dtype 1) is downcast to fp16. Weight-consuming kernels read
    /// the result via their half-weight overload (e.g. <c>MatMulKernel.MatMulHalfWeight</c>), upconverting
    /// to float and accumulating fp32 (ORT-style mixed precision; no accuracy loss). Peak CPU is one chunk,
    /// never the whole tensor. The f16 spike (2026-06-05) confirmed ILGPU.Half storage works on all 6 backends.
    /// </summary>
    public async Task<HalfTensor> AllocateHalfWeightFromStreamAsync(
        Stream stream, long byteOffset, int byteLength, int dataType, int[] shape,
        string? name = null, CancellationToken ct = default)
    {
        int count = shape.Length > 0 ? shape.Aggregate(1, (a, b) => a * b) : 1;
        var buffer = _accelerator.Allocate1D<global::ILGPU.Half>(count);
        _allHalfBuffers.Add(buffer);
        if (count == 0) return new HalfTensor(buffer.View, shape, name);

        int srcElemBytes = dataType switch { 1 => 4, 10 => 2, _ => 0 };
        if (srcElemBytes == 0)
            throw new NotSupportedException(
                $"f16 weight load supports FLOAT32 (1) and FLOAT16 (10) raw_data; got dtype {dataType} for '{name}'.");

        stream.Seek(byteOffset, SeekOrigin.Begin);

        // RAW FLOAT16: ILGPU.Half is IEEE binary16, same 2-byte layout as the source (no conversion), so
        // delegate to CopyFromStreamAsync. It streams in 16 MiB chunks and, on a browser IJSReadStream + browser
        // buffer, goes JS->GPU via CopyFromJS with no managed-heap copy; the WebGPU 4-byte WriteBuffer rule (an
        // odd-count Half tensor = byteLength not a multiple of 4) is handled by its managed padded fallback, so
        // no element-count guard is needed here. FLOAT32 source (dtype 1) needs an fp32->fp16 downcast, so it
        // stays on the byte[] path below. DisableJsZeroCopyWeights forces the managed loop (A/B diagnostic).
        if (dataType == 10 && byteLength == count * 2 && !DisableJsZeroCopyWeights)
        {
            await buffer.View.CopyFromStreamAsync(stream, cancellationToken: ct).ConfigureAwait(false);
            if (stream is SpawnDev.BlazorJS.Toolbox.IJSReadStream && buffer.Buffer is SpawnDev.ILGPU.IBrowserMemoryBuffer)
                ZeroCopyWeightBytes += byteLength; // count only the true JS->GPU zero-copy path
            return new HalfTensor(buffer.View, shape, name);
        }

        const int CHUNK = 262144; // 256K elements
        var byteBuf = new byte[CHUNK * srcElemBytes];
        var halfChunk = new global::ILGPU.Half[CHUNK];
        int uploaded = 0;
        while (uploaded < count)
        {
            int n = Math.Min(CHUNK, count - uploaded);
            int wantBytes = n * srcElemBytes;
            await ReadExactAsync(stream, byteBuf, wantBytes, ct).ConfigureAwait(false);

            if (dataType == 10) // FLOAT16 source — round-trip through float (lossless: both are IEEE fp16)
                for (int i = 0; i < n; i++)
                    halfChunk[i] = (global::ILGPU.Half)(float)BitConverter.ToHalf(byteBuf, i * 2);
            else // FLOAT32 source — downcast to fp16
                for (int i = 0; i < n; i++)
                    halfChunk[i] = (global::ILGPU.Half)BitConverter.ToSingle(byteBuf, i * 4);

            buffer.View.SubView(uploaded, n).CopyFromCPU(n == halfChunk.Length ? halfChunk : halfChunk[..n]);
            uploaded += n;
        }
        return new HalfTensor(buffer.View, shape, name);
    }

    /// <summary>Read exactly <paramref name="count"/> bytes into the start of <paramref name="buf"/>, or throw.</summary>
    private static async Task ReadExactAsync(Stream stream, byte[] buf, int count, CancellationToken ct)
    {
        int got = 0;
        while (got < count)
        {
            int n = await stream.ReadAsync(buf.AsMemory(got, count - got), ct).ConfigureAwait(false);
            if (n == 0) throw new EndOfStreamException($"Stream ended {count - got} bytes short of a weight chunk.");
            got += n;
        }
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
        foreach (var buffer in _allHalfBuffers)
        {
            try { buffer.Dispose(); }
            catch { /* may already be disposed */ }
        }
        _allBuffers.Clear();
        _allHalfBuffers.Clear();
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
