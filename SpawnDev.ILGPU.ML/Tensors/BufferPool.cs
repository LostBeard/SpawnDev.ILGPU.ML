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
    // CUDA-graph capture: an UNNAMED Rent is never Returned (Return keys on name), so it would allocate a fresh
    // buffer every call → a cuMemAlloc mid-capture (illegal). In capture-mode we hand unnamed Rents a STABLE
    // per-forward-ordinal slot (allocated in the WARM pass, reused thereafter; reset on ForwardGeneration). The
    // op's GPU write into the buffer is captured normally; only the buffer address must stay fixed for replay.
    private readonly List<MemoryBuffer1D<float, Stride1D.Dense>> _captureUnnamedSlots = new();
    private int _captureUnnamedNext;
    private long _captureUnnamedGen = -1;
    // fp16 weight buffers (ILGPU.Half) — half the bytes of the fp32 _allBuffers. Tracked separately for
    // disposal (different element type). See AllocateHalfWeightFromStreamAsync.
    private readonly List<MemoryBuffer1D<global::ILGPU.Half, Stride1D.Dense>> _allHalfBuffers = new();
    // fp16 ACTIVATION pool (mixed-precision activations): bucketed Half buffers for graph intermediates,
    // the half-bytes counterpart to the fp32 _buckets/_namedBuffers. Returned buffers reuse by size bucket.
    private readonly Dictionary<int, Stack<MemoryBuffer1D<global::ILGPU.Half, Stride1D.Dense>>> _halfBuckets = new();
    private readonly Dictionary<string, MemoryBuffer1D<global::ILGPU.Half, Stride1D.Dense>> _halfNamedBuffers = new();

    /// <summary>Total fp16 (Half) buffers this pool has allocated (weights + activations).</summary>
    public int AllocatedHalfBufferCount => _allHalfBuffers.Count;

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

    // ── Opt-in peak instrumentation (default OFF = zero cost). Measures whether the pool's high-water is
    //    the SUM of intermediates (reuse failure → memory planning fixes it) or a genuinely large LIVE set
    //    (→ needs fp16 activations / tiling). Static so a console measurement reads it without wiring. ──
    /// <summary>Enable peak-bytes tracking on Rent (diagnostic; default false).</summary>
    public static bool TrackPeaks = false;
    /// <summary>Max total fp32+fp16 bytes ever allocated by ANY pool at one Rent (sum of all live buffers).</summary>
    public static long PeakTotalBytes;
    /// <summary>Max simultaneously-RENTED (named/live, not yet Returned) bytes at one Rent — the true working set.</summary>
    public static long PeakLiveBytes;
    /// <summary>Reset the peak counters before a measured run.</summary>
    public static void ResetPeaks() { PeakTotalBytes = 0; PeakLiveBytes = 0; PeakLiveSnapshot = null; }

    /// <summary>When TrackLivePeakComposition is set, holds the (name, bytes, isHalf) of every live buffer at the
    /// moment the LIVE peak was last raised — i.e. exactly what dominates the working set. Diagnostic only.</summary>
    public static List<(string name, long bytes, bool isHalf)>? PeakLiveSnapshot;
    /// <summary>Capture <see cref="PeakLiveSnapshot"/> on each new LIVE peak (extra cost; default off).</summary>
    public static bool TrackLivePeakComposition = false;

    private void UpdatePeaks()
    {
        if (!TrackPeaks) return;
        long total = 0;
        foreach (var b in _allBuffers) total += b.LengthInBytes;
        foreach (var b in _allHalfBuffers) total += b.LengthInBytes;
        long live = 0;
        foreach (var kv in _namedBuffers) live += kv.Value.LengthInBytes;
        foreach (var kv in _halfNamedBuffers) live += kv.Value.LengthInBytes;
        if (total > PeakTotalBytes) PeakTotalBytes = total;
        if (live > PeakLiveBytes)
        {
            // Throttle the (O(buffers)) composition snapshot: only re-capture when the peak jumps by >2 MiB,
            // else thousands of micro-peak-raises each trigger a full snapshot and dominate runtime.
            bool bigJump = live - PeakLiveBytes > 2L * 1024 * 1024;
            PeakLiveBytes = live;
            if (TrackLivePeakComposition && (bigJump || PeakLiveSnapshot == null))
            {
                var snap = new List<(string, long, bool)>();
                foreach (var kv in _namedBuffers) snap.Add((kv.Key, kv.Value.LengthInBytes, false));
                foreach (var kv in _halfNamedBuffers) snap.Add((kv.Key, kv.Value.LengthInBytes, true));
                PeakLiveSnapshot = snap;
            }
        }
    }

    public BufferPool(Accelerator accelerator) => _accelerator = accelerator;

    /// <summary>Rent a tensor with the given shape. May reuse a pooled buffer.</summary>
    private readonly Dictionary<string, MemoryBuffer1D<float, Stride1D.Dense>> _namedBuffers = new();

    public Tensor Rent(int[] shape, string? name = null)
    {
        int count = TensorHelpers.ElementCount(shape);
        int bucketSize = NextPowerOf2(count);

        // CUDA-graph capture: unnamed Rents get a stable per-forward slot (see _captureUnnamedSlots). Populated
        // during the warm pass (UseCaptureParamSlots set, SuppressDrains not) so no allocation on the capture pass.
        if (name == null && Graph.GraphExecutor.UseCaptureParamSlots)
        {
            long cgen = Graph.GraphExecutor.ForwardGeneration;
            if (cgen != _captureUnnamedGen) { _captureUnnamedGen = cgen; _captureUnnamedNext = 0; }
            int ci = _captureUnnamedNext++;
            MemoryBuffer1D<float, Stride1D.Dense> cbuf;
            if (ci < _captureUnnamedSlots.Count && _captureUnnamedSlots[ci].Length >= bucketSize)
                cbuf = _captureUnnamedSlots[ci];
            else
            {
                cbuf = _accelerator.Allocate1D<float>(bucketSize);
                if (ci < _captureUnnamedSlots.Count) { _captureUnnamedSlots[ci].Dispose(); _captureUnnamedSlots[ci] = cbuf; }
                else _captureUnnamedSlots.Add(cbuf);
            }
            UpdatePeaks();
            return new Tensor(cbuf.View.SubView(0, count), shape, null);
        }

        if (_buckets.TryGetValue(bucketSize, out var stack) && stack.Count > 0)
        {
            var buffer = stack.Pop();
            var tensor = new Tensor(buffer.View, shape, name);
            if (name != null) _namedBuffers[name] = buffer;
            UpdatePeaks();
            return tensor;
        }

        // CUDA-graph capture diagnostic: a pool allocation while SuppressDrains means the warm passes did NOT
        // prime this size-bucket, so the capture pass is about to cuMemAlloc (illegal mid-capture → native crash).
        // Naming it here (survives the crash) reveals which Rent site is under-primed. Inert in production.
        if (Graph.GraphExecutor.SuppressDrains && Graph.GraphExecutor.CaptureTraceFile != null)
        {
            try { System.IO.File.AppendAllText(Graph.GraphExecutor.CaptureTraceFile,
                $"   -> POOL-ALLOC '{name}' count={count} bucket={bucketSize}  (capture priming gap)\n"); } catch { }
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
        UpdatePeaks();
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
        // NEVER free during a graph-capture window: cuMemFree while CUDA stream capture is active
        // corrupts the context (native 0xC0000005 at the NEXT free - SD-Turbo UNet capture under
        // VRAM pressure, 2026-07-03). Reclaiming nothing makes the OOM throw cleanly instead; the
        // capture wrapper falls back to the direct forward.
        if (Graph.GraphExecutor.SuppressDrains) return 0;
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
        // Also reclaim returned-not-live fp16 activation buffers.
        foreach (var stack in _halfBuckets.Values)
            while (stack.Count > 0)
            {
                var buf = stack.Pop();
                freed += buf.LengthInBytes;
                _allHalfBuffers.Remove(buf);
                buf.Dispose();
            }
        _halfBuckets.Clear();
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

    /// <summary>Transfer a live (rented) fp32 buffer's pool ownership from one name to another — the zero-copy
    /// Reshape ownership-handoff (the executor rebinds a single-consumer input's buffer to the reshape output
    /// instead of renting+copying). After this, <see cref="Return"/>(tensor named <paramref name="newName"/>)
    /// frees the buffer. No-op if <paramref name="oldName"/> isn't a live named buffer. Returns true if moved.</summary>
    public bool Rename(string oldName, string newName)
    {
        if (oldName == newName) return true;
        if (!_namedBuffers.TryGetValue(oldName, out var buf)) return false;
        _namedBuffers.Remove(oldName);
        _namedBuffers[newName] = buf;   // overwrites any prior newName entry (the reshape never pre-rents one)
        return true;
    }

    /// <summary>Rent an fp16 (<see cref="ILGPU.Half"/>) ACTIVATION tensor — half the bytes of <see cref="Rent"/>.
    /// Reuses a pooled Half buffer of the right size bucket when free; else allocates (with the same
    /// under-pressure reclaim as the fp32 path). The mixed-precision-activation counterpart to Rent — the
    /// executor keeps heavy intermediates fp16 and converts at fp32 boundaries
    /// (Plans/fp16-bf16-mixed-precision-activations-2026-06-16.md).</summary>
    public HalfTensor RentHalf(int[] shape, string? name = null)
    {
        int count = TensorHelpers.ElementCount(shape);
        int bucketSize = NextPowerOf2(count);

        if (_halfBuckets.TryGetValue(bucketSize, out var stack) && stack.Count > 0)
        {
            var buffer = stack.Pop();
            if (name != null) _halfNamedBuffers[name] = buffer;
            UpdatePeaks();
            return new HalfTensor(buffer.View, shape, name);
        }

        var newBuffer = _accelerator.AllocateWithReclaim(
            () => _accelerator.Allocate1D<global::ILGPU.Half>(bucketSize),
            DisposeBucketedBuffers,
            reclaimed => $"GPU out of memory renting fp16 '{name}' (+{bucketSize * 2L / 1048576}MB) after reclaiming " +
                         $"{reclaimed / 1048576}MB of pooled buffers; {_allHalfBuffers.Count} fp16 + {_allBuffers.Count} fp32 live. Exceeds VRAM.");
        _allHalfBuffers.Add(newBuffer);
        if (name != null) _halfNamedBuffers[name] = newBuffer;
        UpdatePeaks();
        return new HalfTensor(newBuffer.View, shape, name);
    }

    /// <summary>Return an fp16 (Half) tensor's buffer to the pool for reuse by name.</summary>
    public void ReturnHalf(HalfTensor tensor)
    {
        var name = tensor.Name;
        if (name != null && _halfNamedBuffers.TryGetValue(name, out var buffer))
        {
            _halfNamedBuffers.Remove(name);
            int bucketSize = (int)buffer.Length;
            if (!_halfBuckets.TryGetValue(bucketSize, out var stack))
            {
                stack = new Stack<MemoryBuffer1D<global::ILGPU.Half, Stride1D.Dense>>();
                _halfBuckets[bucketSize] = stack;
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

    /// <summary>
    /// Stream a QUANTIZED tensor's RAW BYTES (Q4_K/Q6_K/Q8_0/...) from <paramref name="byteOffset"/> in a
    /// seekable <paramref name="stream"/> straight to a GPU byte buffer — the quantized sibling of
    /// <see cref="AllocatePermanentFromStreamAsync"/> (which is float-only and throws on quantized dtypes).
    /// The bytes stay COMPRESSED on the GPU; FusedDequantMatMul/Gather decode blocks in-register. Peak CPU is
    /// one 4 MB chunk, never the whole tensor — this is what makes a 7 GB GGUF loadable (a single byte[] caps
    /// at ~2 GB). The buffer is padded to a 4-byte multiple (fused kernels read packed int32 words; WebGPU
    /// requires 4-byte buffer sizes). Returns the buffer so the caller owns its lifetime (mirrors the existing
    /// quantized-upload path, whose buffers the session disposes).
    /// </summary>
    public async Task<MemoryBuffer1D<byte, Stride1D.Dense>> AllocateQuantizedBytesFromStreamAsync(
        Stream stream, long byteOffset, int byteLength, CancellationToken ct = default)
    {
        // Guard the range BEFORE allocating/copying: a bad byteOffset/byteLength (e.g. an upstream quant
        // byte-size miscalc producing a negative or over-large length) otherwise hits an uncatchable ILGPU
        // "Index/Extent out of bounds" FailFast inside CopyFromStreamRawAsync. Surface it as a clear,
        // catchable error naming the actual values instead.
        long streamLen = stream.CanSeek ? stream.Length : -1;
        if (byteLength < 0 || byteOffset < 0 || (streamLen >= 0 && byteOffset + (long)byteLength > streamLen))
            throw new InvalidDataException(
                $"Quantized tensor byte range out of bounds: byteOffset={byteOffset}, byteLength={byteLength}, streamLength={streamLen}.");

        int padded = (byteLength + 3) & ~3;
        var buffer = _accelerator.Allocate1D<byte>(padded);
        if (byteLength == 0) return buffer;

        stream.Seek(byteOffset, SeekOrigin.Begin);
        const int CHUNK = 4 * 1024 * 1024; // 4 MB (4-byte aligned, preserves the zero-copy gate below)
        // CopyFromStreamAsync auto-selects the ZERO-COPY browser path: when `stream` is an IJSReadStream
        // (TorrentReadStream / OPFS) and the (offset,length,chunk) are 4-byte aligned, each chunk's Uint8Array
        // goes STRAIGHT to the GPU via queue.writeBuffer (CopyFromJS) — never entering the .NET/WASM managed
        // heap (the whole point on a 7 GB model). 4-aligned weights (the bulk; K-quant block sizes are mostly
        // even) take that path; an odd-length tail gracefully falls back to the managed read. Desktop always
        // takes the managed path. Replaces the old hand-rolled ReadExactAsync + CopyFromCPU marshal loop.
        await buffer.View.SubView(0, byteLength).CopyFromStreamAsync(stream, CHUNK, ct).ConfigureAwait(false);
        return buffer;
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
        foreach (var buffer in _captureUnnamedSlots)
        {
            try { buffer.Dispose(); }
            catch { /* may already be disposed */ }
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
