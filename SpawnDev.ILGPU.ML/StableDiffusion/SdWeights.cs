using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.SafeTensors;

namespace SpawnDev.ILGPU.ML.StableDiffusion;

using HV = ArrayView1D<global::ILGPU.Half, Stride1D.Dense>;

/// <summary>How a checkpoint tensor is stored on the device.</summary>
public enum SdLayout
{
    /// <summary>As in the file (row-major).</summary>
    AsIs,
    /// <summary>A convolution weight [cout, cin, kH, kW] permuted to the gather layout [cout, kH * kW, cin] of the convolution GEMM (a 1x1 kernel is the same bytes either way).</summary>
    ConvGather,
    /// <summary>A 2-D tensor [rows, cols] transposed to [cols, rows] (OpenCLIP's text_projection is applied as x @ P; the linear kernels take [out, in]).</summary>
    Transpose,
}

/// <summary>
/// One model's tensors (UNet, text encoder or VAE) resident on the device as fp16, whatever the file stored (fp32, fp16 or bf16 are converted on the device as they stream in).
/// Names are the loader's INTERNAL names (the original LDM names with the component prefix removed, text encoders normalised to one naming). <see cref="Dispose"/> frees them.
/// </summary>
public sealed class SdWeights : IDisposable
{
    private readonly Dictionary<string, MemoryBuffer1D<global::ILGPU.Half, Stride1D.Dense>> _t = new();
    private readonly Dictionary<string, int[]> _shapes = new();

    public IEnumerable<string> Names => _shapes.Keys;
    public bool Has(string name) => _shapes.ContainsKey(name);
    public int[] Shape(string name) => _shapes.TryGetValue(name, out var s) ? s : throw new KeyNotFoundException($"weight '{name}' not loaded");

    /// <summary>The fp16 tensor <paramref name="name"/>.</summary>
    public HV this[string name] => _t.TryGetValue(name, out var b) ? b.View : throw new KeyNotFoundException($"weight '{name}' not loaded");

    /// <summary>The tensor if it exists (an optional bias).</summary>
    public HV? Opt(string name) => _t.TryGetValue(name, out var b) ? b.View : null;

    /// <summary>Device bytes held.</summary>
    public long Bytes => _t.Values.Sum(b => b.Length * 2L);

    internal void Add(string name, MemoryBuffer1D<global::ILGPU.Half, Stride1D.Dense> buffer, int[] shape)
    {
        if (_t.Remove(name, out var old)) old.Dispose();
        _t[name] = buffer; _shapes[name] = shape;
    }

    /// <summary>Hand tensor <paramref name="name"/> to another store under <paramref name="newName"/> (the buffer moves, nothing is copied).</summary>
    internal void MoveTo(string name, SdWeights other, string newName)
    {
        if (!_t.Remove(name, out var b)) throw new KeyNotFoundException(name);
        var shape = _shapes[name]; _shapes.Remove(name);
        other.Add(newName, b, shape);
    }

    internal void Remove(string name)
    {
        if (_t.Remove(name, out var b)) b.Dispose();
        _shapes.Remove(name);
    }

    /// <summary>
    /// Concatenate tensors along their first dimension into one new tensor (the q, k and v projections of an attention become one [3C, C] matrix: one GEMM instead of three)
    /// and drop the parts. Every part must exist; a missing part leaves everything as it was and returns false.
    /// </summary>
    internal async Task<bool> FuseAsync(Accelerator acc, string fused, params string[] parts)
    {
        foreach (var p in parts) if (!Has(p)) return false;
        var first = Shape(parts[0]);
        int rows = 0; long cols = 1;
        for (int i = 1; i < first.Length; i++) cols *= first[i];
        foreach (var p in parts) rows += Shape(p)[0];
        var buf = acc.Allocate1D<global::ILGPU.Half>(rows * cols);
        long at = 0;
        foreach (var p in parts)
        {
            var v = this[p];
            buf.View.SubView(at, v.Length).CopyFrom(v);
            at += v.Length;
        }
        var shape = (int[])first.Clone(); shape[0] = rows;
        await acc.SynchronizeAsync();               // the parts may only go once the copies ran (a browser batches them; Wasm runs them on workers)
        int row = 0;
        foreach (var p in parts) { int r = Shape(p)[0]; _fusedParts[p] = (fused, row, r); row += r; Remove(p); }
        Add(fused, buf, shape);
        return true;
    }

    private readonly Dictionary<string, (string Fused, int RowOffset, int Rows)> _fusedParts = new();

    /// <summary>
    /// Where a tensor of the file now lives: itself, or rows [RowOffset, RowOffset + Rows) of the fused tensor it went into (a LoRA for <c>attn1.to_k.weight</c> patches
    /// rows C..2C of <c>attn1.qkv.weight</c>). Null when the tensor is not held.
    /// </summary>
    internal (string Name, int RowOffset, int Rows)? Locate(string name)
    {
        if (_fusedParts.TryGetValue(name, out var f)) return (f.Fused, f.RowOffset, f.Rows);
        return Has(name) ? (name, 0, Shape(name)[0]) : null;
    }

    /// <summary>The file-level names this store answers for: the held tensors and the parts fused into them.</summary>
    internal IEnumerable<string> SourceNames => _shapes.Keys.Concat(_fusedParts.Keys);

    public void Dispose()
    {
        foreach (var b in _t.Values) b.Dispose();
        _t.Clear(); _shapes.Clear();
    }
}

/// <summary>
/// Streams a safetensors file once, front to back (no seeking: browser OPFS / HTTP streams work), sending every tensor the <c>route</c> function claims to its <see cref="SdWeights"/>
/// as fp16. fp16 tensors that keep their layout go straight from the stream into their device buffer; everything else (fp32, bf16, permuted or transposed) goes through one
/// reused device staging buffer and a conversion kernel, so the device never holds more than the fp16 weights plus the largest single tensor.
/// </summary>
internal sealed class SdTensorStreamer : IDisposable
{
    private readonly Accelerator _acc;
    private MemoryBuffer1D<uint, Stride1D.Dense>? _staging;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, HV, int, int, int>? _fromF32;
    private Action<Index1D, HV, HV, int, int, int>? _fromF16;
    private Action<Index1D, ArrayView1D<BFloat16, Stride1D.Dense>, HV, int, int, int>? _fromBf16;

    public SdTensorStreamer(Accelerator acc) => _acc = acc;

    // dst index i in the target layout -> source index. layout 0 as is, 1 conv gather ([co][t][ci] <- [co][ci][t]), 2 transpose ([c][r] <- [r][c]); a = cin or cols, b = taps or rows.
    // 32-bit: a launch is at most 2^31 elements (the largest SD tensor, SDXL's token table, is 63 M), and WGSL has no native 64-bit integers (emulated: slower to compile and to run).
    private static int SourceIndex(int i, int layout, int a, int b)
    {
        if (layout == 1)
        {
            int ci = i % a, t = i / a % b, co = i / (a * b);
            return (co * a + ci) * b + t;
        }
        if (layout == 2)
        {
            int r = i % b, c = i / b;       // dst [cols = a][rows = b]
            return r * a + c;
        }
        return i;
    }
    private static void FromF32Impl(Index1D i, ArrayView1D<float, Stride1D.Dense> src, HV dst, int layout, int a, int b)
        => dst[i] = PrecisionConvert.ConvertFromSingle<global::ILGPU.Half>(src[SourceIndex(i, layout, a, b)]);
    private static void FromF16Impl(Index1D i, HV src, HV dst, int layout, int a, int b)
        => dst[i] = src[SourceIndex(i, layout, a, b)];
    private static void FromBf16Impl(Index1D i, ArrayView1D<BFloat16, Stride1D.Dense> src, HV dst, int layout, int a, int b)
        => dst[i] = PrecisionConvert.ConvertFromSingle<global::ILGPU.Half>(PrecisionConvert.ConvertToSingle(src[SourceIndex(i, layout, a, b)]));

    /// <summary>A tensor's destination: the store, the internal name and the layout. A null store = a tiny tensor (a LoRA alpha) read to the host: <c>onScalar</c> gets its first value.</summary>
    public readonly record struct Target(SdWeights? Store, string Name, SdLayout Layout);

    /// <summary>
    /// Stream every tensor of <paramref name="tensors"/> (the parsed header, in file order) from <paramref name="stream"/> positioned just after the header (<paramref name="pos"/> = its
    /// absolute offset). <paramref name="route"/> returns the destination or null (skipped: read through, never uploaded). <paramref name="onBytes"/> reports the absolute stream position.
    /// </summary>
    public async Task StreamAsync(Stream stream, IReadOnlyList<SafeTensorInfo> tensors, long pos, Func<string, Target?> route, Action<long>? onBytes, CancellationToken ct,
        Action<string, float>? onScalar = null)
    {
        var ordered = tensors.OrderBy(t => t.DataOffset).ToList();
        long staging = 0;
        var targets = new Target?[ordered.Count];
        for (int i = 0; i < ordered.Count; i++)
        {
            var t = ordered[i];
            targets[i] = route(t.Name);
            if (targets[i] is { Store: not null } tg && (t.DType != "F16" || tg.Layout != SdLayout.AsIs)) staging = Math.Max(staging, t.DataLength);
            if (targets[i] is { Store: null } && t.DataLength > 64) throw new InvalidDataException($"tensor '{t.Name}' ({t.DataLength} bytes) is too big to read to the host");
            if (targets[i] != null && t.DType is not ("F16" or "F32" or "BF16"))
                throw new NotSupportedException($"tensor '{t.Name}' is {t.DType}: a Stable Diffusion checkpoint holds F32, F16 or BF16 weights");
        }
        if (staging > 0) _staging = _acc.Allocate1D<uint>((staging + 3) / 4);
        byte[]? skipBuf = null;
        for (int i = 0; i < ordered.Count; i++)
        {
            ct.ThrowIfCancellationRequested();
            var t = ordered[i];
            if (targets[i] is not { } tg)
            {
                continue;                                     // skipped: the gap is read through when the next claimed tensor comes
            }
            if (t.DataOffset > pos)
            {
                long gap = t.DataOffset - pos;
                if (stream.CanSeek) stream.Seek(gap, SeekOrigin.Current);
                else { skipBuf ??= new byte[1 << 20]; while (gap > 0) { int r = await stream.ReadAsync(skipBuf.AsMemory(0, (int)Math.Min(skipBuf.Length, gap)), ct); if (r <= 0) throw new EndOfStreamException(t.Name); gap -= r; } }
                pos = t.DataOffset;
            }
            long n = SafeTensorsFileReader.ElementCount(t);
            if (tg.Store == null)
            {
                var raw = new byte[t.DataLength];
                await stream.ReadExactlyAsync(raw, ct);
                float v = t.DType switch
                {
                    "F32" => BitConverter.ToSingle(raw, 0),
                    "F16" => (float)BitConverter.ToHalf(raw, 0),
                    _ => BitConverter.Int32BitsToSingle(BitConverter.ToUInt16(raw, 0) << 16),
                };
                onScalar?.Invoke(tg.Name, v);
                pos += t.DataLength;
                continue;
            }
            var shape = t.Shape;
            int layout = 0, a = 1, b = 1;
            int[] outShape = shape;
            if (tg.Layout == SdLayout.ConvGather && shape.Length == 4 && shape[2] * shape[3] > 1)
            {
                layout = 1; a = shape[1]; b = shape[2] * shape[3];
            }
            else if (tg.Layout == SdLayout.Transpose)
            {
                if (shape.Length != 2) throw new InvalidDataException($"'{t.Name}': a transposed tensor must be 2-D");
                layout = 2; a = shape[1]; b = shape[0]; outShape = new[] { shape[1], shape[0] };
            }
            var dst = _acc.Allocate1D<global::ILGPU.Half>(n);
            try
            {
                if (t.DType == "F16" && layout == 0) await Upload(stream, dst.View, n * 2, ct);
                else
                {
                    var st = _staging!.View;
                    switch (t.DType)
                    {
                        case "F32":
                            { var v = st.Cast<uint, float>().SubView(0, n); await Upload(stream, v, n * 4, ct); _fromF32 ??= _acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, HV, int, int, int>(FromF32Impl); _fromF32((int)n, v, dst.View, layout, a, b); break; }
                        case "F16":
                            { var v = st.Cast<uint, global::ILGPU.Half>().SubView(0, n); await Upload(stream, v, n * 2, ct); _fromF16 ??= _acc.LoadAutoGroupedStreamKernel<Index1D, HV, HV, int, int, int>(FromF16Impl); _fromF16((int)n, v, dst.View, layout, a, b); break; }
                        default:
                            { var v = st.Cast<uint, BFloat16>().SubView(0, n); await Upload(stream, v, n * 2, ct); _fromBf16 ??= _acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<BFloat16, Stride1D.Dense>, HV, int, int, int>(FromBf16Impl); _fromBf16((int)n, v, dst.View, layout, a, b); break; }
                    }
                }
            }
            catch { dst.Dispose(); throw; }
            // the next tensor overwrites the staging buffer: the conversion must have RUN (a flush is not enough on Wasm, whose dispatches wait for the worker pool)
            if (!(t.DType == "F16" && layout == 0)) await _acc.SynchronizeAsync();
            tg.Store.Add(tg.Name, dst, outShape);
            pos += t.DataLength;
            onBytes?.Invoke(pos);
        }
        await _acc.SynchronizeAsync();
    }

    private static async Task Upload<T>(Stream stream, ArrayView1D<T, Stride1D.Dense> dst, long bytes, CancellationToken ct) where T : unmanaged
    {
        // SpawnDev.ILGPU's stream copy: a JS-backed browser stream goes JS -> GPU (queue.writeBuffer) without the .NET heap; writeBuffer needs 4-byte multiples
        if (bytes % 4 == 0) { await dst.CopyFromStreamAsync(stream, cancellationToken: ct); return; }
        int es = System.Runtime.CompilerServices.Unsafe.SizeOf<T>();
        long count = bytes / es;
        var buf = new byte[Math.Min(bytes, 1 << 20)];
        var staging = new T[Math.Min(count, buf.Length / es)];
        for (long done = 0; done < count;)
        {
            int k = (int)Math.Min(staging.Length, count - done);
            await stream.ReadExactlyAsync(buf.AsMemory(0, k * es), ct);
            System.Runtime.InteropServices.MemoryMarshal.Cast<byte, T>(buf.AsSpan(0, k * es)).CopyTo(staging);
            dst.SubView(done, k).CopyFromCPU(k == staging.Length ? staging : staging[..k]);
            done += k;
        }
    }

    public void Dispose()
    {
        _staging?.Dispose(); _staging = null;
    }
}
