using System.Text.Json;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.Wan;

namespace SpawnDev.ILGPU.ML.Trellis2;

using BV = ArrayView1D<BFloat16, Stride1D.Dense>;

/// <summary>
/// A transformer's weights read from one safetensors file, larger than the card may hold: the tensors are split into numbered blocks (a block classifier given by the
/// model) and the rest; the rest and as many blocks as <c>residentBytes</c> allows stay on the device, the remaining blocks are read from the file for every forward
/// pass (one at a time, freed after they ran). A prefix in front of the model's names (ComfyUI's all-in-one checkpoints: <c>model.diffusion_model.</c>) is found from
/// an anchor tensor and removed. 8-bit float (F8_E4M3) matrices, unscaled or with ComfyUI's per-matrix <c>scale_weight</c> (fp8_scaled files), stay 8-bit and are decoded into one shared bf16 scratch just before their GEMM; 8-bit
/// vectors are made bf16 when they load. An optional per-load adapter rewrites each loaded set (fusing, renaming) before use.
/// </summary>
public sealed class StreamedModelWeights : IDisposable
{
    private readonly Accelerator _acc;
    private readonly IModelSource _source;
    private readonly string _repo, _file, _prefix;
    private readonly Func<string, int> _blockOf;
    private readonly Func<GpuWeights, IDictionary<string, float>, Task>? _adapt;                 // the model's per-load adapter, given the 8-bit scales by name (it may fuse parts that share one)
    private readonly GpuWeights _base;
    private readonly GpuWeights?[] _resident;
    private MemoryBuffer1D<BFloat16, Stride1D.Dense>? _scratch;
    // ComfyUI's "fp8_scaled" files: one fp32 scale_weight per 8-bit matrix (the weight is value * scale), by the matrix's prefix-free name; filled as sets load
    private readonly System.Collections.Concurrent.ConcurrentDictionary<string, float> _scales;

    public int BlockCount { get; }
    /// <summary>Device bytes held by the weights that stay (the non-block tensors and the resident blocks), plus the decode scratch.</summary>
    public long ResidentBytes => _base.Bytes + _resident.Sum(r => r?.Bytes ?? 0) + (_scratch?.Length ?? 0) * 2;
    /// <summary>How many blocks stay on the device; the others are read from the file on every forward pass.</summary>
    public int ResidentBlocks => _resident.Count(r => r != null);
    /// <summary>The bytes a forward pass reads from the file (the streamed blocks).</summary>
    public long StreamedBytesPerForward { get; private set; }
    /// <summary>The non-block tensors.</summary>
    public GpuWeights Base => _base;
    /// <summary>Wall time the caller waited for streamed blocks (file read not hidden by the prefetch, plus the upload), and how many loads; for measuring the streaming cost.</summary>
    public double StreamSeconds { get; set; }
    public int StreamedLoads { get; set; }
    /// <summary>Time spent reading block images from the file (on the caller's path or the read-ahead's), and uploading them to the device, summed; for finding the bottleneck.</summary>
    public double ReadSeconds { get; set; }
    public double UploadSeconds { get; set; }
    /// <summary>
    /// Read the next streamed block from the file while the caller's GPU work on this one runs (default on): the disk read hides behind the compute and only the upload
    /// remains in the caller's path.
    /// </summary>
    public bool Prefetch { get; set; } = true;
    /// <summary>
    /// Host RAM (bytes) for keeping streamed blocks' file images between forward passes (default 0): a cached block skips the disk on every later pass (it is still uploaded).
    /// Blocks are cached in order while the budget lasts.
    /// </summary>
    public long HostCacheBytes { get; set; }
    /// <summary>Host RAM the cache holds now.</summary>
    public long HostCacheUsed => _hostCacheUsed;

    // per streamed block: its tensors in the file (names with the prefix, absolute offsets), the file's metadata (kept in every block image: the quantization check reads it)
    private List<SafeTensors.SafeTensorInfo>[] _blockTensors = Array.Empty<List<SafeTensors.SafeTensorInfo>>();
    private Dictionary<string, string> _metadata = new();
    private readonly Dictionary<int, byte[]> _hostCache = new();
    private long _hostCacheUsed;
    private (int Block, Task<byte[]> Image)? _pending;
    /// <summary>
    /// Every tensor name the file's header declares (prefix-free), including ones never loaded: ComfyUI's empty marker tensors (<c>__index_timestep_zero__</c> in Qwen-Image
    /// Edit 2511 files) tell the model how to run.
    /// </summary>
    public IReadOnlySet<string> HeaderNames { get; private set; } = new HashSet<string>();

    // not weights: ComfyUI's empty markers (scaled_fp8, __index_timestep_zero__ ...) and the per-layer quantization json blobs (U8 "comfy_quant") of "fp8mixed" files
    private static bool IsNotAWeight(string prefixFreeName)
        => prefixFreeName == "scaled_fp8" || prefixFreeName.StartsWith("__", StringComparison.Ordinal) || prefixFreeName.EndsWith(".comfy_quant", StringComparison.Ordinal);

    /// <summary>
    /// ComfyUI's mixed-precision files carry <c>_quantization_metadata</c> ({"layers": {name: {"format", "full_precision_matrix_mult"}}}). This loader runs every 8-bit matrix as
    /// value * weight_scale with full-precision activations, which is exactly "float8_e4m3fn" with full_precision_matrix_mult true; any other layer format is refused rather
    /// than run differently from the reference.
    /// </summary>
    private static void CheckQuantization(string file, IReadOnlyDictionary<string, string> metadata)
    {
        if (!metadata.TryGetValue("_quantization_metadata", out var json)) return;
        using var doc = JsonDocument.Parse(json);
        if (!doc.RootElement.TryGetProperty("layers", out var layers)) return;
        foreach (var layer in layers.EnumerateObject())
        {
            string format = layer.Value.TryGetProperty("format", out var f) ? f.GetString() ?? "" : "";
            bool full = layer.Value.TryGetProperty("full_precision_matrix_mult", out var fp) && fp.ValueKind == JsonValueKind.True;
            if (format != "float8_e4m3fn" || !full)
                throw new NotSupportedException($"'{file}': layer {layer.Name} is quantized as '{format}' (full precision matmul: {full}); only float8_e4m3fn weights with full-precision activations are supported");
        }
    }

    private StreamedModelWeights(Accelerator acc, IModelSource source, string repo, string file, string prefix, Func<string, int> blockOf, int blockCount, Func<GpuWeights, IDictionary<string, float>, Task>? adapt, GpuWeights baseWeights,
        System.Collections.Concurrent.ConcurrentDictionary<string, float> scales)
    {
        _acc = acc; _source = source; _repo = repo; _file = file; _prefix = prefix; _blockOf = blockOf; BlockCount = blockCount; _adapt = adapt; _base = baseWeights; _scales = scales;
        _resident = new GpuWeights?[blockCount];
    }

    /// <summary>
    /// Open <paramref name="file"/>: <paramref name="anchor"/> (a tensor name the model always has) finds the prefix; <paramref name="blockOf"/> maps a prefix-free name to its block
    /// (0 .. <paramref name="blockCount"/> - 1) or -1; blocks are kept in order while they fit in <paramref name="residentBytes"/> (as stored), the rest streamed.
    /// </summary>
    public static Task<StreamedModelWeights> OpenAsync(Accelerator accelerator, IModelSource source, string repo, string file, string anchor, Func<string, int> blockOf, int blockCount,
        long residentBytes, Func<GpuWeights, Task>? adapt = null, CancellationToken ct = default)
        => OpenWithScalesAsync(accelerator, source, repo, file, anchor, blockOf, blockCount, residentBytes, adapt == null ? null : (w, _) => adapt(w), ct);

    /// <summary>
    /// The same with an adapter that also sees the 8-bit matrices' scales by name (filled before it runs) and may change them: a model fusing parts (q | k | v) may only fuse
    /// 8-bit parts that share one scale, and then gives the fused matrix that scale.
    /// </summary>
    public static async Task<StreamedModelWeights> OpenWithScalesAsync(Accelerator accelerator, IModelSource source, string repo, string file, string anchor, Func<string, int> blockOf, int blockCount,
        long residentBytes, Func<GpuWeights, IDictionary<string, float>, Task>? adapt, CancellationToken ct = default)
    {
        string prefix; var blockBytes = new long[blockCount]; var headerNames = new HashSet<string>();
        var blockTensors = new List<SafeTensors.SafeTensorInfo>[blockCount]; Dictionary<string, string> metadata;
        await using (var s = await source.OpenAsync(repo, file, ct))
        {
            if (!s.CanSeek) throw new NotSupportedException($"'{file}': the blocks are read by offset; the model source must give a seekable stream");
            var lenBytes = new byte[8]; await s.ReadExactlyAsync(lenBytes, ct);
            long headerSize = BitConverter.ToInt64(lenBytes);
            if (headerSize <= 0 || headerSize > 100_000_000) throw new InvalidDataException($"'{file}': invalid SafeTensors header size {headerSize}");
            var header = new byte[headerSize]; await s.ReadExactlyAsync(header, ct);
            var (tensors, meta) = SafeTensors.SafeTensorsFileReader.ParseHeader(file, header, 8 + headerSize, s.Length - 8 - headerSize);
            metadata = meta;
            for (int k = 0; k < blockCount; k++) blockTensors[k] = new List<SafeTensors.SafeTensorInfo>();
            var a = tensors.FirstOrDefault(t => t.Name == anchor || t.Name.EndsWith("." + anchor, StringComparison.Ordinal))
                ?? throw new InvalidDataException($"'{file}' does not hold the expected model ({anchor})");
            prefix = a.Name[..^anchor.Length];
            foreach (var t in tensors)
            {
                if (!t.Name.StartsWith(prefix, StringComparison.Ordinal)) continue;
                headerNames.Add(t.Name[prefix.Length..]);
                if (IsNotAWeight(t.Name[prefix.Length..])) continue;
                int k = blockOf(t.Name[prefix.Length..]);
                if (k >= 0) { blockBytes[k] += t.DataLength; blockTensors[k].Add(t); }
            }
        }
        var scales = new System.Collections.Concurrent.ConcurrentDictionary<string, float>();
        Func<CancellationToken, Task<Stream>> open = c => source.OpenAsync(repo, file, c);
        var baseW = await LoadAsync(accelerator, open, file, prefix, n => blockOf(n) < 0, adapt, scales, ct);
        var w = new StreamedModelWeights(accelerator, source, repo, file, prefix, blockOf, blockCount, adapt, baseW, scales) { HeaderNames = headerNames, _blockTensors = blockTensors, _metadata = metadata };
        try
        {
            long budget = residentBytes, streamed = 0;
            for (int k = 0; k < blockCount; k++)
            {
                if (blockBytes[k] <= budget) { int kk = k; w._resident[k] = await LoadAsync(accelerator, open, file, prefix, n => blockOf(n) == kk, adapt, scales, ct); budget -= blockBytes[k]; }
                else { budget = 0; streamed += blockBytes[k]; }                       // the resident blocks are a prefix, the streamed ones the tail
            }
            w.StreamedBytesPerForward = streamed;
            return w;
        }
        catch { w.Dispose(); throw; }
    }

    private static async Task<GpuWeights> LoadAsync(Accelerator acc, Func<CancellationToken, Task<Stream>> open, string file, string prefix, Func<string, bool> keep, Func<GpuWeights, IDictionary<string, float>, Task>? adapt,
        System.Collections.Concurrent.ConcurrentDictionary<string, float> scales, CancellationToken ct)
    {
        await using var stream = await open(ct);
        using var empty = JsonDocument.Parse("{}");
        var w = await GpuWeights.LoadAsync(acc, stream, empty.RootElement, file, n => n.StartsWith(prefix, StringComparison.Ordinal) && !IsNotAWeight(n[prefix.Length..]) && keep(n[prefix.Length..]), ct);
        try
        {
            CheckQuantization(file, w.Metadata);
            foreach (var n in w.Names.ToList())
            {
                string to = n[prefix.Length..];
                if (to != n) w.Rename(n, to);
            }
            // fp8_scaled: scale_weight (fp32 scalar) becomes the matrix's decode scale; scale_input (an activation scale for fp8 matmuls) is not used, the activations stay fp32.
            // ComfyUI's mixed-precision ("fp8mixed") files name them weight_scale / input_scale (full-precision matmul, checked by CheckQuantization): the same meaning.
            foreach (var n in w.Names.ToList())
            {
                if (n.EndsWith(".scale_input", StringComparison.Ordinal) || n.EndsWith(".input_scale", StringComparison.Ordinal)) { await acc.SynchronizeAsync(); w.Remove(n)?.Dispose(); continue; }
                string? suffix = n.EndsWith(".scale_weight", StringComparison.Ordinal) ? ".scale_weight" : n.EndsWith(".weight_scale", StringComparison.Ordinal) ? ".weight_scale" : null;
                if (suffix == null) continue;
                if (!w.IsF32(n)) throw new InvalidDataException($"'{file}': {n} is not an fp32 scale");
                scales[n[..^suffix.Length] + ".weight"] = (await w.F32(n).CopyToHostAsync())[0];
                await acc.SynchronizeAsync(); w.Remove(n)?.Dispose();
            }
            foreach (var to in w.Names.ToList())
            {
                if (w.IsF8(to) && w.Shape(to).Length == 1)
                {
                    var dst = acc.Allocate1D<BFloat16>(w.Shape(to)[0]);
                    WanComfyFormat.DecodeF8E4M3(acc, w.F8Words(to), dst.View, scales.TryGetValue(to, out var vs) ? vs : 1f);
                    var old = w.SetBF16(to, dst, w.Shape(to));
                    await acc.SynchronizeAsync(); old?.Dispose();
                }
                else if (w.IsF16(to) && w.Shape(to).Length == 1)
                {
                    // fp16 biases / norm scales (ComfyUI's fp8_scaled FLUX.1 files): the kernels read bf16 vectors, so the vector is held as bf16 hi (name) + bf16 lo (name.lo),
                    // lo = f16 - hi: exact (an fp16 value has 3 more mantissa bits than bf16; the remainder fits bf16). A model that reads name.lo keeps the fp16 value exactly.
                    int n = w.Shape(to)[0];
                    var hi = acc.Allocate1D<BFloat16>(n); var lo = acc.Allocate1D<BFloat16>(n);
                    WanComfyFormat.ToBf16HiLo(acc, w.F16(to), hi.View, lo.View);
                    var old = w.SetBF16(to, hi, w.Shape(to));
                    w.SetBF16(to + ".lo", lo, w.Shape(to));
                    await acc.SynchronizeAsync(); old?.Dispose();
                }
            }
            if (adapt != null) await adapt(w, scales);
            // an 8-bit matrix without its scale in a file that has scales would decode silently wrong (an adapter that fused scaled parts, say)
            if (!scales.IsEmpty)
                foreach (var m in w.Names.Where(w.IsF8))
                    if (w.Shape(m).Length >= 2 && !scales.ContainsKey(m)) throw new NotSupportedException($"'{file}': {m} is an 8-bit matrix without a scale_weight in a scaled file");
            return w;
        }
        catch { w.Dispose(); throw; }
    }

    // the streamed block after k in forward order, wrapping to the first streamed block (the next pass starts there); null when nothing streams
    private int? NextStreamed(int k)
    {
        for (int i = 1; i <= BlockCount; i++) { int b = (k + i) % BlockCount; if (_resident[b] == null) return b; }
        return null;
    }

    /// <summary>
    /// Block <paramref name="k"/> as a self-contained safetensors image in host memory: a header naming its tensors (the file's own names and metadata) and their bytes,
    /// read from the file by offset. The loader then runs on it exactly as on the file.
    /// </summary>
    private async Task<byte[]> ReadBlockImageAsync(int k, CancellationToken ct)
    {
        var list = _blockTensors[k];
        // the header, written directly (trim-safe: no reflection serializer)
        long off = 0;
        var buffer = new System.Buffers.ArrayBufferWriter<byte>();
        using (var jw = new Utf8JsonWriter(buffer))
        {
            jw.WriteStartObject();
            if (_metadata.Count > 0)
            {
                jw.WriteStartObject("__metadata__");
                foreach (var (mk, mv) in _metadata) jw.WriteString(mk, mv);
                jw.WriteEndObject();
            }
            foreach (var t in list)
            {
                jw.WriteStartObject(t.Name);
                jw.WriteString("dtype", t.DType);
                jw.WriteStartArray("shape"); foreach (var d in t.Shape) jw.WriteNumberValue(d); jw.WriteEndArray();
                jw.WriteStartArray("data_offsets"); jw.WriteNumberValue(off); jw.WriteNumberValue(off + t.DataLength); jw.WriteEndArray();
                jw.WriteEndObject();
                off += t.DataLength;
            }
            jw.WriteEndObject();
        }
        var json = buffer.WrittenSpan.ToArray();
        int hlen = (json.Length + 7) / 8 * 8;
        long total = 8 + hlen + off;
        if (total > Array.MaxLength) throw new NotSupportedException($"'{_file}': block {k} is {total} bytes, larger than one host array");
        var image = new byte[total];
        BitConverter.TryWriteBytes(image.AsSpan(0, 8), (long)hlen);
        json.CopyTo(image, 8);
        for (int i = 8 + json.Length; i < 8 + hlen; i++) image[i] = (byte)' ';
        var rd = System.Diagnostics.Stopwatch.StartNew();
        await using var s = await _source.OpenAsync(_repo, _file, ct);
        long pos = 8 + hlen;
        foreach (var t in list)
        {
            s.Seek(t.DataOffset, SeekOrigin.Begin);
            await s.ReadExactlyAsync(image.AsMemory((int)pos, (int)t.DataLength), ct);
            pos += t.DataLength;
        }
        lock (_hostCache) ReadSeconds += rd.Elapsed.TotalSeconds;
        return image;
    }

    /// <summary>Block <paramref name="k"/>'s tensors: the resident set, or a fresh load from the file that the caller disposes after the device has run it (<paramref name="streamed"/> true).</summary>
    public async Task<(GpuWeights Weights, bool Streamed)> BlockAsync(int k, CancellationToken ct = default)
    {
        if (_retired.Count > 0) { await _acc.SynchronizeAsync(); foreach (var old in _retired) old.Dispose(); _retired.Clear(); }
        if (_resident[k] is { } r) return (r, false);
        var sw = System.Diagnostics.Stopwatch.StartNew();
        byte[] image;
        if (_hostCache.TryGetValue(k, out var cached)) image = cached;
        else if (_pending is { } p && p.Block == k) { image = await p.Image; _pending = null; }
        else image = await ReadBlockImageAsync(k, ct);
        if (!_hostCache.ContainsKey(k) && _hostCacheUsed + image.Length <= HostCacheBytes) { _hostCache[k] = image; _hostCacheUsed += image.Length; }
        if (Prefetch && NextStreamed(k) is int next && next != k && !_hostCache.ContainsKey(next) && (_pending == null || _pending.Value.Block != next))
        {
            if (_pending is { } stale) _ = stale.Image.ContinueWith(t => _ = t.Exception, TaskScheduler.Default);   // a prefetch the caller skipped: observed, dropped
            _pending = (next, Task.Run(() => ReadBlockImageAsync(next, CancellationToken.None)));
        }
        var up = System.Diagnostics.Stopwatch.StartNew();
        var loaded = await LoadAsync(_acc, _ => Task.FromResult<Stream>(new MemoryStream(image, writable: false)), _file, _prefix, n => _blockOf(n) == k, _adapt, _scales, ct);
        await _acc.SynchronizeAsync();                                                            // the upload's own time, not a later kernel's
        UploadSeconds += up.Elapsed.TotalSeconds;
        StreamSeconds += sw.Elapsed.TotalSeconds; StreamedLoads++;
        return (loaded, true);
    }

    /// <summary>A matrix of <paramref name="w"/> as bf16: the stored tensor, or (8-bit floats) its decode into the shared scratch, valid until the next call.</summary>
    public BV Matrix(GpuWeights w, string name)
    {
        if (!w.IsF8(name)) return w.BF16(name);
        long count = 1; foreach (int d in w.Shape(name)) count *= d;
        var dst = Scratch(count);
        WanComfyFormat.DecodeF8E4M3(_acc, w.F8Words(name), dst, _scales.TryGetValue(name, out var sc) ? sc : 1f);      // unscaled 8-bit files (ComfyUI's e4m3fn checkpoints): 1
        return dst;
    }

    // the shared bf16 decode scratch, at least count long. Matrix / MatrixExact are synchronous and their callers queue GEMMs on the scratch, so a scratch that is too small is
    // retired, not disposed (work queued on it may still be pending: browser backends cannot drain synchronously), and freed at the next async drain (BlockAsync) or Dispose
    private ArrayView1D<BFloat16, Stride1D.Dense> Scratch(long count)
    {
        if (_scratch == null || _scratch.Length < count)
        {
            if (_scratch != null) _retired.Add(_scratch);
            _scratch = _acc.Allocate1D<BFloat16>(count);
        }
        return _scratch.View.SubView(0, count);
    }
    private readonly List<MemoryBuffer1D<BFloat16, Stride1D.Dense>> _retired = new();

    /// <summary>True when the file is fp8_scaled (its 8-bit matrices carry scales): <see cref="MatrixExact"/> + <see cref="ActivationScale"/> then matter.</summary>
    public bool HasScales => !_scales.IsEmpty;

    /// <summary>
    /// A matrix of <paramref name="w"/> as bf16 with an 8-bit matrix decoded WITHOUT its scale (exact: every F8_E4M3 value is a bf16 value), paired with <see cref="ActivationScale"/>: the
    /// caller multiplies the matrix's input by the scale (s (x W) = (s x) W; the bias is added after), which matches ComfyUI's fp32 decode where <see cref="Matrix"/>'s folded
    /// decode rounds every weight to bf16. Valid until the next call.
    /// </summary>
    public BV MatrixExact(GpuWeights w, string name)
    {
        if (FoldScales) return Matrix(w, name);
        if (!w.IsF8(name)) return w.BF16(name);
        long count = 1; foreach (int d in w.Shape(name)) count *= d;
        var dst = Scratch(count);
        WanComfyFormat.DecodeF8E4M3(_acc, w.F8Words(name), dst, 1f);
        return dst;
    }

    /// <summary>The factor <see cref="MatrixExact"/> leaves to the input: the 8-bit matrix's scale, 1 for every other matrix.</summary>
    public float ActivationScale(GpuWeights w, string name) => !FoldScales && w.IsF8(name) && _scales.TryGetValue(name, out var s) ? s : 1f;

    /// <summary>A/B switch (env ML_FP8_FOLD_SCALES=1): decode scaled 8-bit matrices as bf16(f8 * scale), the folded form, instead of the exact one. For measuring which a reference used.</summary>
    public static bool FoldScales { get; set; } = Environment.GetEnvironmentVariable("ML_FP8_FOLD_SCALES") == "1";

    public void Dispose()
    {
        _scratch?.Dispose(); _scratch = null;
        _hostCache.Clear(); _hostCacheUsed = 0; _pending = null;
        foreach (var r in _retired) r.Dispose(); _retired.Clear();
        foreach (var r in _resident) r?.Dispose();
        Array.Clear(_resident);
        _base.Dispose();
    }
}
