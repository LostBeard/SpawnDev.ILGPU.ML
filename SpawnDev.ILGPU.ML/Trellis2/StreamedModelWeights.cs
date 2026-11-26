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
public sealed class StreamedModelWeights : IDisposable, IDeviceSized
{
    long IDeviceSized.DeviceBytes => ResidentBytes;

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
    private Dictionary<string, SafeTensors.SafeTensorInfo> _hostOnly = new();

    /// <summary>
    /// What a blockOf function returns for a tensor that is never loaded to the device: the caller reads the rows it needs with <see cref="ReadRowsAsync"/> (a 262k-row token
    /// embedding table, say, of which one prompt needs a hundred rows). -1 is the always-resident base set, 0.. the blocks.
    /// </summary>
    public const int HostOnly = -2;

    /// <summary>
    /// Rows <paramref name="rows"/> of the 2-D BF16 tensor <paramref name="name"/> (prefix-free; one the blockOf function marked <see cref="HostOnly"/>), read from the file
    /// by offset, as fp32 [rows.Length, columns] (bf16 to fp32 is exact).
    /// </summary>
    public async Task<float[]> ReadRowsAsync(string name, IReadOnlyList<int> rows, CancellationToken ct = default)
    {
        if (!_hostOnly.TryGetValue(name, out var t)) throw new KeyNotFoundException($"'{_file}': {name} is not a host-only tensor (blockOf must return StreamedModelWeights.HostOnly for it)");
        if (t.DType != "BF16" || t.Shape.Length != 2) throw new NotSupportedException($"'{_file}': {name} is {t.DType} {string.Join("x", t.Shape)}; host rows are read from 2-D BF16 tensors");
        int R = (int)t.Shape[0], C = (int)t.Shape[1];
        var o = new float[(long)rows.Count * C];
        var raw = new byte[C * 2];
        await using var s = await _source.OpenAsync(_repo, _file, ct);
        for (int i = 0; i < rows.Count; i++)
        {
            int r = rows[i];
            if (r < 0 || r >= R) throw new ArgumentOutOfRangeException(nameof(rows), $"row {r} of {name} [{R}, {C}]");
            s.Seek(t.DataOffset + (long)r * C * 2, SeekOrigin.Begin);
            await s.ReadExactlyAsync(raw, ct);
            for (int c = 0; c < C; c++) o[(long)i * C + c] = BitConverter.Int32BitsToSingle(BitConverter.ToUInt16(raw, 2 * c) << 16);
        }
        return o;
    }
    private readonly Dictionary<int, BlockImage> _hostCache = new();
    private long _hostCacheUsed;
    // the read-ahead: the next streamed blocks in forward order (wrapping into the next pass), each read after the one before it (one reader: one disk)
    private readonly List<(int Block, Task<BlockImage> Image, long Bytes)> _ahead = new();

    // A block image in host memory: Length bytes of a REUSED buffer. Buffers come from and go back to a small pool: a fresh 0.4 GB array per streamed load was
    // 54 to 278 gen-2 collections per LTX-2 clip (MEASURED 2026-11-18; 22 with the pool).
    // PAGE-LOCKED on CUDA (PageLockImages): the upload from a registered buffer is a straight DMA (MEASURED 2026-11-18 with a first version, LTX-2, 16.2 GB a call:
    // upload 7.5 -> 5.8 s exact, 5.2 -> 3.4 s on the tensor cores, bit-identical). It waited on SpawnDev.ILGPU 5.4.37: before it the page-lock registered the range
    // without binding the accelerator's context, and these buffers are rented on thread-pool threads ("invalid device context at MemHostRegister", 2 of 3 runs).
    // A page-locked image's registration is released (Lock.Dispose) wherever the image leaves the pool for good: ReturnImage past the pool's size, and Dispose.
    private sealed class BlockImage
    {
        public required byte[] Bytes;
        public int Length;
        public PageLockedArray1D<byte>? Lock;                                                      // the registration behind Bytes (null: a plain array)
    }

    /// <summary>
    /// CUDA: streamed block images are page-locked host buffers (faster uploads; host memory pinned while the weights are open: the pool's two or three buffers of the
    /// largest block's size). Env ML_STREAM_PAGELOCK=0 turns it off for new instances (an A/B switch). Other backends always use plain arrays.
    /// </summary>
    public static bool PageLockImages { get; set; } = Environment.GetEnvironmentVariable("ML_STREAM_PAGELOCK") != "0";

    private static void ReleaseImage(BlockImage image)
    {
        if (image.Lock is { } l) { l.Dispose(); Interlocked.Decrement(ref _liveImageLocks); }
        image.Lock = null;
    }
    private static int _liveImageLocks;
    /// <summary>Page-locked block images not yet released, across every instance (tests: back to 0 once the weights are disposed and their reads have finished).</summary>
    internal static int LiveImageLocks => Volatile.Read(ref _liveImageLocks);
    private readonly List<BlockImage> _freeImages = new();
    private const int MaxFreeImages = 2;                                                           // the one being uploaded + the one being read: what was live before the pool
    private long _maxBlockBytes = -1;
    private bool _disposed;

    private BlockImage RentImage(long total)
    {
        lock (_freeImages)
        {
            for (int i = 0; i < _freeImages.Count; i++)
                if (_freeImages[i].Bytes.Length >= total) { var im = _freeImages[i]; _freeImages.RemoveAt(i); im.Length = (int)total; return im; }
            if (_maxBlockBytes < 0) for (int k = 0; k < BlockCount; k++) if (_resident[k] == null) _maxBlockBytes = Math.Max(_maxBlockBytes, BlockBytes(k));
        }
        // room for the largest streamed block and its header, so one buffer serves every block
        long cap = Math.Min(Array.MaxLength, Math.Max(total, _maxBlockBytes + (1 << 20)));
        if (PageLockImages && _acc.AcceleratorType == AcceleratorType.Cuda)
        {
            var pl = _acc.AllocatePageLocked1D<byte>(cap, uninitialized: true);              // binds the context itself (SpawnDev.ILGPU 5.4.37): any thread
            Interlocked.Increment(ref _liveImageLocks);
            return new BlockImage { Bytes = pl.GetArray(), Length = (int)total, Lock = pl };
        }
        return new BlockImage { Bytes = GC.AllocateUninitializedArray<byte>((int)cap), Length = (int)total };
    }

    private void ReturnImage(BlockImage image)
    {
        lock (_freeImages)
        {
            // as many spare buffers as the read-ahead can have in flight; one beyond that is left to the collector (its page-lock released first)
            int spare = Math.Max(MaxFreeImages, (int)(ReadAheadBytes / Math.Max(1, _maxBlockBytes)) + 2);
            if (!_disposed && _freeImages.Count < spare) { _freeImages.Add(image); return; }
        }
        ReleaseImage(image);
    }

    // a read nobody will consume: its buffer goes back once the read has finished
    private void DropRead(Task<BlockImage> read)
        => _ = read.ContinueWith(t => { if (t.Status == TaskStatus.RanToCompletion) ReturnImage(t.Result); else _ = t.Exception; }, TaskScheduler.Default);

    /// <summary>
    /// Host RAM (bytes) the read-ahead may hold in block images (default 0: the next block only, whatever its size). With a budget the reader keeps reading the
    /// following streamed blocks while the device works, so the disk does not idle through a run of resident or host-cached blocks and catch up later with the
    /// caller waiting. Needs <see cref="Prefetch"/>.
    /// </summary>
    public long ReadAheadBytes { get; set; }

    private long BlockBytes(int k) { long n = 0; foreach (var t in _blockTensors[k]) n += t.DataLength; return n; }

    // after block k was handed out: queue reads of the streamed, uncached blocks that follow, one beyond the last queued while the budget lasts (always at least one)
    private void TopUpReadAhead(int k)
    {
        long held = 0; foreach (var a in _ahead) held += a.Bytes;
        int last = _ahead.Count > 0 ? _ahead[^1].Block : k;
        for (int guard = 0; guard < BlockCount; guard++)
        {
            int? next = null;
            for (int i = 1; i <= BlockCount; i++)                                            // the streamed block after `last` that the host cache does not hold
            {
                int b = (last + i) % BlockCount;
                if (b == k) break;                                                           // a whole pass is queued: nothing more to read ahead
                if (_resident[b] == null && !_hostCache.ContainsKey(b)) { next = b; break; }
            }
            if (next is not int nb || _ahead.Any(a => a.Block == nb)) return;
            long bytes = BlockBytes(nb);
            if (_ahead.Count > 0 && held + bytes > ReadAheadBytes) return;
            var prev = _ahead.Count > 0 ? _ahead[^1].Image : null;
            _ahead.Add((nb, Task.Run(async () =>
            {
                if (prev != null) { try { await prev; } catch { } }                          // one read at a time, in order; a failed read does not stop the next
                return await ReadBlockImageAsync(nb, CancellationToken.None);
            }), bytes));
            held += bytes; last = nb;
        }
    }

    // block k's image from the read-ahead when it is queued (everything queued before it was skipped by the caller: observed and dropped)
    private async Task<BlockImage?> TakeReadAheadAsync(int k)
    {
        int at = _ahead.FindIndex(a => a.Block == k);   // by block, never "the first queued": out of order that hands out another block's image (red-check)
        if (at < 0) return null;
        for (int i = 0; i < at; i++) DropRead(_ahead[i].Image);
        var image = _ahead[at].Image;
        _ahead.RemoveRange(0, at + 1);
        return await image;
    }
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
    private static void CheckLayerFormat(string file, string name, byte[] tag)
    {
        using var doc = JsonDocument.Parse(tag);
        string format = doc.RootElement.TryGetProperty("format", out var f) ? f.GetString() ?? "" : "";
        // float8_e4m3fn: value * weight_scale (fp32), run with full-precision activations; nvfp4: Kernels.Nvfp4 (weight_scale = fp8 block scales, weight_scale_2 = fp32), the same.
        // ComfyUI's CPU reference runs both at full precision (MEASURED: SpawnGen's Gemma dump, Nvfp4_GemmaLayer_MatchesComfyUi); an fp8 tag that says otherwise is refused
        bool fullPrecisionFalse = doc.RootElement.TryGetProperty("full_precision_matrix_mult", out var fp) && fp.ValueKind == JsonValueKind.False;
        if (format is not ("float8_e4m3fn" or "nvfp4") || fullPrecisionFalse)
            throw new NotSupportedException($"'{file}': {name} declares '{System.Text.Encoding.UTF8.GetString(tag)}'; supported layer formats: float8_e4m3fn and nvfp4, full-precision activations");
    }

    private static void CheckQuantization(string file, IReadOnlyDictionary<string, string> metadata)
    {
        if (!metadata.TryGetValue("_quantization_metadata", out var json)) return;
        using var doc = JsonDocument.Parse(json);
        if (!doc.RootElement.TryGetProperty("layers", out var layers)) return;
        foreach (var layer in layers.EnumerateObject())
        {
            string format = layer.Value.TryGetProperty("format", out var f) ? f.GetString() ?? "" : "";
            bool full = !(layer.Value.TryGetProperty("full_precision_matrix_mult", out var fp) && fp.ValueKind == JsonValueKind.False);
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
        string prefix; var blockBytes = new long[blockCount]; var headerNames = new HashSet<string>(); Dictionary<string, SafeTensors.SafeTensorInfo> hostOnly;
        var blockTensors = new List<SafeTensors.SafeTensorInfo>[blockCount]; Dictionary<string, string> metadata;
        await using (var s = await source.OpenAsync(repo, file, ct))
        {
            if (!s.CanSeek) throw new NotSupportedException($"'{file}': the blocks are read by offset; the model source must give a seekable stream");
            var lenBytes = new byte[8]; await s.ReadExactlyAsync(lenBytes, ct);
            long headerSize = BitConverter.ToInt64(lenBytes);
            if (headerSize <= 0 || headerSize > 100_000_000) throw new InvalidDataException($"'{file}': invalid SafeTensors header size {headerSize}");
            var header = new byte[headerSize]; await s.ReadExactlyAsync(header, ct);
            var (tensors, meta) = SafeTensors.SafeTensorsFileReader.ParseHeader(file, header, 8 + headerSize, s.Length - 8 - headerSize);
            var quantTags = new List<SafeTensors.SafeTensorInfo>();
            hostOnly = new Dictionary<string, SafeTensors.SafeTensorInfo>();
            metadata = meta;
            for (int k = 0; k < blockCount; k++) blockTensors[k] = new List<SafeTensors.SafeTensorInfo>();
            var a = tensors.FirstOrDefault(t => t.Name == anchor || t.Name.EndsWith("." + anchor, StringComparison.Ordinal))
                ?? throw new InvalidDataException($"'{file}' does not hold the expected model ({anchor})");
            prefix = a.Name[..^anchor.Length];
            foreach (var t in tensors)
            {
                if (!t.Name.StartsWith(prefix, StringComparison.Ordinal)) continue;
                headerNames.Add(t.Name[prefix.Length..]);
                if (t.Name.EndsWith(".comfy_quant", StringComparison.Ordinal)) quantTags.Add(t);
                if (IsNotAWeight(t.Name[prefix.Length..])) continue;
                int k = blockOf(t.Name[prefix.Length..]);
                if (k >= 0) { blockBytes[k] += t.DataLength; blockTensors[k].Add(t); }
                else if (k == HostOnly) hostOnly[t.Name[prefix.Length..]] = t;
            }
            foreach (var t in quantTags)
            {
                if (t.DataLength > 4096) throw new InvalidDataException($"'{file}': {t.Name} is {t.DataLength} bytes, not a quantization tag");
                var tag = new byte[t.DataLength];
                s.Seek(t.DataOffset, SeekOrigin.Begin); await s.ReadExactlyAsync(tag, ct);
                CheckLayerFormat(file, t.Name, tag);
            }
        }
        var scales = new System.Collections.Concurrent.ConcurrentDictionary<string, float>();
        Func<CancellationToken, Task<Stream>> open = c => source.OpenAsync(repo, file, c);
        var baseW = await LoadAsync(accelerator, open, file, prefix, n => blockOf(n) == -1, adapt, scales, ct);
        var w = new StreamedModelWeights(accelerator, source, repo, file, prefix, blockOf, blockCount, adapt, baseW, scales) { HeaderNames = headerNames, _blockTensors = blockTensors, _metadata = metadata, _hostOnly = hostOnly };
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
                string? suffix = n.EndsWith(".scale_weight", StringComparison.Ordinal) ? ".scale_weight" : n.EndsWith(".weight_scale", StringComparison.Ordinal) ? ".weight_scale"
                    : n.EndsWith(".weight_scale_2", StringComparison.Ordinal) ? ".weight_scale_2" : null;
                if (suffix == null) continue;
                if (suffix == ".weight_scale" && w.IsF8(n)) continue;                                  // NVFP4: the fp8 block scales stay (MatrixExact decodes with them)
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
                    if (w.Shape(m).Length >= 2 && !m.EndsWith(".weight_scale", StringComparison.Ordinal) && !scales.ContainsKey(m)) throw new NotSupportedException($"'{file}': {m} is an 8-bit matrix without a scale_weight in a scaled file");
            // NVFP4: packed codes [N, K / 2] with their block scales (whole swizzled tiles) and a second scale
            foreach (var m in w.Names.Where(w.IsU8).ToList())
            {
                string bs = m[..^"weight".Length] + "weight_scale";
                if (!m.EndsWith(".weight", StringComparison.Ordinal) || w.Shape(m).Length != 2 || !w.IsF8(bs) || !scales.ContainsKey(m))
                    throw new NotSupportedException($"'{file}': {m} is a byte tensor that is not an NVFP4 matrix (weight [N, K / 2] + fp8 weight_scale + fp32 weight_scale_2)");
                int N = w.Shape(m)[0], K = w.Shape(m)[1] * 2;
                long have = 1; foreach (int d in w.Shape(bs)) have *= d;
                if (K % 16 != 0 || have < Kernels.Nvfp4.ScaleBytes(N, K)) throw new InvalidDataException($"'{file}': {bs} holds {have} scales, NVFP4 [{N}, {K}] needs {Kernels.Nvfp4.ScaleBytes(N, K)}");
            }
            return w;
        }
        catch { w.Dispose(); throw; }
    }

    /// <summary>
    /// Block <paramref name="k"/> as a self-contained safetensors image in host memory: a header naming its tensors (the file's own names and metadata) and their bytes,
    /// read from the file by offset. The loader then runs on it exactly as on the file.
    /// </summary>
    private async Task<BlockImage> ReadBlockImageAsync(int k, CancellationToken ct)
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
        var rented = RentImage(total);                                                      // every byte of [0, total) is written below: the buffer comes as it is
        try
        {
            var image = rented.Bytes;
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
            return rented;
        }
        catch { ReturnImage(rented); throw; }
    }

    /// <summary>Block <paramref name="k"/>'s tensors: the resident set, or a fresh load from the file that the caller disposes after the device has run it (<paramref name="streamed"/> true).</summary>
    public async Task<(GpuWeights Weights, bool Streamed)> BlockAsync(int k, CancellationToken ct = default)
    {
        if (_retired.Count > 0) { await _acc.SynchronizeAsync(); foreach (var old in _retired) old.Dispose(); _retired.Clear(); }
        if (_resident[k] is { } r) return (r, false);
        var sw = System.Diagnostics.Stopwatch.StartNew();
        BlockImage image; bool kept = false;                                                      // kept: the host cache holds it, it does not go back to the pool
        if (_hostCache.TryGetValue(k, out var cached)) { image = cached; kept = true; }
        else if (await TakeReadAheadAsync(k) is { } ahead) image = ahead;
        else
        {
            // not what the read-ahead expected (the caller left the forward order): its queue is of no use, drop it and read this block now
            foreach (var a in _ahead) DropRead(a.Image);
            _ahead.Clear();
            image = await ReadBlockImageAsync(k, ct);
        }
        if (!kept && _hostCacheUsed + image.Length <= HostCacheBytes) { _hostCache[k] = image; _hostCacheUsed += image.Length; kept = true; }
        if (Prefetch) TopUpReadAhead(k);
        GpuWeights loaded;
        try
        {
            var up = System.Diagnostics.Stopwatch.StartNew();
            loaded = await LoadAsync(_acc, _ => Task.FromResult<Stream>(new MemoryStream(image.Bytes, 0, image.Length, writable: false, publiclyVisible: true)), _file, _prefix, n => _blockOf(n) == k, _adapt, _scales, ct);
            await _acc.SynchronizeAsync();                                                        // the upload's own time, not a later kernel's; and nothing reads the image after it
            UploadSeconds += up.Elapsed.TotalSeconds;
        }
        finally { if (!kept) ReturnImage(image); }
        StreamSeconds += sw.Elapsed.TotalSeconds; StreamedLoads++;
        return (loaded, true);
    }

    /// <summary>A matrix of <paramref name="w"/> as bf16: the stored tensor, or (8-bit floats) its decode into the shared scratch, valid until the next call.</summary>
    public BV Matrix(GpuWeights w, string name)
    {
        if (w.IsU8(name)) return DecodeNvfp4(w, name, _scales.TryGetValue(name, out var s2) ? s2 : 1f);
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
        if (w.IsU8(name)) return DecodeNvfp4(w, name, 1f);
        if (!w.IsF8(name)) return w.BF16(name);
        long count = 1; foreach (int d in w.Shape(name)) count *= d;
        var dst = Scratch(count);
        WanComfyFormat.DecodeF8E4M3(_acc, w.F8Words(name), dst, 1f);
        return dst;
    }

    /// <summary>The factor <see cref="MatrixExact"/> leaves to the input: the 8-bit matrix's scale, 1 for every other matrix.</summary>
    public float ActivationScale(GpuWeights w, string name) => !FoldScales && (w.IsF8(name) || w.IsU8(name)) && _scales.TryGetValue(name, out var s) ? s : 1f;

    // an NVFP4 matrix [N, K] into the shared scratch: e2m1 x block scale x scale (1: exact, the second scale left to the input)
    private BV DecodeNvfp4(GpuWeights w, string name, float scale)
    {
        int N = w.Shape(name)[0], K = w.Shape(name)[1] * 2;
        var dst = Scratch((long)N * K);
        Kernels.Nvfp4.Decode(_acc, w.U8Words(name), w.F8Words(name[..^"weight".Length] + "weight_scale"), dst, N, K, scale);
        return dst;
    }

    /// <summary>A/B switch (env ML_FP8_FOLD_SCALES=1): decode scaled 8-bit matrices as bf16(f8 * scale), the folded form, instead of the exact one. For measuring which a reference used.</summary>
    public static bool FoldScales { get; set; } = Environment.GetEnvironmentVariable("ML_FP8_FOLD_SCALES") == "1";

    public void Dispose()
    {
        _scratch?.Dispose(); _scratch = null;
        lock (_freeImages) { _disposed = true; foreach (var im in _freeImages) ReleaseImage(im); _freeImages.Clear(); }
        foreach (var im in _hostCache.Values) ReleaseImage(im);
        _hostCache.Clear(); _hostCacheUsed = 0;
        foreach (var a in _ahead) DropRead(a.Image);                                              // reads still in flight: their buffers are released when they finish
        _ahead.Clear();
        foreach (var r in _retired) r.Dispose(); _retired.Clear();
        foreach (var r in _resident) r?.Dispose();
        Array.Clear(_resident);
        _base.Dispose();
    }
}
