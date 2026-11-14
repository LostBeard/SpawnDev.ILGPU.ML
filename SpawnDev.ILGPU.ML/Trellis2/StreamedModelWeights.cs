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
    private readonly Func<GpuWeights, Task>? _adapt;
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

    private StreamedModelWeights(Accelerator acc, IModelSource source, string repo, string file, string prefix, Func<string, int> blockOf, int blockCount, Func<GpuWeights, Task>? adapt, GpuWeights baseWeights,
        System.Collections.Concurrent.ConcurrentDictionary<string, float> scales)
    {
        _acc = acc; _source = source; _repo = repo; _file = file; _prefix = prefix; _blockOf = blockOf; BlockCount = blockCount; _adapt = adapt; _base = baseWeights; _scales = scales;
        _resident = new GpuWeights?[blockCount];
    }

    /// <summary>
    /// Open <paramref name="file"/>: <paramref name="anchor"/> (a tensor name the model always has) finds the prefix; <paramref name="blockOf"/> maps a prefix-free name to its block
    /// (0 .. <paramref name="blockCount"/> - 1) or -1; blocks are kept in order while they fit in <paramref name="residentBytes"/> (as stored), the rest streamed.
    /// </summary>
    public static async Task<StreamedModelWeights> OpenAsync(Accelerator accelerator, IModelSource source, string repo, string file, string anchor, Func<string, int> blockOf, int blockCount,
        long residentBytes, Func<GpuWeights, Task>? adapt = null, CancellationToken ct = default)
    {
        string prefix; var blockBytes = new long[blockCount];
        await using (var s = await source.OpenAsync(repo, file, ct))
        {
            if (!s.CanSeek) throw new NotSupportedException($"'{file}': the blocks are read by offset; the model source must give a seekable stream");
            var lenBytes = new byte[8]; await s.ReadExactlyAsync(lenBytes, ct);
            long headerSize = BitConverter.ToInt64(lenBytes);
            if (headerSize <= 0 || headerSize > 100_000_000) throw new InvalidDataException($"'{file}': invalid SafeTensors header size {headerSize}");
            var header = new byte[headerSize]; await s.ReadExactlyAsync(header, ct);
            var (tensors, _) = SafeTensors.SafeTensorsFileReader.ParseHeader(file, header, 8 + headerSize, s.Length - 8 - headerSize);
            var a = tensors.FirstOrDefault(t => t.Name == anchor || t.Name.EndsWith("." + anchor, StringComparison.Ordinal))
                ?? throw new InvalidDataException($"'{file}' does not hold the expected model ({anchor})");
            prefix = a.Name[..^anchor.Length];
            foreach (var t in tensors)
            {
                if (!t.Name.StartsWith(prefix, StringComparison.Ordinal)) continue;
                int k = blockOf(t.Name[prefix.Length..]);
                if (k >= 0) blockBytes[k] += t.DataLength;
            }
        }
        var scales = new System.Collections.Concurrent.ConcurrentDictionary<string, float>();
        var baseW = await LoadAsync(accelerator, source, repo, file, prefix, n => blockOf(n) < 0, adapt, scales, ct);
        var w = new StreamedModelWeights(accelerator, source, repo, file, prefix, blockOf, blockCount, adapt, baseW, scales);
        try
        {
            long budget = residentBytes, streamed = 0;
            for (int k = 0; k < blockCount; k++)
            {
                if (blockBytes[k] <= budget) { int kk = k; w._resident[k] = await LoadAsync(accelerator, source, repo, file, prefix, n => blockOf(n) == kk, adapt, scales, ct); budget -= blockBytes[k]; }
                else { budget = 0; streamed += blockBytes[k]; }                       // the resident blocks are a prefix, the streamed ones the tail
            }
            w.StreamedBytesPerForward = streamed;
            return w;
        }
        catch { w.Dispose(); throw; }
    }

    private static async Task<GpuWeights> LoadAsync(Accelerator acc, IModelSource source, string repo, string file, string prefix, Func<string, bool> keep, Func<GpuWeights, Task>? adapt,
        System.Collections.Concurrent.ConcurrentDictionary<string, float> scales, CancellationToken ct)
    {
        await using var stream = await source.OpenAsync(repo, file, ct);
        using var empty = JsonDocument.Parse("{}");
        var w = await GpuWeights.LoadAsync(acc, stream, empty.RootElement, file, n => n.StartsWith(prefix, StringComparison.Ordinal) && n != prefix + "scaled_fp8" && keep(n[prefix.Length..]), ct);   // scaled_fp8: ComfyUI's empty marker
        try
        {
            foreach (var n in w.Names.ToList())
            {
                string to = n[prefix.Length..];
                if (to != n) w.Rename(n, to);
            }
            // fp8_scaled: scale_weight (fp32 scalar) becomes the matrix's decode scale; scale_input (an activation scale for fp8 matmuls) is not used, the activations stay fp32
            foreach (var n in w.Names.ToList())
            {
                if (n.EndsWith(".scale_input", StringComparison.Ordinal)) { await acc.SynchronizeAsync(); w.Remove(n)?.Dispose(); continue; }
                if (!n.EndsWith(".scale_weight", StringComparison.Ordinal)) continue;
                if (!w.IsF32(n)) throw new InvalidDataException($"'{file}': {n} is not an fp32 scale");
                scales[n[..^".scale_weight".Length] + ".weight"] = (await w.F32(n).CopyToHostAsync())[0];
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
            if (adapt != null) await adapt(w);
            // an 8-bit matrix without its scale in a file that has scales would decode silently wrong (an adapter that fused scaled parts, say)
            if (!scales.IsEmpty)
                foreach (var m in w.Names.Where(w.IsF8))
                    if (w.Shape(m).Length >= 2 && !scales.ContainsKey(m)) throw new NotSupportedException($"'{file}': {m} is an 8-bit matrix without a scale_weight in a scaled file");
            return w;
        }
        catch { w.Dispose(); throw; }
    }

    /// <summary>Block <paramref name="k"/>'s tensors: the resident set, or a fresh load from the file that the caller disposes after the device has run it (<paramref name="streamed"/> true).</summary>
    public async Task<(GpuWeights Weights, bool Streamed)> BlockAsync(int k, CancellationToken ct = default)
    {
        if (_resident[k] is { } r) return (r, false);
        return (await LoadAsync(_acc, _source, _repo, _file, _prefix, n => _blockOf(n) == k, _adapt, _scales, ct), true);
    }

    /// <summary>A matrix of <paramref name="w"/> as bf16: the stored tensor, or (8-bit floats) its decode into the shared scratch, valid until the next call.</summary>
    public BV Matrix(GpuWeights w, string name)
    {
        if (!w.IsF8(name)) return w.BF16(name);
        long count = 1; foreach (int d in w.Shape(name)) count *= d;
        if (_scratch == null || _scratch.Length < count) { _acc.Synchronize(); _scratch?.Dispose(); _scratch = _acc.Allocate1D<BFloat16>(count); }
        var dst = _scratch.View.SubView(0, count);
        WanComfyFormat.DecodeF8E4M3(_acc, w.F8Words(name), dst, _scales.TryGetValue(name, out var sc) ? sc : 1f);      // unscaled 8-bit files (ComfyUI's e4m3fn checkpoints): 1
        return dst;
    }

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
        if (_scratch == null || _scratch.Length < count) { _acc.Synchronize(); _scratch?.Dispose(); _scratch = _acc.Allocate1D<BFloat16>(count); }
        var dst = _scratch.View.SubView(0, count);
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
        foreach (var r in _resident) r?.Dispose();
        Array.Clear(_resident);
        _base.Dispose();
    }
}
