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
/// an anchor tensor and removed. 8-bit float (F8_E4M3, unscaled) matrices stay 8-bit and are decoded into one shared bf16 scratch just before their GEMM; 8-bit
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

    public int BlockCount { get; }
    /// <summary>Device bytes held by the weights that stay (the non-block tensors and the resident blocks), plus the decode scratch.</summary>
    public long ResidentBytes => _base.Bytes + _resident.Sum(r => r?.Bytes ?? 0) + (_scratch?.Length ?? 0) * 2;
    /// <summary>How many blocks stay on the device; the others are read from the file on every forward pass.</summary>
    public int ResidentBlocks => _resident.Count(r => r != null);
    /// <summary>The bytes a forward pass reads from the file (the streamed blocks).</summary>
    public long StreamedBytesPerForward { get; private set; }
    /// <summary>The non-block tensors.</summary>
    public GpuWeights Base => _base;

    private StreamedModelWeights(Accelerator acc, IModelSource source, string repo, string file, string prefix, Func<string, int> blockOf, int blockCount, Func<GpuWeights, Task>? adapt, GpuWeights baseWeights)
    {
        _acc = acc; _source = source; _repo = repo; _file = file; _prefix = prefix; _blockOf = blockOf; BlockCount = blockCount; _adapt = adapt; _base = baseWeights;
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
        var baseW = await LoadAsync(accelerator, source, repo, file, prefix, n => blockOf(n) < 0, adapt, ct);
        var w = new StreamedModelWeights(accelerator, source, repo, file, prefix, blockOf, blockCount, adapt, baseW);
        try
        {
            long budget = residentBytes, streamed = 0;
            for (int k = 0; k < blockCount; k++)
            {
                if (blockBytes[k] <= budget) { int kk = k; w._resident[k] = await LoadAsync(accelerator, source, repo, file, prefix, n => blockOf(n) == kk, adapt, ct); budget -= blockBytes[k]; }
                else { budget = 0; streamed += blockBytes[k]; }                       // the resident blocks are a prefix, the streamed ones the tail
            }
            w.StreamedBytesPerForward = streamed;
            return w;
        }
        catch { w.Dispose(); throw; }
    }

    private static async Task<GpuWeights> LoadAsync(Accelerator acc, IModelSource source, string repo, string file, string prefix, Func<string, bool> keep, Func<GpuWeights, Task>? adapt, CancellationToken ct)
    {
        await using var stream = await source.OpenAsync(repo, file, ct);
        using var empty = JsonDocument.Parse("{}");
        var w = await GpuWeights.LoadAsync(acc, stream, empty.RootElement, file, n => n.StartsWith(prefix, StringComparison.Ordinal) && keep(n[prefix.Length..]), ct);
        try
        {
            foreach (var n in w.Names.ToList())
            {
                string to = n[prefix.Length..];
                if (to != n) w.Rename(n, to);
                if (w.IsF8(to) && w.Shape(to).Length == 1)
                {
                    var dst = acc.Allocate1D<BFloat16>(w.Shape(to)[0]);
                    WanComfyFormat.DecodeF8E4M3(acc, w.F8Words(to), dst.View, 1f);
                    var old = w.SetBF16(to, dst, w.Shape(to));
                    await acc.SynchronizeAsync(); old?.Dispose();
                }
            }
            if (adapt != null) await adapt(w);
            return w;
        }
        catch { w.Dispose(); throw; }
    }

    /// <summary>Block <paramref name="k"/>'s tensors: the resident set, or a fresh load from the file that the caller disposes after the device has run it (<paramref name="streamed"/> true).</summary>
    public async Task<(GpuWeights Weights, bool Streamed)> BlockAsync(int k, CancellationToken ct = default)
    {
        if (_resident[k] is { } r) return (r, false);
        return (await LoadAsync(_acc, _source, _repo, _file, _prefix, n => _blockOf(n) == k, _adapt, ct), true);
    }

    /// <summary>A matrix of <paramref name="w"/> as bf16: the stored tensor, or (8-bit floats) its decode into the shared scratch, valid until the next call.</summary>
    public BV Matrix(GpuWeights w, string name)
    {
        if (!w.IsF8(name)) return w.BF16(name);
        long count = 1; foreach (int d in w.Shape(name)) count *= d;
        if (_scratch == null || _scratch.Length < count) { _acc.Synchronize(); _scratch?.Dispose(); _scratch = _acc.Allocate1D<BFloat16>(count); }
        var dst = _scratch.View.SubView(0, count);
        WanComfyFormat.DecodeF8E4M3(_acc, w.F8Words(name), dst, 1f);
        return dst;
    }

    public void Dispose()
    {
        _scratch?.Dispose(); _scratch = null;
        foreach (var r in _resident) r?.Dispose();
        Array.Clear(_resident);
        _base.Dispose();
    }
}
