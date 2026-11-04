using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Training;

/// <summary>
/// A PEFT LoRA adapter on the host: its tensors (<c>base_model.model.model.layers.N.{module}.lora_A/B.weight</c>), rank and alpha. Loaded from the
/// two PEFT files by <see cref="Load"/>; consumed by <see cref="LoraRuntime"/> (the batch decoder) and by <c>GraphAutograd.AddLora</c> (the session
/// engine), so either engine serves the same adapter.
/// </summary>
public sealed record LoraAdapterSet(IReadOnlyDictionary<string, float[]> Tensors, int Rank, float Alpha)
{
    /// <summary>Read <c>adapter_model.safetensors</c> bytes and <c>adapter_config.json</c> text; rsLoRA / DoRA / bias adapters are refused, not misapplied.</summary>
    public static LoraAdapterSet Load(byte[] adapterSafeTensors, string adapterConfigJson)
    {
        using var cfg = System.Text.Json.JsonDocument.Parse(adapterConfigJson);
        var root = cfg.RootElement;
        if (root.TryGetProperty("peft_type", out var pt) && pt.GetString() != "LORA") throw new NotSupportedException($"peft_type {pt.GetString()} is not LORA");
        if (root.TryGetProperty("use_rslora", out var rs) && rs.ValueKind == System.Text.Json.JsonValueKind.True) throw new NotSupportedException("rsLoRA scaling (use_rslora) is not supported");
        if (root.TryGetProperty("use_dora", out var dora) && dora.ValueKind == System.Text.Json.JsonValueKind.True) throw new NotSupportedException("DoRA adapters are not supported");
        if (root.TryGetProperty("bias", out var bias) && bias.GetString() is { } b && b != "none") throw new NotSupportedException($"LoRA bias '{b}' is not supported");
        int rank = root.GetProperty("r").GetInt32();
        float alpha = root.TryGetProperty("lora_alpha", out var a) ? a.GetSingle() : rank;
        return new LoraAdapterSet(LoraAdapterFile.ReadSafeTensors(adapterSafeTensors), rank, alpha);
    }

    /// <summary>The GGUF projection names (attn_q, ffn_up, ...) this adapter touches.</summary>
    public IReadOnlyCollection<string> GgufTargets => Tensors.Keys
        .Select(k => System.Text.RegularExpressions.Regex.Match(k, @"^base_model\.model\.model\.layers\.\d+\.(.+)\.lora_A\.weight$"))
        .Where(m => m.Success).Select(m => LoraRuntime.GgufTarget(m.Groups[1].Value)).Distinct().ToList();
}

/// <summary>
/// LoRA adapters applied UNMERGED at inference by the hand-written decoders (<see cref="Pipelines.GgufBatchDecoder"/>): after a projection's
/// <c>o = x W</c> the decoder calls <see cref="Apply"/>, which adds <c>(alpha / rank) * (x A^T) B^T</c> into <c>o</c>'s columns
/// [colOffset, colOffset + out) of every row (a fused q|k|v output holds q, k and v side by side). The base weights stay quantized and
/// untouched; merging a rank-r delta into Q4_K blocks would requantize it away. Loads PEFT tensors (<see cref="LoraAdapterFile"/>).
/// </summary>
public sealed class LoraRuntime : IDisposable
{
    private readonly Accelerator _acc;
    private readonly Dictionary<(int Layer, string Target), (MemoryBuffer1D<float, Stride1D.Dense> AT, MemoryBuffer1D<float, Stride1D.Dense> BT, int In, int Out)> _adapters = new();
    private MemoryBuffer1D<float, Stride1D.Dense>? _t;
    private readonly List<MemoryBuffer1D<float, Stride1D.Dense>> _retired = new();   // a queued launch may still read a replaced scratch
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int>? _down;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        int, int, int, int, float>? _up;

    /// <summary>LoRA rank (r).</summary>
    public int Rank { get; }
    /// <summary>The scale applied to the adapter's product: alpha / rank.</summary>
    public float Scale { get; }

    // PEFT module -> GGUF projection name (the inverse of GraphAutograd's table).
    /// <summary>The GGUF projection name for a PEFT module path (self_attn.q_proj -> attn_q).</summary>
    public static string GgufTarget(string peftModule)
        => Target.TryGetValue(peftModule, out var t) ? t : throw new NotSupportedException($"LoRA module {peftModule} is not supported");

    private static readonly Dictionary<string, string> Target = new()
    {
        ["self_attn.q_proj"] = "attn_q", ["self_attn.k_proj"] = "attn_k", ["self_attn.v_proj"] = "attn_v", ["self_attn.o_proj"] = "attn_output",
        ["mlp.gate_proj"] = "ffn_gate", ["mlp.up_proj"] = "ffn_up", ["mlp.down_proj"] = "ffn_down",
    };

    /// <summary>Upload PEFT adapter tensors (<c>base_model.model.model.layers.N.{module}.lora_A.weight</c> [rank, in] / <c>lora_B.weight</c> [out, rank]).</summary>
    public LoraRuntime(Accelerator accelerator, IReadOnlyDictionary<string, float[]> peft, int rank, float alpha)
    {
        _acc = accelerator; Rank = rank; Scale = alpha / rank;
        var rx = new System.Text.RegularExpressions.Regex(@"^base_model\.model\.model\.layers\.(\d+)\.(.+)\.lora_A\.weight$");
        foreach (var (name, a) in peft)
        {
            var m = rx.Match(name);
            if (!m.Success) continue;
            if (!Target.TryGetValue(m.Groups[2].Value, out var target)) throw new NotSupportedException($"LoRA module {m.Groups[2].Value} is not supported");
            var bName = name[..^"lora_A.weight".Length] + "lora_B.weight";
            if (!peft.TryGetValue(bName, out var b)) throw new ArgumentException($"{name} has no {bName}");
            if (a.Length % rank != 0 || b.Length % rank != 0) throw new ArgumentException($"{name}: sizes are not multiples of rank {rank}");
            int fin = a.Length / rank, fout = b.Length / rank;
            var aT = new float[fin * rank];
            for (int r = 0; r < rank; r++) for (int i = 0; i < fin; i++) aT[i * rank + r] = a[r * fin + i];
            var bT = new float[rank * fout];
            for (int o = 0; o < fout; o++) for (int r = 0; r < rank; r++) bT[r * fout + o] = b[o * rank + r];
            _adapters[(int.Parse(m.Groups[1].Value), target)] = (accelerator.Allocate1D(aT), accelerator.Allocate1D(bT), fin, fout);
        }
        if (_adapters.Count == 0) throw new ArgumentException("no LoRA adapter tensors found");
    }

    /// <summary>
    /// Load a PEFT adapter from its two files: <c>adapter_model.safetensors</c> bytes and <c>adapter_config.json</c> text (rank <c>r</c> and
    /// <c>lora_alpha</c> are read from it). <c>use_rslora</c> / DoRA / bias adapters are refused rather than applied wrongly.
    /// </summary>
    public static LoraRuntime Load(Accelerator accelerator, byte[] adapterSafeTensors, string adapterConfigJson)
        => Load(accelerator, LoraAdapterSet.Load(adapterSafeTensors, adapterConfigJson));

    /// <summary>Upload an adapter set (<see cref="LoraAdapterSet.Load"/>).</summary>
    public static LoraRuntime Load(Accelerator accelerator, LoraAdapterSet adapter) => new(accelerator, adapter.Tensors, adapter.Rank, adapter.Alpha);

    /// <summary>Adapted projections, as (layer, GGUF target name).</summary>
    public IEnumerable<(int Layer, string Target)> Targets => _adapters.Keys;

    /// <summary>True when layer <paramref name="layer"/>'s <paramref name="target"/> (attn_q, attn_v, ...) has an adapter.</summary>
    public bool Has(int layer, string target) => _adapters.ContainsKey((layer, target));

    /// <summary>
    /// o[row, colOffset + j] += scale * sum_r (x[row] . A[r]) * B[j, r] for j &lt; out, rows of <paramref name="rowStride"/> floats. No-op when the
    /// projection has no adapter. Two dispatches: x A^T into a [rows, rank] scratch, then the rank-wide accumulate (one thread per output value).
    /// </summary>
    public void Apply(int layer, string target, ArrayView1D<float, Stride1D.Dense> x, ArrayView1D<float, Stride1D.Dense> o, int rows, int colOffset, int rowStride)
    {
        if (!_adapters.TryGetValue((layer, target), out var ad)) return;
        long need = (long)rows * Rank;
        if (_t == null || _t.Length < need)
        {
            if (_t != null) _retired.Add(_t);
            _t = _acc.Allocate1D<float>(Math.Max(need, 64L * Rank));
        }
        _down ??= _acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, int, int>(DownImpl);
        _up ??= _acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, int, int, int, int, float>(UpImpl);
        if (GroupedDown && _acc.AcceleratorType != AcceleratorType.WebGL && _acc.MaxNumThreadsPerGroup >= DownGroup)
        {
            _downGrouped ??= _acc.LoadStreamKernel<ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>, int, int>(DownGroupedImpl);
            _downGrouped(new KernelConfig((int)need, DownGroup), x, ad.AT.View, _t.View, ad.In, Rank);
        }
        else _down((int)need, x, ad.AT.View, _t.View, ad.In, Rank);
        _up(rows * ad.Out, _t.View, ad.BT.View, o, ad.Out, Rank, colOffset, rowStride, Scale);
    }

    /// <summary>
    /// x A^T with one 64-thread group per output value (default on; false = one thread per value, the A/B control and WebGL's path). At decode
    /// there are only rows * rank values (16 at rank 16), so the per-thread kernel had 16 threads each walking all of x serially.
    /// </summary>
    public static bool GroupedDown { get; set; } = true;
    private const int DownGroup = 64;
    private Action<KernelConfig, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int>? _downGrouped;

    // group g = output (row, r); its threads stride x's row and reduce in shared memory (fixed order: deterministic)
    private static void DownGroupedImpl(ArrayView1D<float, Stride1D.Dense> x, ArrayView1D<float, Stride1D.Dense> aT,
        ArrayView1D<float, Stride1D.Dense> t, int fin, int rank)
    {
        var sh = SharedMemory.Allocate<float>(DownGroup);
        int g = Grid.IdxX, tid = Group.IdxX;
        int row = g / rank, r = g - row * rank;
        int xb = row * fin;
        float s = 0f;
        for (int i = tid; i < fin; i += DownGroup) s += x[xb + i] * aT[i * rank + r];
        sh[tid] = s;
        Group.Barrier();
        for (int w = DownGroup / 2; w > 0; w >>= 1)
        {
            if (tid < w) sh[tid] += sh[tid + w];
            Group.Barrier();
        }
        if (tid == 0) t[g] = sh[0];
    }

    // t[row, r] = sum_i x[row, i] * AT[i, r]
    private static void DownImpl(Index1D idx, ArrayView1D<float, Stride1D.Dense> x, ArrayView1D<float, Stride1D.Dense> aT,
        ArrayView1D<float, Stride1D.Dense> t, int fin, int rank)
    {
        int row = idx / rank, r = idx - row * rank;
        int xb = row * fin;
        float s = 0f;
        for (int i = 0; i < fin; i++) s += x[xb + i] * aT[i * rank + r];
        t[idx] = s;
    }

    // o[row * stride + off + j] += scale * sum_r t[row, r] * BT[r, j]
    private static void UpImpl(Index1D idx, ArrayView1D<float, Stride1D.Dense> t, ArrayView1D<float, Stride1D.Dense> bT,
        ArrayView1D<float, Stride1D.Dense> o, int fout, int rank, int colOffset, int rowStride, float scale)
    {
        int row = idx / fout, j = idx - row * fout;
        float s = 0f;
        for (int r = 0; r < rank; r++) s += t[row * rank + r] * bT[r * fout + j];
        int oi = row * rowStride + colOffset + j;
        o[oi] = o[oi] + scale * s;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        foreach (var (a, b, _, _) in _adapters.Values) { a.Dispose(); b.Dispose(); }
        _adapters.Clear();
        _t?.Dispose(); _t = null;
        foreach (var r in _retired) r.Dispose();
        _retired.Clear();
    }
}
