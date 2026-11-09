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
    private readonly Dictionary<(int Layer, string Target), (float[] AT, float[] BT)> _host = new();   // source of the fused packs
    // fused packs for projections that share an input and write adjacent columns (q|k|v, gate|up); null = not fusable (a part has no adapter)
    private readonly Dictionary<(int Layer, string Key), Fused?> _fused = new();
    private sealed record Fused(MemoryBuffer1D<float, Stride1D.Dense> AT, MemoryBuffer1D<float, Stride1D.Dense> BT, int In, int Out, int Parts, int E1, int E2);
    private MemoryBuffer1D<float, Stride1D.Dense>? _t;
    private readonly List<MemoryBuffer1D<float, Stride1D.Dense>> _retired = new();   // a queued launch may still read a replaced scratch
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int>? _down;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, UpDims>? _up;

    /// <summary>The up kernel's scalars: total output width, rank, the t row stride (parts * rank), the part boundaries E1 / E2 in output
    /// columns (a column j belongs to part 0 below E1, part 1 below E2, else part 2), the column offset and row stride of o, and the scale.</summary>
    public struct UpDims
    {
        /// <summary>Output columns written.</summary>
        public int Out;
        /// <summary>LoRA rank.</summary>
        public int Rank;
        /// <summary>Floats per row of the down scratch (parts * rank).</summary>
        public int TStride;
        /// <summary>First column of part 1.</summary>
        public int E1;
        /// <summary>First column of part 2.</summary>
        public int E2;
        /// <summary>Column of o where output column 0 lands.</summary>
        public int ColOffset;
        /// <summary>Floats per row of o.</summary>
        public int RowStride;
        /// <summary>alpha / rank.</summary>
        public float Scale;
    }

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
            int layer = int.Parse(m.Groups[1].Value);
            _adapters[(layer, target)] = (accelerator.Allocate1D(aT), accelerator.Allocate1D(bT), fin, fout);
            _host[(layer, target)] = (aT, bT);
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
        Launch(ad.AT, ad.BT, ad.In, ad.Out, 1, ad.Out, ad.Out, x, o, rows, colOffset, rowStride);
    }

    /// <summary>
    /// Fuse adjacent projections (default on): <see cref="ApplyAdjacent"/> runs q|k|v (or gate|up) as ONE down and ONE up dispatch over packed
    /// adapters instead of two per projection. false = one <see cref="Apply"/> per projection (the A/B control). Same arithmetic per output value:
    /// each column still sums its own part's rank terms in the same order.
    /// </summary>
    public static bool FuseAdjacent { get; set; } = true;

    /// <summary>
    /// Apply the adapters of up to three projections that read the same input <paramref name="x"/> and write adjacent columns of <paramref name="o"/>
    /// (a fused q|k|v or gate|up output): <paramref name="targets"/>[i] lands at column <paramref name="colOffsets"/>[i]. When every part has an
    /// adapter and the parts are contiguous, one packed launch (<see cref="FuseAdjacent"/>); otherwise one <see cref="Apply"/> per present part.
    /// </summary>
    public void ApplyAdjacent(int layer, string[] targets, int[] colOffsets, ArrayView1D<float, Stride1D.Dense> x, ArrayView1D<float, Stride1D.Dense> o, int rows, int rowStride)
    {
        if (targets.Length != colOffsets.Length || targets.Length is < 1 or > 3) throw new ArgumentException("1 to 3 targets, one column offset each");
        if (FuseAdjacent && targets.Length > 1 && GetFused(layer, targets, colOffsets) is { } f)
        {
            Launch(f.AT, f.BT, f.In, f.Out, f.Parts, f.E1, f.E2, x, o, rows, colOffsets[0], rowStride);
            return;
        }
        for (int i = 0; i < targets.Length; i++) Apply(layer, targets[i], x, o, rows, colOffsets[i], rowStride);
    }

    private Fused? GetFused(int layer, string[] targets, int[] colOffsets)
    {
        var key = (layer, string.Join("|", targets));
        if (_fused.TryGetValue(key, out var hit)) return hit;
        Fused? f = null;
        var parts = targets.Select(t => _adapters.TryGetValue((layer, t), out var a) ? a : default).ToArray();
        bool ok = parts.All(a => a.AT != null) && parts.All(a => a.In == parts[0].In);
        for (int i = 1; ok && i < parts.Length; i++) ok = colOffsets[i] == colOffsets[i - 1] + parts[i - 1].Out;
        if (ok)
        {
            int P = parts.Length, fin = parts[0].In, total = parts.Sum(a => a.Out), tw = P * Rank;
            var aT = new float[(long)fin * tw];
            var bT = new float[(long)Rank * total];
            int cum = 0;
            for (int s = 0; s < P; s++)
            {
                var (ha, hb) = _host[(layer, targets[s])];
                int fo = parts[s].Out;
                for (int i = 0; i < fin; i++) Array.Copy(ha, (long)i * Rank, aT, (long)i * tw + s * Rank, Rank);
                for (int r = 0; r < Rank; r++) Array.Copy(hb, (long)r * fo, bT, (long)r * total + cum, fo);
                cum += fo;
            }
            int e1 = P > 1 ? parts[0].Out : total, e2 = P > 2 ? parts[0].Out + parts[1].Out : total;
            f = new Fused(_acc.Allocate1D(aT), _acc.Allocate1D(bT), fin, total, P, e1, e2);
        }
        _fused[key] = f;
        return f;
    }

    private void Launch(MemoryBuffer1D<float, Stride1D.Dense> aTBuf, MemoryBuffer1D<float, Stride1D.Dense> bTBuf, int fin, int fout, int parts, int e1, int e2,
        ArrayView1D<float, Stride1D.Dense> x, ArrayView1D<float, Stride1D.Dense> o, int rows, int colOffset, int rowStride)
    {
        int tw = parts * Rank;
        long need = (long)rows * tw;
        if (_t == null || _t.Length < need)
        {
            if (_t != null) _retired.Add(_t);
            _t = _acc.Allocate1D<float>(Math.Max(need, 64L * 3 * Rank));
        }
        _down ??= _acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, int, int>(DownImpl);
        _up ??= _acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, UpDims>(UpImpl);
        if (PrefillGemm && rows >= PrefillGemmRows)
        {
            // prefill: x [rows, in] x A^T [in, parts * rank] is a real GEMM; the library's tiled kernel reuses each x tile across 16 outputs
            _mm ??= new MatMulKernel(_acc);
            _mm.MatMul(x, aTBuf.View, _t.View, rows, fin, tw);
        }
        else if (GroupedDown && _acc.AcceleratorType != AcceleratorType.WebGL && _acc.MaxNumThreadsPerGroup >= DownGroup)
        {
            _downGrouped ??= _acc.LoadStreamKernel<ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>, int, int>(DownGroupedImpl);
            _downGrouped(new KernelConfig((int)need, DownGroup), x, aTBuf.View, _t.View, fin, tw);
        }
        else _down((int)need, x, aTBuf.View, _t.View, fin, tw);
        _up(rows * fout, _t.View, bTBuf.View, o, new UpDims { Out = fout, Rank = Rank, TStride = tw, E1 = e1, E2 = e2, ColOffset = colOffset, RowStride = rowStride, Scale = Scale });
    }

    /// <summary>
    /// x A^T with one 64-thread group per output value (default on; false = one thread per value, the A/B control and WebGL's path). At decode
    /// there are only rows * rank values (16 at rank 16), so the per-thread kernel had 16 threads each walking all of x serially.
    /// </summary>
    public static bool GroupedDown { get; set; } = true;

    /// <summary>
    /// From <see cref="PrefillGemmRows"/> rows up (a prefill), x A^T runs as a tiled GEMM (<see cref="MatMulKernel"/>) instead of one reduction
    /// group per value (default on; false = the A/B control). MEASURED 2026-11-04, SmolLM2-360M, 7 projections r16, 256-token prompt, RTX 4070
    /// shared with a training run: the per-value groups re-read each x row once per rank column and added ~50 ms to a 176 ms first token.
    /// </summary>
    public static bool PrefillGemm { get; set; } = true;
    /// <summary>The row count from which <see cref="PrefillGemm"/> applies (decode steps stay on the grouped GEMV).</summary>
    public static int PrefillGemmRows { get; set; } = 16;
    private MatMulKernel? _mm;
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
        for (int i = tid; i < fin; i += DownGroup) s = float.MultiplyAddEstimate(x[xb + i], aT[i * rank + r], s);
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
        for (int i = 0; i < fin; i++) s = float.MultiplyAddEstimate(x[xb + i], aT[i * rank + r], s);
        t[idx] = s;
    }

    // o[row * stride + off + j] += scale * sum_r t[row, part(j) * rank + r] * BT[r, j]; part(j) = 0 below E1, 1 below E2, else 2
    private static void UpImpl(Index1D idx, ArrayView1D<float, Stride1D.Dense> t, ArrayView1D<float, Stride1D.Dense> bT,
        ArrayView1D<float, Stride1D.Dense> o, UpDims d)
    {
        int row = idx / d.Out, j = idx - row * d.Out;
        int part = j < d.E1 ? 0 : j < d.E2 ? 1 : 2;
        int tb = row * d.TStride + part * d.Rank;
        float s = 0f;
        for (int r = 0; r < d.Rank; r++) s = float.MultiplyAddEstimate(t[tb + r], bT[r * d.Out + j], s);
        int oi = row * d.RowStride + d.ColOffset + j;
        o[oi] = float.MultiplyAddEstimate(d.Scale, s, o[oi]);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        foreach (var (a, b, _, _) in _adapters.Values) { a.Dispose(); b.Dispose(); }
        _adapters.Clear();
        foreach (var f in _fused.Values) if (f != null) { f.AT.Dispose(); f.BT.Dispose(); }
        _fused.Clear();
        _host.Clear();
        _t?.Dispose(); _t = null;
        foreach (var r in _retired) r.Dispose();
        _retired.Clear();
    }
}
