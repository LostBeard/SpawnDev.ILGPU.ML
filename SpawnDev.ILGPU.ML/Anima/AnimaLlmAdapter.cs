using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Trellis2;

namespace SpawnDev.ILGPU.ML.Anima;

using FV = ArrayView1D<float, Stride1D.Dense>;

/// <summary>
/// Anima's LLM adapter (<c>net.llm_adapter.*</c> in the Anima file): turns Qwen3-0.6B's final hidden states (after its last norm, ids without
/// BOS/EOS) and the prompt's T5 token ids (EOS = 1 last) into the DiT's context rows [M, 1024]. Derived from the weight header and black-box
/// taps only (tools/anima-ref/ANIMA-SPEC.md section 3; float64 reference within 1.3e-6 end to end):
/// T5 embedding gather (in_proj is the identity) -> 6 pre-norm blocks (RMSNorm eps 1e-6; self-attention on the T5 side; cross-attention with
/// q from the T5 side and k/v from the RAW Qwen3 states; 16 heads x 64 with per-head q/k RMSNorm and rotate-half RoPE base 10000 on q AND k,
/// each side with its own positions 0..len-1; MLP 1024 -> 4096 -> 1024 with biases and exact GELU; plain residuals) -> out_proj -> RMSNorm.
/// No masks anywhere. Pass the output (unpadded) to <see cref="AnimaDiT.SetContext(int, FV, int)"/>, which pads it to 512 rows.
/// </summary>
public sealed class AnimaLlmAdapter : IDisposable
{
    /// <summary>The adapter's tensor-name prefix in the Anima file.</summary>
    public const string Prefix = "net.llm_adapter.";
    /// <summary>Width, heads and head width of the adapter (and of Qwen3-0.6B's states it reads).</summary>
    public const int Dim = 1024, Heads = 16, HeadDim = 64;
    private readonly Accelerator _acc;
    private readonly DiTKernels _k;
    private readonly Wan.WanKernels _wk;
    private readonly AnimaKernels _ak;
    private readonly GpuWeights _w;
    private readonly Flux2.Flux2Weights _mat;
    private readonly int _blocks, _ffn, _vocab;

    private int _capM, _capN;
    private readonly List<MemoryBuffer1D<float, Stride1D.Dense>> _ws = new();
    private MemoryBuffer1D<int, Stride1D.Dense>? _ids;
    private MemoryBuffer1D<float, Stride1D.Dense>? _phases;
    private FV _x, _n, _q, _kv, _attn, _proj, _mlp, _scores, _out;

    /// <summary>Test hook: (stage name, activation view, element count); valid only during the call.</summary>
    public Action<string, FV, int>? Tap { get; set; }

    /// <summary>The adapter over weights already on the device (not owned: the caller disposes them).</summary>
    /// <param name="accelerator">the application's accelerator (never disposed here)</param>
    /// <param name="kernels">the transformer kernels (<see cref="PreciseActivations"/> sets them)</param>
    /// <param name="weights">the Anima file's weights (shared with <see cref="AnimaDiT"/>; not owned)</param>
    public AnimaLlmAdapter(Accelerator accelerator, DiTKernels kernels, GpuWeights weights)
    {
        _acc = accelerator; _k = kernels; _wk = new Wan.WanKernels(accelerator); _ak = new AnimaKernels(accelerator); _w = weights;
        _mat = new Flux2.Flux2Weights(accelerator, weights, ownsWeights: false);
        var embed = weights.Shape(Prefix + "embed.weight");
        if (embed[1] != Dim) throw new NotSupportedException($"AnimaLlmAdapter: width {embed[1]}, expected {Dim}");
        _vocab = embed[0];
        _ffn = weights.Shape(Prefix + "blocks.0.mlp.0.weight")[0];
        _blocks = 0; while (weights.Has($"{Prefix}blocks.{_blocks}.self_attn.q_proj.weight")) _blocks++;
        if (weights.Has(Prefix + "in_proj.weight")) throw new NotSupportedException("AnimaLlmAdapter: this file has an in_proj; the measured checkpoint's is the identity");
    }

    /// <summary>Adapter blocks in the file (6).</summary>
    public int Layers => _blocks;

    /// <summary>CUDA: fp32-class activations through the tensor-core GEMMs (<see cref="DiTKernels.PreciseActivations"/>). Other backends compute in fp32 either way.</summary>
    public bool PreciseActivations { get => _k.PreciseActivations; set => _k.PreciseActivations = value; }

    private FV Alloc(long n) { var b = _acc.Allocate1D<float>(n); _ws.Add(b); return b.View; }

    // Sized for the largest call so far. Growing frees buffers earlier calls' dispatches used, so it drains the queue first (async).
    private async Task EnsureCapacityAsync(int M, int N)
    {
        if (M <= _capM && N <= _capN) return;
        await _acc.SynchronizeAsync();
        foreach (var b in _ws) b.Dispose();
        _ws.Clear();
        _ids?.Dispose(); _phases?.Dispose();
        _capM = Math.Max(M, Math.Max(_capM, 64)); _capN = Math.Max(N, Math.Max(_capN, 64));
        int rows = Math.Max(_capM, _capN), C = Dim;
        _x = Alloc((long)_capM * C); _n = Alloc((long)_capM * C); _q = Alloc((long)_capM * C); _kv = Alloc(2L * rows * C);
        _attn = Alloc((long)_capM * C); _proj = Alloc((long)_capM * C); _mlp = Alloc((long)_capM * _ffn); _out = Alloc((long)_capM * C);
        _scores = Alloc(_k.UsesFlashAttention(HeadDim) ? 1 : (long)Heads * _capM * rows);
        _ids = _acc.Allocate1D<int>(_capM);
        var table = AnimaKernels.AdapterRopePhases(rows);   // one table: both sides use positions 0..len-1 with the same formula
        _phases = _acc.Allocate1D<float>(table.Length);
        _phases.View.CopyFromCPU(table);
    }

    /// <summary>
    /// Context rows for one prompt: <paramref name="qwenStates"/> [N, 1024] (device, Qwen3's final normed hidden states) and
    /// <paramref name="t5Ids"/> [M] (EOS last). The returned view [M, 1024] is the adapter's own buffer: valid until the next call.
    /// </summary>
    public async Task<FV> ForwardAsync(FV qwenStates, int N, int[] t5Ids)
    {
        int M = t5Ids.Length, C = Dim, H = Heads, D = HeadDim;
        if (M < 1 || M > AnimaDiT.ContextRows) throw new ArgumentException($"{M} T5 ids; the DiT context holds 1..{AnimaDiT.ContextRows}");
        if (N < 1 || qwenStates.Length < (long)N * C) throw new ArgumentException($"the Qwen3 states hold fewer than {N} x {C} values");
        foreach (int id in t5Ids) if ((uint)id >= (uint)_vocab) throw new ArgumentOutOfRangeException(nameof(t5Ids), $"T5 id {id} is outside the {_vocab}-row table");
        await EnsureCapacityAsync(M, N);
        _ids!.View.SubView(0, M).CopyFromCPU(t5Ids);
        string P = Prefix;
        FV x = _x.SubView(0, (long)M * C), n = _n.SubView(0, (long)M * C), q = _q.SubView(0, (long)M * C);
        FV attn = _attn.SubView(0, (long)M * C), proj = _proj.SubView(0, (long)M * C), y = _out.SubView(0, (long)M * C);
        FV phases = _phases!.View;

        _ak.GatherRows(_w.BF16(P + "embed.weight"), _ids.View, x, M, C);
        Tap?.Invoke("embed", x, M * C);
        for (int b = 0; b < _blocks; b++)
        {
            string p = $"{P}blocks.{b}.";

            // self attention (T5 side positions on q and k)
            n.CopyFrom(x);
            _wk.RowRmsNorm(n, _w.BF16(p + "norm_self_attn.weight"), M, C, 0, C, 1e-6f);
            FV k = _kv.SubView(0, (long)M * C), v = _kv.SubView((long)M * C, (long)M * C);
            _mat.Linear(_k, n, p + "self_attn.q_proj.weight", null, q, M, C, C);
            _mat.Linear(_k, n, p + "self_attn.k_proj.weight", null, k, M, C, C);
            _mat.Linear(_k, n, p + "self_attn.v_proj.weight", null, v, M, C, C);
            _wk.RowRmsNorm(q, _w.BF16(p + "self_attn.q_norm.weight"), M * H, D, 0, D, 1e-6f);
            _wk.RowRmsNorm(k, _w.BF16(p + "self_attn.k_norm.weight"), M * H, D, 0, D, 1e-6f);
            _ak.RopeHalf(q, phases, M, C, 0, H, D);
            _ak.RopeHalf(k, phases, M, C, 0, H, D);
            _k.Attention(q, C, 0, _kv, C, 0, M * C, _scores, attn, M, M, H, D);
            _mat.Linear(_k, attn, p + "self_attn.o_proj.weight", null, proj, M, C, C);
            _k.GateResidual(x, proj, default, -1, M * C, C);

            // cross attention: q from the T5 side, k / v from the raw Qwen3 states, each with its own positions
            n.CopyFrom(x);
            _wk.RowRmsNorm(n, _w.BF16(p + "norm_cross_attn.weight"), M, C, 0, C, 1e-6f);
            FV kc = _kv.SubView(0, (long)N * C), vc = _kv.SubView((long)N * C, (long)N * C);
            _mat.Linear(_k, n, p + "cross_attn.q_proj.weight", null, q, M, C, C);
            _mat.Linear(_k, qwenStates, p + "cross_attn.k_proj.weight", null, kc, N, C, C);
            _mat.Linear(_k, qwenStates, p + "cross_attn.v_proj.weight", null, vc, N, C, C);
            _wk.RowRmsNorm(q, _w.BF16(p + "cross_attn.q_norm.weight"), M * H, D, 0, D, 1e-6f);
            _wk.RowRmsNorm(kc, _w.BF16(p + "cross_attn.k_norm.weight"), N * H, D, 0, D, 1e-6f);
            _ak.RopeHalf(q, phases, M, C, 0, H, D);
            _ak.RopeHalf(kc, phases, N, C, 0, H, D);
            _k.Attention(q, C, 0, _kv, C, 0, N * C, _scores, attn, M, N, H, D);
            _mat.Linear(_k, attn, p + "cross_attn.o_proj.weight", null, proj, M, C, C);
            _k.GateResidual(x, proj, default, -1, M * C, C);

            // MLP with biases, exact GELU
            n.CopyFrom(x);
            _wk.RowRmsNorm(n, _w.BF16(p + "norm_mlp.weight"), M, C, 0, C, 1e-6f);
            _mat.Linear(_k, n, p + "mlp.0.weight", _w.BF16(p + "mlp.0.bias"), _mlp, M, C, _ffn, DiTKernels.Act.GeluErf);
            _mat.Linear(_k, _mlp, p + "mlp.2.weight", _w.BF16(p + "mlp.2.bias"), proj, M, _ffn, C);
            _k.GateResidual(x, proj, default, -1, M * C, C);

            Tap?.Invoke($"block_{b}", x, M * C);
            _acc.Submit();
        }

        // out_proj BEFORE the final norm
        _mat.Linear(_k, x, P + "out_proj.weight", _w.BF16(P + "out_proj.bias"), y, M, C, C);
        Tap?.Invoke("out_proj", y, M * C);
        _wk.RowRmsNorm(y, _w.BF16(P + "norm.weight"), M, C, 0, C, 1e-6f);
        _acc.Fence();
        return y;
    }

    /// <summary>Frees the work buffers (not the weights, not the accelerator). Drain the queue first (SynchronizeAsync) on browser backends.</summary>
    public void Dispose()
    {
        _mat.Dispose();
        foreach (var b in _ws) b.Dispose();
        _ws.Clear();
        _ids?.Dispose(); _phases?.Dispose();
    }
}
