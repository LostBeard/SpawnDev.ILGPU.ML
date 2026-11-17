using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.SkinTokens;
using SpawnDev.ILGPU.ML.Trellis2;
using FV = ILGPU.Runtime.ArrayView1D<float, ILGPU.Stride1D.Dense>;

namespace SpawnDev.ILGPU.ML.Ltx2;

/// <summary>
/// Gemma 3 12B (Hugging Face transformers' Gemma3TextModel, Apache-2.0) as LTX-2's text encoder: the prompt's real tokens through the 48 layers, every layer's hidden state
/// kept, then LTX-2's per-slot normalisation (diffusers' LTX-2 text connectors, Apache-2.0) into the features the text projection reads, [n, 49 x 3840] channel-major.
/// <para>
/// The weights stream from ComfyUI's text-encoder files (gemma_3_12B_it, _fp8_scaled, _fp4_mixed: NVFP4 and fp8 layers, Kernels.Nvfp4) one layer at a time, so the 12B model
/// runs in a fraction of its size in device memory; the 262k-row token table is never uploaded (the prompt's rows are read from the file).
/// </para>
/// <para>
/// Facts this follows (MEASURED against SpawnGen's ComfyUI taps, 2026-11-17, unless the code is cited): the prompt is left-padded to 1024 ids and the padding never reaches a real
/// token (causal + padding mask), so only the n real tokens are computed, at their absolute positions 1024 - n .. 1023; the embedding is scaled by the fp32 sqrt(3840)
/// (exact); RoPE is rotate_half with theta 1e4 on the sliding layers and 1e6 with a linear factor 8 on every sixth (global) layer (the dumped tables, 6e-8); the sliding
/// window (1024) never binds within 1024 positions; slot 0 is the embedding, slot k the output of layer k - 1 (k = 1..47), slot 48 the final norm's output (layer 47's raw
/// output is not a slot); each slot is normalised 8 (x - mean) / (max - min) over the real tokens x channels.
/// </para>
/// </summary>
public sealed class Gemma3TextEncoder : IDisposable
{
    public const int Hidden = 3840, Layers = 48, QHeads = 16, KvHeads = 8, HeadDim = 256, Inter = 15360, Slots = Layers + 1, SlidingPattern = 6, MaxPositions = 1024;
    public const int QCols = QHeads * HeadDim, KvCols = KvHeads * HeadDim, QkvCols = QCols + 2 * KvCols;
    public const float RmsEps = 1e-6f, LocalTheta = 10_000f, GlobalTheta = 1_000_000f, GlobalLinearFactor = 8f, SlotScale = 8f;

    /// <summary>The width of the features <see cref="EncodeAsync"/> returns per token (49 x 3840 = 188160).</summary>
    public const int FeatureWidth = Slots * Hidden;

    private readonly Accelerator _acc;
    private readonly StreamedModelWeights _w;
    private readonly DiTKernels _k;
    private readonly Gemma3Kernels _g;
    private readonly SkinTokensKernels _u;
    private readonly Flux2.Flux2Kernels _f;
    private readonly List<IDisposable> _owned = new();
    private int _cap;                                                          // the token count the buffers hold
    private MemoryBuffer1D<float, Stride1D.Dense> _h = null!, _hn = null!, _qkv = null!, _part = null!, _attn = null!, _proj = null!, _gate = null!, _up = null!, _act = null!;
    private MemoryBuffer1D<float, Stride1D.Dense> _scores = null!, _kc = null!, _vc = null!, _slots = null!, _features = null!, _xs = null!, _cosL = null!, _sinL = null!, _cosG = null!, _sinG = null!;
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _stats;

    /// <summary>Called with ("embed" | "layer{l}" (0-based layer l's output) | "norm", rows [n, 3840]) as the encode runs (tests compare against the reference's taps).</summary>
    public Action<string, FV, int, int>? Probe;

    /// <summary>CUDA: fp32-class activations through the tensor-core GEMMs (on: the ComfyUI reference runs float32).</summary>
    public bool PreciseActivations { get => _k.PreciseActivations; set => _k.PreciseActivations = value; }

    /// <summary>The streamed weights (timers, prefetch, host cache).</summary>
    public StreamedModelWeights Weights => _w;

    private Gemma3TextEncoder(Accelerator accelerator, StreamedModelWeights weights)
    {
        _acc = accelerator; _w = weights;
        _k = new DiTKernels(accelerator) { PreciseActivations = true }; _g = new Gemma3Kernels(accelerator); _u = new SkinTokensKernels(accelerator); _f = new Flux2.Flux2Kernels(accelerator);
        _stats = accelerator.Allocate1D<float>(3 * Slots); _owned.Add(_stats);
    }

    /// <summary>
    /// Open a Gemma 3 12B text-encoder file (any prefix: ComfyUI's <c>model.</c>). <paramref name="residentBytes"/>: layers kept on the device as stored, in order; the rest
    /// stream per encode.
    /// </summary>
    public static async Task<Gemma3TextEncoder> OpenAsync(Accelerator accelerator, IModelSource source, string repo, string file, long residentBytes, CancellationToken ct = default)
    {
        static int BlockOf(string n)
        {
            if (n.StartsWith("layers.", StringComparison.Ordinal)) { int dot = n.IndexOf('.', 7); return int.Parse(n.AsSpan(7, dot - 7)); }
            if (n == "embed_tokens.weight") return StreamedModelWeights.HostOnly;
            return -1;
        }
        var w = await StreamedModelWeights.OpenAsync(accelerator, source, repo, file, "embed_tokens.weight", BlockOf, Layers, residentBytes, null, ct);
        try
        {
            if (!w.HeaderNames.Contains("norm.weight") || !w.HeaderNames.Contains($"layers.{Layers - 1}.mlp.down_proj.weight") || w.HeaderNames.Contains($"layers.{Layers}.mlp.down_proj.weight"))
                throw new InvalidDataException($"'{file}' is not Gemma 3 12B's 48-layer text model");
            return new Gemma3TextEncoder(accelerator, w);
        }
        catch { w.Dispose(); throw; }
    }

    private MemoryBuffer1D<float, Stride1D.Dense> Alloc(long n) { var b = _acc.Allocate1D<float>(Math.Max(1, n)); _owned.Add(b); return b; }

    private async Task EnsureAsync(int n)
    {
        if (n <= _cap) return;
        if (_cap > 0) await _acc.SynchronizeAsync();                           // the old buffers may still be read by queued work
        foreach (var o in _owned) if (!ReferenceEquals(o, _stats)) o.Dispose();
        _owned.RemoveAll(o => !ReferenceEquals(o, _stats));
        long T = n;
        _h = Alloc(T * Hidden); _hn = Alloc(T * Hidden); _qkv = Alloc(T * QkvCols); _part = Alloc(T * QCols); _attn = Alloc(T * QCols); _proj = Alloc(T * Hidden);
        _gate = Alloc(T * Inter); _up = Alloc(T * Inter); _act = Alloc(T * Inter); _xs = Alloc(T * Inter);
        _scores = Alloc((long)QHeads * T * T); _kc = Alloc(T * KvCols); _vc = Alloc(T * KvCols);
        _slots = Alloc(Slots * T * Hidden); _features = Alloc(T * FeatureWidth);
        _cosL = Alloc(T * (HeadDim / 2)); _sinL = Alloc(T * (HeadDim / 2)); _cosG = Alloc(T * (HeadDim / 2)); _sinG = Alloc(T * (HeadDim / 2));
        _cap = n;
    }

    /// <summary>The rotate_half cos / sin tables [n, D / 2] for positions p0 .. p0 + n - 1: the float32 angle position x inv_freq (inv_freq computed in float32, as the reference), its cos / sin evaluated accurately.</summary>
    public static (float[] Cos, float[] Sin) RopeTables(int p0, int n, float theta, float linearFactor)
    {
        int half = HeadDim / 2;
        var inv = new float[half];
        for (int k = 0; k < half; k++) inv[k] = 1f / MathF.Pow(theta, (float)(2 * k) / HeadDim) / linearFactor;
        var c = new float[n * half]; var s = new float[n * half];
        for (int t = 0; t < n; t++)
            for (int k = 0; k < half; k++)
            {
                float a = (float)(p0 + t) * inv[k];
                c[t * half + k] = (float)Math.Cos(a); s[t * half + k] = (float)Math.Sin(a);
            }
        return (c, s);
    }

    /// <summary>True for Gemma 3's global (full attention) layers: every sixth (5, 11, ..., 47); the others are sliding-window layers.</summary>
    public static bool IsGlobal(int layer) => (layer + 1) % SlidingPattern == 0;

    // nn.Linear over n rows; an 8-bit or NVFP4 matrix runs exact: its decode without the matrix's scale, the input times the scale
    private void Lin(GpuWeights w, FV x, string name, FV o, int n, int K, int N)
    {
        float s = _w.ActivationScale(w, name);
        if (s != 1f) { var xs = _xs.View.SubView(0, (long)n * K); _f.ScaleInto(x, xs, s, n * K); x = xs; }
        _k.Linear(x, _w.MatrixExact(w, name), null, o, n, K, N);
    }

    /// <summary>
    /// Encode the prompt's real token ids (BOS first; at most 1024, the padded length the reference uses: the tokens sit at positions 1024 - n ..). Returns the normalised
    /// stacked features [n, <see cref="FeatureWidth"/>] on the device (valid until the next call).
    /// </summary>
    public async Task<FV> EncodeAsync(int[] ids, CancellationToken ct = default)
    {
        int n = ids.Length;
        if (n < 1 || n > MaxPositions) throw new ArgumentException($"1 to {MaxPositions} token ids, got {n}");
        await EnsureAsync(n);
        int p0 = MaxPositions - n, H = Hidden;
        var (cl, sl) = RopeTables(p0, n, LocalTheta, 1f); var (cg, sg) = RopeTables(p0, n, GlobalTheta, GlobalLinearFactor);
        _cosL.View.SubView(0, cl.Length).CopyFromCPU(cl); _sinL.View.SubView(0, sl.Length).CopyFromCPU(sl);
        _cosG.View.SubView(0, cg.Length).CopyFromCPU(cg); _sinG.View.SubView(0, sg.Length).CopyFromCPU(sg);

        // the embedding: the prompt's rows from the file, times sqrt(hidden) in fp32
        var rows = await _w.ReadRowsAsync("embed_tokens.weight", ids, ct);
        float scale = MathF.Sqrt(Hidden);
        for (int i = 0; i < rows.Length; i++) rows[i] *= scale;
        _h.View.SubView(0, rows.Length).CopyFromCPU(rows);
        Probe?.Invoke("embed", _h.View, n, H);
        long slot = (long)n * H;
        _slots.View.SubView(0, slot).CopyFrom(_h.View.SubView(0, slot));

        for (int l = 0; l < Layers; l++)
        {
            var (w, streamed) = await _w.BlockAsync(l, ct);
            try { Layer(w, $"layers.{l}.", n, IsGlobal(l)); }
            finally { if (streamed) { await _acc.SynchronizeAsync(); w.Dispose(); } }
            Probe?.Invoke($"layer{l}", _h.View, n, H);
            if (l < Layers - 1) _slots.View.SubView((l + 1) * slot, slot).CopyFrom(_h.View.SubView(0, slot));
        }
        _g.RmsNorm(_h.View, _hn.View, _w.Base.BF16("norm.weight"), n, H, RmsEps);
        Probe?.Invoke("norm", _hn.View, n, H);
        _slots.View.SubView(Layers * slot, slot).CopyFrom(_hn.View.SubView(0, slot));
        _g.NormalisedStack(_slots.View, _stats.View, _features.View, Slots, n, H, SlotScale);
        return _features.View.SubView(0, (long)n * FeatureWidth);
    }

    private void Layer(GpuWeights w, string b, int n, bool global)
    {
        int H = Hidden;
        _g.RmsNorm(_h.View, _hn.View, w.BF16(b + "input_layernorm.weight"), n, H, RmsEps);
        Lin(w, _hn.View, b + "self_attn.q_proj.weight", _part.View, n, H, QCols); _f.CopyCols(_part.View, _qkv.View, n, QCols, QCols, QkvCols, 0);
        Lin(w, _hn.View, b + "self_attn.k_proj.weight", _part.View, n, H, KvCols); _f.CopyCols(_part.View, _qkv.View, n, KvCols, KvCols, QkvCols, QCols);
        Lin(w, _hn.View, b + "self_attn.v_proj.weight", _part.View, n, H, KvCols); _f.CopyCols(_part.View, _qkv.View, n, KvCols, KvCols, QkvCols, QCols + KvCols);
        _g.RmsNormHeads(_qkv.View, w.BF16(b + "self_attn.q_norm.weight"), n, QkvCols, 0, QHeads, HeadDim, RmsEps);
        _g.RmsNormHeads(_qkv.View, w.BF16(b + "self_attn.k_norm.weight"), n, QkvCols, QCols, KvHeads, HeadDim, RmsEps);
        var (cos, sin) = global ? (_cosG.View, _sinG.View) : (_cosL.View, _sinL.View);
        _u.RopeTables(_qkv.View, cos, sin, n, QkvCols, 0, QHeads, HeadDim);
        _u.RopeTables(_qkv.View, cos, sin, n, QkvCols, QCols, KvHeads, HeadDim);
        _u.StoreKv(_qkv.View, _kc.View, _vc.View, 0, n, QCols, KvCols);
        _f.PaddedCausalGqa(_qkv.View, _kc.View, _vc.View, _scores.View, _attn.View, n, n, QHeads, HeadDim, QHeads / KvHeads);   // scale 1/sqrt(256) = query_pre_attn_scalar^-0.5
        Lin(w, _attn.View, b + "self_attn.o_proj.weight", _proj.View, n, QCols, H);
        _g.RmsNormAddInto(_h.View, _proj.View, w.BF16(b + "post_attention_layernorm.weight"), n, H, RmsEps);
        _g.RmsNorm(_h.View, _hn.View, w.BF16(b + "pre_feedforward_layernorm.weight"), n, H, RmsEps);
        Lin(w, _hn.View, b + "mlp.gate_proj.weight", _gate.View, n, H, Inter);
        Lin(w, _hn.View, b + "mlp.up_proj.weight", _up.View, n, H, Inter);
        _g.GegluTanh(_gate.View, _up.View, _act.View, n * Inter);
        Lin(w, _act.View, b + "mlp.down_proj.weight", _proj.View, n, Inter, H);
        _g.RmsNormAddInto(_h.View, _proj.View, w.BF16(b + "post_feedforward_layernorm.weight"), n, H, RmsEps);
    }

    public void Dispose()
    {
        foreach (var o in _owned) o.Dispose();
        _owned.Clear();
        _w.Dispose();
    }
}
