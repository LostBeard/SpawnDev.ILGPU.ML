using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.SkinTokens;
using SpawnDev.ILGPU.ML.Trellis2;
using FV = ILGPU.Runtime.ArrayView1D<float, ILGPU.Stride1D.Dense>;

namespace SpawnDev.ILGPU.ML.Ltx2;

/// <summary>
/// The LTX-2 19B audio + video transformer (diffusers' LTX2VideoTransformer3DModel, Apache-2.0; the checkpoint's own config: split RoPE, no gated attention, no prompt adaLN):
/// 48 blocks of a video stream (32 x 128 = 4096) and an audio stream (32 x 64 = 2048). Each block: video and audio self-attention (adaLN-modulated RMS norm, gated
/// residual), text cross-attention (plain RMS norm, ungated), audio-to-video then video-to-audio cross-attention (the second reads the video after the first's update, as
/// ComfyUI runs it; audio head size, time-only RoPE), and the gated feed-forwards. The q / k norms use epsilon 1e-5. Output: velocity for both streams (x0 = x - sigma v).
/// <para>
/// One call runs a batch (the guidance pair: [cond, uncond]) through each block while the block's weights are loaded, so a step streams the 19B weights once. Blocks stream
/// from the checkpoint as stored (fp8 or bf16); the base set (timestep embedders, projections) stays resident.
/// </para>
/// </summary>
public sealed class Ltx2Transformer : IDisposable
{
    public const int Dv = 4096, Da = 2048, Hv = 32, HeadV = 128, Ha = 32, HeadA = 64, Layers = 48, InChannels = 128, Caption = 3840, TimeChannels = 256, Fv = 4 * Dv, Fa = 4 * Da;
    public const float Eps = 1e-6f;
    /// <summary>
    /// The epsilon of the attention q / k RMS norms: 1e-5 (MEASURED 2026-11-17 against ComfyUI's block 0, float64 reference: 1e-6 leaves the video stream 1.9e-3 off, 1e-5 3e-7).
    /// The block norms use <see cref="Eps"/>.
    /// </summary>
    public const float QkEps = 1e-5f;
    public const double RopeTheta = 10_000.0;
    /// <summary>The positional normalisers of the checkpoint config (positional_embedding_max_pos [20, 2048, 2048]; audio [20]).</summary>
    public const float MaxTime = 20f, MaxSpace = 2048f;
    /// <summary>The VAE's compression: 8 frames and 32 x 32 pixels per latent voxel; the audio latent's frame is 4 mel frames of 160 samples at 16 kHz.</summary>
    public const int TimeScale = 8, SpaceScale = 32, AudioScale = 4, AudioHop = 160, AudioRate = 16000;

    private readonly Accelerator _acc;
    private readonly StreamedModelWeights _w;
    private readonly DiTKernels _k;
    private readonly Ltx.LtxKernels _x;
    private readonly Ltx2Kernels _l;
    private readonly SkinTokensKernels _u;
    private readonly Flux2.Flux2Kernels _f;
    private readonly List<IDisposable> _owned = new();
    private readonly int _B, _nv, _na, _L;
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _v, _a, _vn, _an, _vctx, _actx, _q, _kv, _part, _o, _hid, _scores, _xs;
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _sinA, _temb, _tokBias;
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _tA, _tAe, _tCaA, _tGateA;                       // the audio embedders' outputs (per batch item)
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _modA, _caA, _gateA, _headA;
    // the video embedders run per (batch item, frame group): a frame's tokens can carry their own timestep (LTXVAddGuide / image conditioning: sigma x the frame's
    // denoise mask). One group per item when every frame shares the timestep (the plain call, exactly as before)
    private MemoryBuffer1D<float, Stride1D.Dense> _sinV = null!, _t1 = null!, _tembS = null!, _tV = null!, _tVe = null!, _tCaV = null!, _tGateV = null!, _tembV = null!;
    private MemoryBuffer1D<float, Stride1D.Dense> _modV = null!, _caV = null!, _gateV = null!, _headV = null!;
    private int _gCap, _vg = 1, _rpg;                                                                        // group capacity, groups per item, token rows per group
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _ropeV, _ropeVs, _ropeA, _ropeAs, _ropeVx, _ropeVxs, _outV, _outA;

    /// <summary>Called with ("block{k}_video" [B nv, 4096] | "block{k}_audio" [B na, 2048] | "adaln" ... ) as a call goes.</summary>
    public Action<string, FV, int, int>? Probe;

    /// <summary>CUDA: fp32-class activations through the tensor-core GEMMs (on: the ComfyUI reference runs float32).</summary>
    public bool PreciseActivations { get => _k.PreciseActivations; set => _k.PreciseActivations = value; }
    /// <summary>CUDA with tensor cores, <see cref="PreciseActivations"/> on: the attentions run on the tensor cores with q, k, v and the probabilities as two bf16 halves each (<see cref="DiTKernels.ExactTensorCoreAttention"/>). On by default; false runs the fp32 kernel.</summary>
    public bool ExactTensorCoreAttention { get => _k.ExactTensorCoreAttention; set => _k.ExactTensorCoreAttention = value; }

    /// <summary>The streamed weights (timers, prefetch, host cache).</summary>
    public StreamedModelWeights Weights => _w;

    private readonly List<Ltx2Lora> _loras = new();
    private GpuWeights?[] _loraBlock = Array.Empty<GpuWeights?>();     // each LoRA's tensors of the block being run
    private MemoryBuffer1D<float, Stride1D.Dense>? _loraT, _loraU;

    /// <summary>The LoRAs applied to every call, in order (each: <c>+ strength (x A^T) B^T</c> on the Linears it covers). The caller owns and disposes them.</summary>
    public IReadOnlyList<Ltx2Lora> Loras => _loras;

    /// <summary>Apply <paramref name="lora"/> from the next call on (ComfyUI's LoraLoaderModelOnly; its <see cref="Ltx2Lora.Strength"/> is read at every call).</summary>
    public void AddLora(Ltx2Lora lora)
    {
        _loras.Add(lora);
        int rank = _loras.Max(l => l.Rank);
        long rows = (long)_B * Math.Max(_nv, Math.Max(_na, _L));
        if (_loraT == null || _loraT.Length < rows * rank) { _loraT?.Dispose(); _loraT = _acc.Allocate1D<float>(rows * rank); }
        _loraU ??= _acc.Allocate1D<float>((long)_B * Math.Max((long)_nv * Fv, Math.Max((long)_na * Fa, (long)_L * Dv)));
    }

    /// <summary>Stop applying <paramref name="lora"/> (it stays open; the caller disposes it).</summary>
    public bool RemoveLora(Ltx2Lora lora) => _loras.Remove(lora);

    /// <summary>
    /// How a LoRA meets a Linear whose checkpoint weight is 8-bit (F8_E4M3). False (default): exact, <c>x (W + strength B A)^T</c> in float32, the LoRA's full
    /// effect. True: ComfyUI's result: the patched weight is re-rounded to F8_E4M3 with its seeded stochastic rounding (<see cref="Ltx2Fp8Rounding"/>: reproduced
    /// black box; ~4% noise per weight, so outputs equal ComfyUI's and differ from the exact ones by a third of a one-block LoRA's effect, MEASURED). Costs a
    /// re-rounding of every patched 8-bit matrix on every call (the model streams, so nothing is kept): MEASURED 2026-11-17 on the distilled LoRA (all 48 blocks),
    /// 768 x 512 x 121, RTX 4070: 53.4 s a step against 52.4 s; 0.8 GB of device memory and <see cref="CompatHostBytes"/> of host memory. bf16 Linears are exact either way.
    /// </summary>
    public bool ComfyCompatibleFp8Lora { get; set; }

    // the compatible mode's scratch, sized once for the largest patched matrix (16384 x 4096) at the first call that needs it
    private MemoryBuffer1D<float, Stride1D.Dense>? _cw, _cba, _cB, _cRowMax, _cScale;
    private MemoryBuffer1D<BFloat16, Stride1D.Dense>? _cq, _cAt;
    private MemoryBuffer1D<uint, Stride1D.Dense>? _cRnd;
    private uint[]? _cHost;
    private int _cRank;
    // a block's streams, started on the thread pool when its weights are in (a stream depends on its tensor's name alone): each a slice of ONE host array the
    // transformer owns, sized for the largest block. (Pooled arrays per stream, the first form: 5.8 GB of committed host memory after a clip, MEASURED 2026-11-17.)
    private uint[]? _cBlockHost;
    private readonly Dictionary<string, (Task Task, int Offset, int Words)> _cStreams = new();
    /// <summary>Seconds the last call's own thread spent on the compatible mode's random streams, generating them or waiting for the thread pool's (0 when the mode is off).</summary>
    public double CompatStreamSeconds { get; private set; }
    /// <summary>Host memory the compatible mode holds for its streams, bytes (0 when it has not run).</summary>
    public long CompatHostBytes => 4L * ((_cBlockHost?.LongLength ?? 0) + (_cHost?.LongLength ?? 0));

    // Start the streams of every patched 8-bit Linear of the block on the thread pool: a 16384 x 4096 matrix's 67 M draws are sequential by nature (one generator),
    // but the block's 28 matrices are independent. Not in a browser (one thread: LinFp8Compat generates there).
    private void StartCompatStreams(GpuWeights w)
    {
        if (OperatingSystem.IsBrowser()) return;
        var plan = new List<(string Name, int Cnt)>(); long total = 0;
        for (int i = 0; i < _loras.Count; i++)
        {
            if (_loraBlock[i] is not { } lw) continue;
            foreach (var n in lw.Names)
            {
                const string tail = ".lora_A.weight";
                if (!n.EndsWith(tail, StringComparison.Ordinal)) continue;
                string name = n[..^tail.Length];
                if (!w.Has(name + ".weight") || !w.IsF8(name + ".weight") || plan.Any(p => p.Name == name)) continue;
                var shape = w.Shape(name + ".weight");
                int cnt = checked(shape[0] * shape[1]);
                plan.Add((name, cnt)); total += (cnt + 3) / 4;
            }
        }
        if (_cBlockHost == null || _cBlockHost.Length < total) _cBlockHost = GC.AllocateUninitializedArray<uint>(checked((int)total));
        var host = _cBlockHost; int at = 0;
        foreach (var (name, cnt) in plan)
        {
            int words = (cnt + 3) / 4, offset = at;
            _cStreams[name] = (Task.Run(() => Ltx2Fp8Rounding.Low8(Ltx2Fp8Rounding.Seed(name), host.AsSpan(offset, words), cnt)), offset, words);
            at += words;
        }
    }

    // The block is done: every stream it started has to have FINISHED before the next block's streams write the same array (one nothing consumed, a LoRA tensor for
    // a Linear this call did not run or a call that threw, may still be generating).
    private void FinishCompatStreams()
    {
        foreach (var s in _cStreams.Values) { try { s.Task.Wait(); } catch (AggregateException) { } }
        _cStreams.Clear();
    }

    private async Task EnsureCompatAsync()
    {
        int rank = _loras.Max(l => l.Rank);
        if (_cw != null && rank <= _cRank) return;
        if (_cw != null) { await _acc.SynchronizeAsync(); DisposeCompat(); }                 // a larger rank: nothing queued may still read the old scratch
        long cnt = (long)Fv * Dv, side = Fv;
        _cw = _acc.Allocate1D<float>(cnt); _cba = _acc.Allocate1D<float>(cnt); _cq = _acc.Allocate1D<BFloat16>(cnt); _cRnd = _acc.Allocate1D<uint>((cnt + 3) / 4);
        _cB = _acc.Allocate1D<float>(side * rank); _cAt = _acc.Allocate1D<BFloat16>(side * rank); _cRowMax = _acc.Allocate1D<float>(side); _cScale = _acc.Allocate1D<float>(1);
        _cRank = rank;
    }

    private void DisposeCompat()
    {
        _cw?.Dispose(); _cba?.Dispose(); _cB?.Dispose(); _cRowMax?.Dispose(); _cScale?.Dispose(); _cq?.Dispose(); _cAt?.Dispose(); _cRnd?.Dispose();
        _cw = _cba = _cB = _cRowMax = _cScale = null; _cq = _cAt = null; _cRnd = null;
        _cBlockHost = null; _cHost = null;
    }

    // a LoRA-patched Linear on an 8-bit weight, ComfyUI's way: w = W + sum strength B A (float32), re-rounded to F8_E4M3 at scale max|w| / 448 with the tensor's
    // seeded 8-bit stream, then the Linear on the rounded matrix (its scale on the input, as for the checkpoint's own 8-bit matrices)
    private void LinFp8Compat(GpuWeights w, FV x, string name, FV o, int M, int K, int N, DiTKernels.Act act, bool block)
    {
        int cnt = checked(N * K), words = (cnt + 3) / 4;
        if (cnt > _cw!.Length) throw new InvalidOperationException($"{name}: [{N}, {K}] is larger than the compatible mode's scratch");
        var cw = _cw.View.SubView(0, cnt); var cba = _cba!.View.SubView(0, cnt); var cq = _cq!.View.SubView(0, cnt);
        _l.PatchedWeightInit(_w.MatrixExact(w, name + ".weight"), cw, _w.ActivationScale(w, name + ".weight"), cnt);
        for (int i = 0; i < _loras.Count; i++)
        {
            var lw = block ? _loraBlock[i] : _loras[i].Weights.Base;
            if (lw == null || !lw.Has(name + ".lora_A.weight")) continue;
            int r = lw.Shape(name + ".lora_A.weight")[0];
            if (lw.Shape(name + ".lora_A.weight")[1] != K || lw.Shape(name + ".lora_B.weight")[0] != N) throw new InvalidDataException($"LoRA {name}: A [{string.Join(", ", lw.Shape(name + ".lora_A.weight"))}], B [{string.Join(", ", lw.Shape(name + ".lora_B.weight"))}] for a [{N}, {K}] Linear");
            var bF = _cB!.View.SubView(0, (long)N * r); var at = _cAt!.View.SubView(0, (long)K * r);
            _l.Bf16ToF32(lw.BF16(name + ".lora_B.weight"), bF, N * r);
            _l.TransposeBf16(lw.BF16(name + ".lora_A.weight"), at, r, K);
            _k.Linear(bF, at, null, cba, N, r, K);                                           // B A [N, K]: rows of B through A^T's rows
            _l.AddScaled(cw, cba, _loras[i].Strength, cnt);
        }
        _l.Fp8Scale(cw, _cRowMax!.View, _cScale!.View, N, K);
        // the tensor's stream: seeded by its key, the same on every call; from the thread pool when the block started it (StartCompatStreams)
        var sw = System.Diagnostics.Stopwatch.StartNew();
        uint[] host; int offset = 0;
        if (_cStreams.Remove(name, out var started)) { started.Task.GetAwaiter().GetResult(); host = _cBlockHost!; offset = started.Offset; }
        else
        {
            host = _cHost ??= GC.AllocateUninitializedArray<uint>((Fv * Dv + 3) / 4);        // no thread pool stream (a browser, a top-level Linear): generated here
            Ltx2Fp8Rounding.Low8(Ltx2Fp8Rounding.Seed(name), host.AsSpan(0, words), cnt);
        }
        CompatStreamSeconds += sw.Elapsed.TotalSeconds;
        var rnd = _cRnd!.View.SubView(0, words);
        rnd.CopyFromCPU(ref host[offset], (long)words);                                      // has taken the words when it returns: the host arrays are reused
        _l.Fp8StochasticRound(cw, rnd, _cScale.View, cq, cnt);
        var xs = _xs.View.SubView(0, (long)M * K);
        _l.ScaleBy(x, _cScale.View, xs, M * K);
        string bias = name + ".bias";
        _k.Linear(xs, cq, w.Has(bias) ? w.BF16(bias) : null, o, M, K, N, act);
    }

    private Ltx2Transformer(Accelerator accelerator, StreamedModelWeights weights, int batch, int videoTokens, int audioTokens, int contextTokens)
    {
        _acc = accelerator; _w = weights; _B = batch; _nv = videoTokens; _na = audioTokens; _L = contextTokens;
        _k = new DiTKernels(accelerator) { PreciseActivations = true }; _x = new Ltx.LtxKernels(accelerator); _l = new Ltx2Kernels(accelerator); _u = new SkinTokensKernels(accelerator); _f = new Flux2.Flux2Kernels(accelerator);
        long B = batch, nv = videoTokens, na = audioTokens, L = contextTokens, rows = Math.Max(nv, Math.Max(na, L));
        _v = Alloc(B * nv * Dv); _a = Alloc(B * na * Da); _vn = Alloc(B * nv * Dv); _an = Alloc(B * na * Da);
        _vctx = Alloc(B * L * Dv); _actx = Alloc(B * L * Da);
        _q = Alloc(B * rows * Dv); _kv = Alloc(B * rows * 2 * Dv); _part = Alloc(B * rows * Dv); _o = Alloc(B * rows * Dv);
        _hid = Alloc(B * Math.Max(nv * Fv, Math.Max(na * Fa, L * Dv))); _xs = Alloc(B * Math.Max(nv * Fv, Math.Max(na * Fa, L * Math.Max(Caption, Dv))));   // L x Dv: the text cross-attention's k / v input, once its matrix carries a scale
        // the score matrix exists only where no flash attention runs (the reference kernels): at a real clip's token count it is gigabytes (32 heads x 6144^2
        // floats = 4.8 GB), so a flash-capable backend gets a placeholder
        bool flash = _k.UsesFlashAttention(HeadV) && _k.UsesFlashAttention(HeadA) && _k.UsesFlashBiasedAttention(HeadV);
        _scores = Alloc(flash ? 1 : Math.Max(Hv, Ha) * Math.Max(Math.Max(nv * nv, na * na), Math.Max(Math.Max(nv, na) * L, nv * na)));
        _sinA = Alloc(B * TimeChannels); _temb = Alloc(B * Dv); _tokBias = Alloc(nv);
        _tA = Alloc(B * 6 * Da); _tAe = Alloc(B * Da); _tCaA = Alloc(B * 4 * Da); _tGateA = Alloc(B * Da);
        _modA = Alloc(B * 6 * Da); _caA = Alloc(B * 4 * Da); _gateA = Alloc(B * Da); _headA = Alloc(B * 2 * Da);
        AllocGroups(1);
        _ropeV = Alloc(nv * Dv / 2); _ropeVs = Alloc(nv * Dv / 2); _ropeA = Alloc(na * Da / 2); _ropeAs = Alloc(na * Da / 2); _ropeVx = Alloc(nv * Da / 2); _ropeVxs = Alloc(nv * Da / 2);
        _outV = Alloc(B * nv * InChannels); _outA = Alloc(B * na * InChannels);
    }

    private MemoryBuffer1D<float, Stride1D.Dense> Alloc(long n) { var b = _acc.Allocate1D<float>(Math.Max(1, n)); _owned.Add(b); return b; }

    private readonly List<IDisposable> _groupOwned = new();
    private MemoryBuffer1D<float, Stride1D.Dense> GAlloc(long n) { var b = _acc.Allocate1D<float>(Math.Max(1, n)); _groupOwned.Add(b); return b; }
    // the per-group video buffers for g groups per item (B g rows)
    private void AllocGroups(int g)
    {
        foreach (var o in _groupOwned) o.Dispose();
        _groupOwned.Clear();
        long r = (long)_B * g;
        _sinV = GAlloc(r * TimeChannels); _t1 = GAlloc(Math.Max(r, _B) * Dv); _tembS = GAlloc(Math.Max(r, _B) * Dv);
        _tV = GAlloc(r * 6 * Dv); _tVe = GAlloc(r * Dv); _tCaV = GAlloc(r * 4 * Dv); _tGateV = GAlloc(r * Dv); _tembV = GAlloc(r * Dv);
        _modV = GAlloc(r * 6 * Dv); _caV = GAlloc(r * 4 * Dv); _gateV = GAlloc(r * Dv); _headV = GAlloc(r * 2 * Dv);
        _gCap = g;
    }

    /// <summary>
    /// Open the transformer of an LTX-2 checkpoint (any prefix: the original layout's <c>model.diffusion_model.</c>) for calls of <paramref name="batch"/> items of
    /// <paramref name="videoTokens"/> video and <paramref name="audioTokens"/> audio tokens and <paramref name="contextTokens"/> text rows. <paramref name="residentBytes"/>: blocks kept
    /// on the device as stored, in order; the rest stream per call.
    /// </summary>
    public static async Task<Ltx2Transformer> OpenAsync(Accelerator accelerator, IModelSource source, string repo, string file, int batch, int videoTokens, int audioTokens, int contextTokens,
        long residentBytes, CancellationToken ct = default)
    {
        static int BlockOf(string n)
        {
            if (n.StartsWith("transformer_blocks.", StringComparison.Ordinal)) { int dot = n.IndexOf('.', 19); return int.Parse(n.AsSpan(19, dot - 19)); }
            if (n.Contains("_embeddings_connector.", StringComparison.Ordinal)) return StreamedModelWeights.HostOnly;   // Ltx2TextConnectors loads them
            return -1;
        }
        var w = await StreamedModelWeights.OpenAsync(accelerator, source, repo, file, "patchify_proj.weight", BlockOf, Layers, residentBytes, null, ct);
        try
        {
            if (!w.HeaderNames.Contains($"transformer_blocks.{Layers - 1}.audio_ff.net.2.weight") || w.HeaderNames.Contains($"transformer_blocks.{Layers}.ff.net.2.weight"))
                throw new InvalidDataException($"'{file}' is not LTX-2's 48-block audio-video transformer");
            return new Ltx2Transformer(accelerator, w, batch, videoTokens, audioTokens, contextTokens);
        }
        catch { w.Dispose(); throw; }
    }

    // nn.Linear over M rows (+ bias when the checkpoint has one); an 8-bit matrix runs exact (its decode without the scale, the input times the scale).
    // With LoRAs covering the Linear: + strength (x A^T) B^T for each, then the activation (ComfyUI patches the weight: x (W + strength B A)^T, then the activation)
    private void Lin(GpuWeights w, FV x, string name, FV o, int M, int K, int N, DiTKernels.Act act = DiTKernels.Act.None)
    {
        var xin = x;
        bool block = name.StartsWith("transformer_blocks.", StringComparison.Ordinal);
        bool patched = false;
        for (int i = 0; i < _loras.Count; i++)
        {
            var lw = block ? _loraBlock[i] : _loras[i].Weights.Base;
            if (lw != null && lw.Has(name + ".lora_A.weight")) { patched = true; break; }
        }
        if (patched && ComfyCompatibleFp8Lora && w.IsF8(name + ".weight")) { LinFp8Compat(w, x, name, o, M, K, N, act, block); return; }
        float s = _w.ActivationScale(w, name + ".weight");
        if (s != 1f) { var xs = _xs.View.SubView(0, (long)M * K); _f.ScaleInto(x, xs, s, M * K); x = xs; }
        string bias = name + ".bias";
        _k.Linear(x, _w.MatrixExact(w, name + ".weight"), w.Has(bias) ? w.BF16(bias) : null, o, M, K, N, patched ? DiTKernels.Act.None : act);
        if (!patched) return;
        for (int i = 0; i < _loras.Count; i++)
        {
            var lw = block ? _loraBlock[i] : _loras[i].Weights.Base;
            if (lw == null || !lw.Has(name + ".lora_A.weight")) continue;
            int r = lw.Shape(name + ".lora_A.weight")[0];
            if (lw.Shape(name + ".lora_A.weight")[1] != K || lw.Shape(name + ".lora_B.weight")[0] != N) throw new InvalidDataException($"LoRA {name}: A [{string.Join(", ", lw.Shape(name + ".lora_A.weight"))}], B [{string.Join(", ", lw.Shape(name + ".lora_B.weight"))}] for a [{N}, {K}] Linear");
            var t = _loraT!.View.SubView(0, (long)M * r); var u = _loraU!.View.SubView(0, (long)M * N);
            _k.Linear(xin, lw.BF16(name + ".lora_A.weight"), null, t, M, K, r);
            _k.Linear(t, lw.BF16(name + ".lora_B.weight"), null, u, M, r, N);
            _l.AxpyInPlace(o, u, _loras[i].Strength, M * N);
        }
        if (act != DiTKernels.Act.None) _l.ActivateInPlace(o, M * N, (int)act);
    }

    /// <summary>The 256-wide sinusoid of each timestep (diffusers Timesteps(256, flip_sin_to_cos = True, shift 0): [cos | sin], frequency exp(-ln(1e4) i / 128)).</summary>
    public static float[] Sinusoid(IReadOnlyList<float> t)
    {
        const int half = TimeChannels / 2;
        var o = new float[t.Count * TimeChannels];
        for (int b = 0; b < t.Count; b++)
            for (int i = 0; i < half; i++)
            {
                float f = MathF.Exp(-MathF.Log(10000f) * i / half);
                float a = t[b] * f;
                o[b * TimeChannels + i] = (float)Math.Cos(a); o[b * TimeChannels + half + i] = (float)Math.Sin(a);
            }
        return o;
    }

    // one adaLN embedder: emb = linear_2(silu(linear_1(sinusoid))) [B, D]; mod = linear(silu(emb)) [B, N D]
    private void Embedder(string name, int D, FV mod, FV emb, FV sin, int rows)
    {
        var b = _w.Base;
        Lin(b, sin, name + ".emb.timestep_embedder.linear_1", _t1.View, rows, TimeChannels, D, DiTKernels.Act.Silu);
        Lin(b, _t1.View, name + ".emb.timestep_embedder.linear_2", emb, rows, D, D);
        _k.Silu(emb, _tembS.View, rows * D);
        Lin(b, _tembS.View, name + ".linear", mod, rows, D, b.Shape(name + ".linear.weight")[0]);
    }

    /// <summary>
    /// The RoPE tables of a call (MEASURED against ComfyUI's dumped tables, 6e-8): video tokens (f, h, w) of a [F, H, W] latent, audio tokens of <paramref name="audioFrames"/>
    /// latent frames, the video frame rate <paramref name="fps"/>.
    /// </summary>
    /// <summary>
    /// The pixel-space boxes [t0, t1, h0, h1, w0, w1] of an F x H x W latent grid's tokens (f, h, w order): time [max(8 f - 7, 0), 8 f + 1) (causal: frame 0 is one pixel
    /// frame), space 32-pixel cells. A guide's tokens use ComfyUI's keyframe_idxs boxes instead (the guided CallAsync overload's tokenBoxes).
    /// </summary>
    public static float[] GridBoxes(int F, int H, int W)
    {
        var o = new float[F * H * W * 6];
        for (int f = 0, i = 0; f < F; f++)
            for (int h = 0; h < H; h++)
                for (int x = 0; x < W; x++, i++)
                {
                    o[6 * i] = Math.Max(TimeScale * f + 1 - TimeScale, 0); o[6 * i + 1] = Math.Max(TimeScale * (f + 1) + 1 - TimeScale, 0);
                    o[6 * i + 2] = SpaceScale * h; o[6 * i + 3] = SpaceScale * (h + 1); o[6 * i + 4] = SpaceScale * x; o[6 * i + 5] = SpaceScale * (x + 1);
                }
        return o;
    }

    private void UploadRope(int F, int H, int W, int audioFrames, float fps, float[]? boxes = null)
    {
        int nv = F * H * W;
        boxes ??= GridBoxes(F, H, W);
        if (boxes.Length != nv * 6) throw new ArgumentException($"{nv} token boxes of 6 values");
        var tv = new float[nv]; var hv = new float[nv]; var wv = new float[nv];
        for (int i = 0; i < nv; i++)
        {
            tv[i] = (boxes[6 * i] + boxes[6 * i + 1]) / 2f / fps / MaxTime;                // the box midpoints (use_middle_indices_grid), normalised
            hv[i] = (boxes[6 * i + 2] + boxes[6 * i + 3]) / 2f / MaxSpace; wv[i] = (boxes[6 * i + 4] + boxes[6 * i + 5]) / 2f / MaxSpace;
        }
        // video self: 682 frequencies per axis, frequency-major with the axis fastest, 2 identity entries first
        int Fq = Dv / 6, pad = Dv / 2 - 3 * Fq;
        var (ct, st) = Ltx2Kernels.SplitRopeTables(tv, Fq, 0, RopeTheta); var (ch, sh) = Ltx2Kernels.SplitRopeTables(hv, Fq, 0, RopeTheta); var (cw, sw) = Ltx2Kernels.SplitRopeTables(wv, Fq, 0, RopeTheta);
        var cos = new float[nv * Dv / 2]; var sin = new float[nv * Dv / 2];
        for (int i = 0; i < nv; i++)
        {
            int r = i * Dv / 2;
            for (int j = 0; j < pad; j++) { cos[r + j] = 1f; sin[r + j] = 0f; }
            for (int j = 0; j < Fq; j++)
            {
                cos[r + pad + 3 * j] = ct[i * Fq + j]; sin[r + pad + 3 * j] = st[i * Fq + j];
                cos[r + pad + 3 * j + 1] = ch[i * Fq + j]; sin[r + pad + 3 * j + 1] = sh[i * Fq + j];
                cos[r + pad + 3 * j + 2] = cw[i * Fq + j]; sin[r + pad + 3 * j + 2] = sw[i * Fq + j];
            }
        }
        _ropeV.View.SubView(0, cos.Length).CopyFromCPU(cos); _ropeVs.View.SubView(0, sin.Length).CopyFromCPU(sin);
        // audio self and cross: 1024 frequencies of the audio frame's time
        var ta = new float[audioFrames];
        for (int f = 0; f < audioFrames; f++)
        {
            float s0 = Math.Max(AudioScale * f + 1 - AudioScale, 0), s1 = Math.Max(AudioScale * (f + 1) + 1 - AudioScale, 0);
            ta[f] = (s0 + s1) / 2f * AudioHop / AudioRate / MaxTime;
        }
        var (ca, sa) = Ltx2Kernels.SplitRopeTables(ta, Da / 2, 0, RopeTheta);
        _ropeA.View.SubView(0, ca.Length).CopyFromCPU(ca); _ropeAs.View.SubView(0, sa.Length).CopyFromCPU(sa);
        // video cross (to audio): 1024 frequencies of the video token's time
        var (cx, sx) = Ltx2Kernels.SplitRopeTables(tv, Da / 2, 0, RopeTheta);
        _ropeVx.View.SubView(0, cx.Length).CopyFromCPU(cx); _ropeVxs.View.SubView(0, sx.Length).CopyFromCPU(sx);
    }

    /// <summary>
    /// One call: <paramref name="video"/> [B, F H W, 128] and <paramref name="audio"/> [B, audioFrames, 128] (the packed latents), the text <paramref name="videoContext"/> /
    /// <paramref name="audioContext"/> [B, L, 3840] (the connectors' outputs), the timesteps <paramref name="timesteps"/> (sigma x 1000) per item. Returns the velocities
    /// ([B, F H W, 128], [B, audioFrames, 128]) on the device, valid until the next call.
    /// </summary>
    public Task<(FV Video, FV Audio)> CallAsync(FV video, FV audio, FV videoContext, FV audioContext, IReadOnlyList<float> timesteps, int F, int H, int W, int audioFrames, float fps,
        CancellationToken ct = default)
        => CallAsync(video, audio, videoContext, audioContext, timesteps, null, null, null, F, H, W, audioFrames, fps, ct);

    private int? _guideSplit;                                          // the first guide token of a call with a guide attention bias (its per-token bias is in _tokBias)

    /// <summary>
    /// A call with per-frame video timesteps and guide tokens (ComfyUI's LTXVAddGuide / LTXVImgToVideo conditioning): <paramref name="frameMultipliers"/> [B F] or [F] scales
    /// each latent frame's timestep (ComfyUI's denoise_mask per frame; null = all 1); <paramref name="tokenBoxes"/> [F H W, 6] pixel boxes [t0, t1, h0, h1, w0, w1] per
    /// video token (null = <see cref="GridBoxes"/>; a guide's tokens take its keyframe_idxs boxes); <paramref name="guideBias"/> (the first guide token, a bias per video
    /// token [F H W]: 0 for the noisy tokens, ln(strength x mask) for a guide token, at least <see cref="DiTKernels.BlockedTokenBias"/>): the self-attention bias between
    /// noisy and guide tokens (<see cref="DiTKernels.BiasedAttention"/>; null or all 0: none). The audio timestep stays per item.
    /// </summary>
    public async Task<(FV Video, FV Audio)> CallAsync(FV video, FV audio, FV videoContext, FV audioContext, IReadOnlyList<float> timesteps, IReadOnlyList<float>? frameMultipliers,
        float[]? tokenBoxes, (int Split, float[] TokenBias)? guideBias, int F, int H, int W, int audioFrames, float fps, CancellationToken ct = default)
    {
        if (F * H * W != _nv || audioFrames != _na) throw new ArgumentException($"this transformer was opened for {_nv} video / {_na} audio tokens");
        if (timesteps.Count != _B) throw new ArgumentException($"{_B} timesteps (one per batch item)");
        if (frameMultipliers != null && frameMultipliers.Count != F && frameMultipliers.Count != _B * F) throw new ArgumentException($"frame multipliers: [{F}] or [{_B}, {F}]");
        int B = _B, nv = _nv, na = _na, L = _L;
        var b0 = _w.Base;
        // frame groups: one per item when every frame of every item shares its timestep, else one per latent frame
        float Mult(int b, int f) => frameMultipliers == null ? 1f : frameMultipliers.Count == F ? frameMultipliers[f] : frameMultipliers[b * F + f];
        bool uniform = true;
        for (int b = 0; b < B && uniform; b++) for (int f = 0; f < F; f++) if (Mult(b, f) != 1f) { uniform = false; break; }
        _vg = uniform ? 1 : F; _rpg = uniform ? nv : H * W;
        if (_vg > _gCap) { await _acc.SynchronizeAsync(); AllocGroups(_vg); }
        CompatStreamSeconds = 0;
        if (ComfyCompatibleFp8Lora && _loras.Count > 0) await EnsureCompatAsync();
        _guideSplit = null;
        if (guideBias is { } gb && gb.TokenBias.Any(v => v != 0f))
        {
            if (gb.TokenBias.Length != nv || gb.Split < 1 || gb.Split >= nv) throw new ArgumentException($"guide bias: a split inside the {nv} video tokens and one bias per token");
            for (int i = 0; i < nv; i++) if (!(gb.TokenBias[i] >= DiTKernels.BlockedTokenBias) || (i < gb.Split && gb.TokenBias[i] != 0f)) throw new ArgumentException($"guide bias: token {i} has {gb.TokenBias[i]} (0 before the split, finite and at least {DiTKernels.BlockedTokenBias} from it)");
            _tokBias.View.CopyFromCPU(gb.TokenBias);
            _guideSplit = gb.Split;
        }
        UploadRope(F, H, W, audioFrames, fps, tokenBoxes);
        var tvid = new float[B * _vg];
        for (int b = 0; b < B; b++) for (int g = 0; g < _vg; g++) tvid[b * _vg + g] = timesteps[b] * (uniform ? 1f : Mult(b, g));
        _sinV.View.SubView(0, (long)B * _vg * TimeChannels).CopyFromCPU(Sinusoid(tvid));
        _sinA.View.SubView(0, B * TimeChannels).CopyFromCPU(Sinusoid(timesteps));
        int rv = B * _vg;
        Embedder("adaln_single", Dv, _tV.View, _tVe.View, _sinV.View, rv);
        Embedder("audio_adaln_single", Da, _tA.View, _tAe.View, _sinA.View, B);
        Embedder("av_ca_video_scale_shift_adaln_single", Dv, _tCaV.View, _tembV.View, _sinV.View, rv);
        Embedder("av_ca_audio_scale_shift_adaln_single", Da, _tCaA.View, _temb.View, _sinA.View, B);
        // the a2v GATE takes the item's timestep, not the frame's (MEASURED 2026-11-17 on ComfyUI's guided call: v_cross_gate_timestep is identical for the guide
        // frame and the noisy frames, while the adaLN mod and the cross scale / shift differ per frame). do NOT feed _sinV here
        Embedder("av_ca_a2v_gate_adaln_single", Dv, _tGateV.View, _tembV.View, _sinA.View, B);
        Embedder("av_ca_v2a_gate_adaln_single", Da, _tGateA.View, _temb.View, _sinA.View, B);
        Probe?.Invoke("adaln_video", _tV.View, rv, 6 * Dv); Probe?.Invoke("adaln_video_emb", _tVe.View, rv, Dv);
        Probe?.Invoke("adaln_audio", _tA.View, B, 6 * Da); Probe?.Invoke("adaln_audio_emb", _tAe.View, B, Da);

        // text: the caption projections (Linear, GELU-tanh, Linear)
        Lin(b0, videoContext, "caption_projection.linear_1", _hid.View, B * L, Caption, Dv, DiTKernels.Act.GeluTanh);
        Lin(b0, _hid.View, "caption_projection.linear_2", _vctx.View, B * L, Dv, Dv);
        Lin(b0, audioContext, "audio_caption_projection.linear_1", _hid.View, B * L, Caption, Da, DiTKernels.Act.GeluTanh);
        Lin(b0, _hid.View, "audio_caption_projection.linear_2", _actx.View, B * L, Da, Da);
        Probe?.Invoke("caption_video", _vctx.View, B * L, Dv); Probe?.Invoke("caption_audio", _actx.View, B * L, Da);
        Lin(b0, video, "patchify_proj", _v.View, B * nv, InChannels, Dv);
        Lin(b0, audio, "audio_patchify_proj", _a.View, B * na, InChannels, Da);
        Probe?.Invoke("patchify_video", _v.View, B * nv, Dv); Probe?.Invoke("patchify_audio", _a.View, B * na, Da);

        if (_loraBlock.Length != _loras.Count) _loraBlock = new GpuWeights?[_loras.Count];
        var loraStreamed = new bool[_loras.Count];
        for (int k = 0; k < Layers; k++)
        {
            var (w, streamed) = await _w.BlockAsync(k, ct);
            for (int i = 0; i < _loras.Count; i++) (_loraBlock[i], loraStreamed[i]) = await _loras[i].Weights.BlockAsync(k, ct);
            try
            {
                if (ComfyCompatibleFp8Lora && _loras.Count > 0) StartCompatStreams(w);
                Block(w, $"transformer_blocks.{k}.");
            }
            finally
            {
                FinishCompatStreams();
                if (streamed || loraStreamed.Any(b => b)) await _acc.SynchronizeAsync();
                if (streamed) w.Dispose();
                for (int i = 0; i < _loras.Count; i++) { if (loraStreamed[i]) _loraBlock[i]!.Dispose(); _loraBlock[i] = null; }
            }
            Probe?.Invoke($"block{k}_video", _v.View, B * nv, Dv);
            Probe?.Invoke($"block{k}_audio", _a.View, B * na, Da);
        }

        // output heads: LayerNorm (no affine), (1 + scale) + shift with shift = table[0] + embedded timestep, scale = table[1] + embedded timestep
        for (int b = 0; b < B; b++)
        {
            for (int g = 0; g < _vg; g++)
            {
                int r = b * _vg + g;
                for (int j = 0; j < 2; j++) _x.Add(b0.F32("scale_shift_table").SubView(j * Dv, Dv), _tVe.View.SubView((long)r * Dv, Dv), _headV.View.SubView((long)(2 * r + j) * Dv, Dv), Dv);
                _k.LayerNormModulate(VRows(_v, b, g, Dv), VRows(_vn, b, g, Dv), _headV.View, 2 * r * Dv, (2 * r + 1) * Dv, _rpg, Dv, Eps);
            }
            for (int j = 0; j < 2; j++) _x.Add(b0.F32("audio_scale_shift_table").SubView(j * Da, Da), _tAe.View.SubView(b * Da, Da), _headA.View.SubView((2 * b + j) * Da, Da), Da);
            _k.LayerNormModulate(_a.View.SubView((long)b * na * Da, (long)na * Da), _an.View.SubView((long)b * na * Da, (long)na * Da), _headA.View, 2 * b * Da, (2 * b + 1) * Da, na, Da, Eps);
        }
        Lin(b0, _vn.View, "proj_out", _outV.View, B * nv, Dv, InChannels);
        Lin(b0, _an.View, "audio_proj_out", _outA.View, B * na, Da, InChannels);
        return (_outV.View.SubView(0, (long)B * nv * InChannels), _outA.View.SubView(0, (long)B * na * InChannels));
    }

    // mod rows of one block: table [N, D] (fp32) + the embedder's [B, N D] (per item)
    private void Mod(FV table, FV temb, FV mod, int N, int D, int tableRow0 = 0, int rows = -1)
    {
        int R = rows < 0 ? _B : rows;
        for (int b = 0; b < R; b++)
            _x.Add(table.SubView((long)tableRow0 * D, (long)N * D), temb.SubView((long)b * N * D, (long)N * D), mod.SubView((long)b * N * D, (long)N * D), N * D);
    }

    // the token rows of video group g of item b ([_rpg, C]); its modulation row is b _vg + g
    private FV VRows(MemoryBuffer1D<float, Stride1D.Dense> buf, int b, int g, int C) => buf.View.SubView(((long)b * _nv + (long)g * _rpg) * C, (long)_rpg * C);
    private FV VPart(int b, int g) => _part.View.SubView(((long)b * _nv + (long)g * _rpg) * Dv, (long)_rpg * Dv);

    // self- or cross-attention of one stream: q from xq [B Nq, Dq-in], k / v from xkv [B Nk, Dkv-in]; heads H x D (inner H D); optional RoPE tables per side; out [B Nq, Dout] into o
    private void Attn(GpuWeights w, string p, FV xq, int Nq, int dq, FV xkv, int Nk, int dkv, int H, int D, int dout, FV o,
        (FV Cos, FV Sin)? ropeQ, (FV Cos, FV Sin)? ropeK, int? biasSplit = null)
    {
        int B = _B, C = H * D;
        string tag = StageSeconds != null ? p.Split('.')[^2] : "";
        Stage("norm / gate");
        Lin(w, xq, p + "to_q", _q.View, B * Nq, dq, C);
        Lin(w, xkv, p + "to_k", _part.View, B * Nk, dkv, C); _f.CopyCols(_part.View, _kv.View, B * Nk, C, C, 2 * C, 0);
        Lin(w, xkv, p + "to_v", _part.View, B * Nk, dkv, C); _f.CopyCols(_part.View, _kv.View, B * Nk, C, C, 2 * C, C);
        Stage(tag + ": q / k / v projections");
        _u.RmsNormHeads(_q.View, w.BF16(p + "q_norm.weight"), B * Nq, C, 0, 1, C, QkEps);               // over the whole inner dim, with weights
        _u.RmsNormHeads(_kv.View, w.BF16(p + "k_norm.weight"), B * Nk, 2 * C, 0, 1, C, QkEps);
        for (int b = 0; b < B; b++)
        {
            var qb = _q.View.SubView((long)b * Nq * C, (long)Nq * C); var kvb = _kv.View.SubView((long)b * Nk * 2 * C, (long)Nk * 2 * C);
            if (ropeQ is { } rq) _l.RopeSplit(qb, rq.Cos, rq.Sin, Nq, C, 0, H, D);
            if (ropeK is { } rk) _l.RopeSplit(kvb, rk.Cos, rk.Sin, Nk, 2 * C, 0, H, D);
            if (biasSplit is { } split) _k.BiasedAttention(qb, C, 0, kvb, 2 * C, 0, C, _scores.View, _o.View.SubView((long)b * Nq * C, (long)Nq * C), Nq, H, D, split, _tokBias.View);
            else _k.Attention(qb, C, 0, kvb, 2 * C, 0, C, _scores.View, _o.View.SubView((long)b * Nq * C, (long)Nq * C), Nq, Nk, H, D);
        }
        Stage(tag + ": norms, RoPE, attention");
        Lin(w, _o.View, p + "to_out.0", o, B * Nq, C, dout);
        Stage(tag + ": out projection");
    }

    /// <summary>
    /// Set to a dictionary to time the stages of every block (seconds per stage, summed over the blocks and calls): the device is synchronized after each stage,
    /// so this is for desktop backends and a profiled call is slower than a plain one. Null (default): no timing, no synchronizes.
    /// </summary>
    public Dictionary<string, double>? StageSeconds { get; set; }
    private long _stageT0;
    private void Stage(string name)
    {
        if (StageSeconds is not { } d) return;
        _acc.Synchronize();
        long now = System.Diagnostics.Stopwatch.GetTimestamp();
        d[name] = d.GetValueOrDefault(name) + (now - _stageT0) / (double)System.Diagnostics.Stopwatch.Frequency;
        _stageT0 = now;
    }

    private void Block(GpuWeights w, string p)
    {
        int B = _B, nv = _nv, na = _na, L = _L;
        if (StageSeconds != null) { _acc.Synchronize(); _stageT0 = System.Diagnostics.Stopwatch.GetTimestamp(); }
        int G = _vg, rv = B * G;
        Mod(w.F32(p + "scale_shift_table"), _tV.View, _modV.View, 6, Dv, 0, rv);
        Mod(w.F32(p + "audio_scale_shift_table"), _tA.View, _modA.View, 6, Da);
        // rows 0..3 of the per-block cross tables + the scale/shift embedders; row 4 + the gate embedders
        Mod(w.F32(p + "scale_shift_table_a2v_ca_video"), _tCaV.View, _caV.View, 4, Dv, 0, rv);
        Mod(w.F32(p + "scale_shift_table_a2v_ca_audio"), _tCaA.View, _caA.View, 4, Da);
        Mod(w.F32(p + "scale_shift_table_a2v_ca_video"), _tGateV.View, _gateV.View, 1, Dv, 4);          // per item: the gate has no frame groups (see CallAsync)
        Mod(w.F32(p + "scale_shift_table_a2v_ca_audio"), _tGateA.View, _gateA.View, 1, Da, 4);

        Stage("modulation tables");
        // 1. self-attention (video, then audio): adaLN 0 shift, 1 scale, 2 gate
        for (int b = 0; b < B; b++) for (int g = 0; g < G; g++) { int r = b * G + g; _x.RmsNormModulate(VRows(_v, b, g, Dv), VRows(_vn, b, g, Dv), _modV.View, r * 6 * Dv, r * 6 * Dv + Dv, _rpg, Dv, Eps); }
        Attn(w, p + "attn1.", _vn.View, nv, Dv, _vn.View, nv, Dv, Hv, HeadV, Dv, _part.View, (_ropeV.View, _ropeVs.View), (_ropeV.View, _ropeVs.View), _guideSplit);
        for (int b = 0; b < B; b++) for (int g = 0; g < G; g++) { int r = b * G + g; _k.GateResidual(VRows(_v, b, g, Dv), VPart(b, g), _modV.View, r * 6 * Dv + 2 * Dv, _rpg * Dv, Dv); }
        for (int b = 0; b < B; b++) _x.RmsNormModulate(Rows(_a, b, na, Da), Rows(_an, b, na, Da), _modA.View, b * 6 * Da, b * 6 * Da + Da, na, Da, Eps);
        Attn(w, p + "audio_attn1.", _an.View, na, Da, _an.View, na, Da, Ha, HeadA, Da, _part.View, (_ropeA.View, _ropeAs.View), (_ropeA.View, _ropeAs.View));
        for (int b = 0; b < B; b++) _k.GateResidual(Rows(_a, b, na, Da), _part.View.SubView((long)b * na * Da, (long)na * Da), _modA.View, b * 6 * Da + 2 * Da, na * Da, Da);

        Stage("norm / gate");
        // 2. text cross-attention: plain RMS norm, no RoPE, ungated
        _l.RmsNormPlain(_v.View, _vn.View, B * nv, Dv, Eps);
        Attn(w, p + "attn2.", _vn.View, nv, Dv, _vctx.View, L, Dv, Hv, HeadV, Dv, _part.View, null, null);
        _k.GateResidual(_v.View, _part.View, _part.View, -1, B * nv * Dv, Dv);
        _l.RmsNormPlain(_a.View, _an.View, B * na, Da, Eps);
        Attn(w, p + "audio_attn2.", _an.View, na, Da, _actx.View, L, Da, Ha, HeadA, Da, _part.View, null, null);
        _k.GateResidual(_a.View, _part.View, _part.View, -1, B * na * Da, Da);

        Stage("norm / gate");
        // 3. audio <-> video. a2v first; then v2a reads the video stream AFTER a2v's update (MEASURED 2026-11-17 against ComfyUI's block 0, float64 reference: the
        //    pre-update video, as diffusers' code reads it, leaves the audio stream 2.4e-4 off; the updated one 1.1e-7).
        //    video mods: 0 a2v scale, 1 a2v shift, 2 v2a scale, 3 v2a shift; audio the same
        // a2v: q = video modulated by its a2v pair, k / v = audio modulated by its a2v pair
        for (int b = 0; b < B; b++)
        {
            for (int g = 0; g < G; g++) { int r = b * G + g; _x.RmsNormModulate(VRows(_v, b, g, Dv), VRows(_vn, b, g, Dv), _caV.View, r * 4 * Dv + Dv, r * 4 * Dv, _rpg, Dv, Eps); }
            _x.RmsNormModulate(Rows(_a, b, na, Da), Rows(_an, b, na, Da), _caA.View, b * 4 * Da + Da, b * 4 * Da, na, Da, Eps);
        }
        Attn(w, p + "audio_to_video_attn.", _vn.View, nv, Dv, _an.View, na, Da, Ha, HeadA, Dv, _part.View, (_ropeVx.View, _ropeVxs.View), (_ropeA.View, _ropeAs.View));
        for (int b = 0; b < B; b++) _k.GateResidual(Rows(_v, b, nv, Dv), _part.View.SubView((long)b * nv * Dv, (long)nv * Dv), _gateV.View, b * Dv, nv * Dv, Dv);
        // v2a: q = audio modulated by its v2a pair, k / v = the (a2v-updated) video modulated by its v2a pair
        for (int b = 0; b < B; b++)
        {
            for (int g = 0; g < G; g++) { int r = b * G + g; _x.RmsNormModulate(VRows(_v, b, g, Dv), VRows(_vn, b, g, Dv), _caV.View, r * 4 * Dv + 3 * Dv, r * 4 * Dv + 2 * Dv, _rpg, Dv, Eps); }
            _x.RmsNormModulate(Rows(_a, b, na, Da), Rows(_an, b, na, Da), _caA.View, b * 4 * Da + 3 * Da, b * 4 * Da + 2 * Da, na, Da, Eps);
        }
        Attn(w, p + "video_to_audio_attn.", _an.View, na, Da, _vn.View, nv, Dv, Ha, HeadA, Da, _part.View, (_ropeA.View, _ropeAs.View), (_ropeVx.View, _ropeVxs.View));
        for (int b = 0; b < B; b++) _k.GateResidual(Rows(_a, b, na, Da), _part.View.SubView((long)b * na * Da, (long)na * Da), _gateA.View, b * Da, na * Da, Da);

        Stage("norm / gate");
        // 4. feed-forwards: adaLN 3 shift, 4 scale, 5 gate
        for (int b = 0; b < B; b++) for (int g = 0; g < G; g++) { int r = b * G + g; _x.RmsNormModulate(VRows(_v, b, g, Dv), VRows(_vn, b, g, Dv), _modV.View, r * 6 * Dv + 3 * Dv, r * 6 * Dv + 4 * Dv, _rpg, Dv, Eps); }
        Stage("norm / gate");
        Lin(w, _vn.View, p + "ff.net.0.proj", _hid.View, B * nv, Dv, Fv, DiTKernels.Act.GeluTanh);
        Stage("video ff.0 (4096 -> 16384)");
        Lin(w, _hid.View, p + "ff.net.2", _part.View, B * nv, Fv, Dv);
        Stage("video ff.2 (16384 -> 4096)");
        for (int b = 0; b < B; b++) for (int g = 0; g < G; g++) { int r = b * G + g; _k.GateResidual(VRows(_v, b, g, Dv), VPart(b, g), _modV.View, r * 6 * Dv + 5 * Dv, _rpg * Dv, Dv); }
        for (int b = 0; b < B; b++) _x.RmsNormModulate(Rows(_a, b, na, Da), Rows(_an, b, na, Da), _modA.View, b * 6 * Da + 3 * Da, b * 6 * Da + 4 * Da, na, Da, Eps);
        Stage("norm / gate");
        Lin(w, _an.View, p + "audio_ff.net.0.proj", _hid.View, B * na, Da, Fa, DiTKernels.Act.GeluTanh);
        Lin(w, _hid.View, p + "audio_ff.net.2", _part.View, B * na, Fa, Da);
        Stage("audio ff");
        for (int b = 0; b < B; b++) _k.GateResidual(Rows(_a, b, na, Da), _part.View.SubView((long)b * na * Da, (long)na * Da), _modA.View, b * 6 * Da + 5 * Da, na * Da, Da);
    }

    private static FV Rows(MemoryBuffer1D<float, Stride1D.Dense> buf, int b, int n, int C) => buf.View.SubView((long)b * n * C, (long)n * C);

    public void Dispose()
    {
        foreach (var o in _owned) o.Dispose();
        _owned.Clear();
        _loraT?.Dispose(); _loraU?.Dispose(); _loraT = _loraU = null;
        DisposeCompat();
        foreach (var o in _groupOwned) o.Dispose();
        _groupOwned.Clear();
        _w.Dispose();
    }
}
