using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Trellis2;

namespace SpawnDev.ILGPU.ML.StableDiffusion;

using FV = ArrayView1D<float, Stride1D.Dense>;
using HV = ArrayView1D<global::ILGPU.Half, Stride1D.Dense>;
using Buf = MemoryBuffer1D<float, Stride1D.Dense>;

/// <summary>
/// SDXL's extra conditioning: the pooled text embedding of the second text encoder and the six size numbers (original size, crop top-left, target size; ComfyUI's width / height
/// fields, in pixels).
/// </summary>
/// <summary>
/// ComfyUI's FreeU / FreeU_V2 (<see cref="SdUNet.FreeU"/>): in every output block whose incoming features have 4x the model channels (1280 on SD 1.x / SDXL) the first half
/// of those channels is scaled by <see cref="B1"/> (V2: by (B1 - 1) x the min-max normalised per-pixel channel mean + 1) and the skip connection's lowest frequencies by
/// <see cref="S1"/> (ComfyUI's Fourier_filter, threshold 1); at 2x the model channels <see cref="B2"/> / <see cref="S2"/>. ComfyUI's defaults: FreeU 1.1, 1.2, 0.9, 0.2;
/// FreeU_V2 1.3, 1.4, 0.9, 0.2.
/// </summary>
public sealed record SdFreeU(float B1, float B2, float S1, float S2, bool V2 = false);

/// <summary>How a feature map is resized (torch's F.interpolate, align_corners False, as ComfyUI's common_upscale).</summary>
public enum SdResize { Bicubic, Bilinear, NearestExact }

/// <summary>
/// ComfyUI's PatchModelAddDownscale ("Kohya deep shrink", <see cref="SdUNet.DeepShrink"/>): after input block <see cref="Block"/> the features are resized by
/// 1 / <see cref="Factor"/> (Python's round of each side) with <see cref="Down"/> - after the skip connection is stored when <see cref="AfterSkip"/> - and an output
/// block whose features do not meet their skip connection's size resizes them to it with <see cref="Up"/>. ComfyUI applies it only while the step's sigma is inside
/// [sigma(end_percent), sigma(start_percent)] (defaults 0 and 0.35): set and clear the property per step accordingly.
/// </summary>
/// <summary>
/// ComfyUI's HyperTile (<see cref="SdUNet.HyperTile"/>): the self-attention of the blocks at the first <see cref="MaxDepth"/> + 1 resolutions runs inside tiles of
/// about <see cref="TileSize"/> pixels (latent max(32, TileSize) / 8, doubled per depth when <see cref="ScaleDepth"/>). ComfyUI picks each side's tile count at
/// random among the first <see cref="SwapSize"/> - 1 candidates (its torch.randint excludes the last; <see cref="Rng"/>, null = a shared System.Random): SwapSize 1
/// or 2 is deterministic. ComfyUI's defaults:
/// 256, 2, 0, false.
/// <para>
/// ComfyUI's node tiles only the QUERIES: its attn1 patch rearranges q, and k / v are projected from the untiled tokens, then reshaped with q's batch, so query tile t
/// (t = tile row * tiles per row + tile column) attends to the flat tokens [t * L, (t + 1) * L) of its own picture (L = tokens per tile): a band of rows, not its own tile.
/// That is the default here (ComfyUI's pictures; found by SpawnGen's parity test). <see cref="TileKeysAndValues"/> = true gives true tiled self-attention instead (each tile attends to itself).
/// </para>
/// </summary>
public sealed record SdHyperTile(int TileSize = 256, int SwapSize = 2, int MaxDepth = 0, bool ScaleDepth = false, Random? Rng = null, bool TileKeysAndValues = false);

public sealed record SdDeepShrink(int Block = 3, double Factor = 2.0, bool AfterSkip = true, SdResize Down = SdResize.Bicubic, SdResize Up = SdResize.Bicubic);

public sealed record SdxlConditioning(float[] Pooled, int OriginalHeight, int OriginalWidth, int CropTop, int CropLeft, int TargetHeight, int TargetWidth);

/// <summary>
/// The Stable Diffusion UNet (the LDM <c>UNetModel</c>, read from a single-file checkpoint in its original key names): SD 1.x, SD 2.x and SDXL are one code path, the layout
/// (levels, resnets per level, transformer depth, heads, linear or 1x1-conv projections, SDXL's label embedding) is read from the tensors themselves.
/// One call is one noise prediction at one timestep for batch 1 (the sampler, CFG and noise stay with the caller): <see cref="ForwardAsync(float[], int, int, double, float[], int, SdxlConditioning?, CancellationToken)"/>.
/// The latent may have any size (odd sides halve upward and upsample back to their skip connection's size, as in ComfyUI). Activations are channel-last fp32 [pixels, C], weights fp16, convolutions the gathered-row
/// implicit GEMM (tensor cores on CUDA), attention the portable flash kernels. The prediction is whatever the model was trained for (epsilon for SD 1.x / SDXL base, v for the
/// SD 2.x 768-v models: <see cref="SdCheckpoint.Prediction"/>).
/// </summary>
public sealed partial class SdUNet : IDisposable
{
    private bool _ownsWeights;            // loaded alone (LoadAsync): Dispose frees the weights too
    internal bool VPredTensor;            // the file carried a "v_pred" tensor (ComfyUI's marker of a v-prediction fine-tune)
    private SdLoraMerger? _lora;
    private SdPatchState? _patches;
    private SdPatchState Patches => _patches ??= new SdPatchState(_acc, _w);

    /// <summary>The model family by its context width (768 SD 1.x, 1024 SD 2.x, 2048 SDXL).</summary>
    public SdFamily Family => ContextDim switch { 768 => SdFamily.SD1, 1024 => SdFamily.SD2, _ => SdFamily.SDXL };

    /// <summary>
    /// What this UNet predicts, by ComfyUI's rules: a <c>v_pred</c> tensor in the file means V; an SD 2.x UNet is V when the spread of one late layer-norm bias
    /// (<c>output_blocks.11.1.transformer_blocks.0.norm1.bias</c>) is above 0.09 (the 768-v models); anything else is epsilon. Reads 320 numbers back from the device.
    /// </summary>
    public async Task<SdPrediction> InferPredictionAsync()
    {
        if (VPredTensor) return SdPrediction.V;
        if (Family != SdFamily.SD2) return SdPrediction.Epsilon;
        const string key = "output_blocks.11.1.transformer_blocks.0.norm1.bias";
        if (!_w.Has(key)) return SdPrediction.Epsilon;
        var hv = _w[key];
        using var f = _acc.Allocate1D<float>(hv.Length);
        _k.HalfToFloat(hv, f.View);
        await _acc.SynchronizeAsync();
        var v = await f.View.CopyToHostAsync();
        double mean = v.Average(x => (double)x), var = v.Average(x => (x - mean) * (x - mean));
        return Math.Sqrt(var) > 0.09 ? SdPrediction.V : SdPrediction.Epsilon;
    }

    /// <summary>
    /// Merge LoRAs into this UNet (their UNet modules at <see cref="SdLoraSpec.Strength"/>; text-encoder keys are ignored here and reported unmatched), replacing any set
    /// merged before: the originals of patched tensors are kept on the device, so this restores them first (an empty list removes every LoRA).
    /// </summary>
    public Task<SdLoraReport> ApplyLorasAsync(IReadOnlyList<SdLoraSpec> loras, CancellationToken cancellationToken = default)
    {
        _lora ??= new SdLoraMerger(_acc, Patches, null, Family);
        return _lora.ApplyAsync(loras, cancellationToken);
    }

    /// <summary>
    /// Merge another UNet into this one, tensor by tensor: <c>W = W * Self + W_other * Other</c> with the factors <paramref name="factors"/> returns for each tensor name
    /// (null = leave it; <see cref="SdMerge"/> has ComfyUI's ModelMergeSimple / Add / Subtract / Blocks / SD1 / SDXL). <paramref name="other"/> must be loaded on the same
    /// accelerator; it is read, not changed, and may be disposed afterwards. Edits apply in call order as in ComfyUI's graph: the merge reads this UNet's current weights
    /// (LoRAs included, which become part of the weights) and the other's current weights (its own merges and LoRAs included); LoRAs applied after it stay replaceable.
    /// <paramref name="restorable"/> keeps the file's weights of every changed tensor on the device (<see cref="ResetAsync"/>), as much memory again as the merged tensors:
    /// false saves that (an SDXL UNet is 5 GB) and makes <see cref="ResetAsync"/> throw.
    /// </summary>
    public Task<SdMergeReport> MergeAsync(SdUNet other, Func<string, SdMergeFactors?> factors, bool restorable = true, CancellationToken cancellationToken = default)
    {
        if (!ReferenceEquals(other._acc, _acc)) throw new ArgumentException("the other UNet is on another accelerator", nameof(other));
        return SdMerge.RunAsync(_acc, _k, Patches, other._w, factors, restorable, cancellationToken);
    }

    /// <summary>Undo every merge and LoRA: back to the weights the file had.</summary>
    public Task ResetAsync() => _patches?.ResetAsync() ?? Task.CompletedTask;

    /// <summary>
    /// Write this UNet's current weights (merges and LoRAs included) as a .safetensors file with LDM names (<c>model.diffusion_model.*</c>: ComfyUI's ModelSave, which
    /// <see cref="LoadAsync(Accelerator, Stream, string, Action{string, double}?, CancellationToken)"/> and ComfyUI's UNETLoader read back). Streams front to back.
    /// </summary>
    public Task SaveAsync(Stream target, SdSaveOptions? options = null, CancellationToken cancellationToken = default)
    {
        options ??= new SdSaveOptions();
        return SdSafeTensorsWriter.WriteAsync(_acc, target, SdSafeTensorsWriter.Ldm(_w, "model.diffusion_model."), options, options.VPrediction ?? false, cancellationToken);
    }

    internal SdWeights Weights => _w;
    private readonly Accelerator _acc;
    private readonly SdWeights _w;
    private readonly SdKernels _k;
    private readonly DiTKernels _dk;
    private readonly Dictionary<(int, int, int, int, int, int, int, int), MemoryBuffer1D<int, Stride1D.Dense>> _tables = new();
    private readonly Dictionary<int, Buf> _slots = new();
    private readonly List<IDisposable> _retired = new();
    private readonly List<Module[]> _input = new(), _output = new();
    private readonly Module[] _middle;

    public int ModelChannels { get; }
    public int InChannels { get; }
    /// <summary>
    /// The channels an inpainting UNet takes besides the latent (<see cref="InChannels"/> - <see cref="OutChannels"/>: 5 for SD 1.x / 2.x / SDXL inpainting = the mask, then
    /// the masked image's latent, as diffusers and ComfyUI's InpaintModelConditioning concatenate them); 0 for an ordinary UNet. See the <c>concat</c> overloads of ForwardAsync.
    /// </summary>
    public int ConcatChannels => OutChannels == 0 ? 0 : Math.Max(0, InChannels - OutChannels);
    public int OutChannels { get; }
    /// <summary>The text context width (768 SD 1.x, 1024 SD 2.x, 2048 SDXL).</summary>
    public int ContextDim { get; }
    /// <summary>True for SDXL (the pooled text + size conditioning, <see cref="SdxlConditioning"/>).</summary>
    public bool IsXL { get; }
    /// <summary>
    /// Compute the attention q / k / v projections with fp32 activations (no tensor cores), as diffusers' <c>upcast_attention</c> does for SD 2.x: its attention logits are large
    /// enough that fp16-rounded q and k move the softmax (MEASURED on SD 2.1 768-v, CUDA: eps 5.3e-2 off the reference with tensor cores, fp32 paths 6e-5). Default on for the
    /// 1024-wide-context (SD 2.x) models, off for SD 1.x / SDXL, which do not need it.
    /// </summary>
    public bool UpcastAttention { get; set; }

    /// <summary>
    /// Self-Attention Guidance's capture (ComfyUI's SelfAttentionGuidance records the middle block's first self-attention, attn1 of ("middle", 0, 0), in the
    /// unconditional pass): while true, every prediction keeps that attention's per-key mass - softmax(q k^T / sqrt(d)) averaged over the heads and summed over
    /// the queries, ComfyUI's attn.mean(1).sum(1) - for <see cref="ReadMiddleAttentionAsync"/>. The prediction itself is unchanged.
    /// </summary>
    public bool RecordMiddleSelfAttention { get; set; }
    private int _sagN;

    /// <summary>
    /// The middle-attention mass of the last prediction made with <see cref="RecordMiddleSelfAttention"/> on ([tokens] at the middle block's resolution), or null.
    /// Feed it to <see cref="SdGuidance.SagBlur"/> with the unconditional x0 prediction.
    /// </summary>
    public async Task<float[]?> ReadMiddleAttentionAsync()
    {
        if (_sagN == 0) return null;
        await _acc.SynchronizeAsync();
        return await Slot(SlotSagMass, _sagN).CopyToHostAsync();
    }

    /// <summary>HyperTile (<see cref="SdHyperTile"/>), null = off.</summary>
    public SdHyperTile? HyperTile { get; set; }
    private int _latH, _latW;           // the latent size of the running prediction (HyperTile's original_shape)

    /// <summary>Deep shrink (<see cref="SdDeepShrink"/>), null = off.</summary>
    public SdDeepShrink? DeepShrink { get; set; }

    /// <summary>FreeU / FreeU_V2 on the output blocks (<see cref="SdFreeU"/>), null = off. Applied after ControlNet residuals are added to the skips, as ComfyUI does.</summary>
    public SdFreeU? FreeU { get; set; }

    /// <summary>
    /// Perturbed-Attention Guidance's perturbed pass (ComfyUI's PerturbedAttentionGuidance): every self-attention of the middle block returns its values (the attention map
    /// replaced by the identity). Run the conditional prediction once with this on and combine as ComfyUI does: <c>cfg + (cond - perturbed) * scale</c>.
    /// </summary>
    public bool PerturbMiddleSelfAttention { get; set; }

    /// <summary>The width of SDXL's label vector (the input of <c>label_emb.0.0</c>: 2816 for SDXL base), 0 otherwise: what <see cref="SdxlLabel"/> takes.</summary>
    public int LabelWidth => _admDim;
    private readonly int _timeDim, _admDim;
    private readonly bool _fixedHeads;      // SD 1.x: 8 heads at every width; SD 2.x / SDXL: 64-wide heads

    /// <summary>Test hook: (stage name, activation view [pixels, C] channel-last, height, width, channels); valid only during the call.</summary>
    public Action<string, FV, int, int, int>? Tap { get; set; }

    private enum Kind { Res, Attn, Down, Up }
    private sealed record Module(Kind Kind, string Prefix, int Cin, int Cout, int Depth);

    public SdUNet(Accelerator accelerator, SdWeights weights)
    {
        _acc = accelerator; _w = weights; _k = new SdKernels(accelerator); _dk = new DiTKernels(accelerator);
        var convIn = weights.Shape("input_blocks.0.0.weight");
        ModelChannels = convIn[0]; InChannels = convIn[1];
        OutChannels = weights.Has("out.2.weight") ? weights.Shape("out.2.weight")[0] : 0;      // 0: the encoder half only (a ControlNet's copy)
        _timeDim = weights.Shape("time_embed.0.weight")[0];
        IsXL = weights.Has("label_emb.0.0.weight");
        _admDim = IsXL ? weights.Shape("label_emb.0.0.weight")[1] : 0;
        int c = ModelChannels;
        for (int i = 1; weights.Has($"input_blocks.{i}.0.in_layers.0.weight") || weights.Has($"input_blocks.{i}.0.op.weight"); i++)
        {
            var mods = ParseBlock($"input_blocks.{i}.", ref c);
            _input.Add(mods);
        }
        _middle = ParseBlock("middle_block.", ref c);
        for (int i = 0; weights.Has($"output_blocks.{i}.0.in_layers.0.weight"); i++)
            _output.Add(ParseBlock($"output_blocks.{i}.", ref c));
        int ctx = 0;
        foreach (var m in _input.Concat(new[] { _middle }).Concat(_output).SelectMany(b => b))
            if (m.Kind == Kind.Attn) { ctx = weights.Shape(m.Prefix + "transformer_blocks.0.attn2.kv.weight")[1]; break; }
        ContextDim = ctx;
        _fixedHeads = ctx == 768;
        UpcastAttention = ctx == 1024;
    }

    private Module[] ParseBlock(string p, ref int c)
    {
        var mods = new List<Module>();
        for (int j = 0; ; j++)
        {
            string q = $"{p}{j}.";
            if (_w.Has(q + "in_layers.0.weight"))
            {
                int cin = _w.Shape(q + "in_layers.0.weight")[0], cout = _w.Shape(q + "in_layers.2.weight")[0];
                mods.Add(new Module(Kind.Res, q, cin, cout, 0)); c = cout;
            }
            else if (_w.Has(q + "norm.weight") && _w.Has(q + "proj_in.weight"))
            {
                int depth = 0; while (_w.Has($"{q}transformer_blocks.{depth}.norm1.weight")) depth++;
                mods.Add(new Module(Kind.Attn, q, c, c, depth));
            }
            else if (_w.Has(q + "op.weight")) mods.Add(new Module(Kind.Down, q + "op.", c, c, 0));
            else if (_w.Has(q + "conv.weight")) mods.Add(new Module(Kind.Up, q + "conv.", c, c, 0));
            else break;
        }
        return mods.ToArray();
    }

    private int Heads(int C) => _fixedHeads ? 8 : C / 64;

    // ---------------------------------------------------------------------------------------------------
    // workspace: role slots that only grow (a replaced buffer is kept until Dispose: a browser may still have a dispatch on it queued)
    // ---------------------------------------------------------------------------------------------------
    private FV Slot(int slot, long n)
    {
        if (!_slots.TryGetValue(slot, out var b) || b.Length < n)
        {
            if (b != null) _retired.Add(b);
            b = _acc.Allocate1D<float>(Math.Max(n, 64));
            _slots[slot] = b;
        }
        return b.View.SubView(0, n);
    }
    private const int SlotH0 = 0, SlotH1 = 1, SlotNorm = 2, SlotT = 3, SlotWide = 4, SlotMid = 5, SlotQ = 6, SlotKv = 7, SlotEmb = 8, SlotProj = 10, SlotIn = 11, SlotOut = 12,
        SlotTFreq = 13, SlotTHid = 14, SlotY = 15, SlotYHid = 16, SlotCtx = 17, SlotScores = 18, SlotHostIn = 19, SlotHostOut = 20, SlotHostConcat = 21, SlotFreeU = 22, SlotShrink = 23, SlotShrinkUp = 24, SlotTileIn = 25, SlotTileOut = 26, SlotSagMass = 27, SlotSagStats = 28, SlotSagPart = 29,
        SlotSkip = 100;   // + block index: one buffer per skip connection

    private ArrayView1D<int, Stride1D.Dense> Table(int inH, int inW, int outH, int outW, int k, int stride, int pad, int up)
    {
        var key = (inH, inW, outH, outW, k, stride, pad, up);
        if (_tables.TryGetValue(key, out var t)) return t.View;
        t = _acc.Allocate1D<int>((long)outH * outW * 27);
        _k.BuildTable(t.View, inH, inW, outH, outW, k, stride, pad, up);
        _tables[key] = t;
        return t.View;
    }

    // ---------------------------------------------------------------------------------------------------
    // the modules
    // ---------------------------------------------------------------------------------------------------
    private void ResBlock(Module m, FV x, FV dst, FV embAct, int H, int W)
    {
        int rows = H * W, cin = m.Cin, cout = m.Cout;
        string p = m.Prefix;
        var nIn = Slot(SlotNorm, (long)rows * Math.Max(cin, cout)).SubView(0, (long)rows * cin);
        _k.GroupNorm(x, nIn, _w[p + "in_layers.0.weight"], _w[p + "in_layers.0.bias"], rows, cin, 32, 1e-5f, silu: true);
        var c1 = Slot(SlotT, (long)rows * cout);
        _k.Conv(nIn, Table(H, W, H, W, 3, 1, 1, 1), _w[p + "in_layers.2.weight"], _w[p + "in_layers.2.bias"], c1, rows, cin, cout, 9);
        var proj = Slot(SlotProj, cout);
        _k.Linear(embAct, _w[p + "emb_layers.1.weight"], _w[p + "emb_layers.1.bias"], proj, 1, _timeDim, cout);
        _k.AddRowVector(c1, proj, rows, cout);
        var nOut = Slot(SlotNorm, (long)rows * Math.Max(cin, cout)).SubView(0, (long)rows * cout);
        _k.GroupNorm(c1, nOut, _w[p + "out_layers.0.weight"], _w[p + "out_layers.0.bias"], rows, cout, 32, 1e-5f, silu: true);
        _k.Conv(nOut, Table(H, W, H, W, 3, 1, 1, 1), _w[p + "out_layers.3.weight"], _w[p + "out_layers.3.bias"], dst, rows, cout, cout, 9);
        if (_w.Has(p + "skip_connection.weight")) _k.Linear(x, _w[p + "skip_connection.weight"], _w[p + "skip_connection.bias"], dst, rows, cin, cout, 0, dst);
        else _k.AddInPlace(dst, x, (long)rows * cout);
    }

    private void Attention(FV q, int qStride, int qOff, FV kv, int kvStride, int kOff, int vOff, FV o, int Nq, int Nk, int C)
    {
        int H = Heads(C), D = C / H;
        FV scores = _dk.UsesFlashAttention(D) ? Slot(SlotScores, 1) : Slot(SlotScores, (long)H * Nq * Nk);
        _dk.Attention(q, qStride, qOff, kv, kvStride, kOff, vOff, scores, o, Nq, Nk, H, D);
    }

    private void Transformer(Module m, FV x, FV dst, FV ctx, int ctxRows, int H, int W)
    {
        int rows = H * W, C = m.Cin;
        string p = m.Prefix;
        var n = Slot(SlotNorm, (long)rows * C);
        _k.GroupNorm(x, n, _w[p + "norm.weight"], _w[p + "norm.bias"], rows, C, 32, 1e-6f, silu: false);
        var h = Slot(SlotT, (long)rows * C);
        _k.Linear(n, _w[p + "proj_in.weight"], _w[p + "proj_in.bias"], h, rows, C, C);
        for (int j = 0; j < m.Depth; j++)
        {
            string b = $"{p}transformer_blocks.{j}.";
            // self attention
            _k.Sparse.LayerNorm(h, n, _w[b + "norm1.weight"], _w[b + "norm1.bias"], rows, C, 1e-5f);
            var qkv = Slot(SlotWide, (long)rows * 8 * C).SubView(0, (long)rows * 3 * C);
            bool pag = PerturbMiddleSelfAttention && p.StartsWith("middle_block.", StringComparison.Ordinal);
            var grid = pag ? null : HyperTileGrid(rows);
            bool tileKv = HyperTile?.TileKeysAndValues == true;
            FV qkvIn = n;
            if (grid is { } g0 && tileKv)
            {
                // true tiled attention: the tokens in tile order (q, k and v are per token), attention per tile, back to the original order below (to_out is per token)
                qkvIn = Slot(SlotTileIn, (long)rows * C);
                _k.TileRows(n, qkvIn, rows, g0.Th, g0.Tw, g0.Nw, C, inverse: false);
            }
            _k.Linear(qkvIn, _w[b + "attn1.qkv.weight"], null, qkv, rows, C, 3 * C, exact: UpcastAttention);
            if (RecordMiddleSelfAttention && j == 0 && grid == null && !pag && p.StartsWith("middle_block.", StringComparison.Ordinal))
            {
                int Hh = Heads(C);
                _k.AttentionKeyMass(qkv, Slot(SlotSagMass, rows), Slot(SlotSagStats, 2L * Hh * rows), Slot(SlotSagPart, (long)Hh * rows), rows, C, Hh);
                _sagN = rows;
            }
            var a = Slot(SlotMid, (long)rows * 4 * C).SubView(0, (long)rows * C);
            if (pag)
                _k.CopyColumnsFrom(qkv, a, rows, C, 3 * C, 2 * C);                     // PAG: softmax(q k^T) replaced by the identity -> the values
            else if (grid is { } tg)
            {
                int T = tg.Th * tg.Tw, tiles = tg.Nh * tg.Nw;
                var aT = Slot(SlotTileOut, (long)rows * C);
                // ComfyUI: q in tile order, k / v in the original order, so tile t's keys are the band of rows [t * T, (t + 1) * T)
                var qT = tileKv ? qkv : Slot(SlotTileIn, (long)rows * 3 * C);
                if (!tileKv) _k.TileRows(qkv, qT, rows, tg.Th, tg.Tw, tg.Nw, 3 * C, inverse: false);
                for (int t = 0; t < tiles; t++)
                {
                    var qt = qT.SubView((long)t * T * 3 * C, (long)T * 3 * C);
                    var kvt = qkv.SubView((long)t * T * 3 * C, (long)T * 3 * C);
                    Attention(qt, 3 * C, 0, kvt, 3 * C, C, 2 * C, aT.SubView((long)t * T * C, (long)T * C), T, T, C);
                }
                _k.TileRows(aT, a, rows, tg.Th, tg.Tw, tg.Nw, C, inverse: true);
            }
            else Attention(qkv, 3 * C, 0, qkv, 3 * C, C, 2 * C, a, rows, rows, C);
            _k.Linear(a, _w[b + "attn1.to_out.0.weight"], _w[b + "attn1.to_out.0.bias"], h, rows, C, C, 0, h);
            // cross attention to the text
            _k.Sparse.LayerNorm(h, n, _w[b + "norm2.weight"], _w[b + "norm2.bias"], rows, C, 1e-5f);
            var q = Slot(SlotQ, (long)rows * C);
            _k.Linear(n, _w[b + "attn2.to_q.weight"], null, q, rows, C, C, exact: UpcastAttention);
            var kv = Slot(SlotKv, (long)ctxRows * 2 * C);
            _k.Linear(ctx, _w[b + "attn2.kv.weight"], null, kv, ctxRows, ContextDim, 2 * C, exact: UpcastAttention);
            Attention(q, C, 0, kv, 2 * C, 0, C, a, rows, ctxRows, C);
            _k.Linear(a, _w[b + "attn2.to_out.0.weight"], _w[b + "attn2.to_out.0.bias"], h, rows, C, C, 0, h);
            // feed forward (GEGLU)
            _k.Sparse.LayerNorm(h, n, _w[b + "norm3.weight"], _w[b + "norm3.bias"], rows, C, 1e-5f);
            int inner = _w.Shape(b + "ff.net.2.weight")[1];
            var f = Slot(SlotWide, (long)rows * 8 * C).SubView(0, (long)rows * 2 * inner);
            _k.Linear(n, _w[b + "ff.net.0.proj.weight"], _w[b + "ff.net.0.proj.bias"], f, rows, C, 2 * inner);
            var g = Slot(SlotMid, (long)rows * 4 * C).SubView(0, (long)rows * inner);
            _k.Geglu(f, g, rows, inner);
            _k.Linear(g, _w[b + "ff.net.2.weight"], _w[b + "ff.net.2.bias"], h, rows, inner, C, 0, h);
        }
        _k.Linear(h, _w[p + "proj_out.weight"], _w[p + "proj_out.bias"], dst, rows, C, C, 0, x);
    }

    /// <summary>
    /// ComfyUI's hypertile_in, literally: tiles when the token count is (latH / 2^i) * (latW / 2^i) for i &lt;= MaxDepth; the token grid is taken as
    /// round(sqrt(hw * aspect)) x round(sqrt(hw / aspect)) with aspect = latW / latH - that is W x H, transposed for a non-square latent, as ComfyUI does - and each side's
    /// tile count is random_divisor(side, latent tile size, SwapSize). Null = no tiling.
    /// </summary>
    private (int Nh, int Nw, int Th, int Tw)? HyperTileGrid(int hw)
    {
        if (HyperTile is not { } ht) return null;
        int depth = -1;
        for (int i = 0; i <= ht.MaxDepth; i++)
            if ((_latH / Math.Pow(2, i)) * (_latW / Math.Pow(2, i)) == hw) { depth = i; break; }
        if (depth < 0) return null;
        double aspect = (double)_latW / _latH;
        int h = (int)Math.Round(Math.Sqrt(hw * aspect), MidpointRounding.ToEven), w = (int)Math.Round(Math.Sqrt(hw / aspect), MidpointRounding.ToEven);
        int factor = ht.ScaleDepth ? 1 << depth : 1;
        int tile = Math.Max(32, ht.TileSize) / 8 * factor;
        int nh = RandomDivisor(h, tile, ht), nw = RandomDivisor(w, tile, ht);
        if (nh * nw <= 1 || h * w != hw) return null;
        return (nh, nw, h / nh, w / nw);
    }

    private static readonly Random SharedRng = new();
    private static int RandomDivisor(int value, int minValue, SdHyperTile ht)
    {
        minValue = Math.Min(minValue, value);
        var ns = new List<int>();
        for (int i = minValue; i <= value && ns.Count < Math.Max(1, ht.SwapSize); i++) if (value % i == 0) ns.Add(value / i);
        // ComfyUI: randint(low=0, high=len(ns) - 1) - torch's high is exclusive, so the last candidate is never chosen (SwapSize 2 always takes the first)
        int idx = ns.Count > 1 ? (ht.Rng ?? SharedRng).Next(ns.Count - 1) : 0;
        return ns[idx];
    }

    private void Down(Module m, FV x, FV dst, int H, int W)
    {
        int oh = (H + 1) / 2, ow = (W + 1) / 2;
        _k.Conv(x, Table(H, W, oh, ow, 3, 2, 1, 1), _w[m.Prefix + "weight"], _w[m.Prefix + "bias"], dst, oh * ow, m.Cin, m.Cout, 9);
    }

    // nearest upsampling to (oh, ow) fused into the convolution: twice the size, or (ComfyUI's Upsample(output_shape)) the size of the skip connection it meets next
    private void Up(Module m, FV x, FV dst, int H, int W, int oh, int ow)
    {
        _k.Conv(x, Table(H, W, oh, ow, 3, 1, 1, 2), _w[m.Prefix + "weight"], _w[m.Prefix + "bias"], dst, oh * ow, m.Cin, m.Cout, 9);
    }

    /// <summary>
    /// Run one block's modules from <paramref name="x"/>; the last writes into <paramref name="last"/>, or (null) into whichever ping-pong slot its input is not in.
    /// Returns the output view and size.
    /// </summary>
    private (FV View, int H, int W, int C) RunBlock(Module[] mods, FV x, FV? last, FV embAct, FV ctx, int ctxRows, int H, int W, int c, long maxElems,
        (int H, int W)? upTo = null)
    {
        var cur = x;
        for (int i = 0; i < mods.Length; i++)
        {
            var m = mods[i];
            bool isLast = i == mods.Length - 1;
            int oh = m.Kind == Kind.Down ? (H + 1) / 2 : m.Kind == Kind.Up ? upTo?.H ?? 2 * H : H;
            int ow = m.Kind == Kind.Down ? (W + 1) / 2 : m.Kind == Kind.Up ? upTo?.W ?? 2 * W : W;
            long n = (long)oh * ow * m.Cout;
            FV dst = isLast && last.HasValue ? last.Value.SubView(0, n) : PickFree(cur, maxElems).SubView(0, n);
            switch (m.Kind)
            {
                case Kind.Res: ResBlock(m, cur, dst, embAct, H, W); break;
                case Kind.Attn: Transformer(m, cur, dst, ctx, ctxRows, H, W); break;
                case Kind.Down: Down(m, cur, dst, H, W); break;
                case Kind.Up: Up(m, cur, dst, H, W, oh, ow); break;
            }
            cur = dst; H = oh; W = ow; c = m.Cout;
        }
        return (cur, H, W, c);
    }

    private FV PickFree(FV cur, long maxElems)
    {
        var a = Slot(SlotH0, maxElems); var b = Slot(SlotH1, maxElems);
        IContiguousArrayView cv = cur.BaseView, av = a.BaseView;
        return cv.Buffer == av.Buffer ? b : a;
    }

    // ---------------------------------------------------------------------------------------------------
    // forward
    // ---------------------------------------------------------------------------------------------------

    /// <summary>
    /// One prediction on the device. <paramref name="latent"/> is [inChannels, h, w] channel-first (the scaled latent the sampler works on, the model input as diffusers gets it),
    /// <paramref name="context"/> [contextRows, ContextDim] row-major (the text encoder output; any number of rows: ComfyUI concatenates 77-token chunks for long prompts),
    /// <paramref name="y"/> SDXL's label vector [adm] (<see cref="SdxlLabel"/>) or null, <paramref name="output"/> [outChannels, h, w].
    /// </summary>
    public Task ForwardAsync(FV latent, int h, int w, double timestep, FV context, int contextRows, FV? y, FV output, CancellationToken cancellationToken = default)
        => RunAsync(latent, h, w, timestep, context, contextRows, y, output, null, default, null, null, cancellationToken);

    /// <summary>
    /// One prediction of an inpainting UNet (<see cref="ConcatChannels"/> &gt; 0) on the device: <paramref name="latent"/> [OutChannels, h, w] is the latent being denoised and
    /// <paramref name="concat"/> [ConcatChannels, h, w] channel-first what it is concatenated with (the mask [1, h, w] at latent resolution, then the masked image's scaled
    /// latent [4, h, w]: diffusers' StableDiffusionInpaintPipeline order and ComfyUI's c_concat). Joined on the device; the rest as the plain overload, ControlNets included.
    /// </summary>
    public Task ForwardAsync(FV latent, FV concat, int h, int w, double timestep, FV context, int contextRows, FV? y, FV output,
        IReadOnlyList<SdControlResiduals>? controls = null, CancellationToken cancellationToken = default)
    {
        if (ConcatChannels == 0) throw new InvalidOperationException($"this UNet takes {InChannels} input channels and no concatenated input (not an inpainting model)");
        return RunAsync(latent, h, w, timestep, context, contextRows, y, output, controls, default, null, concat, cancellationToken);
    }

    /// <summary>
    /// <see cref="ForwardAsync(FV, int, int, double, FV, int, FV?, FV, CancellationToken)"/> with ControlNet residuals (<see cref="SdControlNet.RunAsync(FV, int, int, double, FV, int, FV, FV?, SdControlResiduals?, CancellationToken)"/>):
    /// each set, times its strength, is added to the skip connections and the middle block's output (several ControlNets add up, as ComfyUI chains them).
    /// </summary>
    public Task ForwardAsync(FV latent, int h, int w, double timestep, FV context, int contextRows, FV? y, FV output, IReadOnlyList<SdControlResiduals>? controls,
        CancellationToken cancellationToken = default)
        => RunAsync(latent, h, w, timestep, context, contextRows, y, output, controls, default, null, null, cancellationToken);

    /// <summary>
    /// The core: the input blocks, the middle block, then (unless <paramref name="encoderOut"/> is given: the ControlNet's encoder pass) the output blocks and the head.
    /// <paramref name="hint"/> (channel-last [h*w, ModelChannels]) is added to conv_in's output; <paramref name="encoderOut"/> sees (index, output, H, W, C) of conv_in, every input
    /// block and the middle block, in that order.
    /// </summary>
    internal async Task RunAsync(FV latent, int h, int w, double timestep, FV context, int contextRows, FV? y, FV output, IReadOnlyList<SdControlResiduals>? controls,
        FV? hint, Action<int, FV, int, int, int>? encoderOut, FV? concat, CancellationToken cancellationToken)
    {
        int latC = concat.HasValue ? InChannels - ConcatChannels : InChannels;
        if (latent.Length != (long)latC * h * w) throw new ArgumentException($"the latent must be [{latC}, {h}, {w}]");
        if (concat.HasValue && concat.Value.Length != (long)ConcatChannels * h * w) throw new ArgumentException($"the concatenated input must be [{ConcatChannels}, {h}, {w}]");
        if (encoderOut == null && OutChannels == 0) throw new InvalidOperationException("this UNet holds the encoder half only (a ControlNet's): it cannot predict");
        if (IsXL != y.HasValue) throw new ArgumentException(IsXL ? "an SDXL UNet needs the label vector (pooled text + size ids)" : "only an SDXL UNet takes a label vector");
        int mc = ModelChannels, rows0 = h * w;
        _latH = h; _latW = w;
        _sagN = 0;
        // the largest activation: a level's pixels x the widest concatenated input of that level (output blocks read h | skip)
        long maxElems = (long)rows0 * mc * 2;
        {
            int H = h, W = w, c = mc;
            foreach (var blk in _input.Concat(new[] { _middle }).Concat(_output))
                foreach (var m in blk)
                {
                    maxElems = Math.Max(maxElems, (long)H * W * (m.Cin + mc));
                    if (m.Kind == Kind.Down) { H = (H + 1) / 2; W = (W + 1) / 2; }
                    if (m.Kind == Kind.Up) { H *= 2; W *= 2; }
                    maxElems = Math.Max(maxElems, (long)H * W * m.Cout * 2);
                }
        }

        // time (and SDXL label) embedding -> SiLU, shared by every resnet
        var tFreq = Slot(SlotTFreq, mc);
        tFreq.CopyFromCPU(SdKernels.TimestepEmbedding(timestep, mc));
        var tHid = Slot(SlotTHid, _timeDim); var emb = Slot(SlotEmb, _timeDim);
        _k.Linear(tFreq, _w["time_embed.0.weight"], _w["time_embed.0.bias"], tHid, 1, mc, _timeDim, 1);
        _k.Linear(tHid, _w["time_embed.2.weight"], _w["time_embed.2.bias"], emb, 1, _timeDim, _timeDim);
        if (IsXL)
        {
            var yh = Slot(SlotYHid, _timeDim);
            _k.Linear(y!.Value, _w["label_emb.0.0.weight"], _w["label_emb.0.0.bias"], yh, 1, _admDim, _timeDim, 1);
            _k.Linear(yh, _w["label_emb.0.2.weight"], _w["label_emb.0.2.bias"], emb, 1, _timeDim, _timeDim, 0, emb);
        }
        _k.Sparse.SiluInPlace(emb, _timeDim);                 // every resnet's emb_layers = SiLU, Linear
        var embAct = emb;

        // conv_in on the channel-last latent
        var xin = Slot(SlotIn, (long)rows0 * InChannels);
        if (concat.HasValue)
        {
            _k.ChannelFirstToLastColumns(latent, xin, rows0, latC, InChannels, 0);
            _k.ChannelFirstToLastColumns(concat.Value, xin, rows0, ConcatChannels, InChannels, latC);
        }
        else _dk.Transpose(latent, xin, InChannels, rows0);
        var skips = new List<(FV View, int H, int W, int C)>();
        var s0 = Slot(SlotSkip, (long)rows0 * mc);
        _k.Conv(xin, Table(h, w, h, w, 3, 1, 1, 1), _w["input_blocks.0.0.weight"], _w["input_blocks.0.0.bias"], s0, rows0, InChannels, mc, 9);
        if (hint.HasValue) _k.AddInPlace(s0, hint.Value, (long)rows0 * mc);
        skips.Add((s0, h, w, mc));
        Tap?.Invoke("conv_in", s0, h, w, mc);
        encoderOut?.Invoke(0, s0, h, w, mc);

        int Hc = h, Wc = w, Cc = mc;
        var cur = s0;
        int level = 0;
        for (int i = 0; i < _input.Count; i++)
        {
            var mods = _input[i];
            int oh = Hc, ow = Wc, oc = mods[^1].Cout;
            foreach (var m in mods) { if (m.Kind == Kind.Down) { oh = (oh + 1) / 2; ow = (ow + 1) / 2; } }
            var dst = Slot(SlotSkip + 1 + i, (long)oh * ow * oc);
            (cur, Hc, Wc, Cc) = RunBlock(mods, cur, dst, embAct, context, contextRows, Hc, Wc, Cc, maxElems);
            if (DeepShrink is { } ds && ds.Block == i + 1 && encoderOut == null)
            {
                // ComfyUI's input_block_patch(_after_skip): round(side / factor) with Python's round (half to even)
                int nh = (int)Math.Round(Hc * (1.0 / ds.Factor), MidpointRounding.ToEven), nw = (int)Math.Round(Wc * (1.0 / ds.Factor), MidpointRounding.ToEven);
                var small = Slot(SlotShrink, (long)nh * nw * Cc);
                if (ds.AfterSkip) skips.Add((cur, Hc, Wc, Cc));
                _k.Resize(cur, small, Hc, Wc, nh, nw, Cc, (int)ds.Down);
                (cur, Hc, Wc) = (small, nh, nw);
                if (!ds.AfterSkip) skips.Add((cur, Hc, Wc, Cc));
            }
            else skips.Add((cur, Hc, Wc, Cc));
            encoderOut?.Invoke(i + 1, cur, Hc, Wc, Cc);
            bool levelEnd = mods.Any(m => m.Kind == Kind.Down) || i == _input.Count - 1;
            if (levelEnd) { Tap?.Invoke($"down_{level}", cur, Hc, Wc, Cc); level++; }
            if (_acc.IsBrowser()) await _acc.SynchronizeAsync();
            cancellationToken.ThrowIfCancellationRequested();
        }

        (cur, Hc, Wc, Cc) = RunBlock(_middle, cur, null, embAct, context, contextRows, Hc, Wc, Cc, maxElems);
        Tap?.Invoke("mid", cur, Hc, Wc, Cc);
        if (encoderOut != null) { encoderOut(_input.Count + 1, cur, Hc, Wc, Cc); return; }
        if (controls != null)
            foreach (var c in controls)
            {
                if (c.Count != skips.Count + 1) throw new ArgumentException($"a ControlNet gave {c.Count} residuals, this UNet takes {skips.Count + 1}");
                _k.AxpyInPlace(cur, c[skips.Count], c.Strength, (long)Hc * Wc * Cc);
            }
        if (_acc.IsBrowser()) await _acc.SynchronizeAsync();

        level = 0;
        for (int i = 0; i < _output.Count; i++)
        {
            var (sv, sh, sw, sc) = skips[^1];
            skips.RemoveAt(skips.Count - 1);
            if (controls != null)
                foreach (var c in controls) _k.AxpyInPlace(sv, c[skips.Count], c.Strength, (long)sh * sw * sc);     // the skip this block concatenates + strength x residual
            if (sh != Hc || sw != Wc)
            {
                // only deep shrink makes the features miss their skip connection (ComfyUI's output_block_patch resizes them to it)
                if (DeepShrink is not { } dsu) throw new InvalidOperationException($"skip {sh}x{sw} does not meet {Hc}x{Wc}");
                var big = Slot(SlotShrinkUp, (long)sh * sw * Cc);
                _k.Resize(cur, big, Hc, Wc, sh, sw, Cc, (int)dsu.Up);
                (cur, Hc, Wc) = (big, sh, sw);
            }
            if (FreeU is { } fu && (Cc == mc * 4 || Cc == mc * 2))
            {
                // ComfyUI's output_block_patch: chosen by the channel count of h (not by the block index, as diffusers does)
                bool deep = Cc == mc * 4;
                float b = deep ? fu.B1 : fu.B2, s = deep ? fu.S1 : fu.S2;
                var scratch = Slot(SlotFreeU, Math.Max((long)Hc * Wc + 2, (long)sc * 8));
                if (fu.V2) _k.FreeUBackboneV2(cur, Hc * Wc, Cc, Cc / 2, b, scratch);
                else _k.FreeUBackbone(cur, Hc * Wc, Cc, Cc / 2, b);
                _k.FreeUFourier(sv, sh, sw, sc, s, scratch);
            }
            var cat = PickFree(cur, maxElems).SubView(0, (long)Hc * Wc * (Cc + sc));
            _k.ConcatChannels(cur, sv, cat, Hc * Wc, Cc, sc);
            var mods = _output[i];
            var first = mods[0];
            if (first.Cin != Cc + sc) throw new InvalidOperationException($"output block {i}: expects {first.Cin} channels, got {Cc + sc}");
            // an upsampling block resizes to the next skip connection's size (ComfyUI: Upsample(output_shape = hs[-1].shape)): 2x unless the sides are odd
            (int, int)? upTo = skips.Count > 0 ? (skips[^1].H, skips[^1].W) : null;
            (cur, Hc, Wc, Cc) = RunBlock(mods, cat, null, embAct, context, contextRows, Hc, Wc, Cc + sc, maxElems, upTo);
            bool levelEnd = mods.Any(m => m.Kind == Kind.Up) || i == _output.Count - 1;
            if (levelEnd) { Tap?.Invoke($"up_{level}", cur, Hc, Wc, Cc); level++; }
            if (_acc.IsBrowser()) await _acc.SynchronizeAsync();
            cancellationToken.ThrowIfCancellationRequested();
        }

        // out: GroupNorm, SiLU, conv to the latent channels, back to channel-first
        var n = Slot(SlotNorm, (long)rows0 * Cc);
        _k.GroupNorm(cur, n, _w["out.0.weight"], _w["out.0.bias"], rows0, Cc, 32, 1e-5f, silu: true);
        var o = Slot(SlotOut, (long)rows0 * OutChannels);
        _k.Conv(n, Table(h, w, h, w, 3, 1, 1, 1), _w["out.2.weight"], _w["out.2.bias"], o, rows0, Cc, OutChannels, 9);
        _dk.Transpose(o, output, rows0, OutChannels);
    }

    /// <summary>
    /// One prediction from host arrays: <paramref name="latent"/> [inChannels, h, w], <paramref name="context"/> [contextRows, ContextDim]; returns [outChannels, h, w].
    /// SDXL needs <paramref name="xl"/>.
    /// </summary>
    public async Task<float[]> ForwardAsync(float[] latent, int h, int w, double timestep, float[] context, int contextRows, SdxlConditioning? xl = null, CancellationToken cancellationToken = default)
        => await ForwardAsync(latent, h, w, timestep, context, contextRows, xl, null, cancellationToken);

    /// <summary>The host-array prediction with ControlNet residuals (see the device overload).</summary>
    public Task<float[]> ForwardAsync(float[] latent, int h, int w, double timestep, float[] context, int contextRows, SdxlConditioning? xl,
        IReadOnlyList<SdControlResiduals>? controls, CancellationToken cancellationToken = default)
        => HostForwardAsync(latent, null, h, w, timestep, context, contextRows, xl, controls, cancellationToken);

    /// <summary>
    /// One prediction of an inpainting UNet from host arrays: <paramref name="latent"/> [OutChannels, h, w], <paramref name="concat"/> [ConcatChannels, h, w] (mask, then the masked
    /// image's latent; see the device overload); returns [OutChannels, h, w].
    /// </summary>
    public Task<float[]> ForwardAsync(float[] latent, float[] concat, int h, int w, double timestep, float[] context, int contextRows, SdxlConditioning? xl = null,
        IReadOnlyList<SdControlResiduals>? controls = null, CancellationToken cancellationToken = default)
    {
        if (ConcatChannels == 0) throw new InvalidOperationException($"this UNet takes {InChannels} input channels and no concatenated input (not an inpainting model)");
        return HostForwardAsync(latent, concat, h, w, timestep, context, contextRows, xl, controls, cancellationToken);
    }

    private async Task<float[]> HostForwardAsync(float[] latent, float[]? concat, int h, int w, double timestep, float[] context, int contextRows, SdxlConditioning? xl,
        IReadOnlyList<SdControlResiduals>? controls, CancellationToken cancellationToken)
    {
        int latC = concat != null ? InChannels - ConcatChannels : InChannels;
        if (latent.Length != (long)latC * h * w) throw new ArgumentException($"the latent must be [{latC}, {h}, {w}]");
        if (context.Length != (long)contextRows * ContextDim) throw new ArgumentException($"the context must be [{contextRows}, {ContextDim}]");
        var lat = Slot(SlotHostIn, latent.Length); lat.CopyFromCPU(latent);
        FV? cc = null;
        if (concat != null)
        {
            if (concat.Length != (long)ConcatChannels * h * w) throw new ArgumentException($"the concatenated input must be [{ConcatChannels}, {h}, {w}]");
            var cv = Slot(SlotHostConcat, concat.Length); cv.CopyFromCPU(concat); cc = cv;
        }
        var ctx = Slot(SlotCtx, context.Length); ctx.CopyFromCPU(context);
        FV? y = null;
        if (IsXL)
        {
            if (xl == null) throw new ArgumentException("an SDXL UNet needs the SDXL conditioning (pooled text embedding and size ids)");
            var yv = Slot(SlotY, _admDim); yv.CopyFromCPU(SdxlLabel(xl, _admDim)); y = yv;
        }
        var o = Slot(SlotHostOut, (long)OutChannels * h * w);
        await RunAsync(lat, h, w, timestep, ctx, contextRows, y, o, controls, null, null, cc, cancellationToken);
        await _acc.SynchronizeAsync();
        return await o.CopyToHostAsync();
    }

    /// <summary>SDXL's label vector: the pooled text embedding, then the timestep embedding (256 wide each) of the six size numbers.</summary>
    public static float[] SdxlLabel(SdxlConditioning c, int admDim)
    {
        var ids = new double[] { c.OriginalHeight, c.OriginalWidth, c.CropTop, c.CropLeft, c.TargetHeight, c.TargetWidth };
        int per = (admDim - c.Pooled.Length) / ids.Length;
        if (per * ids.Length + c.Pooled.Length != admDim) throw new ArgumentException($"the pooled embedding ({c.Pooled.Length}) does not fit the label width {admDim}");
        var y = new float[admDim];
        c.Pooled.CopyTo(y, 0);
        for (int i = 0; i < ids.Length; i++) SdKernels.TimestepEmbedding(ids[i], per).CopyTo(y, c.Pooled.Length + i * per);
        return y;
    }

    public void Dispose()
    {
        foreach (var b in _slots.Values) b.Dispose();
        _slots.Clear();
        foreach (var r in _retired) r.Dispose();
        _retired.Clear();
        foreach (var t in _tables.Values) t.Dispose();
        _tables.Clear();
        _patches?.Dispose();
        if (_ownsWeights) _w.Dispose();
    }
}
