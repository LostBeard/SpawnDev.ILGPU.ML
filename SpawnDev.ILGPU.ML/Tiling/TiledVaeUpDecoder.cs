using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Graph;
using SpawnDev.ILGPU.ML.Kernels;
using SpawnDev.ILGPU.ML.Tensors;

namespace SpawnDev.ILGPU.ML.Tiling;

/// <summary>
/// Exact-stat, SEAM-FREE tiled SD-Turbo VAE decoder. The VAE decode is where GPU peak blows up (latents are tiny
/// but the decoder upsamples 64²→512² across 512/256/128-channel feature maps). This runs the decoder's UP-BLOCKS
/// tiled — one spatial tile on the GPU at a time (the rest offloaded to CPU via <see cref="TiledFeatureMap"/>) —
/// so the GPU working set is bounded by a single tile, not the full 512² map.
///
/// SEAM-FREE because the two tile-coupling ops are made exact, not approximate:
///   • 3×3 SAME convs refresh a 1px halo from neighbor cores before each conv (<see cref="TiledVaeOps.Conv3x3"/>),
///     so every output pixel sees the same neighborhood it would in a full conv.
///   • GroupNorm uses GLOBAL per-group stats combined across all tiles (<see cref="TiledVaeOps.GroupNorm"/>),
///     so normalization is identical to the full-map norm (this is what kills the brightness seams that the
///     approximate per-tile tiling produced).
/// Result is bit-near-identical to the full decode (float-accumulation order only), with a much lower GPU peak.
///
/// Two phases: (A) run the decoder HEAD whole (post_quant_conv → conv_in → mid_block, at 64² it's cheap) via the
/// session with <see cref="GraphExecutor.BreakAtNode"/>, capturing the mid-block output; (B) run up_blocks 0-3 +
/// conv_norm_out + conv_out tiled here. Weights come from the loaded session: conv weights have predictable
/// diffusers names (decoder.up_blocks.{B}.resnets.{R}.conv{1,2}.weight), the GroupNorm γ/β are auto-named
/// (onnx::Mul_N / onnx::Add_N) and resolved by walking norm → Reshape → Mul(γ) → Add(β).
///
/// Plan: Plans/exact-tiled-vae-decode-2026-06-16.md.
/// </summary>
public sealed class TiledVaeUpDecoder : IDisposable
{
    private readonly InferenceSession _vae;
    private readonly Accelerator _acc;
    private readonly TiledVaeOps _ops;
    private readonly PrecisionConvertKernels _convert;

    // norm scope (e.g. "/decoder/up_blocks.0/resnets.0/norm1") → (γ weight name, β weight name)
    private readonly Dictionary<string, (string gamma, string beta)> _normWeights = new();
    // fp32 GPU copies of fp16 per-channel vectors (γ/β/bias), built once + reused across all tiles.
    private readonly Dictionary<string, MemoryBuffer1D<float, Stride1D.Dense>> _f32cache = new();

    private const int Groups = 32;     // SD VAE GroupNorm groups
    private const float Eps = 1e-5f;   // MUST match NormalizationKernels.InstanceNorm's hardcoded 1e-5f (it ignores
                                       // the ONNX eps attr) — the reference the tiled decode is verified against.
    private const int Halo = 1;        // 3×3 SAME conv needs a 1px halo

    /// <summary>The decoder's mid-block output tensor name = the Phase-A → Phase-B boundary.</summary>
    public const string MidBlockOutputName = "/decoder/Cast_output_0";

    /// <summary>DIAGNOSTIC: invoked after each up-block stage with (session-equivalent node output name, current
    /// full [C,H,W] tensor) so a caller can compare against the captured full-decode intermediate to pinpoint a
    /// diverging op. Off (null) by default.</summary>
    public Action<string, float[]>? OnStage { get; set; }

    public TiledVaeUpDecoder(InferenceSession vae, Accelerator acc)
    {
        _vae = vae; _acc = acc;
        _ops = new TiledVaeOps(acc);
        _convert = new PrecisionConvertKernels(acc);
        BuildNormWeightMap();
    }

    // ── weight resolution ───────────────────────────────────────────────────────────────────────────────

    /// <summary>Walk the graph once to map each GroupNorm's scope → its γ/β weight names. The ONNX GroupNorm
    /// decomposes to Reshape → InstanceNormalization → Reshape → Mul(γ) → Add(β); the γ/β are auto-named
    /// initializers, so we follow the chain from each InstanceNormalization to find them.</summary>
    private void BuildNormWeightMap()
    {
        int n = _vae.NodeCount;
        // producer: output name → node idx; consumers: input name → node indices.
        var producer = new Dictionary<string, int>();
        var consumers = new Dictionary<string, List<int>>();
        var nodes = new (string op, string[] ins, string[] outs)[n];
        for (int i = 0; i < n; i++)
        {
            var node = _vae.GetNode(i); nodes[i] = node;
            foreach (var o in node.outputs) producer[o] = i;
            foreach (var inp in node.inputs)
                (consumers.TryGetValue(inp, out var l) ? l : consumers[inp] = new List<int>()).Add(i);
        }

        int FirstConsumerOfOp(string outName, string op)
        {
            if (!consumers.TryGetValue(outName, out var l)) return -1;
            foreach (var idx in l) if (nodes[idx].op == op) return idx;
            return -1;
        }
        string WeightInput(string[] ins, string notThis)
        {
            foreach (var inp in ins) if (inp != notThis && _vae.TryGetWeight(inp) != null) return inp;
            return "";
        }

        const string normSuffix = "/InstanceNormalization_output_0";
        for (int i = 0; i < n; i++)
        {
            if (nodes[i].op != "InstanceNormalization" || nodes[i].outs.Length == 0) continue;
            string normOut = nodes[i].outs[0];
            if (!normOut.EndsWith(normSuffix)) continue;
            string scope = normOut[..^normSuffix.Length];          // e.g. /decoder/up_blocks.0/resnets.0/norm1

            int reshapeIdx = FirstConsumerOfOp(normOut, "Reshape");
            if (reshapeIdx < 0) continue;
            int mulIdx = FirstConsumerOfOp(nodes[reshapeIdx].outs[0], "Mul");
            if (mulIdx < 0) continue;
            string gamma = WeightInput(nodes[mulIdx].ins, nodes[reshapeIdx].outs[0]);
            int addIdx = FirstConsumerOfOp(nodes[mulIdx].outs[0], "Add");
            string beta = addIdx < 0 ? "" : WeightInput(nodes[addIdx].ins, nodes[mulIdx].outs[0]);
            if (gamma.Length == 0 || beta.Length == 0) continue;
            _normWeights[scope] = (gamma, beta);
        }
    }

    private (string gamma, string beta) NormW(string scope) =>
        _normWeights.TryGetValue(scope, out var w) ? w
        : throw new InvalidOperationException($"GroupNorm γ/β not found for scope '{scope}' (have {_normWeights.Count} norms).");

    private Tensor Weight(string name) =>
        _vae.TryGetWeight(name) ?? throw new InvalidOperationException($"VAE weight '{name}' not found.");

    /// <summary>A per-channel vector (γ/β/bias) as a fp32 GPU view: fp32 weights pass through (session owns them);
    /// fp16 weights are upconverted once into a cached buffer.</summary>
    private async Task<ArrayView1D<float, Stride1D.Dense>> F32Async(string name)
    {
        var t = Weight(name);
        if (!t.IsHalf) return t.Data;
        if (_f32cache.TryGetValue(name, out var cached)) return cached.View;
        var buf = _acc.Allocate1D<float>(t.ElementCount);
        _convert.HalfToFloat(t.HalfData, buf.View, t.ElementCount);
        await _acc.SynchronizeAsync();
        _f32cache[name] = buf;
        return buf.View;
    }

    // ── forward ─────────────────────────────────────────────────────────────────────────────────────────

    /// <summary>Phase B: run up_blocks 0-3 + conv_norm_out + conv_out tiled, starting from the mid-block output
    /// [512,64,64] (packed C,H,W). Returns the decoded image [3,512,512] (packed C,H,W). The tile grid is
    /// rows×cols; more tiles = lower GPU peak (bounded by one tile) at more per-tile overhead.</summary>
    public async Task<float[]> DecodeUpBlocksAsync(float[] midOut, int rows, int cols)
    {
        const int C = 512, H = 64, W = 64;
        if (midOut.Length != (long)C * H * W)
            throw new ArgumentException($"mid-block output must be {C}x{H}x{W}={C * H * W}, got {midOut.Length}.");

        var x = TiledFeatureMap.FromFull(midOut, C, H, W, rows, cols, Halo);
        for (int b = 0; b <= 3; b++)
        {
            for (int r = 0; r <= 2; r++)
            {
                x = await ResnetAsync(x, b, r);
                OnStage?.Invoke($"/decoder/up_blocks.{b}/resnets.{r}/Add_output_0", x.ToFull());
            }
            if (b < 3)
            {
                x = await UpsampleAsync(x, b);
                OnStage?.Invoke($"/decoder/up_blocks.{b}/upsamplers.0/conv/Conv_output_0", x.ToFull());
            }
        }

        // head: conv_norm_out (GroupNorm) → SiLU → conv_out (3×3 → 3 channels)
        int headC = x.Channels;
        var gn = NormW("/decoder/conv_norm_out");
        await _ops.GroupNorm(x, await F32Async(gn.gamma), await F32Async(gn.beta), headC, Groups, Eps);
        await _ops.SiLU(x, headC);
        var convOut = Weight("decoder.conv_out.weight");
        int outCh = convOut.Shape[0];                 // 3
        x = await _ops.Conv3x3(x, convOut, await F32Async("decoder.conv_out.bias"), headC, outCh);
        return x.ToFull();                            // [3,512,512]
    }

    /// <summary>One VAE resnet block (tiled): norm1 → SiLU → conv1 → norm2 → SiLU → conv2 → + residual
    /// (1×1 conv_shortcut when channels change). GroupNorm/SiLU mutate in place, so the residual is a clone.</summary>
    private async Task<TiledFeatureMap> ResnetAsync(TiledFeatureMap x, int b, int r)
    {
        string pre = $"decoder.up_blocks.{b}.resnets.{r}";
        string scope = $"/decoder/up_blocks.{b}/resnets.{r}";
        var conv1 = Weight($"{pre}.conv1.weight");
        var conv2 = Weight($"{pre}.conv2.weight");
        int inC = conv1.Shape[1], outC = conv1.Shape[0];
        var n1 = NormW($"{scope}/norm1");
        var n2 = NormW($"{scope}/norm2");

        var residual = x;
        var h = x.Clone();
        await _ops.GroupNorm(h, await F32Async(n1.gamma), await F32Async(n1.beta), inC, Groups, Eps);
        await _ops.SiLU(h, inC);
        h = await _ops.Conv3x3(h, conv1, await F32Async($"{pre}.conv1.bias"), inC, outC);
        await _ops.GroupNorm(h, await F32Async(n2.gamma), await F32Async(n2.beta), outC, Groups, Eps);
        await _ops.SiLU(h, outC);
        h = await _ops.Conv3x3(h, conv2, await F32Async($"{pre}.conv2.bias"), outC, outC);

        if (inC != outC)
        {
            var sc = Weight($"{pre}.conv_shortcut.weight");
            residual = await _ops.Conv1x1(residual, sc, await F32Async($"{pre}.conv_shortcut.bias"), inC, outC);
        }
        await _ops.AddInPlace(h, residual, outC);
        return h;
    }

    /// <summary>Upsampler: nearest-2× then a 3×3 SAME conv (the diffusers VAE upsampler), tiled + grid-aligned.</summary>
    private async Task<TiledFeatureMap> UpsampleAsync(TiledFeatureMap x, int b)
    {
        int C = x.Channels;
        var up = await _ops.Resize2x(x, C);
        var cw = Weight($"decoder.up_blocks.{b}.upsamplers.0.conv.weight");
        int outC = cw.Shape[0];
        return await _ops.Conv3x3(up, cw, await F32Async($"decoder.up_blocks.{b}.upsamplers.0.conv.bias"), C, outC);
    }

    public void Dispose()
    {
        _ops.Dispose();
        _convert.Dispose();
        foreach (var b in _f32cache.Values) b.Dispose();
        _f32cache.Clear();
    }
}
