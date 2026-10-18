using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Trellis2;
using SpawnDev.ILGPU.ML.UniRig;

namespace SpawnDev.ILGPU.ML.SkinTokens;

using FV = ArrayView1D<float, Stride1D.Dense>;

/// <summary>
/// The mesh encoder of TokenRig: the Michelangelo shape encoder (8 layers, 512 query tokens; <see cref="UniRigEncoder"/>) followed by <c>output_proj</c>, a
/// Linear 512 -> 896 and an RMS norm. The norm's epsilon is the SHIPPED one: the model runs in bf16 and torch.nn.RMSNorm(eps=None) uses the machine epsilon of the
/// input dtype, 2^-7 = 0.0078125 (not float32's 1.2e-7).
/// </summary>
public sealed class SkinTokensMeshEncoder : IDisposable
{
    public const int Tokens = 512, Width = SkinTokensDecoder.Hidden;
    public const float NormEps = 0.0078125f;

    private readonly Accelerator _acc;
    private readonly GpuWeights _w;
    private readonly DiTKernels _k;
    private readonly SkinTokensKernels _u;
    private readonly UniRigEncoder _enc;
    private MemoryBuffer1D<float, Stride1D.Dense>? _a, _b;
    private MemoryBuffer1D<float, Stride1D.Dense>? _latents;

    /// <summary>The [512, 512] output of the Michelangelo encoder's ln_post from the last <see cref="Encode"/> (before output_proj); for diagnostics.</summary>
    public MemoryBuffer1D<float, Stride1D.Dense>? LastLatents => _latents;
    /// <summary>Diagnostics hook of the underlying <see cref="UniRigEncoder"/>.</summary>
    public Action<string, FV, int, int>? Probe { get => _enc.Probe; set => _enc.Probe = value; }

    public SkinTokensMeshEncoder(Accelerator accelerator, GpuWeights weights)
    {
        _acc = accelerator; _w = weights;
        _k = new DiTKernels(accelerator); _u = new SkinTokensKernels(accelerator);
        _enc = new UniRigEncoder(accelerator, weights, new UniRigEncoder.Options("mesh_encoder.encoder.", SelfLayers: 8, QueryTokens: Tokens, OutputProj: null));
    }

    /// <summary>[n,3] points and unit normals in [-1,1]; <paramref name="queryRows"/> the 512 farthest-point-sampled query points. Returns [512, 896]; the buffer is reused by the next call.</summary>
    public MemoryBuffer1D<float, Stride1D.Dense> Encode(FV pts, FV nrm, int n, int[] queryRows)
    {
        var tokens = _enc.Encode(pts, nrm, n, queryRows);
        _latents = tokens;
        _a ??= _acc.Allocate1D<float>((long)Tokens * Width); _b ??= _acc.Allocate1D<float>((long)Tokens * Width);
        _k.Linear(tokens.View, _w.BF16("output_proj.0.weight"), _w.BF16("output_proj.0.bias"), _a.View, Tokens, UniRigEncoder.Width, Width);
        _u.RmsNorm(_a.View, _b.View, _w.BF16("output_proj.1.weight"), Tokens, Width, NormEps);
        return _b;
    }

    public void Dispose() { _enc.Dispose(); _a?.Dispose(); _b?.Dispose(); _a = _b = null; }
}

/// <summary>
/// SkinTokens / TokenRig end to end on the device: a point cloud with normals becomes a skeleton AND per-point skin weights in one pass.
/// <code>
/// prefix   = mesh encoder(points)                                        [512, 896]
/// cond     = skin VAE cond encoder(points)                               [384, 512]
/// tokens   = Qwen3 decoder, greedy:  bos cls skeleton... SWITCH  (4 skin tokens) x J                     under the skeleton grammar, then skin tokens only
/// skin     = VAE decoder(tokens of bone j, cond)(every point)            [n, J] sigmoid weights
/// </code>
/// The shipped inference samples (beam search 10, top-k 10, temperature 1.5, repetition penalty 2) which is random; greedy is deterministic and is what the goldens pin.
/// The skeleton grammar is UniRig's (<see cref="UniRigTokenizer"/>; the vocabulary is identical, 267 ids, the switch token is its <c>eos</c> 258); after the switch only skin
/// tokens (ids 267..33034) are allowed until 4 per bone are written, then the end token 33035.
/// </summary>
public sealed class SkinTokensGenerator : IDisposable
{
    public const int SkeletonVocab = UniRigTokenizer.VocabSize, EndToken = SkinTokensDecoder.Vocab - 1, SkinCodes = SkinTokensDecoder.Vocab - SkeletonVocab - 1;

    private readonly Accelerator _acc;
    private readonly SkinTokensMeshEncoder _encoder;
    private readonly SkinTokensDecoder _decoder;
    private readonly SkinTokensVae _vae;

    /// <param name="weights">The SkinTokens export (export_weights.py skintokens).</param>
    public SkinTokensGenerator(Accelerator accelerator, GpuWeights weights)
    {
        _acc = accelerator;
        _encoder = new SkinTokensMeshEncoder(accelerator, weights);
        _decoder = new SkinTokensDecoder(accelerator, weights, SkinTokensMeshEncoder.Tokens + 8);
        _vae = new SkinTokensVae(accelerator, weights);
    }

    /// <summary>Wall seconds per stage of the last <see cref="GenerateAsync"/>: mesh encoder, cond encoder, prefill, skeleton tokens, skin tokens, skin decode.</summary>
    public List<(string Stage, double Seconds)> Timings { get; } = new();

    /// <param name="Ids">Every id, start tokens first: [bos, cls, skeleton..., switch, skin tokens..., end].</param>
    /// <param name="Skeleton">Joints in the normalised ([-1,1]) space and parents.</param>
    /// <param name="Skin">[n, J] weights in 0..1 as the VAE writes them (not normalised to sum 1).</param>
    public sealed record Result(int[] Ids, UniRigTokenizer.Skeleton Skeleton, float[] Skin, int Points);

    /// <summary>The prefix, the cond latents and the generated rig of a point cloud already on the device.</summary>
    public async Task<Result> GenerateAsync(FV pts, FV nrm, int n, int[] meshQueryRows, int[] condQueryRows, int maxSkeletonTokens = 1400, int cls = UniRigTokenizer.ClsArticulationXl)
    {
        Timings.Clear(); var sw = System.Diagnostics.Stopwatch.StartNew();
        void Mark(string stage) { Timings.Add((stage, sw.Elapsed.TotalSeconds)); sw.Restart(); }
        var prefix = _encoder.Encode(pts, nrm, n, meshQueryRows);
        await _acc.WaitAsync(); Mark("mesh encoder");
        _vae.EncodeCond(pts, nrm, n, condQueryRows);
        await _acc.WaitAsync(); Mark("cond encoder");
        var ids = new List<int> { UniRigTokenizer.Bos, cls };
        var logits = await _decoder.PrefillAsync(prefix.View, SkinTokensMeshEncoder.Tokens, ids.ToArray());
        Mark("prefill");

        // phase 1: the skeleton under UniRig's grammar, until the switch token
        var state = UniRigTokenizer.StateAfter(ids.ToArray());
        for (int step = 0; ; step++)
        {
            if (step >= maxSkeletonTokens) throw new InvalidOperationException($"no complete skeleton within {maxSkeletonTokens} tokens");
            int best = -1; float bv = float.NegativeInfinity;
            for (int v = 0; v < SkeletonVocab; v++)
                if (UniRigTokenizer.IsAllowed(state, v) && logits[v] > bv) { bv = logits[v]; best = v; }
            ids.Add(best);
            logits = await _decoder.StepAsync(best);
            if (best == UniRigTokenizer.Eos) break;
            state = UniRigTokenizer.Advance(state, best);
        }
        var skeleton = UniRigTokenizer.Detokenize(ids.ToArray());
        int J = skeleton.Count;
        Mark($"skeleton tokens ({ids.Count - 2})");

        // phase 2: 4 skin tokens per bone, nothing else
        int need = J * SkinTokensVae.TokensPerBone;
        for (int i = 0; i < need; i++)
        {
            int best = SkeletonVocab; float bv = float.NegativeInfinity;
            for (int v = SkeletonVocab; v < EndToken; v++) if (logits[v] > bv) { bv = logits[v]; best = v; }
            ids.Add(best);
            if (i + 1 < need) logits = await _decoder.StepAsync(best);
        }
        ids.Add(EndToken);
        Mark($"skin tokens ({need})");

        // the skin weights: every bone's 4 codes against every point
        using var skin = _acc.Allocate1D<float>((long)n * J);
        _vae.BeginSkin(pts, nrm, n);
        int first = ids.Count - 1 - need;
        for (int j = 0; j < J; j++)
        {
            var codes = new int[SkinTokensVae.TokensPerBone];
            for (int t = 0; t < codes.Length; t++) codes[t] = ids[first + j * SkinTokensVae.TokensPerBone + t] - SkeletonVocab;
            _vae.DecodeBone(codes, skin.View, j, J);
        }
        await _acc.WaitAsync();
        Mark($"skin decode ({J} bones x {n} points)");
        return new Result(ids.ToArray(), skeleton, await skin.View.CopyToHostAsync(), n);
    }

    public void Dispose() { _encoder.Dispose(); _decoder.Dispose(); _vae.Dispose(); }
}
