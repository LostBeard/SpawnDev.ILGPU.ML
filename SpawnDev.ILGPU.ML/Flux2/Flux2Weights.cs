using System.Text.Json;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.SkinTokens;
using SpawnDev.ILGPU.ML.Trellis2;

namespace SpawnDev.ILGPU.ML.Flux2;

using FV = ArrayView1D<float, Stride1D.Dense>;
using IV = ArrayView1D<int, Stride1D.Dense>;
using BV = ArrayView1D<BFloat16, Stride1D.Dense>;

/// <summary>
/// The weights of one FLUX.2 stage, as the encoder and the transformer read them: bf16 tensors straight from the device, and matrices stored as 4-bit blocks
/// (quantize_flux2.py: 5 bits per weight, 2.5 GB for the 3.9B transformer instead of 7.75) dequantised one matrix at a time into a bf16 scratch just before its
/// GEMM, so the normal tensor-core / tiled GEMM runs on them unchanged. <see cref="Matrix"/> returns a view that is only valid until the next call (it may be the shared scratch).
/// </summary>
public sealed class Flux2Weights : IDisposable
{
    private readonly GpuWeights _w;
    private readonly Accelerator _acc;
    private Q8Linear? _q8;
    private readonly Flux2Kernels _f;
    private readonly SkinTokensKernels _u;
    private MemoryBuffer1D<uint, Stride1D.Dense>? _scratch;
    private readonly int _wordsPerBlock;                        // 5 = 4-bit blocks, 9 = 8-bit blocks (the file's metadata "quant" says which)

    /// <summary>4 or 8: the bits per quantised weight of this stage (0 when it holds no quantised matrix).</summary>
    public int QuantBits { get; }

    private readonly bool _owns;

    /// <summary>
    /// Wrap a loaded stage. Disposing this disposes <paramref name="weights"/> too, unless <paramref name="ownsWeights"/> is false (a model that keeps the <see cref="GpuWeights"/> itself and only
    /// borrows the block dequantisation: the Wan 2.2 DiT reads bf16, 8-bit and 4-bit exports through it).
    /// </summary>
    public Flux2Weights(Accelerator accelerator, GpuWeights weights, bool ownsWeights = true)
    {
        _owns = ownsWeights; _acc = accelerator;
        _w = weights; _f = new Flux2Kernels(accelerator); _u = new SkinTokensKernels(accelerator);
        weights.Metadata.TryGetValue("quant", out var quant);
        QuantBits = quant != null && quant.StartsWith("q8", StringComparison.Ordinal) ? 8 : 4;
        _wordsPerBlock = QuantBits == 8 ? 9 : 5;
        long most = 0;
        foreach (var name in weights.Names)
            if (weights.IsU32(name)) { var s = weights.Shape(name); most = Math.Max(most, (long)s[0] * (s[1] / _wordsPerBlock * 32)); }
        if (most > 0) _scratch = accelerator.Allocate1D<uint>(most / 2);
    }

    /// <summary>Stream one stage's safetensors file from a model source (browser OPFS / hub, or HttpClient + a file store on desktop) onto the device.</summary>
    public static async Task<Flux2Weights> LoadAsync(Accelerator accelerator, IModelSource source, string repo, string file, CancellationToken cancellationToken = default)
    {
        await using var stream = await source.OpenAsync(repo, file, cancellationToken);
        using var empty = JsonDocument.Parse("{}");
        return new Flux2Weights(accelerator, await GpuWeights.LoadAsync(accelerator, stream, empty.RootElement, $"{repo}/{file}", cancellationToken: cancellationToken));
    }

    public bool IsQuantized(string name) => _w.IsU32(name);
    /// <summary>Device bytes held (weights plus the dequantisation scratch).</summary>
    public long Bytes => _w.Bytes + (_scratch?.Length ?? 0) * 4;

    /// <summary>A weight matrix [N, K] as bf16: the stored tensor itself, or its dequantisation in the shared scratch (valid until the next call).</summary>
    public BV Matrix(string name)
    {
        if (!_w.IsU32(name)) return _w.BF16(name);
        var s = _w.Shape(name);
        int n = s[0], k = s[1] / _wordsPerBlock * 32;
        if (QuantBits == 8) _f.DequantQ8(_w.U32(name), _scratch!.View, n, k); else _f.DequantQ4(_w.U32(name), _scratch!.View, n, k);
        return _scratch.View.SubView(0, (long)n * k / 2).Cast<uint, BFloat16>();
    }

    /// <summary>
    /// W8A8 (<see cref="Q8Linear"/>): 8-bit matrices multiply straight from their blocks with activations quantised to 8 bits per block of 32, on packed int8 dot products,
    /// instead of a bf16 copy of the matrix and a float GEMM. True turns it on for every 8-bit matrix (ML_W8A8=1): faster on WebGPU with native packed int8 dot products (Chrome:
    /// the packed_4x8_integer_dot_product feature; a Wan layer up to 3.3x, the Wan Turbo transformer 1.27x per step), at a measured cost in quality (<see cref="AutoW8A8"/>).
    /// Null (the default) = only for a model that opts in on WebGPU with native packed dots; false (ML_W8A8=0) = never.
    /// </summary>
    public static bool? EnableW8A8 { get; set; } = Environment.GetEnvironmentVariable("ML_W8A8") switch { "1" => true, "0" => false, _ => null };

    /// <summary>Matrix names containing any of these stay on the float GEMM (ML_W8A8_SKIP, comma separated): for measuring which layers are sensitive.</summary>
    public static string[] SkipW8A8 { get; set; } = (Environment.GetEnvironmentVariable("ML_W8A8_SKIP") ?? "").Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);

    /// <summary>Whether <see cref="Linear"/> runs <paramref name="name"/> as W8A8.</summary>
    public bool UsesW8A8(string name) => QuantBits == 8 && _w.IsU32(name) && !SkipW8A8.Any(s => name.Contains(s, StringComparison.Ordinal))
        && (EnableW8A8 ?? (AutoW8A8 && _acc.AcceleratorType == AcceleratorType.WebGPU && _acc.Capabilities.PackedInt8DotProduct));

    /// <summary>
    /// Whether this model takes W8A8 automatically (<see cref="EnableW8A8"/> null): for a model whose W8A8 clips cost no more than its 8-bit weights do (PSNR against the bf16
    /// output within 1.5 dB of the 8-bit float output's). None does yet. MEASURED: Wan 2.2 Turbo 832x480 x33, 27.4 dB against the bf16 video where the 8-bit weights alone give
    /// 30.8 dB (Chrome: the transformer 5.0 -> 3.9 s per step); LTX-Video 704x480 x97, 22.8 vs 32.1 dB (its error spread over every layer type); the T5 encoders unstable
    /// (0.14 on one prompt). Activation blocks of 32 are too coarse for these models' activations: finer or asymmetric activation scales are the next step.
    /// </summary>
    public bool AutoW8A8 { get; set; }

    /// <summary>nn.Linear with the matrix <paramref name="name"/> [N, K]: W8A8 when <see cref="UsesW8A8"/>, else <paramref name="kernels"/>' GEMM on <see cref="Matrix"/>.</summary>
    /// <param name="allowW8A8">false keeps this layer on the float GEMM (mixed precision: a layer whose input carries outliers that 8-bit activations cannot hold)</param>
    public void Linear(DiTKernels kernels, FV x, string name, BV? bias, FV o, int M, int K, int N, DiTKernels.Act act = DiTKernels.Act.None, bool allowW8A8 = true)
    {
        if (allowW8A8 && UsesW8A8(name)) { (_q8 ??= new Q8Linear(_acc)).Linear(x, _w.U32(name), bias, o, M, K, N, act); return; }
        kernels.Linear(x, Matrix(name), bias, o, M, K, N, act);
    }

    /// <summary>A small bf16 tensor (norm weight, bias).</summary>
    public BV Vector(string name) => _w.BF16(name);

    /// <summary>o[t, :] = row ids[t] of the embedding table <paramref name="name"/> ([vocab, C]), fp32.</summary>
    public void Embed(IV ids, string name, FV o, int count, int C)
    {
        if (_w.IsU32(name)) { if (QuantBits == 8) _f.DequantRows8(_w.U32(name), ids, o, count, C); else _f.DequantRows(_w.U32(name), ids, o, count, C); }
        else _u.Embed(ids, _w.BF16(name), o, count, C);
    }

    public void Dispose()
    {
        _scratch?.Dispose(); _scratch = null;
        _q8?.Dispose(); _q8 = null;
        if (_owns) _w.Dispose();
    }
}
