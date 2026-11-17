using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.SkinTokens;
using SpawnDev.ILGPU.ML.Trellis2;
using System.Text.Json;
using FV = ILGPU.Runtime.ArrayView1D<float, ILGPU.Stride1D.Dense>;
using BV = ILGPU.Runtime.ArrayView1D<ILGPU.BFloat16, ILGPU.Stride1D.Dense>;

namespace SpawnDev.ILGPU.ML.AceStep;

/// <summary>
/// ACE-Step 1.5's audio codes to the DiT's 25 Hz hints (ComfyUI's acestep_v1.5_turbo: <c>tokenizer.quantizer.project_out</c> and <c>detokenizer.*</c>): each 5 Hz code
/// (0 .. 63999) is a finite scalar quantizer index over levels [8, 8, 8, 5, 5, 5] (mixed radix, the first level fastest), each digit scaled to (d - (L - 1) / 2) / ((L - 1) / 2)
/// (MEASURED against ComfyUI's detokenizer input: 5.9e-8), projected 6 -> 2048; the detokenizer then embeds (Linear 2048 -> 2048 + bias), repeats each code 5 times plus
/// its 5 learned special tokens, runs 2 encoder layers over each 5-token group alone (positions 0 .. 4, bidirectional), norms and projects to 64: hints [5 n, 64], time-major.
/// With codes, the hints replace the silence latent as the DiT's source latents (context = [hints | ones]).
/// </summary>
public sealed class AceStepCodeDecoder : IDisposable
{
    public const int Dim = 2048, Pool = 5, LatentChannels = 64;
    /// <summary>The quantizer's levels; the code count is their product, 64000.</summary>
    public static readonly int[] Levels = { 8, 8, 8, 5, 5, 5 };
    private const float Eps = 1e-6f;

    private readonly Accelerator _acc;
    private readonly GpuWeights _w;
    private readonly AceStepEncoderLayers _layers;
    private readonly SkinTokensKernels _u;
    private readonly AceStepKernels _a;
    private readonly int _maxCodes, _layerCount;
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _z, _p, _e, _h, _n, _cos, _sin;

    private AceStepCodeDecoder(Accelerator acc, GpuWeights w, int maxCodes, int layerCount)
    {
        _acc = acc; _w = w; _maxCodes = maxCodes; _layerCount = layerCount;
        _layers = new AceStepEncoderLayers(acc, w, maxCodes * Pool); _u = new SkinTokensKernels(acc); _a = new AceStepKernels(acc);
        _z = acc.Allocate1D<float>((long)maxCodes * Levels.Length); _p = acc.Allocate1D<float>((long)maxCodes * Dim); _e = acc.Allocate1D<float>((long)maxCodes * Dim);
        _h = acc.Allocate1D<float>((long)maxCodes * Pool * Dim); _n = acc.Allocate1D<float>((long)maxCodes * Pool * Dim);
        var (c, s) = AceStepEncoderLayers.RopeTables(Enumerable.Range(0, maxCodes * Pool).Select(r => r % Pool).ToArray());   // every group restarts at position 0
        _cos = acc.Allocate1D(c); _sin = acc.Allocate1D(s);
    }

    /// <summary>Load from ComfyUI's acestep_v1.5_turbo file; buffers sized for up to <paramref name="maxCodes"/> codes (5 per second of audio).</summary>
    public static async Task<AceStepCodeDecoder> LoadAsync(Accelerator accelerator, IModelSource source, string repo, string file, int maxCodes, CancellationToken ct = default)
    {
        GpuWeights w;
        await using (var stream = await source.OpenAsync(repo, file, ct))
        {
            using var empty = JsonDocument.Parse("{}");
            w = await GpuWeights.LoadAsync(accelerator, stream, empty.RootElement, file,
                n => n.StartsWith("detokenizer.", StringComparison.Ordinal) || n.StartsWith("tokenizer.quantizer.project_out.", StringComparison.Ordinal), ct);
        }
        try
        {
            if (w.Shape("tokenizer.quantizer.project_out.weight")[1] != Levels.Length) throw new InvalidDataException($"'{file}': project_out is not {Levels.Length} -> {Dim}");
            int layers = 0; while (w.Has($"detokenizer.layers.{layers}.input_layernorm.weight")) layers++;
            return new AceStepCodeDecoder(accelerator, w, maxCodes, layers);
        }
        catch { w.Dispose(); throw; }
    }

    /// <summary>The quantizer's 6 values of each code (host side): digit k of the mixed-radix index, scaled to [-1, 1].</summary>
    public static float[] FsqValues(IReadOnlyList<int> codes)
    {
        var z = new float[codes.Count * Levels.Length];
        for (int i = 0; i < codes.Count; i++)
        {
            int rest = codes[i];
            if (rest < 0 || rest >= 64000) throw new ArgumentOutOfRangeException(nameof(codes), $"code {rest} is not in 0 .. 63999");
            for (int k = 0; k < Levels.Length; k++)
            {
                int digit = rest % Levels[k]; rest /= Levels[k];
                float half = (Levels[k] - 1) / 2f;
                z[i * Levels.Length + k] = (digit - half) / half;
            }
        }
        return z;
    }

    private BV B(string name) => _w.BF16(name);

    /// <summary>The 25 Hz hints [5 codes, 64] (time-major) for <paramref name="codes"/>, into <paramref name="output"/>.</summary>
    public void Decode(IReadOnlyList<int> codes, FV output)
    {
        int n = codes.Count, L = n * Pool;
        if (n > _maxCodes) throw new ArgumentException($"{n} codes: the decoder was sized for {_maxCodes}");
        var k = _layers.Kernels;
        _z.View.SubView(0, (long)n * Levels.Length).CopyFromCPU(FsqValues(codes));
        k.Linear(_z.View, B("tokenizer.quantizer.project_out.weight"), B("tokenizer.quantizer.project_out.bias"), _p.View, n, Levels.Length, Dim);
        k.Linear(_p.View, B("detokenizer.embed_tokens.weight"), B("detokenizer.embed_tokens.bias"), _e.View, n, Dim, Dim);
        _a.RepeatAdd(_e.View, B("detokenizer.special_tokens"), _h.View, n, Pool, Dim);
        for (int l = 0; l < _layerCount; l++) _layers.Run($"detokenizer.layers.{l}.", _h.View, L, _cos.View, _sin.View, -1, Pool);
        _u.RmsNorm(_h.View, _n.View, B("detokenizer.norm.weight"), L, Dim, Eps);
        k.Linear(_n.View, B("detokenizer.proj_out.weight"), B("detokenizer.proj_out.bias"), output, L, Dim, LatentChannels);
    }

    public void Dispose()
    {
        _layers.Dispose();
        foreach (var b in new[] { _z, _p, _e, _h, _n, _cos, _sin }) b.Dispose();
        _w.Dispose();
    }
}
