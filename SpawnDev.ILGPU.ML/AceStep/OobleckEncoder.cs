using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Hub;
using FV = ILGPU.Runtime.ArrayView1D<float, ILGPU.Stride1D.Dense>;

namespace SpawnDev.ILGPU.ML.AceStep;

/// <summary>
/// The Oobleck audio VAE's encoder (ACE-Step 1.5's <c>ace_1.5_vae</c>, <c>encoder.layers.*</c>): stereo audio [2, 1920 T] at 48 kHz to the posterior over the latent
/// [64, T]. Layout read from the file:
/// <list type="bullet">
/// <item>layers.0: Conv1d k7 2 -> 128 (pad 3);</item>
/// <item>layers.1 .. 5: encoder blocks, each three residual units (<see cref="OobleckLayers"/>), Snake, Conv1d (kernel 2 s, stride s, padding ceil(s / 2);
/// s = 2, 4, 4, 6, 10);</item>
/// <item>layers.6: Snake; layers.7: Conv1d k3 2048 -> 128 (pad 1) = [mean 64 | scale 64].</item>
/// </list>
/// The latent is the posterior's mean (the first 64 channels); its standard deviation is softplus(scale) + 1e-4.
/// </summary>
public sealed class OobleckEncoder : IDisposable
{
    /// <summary>Audio samples per latent frame (2 x 4 x 4 x 6 x 10).</summary>
    public int HopLength { get; }
    /// <summary>Latent channels of the mean.</summary>
    public int LatentChannels { get; }
    /// <summary>Audio channels it takes (2: stereo).</summary>
    public int AudioChannels { get; }

    private readonly Accelerator _acc;
    private readonly OobleckLayers _l;
    private readonly (int Stride, int InC, int OutC)[] _blocks;

    private OobleckEncoder(Accelerator acc, (int, int, int)[] blocks, int audioChannels, int latentChannels)
    {
        _acc = acc; _blocks = blocks; AudioChannels = audioChannels; LatentChannels = latentChannels;
        HopLength = blocks.Aggregate(1, (h, b) => h * b.Item1);
        _l = new OobleckLayers(acc);
    }

    /// <summary>Load the encoder from a safetensors file holding <paramref name="prefix"/>layers.* (ComfyUI's ace_1.5_vae: "encoder.").</summary>
    public static async Task<OobleckEncoder> LoadAsync(Accelerator accelerator, IModelSource source, string repo, string file, string prefix = "encoder.", CancellationToken ct = default)
    {
        var w = await OobleckLayers.ReadAsync(accelerator, source, repo, file, prefix, ct);
        try
        {
            string L(string rest) => prefix + "layers." + rest;
            var blocks = new List<(int, int, int)>();
            for (int i = 1; w.Has(L($"{i}.layers.4.weight_v")); i++)
            {
                var s = w.Shape(L($"{i}.layers.4.weight_v"));                         // Conv1d [outC, inC, 2 stride]
                blocks.Add((s[2] / 2, s[1], s[0]));
            }
            if (blocks.Count == 0) throw new InvalidDataException($"'{file}': no Oobleck encoder blocks under {prefix}layers.");
            int last = blocks.Count + 2;
            var e = new OobleckEncoder(accelerator, blocks.ToArray(), w.Shape(L("0.weight_v"))[1], w.Shape(L($"{last}.weight_v"))[0] / 2);
            try
            {
                await e._l.AddConvAsync(w, L("0"), "in");
                for (int i = 1; i <= blocks.Count; i++)
                {
                    int ii = i;
                    await e._l.AddResidualUnitsAsync(w, r => L($"{ii}.layers.{r}.layers."), $"b{i}");
                    await e._l.AddSnakeAsync(w, L($"{i}.layers.3"), $"b{i}.snake");
                    await e._l.AddConvAsync(w, L($"{i}.layers.4"), $"b{i}.down");
                }
                await e._l.AddSnakeAsync(w, L($"{blocks.Count + 1}"), "out.snake");
                await e._l.AddConvAsync(w, L($"{last}"), "out");
                return e;
            }
            catch { e.Dispose(); throw; }
        }
        finally { await accelerator.SynchronizeAsync(); w.Dispose(); }
    }

    /// <summary>
    /// Encode <paramref name="audio"/> [<see cref="AudioChannels"/>, <see cref="HopLength"/> x <paramref name="frames"/>] (channel-major) into a new buffer
    /// [2 <see cref="LatentChannels"/>, <paramref name="frames"/>] the caller owns: the posterior's mean (rows 0 .. 63) and scale (rows 64 .. 127).
    /// </summary>
    public async Task<MemoryBuffer1D<float, Stride1D.Dense>> EncodeAsync(FV audio, int frames, CancellationToken ct = default)
    {
        int L = frames * HopLength;
        long maxElems = (long)_l["in"].OutC * L;
        foreach (var (st, inC, outC) in _blocks) { maxElems = Math.Max(maxElems, (long)inC * L); L /= st; maxElems = Math.Max(maxElems, (long)outC * L); }
        var a = _acc.Allocate1D<float>(maxElems); var b = _acc.Allocate1D<float>(maxElems); var c = _acc.Allocate1D<float>(maxElems);
        var output = _acc.Allocate1D<float>((long)2 * LatentChannels * frames);
        try
        {
            L = frames * HopLength;
            // the first conv (2 channels: too narrow for the GEMM) channel-major, then channel-last [L, C] inside
            _l.Conv1dChannelMajor("in", audio, c.View, L, 3);
            int ch = _l["in"].OutC;
            _l.Transpose(c.View, a.View, ch, L);
            for (int i = 0; i < _blocks.Length; i++)
            {
                var (s, inC, outC) = _blocks[i];
                _l.ResidualUnits($"b{i + 1}", a.View, b.View, c.View, inC, L);
                _l.Snake($"b{i + 1}.snake", a.View, b.View, inC, L);
                _l.Conv1d($"b{i + 1}.down", b.View, a.View, L, (s + 1) / 2, 1, s);
                L /= s; ch = outC;
            }
            _l.Snake("out.snake", a.View, b.View, ch, L);
            _l.Conv1d("out", b.View, c.View, L, 1);                                               // c: [T, 128]
            _l.Transpose(c.View, output.View, L, 2 * LatentChannels);
            await _acc.SynchronizeAsync();
            return output;
        }
        catch { await _acc.SynchronizeAsync(); output.Dispose(); throw; }
        finally { await _acc.SynchronizeAsync(); a.Dispose(); b.Dispose(); c.Dispose(); }
    }

    public void Dispose() => _l.Dispose();
}
