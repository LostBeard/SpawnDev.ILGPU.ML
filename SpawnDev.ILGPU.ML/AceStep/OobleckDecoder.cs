using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.Trellis2;
using FV = ILGPU.Runtime.ArrayView1D<float, ILGPU.Stride1D.Dense>;

namespace SpawnDev.ILGPU.ML.AceStep;

/// <summary>
/// The Oobleck audio VAE's decoder (diffusers' AutoencoderOobleck; ACE-Step 1.5's <c>ace_1.5_vae</c>): latent [64, T] at 25 frames per second to stereo
/// audio [2, 1920 T] at 48 kHz. Layout read from the file (<c>decoder.layers.*</c>):
/// <list type="bullet">
/// <item>layers.0: Conv1d k7 64 -> 2048 (pad 3);</item>
/// <item>layers.1 .. 5: decoder blocks, each Snake, ConvTranspose1d (kernel 2 s, stride s, padding ceil(s / 2); s = 10, 6, 4, 4, 2), then three residual
/// units (<see cref="OobleckLayers"/>);</item>
/// <item>layers.6: Snake; layers.7: Conv1d k7 128 -> 2, no bias.</item>
/// </list>
/// </summary>
public sealed class OobleckDecoder : IDisposable
{
    /// <summary>Audio samples per latent frame (10 x 6 x 4 x 4 x 2).</summary>
    public int HopLength { get; }
    /// <summary>Latent channels the decoder takes.</summary>
    public int LatentChannels { get; }
    /// <summary>Audio channels it writes (2: stereo).</summary>
    public int AudioChannels { get; }

    private readonly Accelerator _acc;
    private readonly OobleckLayers _l;
    private readonly (int Stride, int InC, int OutC)[] _blocks;

    private OobleckDecoder(Accelerator acc, (int, int, int)[] blocks, int latentChannels, int audioChannels)
    {
        _acc = acc; _blocks = blocks; LatentChannels = latentChannels; AudioChannels = audioChannels;
        HopLength = blocks.Aggregate(1, (h, b) => h * b.Item1);
        _l = new OobleckLayers(acc);
    }

    /// <summary>Load the decoder from a safetensors file holding <paramref name="prefix"/>layers.* (ComfyUI's ace_1.5_vae: "decoder.").</summary>
    public static async Task<OobleckDecoder> LoadAsync(Accelerator accelerator, IModelSource source, string repo, string file, string prefix = "decoder.", CancellationToken ct = default)
    {
        var w = await OobleckLayers.ReadAsync(accelerator, source, repo, file, prefix, ct);
        try
        {
            string L(string rest) => prefix + "layers." + rest;
            var blocks = new List<(int, int, int)>();
            for (int i = 1; w.Has(L($"{i}.layers.1.weight_v")); i++)
            {
                var s = w.Shape(L($"{i}.layers.1.weight_v"));                         // ConvTranspose1d [inC, outC, 2 stride]
                blocks.Add((s[2] / 2, s[0], s[1]));
            }
            if (blocks.Count == 0) throw new InvalidDataException($"'{file}': no Oobleck decoder blocks under {prefix}layers.");
            int last = blocks.Count + 2;
            var d = new OobleckDecoder(accelerator, blocks.ToArray(), w.Shape(L("0.weight_v"))[1], w.Shape(L($"{last}.weight_v"))[0]);
            try
            {
                await d._l.AddConvAsync(w, L("0"), "in");
                for (int i = 1; i <= blocks.Count; i++)
                {
                    await d._l.AddSnakeAsync(w, L($"{i}.layers.0"), $"b{i}.snake");
                    await d._l.AddConvAsync(w, L($"{i}.layers.1"), $"b{i}.up", transposed: true);
                    int ii = i;
                    await d._l.AddResidualUnitsAsync(w, r => L($"{ii}.layers.{r + 2}.layers."), $"b{i}");
                }
                await d._l.AddSnakeAsync(w, L($"{blocks.Count + 1}"), "out.snake");
                await d._l.AddConvAsync(w, L($"{last}"), "out");
                return d;
            }
            catch { d.Dispose(); throw; }
        }
        finally { await accelerator.SynchronizeAsync(); w.Dispose(); }
    }

    /// <summary>
    /// Decode <paramref name="latent"/> [<see cref="LatentChannels"/>, <paramref name="frames"/>] (channel-major, as ComfyUI's [1, 64, T] latent) into a new
    /// buffer [<see cref="AudioChannels"/>, <see cref="HopLength"/> x <paramref name="frames"/>] the caller owns.
    /// </summary>
    public async Task<MemoryBuffer1D<float, Stride1D.Dense>> DecodeAsync(FV latent, int frames, CancellationToken ct = default)
    {
        long maxElems = (long)_l["in"].OutC * frames;
        int len = frames;
        foreach (var (st, _, outC) in _blocks) { len *= st; maxElems = Math.Max(maxElems, (long)outC * len); }
        var a = _acc.Allocate1D<float>(maxElems); var b = _acc.Allocate1D<float>(maxElems); var c = _acc.Allocate1D<float>(maxElems);
        var output = _acc.Allocate1D<float>((long)AudioChannels * frames * HopLength);
        try
        {
            // channel-last [L, C] inside; the latent comes channel-major [64, T] and the audio leaves channel-major [2, L]
            int L = frames, ch;
            _l.Transpose(latent.SubView(0, (long)LatentChannels * frames), c.View, LatentChannels, frames);
            _l.Conv1d("in", c.View, a.View, L, 3); ch = _l["in"].OutC;                          // a: [T, 2048]
            for (int i = 0; i < _blocks.Length; i++)
            {
                // browser: a command buffer per block (DeviceSync.Submit; a no-op on desktop). A whole forward in ONE command buffer can outrun a slow
                // GPU's 2 s watchdog (2026-11-23, Lore: FLUX.2 lost the device on an RX 590 in Chrome until its encoder and transformer submitted per block).
                _acc.Submit();
                var (s, inC, outC) = _blocks[i];
                _l.Snake($"b{i + 1}.snake", a.View, b.View, inC, L);
                _l.Upsample($"b{i + 1}.up", b.View, a.View, L, s);
                L *= s; ch = outC;
                _l.ResidualUnits($"b{i + 1}", a.View, b.View, c.View, ch, L);
            }
            _l.Snake("out.snake", a.View, b.View, ch, L);
            _l.Conv1d("out", b.View, c.View, L, 3);                                               // c: [L, 2]
            _l.Transpose(c.View, output.View, L, AudioChannels);
            await _acc.SynchronizeAsync();
            return output;
        }
        catch { await _acc.SynchronizeAsync(); output.Dispose(); throw; }
        finally { await _acc.SynchronizeAsync(); a.Dispose(); b.Dispose(); c.Dispose(); }
    }

    public void Dispose() => _l.Dispose();
}
