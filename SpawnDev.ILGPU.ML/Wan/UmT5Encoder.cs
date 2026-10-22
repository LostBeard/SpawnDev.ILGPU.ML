using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.SafeTensors;
using SpawnDev.ILGPU.ML.Trellis2;

namespace SpawnDev.ILGPU.ML.Wan;

using FV = ArrayView1D<float, Stride1D.Dense>;
using IV = ArrayView1D<int, Stride1D.Dense>;

/// <summary>
/// The umT5-XXL encoder of Wan 2.2 (5.7B parameters, 24 layers, width 4096, 64 heads of 64, feed-forward 10240): token embedding, then per layer
/// x += o(softmax(q k^T + relative_bias) v) after an RMS norm and x += fc2(fc1(h) * gelu_tanh(gate(h))) after a second one (T5: no attention scaling, a bias table of
/// 32 buckets x 64 heads per layer), then a final RMS norm. Batch 1, only the REAL tokens (the reference pads to 512 and masks the padding out of the keys, which changes no
/// row of the real tokens). Weights: bf16 export of tools/wan-harness/reference/export_t5_bf16.py, activations fp32.
/// Built for a tight device: the 2.1 GB embedding table is never loaded (the rows of the prompt are read from the file), and the layers STREAM: one layer's 0.39 GB of weights is
/// uploaded, used and freed before the next, so the whole 11.4 GB encoder runs in well under 1 GB of VRAM at about the time it takes to read the file.
/// </summary>
public sealed class UmT5Encoder : IDisposable
{
    private readonly Accelerator _acc;
    private readonly string _path;
    private readonly SafeTensorsFileReader _file;
    private readonly DiTKernels _k;
    private readonly WanKernels _wk;
    public const int Dim = 4096, Heads = 64, HeadDim = 64, FfnDim = 10240, DefaultLayers = 24;
    private readonly int _layers;

    /// <summary>Test hook: (stage name, activation view, element count), valid only during the call; names "embedding", "layer_N", "final".</summary>
    public Action<string, FV, int>? Tap { get; set; }

    public UmT5Encoder(Accelerator accelerator, string safetensorsPath)
    {
        _acc = accelerator; _path = safetensorsPath;
        _file = SafeTensorsFileReader.Open(safetensorsPath);
        _k = new DiTKernels(accelerator); _wk = new WanKernels(accelerator);
        _layers = int.Parse(_file.Metadata.GetValueOrDefault("num_layers", "24"));
    }

    /// <summary>The embedding rows of <paramref name="ids"/> as floats [n, 4096], read straight from the file.</summary>
    private float[] Embed(int[] ids)
    {
        var info = _file.Get("token_embedding.weight");
        var x = new float[(long)ids.Length * Dim];
        var row = new byte[Dim * 2];
        for (int i = 0; i < ids.Length; i++)
        {
            _file.ReadBytes(info, (long)ids[i] * Dim * 2, row);
            for (int c = 0; c < Dim; c++) x[(long)i * Dim + c] = BitConverter.Int32BitsToSingle((row[2 * c] | (row[2 * c + 1] << 8)) << 16);     // bf16 -> float
        }
        return x;
    }

    /// <summary>Encode the token ids to the text context [n, 4096] (float32) that the Wan DiT's text embedding consumes.</summary>
    public async Task<float[]> EncodeAsync(int[] ids)
    {
        int L = ids.Length;
        if (L == 0) throw new ArgumentException("no tokens");
        using var xb = _acc.Allocate1D(Embed(ids));
        Tap?.Invoke("embedding", xb.View, L * Dim);
        using var hn = _acc.Allocate1D<float>((long)L * Dim);
        using var qkv = _acc.Allocate1D<float>((long)L * 3 * Dim);
        using var att = _acc.Allocate1D<float>((long)L * Dim);
        using var proj = _acc.Allocate1D<float>((long)L * Dim);
        using var gate = _acc.Allocate1D<float>((long)L * FfnDim);
        using var fc1 = _acc.Allocate1D<float>((long)L * FfnDim);
        using var scores = _acc.Allocate1D<float>((long)Heads * L * L);
        using var bucket = _acc.Allocate1D(WanKernels.T5RelativeBuckets(L));
        for (int b = 0; b < _layers; b++)
        {
            string p = $"blocks.{b}.";
            using var w = GpuWeights.LoadFile(_acc, _path, name => name.StartsWith(p, StringComparison.Ordinal));          // this layer only
            // self-attention
            hn.View.CopyFrom(xb.View);
            _wk.RowRmsNorm(hn.View, w.BF16(p + "norm1.weight"), L, Dim, 0, Dim, 1e-6f);
            _k.Linear(hn.View, w.BF16(p + "attn.qkv.weight"), null, qkv.View, L, Dim, 3 * Dim);
            _wk.T5Attention(qkv.View, w.BF16(p + "pos_embedding.embedding.weight"), bucket.View, scores.View, att.View, L, Heads, HeadDim);
            _k.Linear(att.View, w.BF16(p + "attn.o.weight"), null, proj.View, L, Dim, Dim);
            _k.GateResidual(xb.View, proj.View, default, -1, L * Dim, Dim);
            // gated feed-forward
            hn.View.CopyFrom(xb.View);
            _wk.RowRmsNorm(hn.View, w.BF16(p + "norm2.weight"), L, Dim, 0, Dim, 1e-6f);
            _k.Linear(hn.View, w.BF16(p + "ffn.gate.weight"), null, gate.View, L, Dim, FfnDim, DiTKernels.Act.GeluTanh);
            _k.Linear(hn.View, w.BF16(p + "ffn.fc1.weight"), null, fc1.View, L, Dim, FfnDim);
            _wk.MulInPlace(fc1.View, gate.View, L * FfnDim);
            _k.Linear(fc1.View, w.BF16(p + "ffn.fc2.weight"), null, proj.View, L, FfnDim, Dim);
            _k.GateResidual(xb.View, proj.View, default, -1, L * Dim, Dim);
            _acc.Fence();                                                              // the layer's weights are freed when `w` goes out of scope
            if (Tap != null && (b == 0 || b == _layers / 2 || b == _layers - 1)) Tap($"layer_{b}", xb.View, L * Dim);
        }
        using (var nw = GpuWeights.LoadFile(_acc, _path, name => name == "norm.weight"))
            _wk.RowRmsNorm(xb.View, nw.BF16("norm.weight"), L, Dim, 0, Dim, 1e-6f);
        Tap?.Invoke("final", xb.View, L * Dim);
        return await xb.View.CopyToHostAsync();
    }

    public void Dispose() => _file.Dispose();
}
