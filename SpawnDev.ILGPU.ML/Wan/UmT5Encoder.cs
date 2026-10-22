using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Hub;
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
/// uploaded, used and freed before the next, so the whole 11.4 GB encoder runs in well under 1 GB of VRAM at about the time it takes to read the file. Several prompts (the prompt and
/// the negative prompt) go through ONE pass over the layers, so the file is read once. The file comes from an <see cref="IModelSource"/>: a local folder, or the browser OPFS cache /
/// HTTP ranges (the stream must seek).
/// </summary>
public sealed class UmT5Encoder : IDisposable
{
    private readonly Accelerator _acc;
    private readonly IModelSource _source;
    private readonly string _repo, _file;
    private readonly Stream _embeddings;                         // kept open: the rows of the prompt are read from it
    private readonly SafeTensorInfo _embeddingInfo;
    private readonly DiTKernels _k;
    private readonly WanKernels _wk;
    public const int Dim = 4096, Heads = 64, HeadDim = 64, FfnDim = 10240, DefaultLayers = 24;
    private readonly int _layers;

    /// <summary>Test hook: (stage name, activation view, element count), valid only during the call; names "embedding", "layer_N", "final" (the first sequence of a call).</summary>
    public Action<string, FV, int>? Tap { get; set; }

    private UmT5Encoder(Accelerator accelerator, IModelSource source, string repo, string file, Stream embeddings, SafeTensorInfo info, int layers)
    {
        _acc = accelerator; _source = source; _repo = repo; _file = file; _embeddings = embeddings; _embeddingInfo = info; _layers = layers;
        _k = new DiTKernels(accelerator); _wk = new WanKernels(accelerator);
    }

    /// <summary>Open the encoder file from a model source: reads the header and keeps a stream on the embedding table.</summary>
    public static async Task<UmT5Encoder> CreateAsync(Accelerator accelerator, IModelSource source, string repo, string file, CancellationToken cancellationToken = default)
    {
        var stream = await source.OpenAsync(repo, file, cancellationToken);
        try
        {
            if (!stream.CanSeek) throw new NotSupportedException("the umT5 file must come from a seekable stream (the prompt embedding rows and each layer are read by offset)");
            if (stream is FetchRangeStream ranges) { ranges.ReadAhead = 1 << 16; ranges.Depth = 1; }       // this stream only reads 8 KB rows scattered over a 2 GB table: small ranges, no read-ahead
            var lenBytes = new byte[8];
            await stream.ReadExactlyAsync(lenBytes, cancellationToken);
            long headerSize = BitConverter.ToInt64(lenBytes);
            if (headerSize <= 0 || headerSize > 100_000_000) throw new InvalidDataException($"'{file}': invalid SafeTensors header size {headerSize}");
            var headerBytes = new byte[headerSize];
            await stream.ReadExactlyAsync(headerBytes, cancellationToken);
            long dataStart = 8 + headerSize;
            var (tensors, metadata) = SafeTensorsFileReader.ParseHeader(file, headerBytes, dataStart, stream.Length - dataStart);
            var info = tensors.FirstOrDefault(t => t.Name == "token_embedding.weight") ?? throw new InvalidDataException($"'{file}' has no token_embedding.weight");
            return new UmT5Encoder(accelerator, source, repo, file, stream, info, int.Parse(metadata.GetValueOrDefault("num_layers", "24")));
        }
        catch { stream.Dispose(); throw; }
    }

    /// <summary>Open the encoder file from a path on disk.</summary>
    public static Task<UmT5Encoder> CreateAsync(Accelerator accelerator, string safetensorsPath, CancellationToken cancellationToken = default)
        => CreateAsync(accelerator, new LocalFolderModelSource(""), "", safetensorsPath, cancellationToken);

    /// <summary>The embedding rows of <paramref name="ids"/> as floats [n, 4096], read straight from the file.</summary>
    private async Task<float[]> EmbedAsync(int[] ids, CancellationToken ct)
    {
        var x = new float[(long)ids.Length * Dim];
        var row = new byte[Dim * 2];
        var first = new Dictionary<int, int>();                      // token id -> the first position that holds it: every distinct row is read once, in file order
        for (int i = 0; i < ids.Length; i++) first.TryAdd(ids[i], i);
        foreach (var (id, i) in first.OrderBy(kv => kv.Key))
        {
            _embeddings.Seek(_embeddingInfo.DataOffset + (long)id * Dim * 2, SeekOrigin.Begin);
            await _embeddings.ReadExactlyAsync(row, ct);
            for (int c = 0; c < Dim; c++) x[(long)i * Dim + c] = BitConverter.Int32BitsToSingle((row[2 * c] | (row[2 * c + 1] << 8)) << 16);     // bf16 -> float
        }
        for (int i = 0; i < ids.Length; i++) { int f = first[ids[i]]; if (f != i) Array.Copy(x, (long)f * Dim, x, (long)i * Dim, Dim); }
        return x;
    }

    /// <summary>Load the tensors of one layer (or the final norm) from the file, in file order.</summary>
    private async Task<GpuWeights> LoadLayerAsync(Func<string, bool> filter, CancellationToken ct)
    {
        await using var stream = await _source.OpenAsync(_repo, _file, ct);
        using var empty = System.Text.Json.JsonDocument.Parse("{}");
        return await GpuWeights.LoadAsync(_acc, stream, empty.RootElement, _file, filter, ct);
    }

    /// <summary>Encode the token ids to the text context [n, 4096] (float32) that the Wan DiT text embedding consumes.</summary>
    public async Task<float[]> EncodeAsync(int[] ids, CancellationToken ct = default) => (await EncodeManyAsync(new[] { ids }, ct))[0];

    /// <summary>
    /// Encode several prompts in one pass over the layers (each layer's weights are read and uploaded once, then applied to every sequence): the contexts [n_i, 4096], in order.
    /// </summary>
    public async Task<float[][]> EncodeManyAsync(IReadOnlyList<int[]> sequences, CancellationToken ct = default)
    {
        int count = sequences.Count, maxL = 0;
        foreach (var ids in sequences) { if (ids.Length == 0) throw new ArgumentException("no tokens"); maxL = Math.Max(maxL, ids.Length); }
        var xs = new MemoryBuffer1D<float, Stride1D.Dense>[count];
        var buckets = new MemoryBuffer1D<int, Stride1D.Dense>[count];
        try
        {
            for (int i = 0; i < count; i++)
            {
                xs[i] = _acc.Allocate1D(await EmbedAsync(sequences[i], ct));
                buckets[i] = _acc.Allocate1D(WanKernels.T5RelativeBuckets(sequences[i].Length));
            }
            Tap?.Invoke("embedding", xs[0].View, sequences[0].Length * Dim);
            using var hn = _acc.Allocate1D<float>((long)maxL * Dim);
            using var qkv = _acc.Allocate1D<float>((long)maxL * 3 * Dim);
            using var att = _acc.Allocate1D<float>((long)maxL * Dim);
            using var proj = _acc.Allocate1D<float>((long)maxL * Dim);
            using var gate = _acc.Allocate1D<float>((long)maxL * FfnDim);
            using var fc1 = _acc.Allocate1D<float>((long)maxL * FfnDim);
            using var scores = _acc.Allocate1D<float>((long)Heads * maxL * maxL);
            for (int b = 0; b < _layers; b++)
            {
                ct.ThrowIfCancellationRequested();
                string p = $"blocks.{b}.";
                using var w = await LoadLayerAsync(name => name.StartsWith(p, StringComparison.Ordinal), ct);          // this layer only
                for (int s = 0; s < count; s++)
                {
                    int L = sequences[s].Length; var xb = xs[s];
                    // self-attention
                    hn.View.CopyFrom(xb.View);
                    _wk.RowRmsNorm(hn.View, w.BF16(p + "norm1.weight"), L, Dim, 0, Dim, 1e-6f);
                    _k.Linear(hn.View, w.BF16(p + "attn.qkv.weight"), null, qkv.View, L, Dim, 3 * Dim);
                    _wk.T5Attention(qkv.View, w.BF16(p + "pos_embedding.embedding.weight"), buckets[s].View, scores.View, att.View, L, Heads, HeadDim);
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
                }
                await _acc.SynchronizeAsync();                                         // the layer's weights are freed when `w` goes out of scope
                if (Tap != null && (b == 0 || b == _layers / 2 || b == _layers - 1)) Tap($"layer_{b}", xs[0].View, sequences[0].Length * Dim);
            }
            using (var nw = await LoadLayerAsync(name => name == "norm.weight", ct))
            {
                for (int s = 0; s < count; s++) _wk.RowRmsNorm(xs[s].View, nw.BF16("norm.weight"), sequences[s].Length, Dim, 0, Dim, 1e-6f);
                await _acc.SynchronizeAsync();
            }
            Tap?.Invoke("final", xs[0].View, sequences[0].Length * Dim);
            var result = new float[count][];
            for (int s = 0; s < count; s++) result[s] = await xs[s].View.CopyToHostAsync();
            return result;
        }
        finally
        {
            foreach (var x in xs) x?.Dispose();
            foreach (var b in buckets) b?.Dispose();
        }
    }

    public void Dispose() => _embeddings.Dispose();
}
