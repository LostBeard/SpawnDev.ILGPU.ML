using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Flux2;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.SafeTensors;
using SpawnDev.ILGPU.ML.Trellis2;

namespace SpawnDev.ILGPU.ML.Wan;

using FV = ArrayView1D<float, Stride1D.Dense>;
using IV = ArrayView1D<int, Stride1D.Dense>;

/// <summary>
/// The umT5-XXL encoder of Wan 2.2 (5.7B parameters, 24 layers, width 4096, 64 heads of 64, feed-forward 10240): token embedding, then per layer
/// x += o(softmax(q k^T + relative_bias) v) after an RMS norm and x += fc2(fc1(h) * gelu_tanh(gate(h))) after a second one (T5: no attention scaling, a bias table of
/// 32 buckets x 64 heads per layer), then a final RMS norm. Batch 1 per prompt, only the REAL tokens (the reference pads to 512 and masks the padding out of the keys, which changes no
/// row of the real tokens). Weights: the bf16 export of tools/wan-harness/reference/export_t5_bf16.py (11.4 GB) or its 8-bit blocks (quantize_streaming.py, 6.4 GB, the FLUX.2 block
/// format; the matrices are dequantised one at a time into a bf16 scratch just before their GEMM), activations fp32.
/// Built for a tight device: the 2.1 GB embedding table is never loaded (the rows of the prompt are read from the file), and the layers STREAM: one layer's weights are
/// uploaded, used and freed before the next, so the whole encoder runs in well under 1 GB of VRAM at about the time it takes to read the file. Several prompts (the prompt and
/// the negative prompt) go through ONE pass over the layers (their rows are stacked for every matrix product), so the file is read once. The file comes from an
/// <see cref="IModelSource"/>: a local folder, or the browser OPFS cache / HTTP ranges (the stream must seek).
/// </summary>
public sealed class UmT5Encoder : IDisposable
{
    private readonly Accelerator _acc;
    private readonly IModelSource _source;
    private readonly string _repo, _file;
    private readonly Stream _embeddings;                         // kept open: the rows of the prompt are read from it
    private readonly SafeTensorInfo _embeddingInfo;
    private readonly bool _embeddingQuantized;
    private readonly DiTKernels _k;
    private readonly WanKernels _wk;
    public const int Dim = 4096, Heads = 64, HeadDim = 64, FfnDim = 10240, DefaultLayers = 24;
    private readonly int _layers;

    /// <summary>Test hook: (stage name, activation view, element count), valid only during the call; names "embedding", "layer_N", "final" (the first sequence of a call).</summary>
    public Action<string, FV, int>? Tap { get; set; }

    private UmT5Encoder(Accelerator accelerator, IModelSource source, string repo, string file, Stream embeddings, SafeTensorInfo info, bool quantized, int layers)
    {
        _acc = accelerator; _source = source; _repo = repo; _file = file; _embeddings = embeddings; _embeddingInfo = info; _embeddingQuantized = quantized; _layers = layers;
        _k = new DiTKernels(accelerator); _wk = new WanKernels(accelerator);
    }

    /// <summary>Open the encoder file from a model source: reads the header and keeps a stream on the embedding table.</summary>
    public static async Task<UmT5Encoder> CreateAsync(Accelerator accelerator, IModelSource source, string repo, string file, CancellationToken cancellationToken = default)
    {
        var stream = await source.OpenAsync(repo, file, cancellationToken);
        try
        {
            if (!stream.CanSeek) throw new NotSupportedException("the umT5 file must come from a seekable stream (the prompt embedding rows and each layer are read by offset)");
            if (stream is FetchRangeStream ranges) { ranges.ReadAhead = 1 << 16; ranges.Depth = 1; }       // this stream only reads small rows scattered over a 1-2 GB table: small ranges, no read-ahead
            var lenBytes = new byte[8];
            await stream.ReadExactlyAsync(lenBytes, cancellationToken);
            long headerSize = BitConverter.ToInt64(lenBytes);
            if (headerSize <= 0 || headerSize > 100_000_000) throw new InvalidDataException($"'{file}': invalid SafeTensors header size {headerSize}");
            var headerBytes = new byte[headerSize];
            await stream.ReadExactlyAsync(headerBytes, cancellationToken);
            long dataStart = 8 + headerSize;
            var (tensors, metadata) = SafeTensorsFileReader.ParseHeader(file, headerBytes, dataStart, stream.Length - dataStart);
            var info = tensors.FirstOrDefault(t => t.Name == "token_embedding.weight") ?? throw new InvalidDataException($"'{file}' has no token_embedding.weight");
            bool quantized = info.DType == "I32";
            if (quantized && !(metadata.GetValueOrDefault("quant") ?? "").StartsWith("q8", StringComparison.Ordinal)) throw new NotSupportedException("the quantised umT5 embedding table must be 8-bit blocks");
            return new UmT5Encoder(accelerator, source, repo, file, stream, info, quantized, int.Parse(metadata.GetValueOrDefault("num_layers", "24")));
        }
        catch { stream.Dispose(); throw; }
    }

    /// <summary>Open the encoder file from a path on disk.</summary>
    public static Task<UmT5Encoder> CreateAsync(Accelerator accelerator, string safetensorsPath, CancellationToken cancellationToken = default)
        => CreateAsync(accelerator, new LocalFolderModelSource(""), "", safetensorsPath, cancellationToken);

    /// <summary>The embedding rows of <paramref name="ids"/> as floats [n, 4096], read straight from the file (bf16 rows, or 8-bit blocks dequantised here).</summary>
    private async Task<float[]> EmbedAsync(int[] ids, CancellationToken ct)
    {
        var x = new float[(long)ids.Length * Dim];
        int rowBytes = _embeddingQuantized ? Dim / 32 * 9 * 4 : Dim * 2;
        var row = new byte[rowBytes];
        var first = new Dictionary<int, int>();                      // token id -> the first position that holds it: every distinct row is read once, in file order
        for (int i = 0; i < ids.Length; i++) first.TryAdd(ids[i], i);
        foreach (var (id, i) in first.OrderBy(kv => kv.Key))
        {
            _embeddings.Seek(_embeddingInfo.DataOffset + (long)id * rowBytes, SeekOrigin.Begin);
            await _embeddings.ReadExactlyAsync(row, ct);
            long o = (long)i * Dim;
            if (!_embeddingQuantized)
                for (int c = 0; c < Dim; c++) x[o + c] = BitConverter.Int32BitsToSingle((row[2 * c] | (row[2 * c + 1] << 8)) << 16);     // bf16 -> float
            else
                for (int b = 0; b < Dim / 32; b++)
                {
                    int p = b * 36;                                  // one block: [d f16 | m f16] then 8 words of 4 bytes (element 4 * word + byte)
                    float d = (float)BitConverter.ToHalf(row, p), m = (float)BitConverter.ToHalf(row, p + 2);
                    for (int e = 0; e < 32; e++) x[o + b * 32 + e] = d * row[p + 4 + e] + m;
                }
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
    /// Encode several prompts in one pass over the layers (each layer's weights are read and uploaded once; the rows of all prompts are stacked for the matrix products, attention runs per
    /// prompt): the contexts [n_i, 4096], in order.
    /// </summary>
    public async Task<float[][]> EncodeManyAsync(IReadOnlyList<int[]> sequences, CancellationToken ct = default)
    {
        int count = sequences.Count, maxL = 0, total = 0;
        var offsets = new int[count];
        for (int s = 0; s < count; s++) { if (sequences[s].Length == 0) throw new ArgumentException("no tokens"); offsets[s] = total; total += sequences[s].Length; maxL = Math.Max(maxL, sequences[s].Length); }
        var all = new float[(long)total * Dim];
        for (int s = 0; s < count; s++) Array.Copy(await EmbedAsync(sequences[s], ct), 0, all, (long)offsets[s] * Dim, (long)sequences[s].Length * Dim);
        var buckets = new MemoryBuffer1D<int, Stride1D.Dense>[count];
        using var xb = _acc.Allocate1D(all);
        try
        {
            for (int s = 0; s < count; s++) buckets[s] = _acc.Allocate1D(WanKernels.T5RelativeBuckets(sequences[s].Length));
            Tap?.Invoke("embedding", xb.View, sequences[0].Length * Dim);
            using var hn = _acc.Allocate1D<float>((long)total * Dim);
            using var qkv = _acc.Allocate1D<float>((long)total * 3 * Dim);
            using var att = _acc.Allocate1D<float>((long)total * Dim);
            using var proj = _acc.Allocate1D<float>((long)total * Dim);
            using var gate = _acc.Allocate1D<float>((long)total * FfnDim);
            using var fc1 = _acc.Allocate1D<float>((long)total * FfnDim);
            using var scores = _acc.Allocate1D<float>((long)Heads * maxL * maxL);
            for (int b = 0; b < _layers; b++)
            {
                ct.ThrowIfCancellationRequested();
                string p = $"blocks.{b}.";
                using var w = await LoadLayerAsync(name => name.StartsWith(p, StringComparison.Ordinal), ct);          // this layer only
                using var fw = new Flux2Weights(_acc, w, ownsWeights: false);                                           // matrices: bf16 as stored, or dequantised one at a time
                // self-attention
                hn.View.CopyFrom(xb.View);
                _wk.RowRmsNorm(hn.View, w.BF16(p + "norm1.weight"), total, Dim, 0, Dim, 1e-6f);
                _k.Linear(hn.View, fw.Matrix(p + "attn.qkv.weight"), null, qkv.View, total, Dim, 3 * Dim);
                for (int s = 0; s < count; s++)
                {
                    int L = sequences[s].Length, off = offsets[s];
                    _wk.T5Attention(qkv.View.SubView((long)off * 3 * Dim, (long)L * 3 * Dim), w.BF16(p + "pos_embedding.embedding.weight"), buckets[s].View, scores.View,
                        att.View.SubView((long)off * Dim, (long)L * Dim), L, Heads, HeadDim);
                }
                _k.Linear(att.View, fw.Matrix(p + "attn.o.weight"), null, proj.View, total, Dim, Dim);
                _k.GateResidual(xb.View, proj.View, default, -1, total * Dim, Dim);
                // gated feed-forward
                hn.View.CopyFrom(xb.View);
                _wk.RowRmsNorm(hn.View, w.BF16(p + "norm2.weight"), total, Dim, 0, Dim, 1e-6f);
                _k.Linear(hn.View, fw.Matrix(p + "ffn.gate.weight"), null, gate.View, total, Dim, FfnDim, DiTKernels.Act.GeluTanh);
                _k.Linear(hn.View, fw.Matrix(p + "ffn.fc1.weight"), null, fc1.View, total, Dim, FfnDim);
                _wk.MulInPlace(fc1.View, gate.View, total * FfnDim);
                _k.Linear(fc1.View, fw.Matrix(p + "ffn.fc2.weight"), null, proj.View, total, FfnDim, Dim);
                _k.GateResidual(xb.View, proj.View, default, -1, total * Dim, Dim);
                await _acc.SynchronizeAsync();                                         // the layer's weights are freed when `w` goes out of scope
                if (Tap != null && (b == 0 || b == _layers / 2 || b == _layers - 1)) Tap($"layer_{b}", xb.View, sequences[0].Length * Dim);
            }
            using (var nw = await LoadLayerAsync(name => name == "norm.weight", ct))
            {
                _wk.RowRmsNorm(xb.View, nw.BF16("norm.weight"), total, Dim, 0, Dim, 1e-6f);
                await _acc.SynchronizeAsync();
            }
            Tap?.Invoke("final", xb.View, sequences[0].Length * Dim);
            var host = await xb.View.CopyToHostAsync();
            var result = new float[count][];
            for (int s = 0; s < count; s++) { result[s] = new float[(long)sequences[s].Length * Dim]; Array.Copy(host, (long)offsets[s] * Dim, result[s], 0, result[s].Length); }
            return result;
        }
        finally
        {
            foreach (var b in buckets) b?.Dispose();
        }
    }

    public void Dispose() => _embeddings.Dispose();
}
