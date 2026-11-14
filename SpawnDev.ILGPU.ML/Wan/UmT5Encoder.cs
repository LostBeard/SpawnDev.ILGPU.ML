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
/// format; the matrices are dequantised one at a time into a bf16 scratch just before their GEMM), activations fp32. ComfyUI's repackaged files load as they are
/// (umt5_xxl_fp16.safetensors, umt5_xxl_fp8_e4m3fn_scaled.safetensors: Hugging Face T5 names; each layer is rewritten into this layout as it streams, <see cref="WanComfyFormat"/>).
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
    private readonly bool _embeddingF16;                         // fp16 rows (ComfyUI's files); otherwise bf16, or the 8-bit blocks
    private readonly bool _hfLayout;                             // the file has the Hugging Face T5 names (ComfyUI's umT5 files): each layer is rewritten into ours as it loads (WanComfyFormat)
    private readonly bool _embeddingF8;                          // 8-bit float rows with no scale (ComfyUI's all-in-one FLUX.1 checkpoints)
    private readonly string _prefix;                             // the T5's names inside an all-in-one checkpoint (FLUX.1: "text_encoders.t5xxl.transformer."); "" for a T5 file
    private readonly DiTKernels _k;
    private readonly WanKernels _wk;
    public const int Dim = 4096, Heads = 64, HeadDim = 64, FfnDim = 10240, DefaultLayers = 24;
    private readonly int _layers;
    private readonly bool _sharedBias;                           // T5 v1.1 (LTX-Video): ONE relative-position table, in block 0, used by every layer; umT5 (Wan) has one per layer

    /// <summary>Test hook: (stage name, activation view, element count), valid only during the call; names "embedding", "layer_N", "final" (the first sequence of a call).</summary>
    public Action<string, FV, int>? Tap { get; set; }

    /// <summary>
    /// CUDA: fp32-class activations through the tensor-core GEMMs (<see cref="DiTKernels.PreciseActivations"/>). T5-v1.1-XXL's activations are huge with outliers, so bf16 activations
    /// cost the most here (FLUX.1's T5 against ComfyUI's fp32 run: 6.1e-3 on CUDA's bf16 path, 8.3e-6 on SpawnGen's RX 590). Off by default (the Wan / LTX gates were measured
    /// on the bf16 path); a ComfyUI-parity caller turns it on. Other backends compute in fp32 either way.
    /// </summary>
    public bool PreciseActivations { get => _k.PreciseActivations; set => _k.PreciseActivations = value; }

    private UmT5Encoder(Accelerator accelerator, IModelSource source, string repo, string file, Stream embeddings, SafeTensorInfo info, bool quantized, int layers, bool sharedBias, bool hfLayout = false, string prefix = "")
    {
        _acc = accelerator; _source = source; _repo = repo; _file = file; _embeddings = embeddings; _embeddingInfo = info; _embeddingQuantized = quantized; _layers = layers; _sharedBias = sharedBias;
        _hfLayout = hfLayout; _embeddingF16 = info.DType == "F16"; _embeddingF8 = info.DType == "F8_E4M3"; _prefix = prefix;
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
            // an all-in-one checkpoint keeps the T5 under a prefix (ComfyUI's flux1-dev-fp8: "text_encoders.t5xxl.transformer."): the T5 is the tensors around its shared.weight
            var sharedAny = tensors.FirstOrDefault(t => t.Name == "shared.weight") ?? tensors.FirstOrDefault(t => t.Name.EndsWith(".transformer.shared.weight", StringComparison.Ordinal));
            string prefix = sharedAny == null ? "" : sharedAny.Name[..^"shared.weight".Length];
            if (prefix.Length > 0) tensors = tensors.Where(t => t.Name.StartsWith(prefix, StringComparison.Ordinal)).Select(t => new SafeTensorInfo { Name = t.Name[prefix.Length..], DType = t.DType, Shape = t.Shape, DataOffset = t.DataOffset, DataLength = t.DataLength }).ToList();
            if (WanComfyFormat.IsComfyT5(tensors.Select(t => t.Name)))
            {
                // ComfyUI's T5 files (Hugging Face names, fp16, scaled or unscaled 8-bit floats, no metadata): the layer count and the bias-table layout come from the names
                var names = tensors.Select(t => t.Name).ToHashSet();
                var shared = tensors.First(t => t.Name == "shared.weight");
                if (shared.DType is not ("F16" or "BF16" or "F8_E4M3") || shared.Shape.Length != 2 || shared.Shape[1] != Dim) throw new NotSupportedException($"'{file}': shared.weight is {shared.DType} [{string.Join(", ", shared.Shape)}]; expected fp16, bf16 or 8-bit float [vocab, {Dim}]");
                int n = 0;
                while (names.Contains($"encoder.block.{n}.layer.0.SelfAttention.q.weight")) n++;
                bool oneBias = n > 1 && !names.Contains("encoder.block.1.layer.0.SelfAttention.relative_attention_bias.weight");
                return new UmT5Encoder(accelerator, source, repo, file, stream, shared, false, n, oneBias, hfLayout: true, prefix: prefix);
            }
            var info = tensors.FirstOrDefault(t => t.Name == "token_embedding.weight") ?? throw new InvalidDataException($"'{file}' has no token_embedding.weight");
            bool quantized = info.DType == "I32";
            if (quantized && !(metadata.GetValueOrDefault("quant") ?? "").StartsWith("q8", StringComparison.Ordinal)) throw new NotSupportedException("the quantised umT5 embedding table must be 8-bit blocks");
            return new UmT5Encoder(accelerator, source, repo, file, stream, info, quantized, int.Parse(metadata.GetValueOrDefault("num_layers", "24")), metadata.GetValueOrDefault("shared_bias", "0") == "1");
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
        int rowBytes = _embeddingQuantized ? Dim / 32 * 9 * 4 : _embeddingF8 ? Dim : Dim * 2;
        var row = new byte[rowBytes];
        var first = new Dictionary<int, int>();                      // token id -> the first position that holds it: every distinct row is read once, in file order
        for (int i = 0; i < ids.Length; i++) first.TryAdd(ids[i], i);
        foreach (var (id, i) in first.OrderBy(kv => kv.Key))
        {
            _embeddings.Seek(_embeddingInfo.DataOffset + (long)id * rowBytes, SeekOrigin.Begin);
            await _embeddings.ReadExactlyAsync(row, ct);
            long o = (long)i * Dim;
            if (_embeddingF16)
                for (int c = 0; c < Dim; c++) x[o + c] = (float)BitConverter.ToHalf(row, 2 * c);                                          // fp16 -> float
            else if (_embeddingF8)
                for (int c = 0; c < Dim; c++) x[o + c] = F8E4M3(row[c]);                                                                 // 8-bit float -> float (no scale)
            else if (!_embeddingQuantized)
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

    /// <summary>An E4M3 8-bit float's value (sign, 4 exponent bits with bias 7, 3 mantissa bits; S.1111.111 is NaN).</summary>
    private static float F8E4M3(byte b)
    {
        int e = (b >> 3) & 15, m = b & 7;
        if ((b & 0x7F) == 0x7F) return float.NaN;
        float v = e == 0 ? m * (1f / 512f) : (1f + m / 8f) * MathF.ScaleB(1f, e - 7);
        return (b & 0x80) != 0 ? -v : v;
    }

    /// <summary>Load the tensors of one layer (or the final norm) from the file, in file order.</summary>
    private async Task<GpuWeights> LoadLayerAsync(Func<string, bool> filter, CancellationToken ct)
    {
        await using var stream = await _source.OpenAsync(_repo, _file, ct);
        using var empty = System.Text.Json.JsonDocument.Parse("{}");
        if (!_hfLayout) return await GpuWeights.LoadAsync(_acc, stream, empty.RootElement, _file, filter, ct);
        // the Hugging Face names: load the tensors that make up the ones asked for (our names), then rewrite them into our layout on the device
        var w = await GpuWeights.LoadAsync(_acc, stream, empty.RootElement, _file, name => name.StartsWith(_prefix, StringComparison.Ordinal)
            && WanComfyFormat.T5OurName(name[_prefix.Length..]) is { } ours && ours != "token_embedding.weight" && filter(ours), ct);
        try
        {
            if (_prefix.Length > 0) foreach (var n in w.Names.ToList()) w.Rename(n, n[_prefix.Length..]);
            await WanComfyFormat.AdaptT5Async(_acc, w);
            return w;
        }
        catch { w.Dispose(); throw; }
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
            const string SharedBiasName = "blocks.0.pos_embedding.embedding.weight";
            using var sharedBias = _sharedBias ? await LoadLayerAsync(name => name == SharedBiasName, ct) : null;       // [32, 64] bf16, kept for every layer
            for (int b = 0; b < _layers; b++)
            {
                ct.ThrowIfCancellationRequested();
                string p = $"blocks.{b}.";
                using var w = await LoadLayerAsync(name => name.StartsWith(p, StringComparison.Ordinal), ct);          // this layer only
                using var fw = new Flux2Weights(_acc, w, ownsWeights: false);                                           // matrices: bf16 as stored, or dequantised one at a time
                // self-attention
                hn.View.CopyFrom(xb.View);
                _wk.RowRmsNorm(hn.View, w.BF16(p + "norm1.weight"), total, Dim, 0, Dim, 1e-6f);
                // the text encoders stay on the float GEMM, never W8A8: T5-v1.1-XXL's activations (huge, with outliers) made 8-bit activations unstable (MEASURED, LTX T5 prompt 0:
                // 0.14 relRMS on CUDA, where fc2's float path rounds to bf16, 0.033 on OpenCL; umT5 doubled its error); it runs once per prompt and is cached, so W8A8 bought little
                fw.Linear(_k, hn.View, p + "attn.qkv.weight", null, qkv.View, total, Dim, 3 * Dim, allowW8A8: false);
                for (int s = 0; s < count; s++)
                {
                    int L = sequences[s].Length, off = offsets[s];
                    _wk.T5Attention(qkv.View.SubView((long)off * 3 * Dim, (long)L * 3 * Dim), _sharedBias ? sharedBias!.BF16(SharedBiasName) : w.BF16(p + "pos_embedding.embedding.weight"), buckets[s].View, scores.View,
                        att.View.SubView((long)off * Dim, (long)L * Dim), L, Heads, HeadDim);
                }
                fw.Linear(_k, att.View, p + "attn.o.weight", null, proj.View, total, Dim, Dim, allowW8A8: false);
                _k.GateResidual(xb.View, proj.View, default, -1, total * Dim, Dim);
                // gated feed-forward
                hn.View.CopyFrom(xb.View);
                _wk.RowRmsNorm(hn.View, w.BF16(p + "norm2.weight"), total, Dim, 0, Dim, 1e-6f);
                fw.Linear(_k, hn.View, p + "ffn.gate.weight", null, gate.View, total, Dim, FfnDim, DiTKernels.Act.GeluTanh, allowW8A8: false);
                fw.Linear(_k, hn.View, p + "ffn.fc1.weight", null, fc1.View, total, Dim, FfnDim, allowW8A8: false);
                _wk.MulInPlace(fc1.View, gate.View, total * FfnDim);
                fw.Linear(_k, fc1.View, p + "ffn.fc2.weight", null, proj.View, total, FfnDim, Dim, allowW8A8: false);
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
