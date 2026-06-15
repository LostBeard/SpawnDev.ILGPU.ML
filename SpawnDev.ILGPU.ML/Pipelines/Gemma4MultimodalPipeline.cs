using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.GGUF;
using SpawnDev.ILGPU.ML.Kernels;
using SpawnDev.ILGPU.ML.Multimodal;
using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnDev.ILGPU.ML.Tensors;

namespace SpawnDev.ILGPU.ML.Pipelines;

/// <summary>A decoded image for <see cref="Gemma4MultimodalPipeline"/>: 8-bit interleaved RGB (HWC,
/// length = Width*Height*3). Decoding (PNG/JPEG → RGB) is the caller's job — ImageSharp/System.Drawing on
/// desktop, canvas/<see cref="MediaInterop"/> in the browser — so the library stays decoder-agnostic.</summary>
public readonly record struct ImageInput(byte[] Rgb, int Width, int Height);

/// <summary>
/// First-class MULTIMODAL generation for Gemma 4 12B "Unified" (encoder-free): text + image + audio + video
/// → text, in one call. Wraps the proven pieces so a consumer never touches the low-level plumbing:
/// gemma4 chat template, host-side token-embedding gather, the mmproj vision/audio projection
/// (<see cref="Gemma4MultimodalProjector"/>), the RAW media-embedding splice between the gemma4 media
/// markers, and the O(n) <c>inputs_embeds</c> KV-cache decode. Mirrors <see cref="GgufTextGenerationPipeline"/>
/// (the text-only twin) — same ergonomics, plus media inputs.
///
/// Usage:
/// <code>
/// using var pipe = await Gemma4MultimodalPipeline.CreateAsync(accel, textGgufPath, mmprojPath);
/// string answer = await pipe.GenerateAsync("Describe this.", images: [new ImageInput(rgb, w, h)]);
/// </code>
///
/// Images are preprocessed via <see cref="Gemma4ImagePreprocessor"/> (smart_resize+PAD_CEIL+im2col), audio
/// via <see cref="Gemma4AudioPreprocessor"/> (16 kHz → 640-sample frames); each media item becomes its own
/// marker-wrapped block (multiple images = VIDEO frames). Text rows are gathered ×sqrt(n_embd); media rows
/// are spliced RAW (gemma4 scales only token embeddings). NOTE: the host-side token-row gather reads the
/// GGUF stream synchronously — desktop-ready; a browser (async-only stream) build is a follow-up.
/// </summary>
public sealed class Gemma4MultimodalPipeline : IDisposable
{
    private readonly Accelerator _accel;
    private readonly Stream _textStream;          // kept open for token_embd row gather
    private readonly GGUFModel _model;
    private readonly InferenceSession _session;
    private readonly SentencePieceTokenizer _tok;
    private readonly Gemma4MultimodalProjector _projector;
    private readonly GGUFTensorInfo _tokenEmbd;
    private readonly GGUFDecodeKVCache _cache;
    private readonly bool _ownsSession;

    private readonly int _nEmbd;
    private readonly float _embScale;
    private readonly int _turnO, _turnC, _eos, _bos, _think, _imgBegin, _imgEnd, _audBegin, _audEnd;

    /// <summary>The model's SentencePiece tokenizer (from the GGUF vocab).</summary>
    public SentencePieceTokenizer Tokenizer => _tok;
    /// <summary>True if the mmproj carries the vision projector (gemma4uv).</summary>
    public bool SupportsImages { get; }
    /// <summary>True if the mmproj carries the audio projector (gemma4ua).</summary>
    public bool SupportsAudio { get; }

    /// <summary>
    /// Create a pipeline from the text decoder GGUF + the companion mmproj GGUF. Opens the text model with
    /// the <c>inputs_embeds</c> entry, loads the projector, and wires the KV-cache decode. The caller owns
    /// the <paramref name="accelerator"/>; this pipeline owns everything it creates (disposed in Dispose).
    /// </summary>
    public static async Task<Gemma4MultimodalPipeline> CreateAsync(
        Accelerator accelerator, string textGgufPath, string mmprojPath, int maxSeqLen = 4096)
    {
        var stream = File.OpenRead(textGgufPath);              // stays open for token-row gather
        var model = await GGUFParser.ParseHeaderAsync(stream);
        model.SourceStream = stream;
        var session = await InferenceSession.CreateFromGGUFFileAsync(accelerator, textGgufPath, acceptInputsEmbeds: true);
        var projector = new Gemma4MultimodalProjector(MmprojModel.Load(mmprojPath));
        return new Gemma4MultimodalPipeline(accelerator, stream, model, session, projector, maxSeqLen, ownsSession: true);
    }

    private Gemma4MultimodalPipeline(Accelerator accelerator, Stream textStream, GGUFModel model,
        InferenceSession session, Gemma4MultimodalProjector projector, int maxSeqLen, bool ownsSession)
    {
        _accel = accelerator;
        _textStream = textStream;
        _model = model;
        _session = session;
        _projector = projector;
        _ownsSession = ownsSession;
        _tok = SentencePieceTokenizer.FromGGUF(model)
            ?? throw new InvalidOperationException("GGUF model has no SentencePiece tokenizer metadata.");
        _tokenEmbd = model.Tensors.FirstOrDefault(t => t.Name == "token_embd.weight")
            ?? throw new InvalidOperationException("GGUF model has no token_embd.weight.");
        _nEmbd = (int)model.EmbeddingLength;
        _embScale = MathF.Sqrt(_nEmbd);
        SupportsImages = projector.SupportsVision;
        SupportsAudio = projector.SupportsAudio;

        int Id(string s) => _tok.TryGetId(s, out var v) ? v : -1;
        _turnO = Id("<|turn>"); _turnC = Id("<turn|>"); _eos = Id("<eos>"); _bos = Id("<bos>"); _think = Id("<|think|>");
        _imgBegin = Id("<|image>"); _imgEnd = Id("<image|>"); _audBegin = Id("<|audio>"); _audEnd = Id("<audio|>");

        int nLayers = (int)model.BlockCount, nHeads = (int)model.AttentionHeadCount;
        int defNKV = (int)model.AttentionHeadCountKV; if (defNKV == 0) defNKV = nHeads;
        int defHd = nHeads > 0 ? _nEmbd / nHeads : 0;
        var kvHeads = new int[nLayers]; var hd = new int[nLayers];
        for (int L = 0; L < nLayers; L++)
        { var cfg = GGUFGraphBuilder.GetLayerAttnConfig(model, L, nHeads, defNKV, defHd); kvHeads[L] = cfg.NKVHeads; hd[L] = cfg.HeadDim; }
        _cache = new GGUFDecodeKVCache(accelerator, kvHeads, hd, maxSeqLen);
        _session.EnableGGUFDecode(_cache);
    }

    /// <summary>
    /// Generate a response to <paramref name="prompt"/> with optional <paramref name="images"/> and
    /// <paramref name="audio"/> (16 kHz mono float samples). Returns the decoded assistant text (includes
    /// gemma4's &lt;|channel&gt;thought block — split on the channel markers for the final answer only).
    /// <paramref name="onToken"/> streams (tokenCount, textSoFar) per token. Greedy decode.
    /// </summary>
    public async Task<string> GenerateAsync(string prompt,
        IReadOnlyList<ImageInput>? images = null, IReadOnlyList<float[]>? audio = null,
        int maxNewTokens = 256, bool thinking = true, Func<int, string, Task>? onToken = null)
    {
        if (images is { Count: > 0 } && !SupportsImages) throw new InvalidOperationException("mmproj has no vision encoder.");
        if (audio is { Count: > 0 } && !SupportsAudio) throw new InvalidOperationException("mmproj has no audio encoder.");
        _session.ResetGGUFDecode();

        // Project each media item to its RAW [n, n_embd] embedding block (gemma4 splices media unscaled).
        var imageBlocks = new List<float[]>();
        if (images != null)
            foreach (var im in images)
            {
                var (patches, nCols, nRows) = Gemma4ImagePreprocessor.Preprocess(im.Rgb, im.Width, im.Height);
                imageBlocks.Add(_projector.EncodeImage(patches, nCols * nRows, nCols, nRows));
            }
        var audioBlocks = new List<float[]>();
        if (audio != null)
            foreach (var wav in audio)
            {
                var (frames, nFrames) = Gemma4AudioPreprocessor.Frame(wav);
                audioBlocks.Add(_projector.EncodeAudio(frames, nFrames));
            }

        // Assemble the prefill embedding sequence (gemma4 chat template + media blocks in the user turn).
        var rows = new List<float[]>();
        void Text(string s) { foreach (var t in _tok.Encode(s)) rows.Add(ScaledRow(t)); }
        void Ctrl(int t) => rows.Add(ScaledRow(t));
        void Media(float[] block, int begin, int end)
        {
            Ctrl(begin);
            int n = block.Length / _nEmbd;
            for (int p = 0; p < n; p++) { var o = new float[_nEmbd]; Array.Copy(block, (long)p * _nEmbd, o, 0, _nEmbd); rows.Add(o); }
            Ctrl(end);
        }
        if (_bos >= 0) Ctrl(_bos);
        Ctrl(_turnO); Text("system\n"); if (thinking && _think >= 0) Ctrl(_think); Text("\n"); Ctrl(_turnC); Text("\n");
        Ctrl(_turnO); Text("user\n");
        foreach (var b in imageBlocks) Media(b, _imgBegin, _imgEnd);
        foreach (var b in audioBlocks) Media(b, _audBegin, _audEnd);
        Text(prompt); Ctrl(_turnC); Text("\n");
        Ctrl(_turnO); Text("model\n");

        int seq = rows.Count;
        var prefill = new float[(long)seq * _nEmbd];
        for (int i = 0; i < seq; i++) Array.Copy(rows[i], 0, prefill, (long)i * _nEmbd, _nEmbd);

        var generated = new List<int>();
        float[] stepEmb = prefill; int stepSeq = seq;       // prefill, then 1 token/step
        for (int step = 0; step < maxNewTokens; step++)
        {
            using var inBuf = _accel.Allocate1D(stepEmb);
            var outputs = await _session.RunDecodeStepAsync(new Dictionary<string, Tensor>
            { ["inputs_embeds"] = new Tensor(inBuf.View, new[] { 1, stepSeq, _nEmbd }, "inputs_embeds") });

            var logitsT = outputs.TryGetValue("logits", out var l) ? l : outputs.Values.First();
            int vocab = logitsT.Shape[^1];
            int seqOut = logitsT.ElementCount / vocab;
            long lastOff = (long)(seqOut - 1) * vocab;
            // Browser-portable readback of the last position's logits (async, no sync CopyToCPU).
            using var read = _accel.Allocate1D<float>(vocab);
            await read.View.CopyFromAsync(logitsT.Data.SubView(lastOff, vocab));
            await _accel.SynchronizeAsync();
            var logits = await read.CopyToHostAsync<float>(0, vocab);

            int next = TextGenerationSampler.Greedy(logits);
            generated.Add(next);
            if (onToken != null) await onToken(generated.Count, _tok.Decode(generated.ToArray()));
            if (next == _turnC || next == _eos) break;
            stepEmb = ScaledRow(next); stepSeq = 1;
        }
        return _tok.Decode(generated.ToArray());
    }

    /// <summary>Gather token <paramref name="t"/>'s embedding row from token_embd and scale by sqrt(n_embd)
    /// (the gemma token-embedding scale; media rows bypass this and are spliced raw).</summary>
    private float[] ScaledRow(int t)
    {
        var r = _model.GetTensorRowFloat32(_tokenEmbd, t) ?? throw new InvalidOperationException($"no embedding row for token {t}.");
        var o = new float[_nEmbd];
        for (int e = 0; e < _nEmbd; e++) o[e] = r[e] * _embScale;
        return o;
    }

    /// <summary>Disposes the KV-cache, the owned session, and the text stream. The accelerator is caller-owned.</summary>
    public void Dispose()
    {
        _cache.Dispose();
        if (_ownsSession) _session.Dispose();
        _textStream.Dispose();
    }
}
