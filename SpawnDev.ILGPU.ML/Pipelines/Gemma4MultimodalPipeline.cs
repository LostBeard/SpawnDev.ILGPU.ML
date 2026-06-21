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
    private readonly Kernels.GpuArgMax _argmax;
    private readonly Stream _textStream;          // kept open for token_embd row gather
    private readonly GGUFModel _model;
    private readonly InferenceSession _session;
    private readonly SentencePieceTokenizer _tok;
    private readonly Gemma4MultimodalProjectorGpu _projector;
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
        var projector = new Gemma4MultimodalProjectorGpu(accelerator, MmprojModel.Load(mmprojPath));
        // (Tried raising GraphExecutor.SyncIntervalNodes to drop the mid-forward drains — it was SLOWER:
        // the periodic command-buffer flushes let the GPU start work while the CPU keeps dispatching, so the
        // default 64 stays. The token-by-token prefill is the real win.)
        return new Gemma4MultimodalPipeline(accelerator, stream, model, session, projector, maxSeqLen, ownsSession: true);
    }

    /// <summary>
    /// Browser/stream create: build the pipeline from async, seekable streams instead of file paths — so a
    /// hub <c>TorrentReadStream</c> / OPFS source can feed it (no local file). <paramref name="openTextGguf"/>
    /// opens a FRESH seekable read stream over the text GGUF each call; it's called TWICE — once for the
    /// session's weight upload, once kept open for the per-row token-embedding gather (two concurrent seekable
    /// readers over the same cached torrent are fine). <paramref name="mmprojStream"/> is read fully to bytes
    /// (the mmproj is small next to the text model). The caller owns the accelerator.
    /// </summary>
    public static async Task<Gemma4MultimodalPipeline> CreateFromStreamsAsync(
        Accelerator accelerator, Func<Task<Stream>> openTextGguf, Stream mmprojStream,
        int maxSeqLen = 4096, Action<string, int>? onProgress = null)
    {
        // 1) Upload the text decoder weights to the GPU from one stream (disposed after upload).
        InferenceSession session;
        await using (var uploadStream = await openTextGguf())
            session = await InferenceSession.CreateFromGGUFStreamAsync(accelerator, uploadStream, onProgress, default, acceptInputsEmbeds: true);

        // 2) Keep a second seekable stream open for the host-side token-row gather.
        var gatherStream = await openTextGguf();
        var model = await GGUFParser.ParseHeaderAsync(gatherStream);
        model.SourceStream = gatherStream;

        // 3) mmproj: read to bytes, load the projector.
        using var ms = new MemoryStream();
        await mmprojStream.CopyToAsync(ms);
        var projector = new Gemma4MultimodalProjectorGpu(accelerator, MmprojModel.Load(ms.ToArray()));

        return new Gemma4MultimodalPipeline(accelerator, gatherStream, model, session, projector, maxSeqLen, ownsSession: true);
    }

    private Gemma4MultimodalPipeline(Accelerator accelerator, Stream textStream, GGUFModel model,
        InferenceSession session, Gemma4MultimodalProjectorGpu projector, int maxSeqLen, bool ownsSession)
    {
        _accel = accelerator;
        _argmax = new Kernels.GpuArgMax(accelerator);
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
        // bf16 KV cache (the default — ~½ KV VRAM). Geordi's BFloat16 CUDA store/load fix shipped in
        // SpawnDev.ILGPU 4.13.0-local.4; bf16 store/load is now correct on CUDA/OpenCL/WebGPU/WebGL/Wasm.
        _cache = new GGUFDecodeKVCache(accelerator, kvHeads, hd, maxSeqLen);
        _session.EnableGGUFDecode(_cache);
        // Fixed-shape decode loop: recycle per-step output buffers + warm-cache the proven-stable shape readbacks
        // (the cache self-validates via probe→stable→finalize; this loop argmaxes each step's logits before the
        // next step, satisfying the output-recycling contract). See GgufTextGenerationPipeline.
        _session.CacheShapeReadbacks = true;
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
        _session.ResetGGUFDecode();
        var (imageBlocks, audioBlocks) = await ProjectMediaAsync(images, audio);
        try
        {
            // Full single-turn gemma4 chat template (BOS + system + user[+media] + model), media spliced RAW.
            var rows = new List<EmbRow>();
            if (_bos >= 0) await CtrlAsync(rows, _bos);
            await CtrlAsync(rows, _turnO); await TextAsync(rows, "system\n"); if (thinking && _think >= 0) await CtrlAsync(rows, _think); await TextAsync(rows, "\n"); await CtrlAsync(rows, _turnC); await TextAsync(rows, "\n");
            await CtrlAsync(rows, _turnO); await TextAsync(rows, "user\n");
            await AppendMediaAsync(rows, imageBlocks, audioBlocks);
            await TextAsync(rows, prompt); await CtrlAsync(rows, _turnC); await TextAsync(rows, "\n");
            await CtrlAsync(rows, _turnO); await TextAsync(rows, "model\n");
            return await PrefillAndGenerateAsync(rows, maxNewTokens, onToken);
        }
        finally { DisposeBlocks(imageBlocks); DisposeBlocks(audioBlocks); }
    }

    /// <summary>Start a multi-turn chat. The KV cache is reused ACROSS turns (only each new turn is
    /// prefilled — O(new tokens), not O(whole conversation)), so it stays coherent like llama.cpp/ollama
    /// chat. One chat at a time per pipeline (single KV cache); starting a new chat resets the cache.</summary>
    public Gemma4Chat StartChat(bool thinking = true) => new Gemma4Chat(this, thinking);

    internal void ResetForChat() => _session.ResetGGUFDecode();

    /// <summary>One chat turn: append the user turn (+media) at the RUNNING KV cursor (no reset), generate
    /// the model turn. Turn 0 emits BOS + the system block; later turns first close the prior model turn
    /// (the stop token wasn't cached) before opening the new user turn — so the cached context matches the
    /// canonical multi-turn transcript exactly.</summary>
    internal async Task<string> ChatTurnAsync(Gemma4Chat chat, string text,
        IReadOnlyList<ImageInput>? images, IReadOnlyList<float[]>? audio, int maxNewTokens, Func<int, string, Task>? onToken)
    {
        var (imageBlocks, audioBlocks) = await ProjectMediaAsync(images, audio);
        try
        {
            var rows = new List<EmbRow>();
            if (chat.Turn == 0)
            {
                if (_bos >= 0) await CtrlAsync(rows, _bos);
                await CtrlAsync(rows, _turnO); await TextAsync(rows, "system\n"); if (chat.Thinking && _think >= 0) await CtrlAsync(rows, _think); await TextAsync(rows, "\n"); await CtrlAsync(rows, _turnC); await TextAsync(rows, "\n");
            }
            else
            {
                await CtrlAsync(rows, _turnC); await TextAsync(rows, "\n");   // close the previous model turn (its <turn|> was generated but not cached)
            }
            await CtrlAsync(rows, _turnO); await TextAsync(rows, "user\n");
            await AppendMediaAsync(rows, imageBlocks, audioBlocks);
            await TextAsync(rows, text); await CtrlAsync(rows, _turnC); await TextAsync(rows, "\n");
            await CtrlAsync(rows, _turnO); await TextAsync(rows, "model\n");
            chat.Turn++;
            return await PrefillAndGenerateAsync(rows, maxNewTokens, onToken);
        }
        finally { DisposeBlocks(imageBlocks); DisposeBlocks(audioBlocks); }
    }

    // ── shared building blocks (one-shot GenerateAsync + multi-turn ChatTurnAsync use the same path) ──

    /// <summary>One prefill embedding row, kept ZERO-COPY where possible: a text/control token is gathered
    /// host-side from the GGUF stream (<see cref="Host"/> = the scaled [n_embd] row); a media row is a
    /// sub-view into a GPU projector-output block (<see cref="Gpu"/> + <see cref="Offset"/>) fed straight into
    /// the decoder, no GPU→host→GPU round-trip.</summary>
    private readonly record struct EmbRow(float[]? Host, MemoryBuffer1D<float, Stride1D.Dense>? Gpu, long Offset);

    /// <summary>Project each media item to its RAW [n, n_embd] embedding block (gemma4 splices media unscaled),
    /// kept GPU-RESIDENT: the projector (<see cref="Gemma4MultimodalProjectorGpu"/>) returns a GPU buffer whose
    /// rows are spliced as <c>inputs_embeds</c> sub-views — no readback. Caller disposes the returned buffers
    /// after the prefill consumes them.</summary>
    private async Task<(List<MemoryBuffer1D<float, Stride1D.Dense>> images, List<MemoryBuffer1D<float, Stride1D.Dense>> audio)>
        ProjectMediaAsync(IReadOnlyList<ImageInput>? images, IReadOnlyList<float[]>? audio)
    {
        if (images is { Count: > 0 } && !SupportsImages) throw new InvalidOperationException("mmproj has no vision encoder.");
        if (audio is { Count: > 0 } && !SupportsAudio) throw new InvalidOperationException("mmproj has no audio encoder.");
        var imageBlocks = new List<MemoryBuffer1D<float, Stride1D.Dense>>();
        if (images != null)
            foreach (var im in images)
            {
                var (patches, nCols, nRows) = Gemma4ImagePreprocessor.Preprocess(im.Rgb, im.Width, im.Height);
                imageBlocks.Add(await _projector.EncodeImageToBufferAsync(patches, nCols * nRows, nCols, nRows));
            }
        var audioBlocks = new List<MemoryBuffer1D<float, Stride1D.Dense>>();
        if (audio != null)
            foreach (var wav in audio)
            {
                var (frames, nFrames) = Gemma4AudioPreprocessor.Frame(wav);
                audioBlocks.Add(await _projector.EncodeAudioToBufferAsync(frames, nFrames));
            }
        return (imageBlocks, audioBlocks);
    }

    private static void DisposeBlocks(List<MemoryBuffer1D<float, Stride1D.Dense>> blocks)
    {
        foreach (var b in blocks) b.Dispose();
    }

    // Token-row helpers gather token_embd rows over the (possibly async, browser) stream — all async.
    private async Task TextAsync(List<EmbRow> rows, string s) { foreach (var t in _tok.Encode(s)) rows.Add(new EmbRow(await ScaledRowAsync(t), null, 0)); }
    private async Task CtrlAsync(List<EmbRow> rows, int t) => rows.Add(new EmbRow(await ScaledRowAsync(t), null, 0));
    private async Task MediaAsync(List<EmbRow> rows, MemoryBuffer1D<float, Stride1D.Dense> block, int begin, int end)
    {
        await CtrlAsync(rows, begin);
        int n = (int)(block.Length / _nEmbd);
        for (int p = 0; p < n; p++) rows.Add(new EmbRow(null, block, (long)p * _nEmbd)); // GPU sub-view, no readback
        await CtrlAsync(rows, end);
    }
    private async Task AppendMediaAsync(List<EmbRow> rows,
        List<MemoryBuffer1D<float, Stride1D.Dense>> imageBlocks, List<MemoryBuffer1D<float, Stride1D.Dense>> audioBlocks)
    {
        foreach (var b in imageBlocks) await MediaAsync(rows, b, _imgBegin, _imgEnd);
        foreach (var b in audioBlocks) await MediaAsync(rows, b, _audBegin, _audEnd);
    }

    /// <summary>Prefill the rows TOKEN-BY-TOKEN (seq=1 each) then greedily generate the model turn. Prefill
    /// token-by-token (not one batched seq=N forward) because the executor's per-node CPU residual scales
    /// super-linearly with sequence length (~107x for seq=32 vs seq=1), so N cheap KV-cache steps beat one
    /// big prefill; only the LAST prefill position's logits matter (they predict token 0), so intermediate
    /// readbacks are skipped. The trailing stop token (&lt;turn|&gt;/&lt;eos&gt;) is NOT fed into the cache —
    /// the next chat turn re-emits &lt;turn|&gt; to close the model turn in-context.</summary>
    private async Task<string> PrefillAndGenerateAsync(List<EmbRow> rows, int maxNewTokens, Func<int, string, Task>? onToken)
    {
        int next = -1;
        for (int i = 0; i < rows.Count; i++)
        {
            var outputs = await ForwardAsync(rows[i]);
            if (i == rows.Count - 1) next = await ReadLastArgMaxAsync(outputs); // last prefill position predicts token 0
        }

        var generated = new List<int>();
        for (int step = 0; step < maxNewTokens; step++)
        {
            generated.Add(next);
            if (onToken != null) await onToken(generated.Count, _tok.Decode(generated.ToArray()));
            if (next == _turnC || next == _eos || step == maxNewTokens - 1) break;
            next = await ReadLastArgMaxAsync(await ForwardAsync(new EmbRow(await ScaledRowAsync(next), null, 0)));
        }
        // Strip the trailing stop token from the RETURNED text (it stays out of the cache either way).
        if (generated.Count > 0 && (generated[^1] == _turnC || generated[^1] == _eos)) generated.RemoveAt(generated.Count - 1);
        return _tok.Decode(generated.ToArray());
    }

    /// <summary>Run one seq=1 inputs_embeds forward (KV-cache decode step) for a single [n_embd] embedding row.
    /// A media row feeds its GPU block sub-view DIRECTLY (zero-copy); a host text/control row is uploaded.</summary>
    private async Task<Dictionary<string, Tensor>> ForwardAsync(EmbRow row)
    {
        if (row.Gpu != null)
        {
            var view = row.Gpu.View.SubView(row.Offset, _nEmbd);
            return await _session.RunDecodeStepAsync(new Dictionary<string, Tensor>
            { ["inputs_embeds"] = new Tensor(view, new[] { 1, 1, _nEmbd }, "inputs_embeds") });
        }
        using var inBuf = _accel.Allocate1D(row.Host!);
        return await _session.RunDecodeStepAsync(new Dictionary<string, Tensor>
        { ["inputs_embeds"] = new Tensor(inBuf.View, new[] { 1, 1, _nEmbd }, "inputs_embeds") });
    }

    /// <summary>Greedy next-token from the last position's logits via a GPU argmax — reads back one int, not
    /// the ~1 MB vocab row (this path is greedy-only). Browser-portable (async, no sync CopyToCPU).</summary>
    private async Task<int> ReadLastArgMaxAsync(Dictionary<string, Tensor> outputs)
    {
        var logitsT = outputs.TryGetValue("logits", out var l) ? l : outputs.Values.First();
        int vocab = logitsT.Shape[^1];
        long lastOff = (long)(logitsT.ElementCount / vocab - 1) * vocab;
        return await _argmax.ArgMaxAsync(logitsT.Data.SubView(lastOff, vocab), vocab);
    }

    /// <summary>Gather token <paramref name="t"/>'s embedding row from token_embd and scale by sqrt(n_embd)
    /// (the gemma token-embedding scale; media rows bypass this and are spliced raw).</summary>
    private async Task<float[]> ScaledRowAsync(int t)
    {
        var r = await _model.GetTensorRowFloat32Async(_tokenEmbd, t) ?? throw new InvalidOperationException($"no embedding row for token {t}.");
        var o = new float[_nEmbd];
        for (int e = 0; e < _nEmbd; e++) o[e] = r[e] * _embScale;
        return o;
    }

    /// <summary>Disposes the KV-cache, the owned session, and the text stream. The accelerator is caller-owned.</summary>
    public void Dispose()
    {
        _projector.Dispose();
        _cache.Dispose();
        _argmax.Dispose();
        if (_ownsSession) _session.Dispose();
        _textStream.Dispose();
    }
}

/// <summary>
/// A multi-turn chat session over a <see cref="Gemma4MultimodalPipeline"/>. Holds the turn counter; the
/// conversation state itself lives in the pipeline's KV cache (reused across turns). Create with
/// <see cref="Gemma4MultimodalPipeline.StartChat"/>; each <see cref="SendAsync"/> appends a user turn
/// (optionally with images/audio) and returns the model's reply. The same chat works behind the console
/// and the browser demos.
/// </summary>
public sealed class Gemma4Chat
{
    private readonly Gemma4MultimodalPipeline _pipe;

    internal Gemma4Chat(Gemma4MultimodalPipeline pipe, bool thinking)
    {
        _pipe = pipe;
        Thinking = thinking;
        _pipe.ResetForChat();   // fresh KV cache for this conversation
    }

    /// <summary>Turns sent so far (0 before the first <see cref="SendAsync"/>).</summary>
    public int Turn { get; internal set; }

    /// <summary>Whether the system turn requests gemma4's thinking channel (set at construction).</summary>
    public bool Thinking { get; }

    /// <summary>Send one user message (+ optional images / 16 kHz mono audio) and get the model's reply.
    /// <paramref name="onToken"/> streams (tokenCount, textSoFar) as it generates.</summary>
    public Task<string> SendAsync(string text,
        IReadOnlyList<ImageInput>? images = null, IReadOnlyList<float[]>? audio = null,
        int maxNewTokens = 256, Func<int, string, Task>? onToken = null)
        => _pipe.ChatTurnAsync(this, text, images, audio, maxNewTokens, onToken);
}
