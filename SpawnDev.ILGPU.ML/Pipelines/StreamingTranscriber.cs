using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Kernels;
using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnDev.ILGPU.ML.Tensors;
using System.Diagnostics;
using System.Text;

namespace SpawnDev.ILGPU.ML.Pipelines;

/// <summary>What a streaming recogniser knows after some more audio: the words so far, and the ones that just arrived.</summary>
/// <param name="Text">Everything recognised since the stream began (or since <see cref="StreamingTranscriber.Reset"/>), punctuated and capitalised by the model.</param>
/// <param name="NewTokens">The tokens this call added, each with the encoder frame (80 ms) it was emitted at, so a caller can date a word.</param>
/// <param name="ChunksRun">Encoder chunks (160 ms each) this call ran.</param>
/// <param name="ProcessingMs">Wall time of this call.</param>
public sealed record StreamingUpdate(string Text, IReadOnlyList<(string Token, long Frame)> NewTokens, int ChunksRun, double ProcessingMs);

/// <summary>
/// Speech recognition that keeps up with the speaker: audio goes in as it is spoken and words come out within a fraction of a second, from NVIDIA's cache-aware
/// streaming FastConformer with an RNN-T decoder (<c>nvidia/nemotron-speech-streaming-en-0.6b</c>, the sherpa-onnx export).
/// </summary>
/// <remarks>
/// <para>
/// Cache-aware means the encoder is run on a short window at a time and carries its own attention and convolution context between windows (three cache tensors per layer,
/// held in device buffers here), so each frame of audio is encoded once however long the speech goes on. In the 160 ms export a step takes 25 mel frames (the 16 new ones of a
/// 160 ms chunk plus 9 of overlap) and yields two 80 ms encoder frames; the RNN-T joiner then emits zero or more tokens per frame, greedily (blank = the last token id).
/// </para>
/// <para>
/// A streaming pass is the FAST pass: its words appear while the person is still talking, and the model corrects an earlier word only as far as its small look-ahead allows.
/// For the final text, run the finished utterance through a stronger offline recogniser (<see cref="ParakeetTdtPipeline"/>), the way Google's and Alexa's assistants do;
/// the streaming words are what let an application react EARLY (hear its name, see a sentence stop mid-thought, begin an answer).
/// </para>
/// <para>
/// Not thread-safe: one stream at a time per instance. Features are computed on the host (<see cref="NemoStreamingFeatures"/>, a few thousand samples per chunk);
/// the caches, the encoder output, the prediction network's state and the joiner all stay on the device. Only two argmax indices per joiner step cross.
/// </para>
/// </remarks>
public sealed class StreamingTranscriber : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly InferenceSession _encoder, _decoder, _joiner;
    private readonly string[] _vocab;
    private readonly int _blank;
    private readonly GpuArgMax _argmax;
    private readonly NemoStreamingFeatures _features = new();

    private readonly int _window, _shift, _layers, _cacheChannel, _cacheTime, _dim, _hidden, _predLayers;
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _audio, _length, _cacheCh, _cacheTm, _cacheLen;
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _h, _c, _decOut, _token, _tokenLength;
    private MemoryBuffer1D<float, Stride1D.Dense>? _frames;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int>? _transpose;

    private readonly List<float[]> _pending = new();     // mel frames not yet consumed by a chunk
    private readonly List<int> _tokens = new();
    private long _frameBase;                             // global mel frame index of _pending[0]
    private long _encoderFrame;                          // global encoder frame index of the next frame decoded
    private bool _started;
    private readonly float[] _melFrame = new float[NemoStreamingFeatures.Mels];
    private readonly float[] _chunk;

    /// <summary>Tokens one encoder frame may emit before the decoder moves on (NeMo's default is 10).</summary>
    public int MaxSymbolsPerFrame { get; set; } = 10;

    /// <summary>Everything recognised so far.</summary>
    public string Text => Detokenize(_tokens);

    /// <summary>Milliseconds the encoder spent on the last chunk (device work included), and the decoder on its frames: for latency measurement.</summary>
    public (double EncoderMs, double DecodeMs) LastChunkMs { get; private set; }

    /// <summary>The input shapes the three sessions must be created with (batch 1, one chunk), keyed by session.</summary>
    public static (Dictionary<string, int[]> Encoder, Dictionary<string, int[]> Decoder, Dictionary<string, int[]> Joiner) InputShapes(
        int windowFrames = 25, int layers = 24, int cacheChannel = 70, int cacheTime = 8, int dim = 1024, int predLayers = 2, int hidden = 640, int mels = 128) =>
    (
        new() { ["audio_signal"] = new[] { 1, mels, windowFrames }, ["length"] = new[] { 1 }, ["cache_last_channel"] = new[] { 1, layers, cacheChannel, dim },
                ["cache_last_time"] = new[] { 1, layers, dim, cacheTime }, ["cache_last_channel_len"] = new[] { 1 } },
        new() { ["targets"] = new[] { 1, 1 }, ["target_length"] = new[] { 1 }, ["states.1"] = new[] { predLayers, 1, hidden }, ["onnx::Slice_3"] = new[] { predLayers, 1, hidden } },
        new() { ["encoder_outputs"] = new[] { 1, dim, 1 }, ["decoder_outputs"] = new[] { 1, hidden, 1 } }
    );

    /// <param name="encoder">The cache-aware encoder, created with <see cref="InputShapes"/>.Encoder.</param>
    /// <param name="decoder">The prediction network (LSTM), created with <see cref="InputShapes"/>.Decoder.</param>
    /// <param name="joiner">The joiner, created with <see cref="InputShapes"/>.Joiner.</param>
    /// <param name="vocabulary">One entry per token id; the LAST one is the blank (see <see cref="ParakeetTdtPipeline.ParseVocabulary"/>).</param>
    /// <param name="windowFrames">Mel frames per encoder step (25 for the 160 ms export; the export's <c>window_size</c>).</param>
    /// <param name="shiftFrames">Mel frames the window advances (16 for 160 ms; the export's <c>chunk_shift</c>).</param>
    public StreamingTranscriber(InferenceSession encoder, InferenceSession decoder, InferenceSession joiner, Accelerator accelerator, string[] vocabulary,
        int windowFrames = 25, int shiftFrames = 16, int layers = 24, int cacheChannel = 70, int cacheTime = 8, int encoderDim = 1024, int predLayers = 2, int predHidden = 640)
    {
        _encoder = encoder; _decoder = decoder; _joiner = joiner; _accelerator = accelerator; _vocab = vocabulary;
        _blank = vocabulary.Length - 1;
        _window = windowFrames; _shift = shiftFrames; _layers = layers; _cacheChannel = cacheChannel; _cacheTime = cacheTime;
        _dim = encoderDim; _predLayers = predLayers; _hidden = predHidden;
        _argmax = new GpuArgMax(accelerator);
        _chunk = new float[NemoStreamingFeatures.Mels * windowFrames];
        _audio = accelerator.Allocate1D<float>(_chunk.Length);
        _length = accelerator.Allocate1D(new[] { (float)windowFrames });
        _cacheCh = accelerator.Allocate1D<float>(layers * cacheChannel * encoderDim);
        _cacheTm = accelerator.Allocate1D<float>(layers * encoderDim * cacheTime);
        _cacheLen = accelerator.Allocate1D<float>(1);
        _h = accelerator.Allocate1D<float>(predLayers * predHidden);
        _c = accelerator.Allocate1D<float>(predLayers * predHidden);
        _decOut = accelerator.Allocate1D<float>(predHidden);
        _token = accelerator.Allocate1D<float>(1);
        _tokenLength = accelerator.Allocate1D(new[] { 1f });
        _encoder.OutputsLiveUntilNextRun = true;
        _decoder.OutputsLiveUntilNextRun = true;
        _joiner.OutputsLiveUntilNextRun = true;
    }

    /// <summary>Starts a new stream: caches and the prediction network are cleared, the words forgotten.</summary>
    public void Reset()
    {
        _features.Reset(); _pending.Clear(); _tokens.Clear(); _frameBase = 0; _encoderFrame = 0; _started = false;
        _cacheCh.MemSetToZero(); _cacheTm.MemSetToZero(); _cacheLen.MemSetToZero(); _h.MemSetToZero(); _c.MemSetToZero();
    }

    /// <summary>Feeds the next 16 kHz mono audio in [-1, 1] (any amount) and returns what was recognised from it.</summary>
    public async Task<StreamingUpdate> AcceptAudioAsync(float[] pcm16k)
    {
        ArgumentNullException.ThrowIfNull(pcm16k);
        var total = Stopwatch.StartNew();
        if (!_started) await StartAsync().ConfigureAwait(false);
        _features.Accept(pcm16k);
        return await RunReadyChunksAsync(total).ConfigureAwait(false);
    }

    /// <summary>
    /// Ends the utterance: one second of silence pushes the last words through the look-ahead (as the reference runtime does), and the text is final.
    /// Call <see cref="Reset"/> before the next utterance.
    /// </summary>
    public Task<StreamingUpdate> FinishAsync() => AcceptAudioAsync(new float[NemoStreamingFeatures.SampleRate]);

    private async Task StartAsync()
    {
        _started = true;
        // The prediction network starts from the blank token; its state after that first step is where decoding begins.
        await StepDecoderAsync(_blank).ConfigureAwait(false);
    }

    private async Task<StreamingUpdate> RunReadyChunksAsync(Stopwatch total)
    {
        var added = new List<(string, long)>();
        int chunks = 0;
        while (_features.TryNext(_melFrame)) _pending.Add((float[])_melFrame.Clone());
        while (_pending.Count >= _window)
        {
            var clock = Stopwatch.StartNew();
            for (int t = 0; t < _window; t++)
                for (int m = 0; m < NemoStreamingFeatures.Mels; m++)
                    _chunk[m * _window + t] = _pending[t][m];
            _audio.CopyFromCPU(_chunk);

            var enc = await _encoder.RunAsync(new Dictionary<string, Tensor>
            {
                ["audio_signal"] = new Tensor(_audio.View, new[] { 1, NemoStreamingFeatures.Mels, _window }),
                ["length"] = new Tensor(_length.View, new[] { 1 }),
                ["cache_last_channel"] = new Tensor(_cacheCh.View, new[] { 1, _layers, _cacheChannel, _dim }),
                ["cache_last_time"] = new Tensor(_cacheTm.View, new[] { 1, _layers, _dim, _cacheTime }),
                ["cache_last_channel_len"] = new Tensor(_cacheLen.View, new[] { 1 }),
            }).ConfigureAwait(false);
            var encoded = enc["outputs"];                                   // [1, D, T']
            int frames = encoded.Shape[2];
            // The next caches become this stream's caches (device to device).
            _cacheCh.View.CopyFrom(enc["cache_last_channel_next"].Data.SubView(0, _cacheCh.Length));
            _cacheTm.View.CopyFrom(enc["cache_last_time_next"].Data.SubView(0, _cacheTm.Length));
            _cacheLen.View.CopyFrom(enc["cache_last_channel_next_len"].Data.SubView(0, 1));

            if (_frames == null || _frames.Length < _dim * frames) { _frames?.Dispose(); _frames = _accelerator.Allocate1D<float>(_dim * frames); }
            _transpose ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
                ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int>(TransposeDT);
            _transpose(_dim * frames, encoded.Data, _frames.View, _dim, frames);
            await _accelerator.SynchronizeAsync().ConfigureAwait(false);
            var encoderMs = clock.Elapsed.TotalMilliseconds;

            clock.Restart();
            for (int t = 0; t < frames; t++)
            {
                var frameView = _frames.View.SubView((long)t * _dim, _dim);
                for (int emitted = 0; emitted < MaxSymbolsPerFrame; emitted++)
                {
                    var logits = (await _joiner.RunAsync(new Dictionary<string, Tensor>
                    {
                        ["encoder_outputs"] = new Tensor(frameView, new[] { 1, _dim, 1 }),
                        ["decoder_outputs"] = new Tensor(_decOut.View, new[] { 1, _hidden, 1 }),
                    }).ConfigureAwait(false))[_joiner.OutputNames[0]].Data;
                    int token = await _argmax.ArgMaxAsync(logits.SubView(0, _vocab.Length), _vocab.Length).ConfigureAwait(false);
                    if (token == _blank) break;
                    _tokens.Add(token);
                    added.Add((_vocab[token], _encoderFrame));
                    await StepDecoderAsync(token).ConfigureAwait(false);
                }
                _encoderFrame++;
            }
            await _accelerator.SynchronizeAsync().ConfigureAwait(false);
            LastChunkMs = (encoderMs, clock.Elapsed.TotalMilliseconds);
            _pending.RemoveRange(0, Math.Min(_shift, _pending.Count));
            _frameBase += _shift;
            chunks++;
        }
        return new StreamingUpdate(Text, added, chunks, total.Elapsed.TotalMilliseconds);
    }

    /// <summary>One step of the prediction network on <paramref name="token"/>: its output and state stay on the device.</summary>
    private async Task StepDecoderAsync(int token)
    {
        _token.CopyFromCPU(new[] { (float)token });
        var r = await _decoder.RunAsync(new Dictionary<string, Tensor>
        {
            ["targets"] = new Tensor(_token.View, new[] { 1, 1 }),
            ["target_length"] = new Tensor(_tokenLength.View, new[] { 1 }),
            ["states.1"] = new Tensor(_h.View, new[] { _predLayers, 1, _hidden }),
            ["onnx::Slice_3"] = new Tensor(_c.View, new[] { _predLayers, 1, _hidden }),
        }).ConfigureAwait(false);
        _decOut.View.CopyFrom(r[_decoder.OutputNames[0]].Data.SubView(0, _hidden));
        _h.View.CopyFrom(r[_decoder.OutputNames[2]].Data.SubView(0, _predLayers * _hidden));
        _c.View.CopyFrom(r[_decoder.OutputNames[3]].Data.SubView(0, _predLayers * _hidden));
    }

    private string Detokenize(List<int> tokens)
    {
        var sb = new StringBuilder();
        foreach (var id in tokens)
            if (id >= 0 && id < _blank) sb.Append(_vocab[id]);
        return sb.ToString().Replace('▁', ' ').Trim();
    }

    private static void TransposeDT(Index1D i, ArrayView1D<float, Stride1D.Dense> input, ArrayView1D<float, Stride1D.Dense> output, int dim, int frames)
    {
        int d = i / frames, t = i % frames;
        output[t * dim + d] = input[d * frames + t];
    }

    public void Dispose()
    {
        _frames?.Dispose();
        _audio.Dispose(); _length.Dispose(); _cacheCh.Dispose(); _cacheTm.Dispose(); _cacheLen.Dispose();
        _h.Dispose(); _c.Dispose(); _decOut.Dispose(); _token.Dispose(); _tokenLength.Dispose();
        _argmax.Dispose();
        _encoder.Dispose(); _decoder.Dispose(); _joiner.Dispose();
    }
}
