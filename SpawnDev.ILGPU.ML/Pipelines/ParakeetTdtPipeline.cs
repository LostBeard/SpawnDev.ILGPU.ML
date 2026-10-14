using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Kernels;
using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnDev.ILGPU.ML.Tensors;
using SpawnDev.SpawnJS.JSObjects;
using System.Diagnostics;
using System.Text;

namespace SpawnDev.ILGPU.ML.Pipelines;

/// <summary>
/// Speech recognition with NVIDIA Parakeet TDT (FastConformer encoder + token-and-duration transducer):
/// 16 kHz audio in, text out, every stage on the accelerator.
/// </summary>
/// <remarks>
/// <para>
/// Why this exists next to <see cref="SpeechRecognitionPipeline"/> (Whisper): Whisper pads every call to a fixed
/// 30 s window, so a four-word turn pays for half a minute of audio, and its decoder emits one token per full
/// decoder pass. Parakeet's encoder is as long as the audio, and the TDT joint predicts a token AND how many
/// frames to skip, so decoding is a handful of tiny steps. On the reference run of the 4 s LibriVox clip the same
/// graphs took 303 ms (encoder) + 32 ms (decode) on a CPU, against 3.1 s for Whisper tiny on this GPU in Vertex.
/// </para>
/// <para>
/// The three graphs are the ONNX export used by onnx-asr (<c>istupakov/parakeet-tdt-0.6b-v3-onnx</c>):
/// <c>nemo128.onnx</c> (log-mel, 128 bins), <c>encoder-model.onnx</c> and <c>decoder_joint-model.onnx</c>
/// (prediction LSTM + joint). Greedy TDT, as NeMo and onnx-asr do it: one joint evaluation per step gives
/// <c>VocabSize</c> token logits (the last one is the blank) followed by the duration logits; a non-blank token is
/// emitted and advances the prediction network's state; a duration above 0 skips that many encoder frames, a
/// blank with duration 0 moves on one frame, and at most <see cref="MaxSymbolsPerFrame"/> tokens leave one frame.
/// </para>
/// <para>
/// Bulk data never crosses to the host: the waveform is uploaded once, the mel and encoder output stay on the
/// device, the frame fed to the joint is a view of the (once transposed) encoder output, and the prediction
/// network's state lives in two persistent device buffers. Only the two argmax indices per step cross.
/// </para>
/// </remarks>
public sealed class ParakeetTdtPipeline : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly InferenceSession _preprocessor;
    private readonly InferenceSession _encoder;
    private readonly InferenceSession _decoderJoint;
    private readonly string[] _vocab;
    private readonly int _blank;
    private readonly GpuArgMax _argmax;

    private MemoryBuffer1D<float, Stride1D.Dense>? _frames;       // encoder output transposed to [T, D]
    private MemoryBuffer1D<float, Stride1D.Dense>? _scalar;       // one float read back at a time
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _state1;   // prediction LSTM h, [2, 1, 640]
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _state2;   // prediction LSTM c, [2, 1, 640]
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _lastToken; // the previous token, as a [1, 1] input
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _targetLength;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int>? _transposeKernel;

    /// <summary>Encoder output channels (1024 for the 0.6B models).</summary>
    public int EncoderDim { get; }

    /// <summary>Token logits per joint step, blank included (8193 for v3).</summary>
    public int VocabSize => _vocab.Length;

    /// <summary>Most tokens one encoder frame may emit before the decoder moves on (NeMo's default is 10).</summary>
    public int MaxSymbolsPerFrame { get; set; } = 10;

    /// <summary>Raised once per joint step with (frame, token, duration), for diagnostics.</summary>
    public event Action<int, int, int>? OnStep;

    /// <summary>
    /// Raised with ("features" | "encoded", tensor) when a stage finishes, so a test can compare a stage with a
    /// reference. The tensor is valid until the next call; nothing is copied unless a subscriber reads it.
    /// </summary>
    public event Action<string, Tensor>? OnStage;

    private const int StateElements = 2 * 640;

    /// <param name="preprocessor"><c>nemo128.onnx</c>: waveforms [1, N] + lengths -> features [1, 128, T].</param>
    /// <param name="encoder"><c>encoder-model.onnx</c> (fp32, fp16-stored, or int8).</param>
    /// <param name="decoderJoint"><c>decoder_joint-model.onnx</c>.</param>
    /// <param name="vocabulary">One entry per token id; the LAST one is the blank. See <see cref="ParseVocabulary"/>.</param>
    public ParakeetTdtPipeline(InferenceSession preprocessor, InferenceSession encoder, InferenceSession decoderJoint,
        Accelerator accelerator, string[] vocabulary, int encoderDim = 1024)
    {
        _preprocessor = preprocessor;
        _encoder = encoder;
        _decoderJoint = decoderJoint;
        _accelerator = accelerator;
        _vocab = vocabulary;
        _blank = vocabulary.Length - 1;
        EncoderDim = encoderDim;
        _argmax = new GpuArgMax(accelerator);
        _state1 = accelerator.Allocate1D<float>(StateElements);
        _state2 = accelerator.Allocate1D<float>(StateElements);
        _lastToken = accelerator.Allocate1D<float>(1);
        _targetLength = accelerator.Allocate1D(new[] { 1f });
        // The mel/encoder outputs are consumed before the next call; the joint's outputs are copied out below.
        _preprocessor.OutputsLiveUntilNextRun = true;
        _encoder.OutputsLiveUntilNextRun = true;
        _decoderJoint.OutputsLiveUntilNextRun = true;
    }

    /// <summary>Reads the export's <c>vocab.txt</c>: lines of "token id", the id being the position.</summary>
    public static string[] ParseVocabulary(string vocabTxt)
    {
        var entries = new SortedDictionary<int, string>();
        foreach (var line in vocabTxt.Split('\n', StringSplitOptions.RemoveEmptyEntries))
        {
            var l = line.TrimEnd('\r');
            var cut = l.LastIndexOf(' ');
            if (cut <= 0 || !int.TryParse(l.AsSpan(cut + 1), out var id)) continue;
            entries[id] = l[..cut];
        }
        if (entries.Count == 0) throw new FormatException("vocab.txt has no 'token id' lines");
        var vocab = new string[entries.Keys.Max() + 1];
        foreach (var (id, token) in entries) vocab[id] = token;
        for (int i = 0; i < vocab.Length; i++)
            if (vocab[i] == null) throw new FormatException($"vocab.txt has no entry for id {i}");
        return vocab;
    }

    /// <summary>Transcribe 16 kHz mono PCM (any other rate is resampled on the host first).</summary>
    public async Task<TranscriptionResult> TranscribeAsync(float[] audioSamples, int sampleRate = 16000)
    {
        ArgumentNullException.ThrowIfNull(audioSamples);
        if (sampleRate != AudioPreprocessor.WhisperSampleRate)
            audioSamples = AudioPreprocessor.Resample(audioSamples, sampleRate, AudioPreprocessor.WhisperSampleRate);
        using var pcm = _accelerator.Allocate1D(audioSamples.Length == 0 ? new float[1] : audioSamples);
        return await TranscribeAsync(pcm.View, audioSamples.Length).ConfigureAwait(false);
    }

    /// <summary>Browser path: the PCM is a JS typed array and goes straight to the device.</summary>
    public async Task<TranscriptionResult> TranscribeAsync(Float32Array audioSamples, int sampleRate = 16000)
    {
        ArgumentNullException.ThrowIfNull(audioSamples);
        if (sampleRate != AudioPreprocessor.WhisperSampleRate)
            return await TranscribeAsync(audioSamples.ToArray(), sampleRate).ConfigureAwait(false);
        int n = (int)audioSamples.Length;
        var pcm = _accelerator.Allocate1D<float>(Math.Max(n, 1));
        try
        {
            if (n > 0) MediaInterop.UploadToDevice(audioSamples, pcm);
            return await TranscribeAsync(pcm.View, n).ConfigureAwait(false);
        }
        finally
        {
            await _accelerator.SynchronizeAsync().ConfigureAwait(false);
            pcm.Dispose();
        }
    }

    /// <summary>Device path: 16 kHz mono PCM already on the accelerator (the entry the other two funnel into).</summary>
    public async Task<TranscriptionResult> TranscribeAsync(ArrayView1D<float, Stride1D.Dense> pcm, int length)
    {
        var total = Stopwatch.StartNew();
        if (length < AudioPreprocessor.WhisperSampleRate / 20)   // under 50 ms: nothing to recognise
            return new TranscriptionResult { Text = "", Language = "en", InferenceTimeMs = total.Elapsed.TotalMilliseconds };

        // ── log-mel (nemo128.onnx), on the device ──
        var clock = Stopwatch.StartNew();
        using var lengthBuf = _accelerator.Allocate1D(new[] { (float)length });
        var mel = await _preprocessor.RunAsync(new Dictionary<string, Tensor>
        {
            [_preprocessor.InputNames[0]] = new Tensor(pcm.SubView(0, length), new[] { 1, length }),
            [_preprocessor.InputNames[1]] = new Tensor(lengthBuf.View, new[] { 1 }),
        }).ConfigureAwait(false);
        var features = mel[_preprocessor.OutputNames[0]];
        var featureLengths = mel[_preprocessor.OutputNames[1]];
        await _accelerator.SynchronizeAsync().ConfigureAwait(false);
        var melMs = clock.Elapsed.TotalMilliseconds;
        OnStage?.Invoke("features", features);

        // ── encoder ──
        clock.Restart();
        var enc = await _encoder.RunAsync(new Dictionary<string, Tensor>
        {
            [_encoder.InputNames[0]] = features,
            [_encoder.InputNames[1]] = featureLengths,
        }).ConfigureAwait(false);
        var encoded = enc[_encoder.OutputNames[0]];                 // [1, D, T]
        int dim = encoded.Shape[1], frames = encoded.Shape[2];
        // The one tiny scalar this stage reads back: how many frames are real (the rest is padding).
        _scalar ??= _accelerator.Allocate1D<float>(1);
        _scalar.View.CopyFrom(enc[_encoder.OutputNames[1]].Data.SubView(0, 1));
        await _accelerator.SynchronizeAsync().ConfigureAwait(false);
        int validFrames = (int)MathF.Round((await _scalar.CopyToHostAsync<float>(0, 1).ConfigureAwait(false))[0]);
        validFrames = Math.Clamp(validFrames, 0, frames);
        await _accelerator.SynchronizeAsync().ConfigureAwait(false);
        var encoderMs = clock.Elapsed.TotalMilliseconds;
        OnStage?.Invoke("encoded", encoded);

        // ── transpose [D, T] -> [T, D] once, so frame t is one contiguous view ──
        if (_frames == null || _frames.Length < dim * frames)
        {
            _frames?.Dispose();
            _frames = _accelerator.Allocate1D<float>(dim * frames);
        }
        _transposeKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int>(TransposeDT);
        _transposeKernel(dim * frames, encoded.Data, _frames.View, dim, frames);

        // ── greedy TDT ──
        clock.Restart();
        _state1.MemSetToZero();
        _state2.MemSetToZero();
        var tokens = new List<int>();
        int last = _blank, t = 0, emitted = 0, steps = 0;
        while (t < validFrames)
        {
            steps++;
            var frameView = _frames.View.SubView((long)t * dim, dim);
            // _lastToken is rewritten between steps by a host->device scalar copy (one float).
            _lastToken.CopyFromCPU(new[] { (float)last });
            var r = await _decoderJoint.RunAsync(new Dictionary<string, Tensor>
            {
                ["encoder_outputs"] = new Tensor(frameView, new[] { 1, dim, 1 }),
                ["targets"] = new Tensor(_lastToken.View, new[] { 1, 1 }),
                ["target_length"] = new Tensor(_targetLength.View, new[] { 1 }),
                ["input_states_1"] = new Tensor(_state1.View, new[] { 2, 1, 640 }),
                ["input_states_2"] = new Tensor(_state2.View, new[] { 2, 1, 640 }),
            }).ConfigureAwait(false);
            var logits = r[_decoderJoint.OutputNames[0]].Data;
            int total2 = (int)logits.Length;
            int durations = total2 - _vocab.Length;
            int token = await _argmax.ArgMaxAsync(logits.SubView(0, _vocab.Length), _vocab.Length).ConfigureAwait(false);
            int duration = durations > 1
                ? await _argmax.ArgMaxAsync(logits.SubView(_vocab.Length, durations), durations).ConfigureAwait(false)
                : 0;
            OnStep?.Invoke(t, token, duration);

            if (token != _blank)
            {
                tokens.Add(token);
                last = token;
                // The prediction network only moves on when a token is emitted: keep its new state on the device.
                _state1.View.CopyFrom(r[_decoderJoint.OutputNames[2]].Data.SubView(0, StateElements));
                _state2.View.CopyFrom(r[_decoderJoint.OutputNames[3]].Data.SubView(0, StateElements));
            }
            if (duration > 0) { t += duration; emitted = 0; }
            else if (token == _blank || ++emitted >= MaxSymbolsPerFrame) { t++; emitted = 0; }
        }
        await _accelerator.SynchronizeAsync().ConfigureAwait(false);
        var decodeMs = clock.Elapsed.TotalMilliseconds;

        return new TranscriptionResult
        {
            Text = Detokenize(tokens),
            Language = "en",
            InferenceTimeMs = total.Elapsed.TotalMilliseconds,
            MelTimeMs = melMs,
            ModelTimeMs = total.Elapsed.TotalMilliseconds - melMs,
            EncoderMs = encoderMs,
            DecodeStepsMs = decodeMs,
            DecodeSteps = steps,
            EncoderNodeCount = _encoder.NodeCount,
            DecoderNodeCount = _decoderJoint.NodeCount,
        };
    }

    /// <summary>SentencePiece pieces to text: U+2581 marks a word start.</summary>
    private string Detokenize(List<int> tokens)
    {
        var sb = new StringBuilder();
        foreach (var id in tokens)
            if (id >= 0 && id < _blank) sb.Append(_vocab[id]);
        return sb.ToString().Replace('▁', ' ').Trim();
    }

    private static void TransposeDT(Index1D i, ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output, int dim, int frames)
    {
        int d = i / frames, t = i % frames;
        output[t * dim + d] = input[d * frames + t];
    }

    public void Dispose()
    {
        _frames?.Dispose();
        _scalar?.Dispose();
        _state1.Dispose();
        _state2.Dispose();
        _lastToken.Dispose();
        _targetLength.Dispose();
        _argmax.Dispose();
        _preprocessor.Dispose();
        _encoder.Dispose();
        _decoderJoint.Dispose();
    }
}
