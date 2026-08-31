using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Tensors;
using System;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace SpawnDev.ILGPU.ML.Pipelines;

/// <summary>
/// Silero VAD: one 512-sample frame in, one speech probability out.
/// </summary>
/// <remarks>
/// <para>
/// The model is STATEFUL, and its state is not hidden inside it: <c>h</c> and <c>c</c> arrive as graph
/// INPUTS and the updated state comes back as <c>new_h</c>/<c>new_c</c> on every frame. Threading that
/// back is not an optimisation, it is the model - a detector run with frozen state still produces a
/// plausible probability for each frame and is wrong by up to 0.98 (MEASURED against onnxruntime over 4 s
/// of speech). <c>MLTestBase.SileroVadTests</c> asserts against a frozen-state control for that reason.
/// </para>
/// <para>
/// ⚠️ The frame size is FIXED at 512 samples at 16 kHz - the model declares <c>x</c> as [1, 512]. A caller
/// feeding a different size is feeding the wrong model.
/// </para>
/// </remarks>
public sealed class SileroVad : IDisposable
{
    /// <summary>Silero's native frame at 16 kHz.</summary>
    public const int WindowSize = 512;

    /// <summary>The only sample rate this wrapper supports; the 8 kHz sibling model is not shipped.</summary>
    public const int SampleRate = 16000;

    private const int StateCount = 2 * 1 * 64;

    private readonly Accelerator _accelerator;
    private readonly InferenceSession _session;
    private readonly int[] _xShape = { 1, WindowSize };
    private readonly int[] _stateShape = { 2, 1, 64 };

    // Allocated ONCE and reused. A detector runs ~31 times a second for as long as the microphone is open,
    // so allocating per frame would churn the buffer pool for the life of the session. Owning them for the
    // object's lifetime is also what the browser backends require: a buffer referenced by a pending
    // dispatch must not be freed before the flush, and a method-local `using var` cannot promise that.
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _xBuf;
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _hBuf;
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _cBuf;
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _probBuf;

    /// <remarks>
    /// ⚠️ PERFORMANCE, and the thing to read before optimising this. MEASURED in-browser
    /// (<c>Vad_Benchmark_FrameRate</c>, 60 warmed frames): WebGPU <b>177.9 ms</b> per frame, WebGL 85.6 ms,
    /// Wasm 191.8 ms - against a budget a live microphone fixes at <b>32 ms</b> (512 samples at 16 kHz).
    /// So the browser backends are 2.7x to 6x too slow to follow a microphone, while OpenCL on the desktop
    /// runs 4.99 ms (6.4x realtime).
    /// <para>
    /// ⚠️ The cost is NOT the node dispatches, which is the intuitive answer and the one I asserted before
    /// measuring. The executor's own split on WebGPU reads: mean 167.58 ms, of which <b>readback
    /// 110.94 ms across 16 readbacks</b>, syncDrain 7.06 ms. Two thirds of the frame is host traffic,
    /// because the recurrent operators EXECUTE ON THE HOST - each LSTM node reads X/h/c back, runs the
    /// recurrence in C#, and uploads the result. That is what made LSTM correct on six backends in 5.2.3,
    /// and it is the right thing to attack for speed. `W`/`R`/`B` are already cached across calls.
    /// </para>
    /// <para>
    /// ⚠️ <c>SessionGraphCapture</c> is the obvious answer and it does NOT work here as a drop-in - TRIED
    /// and reverted 2026-08-31. Routing this through it left WebGPU finding <b>0 utterances</b> and crashed
    /// CUDA with an access violation inside <c>GraphExecutor.RunAsync</c>. The reason is the one CLAUDE.md
    /// records for per-step stateful caches: <c>TryCaptureAsync</c> runs the graph SIX times to discover
    /// its patch points, and Silero's <c>h</c>/<c>c</c> are genuine per-frame state rather than a
    /// position-addressed KV cache, so they cannot survive that probing unchanged. Making capture work here
    /// needs the snapshot/restore contract, not a wrapper swap.
    /// ⚠️ Note that <c>Vad_SileroVad_MatchesOnnxRuntimeOverRealSpeech</c> stayed GREEN through all of it -
    /// it drives the session directly and never touches this class, so it cannot see a defect in this path.
    /// The endpointing and flush gates are what caught it.
    /// </para>
    /// </remarks>
    private SileroVad(Accelerator accelerator, InferenceSession session)
    {
        _accelerator = accelerator;
        _session = session;
        _xBuf = accelerator.Allocate1D<float>(WindowSize);
        _hBuf = accelerator.Allocate1D<float>(StateCount);
        _cBuf = accelerator.Allocate1D<float>(StateCount);
        _probBuf = accelerator.Allocate1D<float>(1);
        Reset();
    }

    /// <summary>Loads the detector from silero_vad.onnx (643 KB).</summary>
    public static SileroVad Create(Accelerator accelerator, byte[] modelBytes)
    {
        var session = InferenceSession.CreateFromFile(accelerator, modelBytes,
            inputShapes: new Dictionary<string, int[]>
            {
                ["x"] = new[] { 1, WindowSize },
                ["h"] = new[] { 2, 1, 64 },
                ["c"] = new[] { 2, 1, 64 },
            });
        return new SileroVad(accelerator, session);
    }

    /// <summary>Clears the recurrent state, so the next frame starts a new stream.</summary>
    public void Reset()
    {
        var zero = new float[StateCount];
        _hBuf.View.CopyFromCPU(zero);
        _cBuf.View.CopyFromCPU(zero);
    }

    /// <summary>Speech probability for one frame, advancing the recurrent state.</summary>
    /// <param name="frame">Exactly <see cref="WindowSize"/> mono samples in [-1, 1] at 16 kHz.</param>
    /// <remarks>
    /// ⚠️ The recurrent state NEVER leaves the GPU. `new_h`/`new_c` are copied GPU-to-GPU straight back
    /// into the buffers the next frame reads, so the only readback here is the single float of `prob`.
    /// The obvious implementation reads the state back to the host and re-uploads it - 256 floats each way
    /// plus two extra fences, 31 times a second for as long as the microphone is open - which is the
    /// zero-copy law broken in a loop.
    /// <para>
    /// ⚠️ It is NOT where the frame time goes, and it is worth recording that it is not. MEASURED on
    /// OpenCL with <c>tools/vad-harness bench</c>, removing both state readbacks moved the mean from
    /// 4.97 ms to 4.99 ms - no change at all, because a sync readback there is nearly free. Across a PMT
    /// A/B the win is real but modest and lands where the theory says it should: WebGPU -7.1% and
    /// CUDA -8.8% on the endpointing test, every other backend inside ±3% noise. Those durations include
    /// a constant model load and a 125-node graph compile, so the effect on the frame loop alone is larger
    /// than the headline. It is nevertheless NOT the dominant cost: the executor split (see the note on
    /// the frame budget above) puts 110.94 ms of a 167.58 ms WebGPU frame in the SIXTEEN readbacks the
    /// host-side LSTM performs. Anyone optimising this should start there rather than repeat this
    /// experiment.
    /// </para>
    /// </remarks>
    public async Task<float> ProcessFrameAsync(float[] frame)
    {
        if (frame.Length != WindowSize)
            throw new ArgumentException(
                $"Silero VAD takes exactly {WindowSize} samples per frame, got {frame.Length}.", nameof(frame));

        // CopyFromCPU is an immediate write rather than a queued dispatch, so it carries no command-encoder
        // dependency on the browser backends.
        _xBuf.View.CopyFromCPU(frame);

        var outputs = await _session.RunAsync(new Dictionary<string, Tensor>
        {
            ["x"] = new Tensor(_xBuf.View, _xShape),
            ["h"] = new Tensor(_hBuf.View, _stateShape),
            ["c"] = new Tensor(_cBuf.View, _stateShape),
        });

        // GPU-to-GPU. Native CopyBufferToBuffer on WebGPU - no shader, no dispatch, no host round trip.
        CopyState(outputs, "new_h", _hBuf);
        CopyState(outputs, "new_c", _cBuf);

        if (!outputs.TryGetValue("prob", out var probTensor))
            throw new InvalidOperationException("Silero VAD produced no output named 'prob'.");
        _probBuf.View.CopyFrom(probTensor.Data.SubView(0, 1));
        await _accelerator.SynchronizeAsync();
        var prob = await _probBuf.CopyToHostAsync<float>(0, 1);
        return prob[0];
    }

    private void CopyState(Dictionary<string, Tensor> outputs, string name,
        MemoryBuffer1D<float, Stride1D.Dense> into)
    {
        if (!outputs.TryGetValue(name, out var t))
            throw new InvalidOperationException($"Silero VAD produced no output named '{name}'.");
        if (t.ElementCount < StateCount)
            throw new InvalidOperationException(
                $"Silero VAD '{name}' holds {t.ElementCount} values, expected {StateCount}.");
        into.View.CopyFrom(t.Data.SubView(0, StateCount));
    }

    public void Dispose()
    {
        _session.Dispose();
        _xBuf.Dispose();
        _hBuf.Dispose();
        _cBuf.Dispose();
        _probBuf.Dispose();
    }
}

/// <summary>How aggressively speech is separated from silence.</summary>
/// <remarks>
/// The defaults are the ones RoseEars runs on the robot, chosen for a ten year old talking to it rather
/// than for a dictation app: half a second of silence ends a turn, because shorter clips her off between
/// clauses and longer makes the conversation feel like it is buffering. They are deliberately NOT the
/// silero library defaults, which wait only 100 ms.
/// </remarks>
public sealed class VadOptions
{
    /// <summary>Probabilities at or above this are speech.</summary>
    public float Threshold { get; set; } = 0.5f;

    /// <summary>Probabilities BELOW this end speech. Defaults to <see cref="Threshold"/> - 0.15.</summary>
    /// <remarks>
    /// The gap is hysteresis, and it is in the reference implementation for a reason: with a single
    /// threshold, a probability hovering around 0.5 opens and closes the segment on alternating frames.
    /// </remarks>
    public float? NegativeThreshold { get; set; }

    /// <summary>Silence needed to close a segment.</summary>
    public TimeSpan MinSilenceDuration { get; set; } = TimeSpan.FromMilliseconds(500);

    /// <summary>Segments shorter than this are discarded as noise.</summary>
    public TimeSpan MinSpeechDuration { get; set; } = TimeSpan.FromMilliseconds(250);

    /// <summary>A segment is cut at this length even if the speaker has not stopped.</summary>
    public TimeSpan MaxSpeechDuration { get; set; } = TimeSpan.FromSeconds(20);

    /// <summary>Audio kept either side of a segment, so the first and last phoneme survive.</summary>
    public TimeSpan SpeechPad { get; set; } = TimeSpan.FromMilliseconds(30);

    internal float ResolvedNegativeThreshold => NegativeThreshold ?? MathF.Max(Threshold - 0.15f, 0.01f);
}

/// <summary>One finished utterance.</summary>
/// <param name="Samples">The audio, mono 16 kHz float in [-1, 1].</param>
/// <param name="StartSample">Offset of the first sample within the stream fed to the detector.</param>
public readonly record struct SpeechSegment(float[] Samples, long StartSample)
{
    public double StartSeconds => StartSample / (double)SileroVad.SampleRate;
    public double DurationSeconds => Samples.Length / (double)SileroVad.SampleRate;
}

/// <summary>
/// Turns a stream of microphone samples into finished utterances.
/// </summary>
/// <remarks>
/// <para>
/// Endpointing has to come from a real VAD rather than an energy threshold: someone talking to a computer
/// pauses mid sentence constantly, and an energy gate either cuts them off or waits forever. This is the
/// design RoseEars runs on the robot, ported off sherpa-onnx so that it works in the browser too.
/// </para>
/// <para>
/// The state machine follows silero-vad's own <c>VADIterator</c>: probabilities at or above
/// <see cref="VadOptions.Threshold"/> open a segment, probabilities below the NEGATIVE threshold begin
/// closing it, and silence longer than <see cref="VadOptions.MinSilenceDuration"/> closes it. The two
/// thresholds differ on purpose - see <see cref="VadOptions.NegativeThreshold"/>.
/// </para>
/// <para>
/// Feed it any chunk size. It reframes to the model's fixed 512 internally, because a microphone does not
/// hand over 512-sample buffers and RTP hands over 320.
/// </para>
/// </remarks>
public sealed class VoiceActivityDetector : IDisposable
{
    private readonly SileroVad _vad;
    private readonly VadOptions _options;
    private readonly bool _ownsVad;

    private readonly float[] _frame = new float[SileroVad.WindowSize];
    private int _framed;

    // Audio is retained from the earliest point a segment could still begin, so a closed segment can be
    // handed over complete. _bufferStart is the absolute stream offset of _buffer[0].
    private readonly List<float> _buffer = new();
    private long _bufferStart;

    private long _currentSample;
    private bool _triggered;
    private long _tempEnd;
    private long _segmentStart;

    /// <summary>Raised with each completed utterance.</summary>
    public event Action<SpeechSegment>? OnSegment;

    /// <summary>Whether speech is currently open. A turn held on a clock should consult this.</summary>
    public bool IsSpeechActive => _triggered;

    /// <summary>The most recent frame's speech probability, for meters and diagnostics.</summary>
    public float LastProbability { get; private set; }

    /// <summary>Samples consumed so far, which is the detector's clock.</summary>
    public long SamplesProcessed => _currentSample;

    public VoiceActivityDetector(SileroVad vad, VadOptions? options = null, bool ownsVad = false)
    {
        _vad = vad;
        _options = options ?? new VadOptions();
        _ownsVad = ownsVad;
    }

    private int Samples(TimeSpan t) => (int)(t.TotalSeconds * SileroVad.SampleRate);

    /// <summary>Accepts microphone audio of any length, mono 16 kHz float.</summary>
    public async Task AcceptWaveformAsync(float[] samples, int offset = 0, int count = -1)
    {
        if (count < 0) count = samples.Length - offset;
        for (int i = 0; i < count; i++)
        {
            _frame[_framed++] = samples[offset + i];
            if (_framed < SileroVad.WindowSize) continue;

            _buffer.AddRange(_frame);
            _framed = 0;
            await ProcessFrameAsync();
        }
    }

    private async Task ProcessFrameAsync()
    {
        int window = SileroVad.WindowSize;
        _currentSample += window;
        float prob = await _vad.ProcessFrameAsync(_frame);
        LastProbability = prob;

        int pad = Samples(_options.SpeechPad);
        int minSilence = Samples(_options.MinSilenceDuration);
        int maxSpeech = Samples(_options.MaxSpeechDuration);

        // Speech resumed inside the grace period, so the pending end is cancelled.
        if (prob >= _options.Threshold && _tempEnd != 0) _tempEnd = 0;

        if (prob >= _options.Threshold && !_triggered)
        {
            _triggered = true;
            _segmentStart = Math.Max(0, _currentSample - pad - window);
        }

        // A speaker who never pauses would otherwise hold the turn forever; better a cut than silence.
        if (_triggered && _currentSample - _segmentStart > maxSpeech)
        {
            Emit(_segmentStart, _currentSample);
            _triggered = false;
            _tempEnd = 0;
            TrimBuffer(_currentSample);
            return;
        }

        if (prob < _options.ResolvedNegativeThreshold && _triggered)
        {
            if (_tempEnd == 0) _tempEnd = _currentSample;
            if (_currentSample - _tempEnd < minSilence) return;

            long end = _tempEnd + pad - window;
            _tempEnd = 0;
            _triggered = false;
            // Short blips are noise - a cough, a chair, a door - and handing one to a recogniser makes it
            // hallucinate a sentence nobody said.
            if (end - _segmentStart > Samples(_options.MinSpeechDuration)) Emit(_segmentStart, end);
            TrimBuffer(end);
            return;
        }

        // Nothing open and nothing pending, so audio older than the longest possible lead-in is dead.
        if (!_triggered && _tempEnd == 0) TrimBuffer(_currentSample - pad - window);
    }

    /// <summary>
    /// Closes out speech still in progress.
    /// </summary>
    /// <remarks>
    /// The detector only emits once it has seen enough trailing silence, so audio that ends while someone
    /// is still talking - the end of a recording, or of a session - would otherwise never be emitted.
    /// </remarks>
    public async Task FlushAsync()
    {
        if (_framed > 0)
        {
            // Pad the partial frame rather than drop it: the tail of the last word is inside it.
            Array.Clear(_frame, _framed, SileroVad.WindowSize - _framed);
            _buffer.AddRange(_frame);
            _framed = 0;
            await ProcessFrameAsync();
        }

        if (_triggered && _currentSample - _segmentStart > Samples(_options.MinSpeechDuration))
            Emit(_segmentStart, _currentSample);

        _triggered = false;
        _tempEnd = 0;
        TrimBuffer(_currentSample);
    }

    /// <summary>Drops all state and buffered audio - use when muting, or between sessions.</summary>
    public void Reset()
    {
        _vad.Reset();
        _buffer.Clear();
        _bufferStart = _currentSample;
        _framed = 0;
        _triggered = false;
        _tempEnd = 0;
        _segmentStart = 0;
    }

    private void Emit(long start, long end)
    {
        start = Math.Max(start, _bufferStart);
        end = Math.Min(end, _bufferStart + _buffer.Count);
        int length = (int)(end - start);
        if (length <= 0) return;

        var samples = new float[length];
        _buffer.CopyTo((int)(start - _bufferStart), samples, 0, length);
        OnSegment?.Invoke(new SpeechSegment(samples, start));
    }

    private void TrimBuffer(long keepFrom)
    {
        long drop = keepFrom - _bufferStart;
        if (drop <= 0) return;
        if (drop > _buffer.Count) drop = _buffer.Count;
        _buffer.RemoveRange(0, (int)drop);
        _bufferStart += drop;
    }

    public void Dispose()
    {
        if (_ownsVad) _vad.Dispose();
    }
}
