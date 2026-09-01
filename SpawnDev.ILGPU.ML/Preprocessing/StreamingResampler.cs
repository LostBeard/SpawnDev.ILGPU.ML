using System;
using System.Collections.Generic;

namespace SpawnDev.ILGPU.ML.Preprocessing;

/// <summary>
/// Rate conversion for a LIVE stream: feed it chunks as they arrive and it emits exactly what
/// <see cref="AudioPreprocessor.Resample(float[], int, int)"/> would have produced for the whole signal.
/// </summary>
/// <remarks>
/// <para>
/// ⚠️ <b>Why this exists.</b> <see cref="AudioPreprocessor.Resample(float[], int, int)"/> is a
/// windowed-sinc conversion: every output sample reads a window of source samples either side of its
/// centre, and at the SIGNAL EDGES that window is truncated and the weights renormalised so the edge does
/// not produce a gain step. That is correct for one complete clip and WRONG applied per chunk of a stream,
/// because it then treats every chunk boundary as a signal edge. A microphone hands over a frame every
/// ~10 ms, so resampling frame by frame manufactures an edge artifact 100 times a second - broadband
/// clicks at a steady rate, which is exactly the kind of thing a voice-activity detector reports as
/// speech. <c>MediaInterop.FromAudioDataAsync</c> takes a target rate per frame and did precisely this.
/// </para>
/// <para>
/// This keeps the tail of the previous chunks - enough source samples to fill the kernel window - so an
/// interior output sample sees the same neighbourhood it would have seen inside a whole-buffer call. Only
/// the true start of the stream and the true end (at <see cref="Flush"/>) are treated as edges.
/// </para>
/// <para>
/// ⚠️ The output is EXACT, not approximate: <c>Streaming_MatchesWholeBufferResample</c> asserts
/// bit-equality against the whole-buffer path across chunk sizes that do not divide the frame. That
/// assertion is the point of the class - "close enough" would leave the boundary artifact in place at a
/// smaller amplitude, which is harder to see and just as wrong.
/// </para>
/// <para>
/// The kernel maths below MIRRORS <see cref="AudioPreprocessor.Resample(float[], int, int)"/> deliberately,
/// including the phase table and the renormalisation. If that one changes, this must change with it or the
/// equality test fails - which is the intended alarm, not an inconvenience.
/// </para>
/// </remarks>
public sealed class StreamingResampler
{
    private readonly int _srcRate;
    private readonly int _dstRate;
    private readonly bool _passthrough;

    private readonly double _ratio;
    private readonly double _halfWidth;
    private readonly double _cutoff;

    // Phase table, when the reduced rate pair is small enough to be worth one (48k->16k reduces to a
    // SINGLE phase; 44.1k->16k to 160). Null means evaluate the kernel per output sample.
    private readonly float[][]? _kernels;
    private readonly int[]? _offsets;
    private readonly int _phaseStride;
    private readonly int _phaseCount;

    // Source samples still needed by some future output window. _bufferStart is the absolute index of
    // _buffer[0] in the stream, so windows can be addressed absolutely and old audio dropped.
    private readonly List<float> _buffer = new();
    private long _bufferStart;
    private long _srcSeen;
    private long _outIndex;
    private bool _finished;

    /// <summary>New converter from <paramref name="srcRate"/> to <paramref name="dstRate"/>.</summary>
    public StreamingResampler(int srcRate, int dstRate)
    {
        if (srcRate <= 0) throw new ArgumentOutOfRangeException(nameof(srcRate));
        if (dstRate <= 0) throw new ArgumentOutOfRangeException(nameof(dstRate));

        _srcRate = srcRate;
        _dstRate = dstRate;
        _passthrough = srcRate == dstRate;
        if (_passthrough) return;

        _ratio = (double)dstRate / srcRate;
        _cutoff = Math.Min(1.0, _ratio);
        _halfWidth = AudioPreprocessor.ResampleLobes / _cutoff;

        int g = Gcd(srcRate, dstRate);
        _phaseStride = srcRate / g;
        _phaseCount = dstRate / g;

        if (_phaseCount <= AudioPreprocessor.MaxResamplePhases)
        {
            _kernels = new float[_phaseCount][];
            _offsets = new int[_phaseCount];
            for (int r = 0; r < _phaseCount; r++)
            {
                double c = (double)r * _phaseStride / _phaseCount;
                int f = (int)Math.Ceiling(c - _halfWidth);
                int l = (int)Math.Floor(c + _halfWidth);
                _offsets[r] = f;
                var k = new float[l - f + 1];
                for (int t = 0; t < k.Length; t++)
                {
                    double dt = c - (f + t);
                    k[t] = (float)(AudioPreprocessor.Sinc(_cutoff * dt)
                                 * AudioPreprocessor.BlackmanWindow(dt / _halfWidth));
                }
                _kernels[r] = k;
            }
        }
    }

    /// <summary>Source sample rate.</summary>
    public int SourceRate => _srcRate;

    /// <summary>Destination sample rate.</summary>
    public int DestinationRate => _dstRate;

    /// <summary>Source samples accepted so far.</summary>
    public long SourceSamplesSeen => _srcSeen;

    /// <summary>Output samples emitted so far.</summary>
    public long OutputSamplesEmitted => _outIndex;

    /// <summary>
    /// Accept the next chunk and return whatever output is now fully determined.
    /// </summary>
    /// <remarks>
    /// Output lags the input by about half a kernel window - the converter will not emit a sample whose
    /// window runs past the audio it has been given, because doing so is the truncation this class exists
    /// to avoid. At 48 kHz that lag is well under a millisecond, which no endpointer can perceive.
    /// </remarks>
    public float[] Process(float[] chunk)
    {
        if (_finished) throw new InvalidOperationException("the stream was already flushed");
        if (chunk == null || chunk.Length == 0) return Array.Empty<float>();
        if (_passthrough) { _srcSeen += chunk.Length; _outIndex += chunk.Length; return chunk; }

        _buffer.AddRange(chunk);
        _srcSeen += chunk.Length;
        return Drain(final: false);
    }

    /// <summary>
    /// Close the stream: emits the tail, treating the end of what has been fed as the end of the signal.
    /// </summary>
    public float[] Flush()
    {
        if (_finished) return Array.Empty<float>();
        _finished = true;
        if (_passthrough) return Array.Empty<float>();
        return Drain(final: true);
    }

    /// <summary>Drop all state and start a new stream.</summary>
    public void Reset()
    {
        _buffer.Clear();
        _bufferStart = 0;
        _srcSeen = 0;
        _outIndex = 0;
        _finished = false;
    }

    private float[] Drain(bool final)
    {
        // The whole-buffer path produces (int)(N * ratio) samples for N inputs. Once the signal is closed
        // that count is known, and it is the ONLY thing that bounds the tail.
        long limit = final ? (long)(_srcSeen * _ratio) : long.MaxValue;
        var output = new List<float>();

        while (_outIndex < limit)
        {
            int start, taps;
            float[]? kernel = null;
            double center = 0;

            if (_kernels != null)
            {
                long q = _outIndex / _phaseCount;
                int r = (int)(_outIndex - q * _phaseCount);
                kernel = _kernels[r];
                start = checked((int)(q * _phaseStride + _offsets![r]));
                taps = kernel.Length;
            }
            else
            {
                center = _outIndex / _ratio;
                start = (int)Math.Ceiling(center - _halfWidth);
                taps = (int)Math.Floor(center + _halfWidth) - start + 1;
            }

            long last = (long)start + taps - 1;
            // Not final and the window runs past what we have: stop, and wait for more audio. Emitting
            // here is exactly the truncation-at-a-fake-edge this class exists to prevent.
            if (!final && last >= _srcSeen) break;

            // ⚠️ THE ARITHMETIC MUST MATCH THE WHOLE-BUFFER PATH EXACTLY, INCLUDING ITS PRECISION.
            // The table path there multiplies `samples[j] * k[t]` - float by float, a SINGLE-precision
            // multiply whose result is then added to a double accumulator. Hoisting the weight into a
            // `double` here promotes that to a double-precision multiply, which is MORE accurate and
            // therefore still wrong: it produced 0.37728542 against the whole-buffer 0.37728545 and the
            // equality gate failed on all six backends. "More accurate" is not "identical", and identical
            // is the property this class promises. The two branches are written out separately so each
            // mirrors its counterpart's types rather than sharing a widened one.
            double acc = 0, norm = 0;
            if (kernel != null)
            {
                for (int t = 0; t < taps; t++)
                {
                    long j = start + t;
                    if (j < 0 || j >= _srcSeen) continue;   // the real signal edges, clamped as the
                    acc += _buffer[(int)(j - _bufferStart)] * kernel[t];   // whole-buffer path clamps them
                    norm += kernel[t];
                }
            }
            else
            {
                for (int t = 0; t < taps; t++)
                {
                    long j = start + t;
                    if (j < 0 || j >= _srcSeen) continue;
                    double w = AudioPreprocessor.Sinc(_cutoff * (center - j))
                             * AudioPreprocessor.BlackmanWindow((center - j) / _halfWidth);
                    acc += _buffer[(int)(j - _bufferStart)] * w;
                    norm += w;
                }
            }

            output.Add(norm > 1e-9 ? (float)(acc / norm) : 0f);
            _outIndex++;
        }

        TrimBuffer();
        return output.ToArray();
    }

    /// <summary>Drop source audio no future output window can reach.</summary>
    private void TrimBuffer()
    {
        long keepFrom;
        if (_kernels != null)
        {
            long q = _outIndex / _phaseCount;
            int r = (int)(_outIndex - q * _phaseCount);
            keepFrom = q * _phaseStride + _offsets![r];
        }
        else
        {
            keepFrom = (long)Math.Ceiling(_outIndex / _ratio - _halfWidth);
        }
        if (keepFrom < 0) keepFrom = 0;

        long drop = keepFrom - _bufferStart;
        if (drop <= 0) return;
        if (drop > _buffer.Count) drop = _buffer.Count;
        _buffer.RemoveRange(0, (int)drop);
        _bufferStart += drop;
    }

    private static int Gcd(int a, int b)
    {
        while (b != 0) { (a, b) = (b, a % b); }
        return a < 0 ? -a : a;
    }
}
