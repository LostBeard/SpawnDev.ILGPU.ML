using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Graph;
using SpawnDev.ILGPU.ML.Tensors;

namespace SpawnDev.ILGPU.ML.Pipelines;

/// <summary>
/// ZipVoice's three graphs run on our own engine, on whichever accelerator the app chose.
/// </summary>
/// <remarks>
/// This is the point of the port: the flow-matching decoder is evaluated once per Euler step and is by
/// far the most expensive thing in the pipeline, so it is the piece that wants a GPU. Everything else
/// here is bookkeeping around <see cref="InferenceSession"/>.
/// <para>
/// The graphs declare integer and scalar inputs - token ids are int64 and t, speed and the guidance
/// scale are 0-d tensors. This engine stores every tensor as float32, and a 0-d shape is an empty
/// dimension array rather than a length-1 one; both matter, because a scalar passed as shape [1] would
/// broadcast differently inside the graph than the scalar the exporter declared.
/// </para>
/// </remarks>
public sealed class IlgpuZipVoiceGraphs : IZipVoiceGraphs
{
    private readonly InferenceSession _encoder;
    private readonly InferenceSession _decoder;
    private readonly InferenceSession _vocoder;
    private readonly Accelerator _accelerator;
    private readonly bool _ownsSessions;

    /// <summary>
    /// Wrap three already-loaded sessions.
    /// </summary>
    /// <param name="ownsSessions">
    /// Whether disposing this object should dispose the sessions. The accelerator is NEVER disposed here
    /// - it belongs to the application.
    /// </param>
    public IlgpuZipVoiceGraphs(
        InferenceSession encoder, InferenceSession decoder, InferenceSession vocoder,
        Accelerator accelerator, bool ownsSessions = true)
    {
        _encoder = encoder ?? throw new ArgumentNullException(nameof(encoder));
        _decoder = decoder ?? throw new ArgumentNullException(nameof(decoder));
        _vocoder = vocoder ?? throw new ArgumentNullException(nameof(vocoder));
        _accelerator = accelerator ?? throw new ArgumentNullException(nameof(accelerator));
        _ownsSessions = ownsSessions;

        // ⚠️ THE DECODER IS A DECODE LOOP, which is exactly what this flag is for - and it was never turned
        // on here. MEASURED on WebGPU 2026-09-03, one whole synthesis: 6 graph runs, **1,905 readbacks
        // costing 18,657 ms** against a 42 s total. Nearly half the time is GPU->host round trips for
        // <=64-element shape and scalar tensors that a shape-specialised executor re-reads every single
        // call.
        //
        // ⚠️ That number was invisible until today because the pipeline reported `LastRun*` counters, which
        // are overwritten by the next RunAsync - so the line that read "0 readbacks" was describing the
        // VOCODER, the last graph to run, and not the decoder at all.
        //
        // Correct by construction rather than by assumption: the cache probes the first two runs with full
        // readback and caches ONLY values that were IDENTICAL across them, re-reading the rest every call.
        // The Euler steps feed genuinely different `t` and `x`, so a data-derived value cannot be mistaken
        // for a stable one - which is precisely the mistake to fear here.
        _decoder.CacheShapeReadbacks = true;
    }

    /// <summary>Load the three graphs from raw ONNX bytes.</summary>
    public static IlgpuZipVoiceGraphs Create(
        Accelerator accelerator, byte[] encoderOnnx, byte[] decoderOnnx, byte[] vocoderOnnx) =>
        new(InferenceSession.CreateFromFile(accelerator, encoderOnnx),
            InferenceSession.CreateFromFile(accelerator, decoderOnnx),
            InferenceSession.CreateFromFile(accelerator, vocoderOnnx),
            accelerator);

    /// <summary>
    /// Capture-once/replay-many for the flow-matching decoder.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The decoder runs <c>NumSteps</c> times per utterance at IDENTICAL shapes - only the contents of
    /// <c>t</c> and <c>x</c> change - which is exactly the shape capture/replay wants. It is also where the
    /// time is: after the whole-tensor reduction fix, a synthesis is 8.1 s of which the decoder is 5.4 s,
    /// and only ~3.1 s of the 8.2 s is inside any node's Execute. The remaining ~62% is per-node HOST work
    /// - shape interpretation, pool churn, dispatch setup - which is precisely what a recorded plan skips.
    /// </para>
    /// <para>
    /// ⚠️ A changing scalar input is carried correctly at the INPUT boundary:
    /// <see cref="SessionGraphCapture"/> owns stable input buffers and <c>ReplayAsync</c> copies each
    /// call's tensors into them, so <c>t</c> advancing per Euler step does reach the plan. The hazard is
    /// quieter - a small tensor promoted to a runtime constant can have its dispatch ELIDED, and then stays
    /// frozen at its capture-time value forever. That failure produces confident, plausible audio, so it
    /// cannot be caught by listening; only rendering with and without capture and comparing SAMPLES finds it.
    /// <para>
    /// 🔴 THIS DEFECT IS REAL AND CURRENTLY OPEN. An older note here read "Done: bit-identical", which was
    /// never true of this graph - capture on this decoder was refused for control flow and so had never
    /// actually run. The first time it did run, MEASURED 2026-09-03:
    /// <b>all 73,216 samples differed, worst 0.502935</b> - a completely different utterance, still fluent
    /// enough to pass every amplitude, RMS and duration check in the gate. Replaying this plan is NOT
    /// faithful yet, which is why <see cref="AllowControlFlowCapture"/> defaults OFF.
    /// </para>
    /// </para>
    /// <para>
    /// Capture is best-effort - only CUDA and WebGPU are eligible, and any failure falls through to the
    /// direct forward rather than failing generation.
    /// </para>
    /// </remarks>
    private SessionGraphCapture? _decoderCapture;
    private string? _decoderCaptureShape;

    /// <summary>Enable capture/replay of the decoder. Off disables it entirely (plain RunAsync).</summary>
    public bool EnableGraphCapture { get; set; } = true;

    /// <summary>
    /// Let the decoder be captured even though its graph contains an <c>If</c>.
    /// </summary>
    /// <remarks>
    /// <para>
    /// ⚠️ Without this the decoder is NEVER captured, on any backend, and that is the single largest cost
    /// in a spoken reply. MEASURED on WebGPU 2026-09-03: <c>capture LIVE: False - refused: graph contains
    /// control flow (If)</c>, with the decoder taking 8.4 s per Euler step x 4 steps - 82% of a 48.6 s
    /// synthesis. <c>SubgraphRunner</c>'s own remarks put the refusal's browser cost at ~20x, because a
    /// replayed plan does in microseconds the per-node interop crossings the walk does in ~4.5 ms.
    /// </para>
    /// <para>
    /// ⚠️ THE HISTORY MATTERS, because the first attempt at this was wrong in an instructive way. Turning
    /// it on naively on 2026-09-03 <b>HUNG THE GPU on the first attempt</b> -
    /// <c>DXGI_ERROR_DEVICE_HUNG (0x887A0006)</c>, a D3D12 TDR that resets the display driver - even though
    /// <c>SubgraphRunner</c> caches its compiled plans and <c>WebGPUGraphCapture</c> proves its pool is
    /// primed. Those two facts read like enough. They were not.
    /// </para>
    /// <para>
    /// What settled it was a measurement rather than an argument. With the branch body skipped entirely
    /// (a deliberate diagnostic producing wrong audio), the same decoder recorded <b>8,197 dispatches with
    /// no device hang</b>, and its steady-state Euler step went <b>8,578 ms -> 147 ms (58x)</b>. So it is
    /// executing a BODY inside the recording window that is fatal, not the presence of an If.
    /// </para>
    /// <para>
    /// ⚠️ WHAT IS NOW SOLVED, AND WHAT IS NOT. Capture is REACHABLE: a branch that is a single
    /// <c>Constant</c> is written directly and never reaches SubgraphRunner - a census over a real
    /// utterance says <c>then=21, else=0</c>, so all five of this decoder's Ifs take exactly that path -
    /// and <c>SessionGraphCapture</c> no longer refuses on sight, it runs one forward with the body counter
    /// zeroed and records only if NOTHING ran. The device survives and the plan records 8,197 dispatches.
    /// <para>
    /// 🔴 It still defaults OFF because REPLAY IS NOT FAITHFUL - see the decoder-capture remarks above.
    /// Speed was never the blocker; correctness is. Turning this on today would ship confident, wrong
    /// speech. It flips to on the day the sample-level A/B in
    /// <c>Pipeline_ZipVoice_SpeaksInTheBrowser</c> passes, and not before.
    /// </para>
    /// <para>
    /// ⚠️ The condition here is shape-determined - every leaf of its subtree is an initializer or a
    /// <c>Shape</c> (checked with <c>tools/probe-control-flow.cs</c>, not assumed) - which is what makes a
    /// per-shape capture valid: the branch choice cannot change while the plan is. A data-dependent
    /// condition would not qualify, and that is the caller's obligation, not something capture can verify.
    /// </para>
    /// </remarks>
    public bool AllowControlFlowCapture { get; set; }

    /// <summary>
    /// Whether a decoder capture is actually LIVE - i.e. calls are replaying a recorded plan.
    /// </summary>
    /// <remarks>
    /// ⚠️ "Enabled" is a request, not an outcome. <see cref="SessionGraphCapture"/> falls through to
    /// the direct forward when capture is ineligible (non-CUDA/WebGPU) or when TryCapture returns null -
    /// and the null path prints NOTHING. Without this property a caller measuring "capture on" cannot tell
    /// a replay from a plain run, and would happily report that capture "did not help" when it never
    /// engaged at all.
    /// </remarks>
    public bool DecoderCaptured => _decoderCapture?.IsCaptured ?? false;

    /// <summary>WHY the decoder capture is or is not live - see <see cref="SessionGraphCapture.CaptureStatus"/>.</summary>
    public string DecoderCaptureStatus => _decoderCapture?.CaptureStatus ?? "no capture constructed yet";

    /// <summary>
    /// Run one of the three graphs, naming WHICH one - and with what input shapes - if it throws.
    /// </summary>
    /// <remarks>
    /// ⚠️ Without this, a shape error inside any of the three surfaces only as the failing operator
    /// ("Shapes [106,432] and [106,432,1] are not broadcastable"), with nothing to say whether the encoder,
    /// the decoder or the vocoder produced it. Three graphs deep, that is the difference between a bug
    /// report you can act on and one you have to re-derive by bisection.
    /// </remarks>
    private static async Task<Dictionary<string, Tensor>> RunStageAsync(
        InferenceSession session, string stage, Dictionary<string, Tensor> inputs)
    {
        try
        {
            return await session.RunAsync(inputs);
        }
        catch (Exception ex)
        {
            var shapes = string.Join(", ", inputs.Select(kv =>
                $"{kv.Key}[{string.Join(",", kv.Value.Shape)}]"));
            throw new InvalidOperationException(
                $"ZipVoice {stage} graph failed with inputs {shapes}: {ex.Message}", ex);
        }
    }

    public async Task<ZipVoiceEncoding> RunEncoderAsync(
        long[] tokens, long[] promptTokens, long promptFeatureFrames, float speed)
    {
        // Token ids ride as float32 like every other tensor in this engine. They are small integers -
        // the vocabulary is 360 - so float32 represents them exactly and nothing is lost.
        using var tokenBuffer = _accelerator.Allocate1D(ToFloats(tokens));
        using var promptBuffer = _accelerator.Allocate1D(ToFloats(promptTokens));
        using var lengthBuffer = _accelerator.Allocate1D(new[] { (float)promptFeatureFrames });
        using var speedBuffer = _accelerator.Allocate1D(new[] { speed });

        var inputs = new Dictionary<string, Tensor>
        {
            ["tokens"] = new Tensor(tokenBuffer.View, new[] { 1, tokens.Length }),
            ["prompt_tokens"] = new Tensor(promptBuffer.View, new[] { 1, promptTokens.Length }),
            ["prompt_features_len"] = new Tensor(lengthBuffer.View, Array.Empty<int>()),
            ["speed"] = new Tensor(speedBuffer.View, Array.Empty<int>()),
        };

        var outputs = await RunStageAsync(_encoder, "ENCODER", Rename(inputs, _encoder));
        var condition = outputs[_encoder.OutputNames[0]];

        // [1, frames, features] - the frame count is the encoder's duration prediction, and everything
        // downstream is shaped by it.
        int numFrames = condition.Shape[^2];
        int featDim = condition.Shape[^1];
        return new ZipVoiceEncoding(await ReadAsync(condition), numFrames, featDim);
    }

    public async Task<float[]> RunDecoderAsync(
        float t, float[] x, float[] textCondition, float[] speechCondition,
        float guidanceScale, int numFrames, int featDim)
    {
        var shape = new[] { 1, numFrames, featDim };

        using var tBuffer = _accelerator.Allocate1D(new[] { t });
        using var xBuffer = _accelerator.Allocate1D(x);
        using var textBuffer = _accelerator.Allocate1D(textCondition);
        using var speechBuffer = _accelerator.Allocate1D(speechCondition);
        using var guidanceBuffer = _accelerator.Allocate1D(new[] { guidanceScale });

        var inputs = new Dictionary<string, Tensor>
        {
            ["t"] = new Tensor(tBuffer.View, Array.Empty<int>()),
            ["x"] = new Tensor(xBuffer.View, shape),
            ["text_condition"] = new Tensor(textBuffer.View, shape),
            ["speech_condition"] = new Tensor(speechBuffer.View, shape),
            ["guidance_scale"] = new Tensor(guidanceBuffer.View, Array.Empty<int>()),
        };

        // ⚠️ ONE capture per UTTERANCE, not one per graphs instance. A captured graph is valid only at the
        // shapes it recorded, and this decoder's shape is the utterance length - so a capture taken on
        // "hello" is worthless for the next sentence and a single long-lived instance would either refuse
        // it or replay the wrong thing.
        //
        // Within one utterance the shape is FIXED across all NumSteps Euler steps, which is exactly where
        // capture/replay pays: record step 1, replay the rest. So the capture is rebuilt when the shape
        // changes, which in practice means once per thing said.
        var shapeKey = string.Join("x", shape);
        if (_decoderCaptureShape != shapeKey)
        {
            _decoderCapture?.Dispose();
            _decoderCapture = new SessionGraphCapture(_decoder, _accelerator);
            _decoderCaptureShape = shapeKey;
        }
        _decoderCapture.Enabled = EnableGraphCapture;
        _decoderCapture.AllowControlFlow = AllowControlFlowCapture;
        var renamed = Rename(inputs, _decoder);
        Dictionary<string, Tensor> outputs;
        try
        {
            outputs = await _decoderCapture.RunAsync(renamed);
        }
        catch (Exception ex)
        {
            var shapes = string.Join(", ", renamed.Select(kv => $"{kv.Key}[{string.Join(",", kv.Value.Shape)}]"));
            throw new InvalidOperationException(
                $"ZipVoice DECODER graph failed with inputs {shapes}: {ex.Message}", ex);
        }
        return await ReadAsync(outputs[_decoder.OutputNames[0]]);
    }

    public async Task<ZipVoiceSpectrum> RunVocoderAsync(float[] melChannelsFirst, int channels, int frames)
    {
        using var melBuffer = _accelerator.Allocate1D(melChannelsFirst);
        var inputs = new Dictionary<string, Tensor>
        {
            ["mels"] = new Tensor(melBuffer.View, new[] { 1, channels, frames }),
        };

        var outputs = await RunStageAsync(_vocoder, "VOCODER", Rename(inputs, _vocoder));

        // Three outputs, and their ORDER is not something to assume - "mag" is the magnitude and "x"/"y"
        // are the cosine and sine of the phase, so picking them by name is the only safe read.
        var magnitude = Output(outputs, _vocoder, "mag", 0);
        var cos = Output(outputs, _vocoder, "x", 1);
        var sin = Output(outputs, _vocoder, "y", 2);

        int bins = magnitude.Shape[^2];
        int outFrames = magnitude.Shape[^1];

        return new ZipVoiceSpectrum(
            await ReadAsync(magnitude), await ReadAsync(cos), await ReadAsync(sin), bins, outFrames);
    }

    /// <summary>
    /// Map inputs onto whatever the session actually calls them, falling back to declaration order.
    /// </summary>
    /// <remarks>
    /// The names above are the ones the sherpa-onnx exports use. Keying by name keeps the code readable
    /// and order-independent, but a re-export with different names should degrade to positional binding
    /// rather than throwing a KeyNotFound from deep inside the executor.
    /// </remarks>
    private static Dictionary<string, Tensor> Rename(Dictionary<string, Tensor> inputs, InferenceSession session)
    {
        var names = session.InputNames;
        if (names.All(inputs.ContainsKey)) return inputs;

        var mapped = new Dictionary<string, Tensor>();
        var ordered = inputs.Values.ToArray();
        for (int i = 0; i < names.Length && i < ordered.Length; i++) mapped[names[i]] = ordered[i];
        return mapped;
    }

    private static Tensor Output(Dictionary<string, Tensor> outputs, InferenceSession session, string name, int index)
    {
        if (outputs.TryGetValue(name, out var byName)) return byName;
        return outputs[session.OutputNames[index]];
    }

    private static async Task<float[]> ReadAsync(Tensor tensor)
    {
        var host = new float[tensor.ElementCount];
        var data = await SpawnDev.ILGPU.SpawnDevContextExtensions.CopyToHostAsync<float>(tensor.Data);
        Array.Copy(data, host, Math.Min(data.Length, host.Length));
        return host;
    }

    private static float[] ToFloats(long[] values)
    {
        var floats = new float[values.Length];
        for (int i = 0; i < values.Length; i++) floats[i] = values[i];
        return floats;
    }

    public void Dispose()
    {
        _decoderCapture?.Dispose(); _decoderCapture = null; _decoderCaptureShape = null;
        if (!_ownsSessions) return;
        _encoder.Dispose();
        _decoder.Dispose();
        _vocoder.Dispose();
        // The accelerator is the application's. Disposing it here would take down every other session
        // sharing the device.
    }
}
