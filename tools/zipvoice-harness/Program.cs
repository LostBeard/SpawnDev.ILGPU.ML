using SpawnDev.ILGPU.ML.Tensors;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.ILGPU.ML;
// ZipVoice (zero-shot voice cloning TTS) desktop gate.
//
//   dotnet run --project tools/zipvoice-harness -c Release -- roundtrip [wav]
//
// WHY THIS EXISTS: ZipVoice is four separable pieces - mel features, a text encoder, a flow-matching
// decoder, and a vocoder whose last stage is an inverse STFT we write ourselves. Debugging them
// together, in a browser, from a robot's speaker, would make any one of them able to explain a bad
// result. Each command here isolates one piece and gives it a pass/fail an eye cannot argue with.
//
// ROUNDTRIP is the first gate and needs no tokenizer at all: take a real recording, compute the mel
// exactly as the model expects it, hand that mel to the REAL vocos vocoder, resynthesise, and compare.
// It is a fixed-point test - only a mel built to the right convention survives the loop unchanged, so a
// wrong window, mel scale, normalisation or magnitude-vs-power choice shows up as drift rather than as
// audio nobody can grade. The inverse STFT is on trial in the same pass, since it is what turns the
// vocoder's magnitude and phase back into sound.
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.ILGPU.ML.Preprocessing;
using ZipVoiceHarness;

var modelDir = Environment.GetEnvironmentVariable("ZIPVOICE_MODEL_DIR")
    ?? @"D:\users\tj\Projects\SpawnDev.Reachy\SpawnDev.Reachy\models\sherpa-onnx-zipvoice-distill-zh-en-emilia";

var command = args.Length > 0 ? args[0].ToLowerInvariant() : "roundtrip";

return command switch
{
    "roundtrip" => RoundTrip(args.Length > 1 ? args[1] : Path.Combine(modelDir, "prompt.wav")),
    "synth" => Synth(args.Length > 1 ? args[1] : "fixtures/paint-the-sockets.json",
                     args.Length > 2 ? args[2] : null),
    "compare" => CompareEngines(args.Length > 1 ? args[1] : "fixtures/paint-the-sockets.json"),
    "runonnx" => RunOnnx(args.Length > 1 ? args[1] : "", args.Length > 2 ? args[2] : ""),
    "sensitivity" => RunSensitivity(args.Length > 1 ? args[1] : "fixtures/loaded-classes.json",
                                    args.Length > 2 ? args[2] : null),
    "endtoend" => RunEndToEnd(args.Length > 1 ? args[1] : "fixtures/phase1",
                              args.Length > 2 ? args[2] : null),
    "verify" => RunVerify(args.Length > 1 ? args[1] : "fixtures/phase1",
                          args.Length > 2 ? args[2] : null),
    "trimsweep" => RunTrimSweep(args.Length > 1 ? args[1] : "fixtures/phase1",
                                args.Length > 2 ? args[2] : null),
    _ => Usage(command),
};

int Usage(string bad)
{
    Console.WriteLine($"unknown command '{bad}'. commands: roundtrip [wav] | synth [fixture.json] [out.wav] "
                    + "| compare [fixture.json] | sensitivity [dir] [outDir] | endtoend [dir] [outDir] | verify [dir] [outDir] | trimsweep [dir] [outDir]");
    return 2;
}

// Where should the generated audio be cut away from the reference the model regenerates ahead of it?
// See TrimSweep.cs. Nearly every render opens with a few words of the cloned voice, and the fix is a cut
// point that has to be measured: too little leaves the preamble, too much eats the first word.
int RunTrimSweep(string fixturePath, string? outDir)
{
    var loaded = LoadFixtures(fixturePath);
    if (loaded == null) return 2;
    outDir ??= Path.Combine(Path.GetTempPath(), "zipvoice-trimsweep");
    return TrimSweep.RunAsync(modelDir, loaded, outDir, ReadWav, WriteWav).GetAwaiter().GetResult();
}

// Does listening to your own output rescue a garbled render? See VerifyRun.cs. ZipVoice produces garbage
// on some noise draws - both frontends - and the only way to see it from inside the stack is to transcribe
// the result with the recogniser already present and re-roll when the words come back wrong.
int RunVerify(string fixturePath, string? outDir)
{
    var loaded = LoadFixtures(fixturePath);
    if (loaded == null) return 2;
    outDir ??= Path.Combine(Path.GetTempPath(), "zipvoice-verify");
    return VerifyRun.RunAsync(modelDir, loaded, outDir, ReadWav, WriteWav).GetAwaiter().GetResult();
}

// Does OUR phonemizer sound as good as the reference frontend? See EndToEnd.cs. Speaks each sentence
// twice from the same voice and the same noise seed - once from the reference token ids, once from ours -
// and transcribes both, so any difference is the phonemizer and nothing else.
int RunEndToEnd(string fixturePath, string? outDir)
{
    var loaded = LoadFixtures(fixturePath);
    if (loaded == null) return 2;
    outDir ??= Path.Combine(Path.GetTempPath(), "zipvoice-endtoend");
    return EndToEnd.RunAsync(modelDir, loaded, outDir, ReadWav, WriteWav).GetAwaiter().GetResult();
}

// How much phonemizer error this model actually tolerates - see Sensitivity.cs and
// Plans/mit-phonemizer-2026-08-27.md. Damages the ground-truth tokens the way a CMUdict-based
// frontend will and grades the resulting audio, so the frontend's precision target is measured
// rather than assumed.
// A directory means "every fixture in it". Replication across sentences is what separates a measurement
// from an anecdote, so the many-fixture case is the normal one.
List<(string Path, ZipVoiceFixture Fixture)>? LoadFixtures(string fixturePath)
{
    if (!Directory.Exists(modelDir)) { Console.WriteLine($"no model dir at {modelDir}"); return null; }

    var candidates = new[]
    {
        Path.IsPathRooted(fixturePath) ? fixturePath : Path.Combine(AppContext.BaseDirectory, fixturePath),
        Path.Combine(Environment.CurrentDirectory, fixturePath),
    };
    var resolved = candidates.FirstOrDefault(p => File.Exists(p) || Directory.Exists(p));
    if (resolved == null) { Console.WriteLine($"no fixture or fixture dir at {fixturePath}"); return null; }

    var paths = Directory.Exists(resolved)
        ? Directory.GetFiles(resolved, "*.json").OrderBy(p => p).ToArray()
        : new[] { resolved };
    if (paths.Length == 0) { Console.WriteLine($"no *.json fixtures in {resolved}"); return null; }

    var loaded = new List<(string Path, ZipVoiceFixture Fixture)>();
    foreach (var p in paths)
    {
        var fixture = ZipVoiceFixture.Load(p);
        var promptWav = Sensitivity.ResolvePromptWav(modelDir, fixture);
        if (!File.Exists(promptWav)) { Console.WriteLine($"no prompt wav at {promptWav} (for {Path.GetFileName(p)})"); return null; }
        loaded.Add((p, fixture));
    }
    return loaded;
}

int RunSensitivity(string fixturePath, string? outDir)
{
    var fixtures = LoadFixtures(fixturePath);
    if (fixtures == null) return 2;

    outDir ??= Path.Combine(Path.GetTempPath(), "zipvoice-sensitivity");
    return Sensitivity.RunAsync(modelDir, fixtures, outDir, ReadWav, WriteWav)
                      .GetAwaiter().GetResult();
}

// Full cloning path on the reference engine: ground-truth tokens in, cloned speech out.
// Runs the SHIPPING orchestration (mel features, flow-matching loop, inverse STFT) with onnxruntime
// executing the graphs, so a bad result here is the algorithm and never the engine.
int Synth(string fixturePath, string? outPath)
{
    if (!Directory.Exists(modelDir)) { Console.WriteLine($"no model dir at {modelDir}"); return 2; }

    var resolved = Path.IsPathRooted(fixturePath)
        ? fixturePath
        : Path.Combine(AppContext.BaseDirectory, fixturePath);
    if (!File.Exists(resolved)) resolved = Path.Combine(Environment.CurrentDirectory, fixturePath);
    if (!File.Exists(resolved)) { Console.WriteLine($"no fixture at {fixturePath}"); return 2; }

    var fixture = ZipVoiceFixture.Load(resolved);
    var promptWav = Path.IsPathRooted(fixture.PromptWav)
        ? fixture.PromptWav
        : Path.Combine(modelDir, fixture.PromptWav);
    if (!File.Exists(promptWav)) { Console.WriteLine($"no prompt wav at {promptWav}"); return 2; }

    var (reference, referenceRate, _) = ReadWav(File.ReadAllBytes(promptWav));

    Console.WriteLine($"text     : {fixture.Text}");
    Console.WriteLine($"promptTxt: {fixture.PromptText}");
    Console.WriteLine($"prompt   : {Path.GetFileName(promptWav)} {referenceRate} Hz, " +
                      $"{reference.Length / (double)referenceRate:F2}s");
    Console.WriteLine($"tokens   : {fixture.Tokens.Length} text, {fixture.PromptTokens.Length} prompt");

    var config = new ZipVoiceConfig();
    // The oracle runs the quantized graphs, so being able to select them is what makes a frame-count
    // or duration difference attributable to precision rather than to our code.
    bool int8 = Environment.GetEnvironmentVariable("ZIPVOICE_INT8") == "1";
    var graphDir = int8
        ? modelDir.Replace("zipvoice-distill-zh-en", "zipvoice-distill-int8-zh-en")
        : modelDir;
    // ZIPVOICE_ENGINE=ilgpu renders through OUR engine instead of onnxruntime. Every render mode in this
    // harness used ORT, so our engine had never produced a finished clip - only stage-by-stage diffs in
    // `compare`. Being able to LISTEN to our own output is what turns "the stages match" into "it speaks".
    bool useOurs = string.Equals(Environment.GetEnvironmentVariable("ZIPVOICE_ENGINE"), "ilgpu",
                                 StringComparison.OrdinalIgnoreCase);
    IZipVoiceGraphs graphs;
    IDisposable? ourAccel = null, ourCtx = null;
    if (useOurs)
    {
        var mlCtxBuilder = MLContext.Create();
        mlCtxBuilder.AllAcceleratorsAsync().GetAwaiter().GetResult();
        var mlCtx = mlCtxBuilder.ToContext();
        // ZIPVOICE_ACCELERATOR=cpu|cuda|opencl pins the backend. Without it this always took the
        // PREFERRED device (a real GPU), so a backend-specific failure could not be reproduced here at
        // all - the CPU lane crashed under PMT and the only tool that runs our engine end-to-end could
        // not be pointed at CPU to find out why.
        var want = Environment.GetEnvironmentVariable("ZIPVOICE_ACCELERATOR");
        Accelerator? accelerator = null;
        if (!string.IsNullOrWhiteSpace(want))
        {
            var match = mlCtx.Devices.FirstOrDefault(d =>
                string.Equals(d.AcceleratorType.ToString(), want, StringComparison.OrdinalIgnoreCase));
            if (match == null)
                throw new InvalidOperationException(
                    $"ZIPVOICE_ACCELERATOR='{want}' matches no device. Available: "
                  + string.Join(", ", mlCtx.Devices.Select(d => d.AcceleratorType.ToString())));
            accelerator = match.CreateAccelerator(mlCtx);
        }
        accelerator ??= mlCtx.CreatePreferredAcceleratorAsync().GetAwaiter().GetResult()
            ?? throw new InvalidOperationException("no accelerator available");
        ourCtx = mlCtx; ourAccel = accelerator;
        var encPath = Path.Combine(graphDir, int8 ? "text_encoder_int8.onnx" : "text_encoder.onnx");
        var decPath = Path.Combine(graphDir, int8 ? "fm_decoder_int8.onnx" : "fm_decoder.onnx");
        var vocPath = Path.Combine(graphDir, "vocos_24khz.onnx");
        graphs = new IlgpuZipVoiceGraphs(
            InferenceSession.CreateFromFile(accelerator, File.ReadAllBytes(encPath)),
            InferenceSession.CreateFromFile(accelerator, File.ReadAllBytes(decPath)),
            InferenceSession.CreateFromFile(accelerator, File.ReadAllBytes(vocPath)),
            accelerator);
        // ZIPVOICE_NO_CAPTURE=1 forces the plain forward. Needed as the CONTROL: a capture that froze an
        // elided dispatch still renders confident, plausible audio, so the only way to detect it is to
        // render both ways and compare samples.
        if (Environment.GetEnvironmentVariable("ZIPVOICE_NO_CAPTURE") == "1")
            ((IlgpuZipVoiceGraphs)graphs).EnableGraphCapture = false;
        // Print the accelerator TYPE, not just the device name. "NVIDIA GeForce RTX 4070" is the same
        // string whether ILGPU reached it through CUDA or OpenCL, and graph capture is CUDA-only - so the
        // name alone cannot tell you why a capture silently did not engage.
        Console.WriteLine($"engine   : ILGPU {accelerator.AcceleratorType} ({accelerator.Name}), decoder capture "
                        + $"{(((IlgpuZipVoiceGraphs)graphs).EnableGraphCapture ? "ON" : "OFF")}");
    }
    else
    {
        graphs = new OrtZipVoiceGraphs(graphDir, int8);
        Console.WriteLine("engine   : onnxruntime");
    }
    Console.WriteLine($"graphs   : {(int8 ? "int8" : "fp32")}");
    // `using var` disposes in REVERSE declaration order, and the sessions own buffers that live on the
    // accelerator - disposing the accelerator first throws inside MemoryBuffer.DisposeAcceleratorObject.
    // Declaring graphs LAST is what makes it disposed FIRST.
    using var _ctxOwner = ourCtx;
    using var _accelOwner = ourAccel;
    using var _graphsOwner = graphs;
    using var pipeline = new ZipVoicePipeline(graphs, config)
    {
        // Fixed so two runs of this gate are comparable; production leaves it null and re-rolls.
        NoiseSeed = 1234,
        // The reference implementation does NOT pad, so comparisons against it turn this off.
        ReferenceTailSilenceSeconds =
            Environment.GetEnvironmentVariable("ZIPVOICE_NO_PAD") == "1" ? 0f : 0.25f,
    };
    Console.WriteLine($"tailPad  : {pipeline.ReferenceTailSilenceSeconds}s");

    // ZIPVOICE_NODE_TIMING=1 attributes the time PER NODE instead of per stage. The stage split says the
    // decoder is ~94% of a synthesis; it does not say whether that is a few heavy kernels (worth tuning) or
    // thousands of cheap ones (worth capture/replay), and those lead to opposite work. Measure before cutting.
    var timingMode = Environment.GetEnvironmentVariable("ZIPVOICE_NODE_TIMING");
    bool nodeTiming = timingMode == "1" || timingMode == "2";
    if (nodeTiming)
    {
        SpawnDev.ILGPU.ML.Graph.GraphExecutor.CapturedNodeTimingsMs = new Dictionary<string, double>();
        // ⚠️ =1 times DISPATCH only. Kernels are async, so a node's Execute returns once the work is
        // QUEUED and the real GPU time lands at the next periodic sync - which is why mode 1 accounted for
        // only 2,097 ms of a 24,601 ms synthesis and proved nothing about where the time goes.
        // =2 adds PerOpSync: a flush+wait after every node, so each measurement includes that node's GPU
        // completion. It is slower in absolute terms and the total is NOT comparable to a normal run - but
        // it is the only form that attributes GPU time per op, which is what decides between tuning a
        // kernel and eliminating orchestration.
        if (timingMode == "2")
        {
            SpawnDev.ILGPU.ML.Graph.GraphExecutor.PerOpSync = true;
            Console.WriteLine("timing   : PerOpSync ON - per-node times include GPU completion; "
                            + "TOTALS ARE INFLATED and only the ATTRIBUTION is meaningful");
        }
    }

    var result = pipeline.SynthesizeAsync(fixture.Tokens, fixture.PromptTokens, reference, referenceRate).GetAwaiter().GetResult();

    if (nodeTiming)
    {
        var timings = SpawnDev.ILGPU.ML.Graph.GraphExecutor.CapturedNodeTimingsMs;
        SpawnDev.ILGPU.ML.Graph.GraphExecutor.CapturedNodeTimingsMs = null;   // static: never leave it armed
        SpawnDev.ILGPU.ML.Graph.GraphExecutor.PerOpSync = false;
        if (timings != null && timings.Count > 0)
        {
            double total = timings.Values.Sum();
            Console.WriteLine($"nodes    : {timings.Count} timed, {total:F0}ms accounted");
            // Per-OP totals answer "which kind of work dominates"; the top individual nodes answer "is it
            // one kernel or the long tail". Both, because either alone can mislead.
            Console.WriteLine("by op    :");
            foreach (var g in timings.GroupBy(kv => kv.Key.Split('_')[1])
                                     .Select(g => (Op: g.Key, Ms: g.Sum(x => x.Value), N: g.Count()))
                                     .OrderByDescending(x => x.Ms).Take(30))
                Console.WriteLine($"           {g.Ms,9:F1}ms  {100 * g.Ms / total,5:F1}%  {g.N,5} x {g.Op}");
            Console.WriteLine("slowest  :");
            foreach (var kv in timings.OrderByDescending(kv => kv.Value).Take(10))
                Console.WriteLine($"           {kv.Value,9:F2}ms  {kv.Key}");
            var cheap = timings.Values.Count(v => v < 1.0);
            Console.WriteLine($"under 1ms: {cheap}/{timings.Count} nodes ({100.0 * cheap / timings.Count:F0}%)");
            // ⚠️ The ACCOUNTED-vs-TOTAL gap is the orchestration signal, not the share of cheap nodes.
            // An earlier version of this line asserted "a high share here means ORCHESTRATION" and printed
            // it unconditionally - including on a run where ONE operator was 61% of GPU time and the real
            // answer was a single pathological kernel. Report the gap and name both readings instead of
            // deciding for the reader.
            //
            // Note the per-node keys are node NAMES, so a graph run more than once (the decoder runs
            // NumSteps times) overwrites its own entries: 'accounted' is roughly ONE pass, not all of them.
            double gap = result.TotalMs - total;
            Console.WriteLine($"unacct'd : {gap:F0}ms of {result.TotalMs:F0}ms total is NOT inside any node's "
                            + "Execute (host-side orchestration between nodes: shape interp, pool churn, "
                            + "dispatch setup). A large gap favours capture/replay; a single dominant op "
                            + "above favours fixing that kernel.");
        }
    }

    if (graphs is IlgpuZipVoiceGraphs ig)
        Console.WriteLine($"capture  : requested={ig.EnableGraphCapture}, LIVE={ig.DecoderCaptured}"
                        + (ig.EnableGraphCapture && !ig.DecoderCaptured
                           ? "  <- capture was requested but did NOT engage; timings below are a plain forward"
                           : ""));
    Console.WriteLine($"timing   : encoder {result.EncoderMs:F0}ms, decoder {result.DecoderMs:F0}ms " +
                      $"({config.NumSteps} steps), vocoder {result.VocoderMs:F0}ms, total {result.TotalMs:F0}ms");
    Console.WriteLine($"audio    : {result.Audio.Length} samples @ {result.SampleRate} Hz " +
                      $"({result.DurationSeconds:F2}s), rms {Rms(result.Audio):F4}), " +
                      $"{result.Audio.Length / config.HopLength + 1} generated frames");

    outPath ??= Path.Combine(Path.GetTempPath(), "zipvoice-ours.wav");
    File.WriteAllBytes(outPath, WriteWav(result.Audio, result.SampleRate));
    Console.WriteLine($"wrote    : {outPath}");

    // Silence and noise are the two ways this fails without throwing, and both are visible in the level.
    bool pass = result.DurationSeconds > 0.5 && Rms(result.Audio) > 0.005 && Rms(result.Audio) < 0.5;
    Console.WriteLine(pass ? "RESULT   : PASS (level plausible - grade the audio by transcribing it)" : "RESULT   : FAIL");
    return pass ? 0 : 1;
}

// Every graph on BOTH engines with identical inputs, stage by stage.
// The orchestration is shared and the inputs are pinned, so any difference reported here is our engine.
int CompareEngines(string fixturePath)
{
    if (!Directory.Exists(modelDir)) { Console.WriteLine($"no model dir at {modelDir}"); return 2; }

    var resolved = ResolveFixture(fixturePath);
    if (resolved == null) { Console.WriteLine($"no fixture at {fixturePath}"); return 2; }

    var fixture = ZipVoiceFixture.Load(resolved);
    var promptWav = Path.IsPathRooted(fixture.PromptWav) ? fixture.PromptWav : Path.Combine(modelDir, fixture.PromptWav);
    if (!File.Exists(promptWav)) { Console.WriteLine($"no prompt wav at {promptWav}"); return 2; }

    var (reference, referenceRate, _) = ReadWav(File.ReadAllBytes(promptWav));
    return Compare.RunAsync(modelDir, fixture, reference, referenceRate).GetAwaiter().GetResult();
}

string? ResolveFixture(string fixturePath)
{
    if (Path.IsPathRooted(fixturePath)) return File.Exists(fixturePath) ? fixturePath : null;
    var candidate = Path.Combine(AppContext.BaseDirectory, fixturePath);
    if (File.Exists(candidate)) return candidate;
    candidate = Path.Combine(Environment.CurrentDirectory, fixturePath);
    return File.Exists(candidate) ? candidate : null;
}

// Reference audio -> our mel -> the real vocoder -> our inverse STFT -> audio.
// A correct mel is a fixed point of that loop; a wrong one drifts.
int RoundTrip(string wavPath)
{
    if (!Directory.Exists(modelDir)) { Console.WriteLine($"no model dir at {modelDir}"); return 2; }
    if (!File.Exists(wavPath)) { Console.WriteLine($"no wav at {wavPath}"); return 2; }

    // The vocoder consumes UNSCALED log-mels. The model's 0.1 feature scale belongs to the encoder and
    // decoder, not to it, so it is left out here rather than applied and undone.
    var config = new ZipVoiceConfig { FeatScale = 1.0f };

    var (samples, sampleRate, channels) = ReadWav(File.ReadAllBytes(wavPath));
    Console.WriteLine($"wav      : {Path.GetFileName(wavPath)}  {sampleRate} Hz, {channels} ch, " +
                      $"{samples.Length} samples ({samples.Length / (double)sampleRate:F2}s)");

    var mel = ZipVoiceFeatures.ComputeMel(samples, sampleRate, config, out int frames);
    Console.WriteLine($"mel      : {frames} frames x {config.NumMels} mels, " +
                      $"range [{mel.Min():F3}, {mel.Max():F3}], mean {mel.Average():F3}");

    // Vocos reads [channels, frames]; our features are [frames, channels].
    var melChannelsFirst = new float[config.NumMels * frames];
    for (int f = 0; f < frames; f++)
        for (int c = 0; c < config.NumMels; c++)
            melChannelsFirst[c * frames + f] = mel[f * config.NumMels + c];

    // The oracle runs the quantized graphs, so being able to select them is what makes a frame-count
    // or duration difference attributable to precision rather than to our code.
    bool int8 = Environment.GetEnvironmentVariable("ZIPVOICE_INT8") == "1";
    var graphDir = int8
        ? modelDir.Replace("zipvoice-distill-zh-en", "zipvoice-distill-int8-zh-en")
        : modelDir;
    using var graphs = new OrtZipVoiceGraphs(graphDir, int8);
    Console.WriteLine($"graphs   : {(int8 ? "int8" : "fp32")}");
    var started = System.Diagnostics.Stopwatch.StartNew();
    var spectrum = graphs.RunVocoderAsync(melChannelsFirst, config.NumMels, frames).GetAwaiter().GetResult();
    var audio = ZipVoicePipeline.Vocode(spectrum, config);
    started.Stop();

    Console.WriteLine($"vocoder  : {spectrum.Bins} bins x {spectrum.Frames} frames -> " +
                      $"{audio.Length} samples ({audio.Length / (double)config.SampleRate:F2}s) in {started.ElapsedMilliseconds}ms");

    // The vocoder PREDICTS phase rather than preserving it, so the resynthesised waveform is not
    // sample-aligned with the original and comparing waveforms directly would fail on correct output.
    // The mel is the thing that must survive.
    var melAgain = ZipVoiceFeatures.ComputeMel(audio, config.SampleRate, config, out int framesAgain);
    int common = Math.Min(frames, framesAgain);

    double sumAbs = 0, maxAbs = 0;
    for (int f = 0; f < common; f++)
    {
        for (int c = 0; c < config.NumMels; c++)
        {
            double diff = Math.Abs(mel[f * config.NumMels + c] - melAgain[f * config.NumMels + c]);
            sumAbs += diff;
            if (diff > maxAbs) maxAbs = diff;
        }
    }
    double meanAbs = sumAbs / (common * config.NumMels);

    double rmsIn = Rms(samples), rmsOut = Rms(audio);
    Console.WriteLine($"levels   : rms in {rmsIn:F4}, out {rmsOut:F4}  (ratio {rmsOut / Math.Max(rmsIn, 1e-9):F3})");
    Console.WriteLine($"mel drift: mean |d| {meanAbs:F4} nats, max |d| {maxAbs:F3}, frames {frames} -> {framesAgain}");

    var outPath = Path.Combine(Path.GetDirectoryName(Path.GetFullPath(wavPath))!,
                               Path.GetFileNameWithoutExtension(wavPath) + ".roundtrip.wav");
    File.WriteAllBytes(outPath, WriteWav(audio, config.SampleRate));
    Console.WriteLine($"wrote    : {outPath}");

    // A log-mel is in nats, so 0.25 is roughly a 28% error in magnitude on an average bin - loose
    // enough to allow the vocoder's own reconstruction error, tight enough that a wrong mel convention
    // (which moves whole bands, not single bins) cannot pass.
    bool pass = meanAbs < 0.25 && framesAgain >= frames - 1 && rmsOut > rmsIn * 0.5 && rmsOut < rmsIn * 2.0;
    Console.WriteLine(pass ? "RESULT   : PASS" : "RESULT   : FAIL");
    return pass ? 0 : 1;
}

static double Rms(float[] x)
{
    if (x.Length == 0) return 0;
    double sum = 0;
    foreach (var v in x) sum += (double)v * v;
    return Math.Sqrt(sum / x.Length);
}

static (float[] Samples, int SampleRate, int Channels) ReadWav(byte[] data)
{
    if (data.Length < 44) throw new InvalidDataException("too short to be a WAV");
    if (data[0] != 'R' || data[1] != 'I' || data[2] != 'F' || data[3] != 'F') throw new InvalidDataException("not RIFF");
    if (data[8] != 'W' || data[9] != 'A' || data[10] != 'V' || data[11] != 'E') throw new InvalidDataException("not WAVE");

    int pos = 12, channels = 1, bits = 16, rate = 16000;
    while (pos + 8 <= data.Length)
    {
        var id = System.Text.Encoding.ASCII.GetString(data, pos, 4);
        int size = BitConverter.ToInt32(data, pos + 4);
        int body = pos + 8;
        if (id == "fmt ")
        {
            channels = BitConverter.ToInt16(data, body + 2);
            rate = BitConverter.ToInt32(data, body + 4);
            bits = BitConverter.ToInt16(data, body + 14);
        }
        else if (id == "data")
        {
            if (bits != 16) throw new NotSupportedException($"{bits}-bit WAV not supported (16-bit PCM only)");
            int count = Math.Min(size, data.Length - body) / 2;
            int frames = count / Math.Max(1, channels);
            var outp = new float[frames];
            for (int f = 0; f < frames; f++)
            {
                int sum = 0;
                for (int c = 0; c < channels; c++)
                    sum += BitConverter.ToInt16(data, body + (f * channels + c) * 2);
                outp[f] = sum / (float)channels / 32768f;
            }
            return (outp, rate, channels);
        }
        pos = body + size + (size & 1);   // chunks are word-aligned
    }
    throw new InvalidDataException("no data chunk");
}

static byte[] WriteWav(float[] samples, int sampleRate)
{
    using var stream = new MemoryStream();
    using var writer = new BinaryWriter(stream);
    int dataBytes = samples.Length * 2;

    writer.Write("RIFF"u8.ToArray());
    writer.Write(36 + dataBytes);
    writer.Write("WAVE"u8.ToArray());
    writer.Write("fmt "u8.ToArray());
    writer.Write(16);
    writer.Write((short)1);              // PCM
    writer.Write((short)1);              // mono
    writer.Write(sampleRate);
    writer.Write(sampleRate * 2);        // byte rate
    writer.Write((short)2);              // block align
    writer.Write((short)16);             // bits per sample
    writer.Write("data"u8.ToArray());
    writer.Write(dataBytes);

    foreach (var sample in samples)
        writer.Write((short)Math.Clamp((int)MathF.Round(sample * 32767f), short.MinValue, short.MaxValue));

    writer.Flush();
    return stream.ToArray();
}

// Run ANY onnx model on our engine and print each output's stats. Exists because a failing unit test tells
// you a number is wrong but not what the engine did, and PMT buffers the operator-level diagnostics that
// would say. Feeds inputs from the same JSON the controlflow/recurrent fixtures use:
//   { "inputs": { "name": { "shape": [..], "data": [..] } }, "outputs": { ... } }
//
//   dotnet run --project tools/zipvoice-harness -c Release -- runonnx <model.onnx> [fixture.json]
int RunOnnx(string modelPath, string fixturePath)
{
    if (!File.Exists(modelPath)) { Console.WriteLine($"no model at {modelPath}"); return 2; }

    var mlBuilder = MLContext.Create();
    mlBuilder.AllAcceleratorsAsync().GetAwaiter().GetResult();
    using var mlCtx = mlBuilder.ToContext();
    using var accel = mlCtx.CreatePreferredAcceleratorAsync().GetAwaiter().GetResult()
        ?? throw new InvalidOperationException("no accelerator");
    Console.WriteLine($"device   : {accel.AcceleratorType} {accel.Name}");

    var inputData = new Dictionary<string, (int[] Shape, float[] Data)>();
    var expected = new Dictionary<string, float[]>();
    if (File.Exists(fixturePath))
    {
        using var doc = System.Text.Json.JsonDocument.Parse(File.ReadAllText(fixturePath));
        float ToF(System.Text.Json.JsonElement e) => e.ValueKind switch
        {
            System.Text.Json.JsonValueKind.True => 1f,
            System.Text.Json.JsonValueKind.False => 0f,
            _ => (float)e.GetDouble(),
        };
        if (doc.RootElement.TryGetProperty("inputs", out var ins))
            foreach (var i in ins.EnumerateObject())
            {
                var shape = i.Value.GetProperty("shape").EnumerateArray().Select(e => e.GetInt32()).ToArray();
                var data = i.Value.GetProperty("data").EnumerateArray().Select(ToF).ToArray();
                if (shape.Length == 0) shape = new[] { 1 };
                if (data.Length == 0) data = new[] { 0f };
                inputData[i.Name] = (shape, data);
            }
        if (doc.RootElement.TryGetProperty("outputs", out var outs))
            foreach (var o in outs.EnumerateObject())
                expected[o.Name] = o.Value.GetProperty("data").EnumerateArray().Select(ToF).ToArray();
    }

    using var session = InferenceSession.CreateFromFile(accel, File.ReadAllBytes(modelPath),
        inputShapes: inputData.ToDictionary(kv => kv.Key, kv => kv.Value.Shape));

    Console.WriteLine($"compiled : {session.NodeCount} node(s), ops = {string.Join(", ", session.OperatorTypes)}");
    Console.WriteLine($"inputs   : {string.Join(", ", session.InputNames)}");
    Console.WriteLine($"outputs  : {string.Join(", ", session.OutputNames)}");

    var buffers = new List<MemoryBuffer1D<float, Stride1D.Dense>>();
    var feeds = new Dictionary<string, Tensor>();
    foreach (var (name, v) in inputData)
    {
        var buf = accel.Allocate1D(v.Data);
        buffers.Add(buf);
        feeds[name] = new Tensor(buf.View, v.Shape);
    }

    var results = session.RunAsync(feeds).GetAwaiter().GetResult();
    int bad = 0;
    foreach (var (name, t) in results)
    {
        int n = t.ElementCount;
        using var host = accel.Allocate1D<float>(n);
        host.View.CopyFrom(t.Data.SubView(0, n));
        accel.Synchronize();
        var got = host.GetAsArray1D();
        var head = string.Join(" ", got.Take(6).Select(x => x.ToString("F4")));
        Console.Write($"  {name,-16} shape=[{string.Join(",", t.Shape)}] n={n} head=[{head}]");
        if (expected.TryGetValue(name, out var exp))
        {
            int m = Math.Min(exp.Length, n);
            double worst = 0;
            for (int i = 0; i < m; i++) worst = Math.Max(worst, Math.Abs(got[i] - exp[i]));
            Console.Write($"  vs ORT: max|d|={worst:E2}{(worst > 1e-4 ? "  MISMATCH" : "  OK")}");
            if (worst > 1e-4) bad++;
        }
        Console.WriteLine();
    }
    foreach (var b in buffers) b.Dispose();
    Console.WriteLine(bad == 0 ? "RESULT   : PASS" : $"RESULT   : FAIL ({bad})");
    return bad;
}
