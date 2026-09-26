using System.Diagnostics;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Training;

namespace SpawnDev.ILGPU.ML.SystemOne;

/// <summary>
/// Local System One decision head: float state → typed answers with probabilities.
/// Inspired by the Jev/Laya category (structured decisions, no text generation) —
/// not a port of Laya/Jev weights. Backed by a tiny GPU MLP via <see cref="TrainableModel"/>.
/// </summary>
/// <remarks>
/// Typical use:
/// <code>
/// using var head = SystemOneDecisionHead.CreateForSnake(accelerator);
/// var answer = await head.ChooseAsync(state, question, allowedMask);
/// </code>
/// Pass <c>allowedMask</c> whenever the environment can forbid options (illegal moves, closed gates).
/// </remarks>
public sealed class SystemOneDecisionHead : IDisposable
{
    public const int DefaultHidden = 64;

    private readonly TrainableModel _model;
    private bool _disposed;

    public int StateDim { get; }
    public int NumOptions { get; }
    public int HiddenSize { get; }
    public int ParameterCount => _model.ParameterCount;
    public TrainableModel Model => _model;

    public SystemOneDecisionHead(Accelerator accelerator, int stateDim, int numOptions, int hidden = DefaultHidden, int maxBatchSize = 64)
    {
        if (stateDim < 1) throw new ArgumentOutOfRangeException(nameof(stateDim));
        if (numOptions < 2) throw new ArgumentOutOfRangeException(nameof(numOptions));
        if (hidden < 1) throw new ArgumentOutOfRangeException(nameof(hidden));
        StateDim = stateDim;
        NumOptions = numOptions;
        HiddenSize = hidden;
        _model = new TrainableModel(accelerator);
        _model.AddLinear(stateDim, hidden);
        _model.AddReLU();
        _model.AddLinear(hidden, numOptions);
        _model.Build(maxBatchSize: maxBatchSize);
    }

    /// <summary>4-way direction head sized for <see cref="SystemOneSnakeSpec"/>.</summary>
    public static SystemOneDecisionHead CreateForSnake(Accelerator accelerator) =>
        new(accelerator,
            SystemOneSnakeSpec.StateDim,
            SystemOneSnakeSpec.NumActions,
            SystemOneSnakeSpec.Hidden,
            SystemOneSnakeSpec.TrainMaxBatch);

    /// <summary>
    /// One Softmax-CE training step. Labels are class indices in [0, NumOptions).
    /// Pass <paramref name="readLoss"/> false in hot loops; use <see cref="ReadLastLossAsync"/> per epoch.
    /// </summary>
    public Task<float> TrainStepAsync(
        float[] inputData, int[] targetLabels, int batchSize, float learningRate = 0.05f, bool readLoss = true) =>
        _model.TrainStepAsync(inputData, targetLabels, batchSize, learningRate, readLoss);

    /// <summary>Fire-and-forget train step (no host loss). Pair with <see cref="FlushAsync"/> / <see cref="ReadLastLossAsync"/>.</summary>
    public void TrainStep(float[] inputData, int[] targetLabels, int batchSize, float learningRate) =>
        _model.TrainStep(inputData, targetLabels, batchSize, learningRate);

    public Task FlushAsync() => _model.FlushAsync();
    public Task<float> ReadLastLossAsync(int batchSize) => _model.ReadLastLossAsync(batchSize);

    /// <summary>
    /// Export head identity + MLP weights (FP32). Suitable for localStorage (~tens of KB for Snake).
    /// </summary>
    public async Task<byte[]> ExportWeightsAsync()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var modelBlob = await _model.ExportWeightsAsync().ConfigureAwait(false);
        using var ms = new MemoryStream(16 + modelBlob.Length);
        using (var bw = new BinaryWriter(ms, System.Text.Encoding.UTF8, leaveOpen: true))
        {
            bw.Write(HeadMagic); // 'S1DH' LE
            bw.Write(HeadWeightsVersion);
            bw.Write(StateDim);
            bw.Write(NumOptions);
            bw.Write(HiddenSize);
            bw.Write(modelBlob.Length);
            bw.Write(modelBlob);
        }
        return ms.ToArray();
    }

    /// <summary>Load a blob from <see cref="ExportWeightsAsync"/>. Dimensions must match this head.</summary>
    public void ImportWeights(ReadOnlySpan<byte> blob)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        using var ms = new MemoryStream(blob.ToArray());
        using var br = new BinaryReader(ms);
        int magic = br.ReadInt32();
        if (magic != HeadMagic)
            throw new InvalidDataException($"Bad SystemOneDecisionHead magic 0x{magic:X8}.");
        int ver = br.ReadInt32();
        if (ver != HeadWeightsVersion)
            throw new InvalidDataException($"Unsupported head weights version {ver}.");
        int stateDim = br.ReadInt32();
        int numOptions = br.ReadInt32();
        int hidden = br.ReadInt32();
        if (stateDim != StateDim || numOptions != NumOptions || hidden != HiddenSize)
            throw new InvalidDataException(
                $"Head dims {stateDim}/{numOptions}/{hidden} != {StateDim}/{NumOptions}/{HiddenSize}.");
        int len = br.ReadInt32();
        var modelBlob = br.ReadBytes(len);
        if (modelBlob.Length != len)
            throw new InvalidDataException("Truncated model weight blob.");
        _model.ImportWeights(modelBlob);
    }

    public Task ImportWeightsAsync(byte[] blob)
    {
        ImportWeights(blob);
        return Task.CompletedTask;
    }

    private const int HeadMagic = 0x48443153; // 'S1DH' LE
    private const int HeadWeightsVersion = 1;

    /// <summary>Raw Softmax probabilities (length <see cref="NumOptions"/>), no masking.</summary>
    public Task<float[]> PredictProbsAsync(float[] state) =>
        _model.PredictProbsAsync(state, batchSize: 1);

    /// <summary>Evaluate all questions against the same state (sequential tiny forwards).</summary>
    public Task<SystemOneResponse> DecideAsync(
        float[] state,
        IReadOnlyDictionary<string, SystemOneQuestion> questions) =>
        DecideAsync(state, questions, allowedMask: null);

    /// <summary>
    /// Like <see cref="DecideAsync(float[],IReadOnlyDictionary{string,SystemOneQuestion})"/>,
    /// but applies <paramref name="allowedMask"/> to every <see cref="ChoiceQuestion"/>.
    /// Score/noul answers are unchanged by the mask.
    /// </summary>
    public async Task<SystemOneResponse> DecideAsync(
        float[] state,
        IReadOnlyDictionary<string, SystemOneQuestion> questions,
        bool[]? allowedMask)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ValidateState(state);
        if (questions.Count == 0)
            throw new ArgumentException("At least one question is required.", nameof(questions));
        if (allowedMask != null && allowedMask.Length != NumOptions)
            throw new ArgumentException($"Mask length {allowedMask.Length} != NumOptions {NumOptions}.", nameof(allowedMask));

        var sw = Stopwatch.StartNew();
        var answers = new Dictionary<string, SystemOneAnswer>(questions.Count);

        foreach (var (id, q) in questions)
        {
            answers[id] = q switch
            {
                ChoiceQuestion choice => await ChooseCoreAsync(state, choice, allowedMask).ConfigureAwait(false),
                ScoreQuestion score => await ScoreCoreAsync(state, score).ConfigureAwait(false),
                NoulQuestion noul => await NoulCoreAsync(state, noul).ConfigureAwait(false),
                _ => throw new NotSupportedException($"Unknown question type: {q.GetType().Name}"),
            };
        }

        sw.Stop();
        return new SystemOneResponse
        {
            Answers = answers,
            DecisionLatencyMs = sw.Elapsed.TotalMilliseconds,
        };
    }

    public Task<ChoiceAnswer> ChooseAsync(float[] state, ChoiceQuestion question) =>
        ChooseAsync(state, question, allowedMask: null);

    /// <summary>
    /// Softmax choice, optionally masked to legal options then renormalized.
    /// </summary>
    public async Task<ChoiceAnswer> ChooseAsync(float[] state, ChoiceQuestion question, bool[]? allowedMask)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ValidateState(state);
        if (allowedMask != null && allowedMask.Length != NumOptions)
            throw new ArgumentException($"Mask length {allowedMask.Length} != NumOptions {NumOptions}.", nameof(allowedMask));
        return await ChooseCoreAsync(state, question, allowedMask).ConfigureAwait(false);
    }

    public async Task<ScoreAnswer> ScoreAsync(float[] state, ScoreQuestion question)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ValidateState(state);
        return await ScoreCoreAsync(state, question).ConfigureAwait(false);
    }

    public async Task<NoulAnswer> NoulAsync(float[] state, NoulQuestion question)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ValidateState(state);
        return await NoulCoreAsync(state, question).ConfigureAwait(false);
    }

    private void ValidateState(float[] state)
    {
        if (state.Length != StateDim)
            throw new ArgumentException($"Expected state length {StateDim}, got {state.Length}.", nameof(state));
    }

    private async Task<ChoiceAnswer> ChooseCoreAsync(float[] state, ChoiceQuestion question, bool[]? allowedMask)
    {
        var keys = question.Criteria.Keys.ToArray();
        if (keys.Length != NumOptions)
            throw new ArgumentException(
                $"Choice has {keys.Length} options but this head was built for {NumOptions}.",
                nameof(question));

        var probs = await _model.PredictProbsAsync(state, batchSize: 1).ConfigureAwait(false);
        var raw = BuildChoice(keys, probs);
        if (allowedMask == null)
            return raw;
        return SystemOneChoiceMask.Apply(raw, allowedMask, keys);
    }

    private async Task<ScoreAnswer> ScoreCoreAsync(float[] state, ScoreQuestion question)
    {
        if (question.Criteria.Count != NumOptions)
            throw new ArgumentException(
                $"Score has {question.Criteria.Count} levels but this head was built for {NumOptions}.",
                nameof(question));

        var probs = await _model.PredictProbsAsync(state, batchSize: 1).ConfigureAwait(false);
        float expected = 0f;
        float maxP = 0f;
        var map = new Dictionary<string, float>(NumOptions);
        for (int i = 0; i < NumOptions; i++)
        {
            float p = probs[i];
            expected += i * p;
            if (p > maxP) maxP = p;
            map[i.ToString()] = p;
        }

        return new ScoreAnswer(expected, maxP, map);
    }

    private async Task<NoulAnswer> NoulCoreAsync(float[] state, NoulQuestion question)
    {
        if (NumOptions != 2)
            throw new InvalidOperationException("Noul requires a 2-option decision head.");

        var probs = await _model.PredictProbsAsync(state, batchSize: 1).ConfigureAwait(false);
        _ = question;
        return new NoulAnswer(probs[1]);
    }

    private static ChoiceAnswer BuildChoice(string[] keys, float[] probs)
    {
        int best = 0;
        float maxP = probs[0];
        var map = new Dictionary<string, float>(keys.Length);
        for (int i = 0; i < keys.Length; i++)
        {
            map[keys[i]] = probs[i];
            if (probs[i] > maxP)
            {
                maxP = probs[i];
                best = i;
            }
        }
        return new ChoiceAnswer(keys[best], maxP, map);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _model.Dispose();
    }
}
