using SpawnDev.ILGPU.ML.SystemOne;

namespace SpawnDev.ILGPU.ML.Demo.Shared.Games.Snake;

/// <summary>Progress snapshot for UI (fraction in [0,1], human-readable phase).</summary>
public readonly record struct SnakeTrainProgress(
    float Fraction,
    string Phase,
    int Epoch,
    int Epochs,
    float? Loss);

/// <summary>
/// Behavioral cloning: roll out <see cref="SnakeTeacher"/>, train a System One head to match.
/// Hot path avoids per-batch host loss readback and reuses GPU scratch via <see cref="TrainableModel"/>.
/// </summary>
public static class SnakeSystemOneTrainer
{
    public const int DefaultSampleCount = 4096;
    public const int DefaultBatchSize = 32;
    public const int DefaultEpochs = 80;
    public const float DefaultLearningRate = 0.05f;

    /// <summary>Flush pending GPU commands every N batches (no host copy).</summary>
    public const int FlushEveryBatches = 8;

    /// <summary>
    /// Collect teacher-labeled (state, action) pairs by rolling out games.
    /// Skips the first few steps of each life (trivial early board) for denser hard examples.
    /// </summary>
    public static (float[] Inputs, int[] Labels, int Count) CollectDataset(
        int sampleCount,
        int gridSize = SnakeGame.DefaultGridSize,
        int seed = 42,
        int skipWarmupSteps = 3,
        IProgress<SnakeTrainProgress>? progress = null)
    {
        var inputs = new float[sampleCount * SnakeStateEncoder.StateDim];
        var labels = new int[sampleCount];
        var game = new SnakeGame(gridSize, seed);
        int collected = 0;
        int stepInLife = 0;
        int safety = sampleCount * 80;
        int reportEvery = Math.Max(1, sampleCount / 20);

        while (collected < sampleCount && safety-- > 0)
        {
            if (!game.IsAlive)
            {
                game.Reset();
                stepInLife = 0;
                continue;
            }

            var action = SnakeTeacher.Choose(game);
            if (stepInLife >= skipWarmupSteps)
            {
                SnakeStateEncoder.Encode(game, inputs.AsSpan(collected * SnakeStateEncoder.StateDim));
                labels[collected] = (int)action;
                collected++;
                if (progress != null && (collected % reportEvery == 0 || collected == sampleCount))
                {
                    progress.Report(new SnakeTrainProgress(
                        0.05f + 0.20f * (collected / (float)sampleCount),
                        $"Collecting teacher data ({collected}/{sampleCount})",
                        0, 0, null));
                }
            }

            game.SetAction(action);
            game.Tick();
            stepInLife++;
        }

        if (collected < sampleCount)
            throw new InvalidOperationException($"Only collected {collected}/{sampleCount} samples.");

        return (inputs, labels, collected);
    }

    /// <summary>
    /// Train <paramref name="head"/> on teacher rollouts. Returns last epoch loss (one host read per epoch).
    /// </summary>
    public static async Task<float> TrainAsync(
        SystemOneDecisionHead head,
        int sampleCount = DefaultSampleCount,
        int epochs = DefaultEpochs,
        int batchSize = DefaultBatchSize,
        float learningRate = DefaultLearningRate,
        int seed = 42,
        IProgress<SnakeTrainProgress>? progress = null)
    {
        if (head.StateDim != SnakeStateEncoder.StateDim)
            throw new ArgumentException($"Head state dim {head.StateDim} != encoder {SnakeStateEncoder.StateDim}.");
        if (head.NumOptions != 4)
            throw new ArgumentException("Snake head must have 4 options.");

        progress?.Report(new SnakeTrainProgress(0.02f, "Collecting teacher rollouts…", 0, epochs, null));
        var (inputs, labels, count) = CollectDataset(sampleCount, seed: seed, progress: progress);

        var order = new int[count];
        for (int i = 0; i < count; i++)
            order[i] = i;
        var rng = new Random(seed + 1);
        float lastLoss = 0f;
        // Persistent managed batch staging — filled then CopyFromCPU'd; not grown per step.
        var batchIn = new float[batchSize * SnakeStateEncoder.StateDim];
        var batchLab = new int[batchSize];

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            for (int i = count - 1; i > 0; i--)
            {
                int j = rng.Next(i + 1);
                (order[i], order[j]) = (order[j], order[i]);
            }

            int batchIndex = 0;
            int lastStart = (count / batchSize - 1) * batchSize;
            for (int start = 0; start + batchSize <= count; start += batchSize)
            {
                for (int b = 0; b < batchSize; b++)
                {
                    int src = order[start + b];
                    Array.Copy(inputs, src * SnakeStateEncoder.StateDim, batchIn, b * SnakeStateEncoder.StateDim, SnakeStateEncoder.StateDim);
                    batchLab[b] = labels[src];
                }

                if (start == lastStart)
                {
                    // One host loss read per epoch (tiny float[batchSize] sink).
                    lastLoss = await head.TrainStepAsync(batchIn, batchLab, batchSize, learningRate, readLoss: true)
                        .ConfigureAwait(false);
                }
                else
                {
                    head.TrainStep(batchIn, batchLab, batchSize, learningRate);
                    batchIndex++;
                    if (batchIndex % FlushEveryBatches == 0)
                        await head.FlushAsync().ConfigureAwait(false);
                }
            }

            // 25%..90% reserved for epochs
            float frac = 0.25f + 0.65f * ((epoch + 1) / (float)epochs);
            progress?.Report(new SnakeTrainProgress(
                frac,
                $"Training epoch {epoch + 1}/{epochs}",
                epoch + 1,
                epochs,
                lastLoss));

            // Let Blazor paint without awaiting UI work on the GPU thread path.
            await Task.Yield();
        }

        progress?.Report(new SnakeTrainProgress(0.92f, "Training complete", epochs, epochs, lastLoss));
        return lastLoss;
    }

    /// <summary>
    /// Fraction of held-out states where masked argmax matches the teacher.
    /// </summary>
    public static async Task<float> AgreementAsync(
        SystemOneDecisionHead head,
        int sampleCount = 256,
        int seed = 99)
    {
        var game = new SnakeGame(SnakeGame.DefaultGridSize, seed);
        int correct = 0;
        int total = 0;
        int safety = sampleCount * 80;

        while (total < sampleCount && safety-- > 0)
        {
            if (!game.IsAlive)
            {
                game.Reset();
                continue;
            }

            var teacher = SnakeTeacher.Choose(game);
            var (pred, _, _) = await SnakeSystemOnePolicy.DecideAsync(head, game).ConfigureAwait(false);
            if (pred == teacher) correct++;
            total++;

            game.SetAction(teacher);
            game.Tick();
        }

        return correct / (float)total;
    }

    /// <summary>Play full games with the trained head (masked); return mean score.</summary>
    public static async Task<(double Mean, int Max)> EvaluatePolicyScoresAsync(
        SystemOneDecisionHead head,
        int games = 12,
        int maxSteps = 800,
        int seed = 200,
        IProgress<SnakeTrainProgress>? progress = null)
    {
        var scores = new int[games];
        for (int g = 0; g < games; g++)
        {
            var game = new SnakeGame(SnakeGame.DefaultGridSize, seed + g * 31);
            for (int s = 0; s < maxSteps && game.IsAlive; s++)
            {
                var (action, _, _) = await SnakeSystemOnePolicy.DecideAsync(head, game).ConfigureAwait(false);
                game.SetAction(action);
                game.Tick();
            }
            scores[g] = game.Score;
            progress?.Report(new SnakeTrainProgress(
                0.92f + 0.08f * ((g + 1) / (float)games),
                $"Evaluating policy ({g + 1}/{games})",
                0, 0, null));
            await Task.Yield();
        }

        double sum = 0;
        int max = 0;
        for (int i = 0; i < games; i++)
        {
            sum += scores[i];
            if (scores[i] > max) max = scores[i];
        }
        return (sum / games, max);
    }
}
