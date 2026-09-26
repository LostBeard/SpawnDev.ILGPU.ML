using SpawnDev.ILGPU.ML.Demo.Shared.Games.Snake;
using SpawnDev.ILGPU.ML.SystemOne;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>System One decision head + Classic Snake teacher / behavioral clone tests.</summary>
public abstract partial class MLTestBase
{
    [TestMethod]
    public async Task SystemOne_SnakeTeacher_NeverIllegalWhenSafeExists() => await RunPureTest(() =>
    {
        for (int seed = 0; seed < 40; seed++)
        {
            var game = new SnakeGame(gridSize: 12, seed: seed);
            for (int step = 0; step < 120 && game.IsAlive; step++)
            {
                bool anySafe = false;
                for (int a = 0; a < 4; a++)
                {
                    if (SnakeTeacher.IsLegal(game, (SnakeAction)a))
                    {
                        anySafe = true;
                        break;
                    }
                }

                var choice = SnakeTeacher.Choose(game);
                if (anySafe && !SnakeTeacher.IsLegal(game, choice))
                    throw new Exception($"Seed {seed} step {step}: teacher chose illegal {choice} with safe moves available.");

                if (SnakeGame.IsOpposite(choice, game.Direction))
                    throw new Exception($"Seed {seed} step {step}: teacher reversed into itself ({choice}).");

                game.SetAction(choice);
                game.Tick();
            }
        }

        Console.WriteLine("[SystemOne] SnakeTeacher legality: OK across 40 seeds");
        return Task.CompletedTask;
    });

    [TestMethod]
    public async Task SystemOne_SnakeTeacher_MeanScoreBeatsGreedyFloor() => await RunPureTest(() =>
    {
        // Safe teacher should clearly beat the old greedy-BFS floor (~teens).
        var (mean, median, max) = SnakeTeacher.EvaluateScores(games: 24, maxSteps: 800, seed: 3);
        Console.WriteLine($"[SystemOne] Teacher scores mean={mean:F1} median={median:F1} max={max}");
        if (mean < 20.0)
            throw new Exception($"Teacher mean score {mean:F1} < 20 — safe-food heuristic regressing.");
        if (max < 30)
            throw new Exception($"Teacher max score {max} < 30 — expected at least one strong game.");
        return Task.CompletedTask;
    });

    [TestMethod]
    public async Task SystemOne_SnakeTeacher_NoTailOrbitStall() => await RunPureTest(() =>
    {
        SnakeTeacher.AssertNoFoodStall(seeds: 16, minScore: 28, maxStallSteps: 220, seed0: 9);
        Console.WriteLine("[SystemOne] Teacher: no tail-orbit stall after score 28");
        return Task.CompletedTask;
    });

    [TestMethod]
    public async Task SystemOne_SnakeGame_GrowsOnFood() => await RunPureTest(() =>
    {
        var game = new SnakeGame(gridSize: 8, seed: 1);
        int startLen = game.Length;
        int scoreBefore = game.Score;
        for (int i = 0; i < 400 && game.IsAlive; i++)
        {
            game.SetAction(SnakeTeacher.Choose(game));
            game.Tick();
            if (game.Score > scoreBefore)
            {
                if (game.Length <= startLen)
                    throw new Exception("Score increased but length did not grow.");
                Console.WriteLine($"[SystemOne] Snake grew to length {game.Length} after eating.");
                return Task.CompletedTask;
            }
        }

        throw new Exception("Teacher failed to eat food within 400 steps.");
    });

    [TestMethod]
    public async Task SystemOne_Choice_ProbsSumToOne() => await RunTest(async accelerator =>
    {
        using var head = new SystemOneDecisionHead(accelerator, stateDim: 4, numOptions: 3, hidden: 8);
        var inputs = new float[] { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0 };
        var labels = new int[] { 0, 1, 2 };
        for (int i = 0; i < 30; i++)
            await head.TrainStepAsync(inputs, labels, 3, 0.1f);

        var state = new float[] { 1, 0, 0, 0 };
        var q = new ChoiceQuestion("pick", "a", "b", "c");
        var ans = await head.ChooseAsync(state, q);

        float sum = ans.Probabilities.Values.Sum();
        if (MathF.Abs(sum - 1f) > 0.02f)
            throw new Exception($"Probabilities sum={sum}, expected ~1");
        if (!ans.Probabilities.ContainsKey(ans.Choice))
            throw new Exception("Choice key missing from probabilities.");
        if (ans.Confidence < ans.Probabilities.Values.Max() - 0.001f)
            throw new Exception("Confidence should equal max probability.");

        Console.WriteLine($"[SystemOne] Choice probs sum={sum:F4}, pick={ans.Choice}, conf={ans.Confidence:F3}");
    });

    [TestMethod]
    public async Task SystemOne_ChoiceMask_ZerosIllegalOptions() => await RunTest(async accelerator =>
    {
        using var head = new SystemOneDecisionHead(accelerator, stateDim: 2, numOptions: 4, hidden: 8);
        // Train to prefer index 0 strongly.
        var inputs = new float[32];
        var labels = new int[16];
        for (int i = 0; i < 16; i++)
        {
            inputs[i * 2] = 1f;
            labels[i] = 0;
        }
        for (int e = 0; e < 40; e++)
            await head.TrainStepAsync(inputs, labels, 16, 0.15f);

        var q = new ChoiceQuestion("dir", SystemOneSnakeSpec.ActionKeys);
        var raw = await head.ChooseAsync(new float[] { 1f, 0f }, q);
        // Forbid the raw winner — mask must pick something else among allowed.
        var mask = new bool[4];
        for (int i = 0; i < 4; i++)
            mask[i] = SystemOneSnakeSpec.ActionKeys[i] != raw.Choice;
        if (!mask.Any(x => x))
            throw new Exception("Need at least one allowed option.");

        var masked = await head.ChooseAsync(new float[] { 1f, 0f }, q, mask);
        if (masked.Choice == raw.Choice)
            throw new Exception($"Mask failed: still picked forbidden '{masked.Choice}'.");
        if (masked.Probabilities[raw.Choice] > 1e-5f)
            throw new Exception($"Forbidden option still has mass {masked.Probabilities[raw.Choice]}.");
        float sum = masked.Probabilities.Values.Sum();
        if (MathF.Abs(sum - 1f) > 0.02f)
            throw new Exception($"Masked probs sum={sum}");

        Console.WriteLine($"[SystemOne] Mask: raw={raw.Choice} → masked={masked.Choice}");
    });

    [TestMethod]
    public async Task SystemOne_ScoreAndNoul_InRange() => await RunTest(async accelerator =>
    {
        using var scoreHead = new SystemOneDecisionHead(accelerator, stateDim: 2, numOptions: 4, hidden: 8);
        var scoreQ = new ScoreQuestion("severity", ["low", "med", "high", "crit"]);
        var score = await scoreHead.ScoreAsync(new float[] { 0.5f, -0.2f }, scoreQ);
        if (score.Score < -0.01f || score.Score > 3.01f)
            throw new Exception($"Score {score.Score} out of [0,3]");
        float sSum = score.Probabilities.Values.Sum();
        if (MathF.Abs(sSum - 1f) > 0.02f)
            throw new Exception($"Score probs sum={sSum}");

        using var noulHead = new SystemOneDecisionHead(accelerator, stateDim: 2, numOptions: 2, hidden: 8);
        var noulQ = new NoulQuestion("escalate?");
        var noul = await noulHead.NoulAsync(new float[] { 0.1f, 0.9f }, noulQ);
        if (noul.Noul < 0f || noul.Noul > 1f)
            throw new Exception($"Noul {noul.Noul} out of [0,1]");

        Console.WriteLine($"[SystemOne] Score={score.Score:F3} Noul={noul.Noul:F3}");
    });

    [TestMethod]
    public async Task SystemOne_Decide_ReportsLatency() => await RunTest(async accelerator =>
    {
        using var head = SystemOneDecisionHead.CreateForSnake(accelerator);
        var state = new float[SystemOneSnakeSpec.StateDim];
        state[2] = 1f;
        var questions = new Dictionary<string, SystemOneQuestion>
        {
            ["move"] = new ChoiceQuestion("move", SystemOneSnakeSpec.ActionKeys),
        };
        var resp = await head.DecideAsync(state, questions);
        if (!resp.Answers.ContainsKey("move"))
            throw new Exception("Missing move answer.");
        if (resp.DecisionLatencyMs < 0)
            throw new Exception("Negative latency.");
        if (head.StateDim != SnakeStateEncoder.StateDim)
            throw new Exception("CreateForSnake state dim != encoder.");
        Console.WriteLine($"[SystemOne] Decide latency={resp.DecisionLatencyMs:F3} ms params={head.ParameterCount}");
    });

    [TestMethod(Timeout = 300000)]
    public async Task SystemOne_Snake_BehavioralClone_AgreesWithTeacher() => await RunTest(async accelerator =>
    {
        using var head = SystemOneDecisionHead.CreateForSnake(accelerator);
        float loss = await SnakeSystemOneTrainer.TrainAsync(
            head,
            sampleCount: 4096,
            epochs: 80,
            batchSize: 32,
            learningRate: 0.05f,
            seed: 7);

        float agree = await SnakeSystemOneTrainer.AgreementAsync(head, sampleCount: 256, seed: 123);
        var (mean, max) = await SnakeSystemOneTrainer.EvaluatePolicyScoresAsync(head, games: 10, seed: 50);
        Console.WriteLine($"[SystemOne] BC loss={loss:F4} agree={agree:P1} policy mean={mean:F1} max={max}");
        if (agree < 0.85f)
            throw new Exception($"Held-out teacher agreement {agree:P1} < 85%");
        if (mean < 12.0)
            throw new Exception($"Cloned policy mean score {mean:F1} < 12 — expected lift from safe teacher.");
    });

    [TestMethod(Timeout = 120000)]
    public async Task SystemOne_Weights_RoundTrip_PreservesProbs() => await RunTest(async accelerator =>
    {
        using var src = new SystemOneDecisionHead(accelerator, stateDim: 8, numOptions: 4, hidden: 16);
        // Train a few steps so weights leave init - not full BC, just nonzero structure.
        var rng = new Random(9);
        var input = new float[8];
        var labels = new int[1];
        for (int i = 0; i < 40; i++)
        {
            for (int j = 0; j < 8; j++)
                input[j] = (float)(rng.NextDouble() * 2 - 1);
            labels[0] = rng.Next(4);
            src.TrainStep(input, labels, batchSize: 1, learningRate: 0.1f);
        }
        await src.FlushAsync();

        var probe = new float[8];
        for (int j = 0; j < 8; j++)
            probe[j] = (float)(j * 0.17 - 0.5);
        var before = await src.PredictProbsAsync(probe);
        var blob = await src.ExportWeightsAsync();
        if (blob.Length < 64)
            throw new Exception($"Export blob too small: {blob.Length}");

        using var dst = new SystemOneDecisionHead(accelerator, stateDim: 8, numOptions: 4, hidden: 16);
        dst.ImportWeights(blob);
        var after = await dst.PredictProbsAsync(probe);

        float maxDelta = 0f;
        for (int i = 0; i < before.Length; i++)
            maxDelta = Math.Max(maxDelta, Math.Abs(before[i] - after[i]));
        Console.WriteLine($"[SystemOne] Weights round-trip bytes={blob.Length} maxΔ={maxDelta:E3}");
        if (maxDelta > 1e-5f)
            throw new Exception($"Round-trip probs drifted by {maxDelta:E3}");

        // Mismatched dims must reject.
        using var wrong = new SystemOneDecisionHead(accelerator, stateDim: 8, numOptions: 3, hidden: 16);
        bool rejected = false;
        try { wrong.ImportWeights(blob); }
        catch (InvalidDataException) { rejected = true; }
        if (!rejected)
            throw new Exception("Expected InvalidDataException for mismatched NumOptions.");
    });
}
