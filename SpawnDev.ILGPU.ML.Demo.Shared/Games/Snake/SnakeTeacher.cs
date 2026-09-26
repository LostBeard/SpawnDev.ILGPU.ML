namespace SpawnDev.ILGPU.ML.Demo.Shared.Games.Snake;

/// <summary>
/// Strong deterministic Snake policy for behavioral cloning.
/// Takes food when the first step keeps the tail reachable; otherwise steers
/// with flood-fill space while still seeking food. Never pure tail-chase —
/// that produces endless orbits that ignore food at mid/long length.
/// </summary>
public static class SnakeTeacher
{
    /// <summary>After this many ticks without food, bias hard toward the food.</summary>
    public static int HungerTicks(SnakeGame game) => Math.Max(game.GridSize * 2, game.Length);

    /// <summary>After this many ticks, accept a slightly riskier approach to break a stall.</summary>
    public static int StarvationTicks(SnakeGame game) => Math.Max(game.GridSize * 5, game.Length * 2);

    public static SnakeAction Choose(SnakeGame game)
    {
        var foodStep = SnakeReachability.FirstStepToFood(game);
        if (foodStep.HasValue && SnakeReachability.StepKeepsTailReachable(game, foodStep.Value))
            return foodStep.Value;

        return ChooseSpace(game, foodStep);
    }

    public static int ChooseIndex(SnakeGame game) => (int)Choose(game);

    /// <summary>True if the action is legal this tick (in bounds, not body, not 180° reverse).</summary>
    public static bool IsLegal(SnakeGame game, SnakeAction action)
    {
        if (SnakeGame.IsOpposite(action, game.Direction)) return false;
        return !SnakeStateEncoder.IsDanger(game, game.Head, action, steps: 1);
    }

    /// <summary>Legal-move mask in up/down/left/right order (length 4).</summary>
    public static bool[] LegalMask(SnakeGame game)
    {
        var m = new bool[4];
        for (int i = 0; i < 4; i++)
            m[i] = IsLegal(game, (SnakeAction)i);
        return m;
    }

    /// <summary>True when a safe (tail-reachable) first step toward food exists.</summary>
    public static bool HasSafeFoodPath(SnakeGame game)
    {
        var foodStep = SnakeReachability.FirstStepToFood(game);
        return foodStep.HasValue && SnakeReachability.StepKeepsTailReachable(game, foodStep.Value);
    }

    /// <summary>Safe BFS first step to food, or null.</summary>
    public static SnakeAction? SafeFoodStep(SnakeGame game)
    {
        var foodStep = SnakeReachability.FirstStepToFood(game);
        if (foodStep.HasValue && SnakeReachability.StepKeepsTailReachable(game, foodStep.Value))
            return foodStep;
        return null;
    }

    private static SnakeAction ChooseSpace(SnakeGame game, SnakeAction? foodStep)
    {
        bool hungry = game.StepsSinceFood >= HungerTicks(game);
        bool starving = game.StepsSinceFood >= StarvationTicks(game);

        // Starvation breakout: take the BFS food step even if tail-check fails —
        // better than orbiting forever. Only when a path exists at all.
        if (starving && foodStep.HasValue && IsLegal(game, foodStep.Value))
            return foodStep.Value;

        // Collect legal candidates with metrics.
        Span<int> areas = stackalloc int[4];
        Span<bool> reach = stackalloc bool[4];
        Span<bool> legal = stackalloc bool[4];
        Span<int> foodDelta = stackalloc int[4]; // positive = closer to food
        int maxReachArea = -1;
        int maxAnyArea = -1;
        bool anyReach = false;

        int foodBefore = game.Food.X >= 0
            ? Math.Abs(game.Head.X - game.Food.X) + Math.Abs(game.Head.Y - game.Food.Y)
            : 0;

        for (int i = 0; i < 4; i++)
        {
            var action = (SnakeAction)i;
            legal[i] = IsLegal(game, action);
            if (!legal[i])
            {
                areas[i] = -1;
                continue;
            }

            var (dx, dy) = SnakeGame.Delta(action);
            var next = (X: game.Head.X + dx, Y: game.Head.Y + dy);
            bool eating = next == game.Food;
            areas[i] = SnakeReachability.FloodArea(game, next, eating);
            reach[i] = SnakeReachability.CanReach(game, next, game.Tail, eating);
            int foodAfter = game.Food.X >= 0
                ? Math.Abs(next.X - game.Food.X) + Math.Abs(next.Y - game.Food.Y)
                : foodBefore;
            foodDelta[i] = foodBefore - foodAfter;

            if (areas[i] > maxAnyArea) maxAnyArea = areas[i];
            if (reach[i])
            {
                anyReach = true;
                if (areas[i] > maxReachArea) maxReachArea = areas[i];
            }
        }

        // Prefer the reachable-tail set when any exists; else all legal.
        int areaFloor = anyReach ? maxReachArea : maxAnyArea;
        // Allow small area sacrifice to approach food (critical — old *10 vs *1000 caused tail orbits).
        int tolerance = hungry ? Math.Max(3, game.GridSize / 2) : Math.Max(1, game.GridSize / 4);

        SnakeAction best = game.Direction;
        double bestScore = double.NegativeInfinity;
        bool found = false;

        for (int i = 0; i < 4; i++)
        {
            if (!legal[i]) continue;
            if (anyReach && !reach[i] && !starving) continue;
            if (areas[i] < areaFloor - tolerance) continue;

            // Primary: stay near max free space. Secondary: approach food. Never chase tail.
            double score = areas[i] * 100.0;
            score += foodDelta[i] * (hungry ? 500.0 : 200.0);
            if (reach[i]) score += 50.0;
            // Mild penalty for walking onto food when that move fails the safe check
            // (eating with no escape) — unless starving.
            var action = (SnakeAction)i;
            var (dx, dy) = SnakeGame.Delta(action);
            var next = (X: game.Head.X + dx, Y: game.Head.Y + dy);
            if (!starving && next == game.Food && !reach[i])
                score -= 10_000.0;

            if (!found || score > bestScore)
            {
                bestScore = score;
                best = action;
                found = true;
            }
        }

        if (found) return best;

        // Fallback: any legal move with max area.
        for (int i = 0; i < 4; i++)
        {
            if (!legal[i]) continue;
            if (!found || areas[i] > bestScore)
            {
                bestScore = areas[i];
                best = (SnakeAction)i;
                found = true;
            }
        }

        return best;
    }

    /// <summary>Roll out teacher for many games; returns mean / median / max score.</summary>
    public static (double Mean, double Median, int Max) EvaluateScores(
        int games = 30,
        int maxSteps = 800,
        int gridSize = SnakeGame.DefaultGridSize,
        int seed = 0)
    {
        var scores = new int[games];
        for (int g = 0; g < games; g++)
        {
            var game = new SnakeGame(gridSize, seed + g * 17);
            for (int s = 0; s < maxSteps && game.IsAlive; s++)
            {
                game.SetAction(Choose(game));
                game.Tick();
            }
            scores[g] = game.Score;
        }

        Array.Sort(scores);
        double mean = scores.Average();
        double median = games % 2 == 1
            ? scores[games / 2]
            : (scores[games / 2 - 1] + scores[games / 2]) * 0.5;
        return (mean, median, scores[^1]);
    }

    /// <summary>
    /// After reaching <paramref name="minScore"/>, require another food within
    /// <paramref name="maxStallSteps"/> — catches endless tail orbits.
    /// </summary>
    public static void AssertNoFoodStall(
        int seeds = 12,
        int minScore = 28,
        int maxStallSteps = 200,
        int gridSize = SnakeGame.DefaultGridSize,
        int seed0 = 9)
    {
        for (int s = 0; s < seeds; s++)
        {
            var game = new SnakeGame(gridSize, seed0 + s * 19);
            int guard = gridSize * gridSize * 4;
            while (game.IsAlive && game.Score < minScore && guard-- > 0)
            {
                game.SetAction(Choose(game));
                game.Tick();
            }

            if (!game.IsAlive || game.Score < minScore)
                continue; // this seed died early — skip stall check

            int scoreAt = game.Score;
            for (int i = 0; i < maxStallSteps && game.IsAlive; i++)
            {
                game.SetAction(Choose(game));
                game.Tick();
                if (game.Score > scoreAt)
                    break;
            }

            if (game.IsAlive && game.Score <= scoreAt)
                throw new Exception(
                    $"Seed {seed0 + s * 19}: stalled at score {scoreAt} for {maxStallSteps} steps (tail-orbit).");
        }
    }
}
