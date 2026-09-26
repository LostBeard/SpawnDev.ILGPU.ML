using SpawnDev.ILGPU.ML.SystemOne;

namespace SpawnDev.ILGPU.ML.Demo.Shared.Games.Snake;

/// <summary>
/// Production play helper: encode → System One choice → legal-action mask → SnakeAction.
/// Applies a cheap safe-food shield so a stale/noisy head cannot ignore an obviously safe meal.
/// </summary>
public static class SnakeSystemOnePolicy
{
    public static readonly ChoiceQuestion MoveQuestion =
        new("Which way should the snake move?", SnakeSystemOneSpec.ActionKeys);

    /// <summary>
    /// One decision with wall-clock latency. Always masks illegal moves before argmax.
    /// If the teacher has a safe food step and the head picked something else, override to food
    /// (shield — keeps mid-game from ignoring food after length ~30+).
    /// </summary>
    public static async Task<(SnakeAction Action, ChoiceAnswer Answer, double LatencyMs)> DecideAsync(
        SystemOneDecisionHead head,
        SnakeGame game)
    {
        if (head.StateDim != SnakeSystemOneSpec.StateDim || head.NumOptions != SnakeSystemOneSpec.NumActions)
            throw new ArgumentException(
                $"Head must be SnakeSystemOneSpec.CreateHead() (state={SnakeSystemOneSpec.StateDim}, actions=4).");

        var state = SnakeStateEncoder.Encode(game);
        var mask = SnakeTeacher.LegalMask(game);
        var resp = await head.DecideAsync(
            state,
            new Dictionary<string, SystemOneQuestion> { ["move"] = MoveQuestion },
            mask).ConfigureAwait(false);

        if (resp.Answers["move"] is not ChoiceAnswer choice)
            throw new InvalidOperationException("Expected ChoiceAnswer for move.");

        if (!Enum.TryParse<SnakeAction>(choice.Choice, ignoreCase: true, out var action))
            throw new InvalidOperationException($"Unknown action key '{choice.Choice}'.");

        if (!mask[(int)action] && mask.Any(x => x))
        {
            for (int i = 0; i < 4; i++)
            {
                if (mask[i])
                {
                    action = (SnakeAction)i;
                    break;
                }
            }
        }

        // Safe-food shield: never leave a reachable safe meal on the table.
        var safeFood = SnakeTeacher.SafeFoodStep(game);
        if (safeFood.HasValue && action != safeFood.Value)
            action = safeFood.Value;

        // Hunger shield: if starving and teacher wants food-direction, prefer teacher choice.
        if (game.StepsSinceFood >= SnakeTeacher.HungerTicks(game))
        {
            var teacher = SnakeTeacher.Choose(game);
            if (mask[(int)teacher])
                action = teacher;
        }

        return (action, choice, resp.DecisionLatencyMs);
    }
}
