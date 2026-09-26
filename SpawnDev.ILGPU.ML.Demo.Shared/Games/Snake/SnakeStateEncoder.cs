namespace SpawnDev.ILGPU.ML.Demo.Shared.Games.Snake;

/// <summary>
/// Compact float features for System One. Layout must match <see cref="SnakeSystemOneSpec.StateDim"/>.
/// </summary>
public static class SnakeStateEncoder
{
    public const int StateDim = SnakeSystemOneSpec.StateDim;

    public static readonly string[] ActionKeys = SnakeSystemOneSpec.ActionKeys;

    public static float[] Encode(SnakeGame game)
    {
        var state = new float[StateDim];
        Encode(game, state);
        return state;
    }

    public static void Encode(SnakeGame game, Span<float> state)
    {
        if (state.Length < StateDim)
            throw new ArgumentException($"Need at least {StateDim} floats.", nameof(state));
        state.Clear();

        var head = game.Head;
        var food = game.Food;
        var tail = game.Tail;
        float inv = 1f / Math.Max(1, game.GridSize - 1);
        float invArea = 1f / (game.GridSize * game.GridSize);

        if (food.X >= 0)
        {
            state[0] = (food.X - head.X) * inv;
            state[1] = (food.Y - head.Y) * inv;
        }

        state[2 + (int)game.Direction] = 1f;

        for (int i = 0; i < 4; i++)
        {
            var action = (SnakeAction)i;
            state[6 + i] = IsDanger(game, head, action, steps: 1) ? 1f : 0f;
            state[10 + i] = IsDanger(game, head, action, steps: 2) ? 1f : 0f;
            state[15 + i] = FreeRun(game, head, action) * inv;
            state[19 + i] = FoodInDirection(head, food, action) ? 1f : 0f;

            if (SnakeTeacher.IsLegal(game, action))
            {
                var (dx, dy) = SnakeGame.Delta(action);
                var next = (X: head.X + dx, Y: head.Y + dy);
                bool eating = next == food;
                state[23 + i] = SnakeReachability.FloodArea(game, next, eating) * invArea;
                state[27 + i] = SnakeReachability.CanReach(game, next, tail, eating) ? 1f : 0f;
            }
        }

        state[14] = game.Length * invArea;
        state[31] = SnakeTeacher.HasSafeFoodPath(game) ? 1f : 0f;
        state[32] = SnakeReachability.FloodArea(game, head, eating: false) * invArea;
        state[33] = (tail.X - head.X) * inv;
        state[34] = (tail.Y - head.Y) * inv;

        // Quadrant body density relative to head (NW, NE, SW, SE).
        int[] dens = new int[4];
        foreach (var (x, y) in game.Body)
        {
            if ((x, y) == head) continue;
            int qi = (y < head.Y ? 0 : 2) + (x < head.X ? 0 : 1);
            dens[qi]++;
        }
        float invLen = 1f / Math.Max(1, game.Length - 1);
        for (int i = 0; i < 4; i++)
            state[35 + i] = dens[i] * invLen;

        state[39] = Math.Clamp(game.StepsSinceFood / (float)Math.Max(1, SnakeTeacher.StarvationTicks(game)), 0f, 1f);
    }

    public static bool IsDanger(SnakeGame game, (int X, int Y) from, SnakeAction action, int steps)
    {
        var (dx, dy) = SnakeGame.Delta(action);
        int x = from.X + dx * steps;
        int y = from.Y + dy * steps;
        if (x < 0 || y < 0 || x >= game.GridSize || y >= game.GridSize)
            return true;

        if (game.Occupied(x, y) && (x, y) != game.Tail)
            return true;
        return false;
    }

    private static float FreeRun(SnakeGame game, (int X, int Y) from, SnakeAction action)
    {
        var (dx, dy) = SnakeGame.Delta(action);
        int run = 0;
        int x = from.X;
        int y = from.Y;
        for (int i = 0; i < game.GridSize; i++)
        {
            x += dx;
            y += dy;
            if (x < 0 || y < 0 || x >= game.GridSize || y >= game.GridSize) break;
            if (game.Occupied(x, y) && (x, y) != game.Tail) break;
            run++;
        }
        return run;
    }

    private static bool FoodInDirection((int X, int Y) head, (int X, int Y) food, SnakeAction action)
    {
        if (food.X < 0) return false;
        return action switch
        {
            SnakeAction.Up => food.Y < head.Y && food.X == head.X,
            SnakeAction.Down => food.Y > head.Y && food.X == head.X,
            SnakeAction.Left => food.X < head.X && food.Y == head.Y,
            SnakeAction.Right => food.X > head.X && food.Y == head.Y,
            _ => false,
        };
    }
}
