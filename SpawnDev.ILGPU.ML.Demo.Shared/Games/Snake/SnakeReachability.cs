namespace SpawnDev.ILGPU.ML.Demo.Shared.Games.Snake;

/// <summary>
/// Grid reachability / free-space queries for the safe teacher and encoder.
/// Does not mutate <see cref="SnakeGame"/>.
/// </summary>
public static class SnakeReachability
{
    /// <summary>
    /// Whether <paramref name="cell"/> is blocked after hypothetically stepping onto <paramref name="next"/>.
    /// When <paramref name="eating"/>, the current tail stays occupied (snake grows).
    /// </summary>
    public static bool IsBlockedAfterStep(
        SnakeGame game,
        (int X, int Y) cell,
        (int X, int Y) next,
        bool eating)
    {
        if (cell == next) return false;
        if (cell.X < 0 || cell.Y < 0 || cell.X >= game.GridSize || cell.Y >= game.GridSize)
            return true;

        var tail = game.Tail;
        if (!eating && cell == tail)
            return false;

        return game.Occupied(cell.X, cell.Y);
    }

    /// <summary>Flood-fill free cell count reachable from <paramref name="start"/> after stepping onto it.</summary>
    public static int FloodArea(SnakeGame game, (int X, int Y) start, bool eating)
    {
        int n = game.GridSize;
        if (start.X < 0 || start.Y < 0 || start.X >= n || start.Y >= n)
            return 0;

        var visited = new bool[n * n];
        var q = new Queue<(int X, int Y)>();
        int Idx(int x, int y) => y * n + x;

        q.Enqueue(start);
        visited[Idx(start.X, start.Y)] = true;
        int count = 0;

        while (q.Count > 0)
        {
            var c = q.Dequeue();
            count++;
            for (int a = 0; a < 4; a++)
            {
                var (dx, dy) = SnakeGame.Delta((SnakeAction)a);
                int nx = c.X + dx, ny = c.Y + dy;
                if (nx < 0 || ny < 0 || nx >= n || ny >= n) continue;
                int ni = Idx(nx, ny);
                if (visited[ni]) continue;
                if (IsBlockedAfterStep(game, (nx, ny), start, eating)) continue;
                visited[ni] = true;
                q.Enqueue((nx, ny));
            }
        }

        return count;
    }

    /// <summary>True if <paramref name="to"/> is reachable from <paramref name="from"/> after stepping onto <paramref name="from"/>.</summary>
    public static bool CanReach(SnakeGame game, (int X, int Y) from, (int X, int Y) to, bool eating)
    {
        if (from == to) return true;
        int n = game.GridSize;
        if (from.X < 0 || from.Y < 0 || from.X >= n || from.Y >= n) return false;
        if (to.X < 0 || to.Y < 0 || to.X >= n || to.Y >= n) return false;

        var visited = new bool[n * n];
        var q = new Queue<(int X, int Y)>();
        int Idx(int x, int y) => y * n + x;

        q.Enqueue(from);
        visited[Idx(from.X, from.Y)] = true;

        while (q.Count > 0)
        {
            var c = q.Dequeue();
            for (int a = 0; a < 4; a++)
            {
                var (dx, dy) = SnakeGame.Delta((SnakeAction)a);
                int nx = c.X + dx, ny = c.Y + dy;
                if (nx < 0 || ny < 0 || nx >= n || ny >= n) continue;
                var cell = (nx, ny);
                // Destination may be the tail cell we are trying to reach — allow even if "blocked" as body,
                // because IsBlockedAfterStep frees the tail when !eating; when eating we still allow `to` if it is Tail.
                int ni = Idx(nx, ny);
                if (visited[ni]) continue;

                bool blocked = IsBlockedAfterStep(game, cell, from, eating);
                if (blocked && cell != to) continue;

                if (cell == to) return true;
                visited[ni] = true;
                if (!blocked)
                    q.Enqueue(cell);
                else if (cell == to)
                    return true;
            }
        }

        return false;
    }

    /// <summary>First step of a BFS path from head to food, or null if none.</summary>
    public static SnakeAction? FirstStepToFood(SnakeGame game)
    {
        if (game.Food.X < 0) return null;
        return FirstStepTo(game, game.Head, game.Food, respectReverse: true);
    }

    public static SnakeAction? FirstStepTo(
        SnakeGame game,
        (int X, int Y) from,
        (int X, int Y) to,
        bool respectReverse)
    {
        int n = game.GridSize;
        var came = new SnakeAction?[n * n];
        var visited = new bool[n * n];
        var q = new Queue<(int X, int Y)>();
        int Idx(int x, int y) => y * n + x;

        q.Enqueue(from);
        visited[Idx(from.X, from.Y)] = true;
        bool found = false;

        while (q.Count > 0)
        {
            var (x, y) = q.Dequeue();
            if ((x, y) == to)
            {
                found = true;
                break;
            }

            for (int a = 0; a < 4; a++)
            {
                var action = (SnakeAction)a;
                var (dx, dy) = SnakeGame.Delta(action);
                int nx = x + dx, ny = y + dy;
                if (nx < 0 || ny < 0 || nx >= n || ny >= n) continue;
                int ni = Idx(nx, ny);
                if (visited[ni]) continue;

                // Static occupancy for path search: body blocks except current tail; food is walkable.
                if (game.Occupied(nx, ny) && (nx, ny) != game.Tail && (nx, ny) != to)
                    continue;

                if (respectReverse && (x, y) == from && SnakeGame.IsOpposite(action, game.Direction))
                    continue;

                visited[ni] = true;
                came[ni] = action;
                q.Enqueue((nx, ny));
            }
        }

        if (!found) return null;

        int cx = to.X, cy = to.Y;
        SnakeAction? last = null;
        while ((cx, cy) != from)
        {
            var step = came[Idx(cx, cy)];
            if (step == null) return null;
            last = step;
            var (dx, dy) = SnakeGame.Delta(step.Value);
            cx -= dx;
            cy -= dy;
        }
        return last;
    }

    /// <summary>
    /// After taking <paramref name="action"/>, can the new head still reach the (possibly freed) tail?
    /// </summary>
    public static bool StepKeepsTailReachable(SnakeGame game, SnakeAction action)
    {
        if (!SnakeTeacher.IsLegal(game, action)) return false;
        var (dx, dy) = SnakeGame.Delta(action);
        var next = (X: game.Head.X + dx, Y: game.Head.Y + dy);
        bool eating = next == game.Food;
        return CanReach(game, next, game.Tail, eating);
    }

    public static int StepFloodArea(SnakeGame game, SnakeAction action)
    {
        if (!SnakeTeacher.IsLegal(game, action)) return 0;
        var (dx, dy) = SnakeGame.Delta(action);
        var next = (X: game.Head.X + dx, Y: game.Head.Y + dy);
        bool eating = next == game.Food;
        return FloodArea(game, next, eating);
    }
}
