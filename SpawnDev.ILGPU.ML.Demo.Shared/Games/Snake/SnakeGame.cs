namespace SpawnDev.ILGPU.ML.Demo.Shared.Games.Snake;

/// <summary>
/// Classic grid Snake. Pure C# — no GPU. Coordinates are (x,y) with origin top-left;
/// +Y is down (matches canvas).
/// </summary>
public sealed class SnakeGame
{
    public const int DefaultGridSize = 12;

    private readonly LinkedList<(int X, int Y)> _snake = new();
    private readonly HashSet<(int X, int Y)> _occupied = new();
    private readonly Random _rng;
    private SnakeAction _pending = SnakeAction.Right;
    private SnakeAction _direction = SnakeAction.Right;
    private bool _alive = true;
    private bool _grew;

    public int GridSize { get; }
    public int Score { get; private set; }
    public bool IsAlive => _alive;
    public SnakeAction Direction => _direction;
    public (int X, int Y) Food { get; private set; }
    public IReadOnlyCollection<(int X, int Y)> Body => _snake;
    public int Length => _snake.Count;
    public (int X, int Y) Head => _snake.First!.Value;
    /// <summary>Ticks since last food eaten (0 on the eat tick after Score++). Used for hunger / anti-stall.</summary>
    public int StepsSinceFood { get; private set; }

    public SnakeGame(int gridSize = DefaultGridSize, int? seed = null)
    {
        if (gridSize < 4) throw new ArgumentOutOfRangeException(nameof(gridSize));
        GridSize = gridSize;
        _rng = seed.HasValue ? new Random(seed.Value) : new Random();
        Reset();
    }

    public void Reset()
    {
        _snake.Clear();
        _occupied.Clear();
        Score = 0;
        _alive = true;
        _grew = false;
        StepsSinceFood = 0;
        _direction = SnakeAction.Right;
        _pending = SnakeAction.Right;

        int mid = GridSize / 2;
        // Head at mid, body to the left (facing right).
        PlaceInitial(mid, mid);
        PlaceInitial(mid - 1, mid);
        PlaceInitial(mid - 2, mid);
        SpawnFood();
    }

    private void PlaceInitial(int x, int y)
    {
        var cell = (x, y);
        _snake.AddLast(cell);
        _occupied.Add(cell);
    }

    /// <summary>
    /// Queue a direction for the next tick. 180° reverses are ignored (classic Snake).
    /// </summary>
    public void SetAction(SnakeAction action)
    {
        if (IsOpposite(action, _direction)) return;
        _pending = action;
    }

    /// <summary>Advance one step. Returns false if the snake died this tick.</summary>
    public bool Tick()
    {
        if (!_alive) return false;

        _direction = _pending;
        var (dx, dy) = Delta(_direction);
        var head = Head;
        var next = (X: head.X + dx, Y: head.Y + dy);

        if (next.X < 0 || next.Y < 0 || next.X >= GridSize || next.Y >= GridSize)
        {
            _alive = false;
            return false;
        }

        // Tail frees its cell unless we grow this tick — check food first.
        bool eating = next == Food;
        var tail = _snake.Last!.Value;
        if (!eating)
            _occupied.Remove(tail);

        if (_occupied.Contains(next))
        {
            _alive = false;
            if (!eating) _occupied.Add(tail); // restore for consistent dead state
            return false;
        }

        _snake.AddFirst(next);
        _occupied.Add(next);

        if (eating)
        {
            Score++;
            _grew = true;
            StepsSinceFood = 0;
            SpawnFood();
        }
        else
        {
            _grew = false;
            StepsSinceFood++;
            _snake.RemoveLast();
        }

        return true;
    }

    public bool Occupied(int x, int y) => _occupied.Contains((x, y));

    /// <summary>Cell that would be vacated next tick if not eating (current tail).</summary>
    public (int X, int Y) Tail => _snake.Last!.Value;

    public bool GrewLastTick => _grew;

    public static (int Dx, int Dy) Delta(SnakeAction action) => action switch
    {
        SnakeAction.Up => (0, -1),
        SnakeAction.Down => (0, 1),
        SnakeAction.Left => (-1, 0),
        SnakeAction.Right => (1, 0),
        _ => (0, 0),
    };

    public static bool IsOpposite(SnakeAction a, SnakeAction b) =>
        (a == SnakeAction.Up && b == SnakeAction.Down) ||
        (a == SnakeAction.Down && b == SnakeAction.Up) ||
        (a == SnakeAction.Left && b == SnakeAction.Right) ||
        (a == SnakeAction.Right && b == SnakeAction.Left);

    public static SnakeAction Opposite(SnakeAction a) => a switch
    {
        SnakeAction.Up => SnakeAction.Down,
        SnakeAction.Down => SnakeAction.Up,
        SnakeAction.Left => SnakeAction.Right,
        SnakeAction.Right => SnakeAction.Left,
        _ => a,
    };

    private void SpawnFood()
    {
        int free = GridSize * GridSize - _occupied.Count;
        if (free <= 0)
        {
            Food = (-1, -1);
            return;
        }

        int pick = _rng.Next(free);
        for (int y = 0; y < GridSize; y++)
        {
            for (int x = 0; x < GridSize; x++)
            {
                if (_occupied.Contains((x, y))) continue;
                if (pick-- == 0)
                {
                    Food = (x, y);
                    return;
                }
            }
        }
    }
}
