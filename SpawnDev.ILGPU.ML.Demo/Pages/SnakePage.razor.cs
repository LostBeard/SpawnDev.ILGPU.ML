using ILGPU;
using ILGPU.Runtime;
using Microsoft.AspNetCore.Components;
using Microsoft.AspNetCore.Components.Web;
using SpawnDev.ILGPU.ML.Demo.Games.Snake;
using SpawnDev.ILGPU.ML.Demo.Shared.Games.Snake;
using SpawnDev.ILGPU.ML.SystemOne;
using SpawnDev.ILGPU.WebGPU;
using SpawnDev.ILGPU.WebGPU.Backend;
using SpawnDev.SpawnJS;

namespace SpawnDev.ILGPU.ML.Demo.Pages;

public partial class SnakePage : IAsyncDisposable
{
    private const string _howToCs = """
using SpawnDev.ILGPU.ML.SystemOne;
// Snake helpers below are Demo.Shared — not in the NuGet package.
using SpawnDev.ILGPU.ML.Demo.Shared.Games.Snake;

// Package API (generic). Same as SnakeSystemOneSpec.CreateHead(accelerator):
using var head = new SystemOneDecisionHead(accelerator, stateDim: 40, numOptions: 4, hidden: 128);
await SnakeSystemOneTrainer.TrainAsync(head); // BC from the safe teacher

var (action, answer, ms) = await SnakeSystemOnePolicy.DecideAsync(head, game);
game.SetAction(action); // illegal moves already zeroed via allowedMask
""";

    private enum PlayMode { Human, SystemOne }

    [Inject] SpawnJSRuntime JS { get; set; } = default!;

    private readonly SnakeGame _game = new(SnakeGame.DefaultGridSize, seed: 1);
    private readonly Dictionary<string, float> _lastProbs = new()
    {
        ["up"] = 0, ["down"] = 0, ["left"] = 0, ["right"] = 0,
    };

    private PlayMode _mode = PlayMode.Human;
    private SnakeAction _displayAction = SnakeAction.Right;
    private string _selectedBackend = "WebGPU";
    private string? _status;
    private bool _ready;
    private bool _headReady;
    private bool _isTraining;
    private bool _isPlaying;
    private bool _hasCachedHead;
    private float _trainProgress;
    private double _decisionMs;

    private Context? _context;
    private Accelerator? _accelerator;
    private SystemOneDecisionHead? _head;
    private CancellationTokenSource? _loopCts;
    private ElementReference _focusRef;

    private readonly List<Components.BackendSelector.BackendInfo> _backends = new()
    {
        new() { Id = "WebGPU", DisplayName = "WebGPU" },
        new() { Id = "WebGL", DisplayName = "WebGL" },
        new() { Id = "Wasm", DisplayName = "Wasm" },
    };

    protected override async Task OnAfterRenderAsync(bool firstRender)
    {
        if (!firstRender) return;
        try
        {
            var builder = MLContext.Create();
            await builder.AllAcceleratorsAsync();
            _context = builder.ToContext();
            _accelerator = await CreateAcceleratorForBackendAsync(_selectedBackend);
            _ready = _accelerator != null;
            if (_ready)
                await TryLoadCachedHeadAsync();
            else
                _status = $"No {_selectedBackend} device.";
        }
        catch (Exception ex)
        {
            _status = $"Init failed: {ex.Message}";
            Console.WriteLine($"[Snake] Init: {ex.Message}");
        }
        StateHasChanged();
    }

    private async Task TryLoadCachedHeadAsync()
    {
        if (_accelerator == null) return;
        _hasCachedHead = SnakeHeadCache.TryLoad(JS, out var blob);
        if (!_hasCachedHead)
        {
            _status = "Ready — train the decision head, then play.";
            return;
        }

        try
        {
            _head?.Dispose();
            _head = SnakeSystemOneSpec.CreateHead(_accelerator);
            _head.ImportWeights(blob);
            _headReady = true;
            _status = $"Loaded cached head ({blob.Length / 1024f:F1} KB). Switch to System One, or retrain.";
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[Snake] Cache import failed: {ex.Message}");
            SnakeHeadCache.Clear(JS);
            _hasCachedHead = false;
            _head?.Dispose();
            _head = null;
            _headReady = false;
            _status = "Cached head was stale — train a new one.";
        }

        await Task.CompletedTask;
    }

    private async Task HandleBackendChange(string backend)
    {
        if (backend == _selectedBackend) return;
        await PausePlay();
        _selectedBackend = backend;
        _head?.Dispose();
        _head = null;
        _headReady = false;
        _accelerator?.Dispose();
        _accelerator = null;
        _ready = false;
        _accelerator = await CreateAcceleratorForBackendAsync(backend);
        _ready = _accelerator != null;
        if (_ready)
            await TryLoadCachedHeadAsync();
        else
            _status = $"No {backend} device.";
        StateHasChanged();
    }

    private async Task<Accelerator?> CreateAcceleratorForBackendAsync(string backendId)
    {
        if (_context == null) return null;
        try
        {
            return backendId switch
            {
                "WebGPU" => await CreateWebGPUAsync(),
                "WebGL" => await CreateWebGLAsync(),
                "Wasm" => await CreateWasmAsync(),
                _ => null,
            };
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[Snake] Backend {backendId}: {ex.Message}");
            return null;
        }
    }

    private async Task<Accelerator?> CreateWebGPUAsync()
    {
        var devices = _context!.GetDevices<WebGPUILGPUDevice>();
        return devices.Count > 0 ? await devices[0].CreateAcceleratorAsync(_context) : null;
    }

    private async Task<Accelerator?> CreateWebGLAsync()
    {
        var devices = _context!.GetDevices<SpawnDev.ILGPU.WebGL.WebGLILGPUDevice>();
        return devices.Count > 0 ? await devices[0].CreateAcceleratorAsync(_context) : null;
    }

    private async Task<Accelerator?> CreateWasmAsync()
    {
        var devices = _context!.GetDevices<SpawnDev.ILGPU.Wasm.WasmILGPUDevice>();
        return devices.Count > 0 ? await devices[0].CreateAcceleratorAsync(_context) : null;
    }

    private async Task TrainHeadAsync()
    {
        if (_accelerator == null) return;
        await PausePlay();
        _isTraining = true;
        _trainProgress = 2;
        _status = "Training System One head from safe teacher rollouts…";
        StateHasChanged();

        var progress = new Progress<SnakeTrainProgress>(p =>
        {
            _trainProgress = Math.Clamp(p.Fraction * 100f, 0f, 100f);
            _status = p.Loss is float loss
                ? $"{p.Phase} — loss {loss:F3}"
                : p.Phase;
            // Fire-and-forget UI refresh; Progress may callback on thread-pool.
            _ = InvokeAsync(StateHasChanged);
        });

        try
        {
            _head?.Dispose();
            _head = SnakeSystemOneSpec.CreateHead(_accelerator);

            float loss = await SnakeSystemOneTrainer.TrainAsync(
                _head,
                sampleCount: SnakeSystemOneTrainer.DefaultSampleCount,
                epochs: SnakeSystemOneTrainer.DefaultEpochs,
                batchSize: SnakeSystemOneTrainer.DefaultBatchSize,
                learningRate: SnakeSystemOneTrainer.DefaultLearningRate,
                seed: 11,
                progress: progress);

            // Short eval — enough to show quality without another long stall.
            var (mean, max) = await SnakeSystemOneTrainer.EvaluatePolicyScoresAsync(
                _head, games: 4, seed: 90, progress: progress);
            _trainProgress = 100;
            _headReady = true;

            var blob = await _head.ExportWeightsAsync();
            SnakeHeadCache.Save(JS, blob);
            _hasCachedHead = true;

            _status = $"Head ready (loss {loss:F3}). Eval mean score {mean:F1} (max {max}). Cached for next visit.";
        }
        catch (Exception ex)
        {
            _headReady = false;
            _status = $"Train failed: {ex.Message}";
            Console.WriteLine($"[Snake] Train: {ex}");
        }
        finally
        {
            _isTraining = false;
            StateHasChanged();
        }
    }

    private void ClearCachedHead()
    {
        SnakeHeadCache.Clear(JS);
        _hasCachedHead = false;
        _status = _headReady
            ? "Cache cleared — in-memory head still ready. Retrain to refresh cache."
            : "Cache cleared. Train a new head.";
        StateHasChanged();
    }

    private void SetMode(PlayMode mode)
    {
        if (mode == PlayMode.SystemOne && !_headReady) return;
        _mode = mode;
        StateHasChanged();
    }

    private async Task StartPlay()
    {
        if (_isPlaying || !_game.IsAlive) return;
        _isPlaying = true;
        _loopCts = new CancellationTokenSource();
        var token = _loopCts.Token;
        _ = RunLoopAsync(token);
        try { await _focusRef.FocusAsync(); } catch { /* ignore */ }
        StateHasChanged();
    }

    private Task PausePlay()
    {
        _loopCts?.Cancel();
        _loopCts = null;
        _isPlaying = false;
        StateHasChanged();
        return Task.CompletedTask;
    }

    private async Task ResetGame()
    {
        await PausePlay();
        _game.Reset();
        _displayAction = _game.Direction;
        _decisionMs = 0;
        foreach (var k in SnakeStateEncoder.ActionKeys)
            _lastProbs[k] = 0;
        StateHasChanged();
    }

    private async Task RunLoopAsync(CancellationToken token)
    {
        try
        {
            while (!token.IsCancellationRequested && _game.IsAlive)
            {
                await TickOnceAsync();
                await InvokeAsync(StateHasChanged);
                await Task.Delay(100, token);
            }
        }
        catch (OperationCanceledException) { /* pause */ }
        finally
        {
            _isPlaying = false;
            await InvokeAsync(StateHasChanged);
        }
    }

    private async Task TickOnceAsync()
    {
        if (_mode == PlayMode.SystemOne && _head != null)
        {
            var (action, choice, latencyMs) = await SnakeSystemOnePolicy.DecideAsync(_head, _game);
            _decisionMs = latencyMs;
            foreach (var kv in choice.Probabilities)
                _lastProbs[kv.Key] = kv.Value;
            _game.SetAction(action);
            _displayAction = action;
        }
        else
        {
            _displayAction = _game.Direction;
        }

        _game.Tick();
        if (!_game.IsAlive)
            _status = $"Game over — score {_game.Score}. Reset to try again.";
    }

    private void OnKeyDown(KeyboardEventArgs e)
    {
        if (_mode != PlayMode.Human) return;
        SnakeAction? action = e.Key switch
        {
            "ArrowUp" or "w" or "W" => SnakeAction.Up,
            "ArrowDown" or "s" or "S" => SnakeAction.Down,
            "ArrowLeft" or "a" or "A" => SnakeAction.Left,
            "ArrowRight" or "d" or "D" => SnakeAction.Right,
            _ => null,
        };
        if (action == null) return;
        _game.SetAction(action.Value);
        _displayAction = action.Value;
        foreach (var k in SnakeStateEncoder.ActionKeys)
            _lastProbs[k] = k == SnakeStateEncoder.ActionKeys[(int)action.Value] ? 1f : 0f;
    }

    private string CellClass(int x, int y)
    {
        if (_game.Food == (x, y)) return "food";
        if (_game.Head == (x, y)) return "head";
        if (_game.Occupied(x, y)) return "body";
        return "";
    }

    private string ActiveClass(SnakeAction action) =>
        _displayAction == action ? "active" : "";

    private string Pct(string key)
    {
        float p = _lastProbs.GetValueOrDefault(key, 0f);
        return p > 0.005f ? p.ToString("P0") : "";
    }

    public async ValueTask DisposeAsync()
    {
        await PausePlay();
        _head?.Dispose();
        _accelerator?.Dispose();
        _context?.Dispose();
    }
}
