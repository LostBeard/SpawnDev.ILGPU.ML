using System.Text;
using ILGPU;
using ILGPU.Runtime;
using Microsoft.AspNetCore.Components;
using SpawnDev.SpawnJS;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnDev.ILGPU.WebGPU;

namespace SpawnDev.ILGPU.ML.Demo.Pages;

public partial class SentimentPage : IDisposable
{
    [Inject] SpawnJSRuntime JS { get; set; } = default!;
    [Inject] HttpClient Http { get; set; } = default!;

    private Context? _context;
    private Accelerator? _accelerator;
    private TextClassificationPipeline? _pipeline;
    private LoadedTokenizer? _tokenizer;

    // OPT-IN: nothing downloads on page entry — DistilBERT-SST2 is ~257 MB. The model + tokenizer load
    // on the first Analyze click (see Analyze). Reference opt-in pattern: AiChatPage/ClassifyPage.

    private async Task LoadBackendAndModelAsync()
    {
        try
        {
            _isModelLoading = true;
            _isModelLoaded = false;
            _error = null;
            _modelProgress = 5;
            StateHasChanged();

            if (_context == null)
            {
                var builder = MLContext.Create();
                await builder.AllAcceleratorsAsync();
                _context = builder.ToContext();
            }

            _modelProgress = 15;
            StateHasChanged();
            _accelerator = await CreateAcceleratorForBackendAsync(_selectedBackend);
            if (_accelerator == null)
            {
                _error = $"No {_selectedBackend} device available";
                _isModelLoading = false;
                StateHasChanged();
                return;
            }

            using var hub = new ModelHub(JS);

            // Real WordPiece tokenizer straight from the model repo's tokenizer.json.
            _modelProgress = 30;
            StateHasChanged();
            var tokJson = await hub.LoadAsync(ModelHub.KnownModels.DistilBertSST2, "tokenizer.json");
            _tokenizer = TokenizerLoader.FromTokenizerJson(Encoding.UTF8.GetString(tokJson));

            // The fp32 ONNX classifier.
            _modelProgress = 50;
            StateHasChanged();
            var modelBytes = await hub.LoadAsync(ModelHub.KnownModels.DistilBertSST2, "onnx/model.onnx");
            var session = InferenceSession.CreateFromFile(_accelerator, modelBytes);
            _pipeline = new TextClassificationPipeline(session, _accelerator);

            _isModelLoaded = true;
            _modelProgress = 100;
            Console.WriteLine($"[Sentiment] Model loaded on {_selectedBackend}: tokenizer={_tokenizer.Tokenizer.GetType().Name}");
        }
        catch (Exception ex)
        {
            _error = $"Failed to load: {ex.Message}";
            Console.WriteLine($"[Sentiment] Load error: {ex.Message}");
        }
        finally
        {
            _isModelLoading = false;
            StateHasChanged();
        }
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
                _ => null
            };
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[Sentiment] Backend {backendId} failed: {ex.Message}");
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

    private async Task HandleBackendChange(string backend)
    {
        if (backend == _selectedBackend) return;
        bool wasLoaded = _isModelLoaded;
        _selectedBackend = backend;
        _hasResult = false;

        _pipeline?.Dispose();
        _pipeline = null;
        _tokenizer = null;
        _accelerator?.Dispose();
        _accelerator = null;
        _isModelLoaded = false;

        // Only re-load if the model was ALREADY in use — switching backend before first use must not
        // trigger a 257 MB download (opt-in). The next Analyze loads on the newly selected backend.
        if (wasLoaded)
            await LoadBackendAndModelAsync();
        else
            StateHasChanged();
    }

    private async Task Analyze()
    {
        if (string.IsNullOrWhiteSpace(_text)) return;

        // Opt-in load: pull the model on first Analyze, not on page entry.
        if (!_isModelLoaded && !_isModelLoading)
            await LoadBackendAndModelAsync();
        if (!_isModelLoaded || _pipeline == null || _tokenizer == null) return;

        _isRunning = true;
        _hasResult = false;
        _error = null;
        StateHasChanged();
        await Task.Yield();

        try
        {
            var r = await _pipeline.ClassifyAsync(_text, _tokenizer);
            _inferenceMs = r.InferenceTimeMs;
            _resultPositive = r.TopLabel == "POSITIVE";
            _resultConfidence = r.TopConfidence;
            _hasResult = true;
            Console.WriteLine($"[Sentiment] {r.InferenceTimeMs:F1}ms — {r.TopLabel} ({r.TopConfidence:P1})");
        }
        catch (Exception ex)
        {
            _error = $"Analysis failed: {ex.Message}";
            Console.WriteLine($"[Sentiment] Error: {ex.Message}");
        }

        _isRunning = false;
        StateHasChanged();
    }

    public void Dispose()
    {
        _pipeline?.Dispose();
        _accelerator?.Dispose();
        _context?.Dispose();
    }
}
