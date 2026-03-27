using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Graph;
using SpawnDev.ILGPU.ML.Operators;
using SpawnDev.ILGPU.ML.Tensors;
using System.Text.Json;

namespace SpawnDev.ILGPU.ML;

/// <summary>
/// High-level API for loading and running ONNX models on GPU.
///
/// Usage (from pre-extracted weights + graph JSON):
///   var session = await InferenceSession.CreateAsync(accelerator, httpClient, "models/my_model");
///   var output = session.Run("input", inputTensor);
///
/// The model directory should contain:
///   model_graph.json — Graph structure (nodes, inputs, outputs)
///   weights_fp16.bin — FP16 weight blob
///   manifest_fp16.json — Weight tensor manifest
/// </summary>
public class InferenceSession : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly OperatorRegistry _registry;
    private readonly GraphExecutor _executor;
    private readonly CompiledGraph _compiled;
    private readonly BufferPool _pool;
    private readonly Dictionary<string, Tensor> _weights;

    /// <summary>Enable diagnostic logging to Console.</summary>
    public static bool VerboseLogging { get; set; }

    /// <summary>Model input names and shapes.</summary>
    public string[] InputNames => _compiled.InputNames;
    public Dictionary<string, int[]> InputShapes => _compiled.InputShapes;

    /// <summary>Model output names and shapes.</summary>
    public string[] OutputNames => _compiled.OutputNames;
    public Dictionary<string, int[]> OutputShapes => _compiled.OutputShapes;

    /// <summary>Number of supported ONNX operators.</summary>
    public int SupportedOpCount => _registry.SupportedOps.Count;

    /// <summary>Number of nodes in the compiled graph.</summary>
    public int NodeCount => _compiled.Nodes.Length;

    /// <summary>Number of weight tensors loaded.</summary>
    public int WeightCount => _weights.Count;

    /// <summary>Try to get a weight tensor by name (for diagnostics).</summary>
    public Tensor? TryGetWeight(string name)
        => _weights.TryGetValue(name, out var t) ? t : null;

    /// <summary>Get all weight tensor names (for diagnostics).</summary>
    public IEnumerable<string> GetWeightNames() => _weights.Keys;

    /// <summary>Distinct operator types used in this model.</summary>
    public string[] OperatorTypes => _compiled.Nodes.Select(n => n.OpType).Distinct().OrderBy(s => s).ToArray();

    /// <summary>Model name (from graph metadata).</summary>
    public string ModelName { get; private set; } = "";

    /// <summary>Access to the underlying GraphExecutor (for KV cache management).</summary>
    public GraphExecutor Executor => _executor;

    private InferenceSession(Accelerator accelerator, OperatorRegistry registry,
        CompiledGraph compiled, GraphExecutor executor, BufferPool pool,
        Dictionary<string, Tensor> weights)
    {
        _accelerator = accelerator;
        _registry = registry;
        _compiled = compiled;
        _executor = executor;
        _pool = pool;
        _weights = weights;
    }

    /// <summary>
    /// Create an InferenceSession from pre-extracted model files.
    /// Loads graph JSON + weight manifest + weight blob from the given base path.
    /// </summary>
    /// <param name="onProgress">Optional progress callback: (stage, percent) where stage is
    /// "graph", "weights", "compile", "ready" and percent is 0-100.</param>
    public static async Task<InferenceSession> CreateAsync(
        Accelerator accelerator, HttpClient http, string basePath,
        Action<string, int>? onProgress = null)
    {
        // Load graph
        onProgress?.Invoke("graph", 0);
        var graphJson = await http.GetStringAsync($"{basePath}/model_graph.json");
        var modelGraph = ModelGraph.FromJson(graphJson);
        onProgress?.Invoke("graph", 100);

        // Load weights
        onProgress?.Invoke("weights", 0);
        var weightLoader = new WeightLoader(accelerator, http);
        await weightLoader.LoadAsync(basePath, onProgress);
        onProgress?.Invoke("weights", 100);

        // For Wasm/WebGL, pre-fetch master weight buffer to CPU once.
        // Avoids GPU→GPU SubView copies that cause Wasm OOB and WebGL peer-to-peer issues.
        bool useCpuStaging = accelerator.AcceleratorType == AcceleratorType.Wasm ||
                             accelerator.AcceleratorType == AcceleratorType.WebGL;
        float[]? cpuWeightsAll = null;
        if (useCpuStaging)
        {
            await accelerator.SynchronizeAsync();
            cpuWeightsAll = await weightLoader.CopyAllToHostAsync();
        }

        // Extract small constant values for shape inference AND runtime operator use.
        // Use ONE shared read buffer to avoid allocating hundreds of tiny GPU buffers.
        modelGraph.ConstantData ??= new Dictionary<string, int[]>();
        var constantFloatValues = new Dictionary<string, float[]>();
        {
            // Find max small tensor size, allocate one shared readback buffer
            int maxSmallElems = 0;
            foreach (var (name, shape) in modelGraph.Initializers)
            {
                int elems = shape.Aggregate(1, (a, b) => a * b);
                if (elems > 0 && elems <= 64 && weightLoader.TryGetView(name) != null)
                    maxSmallElems = Math.Max(maxSmallElems, elems);
            }

            if (maxSmallElems > 0)
            {
                using var readBuf = accelerator.Allocate1D<float>(maxSmallElems);
                foreach (var (name, shape) in modelGraph.Initializers)
                {
                    int elems = shape.Aggregate(1, (a, b) => a * b);
                    if (elems > 0 && elems <= 64)
                    {
                        var view = weightLoader.TryGetView(name);
                        if (view != null)
                        {
                            if (useCpuStaging && cpuWeightsAll != null)
                            {
                                var slice = weightLoader.GetSlice(name);
                                if (slice != null)
                                {
                                    var hostBuf = new float[elems];
                                    Array.Copy(cpuWeightsAll, slice.Value.offset, hostBuf, 0, elems);
                                    constantFloatValues[name] = hostBuf;
                                    modelGraph.ConstantData[name] = hostBuf.Select(v => v < int.MinValue ? int.MinValue : v > int.MaxValue ? int.MaxValue : (int)v).ToArray();
                                }
                            }
                            else
                            {
                                readBuf.View.SubView(0, elems).CopyFrom(view.Value.SubView(0, elems));
                                await accelerator.SynchronizeAsync();
                                var hostBuf = await readBuf.CopyToHostAsync<float>(0, elems);
                                constantFloatValues[name] = hostBuf;
                                modelGraph.ConstantData[name] = hostBuf.Select(v => v < int.MinValue ? int.MinValue : v > int.MaxValue ? int.MaxValue : (int)v).ToArray();
                            }
                        }
                    }
                }
            }
        }

        // Compile graph
        onProgress?.Invoke("compile", 0);
        var registry = new OperatorRegistry(accelerator);
        var compiler = new GraphCompiler(registry);
        var compiled = compiler.Compile(modelGraph);
        onProgress?.Invoke("compile", 100);

        // Create weight tensors from WeightLoader
        // Include both compiled initializer names AND all graph initializers
        // (Constant node outputs are stored as initializers by the extraction script)
        var pool = new BufferPool(accelerator);
        var weights = new Dictionary<string, Tensor>();
        var allInitNames = new HashSet<string>(compiled.InitializerNames);
        foreach (var name in modelGraph.Initializers.Keys)
            allInitNames.Add(name);

        // Reuse pre-fetched CPU weights (already fetched before constant extraction)

        int loadedCount = 0;
        foreach (var name in allInitNames)
        {
            var view = weightLoader.TryGetView(name);
            if (view != null && weightLoader.Shapes.TryGetValue(name, out var shape))
            {
                // Copy each weight into its OWN buffer. WeightLoader uses a single
                // shared buffer with SubViews, but WebGPU doesn't allow binding the
                // same GPUBuffer to multiple storage slots in one kernel dispatch.
                int count = Tensors.TensorHelpers.ElementCount(shape);
                var ownBuf = accelerator.Allocate1D<float>(count);
                if (useCpuStaging && cpuWeightsAll != null)
                {
                    // CPU staging: slice from the pre-fetched master buffer
                    var slice = weightLoader.GetSlice(name);
                    if (slice != null)
                    {
                        var weightSlice = new float[count];
                        Array.Copy(cpuWeightsAll, slice.Value.offset, weightSlice, 0, count);
                        ownBuf.CopyFromCPU(weightSlice);
                    }
                }
                else
                {
                    // GPU→GPU copy via Scale kernel
                    registry.ElementWise.Scale(view.Value.SubView(0, count), ownBuf.View, count, 1f);
                }
                weights[name] = new Tensor(ownBuf.View, shape, name);
                loadedCount++;
            }
        }

        // Log weight loading stats
        int missingCount = allInitNames.Count - loadedCount;
        if (VerboseLogging)
        {
            if (missingCount > 0)
            {
                var missing = allInitNames.Where(n => !weights.ContainsKey(n)).Take(5);
                Console.WriteLine($"[InferenceSession] WARNING: {missingCount} initializers not found in weights. First few: {string.Join(", ", missing)}");
            }
            Console.WriteLine($"[InferenceSession] Loaded {loadedCount}/{allInitNames.Count} weights, {compiled.Nodes.Length} nodes compiled");
        }

        // Create executor with pre-read constant values (avoids GPU→CPU readback during inference)
        var executor = new GraphExecutor(accelerator, compiled, weights, constantFloatValues);
        onProgress?.Invoke("ready", 100);

        return new InferenceSession(accelerator, registry, compiled, executor, pool, weights)
        {
            ModelName = modelGraph.Name
        };
    }

    /// <summary>
    /// Create an InferenceSession from a ModelGraph and pre-loaded weight tensors.
    /// For programmatic use without HTTP loading.
    /// </summary>
    public static InferenceSession Create(
        Accelerator accelerator, ModelGraph graph, Dictionary<string, Tensor> weights)
    {
        // Extract small constant values for shape inference (Reshape targets, etc.)
        // Only do sync CPU readback on desktop backends — browser backends require async.
        // Browser callers should use CreateAsync or CreateFromOnnxAsync instead.
        var constantFloatValues = new Dictionary<string, float[]>();
        bool canSyncCopy = accelerator.AcceleratorType != AcceleratorType.WebGPU
                        && accelerator.AcceleratorType != AcceleratorType.WebGL
                        && accelerator.AcceleratorType != AcceleratorType.Wasm;
        if (graph.ConstantData == null)
        {
            graph.ConstantData ??= new Dictionary<string, int[]>();
        }
        if (canSyncCopy)
        {
            foreach (var (name, shape) in graph.Initializers)
            {
                int elems = shape.Aggregate(1, (a, b) => a * b);
                if (elems > 0 && elems <= 64 && weights.TryGetValue(name, out var tensor))
                {
                    var hostBuf = new float[elems];
                    tensor.Data.SubView(0, elems).CopyToCPU(hostBuf);
                    accelerator.Synchronize();
                    constantFloatValues[name] = hostBuf;
                    graph.ConstantData[name] = hostBuf.Select(v => v < int.MinValue ? int.MinValue : v > int.MaxValue ? int.MaxValue : (int)v).ToArray();
                }
            }
        }

        var registry = new OperatorRegistry(accelerator);
        var compiler = new GraphCompiler(registry);
        var compiled = compiler.Compile(graph);
        var pool = new BufferPool(accelerator);
        var executor = new GraphExecutor(accelerator, compiled, weights, constantFloatValues);
        return new InferenceSession(accelerator, registry, compiled, executor, pool, weights)
        {
            ModelName = graph.Name
        };
    }

    /// <summary>
    /// Create an InferenceSession from any supported model file via HTTP.
    /// Auto-detects format from file extension (.onnx, .tflite) or magic bytes.
    /// </summary>
    /// <param name="inputShapes">Optional: override input shapes for models with dynamic dimensions.
    /// e.g. new Dictionary&lt;string, int[]&gt; { ["pixel_values"] = new[] { 1, 3, 518, 518 } }</param>
    public static async Task<InferenceSession> CreateFromFileAsync(
        Accelerator accelerator, HttpClient http, string modelUrl,
        Action<string, int>? onProgress = null,
        Dictionary<string, int[]>? inputShapes = null,
        bool enableOptimization = true)
    {
        onProgress?.Invoke("download", 0);
        var bytes = await DownloadBytesChunkedAsync(http, modelUrl, onProgress);
        onProgress?.Invoke("download", 100);

        return CreateFromFile(accelerator, bytes, onProgress, inputShapes, enableOptimization);
    }

    /// <summary>
    /// Create an InferenceSession from raw model bytes.
    /// Auto-detects format from magic bytes: ONNX (protobuf) or TFLite (FlatBuffers).
    /// </summary>
    public static InferenceSession CreateFromFile(
        Accelerator accelerator, byte[] modelBytes,
        Action<string, int>? onProgress = null,
        Dictionary<string, int[]>? inputShapes = null,
        bool enableOptimization = true)
    {
        var format = DetectModelFormat(modelBytes);
        return format switch
        {
            ModelFormat.ONNX => CreateFromOnnx(accelerator, modelBytes, onProgress, inputShapes, enableOptimization),
            ModelFormat.TFLite => CreateFromTFLite(accelerator, modelBytes, onProgress),
            ModelFormat.GGUF => CreateFromGGUF(accelerator, modelBytes, onProgress),
            _ => throw new NotSupportedException($"Unknown model format. Expected ONNX (.onnx), TFLite (.tflite), or GGUF (.gguf).")
        };
    }

    /// <summary>Detect model format from magic bytes.</summary>
    public static ModelFormat DetectModelFormat(byte[] data)
    {
        if (data.Length < 8) return ModelFormat.Unknown;

        // GGUF: bytes 0-3 = "GGUF"
        if (data[0] == 'G' && data[1] == 'G' && data[2] == 'U' && data[3] == 'F')
            return ModelFormat.GGUF;

        // TFLite: bytes 4-7 = "TFL3" (FlatBuffers file identifier)
        if (data.Length > 7 && data[4] == 'T' && data[5] == 'F' && data[6] == 'L' && data[7] == '3')
            return ModelFormat.TFLite;

        // ONNX: starts with protobuf varint field tag (typically 0x08 for field 1, varint type)
        // More reliable: check for the "onnx" or "pytorch" producer string within first 64 bytes
        for (int i = 0; i < Math.Min(64, data.Length - 4); i++)
        {
            if (data[i] == 'o' && data[i + 1] == 'n' && data[i + 2] == 'n' && data[i + 3] == 'x')
                return ModelFormat.ONNX;
            if (data[i] == 'p' && data[i + 1] == 'y' && data[i + 2] == 't' && data[i + 3] == 'o')
                return ModelFormat.ONNX; // pytorch producer
        }

        // glTF binary: "glTF" magic (0x46546C67)
        if (data[0] == 0x67 && data[1] == 0x6C && data[2] == 0x54 && data[3] == 0x46)
            return ModelFormat.GLTF;

        // SPZ: gzip header (0x1F 0x8B) — decompress and check for SPZ magic
        if (data[0] == 0x1F && data[1] == 0x8B && Formats.SPZParser.IsValidSPZ(data))
            return ModelFormat.SPZ;

        // PLY: starts with "ply\n"
        if (data[0] == 'p' && data[1] == 'l' && data[2] == 'y' && (data[3] == '\n' || data[3] == '\r'))
            return ModelFormat.PLY;

        // OBJ: starts with "# " (comment) or "v " (vertex) or "o " (object)
        if ((data[0] == '#' && data[1] == ' ') || (data[0] == 'v' && data[1] == ' ') || (data[0] == 'o' && data[1] == ' '))
            return ModelFormat.OBJ;

        // PyTorch: ZIP archive (PK header)
        if (PyTorch.PyTorchLoader.IsPyTorchCheckpoint(data))
            return ModelFormat.PyTorch;

        // SafeTensors: starts with uint64 header size, then '{'
        if (SafeTensors.SafeTensorsParser.IsSafeTensors(data))
            return ModelFormat.SafeTensors;

        // Fallback: if first byte is a protobuf field tag (0x08, 0x0A, etc.)
        // Could be ONNX, TF GraphDef, or CoreML — check for ONNX/pytorch strings first
        if (data[0] == 0x08 || data[0] == 0x0A)
            return ModelFormat.ONNX;

        return ModelFormat.Unknown;
    }

    /// <summary>
    /// Create an InferenceSession from a HuggingFace Hub model with OPFS caching.
    /// Downloads the model on first call; subsequent calls load instantly from cache.
    /// <code>
    /// var hub = new ModelHub(js);
    /// var session = await InferenceSession.CreateFromHuggingFaceAsync(
    ///     accelerator, hub, "onnx-community/squeezenet1.1-7", "model.onnx");
    /// </code>
    /// </summary>
    /// <param name="accelerator">GPU accelerator to compile kernels on</param>
    /// <param name="hub">ModelHub instance (provides OPFS caching)</param>
    /// <param name="repoId">HuggingFace repository ID (e.g., "onnx-community/squeezenet1.1-7")</param>
    /// <param name="filename">File path within the repo (e.g., "model.onnx" or "onnx/model.onnx")</param>
    /// <param name="revision">Git revision (default: "main")</param>
    /// <param name="onProgress">Progress callback: (stage, percent)</param>
    /// <param name="inputShapes">Optional: override input shapes for models with dynamic dimensions</param>
    public static async Task<InferenceSession> CreateFromHuggingFaceAsync(
        Accelerator accelerator, Hub.ModelHub hub,
        string repoId, string filename, string revision = "main",
        Action<string, int>? onProgress = null,
        Dictionary<string, int[]>? inputShapes = null)
    {
        onProgress?.Invoke("download", 0);
        var bytes = await hub.LoadAsync(repoId, filename, revision);
        onProgress?.Invoke("download", 100);

        // Check if this ONNX model uses external data format
        byte[]? externalData = null;
        var format = DetectModelFormat(bytes);
        if (format == ModelFormat.ONNX)
        {
            // Quick parse to check for external data — lightweight since we only scan initializer headers
            var quickParse = Onnx.OnnxParser.Parse(bytes, zeroCopyThreshold: 1024 * 1024);
            if (Onnx.OnnxLoader.HasExternalData(quickParse))
            {
                // Download the external data file (typically model.onnx_data)
                var dataFilename = filename + "_data";
                // Check if any initializer specifies a different location
                var firstExtInit = quickParse.Graph.Initializers.FirstOrDefault(i => i.DataLocation == 1 && i.ExternalData != null);
                if (firstExtInit?.ExternalData?.TryGetValue("location", out var loc) == true && !string.IsNullOrEmpty(loc))
                {
                    // External data location is relative to the ONNX file
                    var dir = filename.Contains('/') ? filename[..filename.LastIndexOf('/')] + "/" : "";
                    dataFilename = dir + loc;
                }
                onProgress?.Invoke("download_data", 0);
                externalData = await hub.LoadAsync(repoId, dataFilename, revision);
                onProgress?.Invoke("download_data", 100);

                // Resolve external data in the already-parsed model
                Onnx.OnnxLoader.ResolveExternalData(quickParse, externalData);
                externalData = null; // Free memory — data is now in the tensors

                // Use the already-parsed model directly to avoid re-parsing
                return CreateFromOnnxParsed(accelerator, quickParse, onProgress, inputShapes);
            }
        }

        return CreateFromFile(accelerator, bytes, onProgress, inputShapes);
    }

    /// <summary>
    /// Create an InferenceSession directly from a .onnx file loaded via HTTP.
    /// No Python extraction step needed — uses the native ONNX protobuf parser.
    /// </summary>
    public static async Task<InferenceSession> CreateFromOnnxAsync(
        Accelerator accelerator, HttpClient http, string onnxUrl,
        Action<string, int>? onProgress = null)
    {
        // Download .onnx file using chunked download to avoid browser WASM OOM
        onProgress?.Invoke("download", 0);
        var onnxBytes = await DownloadBytesChunkedAsync(http, onnxUrl, onProgress);
        onProgress?.Invoke("download", 100);

        return CreateFromOnnx(accelerator, onnxBytes, onProgress);
    }

    /// <summary>
    /// Create an InferenceSession directly from raw .onnx bytes.
    /// No Python extraction step needed — uses the native ONNX protobuf parser.
    /// </summary>
    /// <param name="inputShapes">Optional: override input shapes for models with dynamic dimensions.</param>
    /// <param name="externalData">Optional: raw bytes of the external data file (model.onnx_data)
    /// for models that store weights in a separate file.</param>
    public static InferenceSession CreateFromOnnx(
        Accelerator accelerator, byte[] onnxBytes,
        Action<string, int>? onProgress = null,
        Dictionary<string, int[]>? inputShapes = null,
        bool enableOptimization = true,
        byte[]? externalData = null)
    {
        // Single-parse architecture: parse ONNX protobuf ONCE with zero-copy for large tensors.
        // The parsed model is kept in memory — graph info extracted for compilation,
        // then tensors streamed to GPU from the same parsed result.
        // Avoids scanning 652MB of protobuf twice (was the GPT-2 bottleneck).
        onProgress?.Invoke("parse", 0);
        var parsedModel = Onnx.OnnxParser.Parse(onnxBytes, zeroCopyThreshold: 1024 * 1024);

        // Resolve external data if provided (models with separate weight files)
        if (externalData != null && Onnx.OnnxLoader.HasExternalData(parsedModel))
            Onnx.OnnxLoader.ResolveExternalData(parsedModel, externalData);

        var modelInfo = Onnx.OnnxLoader.ExtractModelInfoFromParsed(parsedModel);
        var cpuSmallWeights = new Dictionary<string, float[]>();
        foreach (var init in parsedModel.Graph.Initializers)
        {
            if (init.DataLocation == 1) continue;
            if (init.ElementCount <= 64)
                cpuSmallWeights[init.Name] = init.ToFloatArray();
        }
        foreach (var node in parsedModel.Graph.Nodes)
        {
            if (node.OpType == "Constant" && node.Outputs.Count > 0)
            {
                var valueAttr = node.Attributes.FirstOrDefault(a => a.Name == "value");
                if (valueAttr != null)
                {
                    if (valueAttr.T != null && valueAttr.T.ElementCount <= 64)
                        cpuSmallWeights[node.Outputs[0]] = valueAttr.T.ToFloatArray();
                    else if (valueAttr.T == null)
                    {
                        if (valueAttr.F != 0 || valueAttr.I != 0)
                            cpuSmallWeights[node.Outputs[0]] = new[] { valueAttr.F != 0 ? valueAttr.F : (float)valueAttr.I };
                    }
                }
            }
        }
        onProgress?.Invoke("parse", 100);

        // Apply input shape overrides (for models with dynamic dimensions)
        if (inputShapes != null)
        {
            foreach (var (name, shape) in inputShapes)
                modelInfo.ValueShapes[name] = shape;
        }

        // Convert OnnxModelInfo → ModelGraph
        ModelGraph graph;
        try { graph = ConvertToModelGraph(modelInfo); }
        catch (Exception ex) { throw new InvalidOperationException($"ConvertToModelGraph failed: {ex.GetType().Name}: {ex.Message}", ex); }

        // Extract small constant values — data is already on CPU (from pass 1), no readback needed
        graph.ConstantData ??= new Dictionary<string, int[]>();
        var constantFloatValues = new Dictionary<string, float[]>();
        foreach (var (name, shape) in graph.Initializers)
        {
            int elems = shape.Aggregate(1, (a, b) => a * b);
            if (elems > 0 && elems <= 64 && cpuSmallWeights.TryGetValue(name, out var data))
            {
                constantFloatValues[name] = data;
                // Populate ConstantData and FloatConstantData for ALL small constants
                // (matching the 64-element cpuSmallWeights threshold). The old ≤16 limit
                // broke Upsample/Resize shape inference: scales computed via
                // Shape→Gather→Mul→Floor→Concat need FloatConstantData at every step.
                graph.ConstantData[name] = data.Select(v => v < int.MinValue ? int.MinValue : v > int.MaxValue ? int.MaxValue : (int)v).ToArray();
                graph.FloatConstantData ??= new Dictionary<string, float[]>();
                graph.FloatConstantData[name] = data.ToArray();
            }
        }

        // Compile graph
        onProgress?.Invoke("compile", 0);
        var registry = new OperatorRegistry(accelerator);
        var compiled = new GraphCompiler(registry) { EnableOptimization = enableOptimization }.Compile(graph);
        onProgress?.Invoke("compile", 100);

        // Pass 2: Stream weights to GPU — one tensor at a time, minimal CPU peak.
        // Re-enumerate the ONNX weight stream. Each tensor is uploaded to GPU immediately,
        // then the CPU float[] goes out of scope and can be collected.
        // GC between passes: free Pass 1 intermediates (cpuSmallWeights, modelInfo, etc.)
        // to maximize headroom for large tensor raw data in Pass 2.
        // Critical for browser WASM where 652MB onnxBytes + 147MB tensor = OOM without GC.
        cpuSmallWeights = null;
        GC.Collect();
        GC.WaitForPendingFinalizers();

        onProgress?.Invoke("upload", 0);
        var pool = new BufferPool(accelerator);
        var gpuWeights = new Dictionary<string, Tensor>();
        int loaded = 0;

        // Stream weights to GPU from the already-parsed model (no second parse).
        // Small tensors: standard float[] path. Large tensors (>1M elements): chunked upload
        // via AllocatePermanentChunked — 1MB chunks, never allocates full float[] (fixes GPT-2 OOM).
        foreach (var (name, tensor) in Onnx.OnnxLoader.StreamTensorsFromParsed(parsedModel))
        {
            if (graph.Initializers.TryGetValue(name, out var shape))
            {
                int expectedElems = shape.Length > 0 ? shape.Aggregate(1, (a, b) => a * b) : 1;
                if (tensor.ElementCount == 0 && expectedElems > 0)
                    gpuWeights[name] = pool.AllocatePermanent(new float[expectedElems], shape, name);
                else
                    gpuWeights[name] = pool.AllocatePermanentChunked(tensor, shape, name);
                loaded++;
            }
        }
        // Create tensors for optimizer-folded constants that aren't in the weight dictionary.
        // The optimizer adds these as initializers but they have no weight data — fill from ConstantData/FloatConstantData.
        foreach (var name in compiled.InitializerNames)
        {
            if (gpuWeights.ContainsKey(name)) continue;
            if (constantFloatValues.TryGetValue(name, out var fData))
            {
                var shape = graph.Initializers.TryGetValue(name, out var s) ? s : new[] { fData.Length };
                gpuWeights[name] = pool.AllocatePermanent(fData, shape, name);
                loaded++;
            }
            else if (graph.FloatConstantData != null && graph.FloatConstantData.TryGetValue(name, out var fcdData))
            {
                var shape = graph.Initializers.TryGetValue(name, out var s) ? s : new[] { fcdData.Length };
                gpuWeights[name] = pool.AllocatePermanent(fcdData, shape, name);
                loaded++;
            }
            else if (graph.ConstantData != null && graph.ConstantData.TryGetValue(name, out var iData))
            {
                var fVals = iData.Select(v => (float)v).ToArray();
                var shape = graph.Initializers.TryGetValue(name, out var s) ? s : new[] { fVals.Length };
                gpuWeights[name] = pool.AllocatePermanent(fVals, shape, name);
                loaded++;
            }
        }
        onProgress?.Invoke("upload", 100);

        if (VerboseLogging) Console.WriteLine($"[InferenceSession] ONNX: {modelInfo.Name}, {compiled.Nodes.Length} nodes, {loaded} weights uploaded");

        var executor = new GraphExecutor(accelerator, compiled, gpuWeights, constantFloatValues);
        onProgress?.Invoke("ready", 100);

        return new InferenceSession(accelerator, registry, compiled, executor, pool, gpuWeights)
        {
            ModelName = modelInfo.Name
        };
    }

    /// <summary>
    /// Create an InferenceSession from an already-parsed ONNX model.
    /// Used when external data has been resolved in-place before compilation.
    /// </summary>
    private static InferenceSession CreateFromOnnxParsed(
        Accelerator accelerator, Onnx.OnnxModelProto parsedModel,
        Action<string, int>? onProgress = null,
        Dictionary<string, int[]>? inputShapes = null,
        bool enableOptimization = true)
    {
        onProgress?.Invoke("parse", 0);
        var modelInfo = Onnx.OnnxLoader.ExtractModelInfoFromParsed(parsedModel);
        var cpuSmallWeights = new Dictionary<string, float[]>();
        foreach (var init in parsedModel.Graph.Initializers)
        {
            if (init.DataLocation == 1) continue;
            if (init.ElementCount <= 64)
                cpuSmallWeights[init.Name] = init.ToFloatArray();
        }
        foreach (var node in parsedModel.Graph.Nodes)
        {
            if (node.OpType == "Constant" && node.Outputs.Count > 0)
            {
                var valueAttr = node.Attributes.FirstOrDefault(a => a.Name == "value");
                if (valueAttr != null)
                {
                    if (valueAttr.T != null && valueAttr.T.ElementCount <= 64)
                        cpuSmallWeights[node.Outputs[0]] = valueAttr.T.ToFloatArray();
                    else if (valueAttr.T == null)
                    {
                        // Scalar Constant: "value" attribute with F (float) or I (int) field
                        // instead of T (tensor). Common in PyTorch-exported ONNX models.
                        if (valueAttr.F != 0 || valueAttr.I != 0)
                            cpuSmallWeights[node.Outputs[0]] = new[] { valueAttr.F != 0 ? valueAttr.F : (float)valueAttr.I };
                    }
                }
            }
        }
        onProgress?.Invoke("parse", 100);

        if (inputShapes != null)
        {
            foreach (var (name, shape) in inputShapes)
                modelInfo.ValueShapes[name] = shape;
        }

        ModelGraph graph;
        try { graph = ConvertToModelGraph(modelInfo); }
        catch (Exception ex) { throw new InvalidOperationException($"ConvertToModelGraph failed: {ex.GetType().Name}: {ex.Message}", ex); }

        graph.ConstantData ??= new Dictionary<string, int[]>();
        var constantFloatValues = new Dictionary<string, float[]>();
        foreach (var (name, shape) in graph.Initializers)
        {
            int elems = shape.Aggregate(1, (a, b) => a * b);
            if (elems > 0 && elems <= 64 && cpuSmallWeights.TryGetValue(name, out var data))
            {
                constantFloatValues[name] = data;
                // Populate ConstantData and FloatConstantData for ALL small constants
                // (matching the 64-element cpuSmallWeights threshold). The old ≤16 limit
                // broke Upsample/Resize shape inference: scales computed via
                // Shape→Gather→Mul→Floor→Concat need FloatConstantData at every step.
                graph.ConstantData[name] = data.Select(v => v < int.MinValue ? int.MinValue : v > int.MaxValue ? int.MaxValue : (int)v).ToArray();
                graph.FloatConstantData ??= new Dictionary<string, float[]>();
                graph.FloatConstantData[name] = data.ToArray();
            }
        }

        onProgress?.Invoke("compile", 0);
        var registry = new OperatorRegistry(accelerator);
        var compiled = new GraphCompiler(registry) { EnableOptimization = enableOptimization }.Compile(graph);
        onProgress?.Invoke("compile", 100);

        cpuSmallWeights = null;
        GC.Collect();
        GC.WaitForPendingFinalizers();

        onProgress?.Invoke("upload", 0);
        var pool = new BufferPool(accelerator);
        var gpuWeights = new Dictionary<string, Tensor>();
        int loaded = 0;

        foreach (var (name, tensor) in Onnx.OnnxLoader.StreamTensorsFromParsed(parsedModel))
        {
            if (graph.Initializers.TryGetValue(name, out var shape))
            {
                int expectedElems = shape.Length > 0 ? shape.Aggregate(1, (a, b) => a * b) : 1;
                if (tensor.ElementCount == 0 && expectedElems > 0)
                    gpuWeights[name] = pool.AllocatePermanent(new float[expectedElems], shape, name);
                else
                    gpuWeights[name] = pool.AllocatePermanentChunked(tensor, shape, name);
                loaded++;
            }
        }
        foreach (var name in compiled.InitializerNames)
        {
            if (gpuWeights.ContainsKey(name)) continue;
            if (constantFloatValues.TryGetValue(name, out var fData))
            {
                var shape = graph.Initializers.TryGetValue(name, out var s) ? s : new[] { fData.Length };
                gpuWeights[name] = pool.AllocatePermanent(fData, shape, name);
                loaded++;
            }
            else if (graph.FloatConstantData != null && graph.FloatConstantData.TryGetValue(name, out var fcdData))
            {
                var shape = graph.Initializers.TryGetValue(name, out var s) ? s : new[] { fcdData.Length };
                gpuWeights[name] = pool.AllocatePermanent(fcdData, shape, name);
                loaded++;
            }
            else if (graph.ConstantData != null && graph.ConstantData.TryGetValue(name, out var iData))
            {
                var fVals = iData.Select(v => (float)v).ToArray();
                var shape = graph.Initializers.TryGetValue(name, out var s) ? s : new[] { fVals.Length };
                gpuWeights[name] = pool.AllocatePermanent(fVals, shape, name);
                loaded++;
            }
        }
        onProgress?.Invoke("upload", 100);

        if (VerboseLogging) Console.WriteLine($"[InferenceSession] ONNX (ext): {modelInfo.Name}, {compiled.Nodes.Length} nodes, {loaded} weights uploaded");

        var executor = new GraphExecutor(accelerator, compiled, gpuWeights, constantFloatValues);
        onProgress?.Invoke("ready", 100);

        return new InferenceSession(accelerator, registry, compiled, executor, pool, gpuWeights)
        {
            ModelName = modelInfo.Name
        };
    }

    /// <summary>
    /// Convert OnnxModelInfo (from native parser) to ModelGraph (used by GraphCompiler).
    /// </summary>
    public static ModelGraph ConvertToModelGraph(Onnx.OnnxModelInfo info)
    {
        var graph = new ModelGraph
        {
            Name = info.Name,
            Inputs = info.InputNames.Select(name => new GraphValueInfo
            {
                Name = name,
                Shape = info.ValueShapes.TryGetValue(name, out var s) ? s : Array.Empty<int>()
            }).ToList(),
            Outputs = info.OutputNames.Select(name => new GraphValueInfo
            {
                Name = name,
                Shape = info.ValueShapes.TryGetValue(name, out var s) ? s : Array.Empty<int>()
            }).ToList(),
            Initializers = new Dictionary<string, int[]>(),
        };

        // Register initializer shapes
        foreach (var initName in info.InitializerNames)
        {
            if (info.ValueShapes.TryGetValue(initName, out var shape))
                graph.Initializers[initName] = shape;
        }

        // Register Constant node outputs as initializers so their weight data gets uploaded to GPU.
        // OnnxLoader.ExtractWeights() already extracted the tensor data into cpuWeightsAll,
        // but without registering them here, the weight upload loop skips them.
        foreach (var node in info.Nodes)
        {
            if (node.OpType == "Constant" && node.Outputs.Length > 0)
            {
                var outputName = node.Outputs[0];
                if (!graph.Initializers.ContainsKey(outputName))
                {
                    // Get shape from ValueShapes if available, otherwise from the weight data size
                    if (info.ValueShapes.TryGetValue(outputName, out var constShape))
                        graph.Initializers[outputName] = constShape;
                    else
                        graph.Initializers[outputName] = new[] { 1 }; // Fallback — will be overridden by actual data
                }
            }
        }

        // Convert nodes
        foreach (var node in info.Nodes)
        {
            var graphNode = new GraphNode
            {
                OpType = node.OpType,
                Inputs = node.Inputs.ToList(),
                Outputs = node.Outputs.ToList(),
            };

            // Convert typed attributes to JsonElement-backed attributes
            // The GraphNode uses JsonElement for serialization compatibility,
            // but we have typed objects from the ONNX parser. Serialize and re-parse.
            // ONNX models can contain NaN/Infinity in attributes (e.g., min/max bounds),
            // so AllowNamedFloatingPointLiterals is required.
            if (node.Attributes.Count > 0)
            {
                var nanSafeOptions = new System.Text.Json.JsonSerializerOptions
                {
                    NumberHandling = System.Text.Json.Serialization.JsonNumberHandling.AllowNamedFloatingPointLiterals
                };
                var jsonDict = new Dictionary<string, System.Text.Json.JsonElement>();
                foreach (var (key, value) in node.Attributes)
                {
                    var json = System.Text.Json.JsonSerializer.Serialize(value, nanSafeOptions);
                    jsonDict[key] = System.Text.Json.JsonDocument.Parse(json).RootElement.Clone();
                }
                graphNode.Attributes = jsonDict;
            }

            graph.Nodes.Add(graphNode);
        }

        return graph;
    }

    /// <summary>
    /// Create an InferenceSession from a .tflite file loaded via HTTP.
    /// No Python extraction step needed — uses the native TFLite FlatBuffers parser.
    /// </summary>
    public static async Task<InferenceSession> CreateFromTFLiteAsync(
        Accelerator accelerator, HttpClient http, string tfliteUrl,
        Action<string, int>? onProgress = null)
    {
        onProgress?.Invoke("download", 0);
        var tfliteBytes = await DownloadBytesChunkedAsync(http, tfliteUrl, onProgress);
        onProgress?.Invoke("download", 100);

        return CreateFromTFLite(accelerator, tfliteBytes, onProgress);
    }

    /// <summary>
    /// Create an InferenceSession directly from raw .tflite bytes.
    /// Uses the native TFLite FlatBuffers parser — zero dependencies.
    /// </summary>
    public static InferenceSession CreateFromTFLite(
        Accelerator accelerator, byte[] tfliteBytes,
        Action<string, int>? onProgress = null)
    {
        // Parse TFLite FlatBuffers
        onProgress?.Invoke("parse", 0);
        var (graph, cpuWeightsAll) = TFLite.TFLiteLoader.LoadModel(tfliteBytes);
        onProgress?.Invoke("parse", 100);

        // Extract small constant values for shape inference
        graph.ConstantData ??= new Dictionary<string, int[]>();
        var constantFloatValues = new Dictionary<string, float[]>();
        foreach (var (name, shape) in graph.Initializers)
        {
            int elems = shape.Aggregate(1, (a, b) => a * b);
            if (elems > 0 && elems <= 64 && cpuWeightsAll.TryGetValue(name, out var data))
            {
                constantFloatValues[name] = data;
                graph.ConstantData[name] = data.Select(v => v < int.MinValue ? int.MinValue : v > int.MaxValue ? int.MaxValue : (int)v).ToArray();
                graph.FloatConstantData ??= new Dictionary<string, float[]>();
                graph.FloatConstantData[name] = data.ToArray();
            }
        }

        // Compile graph
        onProgress?.Invoke("compile", 0);
        var registry = new OperatorRegistry(accelerator);
        var compiled = new GraphCompiler(registry) { EnableOptimization = true }.Compile(graph);
        onProgress?.Invoke("compile", 100);

        // Upload weights to GPU
        onProgress?.Invoke("upload", 0);
        var pool = new BufferPool(accelerator);
        var gpuWeights = new Dictionary<string, Tensor>();
        int loaded = 0;
        foreach (var (name, data) in cpuWeightsAll)
        {
            if (graph.Initializers.TryGetValue(name, out var shape))
            {
                gpuWeights[name] = pool.AllocatePermanent(data, shape, name);
                loaded++;
            }
        }
        onProgress?.Invoke("upload", 100);

        if (VerboseLogging) Console.WriteLine($"[InferenceSession] TFLite: {graph.Name}, {compiled.Nodes.Length} nodes, {loaded} weights uploaded");

        var executor = new GraphExecutor(accelerator, compiled, gpuWeights, constantFloatValues);
        onProgress?.Invoke("ready", 100);

        return new InferenceSession(accelerator, registry, compiled, executor, pool, gpuWeights)
        {
            ModelName = graph.Name
        };
    }

    /// <summary>
    /// Create an InferenceSession from a .gguf file loaded via HTTP.
    /// Parses GGUF metadata, constructs transformer graph, uploads weights.
    /// </summary>
    public static async Task<InferenceSession> CreateFromGGUFAsync(
        Accelerator accelerator, HttpClient http, string ggufUrl,
        Action<string, int>? onProgress = null)
    {
        onProgress?.Invoke("download", 0);
        var ggufBytes = await DownloadBytesChunkedAsync(http, ggufUrl, onProgress);
        onProgress?.Invoke("download", 100);

        return CreateFromGGUF(accelerator, ggufBytes, onProgress);
    }

    /// <summary>
    /// Create an InferenceSession from raw .gguf bytes.
    /// Constructs the transformer graph from architecture metadata.
    /// Note: currently only supports F32/F16 weights. Quantized (Q4/Q8) requires dequantization kernels.
    /// </summary>
    public static InferenceSession CreateFromGGUF(
        Accelerator accelerator, byte[] ggufBytes,
        Action<string, int>? onProgress = null)
    {
        // Parse GGUF
        onProgress?.Invoke("parse", 0);
        var ggufModel = GGUF.GGUFParser.Parse(ggufBytes);
        onProgress?.Invoke("parse", 100);

        // Build transformer graph from architecture metadata
        onProgress?.Invoke("build_graph", 0);
        var (graph, cpuWeightsAll) = GGUF.GGUFGraphBuilder.BuildGraph(ggufModel);
        onProgress?.Invoke("build_graph", 100);

        // Extract small constant values
        graph.ConstantData ??= new Dictionary<string, int[]>();
        var constantFloatValues = new Dictionary<string, float[]>();
        foreach (var (name, shape) in graph.Initializers)
        {
            int elems = shape.Aggregate(1, (a, b) => a * b);
            if (elems > 0 && elems <= 64 && cpuWeightsAll.TryGetValue(name, out var data))
            {
                constantFloatValues[name] = data;
                graph.ConstantData[name] = data.Select(v => v < int.MinValue ? int.MinValue : v > int.MaxValue ? int.MaxValue : (int)v).ToArray();
                graph.FloatConstantData ??= new Dictionary<string, float[]>();
                graph.FloatConstantData[name] = data.ToArray();
            }
        }

        // Compile graph
        onProgress?.Invoke("compile", 0);
        var registry = new OperatorRegistry(accelerator);
        var compiled = new GraphCompiler(registry) { EnableOptimization = true }.Compile(graph);
        onProgress?.Invoke("compile", 100);

        // Upload weights to GPU
        onProgress?.Invoke("upload", 0);
        var pool = new BufferPool(accelerator);
        var gpuWeights = new Dictionary<string, Tensor>();
        int loaded = 0;
        foreach (var (name, data) in cpuWeightsAll)
        {
            if (graph.Initializers.TryGetValue(name, out var shape))
            {
                gpuWeights[name] = pool.AllocatePermanent(data, shape, name);
                loaded++;
            }
        }
        onProgress?.Invoke("upload", 100);

        if (VerboseLogging) Console.WriteLine($"[InferenceSession] GGUF: {ggufModel.Name} ({ggufModel.Architecture}), {compiled.Nodes.Length} nodes, {loaded} weights, {ggufModel.BlockCount} layers");

        var executor = new GraphExecutor(accelerator, compiled, gpuWeights, constantFloatValues);
        onProgress?.Invoke("ready", 100);

        return new InferenceSession(accelerator, registry, compiled, executor, pool, gpuWeights)
        {
            ModelName = graph.Name
        };
    }

    /// <summary>Run inference with named input tensors. Returns named output tensors.</summary>
    public Dictionary<string, Tensor> Run(Dictionary<string, Tensor> inputs)
        => _executor.Run(inputs);

    /// <summary>Async inference — required for browser backends (WebGPU/WebGL/Wasm)
    /// which deadlock on synchronous Synchronize(). Periodically flushes GPU commands.</summary>
    public Task<Dictionary<string, Tensor>> RunAsync(Dictionary<string, Tensor> inputs)
        => _executor.RunAsync(inputs);

    /// <summary>Run inference with a single input. Returns the first output tensor.</summary>
    public Tensor Run(string inputName, Tensor input)
    {
        var outputs = _executor.Run(new Dictionary<string, Tensor> { [inputName] = input });
        return outputs.Values.First();
    }

    /// <summary>Summary string for logging/display.</summary>
    public override string ToString()
    {
        var inShape = InputShapes.Count > 0 ? $"[{string.Join(",", InputShapes.Values.First())}]" : "?";
        var outShape = OutputShapes.Count > 0 ? $"[{string.Join(",", OutputShapes.Values.First())}]" : "?";
        return $"{ModelName}: {NodeCount} nodes, {WeightCount} weights, {string.Join("+", OperatorTypes)}, input={inShape} output={outShape}";
    }

    /// <summary>
    /// Download bytes using stream-based chunked reading when possible.
    /// For desktop backends: uses ResponseHeadersRead for streaming download with progress.
    /// For browser WASM: falls back to ReadAsByteArrayAsync which works with fetch API.
    /// Yields periodically to keep the UI thread responsive during long downloads.
    /// </summary>
    /// <summary>
    /// Download bytes using stream-based chunked reading with progress.
    /// Avoids OOM for large files in browser WASM. Yields periodically for UI responsiveness.
    /// </summary>
    public static async Task<byte[]> DownloadBytesChunkedAsync(HttpClient http, string url,
        Action<string, int>? onProgress = null)
    {
        try
        {
            using var response = await http.GetAsync(url, HttpCompletionOption.ResponseHeadersRead);
            response.EnsureSuccessStatusCode();

            var contentLength = response.Content.Headers.ContentLength;
            using var stream = await response.Content.ReadAsStreamAsync();

            // If content length is known, read in 1MB chunks with progress
            if (contentLength.HasValue && contentLength.Value > 0)
            {
                var result = new byte[contentLength.Value];
                int totalRead = 0;
                int lastPercent = -1;
                int yieldCounter = 0;
                var buffer = new byte[1024 * 1024]; // 1MB chunks
                while (totalRead < result.Length)
                {
                    int read = await stream.ReadAsync(buffer, 0, Math.Min(buffer.Length, result.Length - totalRead));
                    if (read == 0) break;
                    Buffer.BlockCopy(buffer, 0, result, totalRead, read);
                    totalRead += read;

                    int pct = (int)(totalRead * 100L / result.Length);
                    if (pct != lastPercent)
                    {
                        lastPercent = pct;
                        onProgress?.Invoke("download", pct);
                    }

                    // Yield every 4MB to keep UI responsive
                    if (++yieldCounter % 4 == 0)
                        await Task.Yield();
                }

                // Verify we got data — browser WASM may return 0 bytes with streaming
                if (totalRead > 0)
                    return totalRead == result.Length ? result : result[..totalRead];
            }
            else
            {
                // Unknown length: stream into expanding MemoryStream
                using var ms = new MemoryStream();
                var unknownBuffer = new byte[1024 * 1024];
                int unknownYield = 0;
                int bytesRead;
                while ((bytesRead = await stream.ReadAsync(unknownBuffer, 0, unknownBuffer.Length)) > 0)
                {
                    ms.Write(unknownBuffer, 0, bytesRead);
                    if (++unknownYield % 4 == 0)
                        await Task.Yield();
                }
                if (ms.Length > 0)
                    return ms.ToArray();
            }
        }
        catch { /* Streaming download not supported — fall through to ReadAsByteArrayAsync */ }

        // Fallback: standard byte array download (works on all platforms including browser WASM)
        onProgress?.Invoke("download", 0);
        using var fallbackResponse = await http.GetAsync(url);
        fallbackResponse.EnsureSuccessStatusCode();
        var bytes = await fallbackResponse.Content.ReadAsByteArrayAsync();
        onProgress?.Invoke("download", 100);
        return bytes;
    }

    public void Dispose()
    {
        _executor.Dispose();
        _pool.Dispose();
    }
}

/// <summary>Supported model file formats.</summary>
public enum ModelFormat
{
    Unknown,
    ONNX,
    TFLite,
    GGUF,
    SafeTensors,
    TFGraphDef,
    PyTorch,
    CoreML,
    SPZ,
    PLY,
    GLTF,
    OBJ,
}
