using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// Unit tests for HuggingFace CDN integration.
/// These tests hit the real HuggingFace API to verify end-to-end functionality.
/// </summary>
public abstract partial class MLTestBase
{
    private static HttpClient CreateHuggingFaceHttpClient()
    {
        var client = new HttpClient();
        client.Timeout = TimeSpan.FromSeconds(30);
        return client;
    }

    // ═══════════════════════════════════════════════════════════
    //  URL Construction (offline — no network needed)
    // ═══════════════════════════════════════════════════════════

    [TestMethod]
    public async Task HF_GetDownloadUrl_ConstructsCorrectUrl()
    {
        await Task.CompletedTask;

        var url = HuggingFaceClient.GetDownloadUrl("onnx-community/squeezenet1.1-7", "model.onnx");
        if (url != "https://huggingface.co/onnx-community/squeezenet1.1-7/resolve/main/model.onnx")
            throw new Exception($"URL mismatch: {url}");

        var urlWithRevision = HuggingFaceClient.GetDownloadUrl("onnx-community/gpt2", "onnx/model.onnx", "refs/pr/1");
        if (urlWithRevision != "https://huggingface.co/onnx-community/gpt2/resolve/refs/pr/1/onnx/model.onnx")
            throw new Exception($"Revision URL mismatch: {urlWithRevision}");

        var urlSubdir = HuggingFaceClient.GetDownloadUrl("onnx-community/depth-anything-v2-small", "onnx/model.onnx");
        if (urlSubdir != "https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/main/onnx/model.onnx")
            throw new Exception($"Subdir URL mismatch: {urlSubdir}");
    }

    // ═══════════════════════════════════════════════════════════
    //  Search API
    // ═══════════════════════════════════════════════════════════

    [TestMethod]
    public async Task HF_SearchModels_ReturnsResults()
    {
        using var http = CreateHuggingFaceHttpClient();
        var hf = new HuggingFaceClient(http);

        var results = await hf.SearchModelsAsync("squeezenet", library: "onnx", limit: 5);
        if (results.Length == 0)
            throw new Exception("Search returned no results for 'squeezenet' with library=onnx");

        var first = results[0];
        if (string.IsNullOrEmpty(first.Id))
            throw new Exception("First result has empty Id");
        if (first.Downloads <= 0)
            throw new Exception($"First result '{first.Id}' has {first.Downloads} downloads — expected > 0");

        Console.WriteLine($"[HF_Search] Found {results.Length} results. Top: {first.Id} ({first.Downloads:N0} downloads)");
    }

    [TestMethod]
    public async Task HF_SearchModels_FilterByTask()
    {
        using var http = CreateHuggingFaceHttpClient();
        var hf = new HuggingFaceClient(http);

        var results = await hf.SearchModelsAsync(
            pipelineTag: HuggingFaceClient.Tasks.DepthEstimation,
            library: HuggingFaceClient.Libraries.ONNX,
            limit: 5);

        if (results.Length == 0)
            throw new Exception("No depth-estimation ONNX models found");

        foreach (var m in results)
        {
            if (m.PipelineTag != "depth-estimation")
                throw new Exception($"Model '{m.Id}' has pipeline_tag '{m.PipelineTag}', expected 'depth-estimation'");
        }

        Console.WriteLine($"[HF_SearchByTask] Found {results.Length} depth-estimation ONNX models. Top: {results[0].Id}");
    }

    // ═══════════════════════════════════════════════════════════
    //  Model Info API
    // ═══════════════════════════════════════════════════════════

    [TestMethod]
    public async Task HF_GetModelInfo_ReturnsDetailedInfo()
    {
        using var http = CreateHuggingFaceHttpClient();
        var hf = new HuggingFaceClient(http);

        var info = await hf.GetModelInfoAsync(ModelHub.KnownModels.SqueezeNet);

        if (info.Id != ModelHub.KnownModels.SqueezeNet)
            throw new Exception($"Id mismatch: expected '{ModelHub.KnownModels.SqueezeNet}', got '{info.Id}'");
        if (info.Siblings == null || info.Siblings.Length == 0)
            throw new Exception("No siblings (files) returned in model info");
        if (info.Downloads <= 0)
            throw new Exception($"Downloads is {info.Downloads}, expected > 0");

        // Verify we can find the model.onnx file in siblings
        var onnxFiles = info.GetOnnxFiles();
        if (onnxFiles.Length == 0)
            throw new Exception("No .onnx files found in siblings");

        Console.WriteLine($"[HF_ModelInfo] {info.Id}: {info.Downloads:N0} downloads, {info.Likes} likes, " +
            $"{info.Siblings.Length} files, {onnxFiles.Length} ONNX files");
        foreach (var f in onnxFiles)
            Console.WriteLine($"  ONNX: {f.Filename} ({f.SizeFormatted})");
    }

    [TestMethod]
    public async Task HF_GetModelInfo_HasOnnxProperty()
    {
        using var http = CreateHuggingFaceHttpClient();
        var hf = new HuggingFaceClient(http);

        var info = await hf.GetModelInfoAsync(ModelHub.KnownModels.SqueezeNet);

        if (!info.HasOnnx)
            throw new Exception($"HasOnnx is false for {info.Id} — expected true");

        Console.WriteLine($"[HF_HasOnnx] {info.Id}: HasOnnx={info.HasOnnx}, HasGGUF={info.HasGGUF}");
    }

    // ═══════════════════════════════════════════════════════════
    //  File Listing API
    // ═══════════════════════════════════════════════════════════

    [TestMethod]
    public async Task HF_ListRepoFiles_ReturnsFiles()
    {
        using var http = CreateHuggingFaceHttpClient();
        var hf = new HuggingFaceClient(http);

        var files = await hf.ListRepoFilesAsync(ModelHub.KnownModels.SqueezeNet);

        if (files.Length == 0)
            throw new Exception("No files returned from repo listing");

        var onnxFile = files.FirstOrDefault(f => f.Path.EndsWith(".onnx"));
        if (onnxFile == null)
            throw new Exception("No .onnx file found in repo listing");
        if (onnxFile.EffectiveSize <= 0)
            throw new Exception($"ONNX file '{onnxFile.Path}' has size {onnxFile.EffectiveSize} — expected > 0");

        Console.WriteLine($"[HF_ListFiles] {ModelHub.KnownModels.SqueezeNet}: {files.Length} entries");
        foreach (var f in files.Where(f => f.IsFile))
            Console.WriteLine($"  {f.Path} ({f.EffectiveSize:N0} bytes)");
    }

    // ═══════════════════════════════════════════════════════════
    //  Download (small model to verify CDN works)
    // ═══════════════════════════════════════════════════════════

    [TestMethod]
    public async Task HF_DownloadFile_SmallModel()
    {
        using var http = CreateHuggingFaceHttpClient();
        var hf = new HuggingFaceClient(http);

        long lastReceived = 0;
        int progressCalls = 0;

        // Download SqueezeNet (~4.8 MB) — small enough for a test
        var bytes = await hf.DownloadFileAsync(
            ModelHub.KnownModels.SqueezeNet, "squeezenet1.1-7.onnx",
            onProgress: (received, total) =>
            {
                lastReceived = received;
                progressCalls++;
            });

        if (bytes.Length < 1_000_000)
            throw new Exception($"Downloaded {bytes.Length} bytes — too small for SqueezeNet (expected ~4.8 MB)");
        if (bytes.Length > 10_000_000)
            throw new Exception($"Downloaded {bytes.Length} bytes — too large for SqueezeNet (expected ~4.8 MB)");
        if (progressCalls == 0)
            throw new Exception("Progress callback was never called");

        // Verify it's a valid ONNX file (check magic bytes / format detection)
        var format = InferenceSession.DetectModelFormat(bytes);
        if (format != ModelFormat.ONNX)
            throw new Exception($"Downloaded file detected as {format}, expected ONNX");

        Console.WriteLine($"[HF_Download] SqueezeNet: {bytes.Length:N0} bytes, " +
            $"{progressCalls} progress callbacks, format={format}");
    }

    // ═══════════════════════════════════════════════════════════
    //  Direct URL Download (ONNX Model Zoo)
    // ═══════════════════════════════════════════════════════════

    [TestMethod]
    public async Task HF_DirectUrl_DownloadStyleModel()
    {
        using var http = CreateHuggingFaceHttpClient();
        var hf = new HuggingFaceClient(http);

        // Super Resolution is tiny (~236 KB) — perfect for a quick test
        var bytes = await hf.DownloadFileAsync(ModelHub.KnownModels.SuperResolution, "super-resolution-10.onnx");

        if (bytes.Length < 100_000)
            throw new Exception($"Downloaded {bytes.Length} bytes — too small for Super Resolution model");
        if (bytes.Length > 1_000_000)
            throw new Exception($"Downloaded {bytes.Length} bytes — too large for Super Resolution model");

        var format = InferenceSession.DetectModelFormat(bytes);
        if (format != ModelFormat.ONNX)
            throw new Exception($"Downloaded file detected as {format}, expected ONNX");

        Console.WriteLine($"[HF_Download] Super Resolution: {bytes.Length:N0} bytes, format={format}");
    }

    // ═══════════════════════════════════════════════════════════
    //  End-to-End: Download → Parse → Load InferenceSession
    // ═══════════════════════════════════════════════════════════

    [TestMethod]
    public async Task HF_DownloadAndLoadSession_SqueezeNet() => await RunTest(async accelerator =>
    {
        using var http = CreateHuggingFaceHttpClient();
        var hf = new HuggingFaceClient(http);

        // Download from HuggingFace CDN
        var url = HuggingFaceClient.GetDownloadUrl(ModelHub.KnownModels.SqueezeNet, "squeezenet1.1-7.onnx");
        var bytes = await hf.DownloadAsync(url);

        // Load into InferenceSession (compiles kernels on accelerator)
        using var session = InferenceSession.CreateFromFile(accelerator, bytes);

        if (session == null)
            throw new Exception("CreateFromFile returned null");

        Console.WriteLine($"[HF_E2E] SqueezeNet loaded from HuggingFace CDN → {session}");
    });

    [TestMethod]
    public async Task HF_DownloadAndLoadSession_SuperResolution() => await RunTest(async accelerator =>
    {
        using var http = CreateHuggingFaceHttpClient();
        var hf = new HuggingFaceClient(http);

        // Download from HuggingFace CDN
        var bytes = await hf.DownloadFileAsync(ModelHub.KnownModels.SuperResolution, "super-resolution-10.onnx");

        using var session = InferenceSession.CreateFromFile(accelerator, bytes);

        if (session == null)
            throw new Exception("CreateFromFile returned null");

        Console.WriteLine($"[HF_E2E] Super Resolution loaded from HuggingFace CDN → {session}");
    });

    // ═══════════════════════════════════════════════════════════
    //  Known Models Constants Validation
    // ═══════════════════════════════════════════════════════════

    [TestMethod]
    public async Task HF_KnownModels_AllReposExist()
    {
        using var http = CreateHuggingFaceHttpClient();
        var hf = new HuggingFaceClient(http);

        // Verify all KnownModels repo IDs are valid (API returns 200)
        var knownRepos = new[]
        {
            (nameof(ModelHub.KnownModels.SqueezeNet), ModelHub.KnownModels.SqueezeNet),
            (nameof(ModelHub.KnownModels.MobileNetV2), ModelHub.KnownModels.MobileNetV2),
            (nameof(ModelHub.KnownModels.DepthAnythingV2Small), ModelHub.KnownModels.DepthAnythingV2Small),
            (nameof(ModelHub.KnownModels.MoveNetLightning), ModelHub.KnownModels.MoveNetLightning),
            (nameof(ModelHub.KnownModels.DistilBertSST2), ModelHub.KnownModels.DistilBertSST2),
            (nameof(ModelHub.KnownModels.GPT2), ModelHub.KnownModels.GPT2),
            (nameof(ModelHub.KnownModels.WhisperTiny), ModelHub.KnownModels.WhisperTiny),
        };

        var errors = new List<string>();
        foreach (var (name, repoId) in knownRepos)
        {
            try
            {
                var info = await hf.GetModelInfoAsync(repoId);
                if (string.IsNullOrEmpty(info.Id))
                    errors.Add($"{name} ({repoId}): returned empty Id");
                else
                    Console.WriteLine($"  OK: {name} → {info.Id} ({info.Downloads:N0} downloads)");
            }
            catch (Exception ex)
            {
                errors.Add($"{name} ({repoId}): {ex.Message}");
            }
        }

        if (errors.Count > 0)
            throw new Exception($"KnownModels validation failed:\n  {string.Join("\n  ", errors)}");

        Console.WriteLine($"[HF_KnownModels] All {knownRepos.Length} repos verified.");
    }

    // ═══════════════════════════════════════════════════════════
    //  Style & Super Resolution Models Validation
    // ═══════════════════════════════════════════════════════════

    [TestMethod]
    public async Task HF_KnownModels_StyleAndSuperRes_AllExist()
    {
        using var http = CreateHuggingFaceHttpClient();
        var hf = new HuggingFaceClient(http);

        var repos = new[]
        {
            (nameof(ModelHub.KnownModels.StyleMosaic), ModelHub.KnownModels.StyleMosaic),
            (nameof(ModelHub.KnownModels.StyleCandy), ModelHub.KnownModels.StyleCandy),
            (nameof(ModelHub.KnownModels.StyleRainPrincess), ModelHub.KnownModels.StyleRainPrincess),
            (nameof(ModelHub.KnownModels.StyleUdnie), ModelHub.KnownModels.StyleUdnie),
            (nameof(ModelHub.KnownModels.StylePointilism), ModelHub.KnownModels.StylePointilism),
            (nameof(ModelHub.KnownModels.SuperResolution), ModelHub.KnownModels.SuperResolution),
        };

        var errors = new List<string>();
        foreach (var (name, repoId) in repos)
        {
            try
            {
                var info = await hf.GetModelInfoAsync(repoId);
                if (string.IsNullOrEmpty(info.Id))
                    errors.Add($"{name} ({repoId}): returned empty Id");
                else
                    Console.WriteLine($"  OK: {name} → {info.Id} ({info.Downloads:N0} downloads)");
            }
            catch (Exception ex)
            {
                errors.Add($"{name} ({repoId}): {ex.Message}");
            }
        }

        if (errors.Count > 0)
            throw new Exception($"KnownModels style/superres validation failed:\n  {string.Join("\n  ", errors)}");

        Console.WriteLine($"[HF_KnownModels] All {repos.Length} style/super-resolution repos verified.");
    }
}
