namespace SpawnDev.ILGPU.ML.Demo.UI.Services;

/// <summary>
/// Utility to download ONNX models from HuggingFace Hub for local development.
/// Run from console to populate wwwroot/models/ without manual downloads.
///
/// Usage:
/// <code>
/// await HuggingFaceModelDownloader.DownloadAllDemoModelsAsync("wwwroot/models");
/// </code>
/// </summary>
public static class HuggingFaceModelDownloader
{
    private static readonly (string RepoId, string FileName, string LocalDir)[] DemoModels = new[]
    {
        // Classification
        ("onnx-community/squeezenet1.1-7", "model.onnx", "squeezenet"),
        ("onnxmodelzoo/mobilenetv2-12", "mobilenetv2-12.onnx", "mobilenetv2"),

        // Depth
        ("onnx-community/depth-anything-v2-small", "onnx/model.onnx", "depth-anything-v2-small"),

        // Pose
        ("Xenova/movenet-singlepose-lightning", "onnx/model.onnx", "movenet-lightning"),

        // Style Transfer (from ONNX Model Zoo — these use direct URLs)
    };

    private static readonly (string Url, string LocalDir, string FileName)[] DirectModels = new[]
    {
        ("https://github.com/onnx/models/raw/main/validated/vision/style_transfer/fast_neural_style/model/mosaic-9.onnx", "style-mosaic", "model.onnx"),
        ("https://github.com/onnx/models/raw/main/validated/vision/style_transfer/fast_neural_style/model/candy-9.onnx", "style-candy", "model.onnx"),
        ("https://github.com/onnx/models/raw/main/validated/vision/style_transfer/fast_neural_style/model/rain-princess-9.onnx", "style-rain-princess", "model.onnx"),
        ("https://github.com/onnx/models/raw/main/validated/vision/style_transfer/fast_neural_style/model/udnie-9.onnx", "style-udnie", "model.onnx"),
        ("https://github.com/onnx/models/raw/main/validated/vision/style_transfer/fast_neural_style/model/pointilism-9.onnx", "style-pointilism", "model.onnx"),
        ("https://github.com/onnx/models/raw/main/validated/vision/super_resolution/sub_pixel_cnn_2016/model/super-resolution-10.onnx", "super-resolution", "model.onnx"),
    };

    /// <summary>
    /// Download all demo models to a local directory.
    /// Skips models that already exist.
    /// </summary>
    public static async Task DownloadAllDemoModelsAsync(string outputDir, Action<string>? log = null)
    {
        using var http = new HttpClient();
        http.Timeout = TimeSpan.FromMinutes(10);

        // HuggingFace models
        foreach (var (repoId, fileName, localDir) in DemoModels)
        {
            var outputPath = Path.Combine(outputDir, localDir, "model.onnx");
            if (File.Exists(outputPath))
            {
                log?.Invoke($"SKIP: {localDir}/model.onnx (already exists)");
                continue;
            }

            var url = $"https://huggingface.co/{repoId}/resolve/main/{fileName}";
            log?.Invoke($"Downloading {repoId}/{fileName} → {localDir}/model.onnx ...");

            try
            {
                Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);
                var bytes = await http.GetByteArrayAsync(url);
                await File.WriteAllBytesAsync(outputPath, bytes);
                log?.Invoke($"  Done: {bytes.Length / 1024.0 / 1024.0:F1} MB");
            }
            catch (Exception ex)
            {
                log?.Invoke($"  FAILED: {ex.Message}");
            }
        }

        // Direct URL models
        foreach (var (url, localDir, fileName) in DirectModels)
        {
            var outputPath = Path.Combine(outputDir, localDir, fileName);
            if (File.Exists(outputPath))
            {
                log?.Invoke($"SKIP: {localDir}/{fileName} (already exists)");
                continue;
            }

            log?.Invoke($"Downloading {localDir}/{fileName} ...");

            try
            {
                Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);
                var bytes = await http.GetByteArrayAsync(url);
                await File.WriteAllBytesAsync(outputPath, bytes);
                log?.Invoke($"  Done: {bytes.Length / 1024.0 / 1024.0:F1} MB");
            }
            catch (Exception ex)
            {
                log?.Invoke($"  FAILED: {ex.Message}");
            }
        }

        log?.Invoke("All downloads complete.");
    }
}
