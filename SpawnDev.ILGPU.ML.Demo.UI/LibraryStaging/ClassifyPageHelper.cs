using Microsoft.AspNetCore.Components;
using SpawnDev.BlazorJS;
using SpawnDev.ILGPU.ML.Demo.UI.Services;
using SpawnDev.ILGPU.ML.Preprocessing;

namespace SpawnDev.ILGPU.ML.Demo.UI.Pages;

/// <summary>
/// Helper logic for the Classification demo page.
/// Provides image decode + a "test with known data" bypass
/// to isolate image decode bugs from pipeline bugs.
/// </summary>
public class ClassifyPageHelper
{
    private readonly BlazorJSRuntime _js;
    private readonly HttpClient _http;
    private readonly ImageDecodeHelper _decoder;

    public ClassifyPageHelper(BlazorJSRuntime js, HttpClient http)
    {
        _js = js;
        _http = http;
        _decoder = new ImageDecodeHelper(js);
    }

    /// <summary>
    /// Decode image file bytes (JPEG/PNG) to RGBA int[] via browser canvas.
    /// This is the normal path used when a user drops an image.
    /// </summary>
    public async Task<(int[] Pixels, int Width, int Height)?> DecodeImageAsync(byte[] imageFileBytes)
    {
        try
        {
            var (pixels, w, h) = await _decoder.DecodeAsync(imageFileBytes);

            if (!ImageDecodeHelper.ValidatePixels(pixels, w, h))
            {
                if (VerboseLogging) Console.WriteLine("[ClassifyHelper] WARNING: Decoded pixels failed validation");
                return null;
            }

            if (VerboseLogging) Console.WriteLine($"[ClassifyHelper] Decoded: {w}x{h}, {pixels.Length} pixels");
            return (pixels, w, h);
        }
        catch (Exception ex)
        {
            if (VerboseLogging) Console.WriteLine($"[ClassifyHelper] Decode error: {ex.Message}");
            return null;
        }
    }

    /// <summary>
    /// Load the pre-decoded cat_rgba.bin test image.
    /// Bypasses canvas decode entirely — uses the exact same path as the unit test.
    /// If this produces correct classification results, the pipeline is fine
    /// and the bug is in the canvas image decode path.
    /// </summary>
    public async Task<(int[] Pixels, int Width, int Height)?> LoadTestImageAsync()
    {
        try
        {
            var binData = await _http.GetByteArrayAsync("samples/cat_rgba.bin");
            var (pixels, w, h) = ImageDecodeHelper.LoadRawRGBA(binData);

            if (VerboseLogging) Console.WriteLine($"[ClassifyHelper] Loaded test image: {w}x{h}, {pixels.Length} pixels");

            if (!ImageDecodeHelper.ValidatePixels(pixels, w, h))
            {
                if (VerboseLogging) Console.WriteLine("[ClassifyHelper] WARNING: Test image pixels failed validation");
                return null;
            }

            return (pixels, w, h);
        }
        catch (Exception ex)
        {
            if (VerboseLogging) Console.WriteLine($"[ClassifyHelper] Test image load error: {ex.Message}");
            return null;
        }
    }

    /// <summary>
    /// Compare two classification results to check if they're meaningfully different.
    /// Used to diagnose whether canvas decode produces different inference results.
    /// </summary>
    public static string CompareResults(
        (string Label, float Confidence)[] canvasResults,
        (string Label, float Confidence)[] testResults)
    {
        if (canvasResults.Length == 0 || testResults.Length == 0)
            return "One or both results empty";

        bool sameTopLabel = canvasResults[0].Label == testResults[0].Label;
        float canvasTopConf = canvasResults[0].Confidence;
        float testTopConf = testResults[0].Confidence;

        if (sameTopLabel && MathF.Abs(canvasTopConf - testTopConf) < 0.05f)
            return $"MATCH: Both say '{canvasResults[0].Label}' ({canvasTopConf:P1} vs {testTopConf:P1})";

        if (sameTopLabel)
            return $"SAME LABEL, DIFFERENT CONFIDENCE: '{canvasResults[0].Label}' ({canvasTopConf:P1} vs {testTopConf:P1})";

        return $"DIFFERENT: Canvas='{canvasResults[0].Label}' ({canvasTopConf:P1}), Test='{testResults[0].Label}' ({testTopConf:P1})";
    }

    /// <summary>Gated logging.</summary>
    public bool VerboseLogging { get; set; }
}
