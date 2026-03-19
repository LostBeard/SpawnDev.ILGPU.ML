using SpawnDev.BlazorJS;
using SpawnDev.ILGPU.ML.Demo.UI.Services;
using SpawnDev.ILGPU.ML.Preprocessing;

namespace SpawnDev.ILGPU.ML.Demo.UI.Pages;

/// <summary>
/// Helper logic for the Style Transfer demo page.
/// Handles image decode and result display for style transfer.
///
/// #1's note: Models are in wwwroot/models/style-*.
/// Style transfer models expect float [0,255] input (NOT [0,1] normalized).
/// Output is float [0,255] NCHW — clip and convert to RGBA for display.
/// </summary>
public class StylePageHelper
{
    private readonly BlazorJSRuntime _js;
    private readonly ImageDecodeHelper _decoder;

    public static readonly string[] AvailableStyles = { "mosaic", "candy", "rain-princess", "udnie", "pointilism" };

    public StylePageHelper(BlazorJSRuntime js)
    {
        _js = js;
        _decoder = new ImageDecodeHelper(js);
    }

    /// <summary>
    /// Decode image for style transfer input.
    /// Returns RGBA int[] at original dimensions.
    /// </summary>
    public async Task<(int[] Pixels, int Width, int Height)?> DecodeImageAsync(byte[] imageFileBytes)
    {
        try
        {
            var (pixels, w, h) = await _decoder.DecodeAsync(imageFileBytes);
            return (pixels, w, h);
        }
        catch
        {
            return null;
        }
    }

    /// <summary>
    /// Convert style transfer result to displayable data URL.
    /// Result pixels are RGBA int[].
    /// </summary>
    public static string ResultToDataUrl(int[] resultPixels, int width, int height)
    {
        return ImageDecodeHelper.ToDataUrl(resultPixels, width, height);
    }

    /// <summary>
    /// Get the model path for a style name.
    /// </summary>
    public static string GetModelPath(string styleName)
    {
        return $"models/style-{styleName.ToLowerInvariant()}";
    }
}
