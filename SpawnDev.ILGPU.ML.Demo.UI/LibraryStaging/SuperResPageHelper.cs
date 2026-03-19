using SpawnDev.BlazorJS;
using SpawnDev.ILGPU.ML.Demo.UI.Services;
using SpawnDev.ILGPU.ML.Preprocessing;

namespace SpawnDev.ILGPU.ML.Demo.UI.Pages;

/// <summary>
/// Helper logic for the Super Resolution demo page.
/// Handles image decode, center crop to model input size, and result display.
///
/// #1's note: ESPCN model expects 224x224 input (static shapes).
/// Resize input to 224x224 before inference. Output is 672x672 (3x).
/// </summary>
public class SuperResPageHelper
{
    private readonly BlazorJSRuntime _js;
    private readonly ImageDecodeHelper _decoder;

    public SuperResPageHelper(BlazorJSRuntime js)
    {
        _js = js;
        _decoder = new ImageDecodeHelper(js);
    }

    /// <summary>
    /// Prepare image for ESPCN super resolution.
    /// Center crops to 224x224 (model's static input size).
    /// Returns RGBA int[] at 224x224.
    /// </summary>
    public async Task<(int[] Pixels, int Width, int Height, string CroppedDataUrl)?> PrepareImageAsync(byte[] imageFileBytes)
    {
        var (pixels, w, h) = await _decoder.DecodeAsync(imageFileBytes);

        // Convert int[] to byte[] for image ops
        var rgba = new byte[pixels.Length * 4];
        Buffer.BlockCopy(pixels, 0, rgba, 0, rgba.Length);

        // Resize to fit (shortest side to 224)
        var (resized, rw, rh) = ImageOps.ResizePreserveAspect(rgba, w, h, 224, 224);

        // Center crop to 224x224
        byte[] cropped;
        if (rw >= 224 && rh >= 224)
        {
            cropped = ImageOps.CenterCrop(resized, rw, rh, 224, 224);
        }
        else
        {
            // Pad if needed
            int padX = Math.Max(0, (224 - rw) / 2);
            int padY = Math.Max(0, (224 - rh) / 2);
            cropped = ImageOps.PadConstant(resized, rw, rh, padY, 224 - rh - padY, padX, 224 - rw - padX);
        }

        // Convert back to int[]
        var croppedPixels = new int[224 * 224];
        Buffer.BlockCopy(cropped, 0, croppedPixels, 0, 224 * 224 * 4);

        // Generate preview of cropped input
        var croppedDataUrl = ImageDecodeHelper.ToDataUrl(croppedPixels, 224, 224);

        return (croppedPixels, 224, 224, croppedDataUrl);
    }

    /// <summary>
    /// Convert super resolution result (int[] 672x672) to displayable data URL.
    /// </summary>
    public static string ResultToDataUrl(int[] resultPixels, int width, int height)
    {
        return ImageDecodeHelper.ToDataUrl(resultPixels, width, height);
    }
}
