using SpawnDev.BlazorJS;
using SpawnDev.BlazorJS.JSObjects;
using SpawnDev.ILGPU.ML.Preprocessing;

namespace SpawnDev.ILGPU.ML.Demo.UI.Services;

/// <summary>
/// Reliable image decoding for ML inference.
/// Handles JPEG/PNG/WebP → RGBA int[] with validation.
/// Uses browser-native decoding via canvas (SpawnDev.BlazorJS),
/// with fallback to our PNG decoder for PNG files.
/// </summary>
public class ImageDecodeHelper
{
    private readonly BlazorJSRuntime _js;

    public ImageDecodeHelper(BlazorJSRuntime js) => _js = js;

    /// <summary>
    /// Decode image file bytes (JPEG/PNG/WebP) to RGBA int[] pixels.
    /// Returns (pixels, width, height).
    /// This is the method demo pages should use for image input.
    /// </summary>
    public async Task<(int[] Pixels, int Width, int Height)> DecodeAsync(byte[] imageFileBytes)
    {
        var format = ImageFormat.Detect(imageFileBytes);

        // For PNG, try our native decoder first (avoids canvas overhead)
        if (format == ImageFormat.Format.PNG)
        {
            var pngPixels = PngDecoder.DecodePixels(imageFileBytes);
            if (pngPixels != null)
            {
                var (w, h) = PngDecoder.GetDimensions(imageFileBytes);
                var intPixels = ConvertRGBABytesToInts(pngPixels);
                return (intPixels, w, h);
            }
        }

        // Canvas-based decoding (handles JPEG, WebP, complex PNGs)
        return await DecodeViaCanvasAsync(imageFileBytes);
    }

    /// <summary>
    /// Load pre-decoded RGBA binary (cat_rgba.bin format: 8-byte header + raw pixels).
    /// Used for testing / bypassing image decode.
    /// </summary>
    public static (int[] Pixels, int Width, int Height) LoadRawRGBA(byte[] binData)
    {
        int width = BitConverter.ToInt32(binData, 0);
        int height = BitConverter.ToInt32(binData, 4);
        var pixels = new int[width * height];
        Buffer.BlockCopy(binData, 8, pixels, 0, width * height * 4);
        return (pixels, width, height);
    }

    /// <summary>
    /// Decode via browser canvas (JPEG, WebP, and complex PNGs).
    /// Uses SpawnDev.BlazorJS — no raw JavaScript.
    /// </summary>
    private async Task<(int[] Pixels, int Width, int Height)> DecodeViaCanvasAsync(byte[] imageFileBytes)
    {
        // Create blob from image bytes
        var mimeType = ImageFormat.GetMimeType(ImageFormat.Detect(imageFileBytes));
        using var blob = new Blob(new[] { imageFileBytes }, new BlobOptions { Type = mimeType });

        // Decode via createImageBitmap (browser-native, handles EXIF orientation)
        using var window = _js.Get<Window>("window");
        using var bitmap = await window.CreateImageBitmap(blob);

        int w = (int)bitmap.Width;
        int h = (int)bitmap.Height;

        // Draw to offscreen canvas
        using var canvas = new HTMLCanvasElement();
        canvas.Width = w;
        canvas.Height = h;
        using var ctx = canvas.Get2DContext();
        ctx.DrawImage(bitmap, 0, 0, w, h);

        // Extract pixels
        using var imageData = ctx.GetImageData(0, 0, w, h);
        using var data = imageData.Data;
        int[] pixels = data.Read<int>(); // Each int = RGBA packed as uint32

        return (pixels, w, h);
    }

    /// <summary>
    /// Convert RGBA byte array to int[] (packed RGBA pixels).
    /// Same layout as Buffer.BlockCopy on little-endian.
    /// </summary>
    private static int[] ConvertRGBABytesToInts(byte[] rgba)
    {
        int pixelCount = rgba.Length / 4;
        var result = new int[pixelCount];
        Buffer.BlockCopy(rgba, 0, result, 0, pixelCount * 4);
        return result;
    }

    /// <summary>
    /// Convert RGBA int[] pixels to a displayable PNG data URL.
    /// Uses our PngEncoder — no canvas round-trip needed.
    /// </summary>
    public static string ToDataUrl(int[] rgbaPixels, int width, int height)
    {
        // Convert int[] back to byte[]
        var rgba = new byte[rgbaPixels.Length * 4];
        Buffer.BlockCopy(rgbaPixels, 0, rgba, 0, rgba.Length);

        var pngBytes = PngEncoder.Encode(rgba, width, height);
        return $"data:image/png;base64,{Convert.ToBase64String(pngBytes)}";
    }

    /// <summary>
    /// Validate decoded pixels are reasonable (not all zeros, not all same value).
    /// Returns true if pixels look valid.
    /// </summary>
    public static bool ValidatePixels(int[] pixels, int width, int height)
    {
        if (pixels == null || pixels.Length == 0) return false;
        if (pixels.Length != width * height) return false;

        // Check not all same value
        int firstPixel = pixels[0];
        bool allSame = true;
        for (int i = 1; i < Math.Min(pixels.Length, 1000); i++)
        {
            if (pixels[i] != firstPixel) { allSame = false; break; }
        }
        if (allSame) return false;

        // Check not all zeros
        bool allZero = true;
        for (int i = 0; i < Math.Min(pixels.Length, 1000); i++)
        {
            if (pixels[i] != 0) { allZero = false; break; }
        }
        if (allZero) return false;

        return true;
    }
}
