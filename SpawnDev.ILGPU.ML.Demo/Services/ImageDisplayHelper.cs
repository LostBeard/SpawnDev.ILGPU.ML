using SpawnDev.BlazorJS;
using SpawnDev.BlazorJS.JSObjects;

namespace SpawnDev.ILGPU.ML.Demo.Services;

/// <summary>
/// Utility to convert RGBA int[] pixels to a data URL for display in &lt;img&gt; tags.
/// Uses an OffscreenCanvas to pack pixels into a PNG data URL.
/// </summary>
public static class ImageDisplayHelper
{
    /// <summary>
    /// Convert RGBA int[] to a base64 PNG data URL.
    /// Each int is packed as R | (G << 8) | (B << 16) | (A << 24).
    /// </summary>
    public static string ToDataUrl(BlazorJSRuntime js, int[] rgbaPixels, int width, int height)
    {
        using var canvas = new HTMLCanvasElement();
        canvas.Width = width;
        canvas.Height = height;
        using var ctx = canvas.Get2DContext();

        // Create ImageData from pixel array
        using var imageData = ctx.CreateImageData(width, height);
        using var data = imageData.Data;

        // Write pixels — Uint8ClampedArray expects RGBA bytes
        var bytes = new byte[width * height * 4];
        for (int i = 0; i < width * height; i++)
        {
            int pixel = rgbaPixels[i];
            bytes[i * 4] = (byte)(pixel & 0xFF);         // R
            bytes[i * 4 + 1] = (byte)((pixel >> 8) & 0xFF);  // G
            bytes[i * 4 + 2] = (byte)((pixel >> 16) & 0xFF); // B
            bytes[i * 4 + 3] = (byte)((pixel >> 24) & 0xFF); // A
        }
        data.WriteBytes(bytes);

        ctx.PutImageData(imageData, 0, 0);
        return canvas.ToDataURL("image/png");
    }

    /// <summary>
    /// Convert RGBA byte[] to a base64 PNG data URL.
    /// Byte layout: [R, G, B, A, R, G, B, A, ...] — 4 bytes per pixel.
    /// </summary>
    public static string ToDataUrl(BlazorJSRuntime js, byte[] rgbaBytes, int width, int height)
    {
        using var canvas = new HTMLCanvasElement();
        canvas.Width = width;
        canvas.Height = height;
        using var ctx = canvas.Get2DContext();
        using var imageData = ctx.CreateImageData(width, height);
        using var data = imageData.Data;
        data.WriteBytes(rgbaBytes);
        ctx.PutImageData(imageData, 0, 0);
        return canvas.ToDataURL("image/png");
    }
}
