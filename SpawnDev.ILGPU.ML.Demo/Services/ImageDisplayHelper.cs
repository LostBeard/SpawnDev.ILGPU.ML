using SpawnDev.SpawnJS;
using SpawnDev.SpawnJS.JSObjects;

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
    public static string ToDataUrl(SpawnJSRuntime js, int[] rgbaPixels, int width, int height)
    {
        using var canvas = new HTMLCanvasElement();
        canvas.Width = width;
        canvas.Height = height;
        using var ctx = canvas.Get2DContext();

        // Create ImageData from pixel array
        using var imageData = ctx.CreateImageData(width, height);
        using var data = imageData.Data;

        // Write pixels — Uint8ClampedArray expects RGBA bytes
        // a packed RGBA int is R, G, B, A in memory (little endian): one block copy, no per-pixel loop
        var bytes = new byte[width * height * 4];
        Buffer.BlockCopy(rgbaPixels, 0, bytes, 0, bytes.Length);
        data.WriteBytes(bytes);

        ctx.PutImageData(imageData, 0, 0);
        return canvas.ToDataURL("image/png");
    }

    /// <summary>
    /// Convert RGBA byte[] to a base64 PNG data URL.
    /// Byte layout: [R, G, B, A, R, G, B, A, ...] — 4 bytes per pixel.
    /// </summary>
    public static string ToDataUrl(SpawnJSRuntime js, byte[] rgbaBytes, int width, int height)
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

    /// <summary>
    /// An encoded picture (PNG, JPEG, WebP ...) decoded to RGBA bytes by the browser, shrunk so its longer side is at most <paramref name="maxSide"/> pixels (never below 16).
    /// FLUX reference images cost transformer tokens in proportion to their area, so the edit steps and the reference-image list shrink pictures the same way.
    /// </summary>
    public static async Task<(int Width, int Height, byte[] Rgba)> DecodeAsync(SpawnJSRuntime js, byte[] imageBytes, int maxSide = 768)
    {
        using var blob = new Blob(new[] { imageBytes });
        using var window = js.Get<Window>("window");
        using var bitmap = await window.CreateImageBitmap(blob);
        int w = (int)bitmap.Width, h = (int)bitmap.Height;
        double shrink = Math.Min(1.0, (double)maxSide / Math.Max(w, h));
        int nw = Math.Max(16, (int)Math.Round(w * shrink)), nh = Math.Max(16, (int)Math.Round(h * shrink));
        using var canvas = new HTMLCanvasElement();
        canvas.Width = nw; canvas.Height = nh;
        using var ctx = canvas.Get2DContext();
        ctx.DrawImage(bitmap, 0, 0, nw, nh);
        using var imageData = ctx.GetImageData(0, 0, nw, nh);
        using var data = imageData.Data;
        return (nw, nh, data.ReadBytes());
    }
}
