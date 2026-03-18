using SpawnDev.BlazorJS;
using SpawnDev.BlazorJS.JSObjects;

namespace SpawnDev.ILGPU.ML.Demo.UI.Services;

/// <summary>
/// Browser utility methods using SpawnDev.BlazorJS for all JS interop.
/// </summary>
public class BrowserUtils
{
    private readonly BlazorJSRuntime _js;

    public BrowserUtils(BlazorJSRuntime js)
    {
        _js = js;
    }

    /// <summary>
    /// Copy text to the system clipboard.
    /// </summary>
    public async Task CopyToClipboardAsync(string text)
    {
        using var navigator = _js.Get<Navigator>("navigator");
        using var clipboard = navigator.Clipboard;
        await clipboard.WriteText(text);
    }

    /// <summary>
    /// Download a data URL as a file.
    /// </summary>
    public void DownloadDataUrl(string dataUrl, string filename)
    {
        using var document = _js.Get<Document>("document");
        using var link = document.CreateElement<HTMLAnchorElement>("a");
        link.Href = dataUrl;
        link.Download = filename;
        using var body = document.Body!;
        body.AppendChild(link);
        link.Click();
        body.RemoveChild(link);
    }

    /// <summary>
    /// Download a byte array as a file.
    /// </summary>
    public async Task DownloadBytesAsync(byte[] bytes, string filename, string mimeType = "application/octet-stream")
    {
        using var blob = new Blob(new[] { bytes }, new BlobOptions { Type = mimeType });
        await blob.StartDownload(filename);
    }

    /// <summary>
    /// Download canvas content as a PNG file.
    /// </summary>
    public async Task DownloadCanvasAsync(HTMLCanvasElement canvas, string filename)
    {
        using var blob = await canvas.ToBlobAsync("image/png");
        await blob.StartDownload(filename);
    }
}
