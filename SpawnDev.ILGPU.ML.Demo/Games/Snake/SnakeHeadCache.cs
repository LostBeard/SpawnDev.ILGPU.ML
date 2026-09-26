using SpawnDev.ILGPU.ML.SystemOne;
using SpawnDev.SpawnJS;
using SpawnDev.SpawnJS.JSObjects;

namespace SpawnDev.ILGPU.ML.Demo.Games.Snake;

/// <summary>
/// Persist a trained Snake System One head in browser localStorage (base64 of the FP32 blob).
/// Cache key includes <see cref="SystemOneSnakeSpec.WeightsCacheVersion"/> so encoder/teacher
/// bumps invalidate stale heads automatically.
/// </summary>
internal static class SnakeHeadCache
{
    private static string StorageKey =>
        $"spawndev_ilgpu_ml_snake_s1_v{SystemOneSnakeSpec.WeightsCacheVersion}";

    public static bool TryLoad(SpawnJSRuntime js, out byte[] blob)
    {
        blob = System.Array.Empty<byte>();
        try
        {
            using var window = js.Get<Window>("window");
            using var storage = window.LocalStorage;
            var b64 = storage.GetItem(StorageKey);
            if (string.IsNullOrEmpty(b64))
                return false;
            blob = Convert.FromBase64String(b64);
            return blob.Length > 0;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[Snake] Cache load: {ex.Message}");
            return false;
        }
    }

    public static void Save(SpawnJSRuntime js, byte[] blob)
    {
        try
        {
            using var window = js.Get<Window>("window");
            using var storage = window.LocalStorage;
            storage.SetItem(StorageKey, Convert.ToBase64String(blob));
            Console.WriteLine($"[Snake] Cached head ({blob.Length} bytes) → {StorageKey}");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[Snake] Cache save: {ex.Message}");
        }
    }

    public static void Clear(SpawnJSRuntime js)
    {
        try
        {
            using var window = js.Get<Window>("window");
            using var storage = window.LocalStorage;
            storage.RemoveItem(StorageKey);
        }
        catch { /* ignore */ }
    }
}
