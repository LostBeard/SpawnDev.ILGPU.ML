using System.Reflection;
using System.Text;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Hub;

namespace SpawnDev.ILGPU.ML;

/// <summary>
/// Keeps the WebGPU shaders ILGPU generated in one visit for the next (ILGPU's session export / import, Docs/precompiled-shaders.md in SpawnDev.ILGPU), in the
/// browser's OPFS. Translating a kernel from .NET to WGSL runs in the browser's .NET and is most of a model's cold first run; an imported kernel skips it (and
/// ILGPU's IR). MEASURED (RTX 4070, Chrome, MobileNetV2, ColdStart_MobileNetV2_ShaderCacheImport): a fresh accelerator's first run 279-326 ms with the cache
/// cleared, 45-60 ms with the previous export imported (7 of 7 kernels hit, IR skipped); the export is 201 KB, 30 ms; the import 11-28 ms.
/// </summary>
/// <remarks>
/// <para>Automatic in the browser on WebGPU (<see cref="Enabled"/>): every async session factory and <see cref="InferenceSession.RunAsync"/> import once per page
/// before kernels load; after a run that made ILGPU generate new shaders, the store is rewritten in the background. Nothing on the desktop (the cache is WebGPU).</para>
/// <para>STALENESS. ILGPU keys an entry by the kernel's signature, the device profile and the ILGPU build, not by the kernel's code: a kernel changed and rebuilt
/// without an ILGPU change would import its OLD shader. So the file is named by a hash of the ILGPU build and the build identities (MVID, new on every compile)
/// of this assembly, SpawnDev.ILGPU.ML.Kernels and the app's entry assembly; any rebuild of those starts a fresh file. A kernel in another library changed without
/// an app rebuild is not covered; raised with SpawnDev.ILGPU (an MVID or IL hash in the entry key) 2026-11-26.</para>
/// </remarks>
public static class ShaderCacheStore
{
    /// <summary>On by default in the browser. Off: no import, no export (pure runtime generation, ILGPU's in-page cache only).</summary>
    public static bool Enabled { get; set; } = OperatingSystem.IsBrowser();

    /// <summary>The OPFS directory the store lives in (its own, so the model cache never lists or trims it).</summary>
    public const string DirectoryName = "ilgpu-ml-shaders";

    /// <summary>Kernels the last import registered (0: none stored, another build's, or not imported).</summary>
    public static int LastImported { get; private set; }

    /// <summary>The last export's size in bytes (0: none yet).</summary>
    public static int LastExportBytes { get; private set; }

    private static Task? _import;
    private static int _bytesPerKernel = 29 * 1024;   // MobileNetV2: 201 KB for 7 kernels; refined by every export
    private static long _exportedAtMisses = -1;
    private static Task? _export;
    private static OpfsModelCache? _store;

    private static string? _key;
    /// <summary>The store's file name: a hash of the ILGPU build and the MVIDs of ML, ML.Kernels and the entry assembly (see the remarks).</summary>
    public static string Key => _key ??= ComputeKey();

    private static string ComputeKey()
    {
        var sb = new StringBuilder(ShaderArtifactSerializer.LibraryVersion);
        sb.Append('|').Append(typeof(ShaderCacheStore).Assembly.ManifestModule.ModuleVersionId);
        Guid kernels = Guid.Empty;
        try { kernels = Assembly.Load("SpawnDev.ILGPU.ML.Kernels").ManifestModule.ModuleVersionId; } catch { /* not deployed: only ML's own kernels then */ }
        sb.Append('|').Append(kernels);
        sb.Append('|').Append(Assembly.GetEntryAssembly()?.ManifestModule.ModuleVersionId.ToString() ?? "-");
        uint h = 2166136261u;
        foreach (char c in sb.ToString()) { h ^= c; h *= 16777619u; }
        return $"shaders-{h:x8}.json";
    }

    private static bool Applies(Accelerator accelerator) => Enabled && OperatingSystem.IsBrowser() && accelerator.AcceleratorType == AcceleratorType.WebGPU;

    private static OpfsModelCache Store => _store ??= new OpfsModelCache(SpawnDev.SpawnJS.SpawnJSRuntime.Instance!) { CacheDirectoryName = DirectoryName };

    /// <summary>Imports the stored shaders once per page (later calls return the same task). A failure is logged and ignored: the shaders are then generated as before.</summary>
    public static Task ImportOnceAsync(Accelerator accelerator) => Applies(accelerator) ? _import ??= ImportAsync() : Task.CompletedTask;

    private static async Task ImportAsync()
    {
        try
        {
            var stream = await ((IModelStore)Store).OpenReadAsync(Key).ConfigureAwait(false);
            if (stream == null) return;
            // the file is small (MaxBytes): one large read out of OPFS, then the import from memory. Reading it through the OPFS stream in the parser's own small
            // pieces crossed into JS for each (Bones, 2026-11-26: ~72 ms at the start of a return visit's session create, against 11-28 ms from memory)
            var bytes = new MemoryStream();
            await using (stream.ConfigureAwait(false))
                await stream.CopyToAsync(bytes, 1 << 20).ConfigureAwait(false);
            bytes.Position = 0;
            LastImported = await ShaderArtifactSerializer.ImportCacheFromAsync(bytes).ConfigureAwait(false);
            _exportedAtMisses = ShaderArtifactCache.Misses;   // what is stored is what the page has: export again only when it generates more
        }
        catch (Exception e) { Console.WriteLine($"[ShaderCacheStore] import failed, shaders will be generated: {e.GetType().Name}: {e.Message}"); }
    }

    /// <summary>How long no new shaders must appear before the store is rewritten: a page that keeps compiling (a test suite, a model being explored) exports
    /// once it settles, not after every run (an export writes the whole set).</summary>
    public static int ExportQuietMs { get; set; } = 3000;

    /// <summary>The quiet period before a page's FIRST export: short, because a tab may close soon after its first answer (Bones, 2026-11-26: a page closed
    /// within 3 s of its first run stored nothing, so its next visit generated every shader again; a closing page cannot finish an async OPFS write).</summary>
    public static int FirstExportQuietMs { get; set; } = 250;
    private static bool _exportedThisPage;

    /// <summary>The largest store written (the import cost grows with it: 201 KB imports in 11-28 ms); a larger export is skipped and logged.</summary>
    public static int MaxBytes { get; set; } = 8 * 1024 * 1024;

    /// <summary>After a run: when ILGPU generated shaders the store lacks, rewrite it in the background once no new ones have appeared for <see cref="ExportQuietMs"/>.</summary>
    public static void NoteRun(Accelerator accelerator)
    {
        if (!Applies(accelerator) || ShaderArtifactCache.Misses == _exportedAtMisses) return;
        if (_export is { IsCompleted: false }) return;   // a pending export re-checks the count when its quiet period ends
        _export = ExportWhenQuietAsync();
    }

    private static async Task ExportWhenQuietAsync()
    {
        try
        {
            long seen;
            do { seen = ShaderArtifactCache.Misses; await Task.Delay(_exportedThisPage ? ExportQuietMs : FirstExportQuietMs).ConfigureAwait(false); }
            while (ShaderArtifactCache.Misses != seen);
            _exportedAtMisses = seen;
            // estimate first and build nothing MaxBytes would reject (MEASURED: a test page that compiled hundreds of kernels exported 4.35 MB at a time; built as a
            // string then UTF-8 bytes, ~3x that at once, it ran the page's heap out). ILGPU 5.4.54 writes the export as UTF-8 straight into one buffer.
            long estimate = (long)ShaderArtifactCache.Count * _bytesPerKernel;
            if (estimate > MaxBytes)
            {
                Console.WriteLine($"[ShaderCacheStore] ~{estimate / 1024} KB of shaders ({ShaderArtifactCache.Count} kernels) exceeds MaxBytes ({MaxBytes / 1024} KB): not stored");
                return;
            }
            var buffer = new MemoryStream();
            ShaderArtifactSerializer.ExportCacheTo(buffer);
            int length = (int)buffer.Length;
            if (ShaderArtifactCache.Count > 0) _bytesPerKernel = Math.Max(1, length / ShaderArtifactCache.Count);
            if (length > MaxBytes)
            {
                Console.WriteLine($"[ShaderCacheStore] {length / 1024} KB of shaders exceeds MaxBytes ({MaxBytes / 1024} KB): not stored");
                return;
            }
            buffer.Position = 0;
            await Store.PutAsync(Key, buffer).ConfigureAwait(false);
            LastExportBytes = length;
            _exportedThisPage = true;
            // files of other builds can never be imported again (their key no longer matches): remove them
            foreach (var e in await ((IModelStore)Store).ListAsync().ConfigureAwait(false))
                if (e.Key != Key) await Store.RemoveAsync(e.Key).ConfigureAwait(false);
        }
        catch (Exception e) { Console.WriteLine($"[ShaderCacheStore] export failed: {e.GetType().Name}: {e.Message}"); }
    }

    /// <summary>Waits for a background export in progress, quiet period included (tests, or a page about to close).</summary>
    public static Task FlushAsync() => _export ?? Task.CompletedTask;

    /// <summary>Tests: forget this page's import, so the next session imports the stored file again as a new visit would.</summary>
    internal static void ResetImportForTest() { _import = null; _exportedAtMisses = -1; LastImported = 0; _exportedThisPage = false; }

    /// <summary>Deletes every stored shader file (all builds' keys).</summary>
    public static async Task ClearAsync()
    {
        if (!OperatingSystem.IsBrowser()) return;
        await ((IModelStore)Store).ClearAsync().ConfigureAwait(false);
        _import = null; _exportedAtMisses = -1; LastImported = 0;
    }
}
