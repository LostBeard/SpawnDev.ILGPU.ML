using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.Ltx;
using SpawnDev.ILGPU.ML.Ltx2;
using SpawnDev.ILGPU.ML.Wan;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// What a video page shows while a model file downloads. Opening a hub file waits for its whole download (HubModelSource.OpenAsync ->
/// OpfsModelCache.OpenOrDownloadAsync), so the pipeline's progress message at the moment it opens a file is what the visitor reads for the whole
/// download. 2026-11-27 the live /video page said "encoding the prompt" through the 6.4 GB umT5 download and "decoding the video" through the
/// 1.4 GB VAE download. The rule: every stage says "loading ..." before it opens its first file and names its work only after.
/// </summary>
public abstract partial class MLTestBase
{
    /// <summary>Thrown by <see cref="OpenRecordingSource"/> to end a run at its first file once the message there is recorded.</summary>
    private sealed class StopAtOpenException : Exception { }

    /// <summary>Reports immediately on the caller's thread (Progress&lt;T&gt; posts to the thread pool, so its "latest message" would race the open).</summary>
    private sealed class SyncWanProgress(Action<WanProgress> report) : IProgress<WanProgress> { public void Report(WanProgress value) => report(value); }

    /// <summary>A model source that records, at every open or fetch, the file and the progress message shown at that moment.</summary>
    private sealed class OpenRecordingSource(IModelSource? inner, Func<string> shown) : IModelSource
    {
        public readonly List<(string File, string Shown)> Opens = new();
        private Exception? Record(string file) { lock (Opens) Opens.Add((file, shown())); return inner == null ? new StopAtOpenException() : null; }
        public async Task<Stream> OpenAsync(string repoId, string filePath, CancellationToken cancellationToken = default)
        {
            if (Record(filePath) is { } stop) throw stop;
            return await inner!.OpenAsync(repoId, filePath, cancellationToken);
        }
        public async Task<byte[]> FetchBytesAsync(string repoId, string filePath, CancellationToken cancellationToken = default)
        {
            if (Record(filePath) is { } stop) throw stop;
            return await inner!.FetchBytesAsync(repoId, filePath, cancellationToken);
        }
        public Task<long?> GetSizeAsync(string repoId, string filePath, CancellationToken cancellationToken = default)
            => inner?.GetSizeAsync(repoId, filePath, cancellationToken) ?? Task.FromResult<long?>(null);
    }

    /// <summary>
    /// The first file each video pipeline opens (the text encoder's: umT5 for Wan and LTX, Gemma 3 for LTX-2) must be opened under a "loading" message, not
    /// under the encoding's. No model is read: the source stops the run at its first open. Runs on every backend.
    /// </summary>
    [TestMethod]
    public async Task VideoPipelines_SayLoadingBeforeTheFirstFileOpens() => await RunTest(async accelerator =>
    {
        async Task<string> FirstOpen(Func<IModelSource, IProgress<WanProgress>, Task> run, string name)
        {
            string current = "(no message yet)";
            var source = new OpenRecordingSource(null, () => current);
            try { await run(source, new SyncWanProgress(p => current = $"{p.Stage}: {p.Message}")); }
            catch (StopAtOpenException) { }
            if (source.Opens.Count == 0) throw new Exception($"{name}: the pipeline opened no file");
            var (file, shown) = source.Opens[0];
            if (!shown.Contains("loading the text encoder")) throw new Exception($"{name}: opened '{file}' while the page said \"{shown}\" (should say it is loading the text encoder)");
            return $"{name} '{file}' under \"{shown}\"";
        }
        var wan = await FirstOpen((s, p) => new Wan22Pipeline(accelerator, s, "", Wan22Paths.InRepo()).GenerateAsync(new WanRequest { Prompt = "a fox", Width = 320, Height = 192, Frames = 1, Steps = 1 }, p), "Wan");
        var ltx = await FirstOpen((s, p) => new LtxPipeline(accelerator, s, "", LtxPaths.InRepo()).GenerateAsync(new LtxRequest { Prompt = "a fox", Width = 320, Height = 192, Frames = 9 }, p), "LTX");
        var ltx2 = await FirstOpen((s, p) => new Ltx2Pipeline(accelerator, s, Ltx2Paths.InFolder()).GenerateAsync(new Ltx2Request { Prompt = "a fox", Width = 320, Height = 192, Frames = 9 }, p), "LTX-2");
        Console.WriteLine($"[VideoProgress] {wan}; {ltx}; {ltx2}");
        return $"{wan}; {ltx}; {ltx2}";
    });

    /// <summary>
    /// The whole Wan pipeline on the local model files (WAN_DIR, 8-bit DiT and umT5), smallest clip (320x192, 1 frame, 1 step): the FIRST open of every file
    /// (tokenizer, umT5, DiT, VAE) must happen under a "loading" message of its stage. Later opens of an already opened file (umT5 reopens per layer) are not
    /// downloads and are not checked.
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task Wan_Pipeline_EveryFileOpensUnderALoadingMessage() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is AcceleratorType.WebGL or AcceleratorType.Wasm or AcceleratorType.WebGPU or AcceleratorType.CPU or AcceleratorType.Velocity)
            throw new UnsupportedTestException("the desktop pipeline test needs a desktop GPU (the browser pipeline is separate)");
        var paths = Wan22Paths.InFolder(WanDir, "wan22_ti2v_5b_dit_q8.safetensors", "umt5_xxl_enc_q8.safetensors");
        if (!File.Exists(paths.Dit) || !File.Exists(paths.Vae) || !File.Exists(paths.TextEncoder) || !File.Exists(paths.Tokenizer)) throw new UnsupportedTestException($"Wan exports not found in {WanDir}");
        string current = "(no message yet)";
        var source = new OpenRecordingSource(new LocalFolderModelSource(""), () => current);
        var pipe = new Wan22Pipeline(accelerator, source, "", paths);
        var video = await pipe.GenerateAsync(new WanRequest { Prompt = "A red fox walking through fresh snow.", Width = 320, Height = 192, Frames = 1, Steps = 1 },
            new SyncWanProgress(p => current = $"{p.Stage}: {p.Message}"));
        if (video.Frames != 1) throw new Exception($"expected 1 frame, got {video.Frames}");
        var firsts = source.Opens.GroupBy(o => o.File).Select(g => g.First()).ToList();
        var lines = string.Join("; ", firsts.Select(o => $"{Path.GetFileName(o.File)} under \"{o.Shown}\""));
        Console.WriteLine("[Wan] first opens: " + lines);
        foreach (var name in new[] { paths.Tokenizer, paths.TextEncoder, paths.Dit, paths.Vae })
            if (!firsts.Any(o => o.File == name)) throw new Exception($"{Path.GetFileName(name)} was never opened: {lines}");
        var bad = firsts.Where(o => !o.Shown.Contains(": loading ")).ToList();
        if (bad.Count > 0) throw new Exception("opened while the page named other work: " + string.Join("; ", bad.Select(o => $"{Path.GetFileName(o.File)} under \"{o.Shown}\"")));
        return lines;
    });
}
