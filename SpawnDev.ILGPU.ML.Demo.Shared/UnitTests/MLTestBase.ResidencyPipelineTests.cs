using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Flux2;
using SpawnDev.ILGPU.ML.Flux3;
using SpawnDev.ILGPU.ML.Ltx;
using SpawnDev.ILGPU.ML.Trellis2;
using SpawnDev.ILGPU.ML.Wan;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// The pipelines that keep big models (FLUX.2, FLUX 3, LTX) on a <see cref="ModelResidency"/>: the same output, bit for bit, when the model was kept, and when the manager dropped it and the next run
/// loaded it again. The weights are loaded once while kept, and once more after the drop.
/// </summary>
public abstract partial class MLTestBase
{
    private static void Rp(bool ok, string what) { if (!ok) throw new Exception(what); }

    private static void Rp_SameBytes(byte[] a, byte[] b, string what)
    {
        Rp(a.Length == b.Length, $"{what}: {a.Length} vs {b.Length} bytes");
        int diff = 0, first = -1;
        for (int i = 0; i < a.Length; i++) if (a[i] != b[i]) { diff++; if (first < 0) first = i; }
        Rp(diff == 0, $"{what}: {diff} of {a.Length} bytes differ (first at {first})");
    }

    private static async Task<ModelResidency> Rp_Residency(Accelerator accelerator, List<ModelEviction> evicted, List<string> log)
    {
        var r = new ModelResidency(accelerator) { Log = log.Add };
        r.Evicted += evicted.Add;
        await Task.CompletedTask;
        return r;
    }

    /// <summary>FLUX.2 [klein] 4-bit on a manager: kept = not loaded again and the same picture; dropped = loaded again and the same picture.</summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task Flux2_Residency_KeptAndReloaded_SamePicture() => await RunTest(async accelerator =>
    {
        var dir = F2_Require(accelerator);
        var enc = Path.Combine(dir, "weights", "text_encoder_q4.safetensors"); var trf = Path.Combine(dir, "weights", "transformer_q4.safetensors");
        if (!File.Exists(enc) || !File.Exists(trf)) throw new UnsupportedTestException("the 4-bit exports are not found (quantize_flux2.py)");
        var tok = new Qwen2Tokenizer(File.ReadAllText(Path.Combine(dir, "klein4b", "tokenizer", "tokenizer.json")));
        using var vae = InferenceSession.CreateFromFile(accelerator, File.ReadAllBytes(Path.Combine(dir, "weights", "vae_decoder.onnx")), inputShapes: new Dictionary<string, int[]> { ["latents"] = new[] { 1, 32, 64, 64 } });
        int trfLoads = 0;
        var evicted = new List<ModelEviction>(); var log = new List<string>();
        await using var r = await Rp_Residency(accelerator, evicted, log);
        using var pipe = new Flux2Pipeline(accelerator, tok,
            () => Task.FromResult(new Flux2Weights(accelerator, GpuWeights.LoadFile(accelerator, enc))),
            () => { trfLoads++; return Task.FromResult(new Flux2Weights(accelerator, GpuWeights.LoadFile(accelerator, trf))); }, vae, F2_Norm(dir)) { Residency = r };
        var request = new Flux2Request { Prompt = "a knight in plate armor, full body, white background", Seed = 7 };
        var first = await pipe.RunAsync(request);
        Rp(trfLoads == 1 && r.Resident.Count == 1, $"the first run loads the transformer once and the manager keeps it (loads {trfLoads}, resident {r.Resident.Count})");
        long bytes = r.Resident[0].Bytes;
        var kept = await pipe.RunAsync(request);
        Rp(trfLoads == 1, $"a kept transformer is not loaded again (loads {trfLoads})");
        Rp_SameBytes(first.Rgba, kept.Rgba, "kept transformer, same seed");
        Rp(await r.ReleaseAllAsync() == 1 && evicted.Count == 1 && r.Resident.Count == 0, $"the manager drops it ({evicted.Count} drops, {r.Resident.Count} resident)");
        var reloaded = await pipe.RunAsync(request);
        Rp(trfLoads == 2, $"a dropped transformer is loaded again (loads {trfLoads})");
        Rp_SameBytes(first.Rgba, reloaded.Rgba, "reloaded transformer, same seed");
        Console.WriteLine($"[Residency] Flux2 on {accelerator.AcceleratorType}: transformer {bytes >> 20} MB kept, dropped, reloaded; 3 pictures identical ({first.Width}x{first.Height}); {string.Join(" | ", log)}");
    });

    /// <summary>FLUX 3 (4-bit) over HTTP on a manager: kept = not loaded again and the same picture; dropped = loaded again and the same picture.</summary>
    [TestMethod(Timeout = 3600000, Category = "HeavyModel")]
    public async Task Flux3_Residency_KeptAndReloaded_SamePicture() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient() ?? throw new UnsupportedTestException("HttpClient not available");
        try { await http.GetByteArrayAsync(F3Server + "vae_norm.json"); }
        catch (Exception e) { throw new UnsupportedTestException($"the local file server is not running ({e.GetType().Name}): node tools/unirig-harness/serve-files.mjs E:/spawndev-flux3/hf_upload 8767"); }
        var evicted = new List<ModelEviction>(); var log = new List<string>();
        await using var r = await Rp_Residency(accelerator, evicted, log);
        using var pipe = await Flux3Pipeline.CreateAsync(accelerator, new F3HttpSource(http), "local");
        pipe.Residency = r;
        var request = new Flux3Request { Caption = "Full-color studio photograph of a small red ceramic teapot on a plain light grey background, soft light.", Seed = 3, Width = 512, Height = 512 };
        var first = await pipe.RunAsync(request);
        Rp(r.Resident.Count == 1, $"the manager keeps the transformer ({r.Resident.Count} resident)");
        long bytes = r.Resident[0].Bytes;
        var kept = await pipe.RunAsync(request);
        Rp(pipe.LastTimings.FirstOrDefault(t => t.Stage.StartsWith("transformer weights")).Seconds < 1, "a kept transformer is not loaded again: " + string.Join(", ", pipe.LastTimings.Select(t => $"{t.Stage} {t.Seconds:F1} s")));
        Rp_SameBytes(first.Rgba, kept.Rgba, "kept transformer, same seed");
        Rp(await r.ReleaseAllAsync() == 1 && evicted.Count == 1 && r.Resident.Count == 0, $"the manager drops it ({evicted.Count} drops, {r.Resident.Count} resident)");
        var reloaded = await pipe.RunAsync(request);
        double load = pipe.LastTimings.FirstOrDefault(t => t.Stage.StartsWith("transformer weights")).Seconds;
        Rp(r.Resident.Count == 1, "the next run loads it again and the manager keeps it");
        Rp_SameBytes(first.Rgba, reloaded.Rgba, "reloaded transformer, same seed");
        Console.WriteLine($"[Residency] Flux3 on {accelerator.AcceleratorType}: transformer {bytes >> 20} MB kept, dropped, reloaded in {load:F1} s; 3 pictures identical; {string.Join(" | ", log)}");
    });

    /// <summary>LTX-Video (8-bit DiT and T5, from disk) on a manager, a short clip: every stage's weights kept, then all dropped and loaded again; the frames are the same bit for bit.</summary>
    [TestMethod(Timeout = 3600000, Category = "HeavyModel")]
    public async Task Ltx_Residency_KeptAndReloaded_SameClip() => await RunTest(async accelerator =>
    {
        if (OperatingSystem.IsBrowser()) throw new UnsupportedTestException("reads the LTX files from disk");
        var paths = LtxPaths.InFolder(LtxDir, LtxPaths.DitFile("q8"), LtxPaths.TextEncoderFile("q8"));
        if (!File.Exists(paths.Dit) || !File.Exists(paths.Vae) || !File.Exists(paths.TextEncoder)) throw new UnsupportedTestException($"LTX files not found in {LtxDir} (set LTX_DIR)");
        var evicted = new List<ModelEviction>(); var log = new List<string>();
        await using var r = await Rp_Residency(accelerator, evicted, log);
        using var pipe = new LtxPipeline(accelerator, paths) { Residency = r };
        var request = new LtxRequest { Prompt = "A red fox walking through fresh snow at golden hour, soft light.", Width = 256, Height = 256, Frames = 9, Seed = 42 };
        static byte[] Bytes(WanVideo v) { var b = new byte[v.Data.Length * 4]; Buffer.BlockCopy(v.Data, 0, b, 0, b.Length); return b; }
        var first = Bytes(await pipe.GenerateAsync(request));
        int resident = r.Resident.Count;
        Rp(resident >= 2, $"the manager keeps the stages' weights ({resident} resident: {string.Join(", ", r.Resident.Select(m => m.Key))})");
        long bytes = r.Resident.Sum(m => m.Bytes);
        var kept = Bytes(await pipe.GenerateAsync(request));
        Rp(evicted.Count == 0 && r.Resident.Count == resident, $"nothing dropped or added while kept ({evicted.Count} drops, {r.Resident.Count} resident)");
        Rp_SameBytes(first, kept, "kept weights, same seed");
        Rp(await r.ReleaseAllAsync() == resident && r.Resident.Count == 0, $"the manager drops them all ({evicted.Count} drops)");
        var reloaded = Bytes(await pipe.GenerateAsync(request));
        Rp(r.Resident.Count == resident, $"the next clip loads them again ({r.Resident.Count} resident)");
        Rp_SameBytes(first, reloaded, "reloaded weights, same seed");
        Console.WriteLine($"[Residency] Ltx on {accelerator.AcceleratorType}: {resident} models, {bytes >> 20} MB kept, dropped, reloaded; 3 clips identical; {string.Join(" | ", log)}");
    });
}
