using System.Diagnostics;
using System.Text.Json;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Hub;

namespace SpawnDev.ILGPU.ML.Trellis2;

/// <summary>
/// Where the TRELLIS.2 checkpoints come from: local files (<see cref="FromFiles"/>, the desktop path with page-cache prefetch) or any
/// <see cref="IModelSource"/> (<see cref="FromModelSource"/>: the Hugging Face hub streamed into OPFS in the browser, an HTTP + disk cache on
/// desktop). The pipeline asks for one stage's checkpoint at a time and frees it before the next.
/// </summary>
public abstract class Trellis2Weights
{
    /// <summary>pipeline.json of TRELLIS.2-4B (sampler settings, latent normalisation).</summary>
    public abstract Task<string> ReadPipelineJsonAsync(CancellationToken ct = default);
    /// <summary>A TRELLIS.2-4B checkpoint by name (ckpts/&lt;name&gt;.json + .safetensors).</summary>
    public abstract Task<GpuWeights> LoadCheckpointAsync(Accelerator accelerator, string name, CancellationToken ct = default);
    /// <summary>The sparse-structure decoder (TRELLIS-image-large ckpts/ss_dec_conv3d_16l8_fp16).</summary>
    public abstract Task<GpuWeights> LoadSsDecoderAsync(Accelerator accelerator, CancellationToken ct = default);
    /// <summary>DINOv3 ViT-L/16 (a bare .safetensors in the Hugging Face or timm layout).</summary>
    public abstract Task<GpuWeights> LoadDinoAsync(Accelerator accelerator, CancellationToken ct = default);
    /// <summary>Hint: these checkpoints are needed soon (read ahead so the stage does not wait on the disk / network). Optional.</summary>
    public virtual void Prefetch(IEnumerable<string> checkpointNames, bool includeDino = false, bool includeSsDecoder = false) { }

    public static Trellis2Weights FromFiles(Trellis2Paths paths) => new FileWeights(paths);

    /// <param name="source">Hub / HTTP / OPFS source (for example the demo's HubModelSource, or HttpClientModelSource + FileModelStore on desktop).</param>
    public static Trellis2Weights FromModelSource(IModelSource source, string trellisRepo = "microsoft/TRELLIS.2-4B",
        string ssDecoderRepo = "microsoft/TRELLIS-image-large", string dinoRepo = "timm/vit_large_patch16_dinov3_qkvb.lvd1689m")
        => new SourceWeights(source, trellisRepo, ssDecoderRepo, dinoRepo);

    /// <summary>
    /// The weight files a <see cref="FromModelSource"/> run reads at <paramref name="resolution"/> (512 or 1024), as (repo, path): for storage
    /// planning (<see cref="RangeStreamModelSource.PlanCacheAsync"/>). The small .json files beside them are not listed. A <paramref name="cascade"/> run (<c>1024_cascade</c> / <c>1536_cascade</c>) reads the 512 shape model and the 1024 shape and texture models.
    /// </summary>
    public static IReadOnlyList<(string Repo, string Path)> HubWeightFiles(int resolution = 512, string trellisRepo = "microsoft/TRELLIS.2-4B",
        string ssDecoderRepo = "microsoft/TRELLIS-image-large", string dinoRepo = "timm/vit_large_patch16_dinov3_qkvb.lvd1689m", bool cascade = false)
    {
        if (cascade)
            return new[]
            {
                (dinoRepo, "model.safetensors"),
                (trellisRepo, "ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors"),
                (ssDecoderRepo, "ckpts/ss_dec_conv3d_16l8_fp16.safetensors"),
                (trellisRepo, "ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors"),
                (trellisRepo, "ckpts/shape_dec_next_dc_f16c32_fp16.safetensors"),
                (trellisRepo, "ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16.safetensors"),
                (trellisRepo, "ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.safetensors"),
                (trellisRepo, "ckpts/tex_dec_next_dc_f16c32_fp16.safetensors"),
            };
        string s = resolution >= 1024 ? "1024" : "512";
        return new[]
        {
            (dinoRepo, "model.safetensors"),
            (trellisRepo, "ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors"),
            (ssDecoderRepo, "ckpts/ss_dec_conv3d_16l8_fp16.safetensors"),
            (trellisRepo, $"ckpts/slat_flow_img2shape_dit_1_3B_{s}_bf16.safetensors"),
            (trellisRepo, "ckpts/shape_dec_next_dc_f16c32_fp16.safetensors"),
            (trellisRepo, $"ckpts/slat_flow_imgshape2tex_dit_1_3B_{s}_bf16.safetensors"),
            (trellisRepo, "ckpts/tex_dec_next_dc_f16c32_fp16.safetensors"),
        };
    }

    internal static JsonElement Args(string json)
    {
        using var doc = JsonDocument.Parse(json);
        return doc.RootElement.TryGetProperty("args", out var a) ? a.Clone() : doc.RootElement.Clone();
    }

    private sealed class FileWeights : Trellis2Weights
    {
        private readonly Trellis2Paths _p;
        private Task _prefetch = Task.CompletedTask;
        public FileWeights(Trellis2Paths p) => _p = p;
        private string Ckpt(string name) => Path.Combine(_p.ModelDir, "ckpts", name);

        public override Task<string> ReadPipelineJsonAsync(CancellationToken ct = default) => File.ReadAllTextAsync(Path.Combine(_p.ModelDir, "pipeline.json"), ct);
        public override Task<GpuWeights> LoadCheckpointAsync(Accelerator acc, string name, CancellationToken ct = default) => Task.FromResult(GpuWeights.Load(acc, Ckpt(name)));
        public override Task<GpuWeights> LoadSsDecoderAsync(Accelerator acc, CancellationToken ct = default) => Task.FromResult(GpuWeights.Load(acc, _p.SparseStructureDecoder));
        public override Task<GpuWeights> LoadDinoAsync(Accelerator acc, CancellationToken ct = default) => Task.FromResult(GpuWeights.LoadFile(acc, _p.DinoWeights));

        // Page-cache read-ahead (measured: a cold read of a 2.4 GB flow checkpoint ~2.5 s, a cached one ~0.3 s). RandomAccess, not
        // SequentialScan: Windows drops SequentialScan pages right behind the read pointer. Queued behind earlier requests.
        public override void Prefetch(IEnumerable<string> checkpointNames, bool includeDino = false, bool includeSsDecoder = false)
        {
            var files = new List<string>();
            if (includeDino) files.Add(_p.DinoWeights);
            files.AddRange(checkpointNames.Select(n => Ckpt(n) + ".safetensors"));
            if (includeSsDecoder) files.Add(_p.SparseStructureDecoder + ".safetensors");
            var previous = _prefetch;
            _prefetch = Task.Run(() =>
            {
                try { previous.Wait(); } catch { }
                var buf = new byte[8 << 20]; var psw = Stopwatch.StartNew();
                foreach (var f in files)
                {
                    if (Environment.GetEnvironmentVariable("TRELLIS2_PREFETCH_LOG") == "1") Console.WriteLine($"  [prefetch] {Path.GetFileName(f)} at {psw.Elapsed.TotalSeconds:F1}s");
                    try
                    {
                        using var fs = new FileStream(f, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 16, FileOptions.RandomAccess);
                        while (fs.Read(buf, 0, buf.Length) > 0) { }
                    }
                    catch (IOException) { }                      // a missing file surfaces with its real error when the stage opens it
                    catch (UnauthorizedAccessException) { }
                }
            });
        }
    }

    private sealed class SourceWeights : Trellis2Weights
    {
        private readonly IModelSource _src;
        private readonly string _trellis, _ssDec, _dino;
        public SourceWeights(IModelSource src, string trellis, string ssDec, string dino) { _src = src; _trellis = trellis; _ssDec = ssDec; _dino = dino; }

        private async Task<string> TextAsync(string repo, string path, CancellationToken ct)
        {
            await using var s = await _src.OpenAsync(repo, path, ct);
            using var r = new StreamReader(s);
            return await r.ReadToEndAsync(ct);
        }
        private async Task<GpuWeights> LoadAsync(Accelerator acc, string repo, string basePath, bool hasJson, CancellationToken ct)
        {
            var args = hasJson ? Args(await TextAsync(repo, basePath + ".json", ct)) : Args("{}");
            await using var s = await _src.OpenAsync(repo, basePath + ".safetensors", ct);
            return await GpuWeights.LoadAsync(acc, s, args, $"{repo}/{basePath}.safetensors", cancellationToken: ct);
        }

        public override Task<string> ReadPipelineJsonAsync(CancellationToken ct = default) => TextAsync(_trellis, "pipeline.json", ct);
        public override Task<GpuWeights> LoadCheckpointAsync(Accelerator acc, string name, CancellationToken ct = default) => LoadAsync(acc, _trellis, "ckpts/" + name, true, ct);
        public override Task<GpuWeights> LoadSsDecoderAsync(Accelerator acc, CancellationToken ct = default) => LoadAsync(acc, _ssDec, "ckpts/ss_dec_conv3d_16l8_fp16", true, ct);
        public override Task<GpuWeights> LoadDinoAsync(Accelerator acc, CancellationToken ct = default) => LoadAsync(acc, _dino, "model", false, ct);
    }
}
