using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.SafeTensors;

namespace SpawnDev.ILGPU.ML.StableDiffusion;

/// <summary>Streams one standalone component file (ComfyUI's split loaders) into an <see cref="SdWeights"/> store.</summary>
internal static class SdComponentFile
{
    /// <summary>
    /// Parse the header of <paramref name="stream"/>, let <paramref name="makeRoute"/> look at every tensor name (and shape) and build the route, then stream the claimed
    /// tensors into <paramref name="store"/> (a new store when null). Returns the store.
    /// </summary>
    public static async Task<SdWeights> LoadAsync(Accelerator acc, Stream stream, string name, Func<IReadOnlyList<SafeTensorInfo>, Func<string, (string Local, SdLayout Layout)?>> makeRoute,
        SdWeights? store, Action<string, double>? onProgress, CancellationToken ct)
    {
        var lenBytes = new byte[8];
        await stream.ReadExactlyAsync(lenBytes, ct);
        long headerSize = BitConverter.ToInt64(lenBytes);
        if (headerSize <= 0 || headerSize > 100_000_000) throw new InvalidDataException($"'{name}': not a safetensors file (header size {headerSize})");
        var header = new byte[headerSize];
        await stream.ReadExactlyAsync(header, ct);
        long pos = 8 + headerSize, length = -1;
        try { if (stream.CanSeek) length = stream.Length - pos; } catch (NotSupportedException) { }
        var (tensors, _) = SafeTensorsFileReader.ParseHeader(name, header, pos, length);
        var route = makeRoute(tensors);
        long total = tensors.Count == 0 ? 1 : tensors.Max(t => t.DataOffset + t.DataLength);
        var w = store ?? new SdWeights();
        try
        {
            using var st = new SdTensorStreamer(acc);
            await st.StreamAsync(stream, tensors, pos, n => route(n) is { } r ? new SdTensorStreamer.Target(w, r.Local, r.Layout) : null, p => onProgress?.Invoke(name, (double)p / total), ct);
            return w;
        }
        catch
        {
            if (store == null) w.Dispose();
            throw;
        }
    }

    public static string StripPrefix(string n, params string[] prefixes)
    {
        foreach (var p in prefixes) if (n.StartsWith(p, StringComparison.Ordinal)) return n[p.Length..];
        return n;
    }
}

public sealed partial class SdUNet
{
    /// <summary>
    /// Load a UNet-only file (ComfyUI's UNETLoader, models/diffusion_models): LDM names with or without <c>model.diffusion_model.</c>, or diffusers names (a diffusers
    /// <c>unet/diffusion_pytorch_model.safetensors</c>, with or without <c>unet.</c>). The returned UNet owns its weights.
    /// </summary>
    public static async Task<SdUNet> LoadAsync(Accelerator accelerator, IModelSource source, string repo, string file, Action<string, double>? onProgress = null, CancellationToken cancellationToken = default)
    {
        await using var stream = await source.OpenAsync(repo, file, cancellationToken);
        return await LoadAsync(accelerator, stream, $"{repo}/{file}", onProgress, cancellationToken);
    }

    /// <summary>Load a UNet-only file from a stream (read once, front to back).</summary>
    public static async Task<SdUNet> LoadAsync(Accelerator accelerator, Stream stream, string name, Action<string, double>? onProgress = null, CancellationToken cancellationToken = default)
    {
        var w = await SdComponentFile.LoadAsync(accelerator, stream, name, tensors =>
        {
            var names = tensors.Select(t => SdComponentFile.StripPrefix(t.Name, "model.diffusion_model.", "diffusion_model.", "unet.")).ToHashSet();
            bool diffusers = names.Any(n => n.StartsWith("down_blocks.", StringComparison.Ordinal));
            return n =>
            {
                if (!n.StartsWith("model.diffusion_model.", StringComparison.Ordinal) && !n.StartsWith("diffusion_model.", StringComparison.Ordinal) && !n.StartsWith("unet.", StringComparison.Ordinal)
                    && (n.StartsWith("first_stage_model.", StringComparison.Ordinal) || n.StartsWith("cond_stage_model.", StringComparison.Ordinal) || n.StartsWith("conditioner.", StringComparison.Ordinal)))
                    return null;                                          // a whole checkpoint passed to the UNet loader: only its UNet
                var s = SdComponentFile.StripPrefix(n, "model.diffusion_model.", "diffusion_model.", "unet.");
                var local = diffusers ? SdNames.DiffusersToLdmUNet(s, i => names.Any(x => x.StartsWith($"up_blocks.{i}.attentions.", StringComparison.Ordinal))) : s;
                return local == null ? null : (local, SdLayout.ConvGather);
            };
        }, null, onProgress, cancellationToken);
        try
        {
            if (!w.Has("input_blocks.0.0.weight")) throw new InvalidDataException($"'{name}': no UNet tensors (conv_in / input_blocks.0.0) in this file");
            await SdNames.FuseUNetAsync(accelerator, w);
            return new SdUNet(accelerator, w) { _ownsWeights = true };
        }
        catch { w.Dispose(); throw; }
    }
}

public sealed partial class SdVae
{
    /// <summary>
    /// Load a standalone VAE file (ComfyUI's VAELoader, models/vae: <c>sdxl_vae.safetensors</c>, <c>vae-ft-mse-840000-ema-pruned.safetensors</c>, a diffusers
    /// <c>vae/diffusion_pytorch_model.safetensors</c>): LDM names with or without <c>first_stage_model.</c>, or diffusers names. The returned VAE owns its weights.
    /// The latent scale is the model family's (0.18215 SD 1.x / 2.x, 0.13025 SDXL): the caller knows which one the VAE belongs to.
    /// </summary>
    public static async Task<SdVae> LoadAsync(Accelerator accelerator, IModelSource source, string repo, string file, Action<string, double>? onProgress = null, CancellationToken cancellationToken = default)
    {
        await using var stream = await source.OpenAsync(repo, file, cancellationToken);
        return await LoadAsync(accelerator, stream, $"{repo}/{file}", onProgress, cancellationToken);
    }

    /// <summary>Load a standalone VAE file from a stream (read once, front to back).</summary>
    public static async Task<SdVae> LoadAsync(Accelerator accelerator, Stream stream, string name, Action<string, double>? onProgress = null, CancellationToken cancellationToken = default)
    {
        var w = await SdComponentFile.LoadAsync(accelerator, stream, name, tensors =>
        {
            var names = tensors.Select(t => SdComponentFile.StripPrefix(t.Name, "first_stage_model.", "vae.")).ToList();
            int levels = 0;
            while (names.Any(n => n.StartsWith($"encoder.down_blocks.{levels}.", StringComparison.Ordinal) || n.StartsWith($"decoder.up_blocks.{levels}.", StringComparison.Ordinal)
                || n.StartsWith($"encoder.down.{levels}.", StringComparison.Ordinal) || n.StartsWith($"decoder.up.{levels}.", StringComparison.Ordinal))) levels++;
            return n =>
            {
                if (n.StartsWith("model.diffusion_model.", StringComparison.Ordinal) || n.StartsWith("cond_stage_model.", StringComparison.Ordinal) || n.StartsWith("conditioner.", StringComparison.Ordinal)) return null;
                var s = SdComponentFile.StripPrefix(n, "first_stage_model.", "vae.");
                if (!(s.StartsWith("encoder.", StringComparison.Ordinal) || s.StartsWith("decoder.", StringComparison.Ordinal) || s.StartsWith("quant_conv.", StringComparison.Ordinal) || s.StartsWith("post_quant_conv.", StringComparison.Ordinal)))
                    return null;
                var local = SdNames.DiffusersToLdmVae(s, levels);
                return local == null ? null : (local, SdLayout.ConvGather);
            };
        }, null, onProgress, cancellationToken);
        try
        {
            if (!w.Has("decoder.conv_in.weight") && !w.Has("encoder.conv_in.weight")) throw new InvalidDataException($"'{name}': no VAE tensors (encoder / decoder conv_in) in this file");
            await SdNames.FuseVaeAsync(accelerator, w);
            return new SdVae(accelerator, w) { _ownsWeights = true };
        }
        catch { w.Dispose(); throw; }
    }
}

public sealed partial class SdTextEncoder
{
    /// <summary>
    /// Load one standalone text encoder (ComfyUI's CLIPLoader, type stable_diffusion: <c>clip_l.safetensors</c> = CLIP ViT-L for SD 1.x, an OpenCLIP ViT-H for SD 2.x), in
    /// Hugging Face (<c>text_model.*</c>) or OpenCLIP names. The family comes from the width (768 = SD 1.x, 1024 = SD 2.x). The returned encoder owns its weights.
    /// </summary>
    public static async Task<SdTextEncoder> LoadAsync(Accelerator accelerator, IModelSource source, string repo, string file, Action<string, double>? onProgress = null, CancellationToken cancellationToken = default)
    {
        await using var stream = await source.OpenAsync(repo, file, cancellationToken);
        return await LoadAsync(accelerator, stream, $"{repo}/{file}", onProgress, cancellationToken);
    }

    /// <summary>Load one standalone text encoder from a stream.</summary>
    public static async Task<SdTextEncoder> LoadAsync(Accelerator accelerator, Stream stream, string name, Action<string, double>? onProgress = null, CancellationToken cancellationToken = default)
    {
        SdFamily family = SdFamily.SD1;
        var w = await SdComponentFile.LoadAsync(accelerator, stream, name, tensors =>
        {
            int width = TokenWidth(tensors, name);
            (family, var prefix) = width switch
            {
                768 => (SdFamily.SD1, "l."),
                1024 => (SdFamily.SD2, "h."),
                1280 => throw new NotSupportedException($"'{name}': an OpenCLIP bigG tower alone is SDXL's second encoder: load it with LoadSdxlAsync together with CLIP-L"),
                _ => throw new NotSupportedException($"'{name}': text width {width} is not CLIP-L (768) or OpenCLIP-H (1024)"),
            };
            return n => SdNames.MapClipFile(n, prefix, out var layout) is { } local ? (local, layout) : null;
        }, null, onProgress, cancellationToken);
        try
        {
            await SdNames.FuseTextAsync(accelerator, w);
            return new SdTextEncoder(accelerator, w, family) { _ownsWeights = true };
        }
        catch { w.Dispose(); throw; }
    }

    /// <summary>
    /// Load SDXL's two text encoders from separate files (ComfyUI's DualCLIPLoader, type sdxl: <c>clip_l.safetensors</c> + <c>clip_g.safetensors</c>, or a diffusers
    /// <c>text_encoder</c> + <c>text_encoder_2</c>), in Hugging Face or OpenCLIP names. The returned encoder owns its weights.
    /// </summary>
    public static async Task<SdTextEncoder> LoadSdxlAsync(Accelerator accelerator, IModelSource source, string repo, string clipLFile, string clipGFile,
        Action<string, double>? onProgress = null, CancellationToken cancellationToken = default)
    {
        await using var l = await source.OpenAsync(repo, clipLFile, cancellationToken);
        await using var g = await source.OpenAsync(repo, clipGFile, cancellationToken);
        return await LoadSdxlAsync(accelerator, l, $"{repo}/{clipLFile}", g, $"{repo}/{clipGFile}", onProgress, cancellationToken);
    }

    /// <summary>Load SDXL's two text encoders from two streams.</summary>
    public static async Task<SdTextEncoder> LoadSdxlAsync(Accelerator accelerator, Stream clipL, string nameL, Stream clipG, string nameG,
        Action<string, double>? onProgress = null, CancellationToken cancellationToken = default)
    {
        var w = new SdWeights();
        try
        {
            foreach (var (stream, name, want, prefix) in new[] { (clipL, nameL, 768, "l."), (clipG, nameG, 1280, "g.") })
                await SdComponentFile.LoadAsync(accelerator, stream, name, tensors =>
                {
                    int width = TokenWidth(tensors, name);
                    if (width != want) throw new InvalidDataException($"'{name}': text width {width}, SDXL's {(prefix == "l." ? "first (CLIP-L)" : "second (OpenCLIP bigG)")} encoder is {want}");
                    return n => SdNames.MapClipFile(n, prefix, out var layout) is { } local ? (local, layout) : null;
                }, w, onProgress, cancellationToken);
            if (!w.Has("g.proj.weight")) throw new InvalidDataException($"'{nameG}': no text projection (the pooled embedding needs it)");
            await SdNames.FuseTextAsync(accelerator, w);
            return new SdTextEncoder(accelerator, w, SdFamily.SDXL) { _ownsWeights = true };
        }
        catch { w.Dispose(); throw; }
    }

    private static int TokenWidth(IReadOnlyList<SafeTensorInfo> tensors, string name)
    {
        var tok = tensors.FirstOrDefault(t => t.Name.EndsWith("token_embedding.weight", StringComparison.Ordinal))
            ?? throw new InvalidDataException($"'{name}': no token embedding: not a CLIP text encoder");
        return tok.Shape[1];
    }
}
