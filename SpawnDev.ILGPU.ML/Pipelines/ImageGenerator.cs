using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Flux2;
using SpawnDev.ILGPU.ML.Flux3;
using SpawnDev.ILGPU.ML.Hub;

namespace SpawnDev.ILGPU.ML.Pipelines;

/// <summary>The text-to-image models <see cref="ImageGenerator"/> draws with.</summary>
public enum ImageEngine
{
    /// <summary>FLUX 3 Image Klein 5.6B, 16 steps (Black Forest Labs, FLUX Kommunity License: non-commercial). Draws from a full scene caption; up to 10 reference images.</summary>
    Flux3,
    /// <summary>FLUX.2 klein 4B, 4 steps (Black Forest Labs, Apache-2.0). Up to 10 reference images.</summary>
    Flux2,
    /// <summary>SD-Turbo, one step (Stability AI): the fast, small one.</summary>
    SdTurbo,
}

/// <summary>What <see cref="ImageGenerator.GenerateAsync"/> draws: the same settings the browser demo's Image Generation page offers.</summary>
public sealed class ImageGenerationRequest
{
    public string Prompt { get; init; } = "";
    public ImageEngine Engine { get; init; } = ImageEngine.Flux2;
    /// <summary>The first picture's seed; the next pictures use the following seeds.</summary>
    public int Seed { get; init; } = 1;
    /// <summary>How many pictures to draw of the prompt (consecutive seeds).</summary>
    public int Pictures { get; init; } = 1;
    /// <summary>FLUX: the picture's side (a multiple of 16; the longer side when <see cref="References"/> set the proportions). SD-Turbo: 256 or 512.</summary>
    public int Size { get; init; } = 512;
    /// <summary>FLUX: up to 10 pictures to edit or combine (the prompt says what to do); the picture takes the first one's proportions.</summary>
    public IReadOnlyList<Flux2ReferenceImage>? References { get; init; }
    /// <summary>FLUX 3 without references: a local language model (Qwen3 4B) writes the full scene caption FLUX 3 is trained on from the prompt.</summary>
    public bool WriteCaption { get; init; }
    /// <summary>SD-Turbo only.</summary>
    public string NegativePrompt { get; init; } = "";
    /// <summary>SD-Turbo only: denoising steps (designed for 1).</summary>
    public int SdSteps { get; init; } = 1;
    /// <summary>SD-Turbo only: classifier-free guidance (designed for 0).</summary>
    public float SdGuidance { get; init; }
}

/// <summary>One picture drawn by <see cref="ImageGenerator"/>.</summary>
public sealed record GeneratedImage(int Seed, int Width, int Height, byte[] Rgba, string? Caption);

/// <summary>
/// Text to image (and picture editing) with FLUX 3 Image Klein, FLUX.2 klein or SD-Turbo behind one call: the pipeline of the browser demo's Image Generation page, for any host (a console, a desktop app, a server).
/// One instance keeps the chosen engine's pipeline loaded between calls, so a second picture skips the load. The caller owns the accelerator and the model source.
/// </summary>
public sealed class ImageGenerator : IDisposable
{
    public const string Flux3Repo = "LostBeard/flux-3-image-klein-q4-ILGPU", Flux2Repo = "LostBeard/FLUX.2-klein-4B-q4-ILGPU";

    private readonly Accelerator _acc;
    private readonly IModelSource _source;
    private readonly Flux3CaptionWriter.OpenModel _openModel;
    private Flux3Pipeline? _flux3;
    private Flux2Pipeline? _flux2;
    private ImageGenerationPipeline? _sd;

    /// <param name="openModel">how the caption language model's file is opened with progress (for <see cref="ImageGenerationRequest.WriteCaption"/>); null = through <paramref name="source"/> without progress</param>
    public ImageGenerator(Accelerator accelerator, IModelSource source, Flux3CaptionWriter.OpenModel? openModel = null)
    {
        _acc = accelerator; _source = source;
        _openModel = openModel ?? ((repo, file, _) => source.OpenAsync(repo, file));
    }

    /// <summary>The weights of FLUX.2 and FLUX 3: 4-bit (the browser's) or 8-bit (near lossless, about 4.4 GB / 10 GB: a desktop GPU's choice). Set before the first picture of an engine.</summary>
    public int FluxBits { get; set; } = 4;

    /// <summary>Progress text ("FLUX 3 Klein (2/4) — denoise 5/16") for a status line.</summary>
    public Action<string>? Status { get; set; }

    /// <summary>(step, steps) of the picture being drawn, for a progress bar; (0, steps) between pictures.</summary>
    public Action<int, int>? StepProgress { get; set; }

    /// <summary>The caption as the language model writes it (whole text so far).</summary>
    public Action<string>? CaptionText { get; set; }

    /// <summary>0 to 100 while a model downloads and loads.</summary>
    public Action<double>? LoadProgress { get; set; }

    /// <summary>Seconds each stage of the last FLUX 3 picture took.</summary>
    public IReadOnlyList<(string Stage, double Seconds)> LastTimings { get; private set; } = Array.Empty<(string, double)>();

    /// <summary>The picture size: a square of <paramref name="side"/> pixels, or with reference images the first one's proportions with that longer side (multiples of 16, as the browser demo computes it).</summary>
    public static (int Width, int Height) PictureSize(int side, IReadOnlyList<Flux2ReferenceImage>? references)
    {
        if (references == null || references.Count == 0) return (side, side);
        var first = references[0];
        double aspect = (double)first.Width / first.Height;
        int w = aspect >= 1 ? side : (int)Math.Round(side * aspect / 16) * 16, h = aspect >= 1 ? (int)Math.Round(side / aspect / 16) * 16 : side;
        return (Math.Max(64, w), Math.Max(64, h));
    }

    /// <summary>The SD-Turbo pipeline once it is loaded (for diagnostics); null before.</summary>
    public ImageGenerationPipeline? SdPipeline => _sd;

    /// <summary>Load an engine's pipeline now (<see cref="GenerateAsync"/> does it on first use): FLUX loads only the tokenizer and the small VAE decoder, the big stages stream in per picture; SD-Turbo loads all three models.</summary>
    public async Task LoadAsync(ImageEngine engine)
    {
        switch (engine)
        {
            case ImageEngine.Flux3 when _flux3 == null:
                // the text encoder and the transformer stream in per picture (the transformer then stays: the next picture skips the reload)
                _flux3 = await Flux3Pipeline.CreateAsync(_acc, _source, Flux3Repo, (stage, pct) => { Status?.Invoke($"FLUX 3 Klein — {stage}"); LoadProgress?.Invoke(pct); }, default, FluxBits);
                _flux3.KeepTransformerResident = true;
                break;
            case ImageEngine.Flux2 when _flux2 == null:
                _flux2 = await Flux2Pipeline.CreateAsync(_acc, _source, Flux2Repo, 512, (stage, pct) => { Status?.Invoke($"FLUX.2 klein — {stage}"); LoadProgress?.Invoke(pct); }, default, FluxBits);
                _flux2.KeepTransformerResident = true;
                break;
            case ImageEngine.SdTurbo when _sd == null:
                _sd = await ImageGenerationPipeline.CreateAsync(_acc, _source, onProgress: (stage, pct) => { Status?.Invoke($"SD Turbo — {stage}"); LoadProgress?.Invoke(pct); });
                if (!_sd.IsReady) { _sd.Dispose(); _sd = null; throw new InvalidOperationException("SD-Turbo's pipeline failed to become ready after load."); }
                break;
        }
    }

    /// <summary>Draw <see cref="ImageGenerationRequest.Pictures"/> pictures; <paramref name="onPicture"/> gets each as soon as it is drawn. Returns them all.</summary>
    public async Task<List<GeneratedImage>> GenerateAsync(ImageGenerationRequest request, Func<GeneratedImage, Task>? onPicture = null)
    {
        if (string.IsNullOrWhiteSpace(request.Prompt)) throw new ArgumentException("The prompt is empty.", nameof(request));
        bool refs = request.References is { Count: > 0 };
        if (refs && request.Engine == ImageEngine.SdTurbo) throw new ArgumentException("SD-Turbo does not take reference images; use FLUX.2 or FLUX 3.", nameof(request));
        int count = Math.Max(1, request.Pictures);
        var result = new List<GeneratedImage>();
        var (pw, ph) = request.Engine == ImageEngine.SdTurbo ? (request.Size, request.Size) : PictureSize(request.Size, request.References);

        switch (request.Engine)
        {
            case ImageEngine.Flux3:
            {
                string caption = request.Prompt;
                bool expand = request.WriteCaption && !refs;
                await LoadAsync(ImageEngine.Flux3);
                if (expand)
                {
                    _flux3.ReleaseResident();                       // the language model and the transformer never share the GPU
                    caption = await Flux3CaptionWriter.WriteAsync(_acc, _openModel, request.Prompt, pw, ph, s => Status?.Invoke(s), LoadProgress, text => CaptionText?.Invoke(text));
                }
                for (int n = 0; n < count; n++)
                {
                    int seed = request.Seed + n; string label = count > 1 ? $" ({n + 1}/{count})" : "";
                    StepProgress?.Invoke(0, 16);
                    var image = await _flux3.RunAsync(new Flux3Request
                    {
                        Caption = caption, Seed = seed, Width = pw, Height = ph, References = refs ? request.References : null,
                        Progress = (stage, i, c) => { Status?.Invoke($"FLUX 3 Klein{label} — {stage}"); StepProgress?.Invoke(stage.StartsWith("denoise") ? i + 1 : 0, 16); },
                    });
                    LastTimings = _flux3.LastTimings;
                    await Emit(new GeneratedImage(seed, image.Width, image.Height, image.Rgba, expand ? caption : null));
                }
                break;
            }
            case ImageEngine.Flux2:
            {
                await LoadAsync(ImageEngine.Flux2);
                for (int n = 0; n < count; n++)
                {
                    int seed = request.Seed + n; string label = count > 1 ? $" ({n + 1}/{count})" : "";
                    StepProgress?.Invoke(0, 4);
                    var image = await _flux2.RunAsync(new Flux2Request
                    {
                        Prompt = request.Prompt, Seed = seed, Size = request.Size, Width = refs ? pw : 0, Height = refs ? ph : 0, References = refs ? request.References : null, Steps = 4,
                        Progress = (stage, i, c) => { Status?.Invoke($"FLUX.2 klein{label} — {stage}"); StepProgress?.Invoke(stage.StartsWith("denoise") ? i + 1 : 0, 4); },
                    });
                    await Emit(new GeneratedImage(seed, image.Width, image.Height, image.Rgba, null));
                }
                break;
            }
            default:
            {
                await LoadAsync(ImageEngine.SdTurbo);
                var sd = _sd!;
                int steps = Math.Max(1, request.SdSteps);
                sd.NumInferenceSteps = steps; sd.GuidanceScale = request.SdGuidance; sd.Width = request.Size; sd.Height = request.Size;
                Action<int, int> onStep = (step, total) => StepProgress?.Invoke(step, total);
                sd.OnProgress += onStep;
                try
                {
                    for (int n = 0; n < count; n++)
                    {
                        sd.Seed = request.Seed + n;
                        StepProgress?.Invoke(0, steps);
                        var r = await sd.RunAsync(new ImageGenerationInput { Prompt = request.Prompt, NegativePrompt = request.NegativePrompt });
                        await Emit(new GeneratedImage(r.Seed, r.Width, r.Height, r.ImageRGBA, null));
                    }
                }
                finally { sd.OnProgress -= onStep; }
                break;
            }
        }
        return result;

        async Task Emit(GeneratedImage picture)
        {
            result.Add(picture);
            if (onPicture != null) await onPicture(picture);
        }
    }

    /// <summary>Free the loaded engines (the accelerator stays: the caller owns it).</summary>
    public void ReleaseModels()
    {
        _flux3?.Dispose(); _flux2?.Dispose(); _sd?.Dispose();
        _flux3 = null; _flux2 = null; _sd = null;
    }

    public void Dispose() => ReleaseModels();
}
