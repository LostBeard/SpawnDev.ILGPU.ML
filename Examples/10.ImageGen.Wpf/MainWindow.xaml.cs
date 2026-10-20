using System.Collections.ObjectModel;
using System.IO;
using System.Net.Http;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using ILGPU.Runtime.OpenCL;
using Microsoft.Win32;
using SpawnDev.ILGPU.ML.Flux2;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.Pipelines;

namespace SpawnDev.ILGPU.ML.Examples.ImageGenWpf;

/// <summary>
/// Example 10: the browser demo's Image Generation page as a desktop app on CUDA / OpenCL: FLUX 3 Image Klein (with the Qwen3 caption writer), FLUX.2 klein or SD-Turbo, up to 10 reference images to edit or
/// combine, 1 / 4 / 8 pictures, a seed, SD-Turbo's steps / guidance / negative prompt, and Save. All of the work is SpawnDev.ILGPU.ML's <see cref="ImageGenerator"/>; this file is only the window.
/// </summary>
public partial class MainWindow : Window
{
    public sealed record PictureItem(int Seed, int Width, int Height, byte[] Rgba, BitmapSource Image, string? WrittenCaption)
    {
        public string Caption => $"  seed {Seed}, {Width} x {Height}";
        public double DisplayWidth { get; init; } = 240;
    }

    public sealed record RefItem(Flux2ReferenceImage Reference, BitmapSource Image);

    private const int ReferenceMaxSide = 768;      // the browser demo's limit: a reference is shrunk to this before it is encoded

    private readonly ObservableCollection<PictureItem> _items = new();
    private readonly ObservableCollection<RefItem> _refs = new();
    private readonly HttpClient _http = new() { Timeout = Timeout.InfiniteTimeSpan };
    private Context? _context;
    private Accelerator? _accelerator;
    private ImageGenerator? _gen;
    private ImageEngine _genEngine;
    private int _genBits;
    private string _cacheDir = Environment.GetEnvironmentVariable("IMAGEGEN_CACHE") ?? FileModelStore.Default().RootDirectory;   // IMAGEGEN_CACHE: start with the models somewhere else (a big drive)
    private bool _ready;

    public MainWindow()
    {
        InitializeComponent();
        Pictures.ItemsSource = _items;
        Refs.ItemsSource = _refs;
        _ready = true;
        ModelChanged(this, null!);
        CacheText.Text = $"Models are cached in {_cacheDir}: use Model folder... to put them on a big drive. Nothing leaves your machine except the model downloads.";
    }

    // ───────────────────────────── helpers ─────────────────────────────

    private ImageEngine Engine => ModelBox.SelectedIndex switch { 0 => ImageEngine.Flux3, 1 => ImageEngine.Flux2, _ => ImageEngine.SdTurbo };

    private void Log(string line) => Dispatcher.InvokeAsync(() => { LogBox.AppendText(line + Environment.NewLine); LogBox.ScrollToEnd(); });

    private void Busy(bool busy)
    {
        foreach (var c in new Control[] { GenerateButton, ClearButton, CacheButton, AddRefButton, ModelBox, SizeBox, CountBox, SeedBox, BitsBox, CaptionBox, StepsBox, GuidanceBox, NegativeBox, PromptBox, Refs })
            c.IsEnabled = !busy;
        if (!busy) UpdateCaptionBox();
    }

    private void UpdateCaptionBox()
    {
        CaptionBox.Visibility = Engine == ImageEngine.Flux3 ? Visibility.Visible : Visibility.Collapsed;
        CaptionBox.IsEnabled = _refs.Count == 0;       // the caption writer is for text to image; with references the prompt is the edit instruction
        CaptionBox.ToolTip = _refs.Count == 0 ? CaptionBox.ToolTip : "Off while reference images are set: the prompt is then the instruction for the edit.";
        SizeLabel.Text = _refs.Count > 0 && Engine != ImageEngine.SdTurbo ? "Longer side" : "Size";
    }

    private void ModelChanged(object sender, SelectionChangedEventArgs e)
    {
        if (!_ready) return;
        var engine = Engine;
        bool sd = engine == ImageEngine.SdTurbo;
        FluxPanel.Visibility = sd ? Visibility.Collapsed : Visibility.Visible;
        SdPanel.Visibility = sd ? Visibility.Visible : Visibility.Collapsed;
        RefPanel.Visibility = sd ? Visibility.Collapsed : Visibility.Visible;
        if (sd) _refs.Clear();
        var sizes = sd ? new[] { "256", "512" } : new[] { "512", "768", "1024" };
        if (SizeBox.Items.Count != sizes.Length || SizeBox.Items.Cast<string>().Except(sizes).Any())
        {
            SizeBox.ItemsSource = sizes;
            SizeBox.SelectedIndex = sd ? 1 : 0;
        }
        LicenceText.Text = engine == ImageEngine.Flux3
            ? "FLUX 3 Image Klein: weights under the FLUX Kommunity License (non-commercial use). Modified 4-bit export LostBeard/flux-3-image-klein-q4-ILGPU; about 5.7 GB downloaded."
            : engine == ImageEngine.Flux2 ? "FLUX.2 klein 4B (Black Forest Labs, Apache-2.0): 4-bit export LostBeard/FLUX.2-klein-4B-q4-ILGPU, about 4.6 GB downloaded."
            : "SD-Turbo (Stability AI): about 2.5 GB downloaded.";
        UpdateCaptionBox();
    }

    private static BitmapSource ToBitmap(int w, int h, byte[] rgba)
    {
        var bgra = new byte[rgba.Length];
        for (int i = 0; i < bgra.Length; i += 4) { bgra[i] = rgba[i + 2]; bgra[i + 1] = rgba[i + 1]; bgra[i + 2] = rgba[i]; bgra[i + 3] = 255; }
        var bmp = BitmapSource.Create(w, h, 96, 96, PixelFormats.Bgra32, null, bgra, w * 4);
        bmp.Freeze();
        return bmp;
    }

    /// <summary>The generator for the chosen model, kept between pictures (a second picture skips the model load); a different model or weight size starts a new one.</summary>
    private async Task<ImageGenerator> EnsureGeneratorAsync()
    {
        int bits = BitsBox.SelectedIndex == 1 ? 8 : 4;
        var engine = Engine;
        if (_gen != null && _genEngine == engine && _genBits == bits) return _gen;
        _gen?.Dispose(); _gen = null;
        await Task.Run(() =>
        {
            if (_accelerator == null)
            {
                _context = MLContext.Create().ToContext();
                var cuda = _context.GetCudaDevices(); var cl = _context.GetCLDevices();
                Device device = cuda.Count > 0 ? (Device)cuda[0] : cl.Count > 0 ? (Device)cl[0] : _context.GetPreferredDevice(preferCPU: false);
                _accelerator = device.CreateAccelerator(_context);
                Log($"Accelerator: {_accelerator.Name} ({_accelerator.AcceleratorType})");
            }
        });
        var source = new HttpClientModelSource(_http, new FileModelStore(_cacheDir));
        _gen = new ImageGenerator(_accelerator!, source, (repo, file, progress) => source.OpenAsync(repo, file, progress)) { FluxBits = bits };
        _gen.Status = s => Dispatcher.InvokeAsync(() => StatusText.Text = s);
        _gen.LoadProgress = p => Dispatcher.InvokeAsync(() => Bar.Value = p);
        _gen.StepProgress = (step, total) => Dispatcher.InvokeAsync(() => Bar.Value = total > 0 ? step * 100.0 / total : 0);
        _gen.CaptionText = text => Dispatcher.InvokeAsync(() => CaptionText.Text = text);
        _genEngine = engine; _genBits = bits;
        return _gen;
    }

    // ───────────────────────────── reference images ─────────────────────────────

    private void AddRefClick(object sender, RoutedEventArgs e)
    {
        var dlg = new OpenFileDialog { Title = "Reference images", Multiselect = true, Filter = "Pictures (*.png;*.jpg;*.jpeg;*.bmp;*.webp)|*.png;*.jpg;*.jpeg;*.bmp;*.webp" };
        if (dlg.ShowDialog() != true) return;
        foreach (var path in dlg.FileNames)
        {
            if (_refs.Count >= 10) { Log("At most 10 reference images."); break; }
            try { _refs.Add(LoadReference(path)); }
            catch (Exception ex) { Log($"Could not read {Path.GetFileName(path)}: {ex.Message}"); }
        }
        UpdateCaptionBox();
    }

    private void RemoveRefClick(object sender, RoutedEventArgs e)
    {
        if (((Button)sender).Tag is RefItem item) _refs.Remove(item);
        UpdateCaptionBox();
    }

    /// <summary>A reference picture as RGBA, shrunk so its longer side is at most <see cref="ReferenceMaxSide"/> (the browser demo does the same).</summary>
    private static RefItem LoadReference(string path)
    {
        var decoder = BitmapDecoder.Create(new Uri(path), BitmapCreateOptions.PreservePixelFormat, BitmapCacheOption.OnLoad);
        BitmapSource src = decoder.Frames[0];
        int longer = Math.Max(src.PixelWidth, src.PixelHeight);
        if (longer > ReferenceMaxSide)
        {
            double scale = (double)ReferenceMaxSide / longer;
            src = new TransformedBitmap(src, new ScaleTransform(scale, scale));
        }
        var bgra = new FormatConvertedBitmap(src, PixelFormats.Bgra32, null, 0);
        int w = bgra.PixelWidth, h = bgra.PixelHeight;
        var px = new byte[w * h * 4];
        bgra.CopyPixels(px, w * 4, 0);
        for (int i = 0; i < px.Length; i += 4) (px[i], px[i + 2]) = (px[i + 2], px[i]);     // BGRA -> RGBA
        bgra.Freeze();
        return new RefItem(new Flux2ReferenceImage(w, h, px), ToBitmap(w, h, px));
    }

    // ───────────────────────────── actions ─────────────────────────────

    private void CacheClick(object sender, RoutedEventArgs e)
    {
        var dlg = new OpenFolderDialog { Title = "Where should downloaded models live?", InitialDirectory = _cacheDir };
        if (dlg.ShowDialog() == true)
        {
            _cacheDir = dlg.FolderName; _gen?.Dispose(); _gen = null;
            CacheText.Text = $"Models are cached in {_cacheDir}.";
        }
    }

    private void ClearClick(object sender, RoutedEventArgs e)
    {
        _items.Clear(); CaptionExpander.Visibility = Visibility.Collapsed; CaptionText.Text = ""; InfoText.Text = ""; StatusText.Text = ""; Bar.Value = 0;
    }

    private async void GenerateClick(object sender, RoutedEventArgs e)
    {
        string prompt = PromptBox.Text.Trim();
        if (prompt.Length == 0) return;
        var engine = Engine;
        int seed = int.TryParse(SeedBox.Text, out var s) ? s : Environment.TickCount & 0x7FFFFFFF;
        int count = int.Parse(((ComboBoxItem)CountBox.SelectedItem).Content.ToString()!);
        int size = int.Parse((string)SizeBox.SelectedItem);
        var references = _refs.Count > 0 ? _refs.Select(r => r.Reference).ToList() : null;
        bool writeCaption = engine == ImageEngine.Flux3 && references == null && CaptionBox.IsChecked == true;
        int steps = int.TryParse(StepsBox.Text, out var st) ? Math.Max(1, st) : 1;
        float guidance = float.TryParse(GuidanceBox.Text, System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out var g) ? g : 0f;

        Busy(true); _items.Clear(); InfoText.Text = ""; CaptionText.Text = ""; Bar.Value = 0;
        CaptionExpander.Visibility = writeCaption ? Visibility.Visible : Visibility.Collapsed; CaptionExpander.IsExpanded = true;
        string? usedCaption = null; string timings = "";
        var total = System.Diagnostics.Stopwatch.StartNew();
        try
        {
            var gen = await EnsureGeneratorAsync();
            Log($"Drawing {count} picture(s) with {engine}{(references != null ? $" from {references.Count} reference image(s)" : "")} from seed {seed}; the first run downloads the model.");
            var request = new ImageGenerationRequest
            {
                Prompt = prompt, Engine = engine, Seed = seed, Pictures = count, Size = size, References = references, WriteCaption = writeCaption,
                NegativePrompt = NegativeBox.Text.Trim(), SdSteps = steps, SdGuidance = guidance,
            };
            await Task.Run(() => gen.GenerateAsync(request, picture => Dispatcher.InvokeAsync(() =>
            {
                _items.Add(new PictureItem(picture.Seed, picture.Width, picture.Height, picture.Rgba, ToBitmap(picture.Width, picture.Height, picture.Rgba), picture.Caption) { DisplayWidth = count == 1 ? 512 : 240 });
                usedCaption = picture.Caption ?? usedCaption;
                InfoText.Text = $"{total.Elapsed.TotalMilliseconds:F0} ms   {picture.Width} x {picture.Height}   seed {picture.Seed}";
            }).Task));
            timings = gen.LastTimings.Count > 0 ? "   last picture: " + string.Join(" · ", gen.LastTimings.Select(t => $"{t.Stage} {t.Seconds:F1} s")) : "";
            InfoText.Text += timings;
            if (usedCaption != null) { CaptionExpander.Header = "Caption used"; CaptionText.Text = usedCaption; CaptionExpander.IsExpanded = false; }
            else CaptionExpander.Visibility = Visibility.Collapsed;
            StatusText.Text = "Done."; Bar.Value = 100;
        }
        catch (Exception ex) { Log("Failed: " + ex.Message); MessageBox.Show(ex.ToString(), "Generation failed"); StatusText.Text = "Failed."; }
        finally { Busy(false); }
    }

    private void SaveClick(object sender, RoutedEventArgs e)
    {
        if (((Button)sender).Tag is not PictureItem item) return;
        var dlg = new SaveFileDialog { Filter = "PNG (*.png)|*.png", FileName = $"generated-{item.Seed}.png" };
        if (dlg.ShowDialog() != true) return;
        var encoder = new PngBitmapEncoder();
        encoder.Frames.Add(BitmapFrame.Create(item.Image));
        using var fs = File.Create(dlg.FileName);
        encoder.Save(fs);
        Log($"Saved {dlg.FileName}");
    }

    protected override void OnClosed(EventArgs e)
    {
        _gen?.Dispose(); _accelerator?.Dispose(); _context?.Dispose(); _http.Dispose();
        base.OnClosed(e);
    }
}
