using System.IO;
using System.Net.Http;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Threading;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using ILGPU.Runtime.OpenCL;
using Microsoft.Win32;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.Ltx;
using SpawnDev.ILGPU.ML.Wan;

namespace SpawnDev.ILGPU.ML.Examples.VideoGenWpf;

/// <summary>
/// Example 12: the browser demo's Video Generation page as a desktop app on CUDA / OpenCL: Wan 2.2 TI2V-5B text to video, or image to video with a first frame you choose. All of the work is
/// SpawnDev.ILGPU.ML's <see cref="Wan22Pipeline"/>; this file is only the window: the settings, a progress bar, a looping player and Save.
/// </summary>
public partial class MainWindow : Window
{
    private readonly HttpClient _http = new() { Timeout = Timeout.InfiniteTimeSpan };
    private Context? _context;
    private Accelerator? _accelerator;
    private string? _modelFolder = Environment.GetEnvironmentVariable("WAN_DIR");        // a local folder of the exports; null = download from the models host into the cache
    private bool _folderPicked;                                                          // a folder chosen in the window wins over WAN_DIR / LTX_DIR when the model changes
    private bool Ltx => ModelBox.SelectedIndex == 1;
    private string _cacheDir = FileModelStore.Default().RootDirectory;
    private byte[]? _pictureRgb;
    private int _pictureW, _pictureH;
    private WanVideo? _video;
    private List<BitmapSource> _frames = new();
    private readonly DispatcherTimer _timer = new() { Interval = TimeSpan.FromMilliseconds(1000.0 / 24) };
    private int _frameIndex;

    public MainWindow()
    {
        InitializeComponent();
        _timer.Tick += (_, _) => { if (_frames.Count == 0) return; _frameIndex = (_frameIndex + 1) % _frames.Count; Player.Source = _frames[_frameIndex]; };
        ShowFolder();
    }

    private void ShowFolder() => FolderText.Text = _modelFolder != null
        ? $"Models are read from {_modelFolder} (the exports of tools/{(Ltx ? "ltx" : "wan")}-harness/reference)."
        : $"Models download from the SpawnDev models host into {_cacheDir} on first use (Model folder... uses a local folder of the exports instead).";

    private void Log(string line) => Dispatcher.InvokeAsync(() => { LogBox.AppendText(line + Environment.NewLine); LogBox.ScrollToEnd(); });

    private void Busy(bool busy)
    {
        foreach (var c in new Control[] { GenerateButton, FolderButton, ModelBox, SizeBox, FramesBox, SeedBox, DitBox, TextEncBox, PromptBox }) c.IsEnabled = !busy;
        foreach (var c in new Control[] { PickButton, ClearPictureButton, StepsBox, GuidanceBox, NegativeBox }) c.IsEnabled = !busy && !Ltx;      // Wan only: LTX is text to video with a fixed 7-step recipe and no guidance
    }

    private static void Fill(ComboBox box, int selected, params string[] items)
    {
        box.Items.Clear();
        foreach (var i in items) box.Items.Add(new ComboBoxItem { Content = i });
        box.SelectedIndex = selected;
    }

    private void ModelChanged(object sender, SelectionChangedEventArgs e)
    {
        if (!IsLoaded) return;
        if (Ltx)
        {
            Fill(SizeBox, 0, "480x320", "640x384", "704x480");
            Fill(FramesBox, 3, "9", "17", "33", "49", "65", "97");                          // 8n + 1
            Fill(DitBox, 0, "8-bit (2.0 GB)", "16-bit (3.6 GB)");
            Fill(TextEncBox, 0, "8-bit (5.0 GB)", "16-bit (8.9 GB)");
            PromptBox.Text = "A red fox running through fresh snow at sunrise, cinematic, shallow depth of field.";
            if (!_folderPicked) _modelFolder = Environment.GetEnvironmentVariable("LTX_DIR");
            _pictureRgb = null; PicturePreview.Source = null;
        }
        else
        {
            Fill(SizeBox, 1, "512x288", "640x352", "832x480", "1280x704");
            Fill(FramesBox, 1, "9", "17", "25", "33", "49", "81", "121");
            Fill(DitBox, 0, "Turbo 8-bit (5.6 GB)", "Standard 8-bit (5.6 GB)", "Standard 4-bit (3.1 GB)", "Standard 16-bit (10 GB)");
            Fill(TextEncBox, 0, "8-bit (6.4 GB)", "16-bit (11.4 GB)");
            PromptBox.Text = "A red fox walking through fresh snow at golden hour, the camera follows it slowly, soft light, shallow depth of field.";
            if (!_folderPicked) _modelFolder = Environment.GetEnvironmentVariable("WAN_DIR");
        }
        Busy(false);
        ShowFolder();
    }

    private static string Content(ComboBox box) => ((ComboBoxItem)box.SelectedItem).Content.ToString()!;

    private static BitmapSource ToBitmap(int w, int h, byte[] rgba)
    {
        var bgra = new byte[rgba.Length];
        for (int i = 0; i < bgra.Length; i += 4) { bgra[i] = rgba[i + 2]; bgra[i + 1] = rgba[i + 1]; bgra[i + 2] = rgba[i]; bgra[i + 3] = 255; }
        var bmp = BitmapSource.Create(w, h, 96, 96, PixelFormats.Bgra32, null, bgra, w * 4);
        bmp.Freeze();
        return bmp;
    }

    private void PickClick(object sender, RoutedEventArgs e)
    {
        var dlg = new OpenFileDialog { Title = "First frame", Filter = "Pictures (*.png;*.jpg;*.jpeg;*.bmp;*.webp)|*.png;*.jpg;*.jpeg;*.bmp;*.webp" };
        if (dlg.ShowDialog() != true) return;
        try
        {
            var bmp = new BitmapImage();
            bmp.BeginInit(); bmp.UriSource = new Uri(dlg.FileName); bmp.CacheOption = BitmapCacheOption.OnLoad; bmp.EndInit(); bmp.Freeze();
            var bgr = new FormatConvertedBitmap(bmp, PixelFormats.Bgr24, null, 0);
            int w = bgr.PixelWidth, h = bgr.PixelHeight, stride = w * 3;
            var raw = new byte[stride * h];
            bgr.CopyPixels(raw, stride, 0);
            var rgb = new byte[raw.Length];
            for (int i = 0; i < raw.Length; i += 3) { rgb[i] = raw[i + 2]; rgb[i + 1] = raw[i + 1]; rgb[i + 2] = raw[i]; }
            _pictureRgb = rgb; _pictureW = w; _pictureH = h; PicturePreview.Source = bmp;
            Log($"First frame: {Path.GetFileName(dlg.FileName)} ({w} x {h}); it is resized and centre-cropped to the video size.");
        }
        catch (Exception ex) { Log($"Could not read {Path.GetFileName(dlg.FileName)}: {ex.Message}"); }
    }

    private void ClearPictureClick(object sender, RoutedEventArgs e) { _pictureRgb = null; PicturePreview.Source = null; }

    private void FolderClick(object sender, RoutedEventArgs e)
    {
        var dlg = new OpenFolderDialog { Title = "Folder with the Wan 2.2 exports (wan22_ti2v_5b_dit_q8.safetensors, wan22_vae_f16.safetensors, umt5_xxl_enc_q8.safetensors, google/umt5-xxl/tokenizer.json)" };
        if (dlg.ShowDialog() != true) return;
        _modelFolder = dlg.FolderName; _folderPicked = true;
        ShowFolder();
    }

    private async Task EnsureAcceleratorAsync()
    {
        if (_accelerator != null) return;
        await Task.Run(() =>
        {
            _context = MLContext.Create().ToContext();
            var cuda = _context.GetCudaDevices(); var cl = _context.GetCLDevices();
            Device device = cuda.Count > 0 ? (Device)cuda[0] : cl.Count > 0 ? (Device)cl[0] : _context.GetPreferredDevice(preferCPU: false);
            _accelerator = device.CreateAccelerator(_context);
        });
        Log($"Accelerator: {_accelerator!.Name} ({_accelerator.AcceleratorType})");
    }

    private async void GenerateClick(object sender, RoutedEventArgs e)
    {
        string prompt = PromptBox.Text.Trim();
        if (prompt.Length == 0) return;
        var size = Content(SizeBox).Split('x');
        int width = int.Parse(size[0]), height = int.Parse(size[1]);
        bool ltx = Ltx;
        int frames = int.Parse(Content(FramesBox)), steps = ltx ? 7 : int.Parse(Content(StepsBox));
        double guidance = double.TryParse(GuidanceBox.Text, System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out var g) ? g : 5.0;
        long seed = long.TryParse(SeedBox.Text, out var sd) ? sd : Random.Shared.NextInt64(1, int.MaxValue);
        bool turbo = !ltx && DitBox.SelectedIndex == 0;
        string dit = ltx ? (DitBox.SelectedIndex == 0 ? "q8" : "bf16") : DitBox.SelectedIndex switch { 0 => "turbo_q8", 1 => "q8", 2 => "q4", _ => "bf16" }, text = TextEncBox.SelectedIndex == 0 ? "q8" : "bf16";
        if (turbo) guidance = 1.0;

        Busy(true); _timer.Stop(); ReplayButton.IsEnabled = SaveButton.IsEnabled = false; Bar.Value = 0; InfoText.Text = "";
        var total = System.Diagnostics.Stopwatch.StartNew();
        try
        {
            await EnsureAcceleratorAsync();
            if (ltx)
            {
                LtxPipeline lp = _modelFolder != null
                    ? new LtxPipeline(_accelerator!, LtxPaths.InFolder(_modelFolder, LtxPaths.DitFile(dit), LtxPaths.TextEncoderFile(text)))
                    : new LtxPipeline(_accelerator!, new HttpClientModelSource(_http, new FileModelStore(_cacheDir)), LtxPipeline.DefaultRepo, LtxPaths.InRepo(LtxPaths.DitFile(dit), LtxPaths.TextEncoderFile(text)));
                Log($"Text to video (LTX-Video 2B): {width}x{height}, {frames} frames, 7 steps, seed {seed}.");
                var lprogress = new Progress<WanProgress>(p =>
                {
                    StatusText.Text = $"{p.Stage}: {p.Message}";
                    Bar.Value = 100 * p.Stage switch { "text" => 0.3 * p.Fraction, "denoise" => 0.3 + 0.55 * p.Fraction, "decode" => 0.85 + 0.15 * p.Fraction, _ => 0 };
                });
                _video = await Task.Run(() => lp.GenerateAsync(new LtxRequest { Prompt = prompt, Width = width, Height = height, Frames = frames, Seed = seed }, lprogress));
                ShowVideo(seed, total);
                return;
            }
            Wan22Pipeline pipe = _modelFolder != null
                ? new Wan22Pipeline(_accelerator!, Wan22Paths.InFolder(_modelFolder, Wan22Paths.DitFile(dit), Wan22Paths.TextEncoderFile(text)))
                : new Wan22Pipeline(_accelerator!, new HttpClientModelSource(_http, new FileModelStore(_cacheDir)), Wan22Pipeline.DefaultRepo, Wan22Paths.InRepo(Wan22Paths.DitFile(dit), Wan22Paths.TextEncoderFile(text)));
            var request = new WanRequest
            {
                Prompt = prompt, NegativePrompt = string.IsNullOrWhiteSpace(NegativeBox.Text) ? null : NegativeBox.Text.Trim(), Width = width, Height = height, Frames = frames, Steps = steps,
                GuidanceScale = guidance, Seed = seed, Sampler = turbo ? WanSampler.FewStep : WanSampler.UniPC, ImageRgb = _pictureRgb, ImageWidth = _pictureW, ImageHeight = _pictureH,
            };
            Log($"{(_pictureRgb != null ? "Image to video" : "Text to video")}: {width}x{height}, {frames} frames, {steps} steps, seed {seed}.");
            var progress = new Progress<WanProgress>(p =>
            {
                StatusText.Text = $"{p.Stage}: {p.Message}";
                Bar.Value = 100 * p.Stage switch { "text" => 0.1 * p.Fraction, "image" => 0.1, "denoise" => 0.1 + 0.8 * p.Fraction, "decode" => 0.9 + 0.1 * p.Fraction, _ => 0 };
            });
            _video = await Task.Run(() => pipe.GenerateAsync(request, progress));
            ShowVideo(seed, total);
        }
        catch (Exception ex) { Log("Failed: " + ex.Message); MessageBox.Show(ex.ToString(), "Generation failed"); StatusText.Text = "Failed."; }
        finally { Busy(false); }
    }

    private void ShowVideo(long seed, System.Diagnostics.Stopwatch total)
    {
        var video = _video!;
        _frames = Enumerable.Range(0, video.Frames).Select(f => ToBitmap(video.Width, video.Height, video.GetFrameRgba(f))).ToList();
        _frameIndex = 0; Player.Source = _frames[0]; _timer.Start();
        ReplayButton.IsEnabled = SaveButton.IsEnabled = true;
        InfoText.Text = $"{video.Frames} frames {video.Width} x {video.Height} at 24 fps, seed {seed}, {total.Elapsed.TotalSeconds:F0} s";
        StatusText.Text = "Done."; Bar.Value = 100;
    }

    private void DitChanged(object sender, SelectionChangedEventArgs e)
    {
        if (!IsLoaded) return;
        if (DitBox.SelectedIndex == 0) { StepsBox.SelectedIndex = 0; GuidanceBox.Text = "1"; }            // Turbo: 4 steps, guidance built in
        else { if (StepsBox.SelectedIndex < 3) StepsBox.SelectedIndex = 4; GuidanceBox.Text = "5"; }       // the standard recipe: 30 steps, guidance 5
    }

    private void ReplayClick(object sender, RoutedEventArgs e) { _frameIndex = 0; if (_frames.Count > 0) Player.Source = _frames[0]; _timer.Start(); }

    private void SaveClick(object sender, RoutedEventArgs e)
    {
        if (_frames.Count == 0) return;
        var dlg = new OpenFolderDialog { Title = "Folder for the PNG frames" };
        if (dlg.ShowDialog() != true) return;
        for (int i = 0; i < _frames.Count; i++)
        {
            var encoder = new PngBitmapEncoder();
            encoder.Frames.Add(BitmapFrame.Create(_frames[i]));
            using var fs = File.Create(Path.Combine(dlg.FolderName, $"frame_{i:D4}.png"));
            encoder.Save(fs);
        }
        Log($"Saved {_frames.Count} frames to {dlg.FolderName}");
    }

    protected override void OnClosed(EventArgs e)
    {
        _timer.Stop(); _accelerator?.Dispose(); _context?.Dispose(); _http.Dispose();
        base.OnClosed(e);
    }
}
