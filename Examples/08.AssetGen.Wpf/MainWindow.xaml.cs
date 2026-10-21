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
using Microsoft.Web.WebView2.Core;
using Microsoft.Web.WebView2.Wpf;
using Microsoft.Win32;
using SpawnDev.ILGPU.ML.Assets;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.Trellis2;

namespace SpawnDev.ILGPU.ML.Examples.AssetGenWpf;

/// <summary>
/// Example 08: the browser demo's Text and Image to 3D page as a desktop app on CUDA / OpenCL. Describe something (FLUX 3 / FLUX.2 / SD-Turbo draws it, FLUX 3 with Qwen3 writing the full caption) or choose a picture
/// of your own, let it auto-pick the best fit or draw 4 / 8 and pick (and draw more), build it in 3D (TRELLIS.2) at the triangle budget and texture size you choose, rig it (SkinTokens) and animate it (Kimodo clips, with
/// arm clearance); the 3D preview is the demo's own WebGL viewer hosted in WebView2. All of the work is SpawnDev.ILGPU.ML's <see cref="AssetGenerator"/>; this file is only the window.
/// </summary>
public partial class MainWindow : Window
{
    public sealed record PictureItem(AssetPicture Picture, BitmapSource Image) { public string Caption => $"seed {Picture.Seed}"; }

    private readonly ObservableCollection<PictureItem> _items = new();
    private readonly HttpClient _http = new() { Timeout = Timeout.InfiniteTimeSpan };
    private Context? _context;
    private Accelerator? _accelerator;
    private AssetGenerator? _gen;
    private int _genBits;
    private string _cacheDir = Environment.GetEnvironmentVariable("ASSETGEN_CACHE") ?? FileModelStore.Default().RootDirectory;   // ASSETGEN_CACHE: start with the models somewhere else (a big drive)
    private byte[]? _glb, _rigged, _animated;
    private PictureItem? _built;
    private AssetPicture? _photo;
    private int _nextSeed;
    private string _drawPrompt = "";
    private readonly string _viewerDir = Path.Combine(Path.GetTempPath(), "AssetGen.Wpf", "viewer");
    private int _glbCounter;
    private IReadOnlyList<string> _moveIds = Array.Empty<string>();
    private bool _ready;

    public MainWindow()
    {
        InitializeComponent();
        Pictures.ItemsSource = _items;
        _ready = true;
        ModelChanged(this, null!);
        CacheText.Text = $"Models are cached in {_cacheDir} (about 25 GB when everything is downloaded): use Model folder... to put them on a big drive. FLUX 3's weights are under the FLUX Kommunity License (non-commercial use).";
        Loaded += async (_, _) => await InitViewerAsync();
    }

    private async Task InitViewerAsync()
    {
        try
        {
            Directory.CreateDirectory(_viewerDir);
            // the viewer page, glb-viewer.js and the three.js files it imports (viewer/lib/...): the whole folder, so the relative module imports resolve
            var source = Path.Combine(AppContext.BaseDirectory, "viewer");
            foreach (var f in Directory.GetFiles(source, "*", SearchOption.AllDirectories))
            {
                var target = Path.Combine(_viewerDir, Path.GetRelativePath(source, f));
                Directory.CreateDirectory(Path.GetDirectoryName(target)!);
                File.Copy(f, target, true);
            }
            Viewer.CreationProperties = new CoreWebView2CreationProperties { UserDataFolder = Path.Combine(Path.GetTempPath(), "AssetGen.Wpf", "webview") };
            await Viewer.EnsureCoreWebView2Async();
            Viewer.CoreWebView2.SetVirtualHostNameToFolderMapping("assets.local", _viewerDir, CoreWebView2HostResourceAccessKind.Allow);
            Viewer.CoreWebView2.Navigate("https://assets.local/viewer.html");
        }
        catch (Exception ex) { Log("3D preview unavailable (is the WebView2 runtime installed?): " + ex.Message); }
    }

    // ───────────────────────────── helpers ─────────────────────────────

    private void Log(string line) => Dispatcher.InvokeAsync(() => { LogBox.AppendText(line + Environment.NewLine); LogBox.ScrollToEnd(); });

    private ImageModel Model => ModelBox.SelectedIndex switch { 0 => ImageModel.Flux3Klein, 1 => ImageModel.Flux2Klein, _ => ImageModel.SdTurbo };
    private bool IsImageMode => ImageMode.IsChecked == true;
    private int Picks => int.Parse((string)((ComboBoxItem)CountBox.SelectedItem).Tag);        // 0 = auto-pick the best fit
    private int Size => int.Parse(((ComboBoxItem)SizeBox.SelectedItem).Content.ToString()!);
    private int Seed => int.TryParse(SeedBox.Text, out var s) ? s : 1;

    private Trellis2Options TrellisOptions(int seed) => new()
    {
        Seed = seed, MaxFaces = int.Parse((string)((ComboBoxItem)FacesBox.SelectedItem).Tag), TextureSize = int.Parse(((ComboBoxItem)TextureBox.SelectedItem).Content.ToString()!),
    };

    private void Busy(bool busy)
    {
        DrawButton.IsEnabled = !busy; CacheButton.IsEnabled = !busy; ChooseImageButton.IsEnabled = !busy;
        MoreButton.IsEnabled = !busy && _items.Count > 0 && !IsImageMode && _nextSeed > 0;
        RigButton.IsEnabled = !busy && _glb != null; AnimateButton.IsEnabled = !busy && _rigged != null; ClearBox.IsEnabled = !busy && _rigged != null;
        SaveButton.IsEnabled = !busy && _glb != null; SaveRiggedButton.IsEnabled = !busy && _rigged != null; SaveAnimatedButton.IsEnabled = !busy && _animated != null;
        foreach (var b in new Control[] { ModelBox, CountBox, SeedBox, BitsBox, SizeBox, PromptBox, FacesBox, TextureBox, CaptionBox, TextMode, ImageMode }) b.IsEnabled = !busy;
        if (!busy) UpdateModelControls();
    }

    private void UpdateModelControls()
    {
        CaptionBox.Visibility = Model == ImageModel.Flux3Klein ? Visibility.Visible : Visibility.Collapsed;
        SizeBox.IsEnabled = Model != ImageModel.SdTurbo;                       // SD-Turbo draws 512
    }

    private void ModelChanged(object sender, SelectionChangedEventArgs e) { if (_ready) UpdateModelControls(); }

    private void ModeChanged(object sender, RoutedEventArgs e)
    {
        if (!_ready) return;
        TextPanel.Visibility = IsImageMode ? Visibility.Collapsed : Visibility.Visible;
        ImagePanel.Visibility = IsImageMode ? Visibility.Visible : Visibility.Collapsed;
        DrawButton.Content = IsImageMode ? "Build 3D from the picture" : "Generate";
        MoreButton.Visibility = IsImageMode ? Visibility.Collapsed : Visibility.Visible;
    }

    private static BitmapSource ToBitmap(AssetPicture p)
    {
        var bgra = new byte[p.Rgba.Length];
        for (int i = 0; i < bgra.Length; i += 4) { bgra[i] = p.Rgba[i + 2]; bgra[i + 1] = p.Rgba[i + 1]; bgra[i + 2] = p.Rgba[i]; bgra[i + 3] = 255; }
        var bmp = BitmapSource.Create(p.Width, p.Height, 96, 96, PixelFormats.Bgra32, null, bgra, p.Width * 4);
        bmp.Freeze();
        return bmp;
    }

    private async Task<AssetGenerator> EnsureGeneratorAsync()
    {
        int bits = BitsBox.SelectedIndex == 1 ? 8 : 4;
        if (_gen != null && _genBits == bits) return _gen;
        _gen?.Dispose();
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
        var (loader, ids) = AssetGenerator.MotionsFromDirectory(Path.Combine(AppContext.BaseDirectory, "motions"));
        _moveIds = ids;
        _gen = new AssetGenerator(_accelerator!, source, loader)
        {
            ImageModelBits = bits, KeepImageModelResident = true,
            OpenCaptionModel = (repo, file, progress) => source.OpenAsync(repo, file, progress),
        };
        _gen.StageDone = (stage, s) => Log($"{stage,-40} {s,6:F1} s");
        _gen.Status = s => Dispatcher.InvokeAsync(() => StatusText.Text = s);
        _gen.CaptionText = text => Dispatcher.InvokeAsync(() => { CaptionText.Text = text; });
        _genBits = bits;
        return _gen;
    }

    private async Task ShowGlbAsync(byte[] glb)
    {
        if (Viewer.CoreWebView2 == null) return;
        string name = $"model_{++_glbCounter}.glb";
        await File.WriteAllBytesAsync(Path.Combine(_viewerDir, name), glb);
        await Viewer.ExecuteScriptAsync($"loadGlb('https://assets.local/{name}')");
    }

    /// <summary>A model was built: show it, and reset what depended on the previous one.</summary>
    private async Task OnBuiltAsync(byte[] glb, PictureItem? source)
    {
        _glb = glb; _rigged = _animated = null; _built = source;
        await ShowGlbAsync(glb);
        Log($"3D model: {glb.Length / 1048576.0:F1} MB.");
        MoveBox.ItemsSource = null; MoveBox.IsEnabled = false;
    }

    private void ShowCaptionUsed(AssetGenerator gen)
    {
        if (gen.UsedCaption != null) { CaptionExpander.Visibility = Visibility.Visible; CaptionExpander.Header = "Caption used"; CaptionText.Text = gen.UsedCaption; CaptionExpander.IsExpanded = false; }
        else CaptionExpander.Visibility = Visibility.Collapsed;
    }

    private void BeginCaption()
    {
        bool write = !IsImageMode && Model == ImageModel.Flux3Klein && CaptionBox.IsChecked == true;
        CaptionText.Text = "";
        CaptionExpander.Visibility = write ? Visibility.Visible : Visibility.Collapsed; CaptionExpander.Header = "Writing the caption"; CaptionExpander.IsExpanded = true;
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

    private void ChooseImageClick(object sender, RoutedEventArgs e)
    {
        var dlg = new OpenFileDialog { Title = "A picture of one whole object", Filter = "Pictures (*.png;*.jpg;*.jpeg;*.bmp;*.webp)|*.png;*.jpg;*.jpeg;*.bmp;*.webp" };
        if (dlg.ShowDialog() != true) return;
        try
        {
            _photo = LoadPicture(dlg.FileName, Seed);
            ImageName.Text = $"{Path.GetFileName(dlg.FileName)} ({_photo.Width} x {_photo.Height}{(AssetGenerator.HasTransparency(_photo.Rgba) ? ", transparent: used as is" : ", will be cut out")})";
            _items.Clear(); _items.Add(new PictureItem(_photo, ToBitmap(_photo)));
        }
        catch (Exception ex) { Log($"Could not read {Path.GetFileName(dlg.FileName)}: {ex.Message}"); }
    }

    /// <summary>A picture file as RGBA with its alpha kept (a transparent background becomes the 3D stage's cut-out); a longer side above 2048 is shrunk.</summary>
    private static AssetPicture LoadPicture(string path, int seed)
    {
        var decoder = BitmapDecoder.Create(new Uri(path), BitmapCreateOptions.PreservePixelFormat, BitmapCacheOption.OnLoad);
        BitmapSource src = decoder.Frames[0];
        int longer = Math.Max(src.PixelWidth, src.PixelHeight);
        if (longer > 2048) { double scale = 2048.0 / longer; src = new TransformedBitmap(src, new ScaleTransform(scale, scale)); }
        var bgra = new FormatConvertedBitmap(src, PixelFormats.Bgra32, null, 0);
        int w = bgra.PixelWidth, h = bgra.PixelHeight;
        var px = new byte[w * h * 4];
        bgra.CopyPixels(px, w * 4, 0);
        for (int i = 0; i < px.Length; i += 4) (px[i], px[i + 2]) = (px[i + 2], px[i]);     // BGRA -> RGBA
        return new AssetPicture(seed, w, h, px);
    }

    private async void GenerateClick(object sender, RoutedEventArgs e)
    {
        if (IsImageMode)
        {
            if (_photo == null) { Log("Choose a picture first."); return; }
            await BuildFromPhotoAsync(_photo with { Seed = Seed });
            return;
        }
        string prompt = PromptBox.Text.Trim();
        if (prompt.Length == 0) return;
        int seed = Seed, picks = Picks, size = Size;
        var model = Model;
        Busy(true); _items.Clear(); _glb = _rigged = _animated = null; _nextSeed = 0; BeginCaption();
        try
        {
            var gen = await EnsureGeneratorAsync();
            gen.WriteCaption = CaptionBox.IsChecked == true;
            _drawPrompt = prompt;
            if (picks == 0)
            {   // auto: draw two, keep the one that matches the words and fits its frame best, build it
                Log($"Drawing and building \"{prompt}\" ({model}, {size} px) from seed {seed}; the first run downloads the models.");
                var asset = await Task.Run(() => gen.GenerateAsync(prompt, seed, 2, model, size, TrellisOptions(seed), rig: false));
                var item = new PictureItem(asset.Picture, ToBitmap(asset.Picture));
                _items.Add(item);
                await OnBuiltAsync(asset.Glb, item);
                ShowCaptionUsed(gen);
            }
            else
            {   // the user chooses: draw them all and wait
                Log($"Drawing {picks} picture(s) of \"{prompt}\" ({model}, {size} px) from seed {seed}; the first run downloads the model.");
                await DrawAsync(gen, prompt, seed, picks, model, size);
                _nextSeed = seed + picks;
            }
        }
        catch (Exception ex) { Log("Failed: " + ex.Message); MessageBox.Show(ex.ToString(), "Generation failed"); }
        finally { Busy(false); StatusText.Text = _glb != null ? "Done. Rig it to animate it, or save the .glb." : _items.Count > 0 ? "Pick a picture to build in 3D." : ""; }
    }

    private async Task DrawAsync(AssetGenerator gen, string prompt, int firstSeed, int count, ImageModel model, int size)
    {
        await Task.Run(() => gen.DrawAsync(prompt, Enumerable.Range(firstSeed, count).ToArray(), model, size,
            pic => Dispatcher.InvokeAsync(() => _items.Add(new PictureItem(pic, ToBitmap(pic))), System.Windows.Threading.DispatcherPriority.Normal).Task));
        ShowCaptionUsed(gen);
    }

    /// <summary>"Draw 4 more": the same prompt with the next seeds, added to the pictures to choose from.</summary>
    private async void MoreClick(object sender, RoutedEventArgs e)
    {
        if (_nextSeed <= 0 || _drawPrompt.Length == 0) return;
        Busy(true); BeginCaption();
        try
        {
            var gen = await EnsureGeneratorAsync();
            gen.WriteCaption = CaptionBox.IsChecked == true;
            Log($"Drawing 4 more of \"{_drawPrompt}\" from seed {_nextSeed}.");
            await DrawAsync(gen, _drawPrompt, _nextSeed, 4, Model, Size);
            _nextSeed += 4;
        }
        catch (Exception ex) { Log("Failed: " + ex.Message); MessageBox.Show(ex.ToString(), "Drawing failed"); }
        finally { Busy(false); StatusText.Text = "Pick a picture to build in 3D."; }
    }

    private async Task BuildFromPhotoAsync(AssetPicture photo)
    {
        Busy(true); _rigged = _animated = null;
        try
        {
            var gen = await EnsureGeneratorAsync();
            Log("Building your picture in 3D (first run downloads about 11 GB of TRELLIS.2 weights)...");
            var asset = await Task.Run(() => gen.GenerateFromPictureAsync(photo, TrellisOptions(photo.Seed), rig: false));
            await OnBuiltAsync(asset.Glb, _items.FirstOrDefault());
        }
        catch (Exception ex) { Log("Failed: " + ex.Message); MessageBox.Show(ex.ToString(), "Building failed"); }
        finally { Busy(false); StatusText.Text = _glb != null ? "Done. Rig it to animate it, or save the .glb." : ""; }
    }

    private async void BuildClick(object sender, RoutedEventArgs e)
    {
        if (((Button)sender).Tag is not PictureItem item) return;
        Busy(true); _rigged = _animated = null;
        try
        {
            var gen = await EnsureGeneratorAsync();
            Log($"Building seed {item.Picture.Seed} in 3D (first run downloads about 11 GB of TRELLIS.2 weights)...");
            var glb = await Task.Run(async () =>
            {
                if (AssetGenerator.HasTransparency(item.Picture.Rgba)) return (await gen.GenerateFromPictureAsync(item.Picture, TrellisOptions(item.Picture.Seed), rig: false)).Glb;
                var (cut, touch) = await gen.CutOutAsync(item.Picture);
                if (touch > 0.02) Log($"note: the subject touches {touch:P0} of the picture's edge, so the model may be cropped; pick another picture for a whole object");
                return await gen.BuildModelAsync(cut, item.Picture.Width, item.Picture.Height, TrellisOptions(item.Picture.Seed));
            });
            await OnBuiltAsync(glb, item);
        }
        catch (Exception ex) { Log("Failed: " + ex.Message); MessageBox.Show(ex.ToString(), "Building failed"); }
        finally { Busy(false); StatusText.Text = _glb != null ? "Done. Rig it to animate it, or save the .glb." : ""; }
    }

    private async void RigClick(object sender, RoutedEventArgs e)
    {
        if (_glb == null) return;
        Busy(true);
        try
        {
            var gen = await EnsureGeneratorAsync();
            _rigged = await Task.Run(() => gen.RigAsync(_glb));
            _animated = null;
            await ShowGlbAsync(_rigged);
            Log("Rigged: a skeleton and learned skin weights.");
        }
        catch (Exception ex) { Log("Failed: " + ex.Message); MessageBox.Show(ex.ToString(), "Rigging failed"); }
        finally { Busy(false); StatusText.Text = _rigged != null ? "Rigged. Press Animate to add the moves." : ""; }
    }

    private async void AnimateClick(object sender, RoutedEventArgs e)
    {
        if (_rigged == null) return;
        Busy(true);
        try
        {
            var gen = await EnsureGeneratorAsync();
            double clearance = double.Parse(((ComboBoxItem)ClearBox.SelectedItem).Content.ToString()!);
            _animated = await Task.Run(() => gen.AnimateAsync(_rigged, _moveIds, clearance));
            await ShowGlbAsync(_animated);
            MoveBox.ItemsSource = _moveIds; MoveBox.IsEnabled = true;
            MoveBox.SelectedItem = _moveIds.Contains("walk") ? "walk" : _moveIds.FirstOrDefault();
            Log($"Animated: {_moveIds.Count} moves.");
        }
        catch (SpawnDev.ILGPU.ML.Animation.NotAHumanoidException ex) { Log("This model does not look like a standing humanoid, so the moves cannot be mapped onto it: " + ex.Message); }
        catch (Exception ex) { Log("Failed: " + ex.Message); MessageBox.Show(ex.ToString(), "Animating failed"); }
        finally { Busy(false); StatusText.Text = _animated != null ? "Pick a move; drag to orbit. Save keeps the animations in the .glb." : ""; }
    }

    /// <summary>Arm clearance is applied when the moves are retargeted, so changing it animates again (as the browser demo does).</summary>
    private void ClearanceChanged(object sender, SelectionChangedEventArgs e)
    {
        if (!_ready || !ClearBox.IsEnabled || _animated == null) return;
        AnimateClick(sender, new RoutedEventArgs());
    }

    private async void MoveChanged(object sender, RoutedEventArgs e)
    {
        if (_animated == null || MoveBox.SelectedItem is not string move || Viewer.CoreWebView2 == null) return;
        await Viewer.ExecuteScriptAsync($"playMove('{move}', {(InPlaceBox.IsChecked == true ? "true" : "false")})");
    }

    private void Save(byte[]? data, string suffix)
    {
        if (data == null) return;
        var dlg = new SaveFileDialog { Filter = "glTF binary (*.glb)|*.glb", FileName = $"asset_seed{_built?.Picture.Seed}{suffix}.glb" };
        if (dlg.ShowDialog() == true) { File.WriteAllBytes(dlg.FileName, data); Log($"Saved {dlg.FileName}"); }
    }

    private void SaveClick(object sender, RoutedEventArgs e) => Save(_glb, "");
    private void SaveRiggedClick(object sender, RoutedEventArgs e) => Save(_rigged, "_rigged");
    private void SaveAnimatedClick(object sender, RoutedEventArgs e) => Save(_animated, "_animated");

    protected override void OnClosed(EventArgs e)
    {
        _gen?.Dispose(); _accelerator?.Dispose(); _context?.Dispose(); _http.Dispose();
        base.OnClosed(e);
    }
}
