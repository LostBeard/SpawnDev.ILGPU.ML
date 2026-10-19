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
/// Example 08: the browser demo's text-to-character chain as a desktop app on CUDA / OpenCL. Describe something, draw several pictures, pick one, build it in 3D (TRELLIS.2), rig it (SkinTokens)
/// and animate it (Kimodo clips); the 3D preview is the demo's own WebGL viewer hosted in WebView2. All of the work is SpawnDev.ILGPU.ML's <see cref="AssetGenerator"/>; this file is only the window.
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
    private readonly string _viewerDir = Path.Combine(Path.GetTempPath(), "AssetGen.Wpf", "viewer");
    private int _glbCounter;
    private IReadOnlyList<string> _moveIds = Array.Empty<string>();

    public MainWindow()
    {
        InitializeComponent();
        Pictures.ItemsSource = _items;
        CacheText.Text = $"Models are cached in {_cacheDir} (about 25 GB when everything is downloaded): use Model folder... to put them on a big drive.";
        Loaded += async (_, _) => await InitViewerAsync();
    }

    private async Task InitViewerAsync()
    {
        try
        {
            Directory.CreateDirectory(_viewerDir);
            foreach (var f in new[] { "viewer.html", "glb-viewer.js" }) File.Copy(Path.Combine(AppContext.BaseDirectory, "viewer", f), Path.Combine(_viewerDir, f), true);
            Viewer.CreationProperties = new CoreWebView2CreationProperties { UserDataFolder = Path.Combine(Path.GetTempPath(), "AssetGen.Wpf", "webview") };
            await Viewer.EnsureCoreWebView2Async();
            Viewer.CoreWebView2.SetVirtualHostNameToFolderMapping("assets.local", _viewerDir, CoreWebView2HostResourceAccessKind.Allow);
            Viewer.CoreWebView2.Navigate("https://assets.local/viewer.html");
        }
        catch (Exception ex) { Log("3D preview unavailable (is the WebView2 runtime installed?): " + ex.Message); }
    }

    // ───────────────────────────── helpers ─────────────────────────────

    private void Log(string line) => Dispatcher.InvokeAsync(() => { LogBox.AppendText(line + Environment.NewLine); LogBox.ScrollToEnd(); });

    private void Busy(bool busy)
    {
        DrawButton.IsEnabled = !busy; CacheButton.IsEnabled = !busy;
        RigButton.IsEnabled = !busy && _glb != null; AnimateButton.IsEnabled = !busy && _rigged != null; ClearBox.IsEnabled = !busy && _rigged != null;
        SaveButton.IsEnabled = !busy && _glb != null;
        foreach (var b in new Control[] { ModelBox, CountBox, SeedBox, BitsBox, SizeBox, PromptBox }) b.IsEnabled = !busy;
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
        var sw = System.Diagnostics.Stopwatch.StartNew();
        _gen = new AssetGenerator(_accelerator!, source, loader) { ImageModelBits = bits, KeepImageModelResident = true };
        _gen.StageDone = (stage, s) => Log($"{stage,-40} {s,6:F1} s");
        _gen.Status = s => Dispatcher.InvokeAsync(() => StatusText.Text = s);
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

    private async void DrawClick(object sender, RoutedEventArgs e)
    {
        string prompt = PromptBox.Text.Trim();
        if (prompt.Length == 0) return;
        int seed = int.TryParse(SeedBox.Text, out var s) ? s : 1;
        int count = int.Parse(((ComboBoxItem)CountBox.SelectedItem).Content.ToString()!);
        int size = int.Parse(((ComboBoxItem)SizeBox.SelectedItem).Content.ToString()!);
        var model = ModelBox.SelectedIndex == 1 ? ImageModel.SdTurbo : ImageModel.Flux2Klein;
        Busy(true); _items.Clear();
        try
        {
            var gen = await EnsureGeneratorAsync();
            Log($"Drawing {count} picture(s) of \"{prompt}\" ({model}, {size} px) from seed {seed}; the first run downloads the model.");
            await Task.Run(() => gen.DrawAsync(prompt, Enumerable.Range(seed, count).ToArray(), model, size,
                pic => Dispatcher.InvokeAsync(() => _items.Add(new PictureItem(pic, ToBitmap(pic))), System.Windows.Threading.DispatcherPriority.Normal).Task));
        }
        catch (Exception ex) { Log("Failed: " + ex.Message); MessageBox.Show(ex.ToString(), "Drawing failed"); }
        finally { Busy(false); StatusText.Text = _items.Count > 0 ? "Pick a picture to build in 3D." : ""; }
    }

    private async void BuildClick(object sender, RoutedEventArgs e)
    {
        if (((Button)sender).Tag is not PictureItem item) return;
        Busy(true); _rigged = _animated = null; _built = item;
        try
        {
            var gen = await EnsureGeneratorAsync();
            Log($"Building seed {item.Picture.Seed} in 3D (first run downloads about 11 GB of TRELLIS.2 weights)...");
            _glb = await Task.Run(async () =>
            {
                var (cut, touch) = await gen.CutOutAsync(item.Picture);
                if (touch > 0.02) Log($"note: the subject touches {touch:P0} of the picture's edge, so the model may be cropped; pick another picture for a whole object");
                return await gen.BuildModelAsync(cut, item.Picture.Width, item.Picture.Height, new Trellis2Options { Seed = item.Picture.Seed });
            });
            await ShowGlbAsync(_glb);
            Log($"3D model: {_glb.Length / 1048576.0:F1} MB.");
        }
        catch (Exception ex) { Log("Failed: " + ex.Message); MessageBox.Show(ex.ToString(), "Building failed"); }
        finally { Busy(false); StatusText.Text = _glb != null ? "Done. Rig it to animate it, or save the .glb." : ""; MoveBox.ItemsSource = null; MoveBox.IsEnabled = false; }
    }

    private async void RigClick(object sender, RoutedEventArgs e)
    {
        if (_glb == null) return;
        Busy(true);
        try
        {
            var gen = await EnsureGeneratorAsync();
            _rigged = await Task.Run(() => gen.RigAsync(_glb));
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

    private async void MoveChanged(object sender, RoutedEventArgs e)
    {
        if (_animated == null || MoveBox.SelectedItem is not string move || Viewer.CoreWebView2 == null) return;
        await Viewer.ExecuteScriptAsync($"playMove('{move}', {(InPlaceBox.IsChecked == true ? "true" : "false")})");
    }

    private void SaveClick(object sender, RoutedEventArgs e)
    {
        var data = _animated ?? _rigged ?? _glb;
        if (data == null) return;
        string suffix = _animated != null ? "_animated" : _rigged != null ? "_rigged" : "";
        var dlg = new SaveFileDialog { Filter = "glTF binary (*.glb)|*.glb", FileName = $"asset_seed{_built?.Picture.Seed}{suffix}.glb" };
        if (dlg.ShowDialog() == true) { File.WriteAllBytes(dlg.FileName, data); Log($"Saved {dlg.FileName}"); }
    }

    protected override void OnClosed(EventArgs e)
    {
        _gen?.Dispose(); _accelerator?.Dispose(); _context?.Dispose(); _http.Dispose();
        base.OnClosed(e);
    }
}
