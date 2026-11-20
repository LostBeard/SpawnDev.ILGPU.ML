using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.Trellis2;
using SpawnDev.ILGPU.ML.StableDiffusion;
using SpawnDev.ILGPU.ML.Wan;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// Rule 4 step 3 (2026-11-20, Lore / TJ: pictures and latents stay on the GPU): the VAEs' device encode and ComfyUI IMAGE ([F, H, W, 3] in 0..1) entry points against
/// their host paths. The pictures are built from 0..1 values k / 255, so the host's channel-first 2q - 1 and the device's FromImage01 are the same float (2q is exact,
/// one rounding either way): every comparison is on identical inputs.
/// </summary>
public abstract partial class MLTestBase
{
    // a smooth synthetic picture / clip as ComfyUI's IMAGE [frames, H, W, 3] in 0..1 (k / 255) and the same values channel-first [3, frames, H, W] in [-1, 1]
    private static (float[] Image01, float[] ChannelFirst) Dv_Picture(int frames, int H, int W)
    {
        var img = new float[3L * frames * H * W]; var cf = new float[img.Length];
        long plane = (long)frames * H * W;
        for (int f = 0; f < frames; f++)
            for (int y = 0; y < H; y++)
                for (int x = 0; x < W; x++)
                    for (int c = 0; c < 3; c++)
                    {
                        int k = (c == 0 ? x * 255 / Math.Max(1, W - 1) : c == 1 ? y * 255 / Math.Max(1, H - 1) : (x + y + 17 * f) * 3) % 256;
                        float q = k / 255f;
                        long p = ((long)f * H + y) * W + x;
                        img[p * 3 + c] = q;
                        cf[c * plane + p] = 2f * q - 1f;
                    }
        return (img, cf);
    }

    private static int Dv_BitDiff(float[] a, float[] b)
    {
        if (a.Length != b.Length) return int.MaxValue;
        int d = 0; for (int i = 0; i < a.Length; i++) if (BitConverter.SingleToInt32Bits(a[i]) != BitConverter.SingleToInt32Bits(b[i])) d++;
        return d;
    }

    /// <summary>
    /// Rule 4 step 4: a transformer call into the caller's own buffer (<paramref name="callInto"/>) must equal the same call's <paramref name="output"/> (the transformer's
    /// Output, already computed for the same inputs) bit for bit, and leave Output untouched.
    /// </summary>
    private static async Task<string> Dv_CallerView(Accelerator accelerator, ArrayView1D<float, Stride1D.Dense> output, Func<ArrayView1D<float, Stride1D.Dense>, Task> callInto)
    {
        var before = await output.CopyToHostAsync();
        using var mine = accelerator.Allocate1D<float>(output.Length);
        await callInto(mine.View);
        var a = await mine.View.CopyToHostAsync(); var after = await output.CopyToHostAsync();
        int d1 = Dv_BitDiff(before, a), d2 = Dv_BitDiff(before, after);
        string line = $"caller-view output: {d1} of {a.Length} values differ from Output, Output changed in {d2}";
        Ur_Expect(d1 == 0 && d2 == 0, line);
        return line;
    }

    private static double Dv_Rel(float[] a, float[] b)
    {
        double n = 0, d = 0; for (int i = 0; i < a.Length; i++) { double e = (double)a[i] - b[i]; n += e * e; d += (double)a[i] * a[i]; }
        return d > 0 ? Math.Sqrt(n / d) : 0;
    }

    /// <summary>
    /// The device encodes against the host encodes, same picture: WanVae (2.2, and 2.1 = Qwen-Image's VAE when present) EncodeImageDeviceAsync / EncodeImage01Async vs
    /// EncodeImageAsync (the normalisation divides on the device: within 1e-6, bit differences reported), LtxVaeEncoder and Ltx2VideoEncoder EncodeDeviceAsync /
    /// EncodeFrames01Async vs EncodeAsync (the same kernels: bit for bit). The IMAGE path must equal the channel-first device path bit for bit everywhere.
    /// </summary>
    [TestMethod(Timeout = 3600000, Category = "HeavyModel")]
    public async Task Vae_DeviceEncode_MatchesHostEncode() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is not (AcceleratorType.Cuda or AcceleratorType.OpenCL)) throw new UnsupportedTestException("the VAE files are read from disk: a desktop GPU");
        var report = new List<string>();

        async Task WanCase(string label, WanVae vae, int H, int W)
        {
            var (img01, cf) = Dv_Picture(1, H, W);
            var host = await vae.EncodeImageAsync(cf, H, W);
            using var cfDev = accelerator.Allocate1D(cf); using var imgDev = accelerator.Allocate1D(img01);
            using var d1 = await vae.EncodeImageDeviceAsync(cfDev.View, H, W);
            using var d2 = await vae.EncodeImage01Async(imgDev.View, H, W);
            var a = await d1.View.CopyToHostAsync(); var b = await d2.View.CopyToHostAsync();
            double rel = Dv_Rel(host, a); int bits = Dv_BitDiff(host, a), bits2 = Dv_BitDiff(a, b);
            report.Add($"{label} {W}x{H}: device vs host {rel:E2} ({bits} of {host.Length} values differ in a bit), IMAGE vs channel-first {bits2} differ");
            Ur_Expect(a.Length == host.Length && rel < 1e-6 && bits2 == 0, report[^1]);
        }

        var wan22 = Path.Combine(WanDir, "wan22_vae_f16.safetensors");
        if (File.Exists(wan22))
        {
            using var w = GpuWeights.LoadFile(accelerator, wan22); using var vae = new WanVae(accelerator, w);
            await WanCase("Wan 2.2", vae, 64, 96);
        }
        else report.Add("Wan 2.2: file not here");
        string? qi = null;
        try { qi = Qi_Require(accelerator, Path.Combine("vae", "qwen_image_vae.safetensors")); } catch (UnsupportedTestException) { report.Add("Wan 2.1: file not here"); }
        if (qi != null)
        {
            using var w = GpuWeights.LoadFile(accelerator, qi);
            await WanComfyFormat.AdaptVaeAsync(accelerator, w);
            using var vae = new WanVae(accelerator, w);
            await WanCase("Wan 2.1", vae, 64, 96);
        }

        async Task LtxCase(string label, Func<float[], Task<float[]>> host, Func<ArrayView1D<float, Stride1D.Dense>, Task<MemoryBuffer1D<float, Stride1D.Dense>>> dev,
            Func<ArrayView1D<float, Stride1D.Dense>, Task<MemoryBuffer1D<float, Stride1D.Dense>>> dev01, int frames, int H, int W)
        {
            var (img01, cf) = Dv_Picture(frames, H, W);
            var h = await host(cf);
            using var cfDev = accelerator.Allocate1D(cf); using var imgDev = accelerator.Allocate1D(img01);
            using var d1 = await dev(cfDev.View); using var d2 = await dev01(imgDev.View);
            var a = await d1.View.CopyToHostAsync(); var b = await d2.View.CopyToHostAsync();
            int bits = Dv_BitDiff(h, a), bits2 = Dv_BitDiff(a, b);
            report.Add($"{label} {frames}f {W}x{H}: device vs host {bits} of {h.Length} differ, IMAGE vs channel-first {bits2} differ");
            Ur_Expect(bits == 0 && bits2 == 0, report[^1]);
        }

        var ltxEnc = Path.Combine(LtxDir, "ltx_vae_encoder_f16.safetensors");
        if (File.Exists(ltxEnc))
        {
            using var w = GpuWeights.LoadFile(accelerator, ltxEnc); using var enc = new Ltx.LtxVaeEncoder(accelerator, w);
            await LtxCase("LTX", cf => enc.EncodeAsync(cf, 9, 64, 96), v => enc.EncodeDeviceAsync(v, 9, 64, 96), v => enc.EncodeFrames01Async(v, 9, 64, 96), 9, 64, 96);
        }
        else report.Add("LTX: file not here");
        var ltx2 = Path.Combine(Ltx2_Models(), "lightricks", "ltx-2-19b-dev-fp8.safetensors");
        if (File.Exists(ltx2))
        {
            using var enc = await Ltx2.Ltx2VideoEncoder.OpenAsync(accelerator, new LocalFolderModelSource(Path.GetDirectoryName(ltx2)!), "", Path.GetFileName(ltx2));
            await LtxCase("LTX-2", cf => enc.EncodeAsync(cf, 9, 64, 64), v => enc.EncodeDeviceAsync(v, 9, 64, 64), v => enc.EncodeFrames01Async(v, 9, 64, 64), 9, 64, 64);
        }
        else report.Add("LTX-2: file not here");

        string line = string.Join("; ", report);
        Console.WriteLine("[Vae] device encode: " + line);
        return line;
    });

    /// <summary>
    /// SdVae (vae-ft-mse) and SdTaesd: DecodeImage01Async against the host DecodeAsync through ComfyUI's IMAGE formula, and EncodeImage01Async against the host
    /// EncodeAsync, bit for bit (the same kernels on the same values).
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task SdVae_Image01_MatchHostPaths() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is not (AcceleratorType.Cuda or AcceleratorType.OpenCL)) throw new UnsupportedTestException("the VAE files are read from disk: a desktop GPU");
        var report = new List<string>();
        const int H = 64, W = 96, h = H / 8, w = W / 8;
        var (img01, cf) = Dv_Picture(1, H, W);
        var z = Wan22Pipeline.Gaussian(4L * h * w, 21);

        async Task Check(string label, Func<float[], Task<float[]>> decHost, Func<ArrayView1D<float, Stride1D.Dense>, Task<MemoryBuffer1D<float, Stride1D.Dense>>> dec01,
            Func<float[], Task<float[]>> encHost, Func<ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, Task> enc01)
        {
            using var zDev = accelerator.Allocate1D(z);
            var host = await decHost(z);
            using var dev = await dec01(zDev.View);
            string d = DeviceFramesCheck(host, await dev.View.CopyToHostAsync(), 1, H, W);
            var e = await encHost(cf);
            using var imgDev = accelerator.Allocate1D(img01); using var lat = accelerator.Allocate1D<float>(4L * h * w);
            await enc01(imgDev.View, lat.View);
            var le = await lat.View.CopyToHostAsync();
            int bits = Dv_BitDiff(e, le);
            report.Add($"{label}: decode {d}; encode from the IMAGE {bits} of {e.Length} differ");
            Ur_Expect(bits == 0, report[^1]);
        }

        var vaeFile = Path.Combine(SdDir, "split", "vae", "vae-ft-mse-840000-ema-pruned.safetensors");
        if (File.Exists(vaeFile))
        {
            await using var s = new FileStream(vaeFile, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 20, useAsync: true);
            using var vae = await SdVae.LoadAsync(accelerator, s, "vae-ft-mse");
            await Check("SdVae (vae-ft-mse)", x => vae.DecodeAsync(x, h, w), v => vae.DecodeImage01Async(v, h, w), x => vae.EncodeAsync(x, H, W), (i, m) => vae.EncodeImage01Async(i, H, W, m));
        }
        else report.Add("SdVae: file not here");
        string dir = Path.Combine(SdDir, "taesd");
        if (File.Exists(Path.Combine(dir, "taesd_decoder.safetensors")) && File.Exists(Path.Combine(dir, "taesd_encoder.safetensors")))
        {
            FileStream Open(string f) => new(Path.Combine(dir, f), FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 20, useAsync: true);
            await using var ds = Open("taesd_decoder.safetensors"); await using var es = Open("taesd_encoder.safetensors");
            using var t = await SdTaesd.LoadAsync(accelerator, ds, "taesd_decoder.safetensors", es, "taesd_encoder.safetensors", 0.18215f);
            await Check("SdTaesd", x => t.DecodeAsync(x, h, w), v => t.DecodeImage01Async(v, h, w), x => t.EncodeAsync(x, H, W), (i, m) => t.EncodeImage01Async(i, H, W, m));
        }
        else report.Add("SdTaesd: files not here");
        string line = string.Join("; ", report);
        Console.WriteLine("[Vae] IMAGE paths: " + line);
        return line;
    });
}
