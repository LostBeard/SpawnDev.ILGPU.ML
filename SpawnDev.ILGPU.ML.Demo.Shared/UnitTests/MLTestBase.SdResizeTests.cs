using System.Text.Json;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.StableDiffusion;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    /// <summary>
    /// Deep shrink's two remaining resize methods (Lore, SpawnGen's PatchModelAddDownscale): <see cref="SdResize.Area"/> = torch's F.interpolate(mode="area")
    /// and <see cref="SdResize.Bislerp"/> = ComfyUI's comfy.utils.bislerp, against references made by running those exact functions
    /// (tools/gen_sd_resize_reference.py, torch 2.11 CPU) on channel-last 48-channel maps: integer and non-integer shrinks and growths, with a zero
    /// vector, identical neighbours and opposite neighbours planted for bislerp's edge cases.
    /// </summary>
    [TestMethod(Timeout = 120000)]
    public async Task Sd_Resize_AreaAndBislerp_MatchComfyUI() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient() ?? throw new UnsupportedTestException("HttpClient not available");
        using var doc = JsonDocument.Parse(await http.GetStringAsync("references/sd_resize/cases.json"));
        var k = new SdKernels(accelerator);
        static float[] F(byte[] b) { var f = new float[b.Length / 4]; Buffer.BlockCopy(b, 0, f, 0, b.Length); return f; }
        var report = new List<string>();
        foreach (var cs in doc.RootElement.GetProperty("cases").EnumerateArray())
        {
            string name = cs.GetProperty("name").GetString()!;
            int h = cs.GetProperty("h").GetInt32(), w = cs.GetProperty("w").GetInt32(), oh = cs.GetProperty("oh").GetInt32(), ow = cs.GetProperty("ow").GetInt32(), c = cs.GetProperty("c").GetInt32();
            using var input = accelerator.Allocate1D(F(await http.GetByteArrayAsync($"references/sd_resize/{name}_in.bin")));
            using var output = accelerator.Allocate1D<float>((long)oh * ow * c);
            using var mid = accelerator.Allocate1D<float>((long)h * ow * c);
            foreach (var method in new[] { "area", "bislerp" })
            {
                var want = F(await http.GetByteArrayAsync($"references/sd_resize/{name}_{method}.bin"));
                if (method == "area") k.Resize(input.View, output.View, h, w, oh, ow, c, (int)SdResize.Area);
                else k.Bislerp(input.View, output.View, mid.View, h, w, oh, ow, c);
                var got = await output.CopyToHostAsync<float>(0, want.Length);
                double worst = 0; int at = 0;
                for (int i = 0; i < want.Length; i++) { double d = Math.Abs(got[i] - want[i]); if (!(d <= worst)) { worst = d; at = i; } }
                report.Add($"{name} {method} {worst:E1}");
                if (!(worst < 1e-4))
                    throw new Exception($"{name} {method}: max|diff| {worst:E2} at {at} (pixel {at / c}, channel {at % c}): ours {got[at]:G6}, ComfyUI {want[at]:G6} on {BackendName}");
            }
        }
        Console.WriteLine($"[SdResize] {string.Join("; ", report)} on {BackendName}");
    });
}
