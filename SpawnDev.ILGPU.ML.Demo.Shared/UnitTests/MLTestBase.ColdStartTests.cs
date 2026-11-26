using System.Diagnostics;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Tensors;
using SpawnDev.ILGPU.WebGPU;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// Where a model's cold first run goes in the browser (Bones' comparison, 2026-11-25: MobileNetV2 on WebGPU, our first run 7.4 s, ONNX Runtime Web's whole first visit 0.9 s).
/// A measurement, not a gate.
/// </summary>
public abstract partial class MLTestBase
{
    /// <summary>
    /// MobileNetV2 on WebGPU: create, first and second run; then the distinct WGSL shaders the first run compiled, compiled again in Chrome one at a time and all at once
    /// (<c>createComputePipelineAsync</c>, ILGPU's PrecompileComputeShaderAsync). Each copy gets its own entry-point name, so neither Chrome's in-process nor its disk shader cache
    /// (keyed on the generated code) can answer it: both numbers are real compiles. The difference is what compiling a model's shaders in parallel, before its first run, can save.
    /// </summary>
    [TestMethod(Timeout = 600000)]
    public async Task<string> ColdStart_MobileNetV2_ShaderCompile_SerialVsParallel() => await RunTest(accelerator => ColdStartMobileNetV2(accelerator));

    private async Task<string> ColdStartMobileNetV2(Accelerator accelerator)
    {
        if (accelerator is not WebGPUAccelerator web) throw new UnsupportedTestException("the browser's WebGPU shader compile");
        var http = GetHttpClient() ?? throw new UnsupportedTestException("HttpClient not available");
        var shaders = new List<(string Name, string Wgsl)>();
        var prev = global::SpawnDev.ILGPU.WebGPU.Backend.WebGPUBackend.OnShaderCompiled;
        global::SpawnDev.ILGPU.WebGPU.Backend.WebGPUBackend.OnShaderCompiled = (n, s, e) =>
        {
            lock (shaders) shaders.Add((n, s));
            if (n is "Gemm" or "BicubicResizeImpl" or "RMSNormStatsImpl" or "GELUImpl")       // who loads a kernel MobileNetV2 has no op for
            {
                var frames = Environment.StackTrace.Split('\n').Select(f => f.Trim()).Where(f => f.Contains("SpawnDev.ILGPU.ML") && !f.Contains("UnitTests")).Take(4);
                Console.WriteLine($"[ColdStart] loaded {n} from: {string.Join(" <- ", frames)}");
            }
            prev?.Invoke(n, s, e);
        };
        double create, first, second;
        var input = new float[3 * 224 * 224];
        var rng = new Random(7);
        for (int i = 0; i < input.Length; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);
        try
        {
            var bytes = await http.GetByteArrayAsync("models/mobilenetv2/model.onnx");
            var sw = Stopwatch.StartNew();
            using var session = await InferenceSession.CreateFromStreamAsync(accelerator, new MemoryStream(bytes, writable: false));
            create = sw.Elapsed.TotalMilliseconds;
            async Task Run()
            {
                using var t = OwnedTensor<float>.FromHost(accelerator, input, new[] { 1, 3, 224, 224 });
                using var o = await session.RunOwnedAsync(new Dictionary<string, Tensor<float>> { [session.InputNames[0]] = t });
                await o[session.OutputNames[0]].ToHostAsync();
            }
            sw.Restart(); await Run(); first = sw.Elapsed.TotalMilliseconds;
            sw.Restart(); await Run(); second = sw.Elapsed.TotalMilliseconds;
            // MEASURED 2026-11-25: TiledGemm.Gemm (308 KB, 3.9 s in Chrome on a quiet machine) came with every fp32 convolution through Conv2DGemm.RunF32 until that route was
            // removed (Conv_DirectVsTiled_Crossover: the direct implicit GEMM is as fast or faster); its single-mode variants are as large (239 and 203 KB), so the size is the
            // register tile, not the modes; keeping its inner loops rolled made it 285 KB and slower to compile. After: 7 shaders, 173 KB, 0.23 s, first run 849 ms (was 2388).
        }
        finally { global::SpawnDev.ILGPU.WebGPU.Backend.WebGPUBackend.OnShaderCompiled = prev; }

        var distinct = shaders.GroupBy(s => s.Wgsl).Select(g => g.First()).Where(s => s.Wgsl.Contains("fn main(")).ToList();
        if (distinct.Count == 0)
            throw new UnsupportedTestException("earlier tests on this page compiled every kernel MobileNetV2 uses: a cold start needs this test alone (PMT_FILTER=ColdStart_)");
        long kb = distinct.Sum(s => (long)s.Wgsl.Length) / 1024;
        var native = web.NativeAccelerator;
        var perShader = new List<(string Name, int Kb, double Ms)>();
        async Task<(double Ms, int Ok)> Compile(string tag, bool parallel)
        {
            int before = native.CachedShaderCount;
            var copies = distinct.Select((s, i) => (Src: s.Wgsl.Replace("fn main(", $"fn {tag}_{i}("), Entry: $"{tag}_{i}", s.Name)).ToList();
            var sw = Stopwatch.StartNew();
            if (parallel) await Task.WhenAll(copies.Select(c => native.PrecompileComputeShaderAsync(c.Src, c.Entry, c.Name)));
            else
                foreach (var c in copies)
                {
                    var one = Stopwatch.StartNew();
                    await native.PrecompileComputeShaderAsync(c.Src, c.Entry, c.Name);
                    perShader.Add((c.Name, c.Src.Length / 1024, one.Elapsed.TotalMilliseconds));
                }
            return (sw.Elapsed.TotalMilliseconds, native.CachedShaderCount - before);
        }
        string salt = "s" + Guid.NewGuid().ToString("N")[..8];
        var serial = await Compile(salt + "a", parallel: false);
        var all = await Compile(salt + "b", parallel: true);
        var biggest = distinct.OrderByDescending(s => s.Wgsl.Length).Take(3).Select(s => $"{s.Name} {s.Wgsl.Length / 1024} KB");
        var report = $"MobileNetV2 WebGPU: create {create:F0} ms, first run {first:F0} ms, second {second:F0} ms; {distinct.Count} distinct shaders from {shaders.Count} compiles, {kb} KB WGSL " +
                     $"(largest: {string.Join(", ", biggest)}); compiled again in Chrome: one at a time {serial.Ms:F0} ms ({serial.Ok} ok), all at once {all.Ms:F0} ms ({all.Ok} ok)";
        Console.WriteLine("[ColdStart] " + report);
        foreach (var s in perShader.OrderByDescending(p => p.Ms))
            Console.WriteLine($"[ColdStart] shader {s.Name}: {s.Kb} KB, {s.Ms:F0} ms");
        if (serial.Ok == 0 || all.Ok == 0) throw new Exception("no shader compiled: the measurement measured nothing. " + report);
        return report;
    }
}
