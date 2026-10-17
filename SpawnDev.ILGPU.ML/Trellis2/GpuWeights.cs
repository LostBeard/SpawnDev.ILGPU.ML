using System.Text.Json;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.SafeTensors;

namespace SpawnDev.ILGPU.ML.Trellis2;

/// <summary>
/// One checkpoint's tensors resident on the device in their NATIVE storage type (bf16 stays bf16, fp16 stays fp16),
/// streamed from the file in bounded chunks (<see cref="SafeTensorsFileReader"/>). Dispose frees the whole stage:
/// the pipeline keeps one stage's weights on a 12 GB card at a time.
/// </summary>
public sealed class GpuWeights : IDisposable
{
    private readonly Dictionary<string, MemoryBuffer1D<BFloat16, Stride1D.Dense>> _bf16 = new();
    private readonly Dictionary<string, MemoryBuffer1D<global::ILGPU.Half, Stride1D.Dense>> _f16 = new();
    private readonly Dictionary<string, MemoryBuffer1D<float, Stride1D.Dense>> _f32 = new();
    private readonly Dictionary<string, int[]> _shapes = new();

    /// <summary>The model's JSON config (<c>args</c> object of the checkpoint's sibling .json).</summary>
    public JsonElement Args { get; }

    private GpuWeights(JsonElement args) => Args = args;

    /// <summary>
    /// Load <paramref name="basePath"/> + ".safetensors" / ".json" (the checkpoint layout of TRELLIS.2).
    /// <paramref name="filter"/> (name -> keep) lets a caller skip tensors it will never read.
    /// </summary>
    public static GpuWeights Load(Accelerator accelerator, string basePath, Func<string, bool>? filter = null)
    {
        using var json = JsonDocument.Parse(File.ReadAllText(basePath + ".json"));
        return LoadCore(accelerator, basePath + ".safetensors", json.RootElement.GetProperty("args").Clone(), filter);
    }

    /// <summary>Load a bare .safetensors file (a Hugging Face checkpoint has no TRELLIS-style sibling args json); <see cref="Args"/> is empty.</summary>
    public static GpuWeights LoadFile(Accelerator accelerator, string safetensorsPath, Func<string, bool>? filter = null)
    {
        using var empty = JsonDocument.Parse("{}");
        return LoadCore(accelerator, safetensorsPath, empty.RootElement.Clone(), filter);
    }

    private static GpuWeights LoadCore(Accelerator accelerator, string safetensorsPath, JsonElement args, Func<string, bool>? filter)
    {
        var w = new GpuWeights(args);
        try
        {
            using var reader = SafeTensorsFileReader.Open(safetensorsPath);
            var prof = Environment.GetEnvironmentVariable("TRELLIS2_PROFILE") == "1"; var psw = System.Diagnostics.Stopwatch.StartNew(); double tAlloc = 0, tUp = 0;
            foreach (var t in reader.Tensors)
            {
                if (prof) psw.Restart();
                if (filter != null && !filter(t.Name)) continue;
                long n = SafeTensorsFileReader.ElementCount(t);
                switch (t.DType)
                {
                    case "BF16":
                        { var b = accelerator.Allocate1D<BFloat16>(n); w._bf16[t.Name] = b; reader.Upload(t, b.View); break; }
                    case "F16":
                        { var b = accelerator.Allocate1D<global::ILGPU.Half>(n); w._f16[t.Name] = b; reader.Upload(t, b.View); break; }
                    case "F32":
                        { var b = accelerator.Allocate1D<float>(n); w._f32[t.Name] = b; reader.Upload(t, b.View); break; }
                    default:
                        throw new NotSupportedException($"'{safetensorsPath}': tensor '{t.Name}' is {t.DType}; this loader keeps BF16/F16/F32 native only");
                }
                w._shapes[t.Name] = t.Shape;
                if (prof) tUp += psw.Elapsed.TotalSeconds;
            }
            if (prof) Console.WriteLine($"  [weights] {Path.GetFileName(safetensorsPath)} alloc+upload {tUp:F2}s (read {SafeTensorsFileReader.ReadTicks / (double)System.Diagnostics.Stopwatch.Frequency:F2}s, copy {SafeTensorsFileReader.CopyTicks / (double)System.Diagnostics.Stopwatch.Frequency:F2}s cumulative)");
            accelerator.Synchronize();
            return w;
        }
        catch
        {
            w.Dispose();
            throw;
        }
    }

    public bool Has(string name) => _shapes.ContainsKey(name);
    public int[] Shape(string name) => _shapes.TryGetValue(name, out var s) ? s : throw new KeyNotFoundException($"weight '{name}' not loaded");

    /// <summary>The bf16 tensor <paramref name="name"/> (throws when it is absent or stored as fp16).</summary>
    public ArrayView1D<BFloat16, Stride1D.Dense> BF16(string name)
        => _bf16.TryGetValue(name, out var b) ? b.View : throw new KeyNotFoundException($"bf16 weight '{name}' not loaded");

    public ArrayView1D<global::ILGPU.Half, Stride1D.Dense> F16(string name)
        => _f16.TryGetValue(name, out var b) ? b.View : throw new KeyNotFoundException($"fp16 weight '{name}' not loaded");

    public ArrayView1D<float, Stride1D.Dense> F32(string name)
        => _f32.TryGetValue(name, out var b) ? b.View : throw new KeyNotFoundException($"fp32 weight '{name}' not loaded");

    /// <summary>True when the tensor was stored (and is held) as fp32.</summary>
    public bool IsF32(string name) => _f32.ContainsKey(name);
    public bool IsF16(string name) => _f16.ContainsKey(name);

    /// <summary>Total device bytes held.</summary>
    public long Bytes => _bf16.Values.Sum(b => b.Length * 2L) + _f16.Values.Sum(b => b.Length * 2L) + _f32.Values.Sum(b => b.Length * 4L);

    public void Dispose()
    {
        foreach (var b in _bf16.Values) b.Dispose();
        foreach (var b in _f16.Values) b.Dispose();
        foreach (var b in _f32.Values) b.Dispose();
        _bf16.Clear(); _f16.Clear(); _f32.Clear(); _shapes.Clear();
    }
}
