using System.Text.Json;
using System.Text.Json.Serialization;
using System.Text.Json.Serialization.Metadata;
using SpawnDev.ILGPU.ML.Graph;
using SpawnDev.ILGPU.ML.Hub;

namespace SpawnDev.ILGPU.ML;

/// <summary>
/// Source-generated JSON metadata for every type the engine serializes. Trim and AOT safe: the reflection
/// serializer (JsonSerializer without a JsonTypeInfo) needs metadata a trimmed app may have removed.
/// Every option set the engine used before is reproduced by an options object over this context, so the JSON
/// text is unchanged.
/// </summary>
[JsonSerializable(typeof(bool))]
[JsonSerializable(typeof(int))]
[JsonSerializable(typeof(long))]
[JsonSerializable(typeof(float))]
[JsonSerializable(typeof(double))]
[JsonSerializable(typeof(string))]
[JsonSerializable(typeof(bool[]))]
[JsonSerializable(typeof(int[]))]
[JsonSerializable(typeof(long[]))]
[JsonSerializable(typeof(float[]))]
[JsonSerializable(typeof(double[]))]
[JsonSerializable(typeof(string[]))]
[JsonSerializable(typeof(Dictionary<string, JsonElement>))]
[JsonSerializable(typeof(Dictionary<string, WeightLoader.TensorInfo>))]
[JsonSerializable(typeof(ModelGraph))]
[JsonSerializable(typeof(HFModelInfo))]
[JsonSerializable(typeof(HFModelInfo[]))]
[JsonSerializable(typeof(HFRepoFile[]))]
internal partial class MLJsonContext : JsonSerializerContext { }

/// <summary>
/// The engine's JSON entry points. Node attributes are built from a closed set of value types (the ONNX / TF /
/// TFLite / GGUF / SafeTensors attribute kinds): typed overloads make an unsupported type a COMPILE error, and
/// the object overload names the type at runtime instead of silently emitting nothing.
/// </summary>
internal static class MLJson
{
    /// <summary>Default options (what JsonSerializer used with no options), resolved from the source-generated context.</summary>
    internal static readonly JsonSerializerOptions Default = new() { TypeInfoResolver = MLJsonContext.Default };

    /// <summary>ONNX attributes can carry NaN / Infinity (e.g. Clip bounds): same options the loader used before.</summary>
    internal static readonly JsonSerializerOptions NanSafe = new()
    {
        NumberHandling = JsonNumberHandling.AllowNamedFloatingPointLiterals,
        TypeInfoResolver = MLJsonContext.Default,
    };

    /// <summary>Indented output (ModelGraph.ToJson).</summary>
    internal static readonly JsonSerializerOptions Indented = new() { WriteIndented = true, TypeInfoResolver = MLJsonContext.Default };

    /// <summary>The metadata for <typeparamref name="T"/> under <paramref name="options"/>; throws naming the type if it is not in MLJsonContext.</summary>
    internal static JsonTypeInfo<T> Info<T>(JsonSerializerOptions options) =>
        options.GetTypeInfo(typeof(T)) as JsonTypeInfo<T>
        ?? throw new NotSupportedException($"MLJson: {typeof(T)} is not registered in MLJsonContext.");

    internal static string Serialize<T>(T value, JsonSerializerOptions? options = null) => JsonSerializer.Serialize(value, Info<T>(options ?? Default));
    internal static T? Deserialize<T>(string json, JsonSerializerOptions? options = null) => JsonSerializer.Deserialize(json, Info<T>(options ?? Default));

    internal static JsonElement ToElement(bool v) => JsonSerializer.SerializeToElement(v, MLJsonContext.Default.Boolean);
    internal static JsonElement ToElement(int v) => JsonSerializer.SerializeToElement(v, MLJsonContext.Default.Int32);
    internal static JsonElement ToElement(long v) => JsonSerializer.SerializeToElement(v, MLJsonContext.Default.Int64);
    internal static JsonElement ToElement(float v) => JsonSerializer.SerializeToElement(v, MLJsonContext.Default.Single);
    internal static JsonElement ToElement(double v) => JsonSerializer.SerializeToElement(v, MLJsonContext.Default.Double);
    internal static JsonElement ToElement(string v) => JsonSerializer.SerializeToElement(v, MLJsonContext.Default.String);
    internal static JsonElement ToElement(bool[] v) => JsonSerializer.SerializeToElement(v, MLJsonContext.Default.BooleanArray);
    internal static JsonElement ToElement(int[] v) => JsonSerializer.SerializeToElement(v, MLJsonContext.Default.Int32Array);
    internal static JsonElement ToElement(long[] v) => JsonSerializer.SerializeToElement(v, MLJsonContext.Default.Int64Array);
    internal static JsonElement ToElement(float[] v) => JsonSerializer.SerializeToElement(v, MLJsonContext.Default.SingleArray);
    internal static JsonElement ToElement(double[] v) => JsonSerializer.SerializeToElement(v, MLJsonContext.Default.DoubleArray);
    internal static JsonElement ToElement(string[] v) => JsonSerializer.SerializeToElement(v, MLJsonContext.Default.StringArray);

    /// <summary>
    /// A boxed attribute value (ONNX / TF attribute dictionaries are object-typed). Dispatches on the runtime type to
    /// the same metadata the typed overloads use, under <paramref name="options"/> (Default or NanSafe).
    /// </summary>
    internal static JsonElement ToElement(object? value, JsonSerializerOptions? options = null)
    {
        var o = options ?? Default;
        return value switch
        {
            null => JsonSerializer.SerializeToElement<string?>(null, Info<string>(o)!),
            bool v => JsonSerializer.SerializeToElement(v, Info<bool>(o)),
            int v => JsonSerializer.SerializeToElement(v, Info<int>(o)),
            long v => JsonSerializer.SerializeToElement(v, Info<long>(o)),
            float v => JsonSerializer.SerializeToElement(v, Info<float>(o)),
            double v => JsonSerializer.SerializeToElement(v, Info<double>(o)),
            string v => JsonSerializer.SerializeToElement(v, Info<string>(o)),
            bool[] v => JsonSerializer.SerializeToElement(v, Info<bool[]>(o)),
            int[] v => JsonSerializer.SerializeToElement(v, Info<int[]>(o)),
            long[] v => JsonSerializer.SerializeToElement(v, Info<long[]>(o)),
            float[] v => JsonSerializer.SerializeToElement(v, Info<float[]>(o)),
            double[] v => JsonSerializer.SerializeToElement(v, Info<double[]>(o)),
            string[] v => JsonSerializer.SerializeToElement(v, Info<string[]>(o)),
            JsonElement v => v.Clone(),
            _ => throw new NotSupportedException($"MLJson: unsupported attribute value type {value.GetType()} (add it to MLJsonContext and this switch)."),
        };
    }
}
