using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SpawnDev.ILGPU.ML.Pipelines;

namespace ZipVoiceHarness;

/// <summary>
/// ZipVoice's three graphs run on onnxruntime - the reference engine the ILGPU implementation is
/// measured against.
/// </summary>
/// <remarks>
/// This exists to make failures attributable. The orchestration around the graphs (mel features, the
/// flow-matching loop, the inverse STFT) is shared with the shipping pipeline, so if this produces
/// good audio the algorithm is right, and any difference in our engine's output is our engine.
/// </remarks>
public sealed class OrtZipVoiceGraphs : IZipVoiceGraphs
{
    private readonly InferenceSession _encoder;
    private readonly InferenceSession _decoder;
    private readonly InferenceSession _vocoder;

    /// <summary>Open the three graphs from a sherpa-onnx ZipVoice model directory.</summary>
    /// <param name="modelDir">Directory holding the encoder, decoder and vocoder.</param>
    /// <param name="int8">Load the quantized encoder/decoder instead of the full-precision pair.</param>
    public OrtZipVoiceGraphs(string modelDir, bool int8 = false)
    {
        var encoderPath = Resolve(modelDir, int8 ? "encoder.int8.onnx" : "text_encoder.onnx");
        var decoderPath = Resolve(modelDir, int8 ? "decoder.int8.onnx" : "fm_decoder.onnx");
        // The vocoder is shipped separately from the quantized package and is shared between them, so
        // fall back to the sibling folder rather than demanding a second copy on disk.
        var vocoderPath = Resolve(modelDir, "vocos_24khz.onnx", SiblingModelDir(modelDir));

        _encoder = new InferenceSession(encoderPath);
        _decoder = new InferenceSession(decoderPath);
        _vocoder = new InferenceSession(vocoderPath);
    }

    /// <summary>Names of the loaded graph files, for the record a test run prints.</summary>
    public string Description { get; private set; } = "";

    private string Resolve(string modelDir, string name, string? fallbackDir = null)
    {
        var path = Path.Combine(modelDir, name);
        if (!File.Exists(path) && fallbackDir != null) path = Path.Combine(fallbackDir, name);
        if (!File.Exists(path))
            throw new FileNotFoundException($"ZipVoice model file not found: {Path.Combine(modelDir, name)}");
        Description += (Description.Length > 0 ? ", " : "") + $"{name} ({new FileInfo(path).Length / 1048576} MB)";
        return path;
    }

    /// <summary>The other precision's package folder, where a shared file may live instead.</summary>
    private static string SiblingModelDir(string modelDir) =>
        modelDir.Contains("-int8-", StringComparison.OrdinalIgnoreCase)
            ? modelDir.Replace("-int8-", "-", StringComparison.OrdinalIgnoreCase)
            : modelDir.Replace("zipvoice-distill-", "zipvoice-distill-int8-", StringComparison.OrdinalIgnoreCase);

    public Task<ZipVoiceEncoding> RunEncoderAsync(
        long[] tokens, long[] promptTokens, long promptFeatureFrames, float speed)
    {
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("tokens", new DenseTensor<long>(tokens, new[] { 1, tokens.Length })),
            NamedOnnxValue.CreateFromTensor("prompt_tokens", new DenseTensor<long>(promptTokens, new[] { 1, promptTokens.Length })),
            // Declared as a 0-d scalar in the graph, so the shape is empty, not { 1 }.
            NamedOnnxValue.CreateFromTensor("prompt_features_len", new DenseTensor<long>(new[] { promptFeatureFrames }, Array.Empty<int>())),
            NamedOnnxValue.CreateFromTensor("speed", new DenseTensor<float>(new[] { speed }, Array.Empty<int>())),
        };

        using var results = _encoder.Run(inputs);
        var output = results.First().AsTensor<float>();
        return Task.FromResult(new ZipVoiceEncoding(
            output.ToArray(), output.Dimensions[1], output.Dimensions[2]));
    }

    public Task<float[]> RunDecoderAsync(
        float t, float[] x, float[] textCondition, float[] speechCondition,
        float guidanceScale, int numFrames, int featDim)
    {
        var shape = new[] { 1, numFrames, featDim };
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("t", new DenseTensor<float>(new[] { t }, Array.Empty<int>())),
            NamedOnnxValue.CreateFromTensor("x", new DenseTensor<float>(x, shape)),
            NamedOnnxValue.CreateFromTensor("text_condition", new DenseTensor<float>(textCondition, shape)),
            NamedOnnxValue.CreateFromTensor("speech_condition", new DenseTensor<float>(speechCondition, shape)),
            NamedOnnxValue.CreateFromTensor("guidance_scale", new DenseTensor<float>(new[] { guidanceScale }, Array.Empty<int>())),
        };

        using var results = _decoder.Run(inputs);
        return Task.FromResult(results.First().AsTensor<float>().ToArray());
    }

    public Task<ZipVoiceSpectrum> RunVocoderAsync(float[] melChannelsFirst, int channels, int frames)
    {
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("mels", new DenseTensor<float>(melChannelsFirst, new[] { 1, channels, frames })),
        };

        using var results = _vocoder.Run(inputs);
        var byName = results.ToDictionary(r => r.Name, r => r.AsTensor<float>());
        var magnitude = byName["mag"];
        var cos = byName["x"];
        var sin = byName["y"];

        return Task.FromResult(new ZipVoiceSpectrum(
            magnitude.ToArray(), cos.ToArray(), sin.ToArray(),
            Bins: magnitude.Dimensions[1], Frames: magnitude.Dimensions[2]));
    }

    public void Dispose()
    {
        _encoder.Dispose();
        _decoder.Dispose();
        _vocoder.Dispose();
    }
}
