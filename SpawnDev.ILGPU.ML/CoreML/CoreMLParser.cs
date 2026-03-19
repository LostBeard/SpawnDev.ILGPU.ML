using SpawnDev.ILGPU.ML.Onnx; // Reuse ProtobufReader

namespace SpawnDev.ILGPU.ML.CoreML;

/// <summary>
/// Minimal Core ML model parser (.mlmodel).
/// Core ML uses protocol buffers (Model.proto from coremltools).
/// Reuses our ProtobufReader from the ONNX parser.
///
/// .mlmodel file = single protobuf message (Model)
/// .mlpackage = directory with Manifest.json + model files
///
/// Model protobuf structure:
///   Model { specificationVersion, description, neuralNetwork/pipeline/... }
///   NeuralNetwork { layers[], preprocessing[], ... }
///   NeuralNetworkLayer { name, input[], output[], layer_type_oneof }
/// </summary>
public static class CoreMLParser
{
    /// <summary>
    /// Parse a Core ML .mlmodel file from raw bytes.
    /// </summary>
    public static CoreMLModel Parse(byte[] data)
    {
        var model = new CoreMLModel();
        var reader = new ProtobufReader(data);

        while (reader.HasMore)
        {
            var (fieldNumber, wireType) = reader.ReadTag();
            switch (fieldNumber)
            {
                case 1: // specificationVersion (int32)
                    model.SpecVersion = (int)reader.ReadVarint();
                    break;
                case 2: // description (ModelDescription)
                    var descBytes = reader.ReadBytes().ToArray();
                    ParseModelDescription(descBytes, model);
                    break;
                case 5: // neuralNetwork
                    var nnBytes = reader.ReadBytes().ToArray();
                    ParseNeuralNetwork(nnBytes, model);
                    break;
                case 200: // neuralNetworkClassifier
                    var nncBytes = reader.ReadBytes().ToArray();
                    ParseNeuralNetwork(nncBytes, model);
                    model.IsClassifier = true;
                    break;
                case 201: // neuralNetworkRegressor
                    var nnrBytes = reader.ReadBytes().ToArray();
                    ParseNeuralNetwork(nnrBytes, model);
                    break;
                default:
                    reader.SkipField(wireType);
                    break;
            }
        }

        return model;
    }

    /// <summary>Check if data looks like a Core ML model (protobuf with spec version).</summary>
    public static bool IsCoreML(byte[] data)
    {
        if (data.Length < 4) return false;
        // Field 1 (specificationVersion), varint type: tag = 0x08
        // Spec versions are typically 1-8
        return data[0] == 0x08 && data[1] >= 1 && data[1] <= 10;
    }

    /// <summary>Get a summary string.</summary>
    public static string GetSummary(CoreMLModel model)
    {
        return $"CoreML v{model.SpecVersion}: {model.Layers.Count} layers, " +
               $"inputs: {string.Join(", ", model.InputNames)}, " +
               $"outputs: {string.Join(", ", model.OutputNames)}";
    }

    private static void ParseModelDescription(byte[] data, CoreMLModel model)
    {
        var reader = new ProtobufReader(data);
        while (reader.HasMore)
        {
            var (fieldNumber, wireType) = reader.ReadTag();
            switch (fieldNumber)
            {
                case 1: // input (repeated FeatureDescription)
                    var inputBytes = reader.ReadBytes().ToArray();
                    var inputName = ExtractFeatureName(inputBytes);
                    if (inputName != null) model.InputNames.Add(inputName);
                    break;
                case 2: // output (repeated FeatureDescription)
                    var outputBytes = reader.ReadBytes().ToArray();
                    var outputName = ExtractFeatureName(outputBytes);
                    if (outputName != null) model.OutputNames.Add(outputName);
                    break;
                case 5: // metadata
                    reader.SkipField(wireType);
                    break;
                default:
                    reader.SkipField(wireType);
                    break;
            }
        }
    }

    private static string? ExtractFeatureName(byte[] data)
    {
        var reader = new ProtobufReader(data);
        while (reader.HasMore)
        {
            var (fieldNumber, wireType) = reader.ReadTag();
            if (fieldNumber == 1) return reader.ReadString(); // name field
            reader.SkipField(wireType);
        }
        return null;
    }

    private static void ParseNeuralNetwork(byte[] data, CoreMLModel model)
    {
        var reader = new ProtobufReader(data);
        while (reader.HasMore)
        {
            var (fieldNumber, wireType) = reader.ReadTag();
            switch (fieldNumber)
            {
                case 1: // layers (repeated NeuralNetworkLayer)
                    var layerBytes = reader.ReadBytes().ToArray();
                    var layer = ParseLayer(layerBytes);
                    if (layer != null) model.Layers.Add(layer);
                    break;
                default:
                    reader.SkipField(wireType);
                    break;
            }
        }
    }

    private static CoreMLLayer? ParseLayer(byte[] data)
    {
        var layer = new CoreMLLayer();
        var reader = new ProtobufReader(data);
        while (reader.HasMore)
        {
            var (fieldNumber, wireType) = reader.ReadTag();
            switch (fieldNumber)
            {
                case 1: // name
                    layer.Name = reader.ReadString();
                    break;
                case 2: // input (repeated string)
                    layer.Inputs.Add(reader.ReadString());
                    break;
                case 3: // output (repeated string)
                    layer.Outputs.Add(reader.ReadString());
                    break;
                default:
                    // The layer type is a oneof with field numbers 100+ for each type
                    // (convolution=100, innerProduct=101, pooling=120, etc.)
                    if (fieldNumber >= 100 && layer.LayerType == null)
                        layer.LayerType = CoreMLLayerTypeNames.GetName(fieldNumber);
                    reader.SkipField(wireType);
                    break;
            }
        }
        return layer.Name.Length > 0 ? layer : null;
    }
}

/// <summary>Parsed Core ML model.</summary>
public class CoreMLModel
{
    public int SpecVersion { get; set; }
    public bool IsClassifier { get; set; }
    public List<string> InputNames { get; set; } = new();
    public List<string> OutputNames { get; set; } = new();
    public List<CoreMLLayer> Layers { get; set; } = new();
}

/// <summary>A layer in a Core ML neural network.</summary>
public class CoreMLLayer
{
    public string Name { get; set; } = "";
    public string? LayerType { get; set; }
    public List<string> Inputs { get; set; } = new();
    public List<string> Outputs { get; set; } = new();
}

/// <summary>Maps Core ML layer type field numbers to names.</summary>
public static class CoreMLLayerTypeNames
{
    public static string GetName(int fieldNumber) => fieldNumber switch
    {
        100 => "Convolution",
        101 => "InnerProduct",
        110 => "BatchNorm",
        120 => "Pooling",
        130 => "Padding",
        140 => "Concat",
        141 => "LRN",
        148 => "Softmax",
        150 => "Split",
        160 => "Add",
        161 => "Multiply",
        162 => "UnaryFunction",
        170 => "Upsample",
        175 => "Bias",
        180 => "Activation",  // covers relu, sigmoid, tanh, etc.
        190 => "Reshape",
        200 => "Flatten",
        210 => "Permute",
        220 => "Reduce",
        230 => "LoadConstant",
        _ => $"LayerType_{fieldNumber}"
    };
}
