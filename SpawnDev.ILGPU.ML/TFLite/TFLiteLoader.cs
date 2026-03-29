using System.Text.Json;
using SpawnDev.ILGPU.ML.Graph;

namespace SpawnDev.ILGPU.ML.TFLite;

/// <summary>
/// Converts a parsed TFLiteModel into the shared ModelGraph IR + CPU weights.
/// This bridges TFLite models into the same compilation pipeline as ONNX.
///
/// Usage:
///   var tfliteBytes = await http.GetByteArrayAsync("model.tflite");
///   var (graph, weights) = TFLiteLoader.LoadModel(tfliteBytes);
///   var session = InferenceSession.CreateFromModelGraph(accelerator, graph, weights);
/// </summary>
public static class TFLiteLoader
{
    /// <summary>
    /// Load a TFLite model from raw bytes.
    /// Returns a ModelGraph + CPU weight dictionary ready for InferenceSession.
    /// </summary>
    public static (ModelGraph Graph, Dictionary<string, float[]> Weights) LoadModel(byte[] tfliteBytes)
    {
        var model = TFLiteParser.Parse(tfliteBytes);
        return ConvertToModelGraph(model);
    }

    /// <summary>
    /// Parse a TFLite file and return a summary string (for logging/display).
    /// </summary>
    public static string GetModelSummary(byte[] tfliteBytes)
    {
        var model = TFLiteParser.Parse(tfliteBytes);
        return TFLiteParser.GetSummary(model);
    }

    /// <summary>
    /// Convert a parsed TFLiteModel to ModelGraph + weight dictionary.
    /// </summary>
    public static (ModelGraph Graph, Dictionary<string, float[]> Weights) ConvertToModelGraph(TFLiteModel model)
    {
        if (model.Subgraphs.Length == 0)
            throw new InvalidOperationException("TFLite model has no subgraphs");

        var sg = model.Subgraphs[0]; // TFLite models typically have one subgraph
        var graph = new ModelGraph { Name = model.Description.Length > 0 ? model.Description : "tflite_model" };
        var weights = new Dictionary<string, float[]>();

        // Build tensor name map — TFLite uses indices, we need names
        // Use the tensor's name field if available, otherwise generate one
        var tensorNames = new string[sg.Tensors.Length];
        for (int i = 0; i < sg.Tensors.Length; i++)
        {
            tensorNames[i] = !string.IsNullOrEmpty(sg.Tensors[i].Name)
                ? sg.Tensors[i].Name
                : $"tensor_{i}";
        }

        // Identify inputs (subgraph inputs that are NOT initializers/constants)
        foreach (int inputIdx in sg.Inputs)
        {
            var tensor = sg.Tensors[inputIdx];
            var buffer = model.Buffers[tensor.BufferIndex];
            // Skip inputs that have buffer data (they're constants, not real inputs)
            if (buffer.DataLength > 0) continue;

            graph.Inputs.Add(new GraphValueInfo
            {
                Name = tensorNames[inputIdx],
                Shape = ConvertShape(tensor.Shape, tensor.Type)
            });
        }

        // Identify outputs
        foreach (int outputIdx in sg.Outputs)
        {
            graph.Outputs.Add(new GraphValueInfo
            {
                Name = tensorNames[outputIdx],
                Shape = sg.Tensors[outputIdx].Shape
            });
        }

        // Extract constant tensors (weights/biases) as initializers
        for (int i = 0; i < sg.Tensors.Length; i++)
        {
            var tensor = sg.Tensors[i];
            var buffer = model.Buffers[tensor.BufferIndex];
            if (buffer.DataLength == 0) continue; // not a constant

            var name = tensorNames[i];
            graph.Initializers[name] = tensor.Shape;

            // Extract weight data as float[]
            var data = model.GetTensorData(tensor);
            if (data != null)
            {
                weights[name] = data;
                // TFLite graphs reference dequantized/converted tensor names with
                // _dequantize suffix. Register under both names for compatibility.
                weights[name + "_dequantize"] = data;
                graph.Initializers[name + "_dequantize"] = tensor.Shape;
            }
        }

        // Convert operators to graph nodes
        for (int opIdx = 0; opIdx < sg.Operators.Length; opIdx++)
        {
            var op = sg.Operators[opIdx];
            var opName = model.GetOperatorName(op.OpcodeIndex);
            var onnxOpType = TFLiteBuiltinOps.ToOnnxOpType(
                model.OperatorCodes[op.OpcodeIndex].BuiltinCode);

            if (onnxOpType == null)
            {
                // No ONNX equivalent — create a passthrough or skip
                // Log for debugging
                if (InferenceSession.VerboseLogging)
                    Console.WriteLine($"[TFLite] Unmapped operator: {opName} (builtin={model.OperatorCodes[op.OpcodeIndex].BuiltinCode})");
                continue;
            }

            int builtinCodeForNode = model.OperatorCodes[op.OpcodeIndex].BuiltinCode;

            // FULLY_CONNECTED: split into MatMul + optional bias Add
            if (builtinCodeForNode == 9 && op.Inputs.Length >= 2)
            {
                var matmulInputs = new List<string>();
                if (op.Inputs[0] >= 0) matmulInputs.Add(tensorNames[op.Inputs[0]]);
                if (op.Inputs[1] >= 0) matmulInputs.Add(tensorNames[op.Inputs[1]]);

                bool hasBias = op.Inputs.Length > 2 && op.Inputs[2] >= 0;
                string matmulOutput = hasBias ? $"{tensorNames[op.Outputs[0]]}_pre_bias" : tensorNames[op.Outputs[0]];

                graph.Nodes.Add(new GraphNode
                {
                    OpType = "MatMul",
                    Inputs = matmulInputs,
                    Outputs = new List<string> { matmulOutput }
                });

                if (hasBias)
                {
                    graph.Nodes.Add(new GraphNode
                    {
                        OpType = "Add",
                        Inputs = new List<string> { matmulOutput, tensorNames[op.Inputs[2]] },
                        Outputs = new List<string> { tensorNames[op.Outputs[0]] }
                    });
                }

                // Handle fused activation for FC (same pattern as Conv)
                HandleFusedActivation(graph, model, op, tensorNames, 3);
                continue;
            }

            var node = new GraphNode
            {
                OpType = onnxOpType,
                Inputs = op.Inputs.Where(i => i >= 0).Select(i => tensorNames[i]).ToList(),
                Outputs = op.Outputs.Select(i => tensorNames[i]).ToList(),
            };

            // Extract operator-specific attributes from builtin options
            var attrs = ExtractAttributes(model, op, sg);
            if (attrs.Count > 0)
                node.Attributes = attrs;

            graph.Nodes.Add(node);

            // Handle fused activations (TFLite fuses RELU/RELU6 into Conv/Pool ops)
            if (builtinCodeForNode is 3 or 4 or 1 or 17)
                HandleFusedActivation(graph, model, op, tensorNames, 3);
        }

        return (graph, weights);
    }

    /// <summary>
    /// Convert TFLite shape to ONNX-style shape.
    /// TFLite uses NHWC by default, ONNX uses NCHW.
    /// For 4D tensors, transpose the shape.
    /// </summary>
    private static int[] ConvertShape(int[] tfliteShape, TFLiteTensorType type)
    {
        // TFLite 4D: [N, H, W, C] → ONNX 4D: [N, C, H, W]
        if (tfliteShape.Length == 4)
            return new[] { tfliteShape[0], tfliteShape[3], tfliteShape[1], tfliteShape[2] };
        return tfliteShape;
    }

    /// <summary>
    /// Extract operator attributes from TFLite builtin options.
    /// Maps TFLite-specific attributes to ONNX-equivalent attribute names.
    /// </summary>
    private static Dictionary<string, JsonElement> ExtractAttributes(
        TFLiteModel model, TFLiteOperator op, TFLiteSubGraph sg)
    {
        var attrs = new Dictionary<string, JsonElement>();
        var fb = new FlatBufferReader(model.RawData);
        int builtinCode = model.OperatorCodes[op.OpcodeIndex].BuiltinCode;
        int optOffset = op.BuiltinOptionsOffset;

        if (optOffset == 0) return attrs;

        switch (builtinCode)
        {
            case 3: // CONV_2D
            case 4: // DEPTHWISE_CONV_2D
                ExtractConvAttributes(fb, optOffset, builtinCode, attrs);
                break;
            case 1: // AVERAGE_POOL_2D
            case 17: // MAX_POOL_2D
                ExtractPoolAttributes(fb, optOffset, attrs);
                break;
            case 22: // RESHAPE
                // Reshape target shape comes from the second input tensor
                if (op.Inputs.Length > 1 && op.Inputs[1] >= 0)
                {
                    var shapeTensor = sg.Tensors[op.Inputs[1]];
                    var shapeData = model.GetTensorData(shapeTensor);
                    if (shapeData != null)
                        attrs["shape"] = JsonSerializer.SerializeToElement(shapeData.Select(v => (long)v).ToArray());
                }
                break;
            case 25: // SOFTMAX
                // Softmax axis — TFLite defaults to -1
                attrs["axis"] = JsonSerializer.SerializeToElement(-1L);
                break;
        }

        return attrs;
    }

    private static void ExtractConvAttributes(FlatBufferReader fb, int offset, int builtinCode,
        Dictionary<string, JsonElement> attrs)
    {
        // Conv2DOptions / DepthwiseConv2DOptions:
        // 0: padding (Padding enum: 0=SAME, 1=VALID)
        // 1: stride_w (int)
        // 2: stride_h (int)
        // For DepthwiseConv2D: 4: depth_multiplier (int)
        byte padding = fb.ReadFieldByte(offset, 0);
        int strideW = fb.ReadFieldInt32(offset, 1, 1);
        int strideH = fb.ReadFieldInt32(offset, 2, 1);

        attrs["strides"] = JsonSerializer.SerializeToElement(new long[] { strideH, strideW });

        if (padding == 0) // SAME
            attrs["auto_pad"] = JsonSerializer.SerializeToElement("SAME_UPPER");
        else // VALID
            attrs["auto_pad"] = JsonSerializer.SerializeToElement("VALID");

        if (builtinCode == 4) // DEPTHWISE_CONV_2D
        {
            // TFLite depthwise: group = inC (each input channel is its own group)
            // Set group = -1 as sentinel — ConvOperator resolves to inC at execution time
            attrs["group"] = JsonSerializer.SerializeToElement((long)-1);
        }
    }

    private static void ExtractPoolAttributes(FlatBufferReader fb, int offset,
        Dictionary<string, JsonElement> attrs)
    {
        // Pool2DOptions:
        // 0: padding (Padding enum)
        // 1: stride_w (int)
        // 2: stride_h (int)
        // 3: filter_width (int)
        // 4: filter_height (int)
        byte padding = fb.ReadFieldByte(offset, 0);
        int strideW = fb.ReadFieldInt32(offset, 1, 1);
        int strideH = fb.ReadFieldInt32(offset, 2, 1);
        int filterW = fb.ReadFieldInt32(offset, 3, 1);
        int filterH = fb.ReadFieldInt32(offset, 4, 1);

        attrs["kernel_shape"] = JsonSerializer.SerializeToElement(new long[] { filterH, filterW });
        attrs["strides"] = JsonSerializer.SerializeToElement(new long[] { strideH, strideW });

        if (padding == 0)
            attrs["auto_pad"] = JsonSerializer.SerializeToElement("SAME_UPPER");
        else
            attrs["auto_pad"] = JsonSerializer.SerializeToElement("VALID");
    }

    /// <summary>
    /// Handle TFLite fused activation functions.
    /// TFLite fuses RELU/RELU6/TANH into Conv, Pool, and FC operators.
    /// We insert a separate activation node after the op to match ONNX graph structure.
    /// </summary>
    private static void HandleFusedActivation(ModelGraph graph, TFLiteModel model,
        TFLiteOperator op, string[] tensorNames, int fusedActFieldIndex)
    {
        var fb = new FlatBufferReader(model.RawData);
        int optOffset = op.BuiltinOptionsOffset;
        if (optOffset == 0) return;

        byte fusedAct = fb.ReadFieldByte(optOffset, fusedActFieldIndex);
        if (fusedAct == 0) return; // NONE

        string actOpType = fusedAct switch
        {
            1 => "Relu",
            3 => "Clip",       // RELU6
            4 => "Tanh",
            5 => "Sigmoid",    // SIGN_BIT → Sigmoid approximation
            _ => "Relu"
        };

        // Rename the last node's output, insert activation
        var lastNode = graph.Nodes[^1];
        string origOutput = lastNode.Outputs[^1];
        string preActOutput = $"{origOutput}_pre_act";
        lastNode.Outputs[^1] = preActOutput;

        var actNode = new GraphNode
        {
            OpType = actOpType,
            Inputs = new List<string> { preActOutput },
            Outputs = new List<string> { origOutput },
        };

        if (fusedAct == 3) // RELU6 → Clip [0, 6]
        {
            actNode.Attributes = new Dictionary<string, JsonElement>
            {
                ["min"] = JsonSerializer.SerializeToElement(0.0),
                ["max"] = JsonSerializer.SerializeToElement(6.0)
            };
        }

        graph.Nodes.Add(actNode);
    }
}
