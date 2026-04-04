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
                Shape = tensor.Shape // Keep native NHWC — dual layout handles it
            });
        }

        // Identify outputs
        foreach (int outputIdx in sg.Outputs)
        {
            graph.Outputs.Add(new GraphValueInfo
            {
                Name = tensorNames[outputIdx],
                Shape = sg.Tensors[outputIdx].Shape // Keep native NHWC
            });
        }


        // Identify depthwise conv weight tensors via DEQUANTIZE chain trace.
        // For float16 models, weights go through DEQUANTIZE → DEPTHWISE_CONV_2D.
        // We trace back through the chain to find the source tensor with buffer data.
        var depthwiseWeightIndices = new HashSet<int>();
        var tensorProducer = new Dictionary<int, (int opIdx, int builtinCode)>();
        for (int opIdx = 0; opIdx < sg.Operators.Length; opIdx++)
        {
            var op2 = sg.Operators[opIdx];
            int bc = model.OperatorCodes[op2.OpcodeIndex].BuiltinCode;
            foreach (var outIdx in op2.Outputs)
                if (outIdx >= 0) tensorProducer[outIdx] = (opIdx, bc);
        }
        for (int opIdx = 0; opIdx < sg.Operators.Length; opIdx++)
        {
            var op2 = sg.Operators[opIdx];
            int bc = model.OperatorCodes[op2.OpcodeIndex].BuiltinCode;
            if (bc == 4 && op2.Inputs.Length >= 2 && op2.Inputs[1] >= 0) // DEPTHWISE_CONV_2D
            {
                int weightIdx = op2.Inputs[1];
                depthwiseWeightIndices.Add(weightIdx);
                // Look-back through DEQUANTIZE chain to find source with buffer data
                int traceIdx = weightIdx;
                for (int depth = 0; depth < 3; depth++) // max 3 hops
                {
                    if (tensorProducer.TryGetValue(traceIdx, out var producer) && producer.builtinCode == 6)
                    {
                        var deqOp = sg.Operators[producer.opIdx];
                        if (deqOp.Inputs.Length > 0 && deqOp.Inputs[0] >= 0)
                        {
                            depthwiseWeightIndices.Add(deqOp.Inputs[0]);
                            traceIdx = deqOp.Inputs[0];
                        }
                        else break;
                    }
                    else break;
                }
            }
        }

        // Extract constant tensors (weights/biases) as initializers
        for (int i = 0; i < sg.Tensors.Length; i++)
        {
            var tensor = sg.Tensors[i];
            var buffer = model.Buffers[tensor.BufferIndex];
            if (buffer.DataLength == 0) continue;

            var name = tensorNames[i];

            // Extract weight data as float[] — keep native NHWC layout
            var data = model.GetTensorData(tensor);
            if (data != null)
            {
                graph.Initializers[name] = tensor.Shape;
                weights[name] = data;
                weights[name + "_dequantize"] = data;
                graph.Initializers[name + "_dequantize"] = tensor.Shape;
            }
            else
            {
                graph.Initializers[name] = tensor.Shape;
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

            // ── Multi-node decompositions for ops that don't map 1:1 ──

            // RSQRT (76): 1/sqrt(x) → Sqrt then Reciprocal
            if (builtinCodeForNode == 76)
            {
                string sqrtOut = tensorNames[op.Outputs[0]] + "_sqrt";
                graph.Nodes.Add(new GraphNode { OpType = "Sqrt",
                    Inputs = op.Inputs.Where(i => i >= 0).Select(i => tensorNames[i]).ToList(),
                    Outputs = new List<string> { sqrtOut } });
                graph.Nodes.Add(new GraphNode { OpType = "Reciprocal",
                    Inputs = new List<string> { sqrtOut },
                    Outputs = op.Outputs.Select(i => tensorNames[i]).ToList() });
                HandleFusedActivation(graph, model, op, tensorNames, builtinCodeForNode == 3 || builtinCodeForNode == 4 ? 6 : 3);
                continue;
            }

            // SQUARE (90): x*x → Mul(x, x)
            if (builtinCodeForNode == 90)
            {
                var inputs = op.Inputs.Where(i => i >= 0).Select(i => tensorNames[i]).ToList();
                graph.Nodes.Add(new GraphNode { OpType = "Mul",
                    Inputs = new List<string> { inputs[0], inputs[0] },
                    Outputs = op.Outputs.Select(i => tensorNames[i]).ToList() });
                continue;
            }

            // NOT_EQUAL (72): Equal(a,b) then Not → but we map to Not which is unary.
            // Decompose: Equal then Sub from 1 (1 - Equal = NotEqual)
            if (builtinCodeForNode == 72)
            {
                var inputs = op.Inputs.Where(i => i >= 0).Select(i => tensorNames[i]).ToList();
                string eqOut = tensorNames[op.Outputs[0]] + "_eq";
                graph.Nodes.Add(new GraphNode { OpType = "Equal",
                    Inputs = inputs,
                    Outputs = new List<string> { eqOut } });
                // 1 - Equal = NotEqual (using Sub with a ones constant)
                string onesName = $"_ones_{op.Outputs[0]}";
                weights[onesName] = new float[] { 1f };
                graph.Initializers[onesName] = new[] { 1 };
                graph.Nodes.Add(new GraphNode { OpType = "Sub",
                    Inputs = new List<string> { onesName, eqOut },
                    Outputs = op.Outputs.Select(i => tensorNames[i]).ToList() });
                continue;
            }

            // EXPAND_DIMS (70): Unsqueeze with axis from second input
            if (builtinCodeForNode == 70 && onnxOpType == "Unsqueeze")
            {
                // The axis comes from the second input tensor (constant)
                var inputs = op.Inputs.Where(i => i >= 0).Select(i => tensorNames[i]).ToList();
                var attrs2 = ExtractAttributes(model, op, sg);
                if (op.Inputs.Length > 1 && op.Inputs[1] >= 0)
                {
                    var axisTensor = sg.Tensors[op.Inputs[1]];
                    var axisData = model.GetTensorData(axisTensor);
                    if (axisData != null && axisData.Length > 0)
                        attrs2["axes"] = System.Text.Json.JsonSerializer.SerializeToElement(new long[] { (long)axisData[0] });
                }
                graph.Nodes.Add(new GraphNode { OpType = "Unsqueeze",
                    Inputs = new List<string> { inputs[0] },
                    Outputs = op.Outputs.Select(i => tensorNames[i]).ToList(),
                    Attributes = attrs2 });
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
            attrs["_data_format"] = System.Text.Json.JsonSerializer.SerializeToElement("NHWC");
            node.Attributes = attrs;

            graph.Nodes.Add(node);

            // TFLite PAD constants: deinterleave [b0,e0,b1,e1,...] → ONNX [b0,b1,...,e0,e1,...]
            // No NHWC→NCHW reorder needed — model runs in native NHWC layout.
            if (onnxOpType == "Pad" && op.Inputs.Length >= 2 && op.Inputs[1] >= 0)
            {
                var padName = tensorNames[op.Inputs[1]];
                if (weights.TryGetValue(padName, out var padData) && padData.Length >= 4)
                {
                    int rank = padData.Length / 2;
                    var onnxPad = new float[padData.Length];
                    for (int d = 0; d < rank; d++)
                    {
                        onnxPad[d] = padData[d * 2];
                        onnxPad[rank + d] = padData[d * 2 + 1];
                    }
                    weights[padName] = onnxPad;
                    if (weights.ContainsKey(padName + "_dequantize"))
                        weights[padName + "_dequantize"] = onnxPad;
                }
            }

            // Handle fused activations (TFLite fuses RELU/RELU6 into Conv/Pool ops)
            if (builtinCodeForNode is 3 or 4 or 1 or 17)
                HandleFusedActivation(graph, model, op, tensorNames, 3);
        }

        return (graph, weights);
    }

    /// <summary>
    /// Transpose depthwise conv weights: [1, kH, kW, outC] → [outC, 1, kH, kW]
    /// </summary>
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
                attrs["axis"] = JsonSerializer.SerializeToElement(-1L);
                break;
            case 45: // STRIDED_SLICE
                // StridedSlice has begin_mask, end_mask, etc.
                attrs["begin_mask"] = JsonSerializer.SerializeToElement((long)fb.ReadFieldInt32(optOffset, 0, 0));
                attrs["end_mask"] = JsonSerializer.SerializeToElement((long)fb.ReadFieldInt32(optOffset, 1, 0));
                attrs["ellipsis_mask"] = JsonSerializer.SerializeToElement((long)fb.ReadFieldInt32(optOffset, 2, 0));
                attrs["new_axis_mask"] = JsonSerializer.SerializeToElement((long)fb.ReadFieldInt32(optOffset, 3, 0));
                attrs["shrink_axis_mask"] = JsonSerializer.SerializeToElement((long)fb.ReadFieldInt32(optOffset, 4, 0));
                break;
            case 49: // SPLIT
                attrs["num_outputs"] = JsonSerializer.SerializeToElement((long)fb.ReadFieldInt32(optOffset, 0, 2));
                break;
            case 36: // GATHER
                attrs["axis"] = JsonSerializer.SerializeToElement((long)fb.ReadFieldInt32(optOffset, 0, 0));
                break;
            case 56: // ARG_MAX
                attrs["axis"] = JsonSerializer.SerializeToElement(-1L); // default
                attrs["keepdims"] = JsonSerializer.SerializeToElement(0L);
                break;
            case 53: // CAST
                attrs["to"] = JsonSerializer.SerializeToElement(1L); // float32
                break;
            case 23: // RESIZE_BILINEAR
                attrs["mode"] = JsonSerializer.SerializeToElement("linear");
                break;
            case 95: // RESIZE_NEAREST_NEIGHBOR
                attrs["mode"] = JsonSerializer.SerializeToElement("nearest");
                break;
            case 39: // TRANSPOSE
                // perm comes from second input tensor
                if (op.Inputs.Length > 1 && op.Inputs[1] >= 0)
                {
                    var permTensor = sg.Tensors[op.Inputs[1]];
                    var permData = model.GetTensorData(permTensor);
                    if (permData != null)
                        attrs["perm"] = JsonSerializer.SerializeToElement(permData.Select(v => (long)v).ToArray());
                }
                break;
            case 96: // LEAKY_RELU
                // alpha from builtin options field 0
                {
                    float alpha = fb.ReadFieldFloat(optOffset, 0, 0.01f);
                    attrs["alpha"] = JsonSerializer.SerializeToElement((double)alpha);
                }
                break;
            case 20: // RELU_N1_TO_1
                attrs["min"] = JsonSerializer.SerializeToElement(-1.0);
                attrs["max"] = JsonSerializer.SerializeToElement(1.0);
                break;
            case 21: // RELU6
                attrs["min"] = JsonSerializer.SerializeToElement(0.0);
                attrs["max"] = JsonSerializer.SerializeToElement(6.0);
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
