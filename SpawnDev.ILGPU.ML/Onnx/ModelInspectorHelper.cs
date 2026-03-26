namespace SpawnDev.ILGPU.ML.Onnx;

/// <summary>
/// Helper for the Model Inspector demo page.
/// Drop any .onnx file → see architecture, operators, shapes, weights.
/// A developer tool that showcases our native ONNX parser.
/// </summary>
public static class ModelInspectorHelper
{
    /// <summary>
    /// Inspect a model file and return a structured summary.
    /// Auto-detects format (ONNX or TFLite) from magic bytes.
    /// </summary>
    public static InspectionResult Inspect(byte[] modelBytes)
    {
        var format = InferenceSession.DetectModelFormat(modelBytes);
        return format switch
        {
            ModelFormat.TFLite => InspectTFLite(modelBytes),
            ModelFormat.GGUF => InspectGGUF(modelBytes),
            ModelFormat.SafeTensors => InspectSafeTensors(modelBytes),
            ModelFormat.SPZ => InspectSPZ(modelBytes),
            ModelFormat.PLY => InspectPLY(modelBytes),
            ModelFormat.GLTF => InspectGLTF(modelBytes),
            ModelFormat.OBJ => InspectOBJ(modelBytes),
            _ => InspectOnnx(modelBytes),
        };
    }

    /// <summary>Inspect an ONNX model.</summary>
    public static InspectionResult InspectOnnx(byte[] onnxBytes)
    {
        var model = OnnxParser.Parse(onnxBytes);
        var graph = model.Graph;

        var opset = model.OpsetImports.FirstOrDefault(o => o.Domain == "")?.Version ?? 0;

        // Operator usage
        var opCounts = graph.Nodes
            .GroupBy(n => n.OpType)
            .OrderByDescending(g => g.Count())
            .Select(g => new OpUsage { OpType = g.Key, Count = g.Count() })
            .ToArray();

        // Weight statistics
        long totalParams = 0;
        long totalBytes = 0;
        var largestWeights = new List<WeightInfo>();

        foreach (var init in graph.Initializers)
        {
            long elements = init.ElementCount;
            long bytes = init.RawData?.Length ?? (elements * DataTypeSize(init.DataType));
            totalParams += elements;
            totalBytes += bytes;

            largestWeights.Add(new WeightInfo
            {
                Name = init.Name,
                Shape = init.Dims.Select(d => (int)d).ToArray(),
                Elements = elements,
                SizeBytes = bytes,
                DataType = DataTypeName(init.DataType),
            });
        }

        largestWeights.Sort((a, b) => b.SizeBytes.CompareTo(a.SizeBytes));

        // Inputs and outputs
        var inputs = graph.Inputs
            .Where(i => !graph.Initializers.Any(init => init.Name == i.Name))
            .Select(i => new TensorInfo
            {
                Name = i.Name,
                Shape = i.Shape.Select(d => d.ToString()).ToArray(),
                DataType = DataTypeName(i.ElemType),
            }).ToArray();

        var outputs = graph.Outputs.Select(o => new TensorInfo
        {
            Name = o.Name,
            Shape = o.Shape.Select(d => d.ToString()).ToArray(),
            DataType = DataTypeName(o.ElemType),
        }).ToArray();

        return new InspectionResult
        {
            GraphName = graph.Name,
            ProducerName = model.ProducerName,
            ProducerVersion = model.ProducerVersion,
            IrVersion = model.IrVersion,
            OpsetVersion = opset,
            NodeCount = graph.Nodes.Count,
            InitializerCount = graph.Initializers.Count,
            TotalParameters = totalParams,
            TotalWeightBytes = totalBytes,
            Operators = opCounts,
            Inputs = inputs,
            Outputs = outputs,
            LargestWeights = largestWeights.Take(20).ToArray(),
            FileSizeBytes = onnxBytes.Length,
        };
    }

    /// <summary>Inspect a TFLite model.</summary>
    public static InspectionResult InspectTFLite(byte[] tfliteBytes)
    {
        var model = TFLite.TFLiteParser.Parse(tfliteBytes);
        if (model.Subgraphs.Length == 0)
            return new InspectionResult { GraphName = "Empty TFLite model", FileSizeBytes = tfliteBytes.Length };

        var sg = model.Subgraphs[0];

        // Operator usage
        var opCounts = sg.Operators
            .Select(o => model.GetOperatorName(o.OpcodeIndex))
            .GroupBy(n => n)
            .OrderByDescending(g => g.Count())
            .Select(g => new OpUsage { OpType = g.Key, Count = g.Count() })
            .ToArray();

        // Weight statistics
        long totalParams = 0;
        long totalBytes = 0;
        var largestWeights = new List<WeightInfo>();
        for (int i = 0; i < sg.Tensors.Length; i++)
        {
            var tensor = sg.Tensors[i];
            var buffer = model.Buffers[tensor.BufferIndex];
            if (buffer.DataLength == 0) continue;

            long elems = 1;
            foreach (var d in tensor.Shape) elems *= d;
            totalParams += elems;
            totalBytes += buffer.DataLength;

            largestWeights.Add(new WeightInfo
            {
                Name = tensor.Name,
                Shape = tensor.Shape,
                Elements = elems,
                SizeBytes = buffer.DataLength,
                DataType = tensor.Type.ToString()
            });
        }
        largestWeights.Sort((a, b) => b.SizeBytes.CompareTo(a.SizeBytes));

        // Inputs/outputs
        var inputs = sg.Inputs.Where(i => model.Buffers[sg.Tensors[i].BufferIndex].DataLength == 0)
            .Select(i => new TensorInfo
            {
                Name = sg.Tensors[i].Name,
                Shape = sg.Tensors[i].Shape.Select(d => d.ToString()).ToArray(),
                DataType = sg.Tensors[i].Type.ToString()
            }).ToArray();

        var outputs = sg.Outputs.Select(i => new TensorInfo
        {
            Name = sg.Tensors[i].Name,
            Shape = sg.Tensors[i].Shape.Select(d => d.ToString()).ToArray(),
            DataType = sg.Tensors[i].Type.ToString()
        }).ToArray();

        return new InspectionResult
        {
            GraphName = model.Description.Length > 0 ? model.Description : "TFLite Model",
            ProducerName = "TensorFlow Lite",
            ProducerVersion = $"v{model.Version}",
            IrVersion = model.Version,
            NodeCount = sg.Operators.Length,
            InitializerCount = largestWeights.Count,
            TotalParameters = totalParams,
            TotalWeightBytes = totalBytes,
            Operators = opCounts,
            Inputs = inputs,
            Outputs = outputs,
            LargestWeights = largestWeights.Take(20).ToArray(),
            FileSizeBytes = tfliteBytes.Length,
        };
    }

    /// <summary>Inspect a GGUF model (LLM weights).</summary>
    public static InspectionResult InspectGGUF(byte[] ggufBytes)
    {
        var model = GGUF.GGUFParser.Parse(ggufBytes);

        // Count tensor types as "operators"
        var typeCounts = model.Tensors
            .GroupBy(t => t.Type.ToString())
            .OrderByDescending(g => g.Count())
            .Select(g => new OpUsage { OpType = g.Key, Count = g.Count() })
            .ToArray();

        long totalParams = 0;
        long totalBytes = 0;
        var largestWeights = new List<WeightInfo>();

        foreach (var tensor in model.Tensors)
        {
            long elems = model.GetTensorElementCount(tensor);
            long bytes = GGUF.GGMLTypes.TypeSize(tensor.Type, elems);
            totalParams += elems;
            totalBytes += bytes;

            largestWeights.Add(new WeightInfo
            {
                Name = tensor.Name,
                Shape = tensor.Shape,
                Elements = elems,
                SizeBytes = bytes,
                DataType = tensor.Type.ToString()
            });
        }
        largestWeights.Sort((a, b) => b.SizeBytes.CompareTo(a.SizeBytes));

        return new InspectionResult
        {
            GraphName = $"{model.Name} ({model.Architecture})",
            ProducerName = "GGUF / llama.cpp",
            ProducerVersion = $"v{model.Version}",
            IrVersion = model.Version,
            OpsetVersion = model.ContextLength,
            NodeCount = model.Tensors.Length,
            InitializerCount = model.Tensors.Length,
            TotalParameters = totalParams,
            TotalWeightBytes = totalBytes,
            Operators = typeCounts,
            Inputs = new[] { new TensorInfo
            {
                Name = "Architecture",
                Shape = new[] { model.Architecture, $"{model.BlockCount} layers", $"{model.EmbeddingLength}d", $"{model.AttentionHeadCount} heads" },
                DataType = $"ctx={model.ContextLength}"
            }},
            Outputs = new[] { new TensorInfo
            {
                Name = "Vocab",
                Shape = new[] { $"{model.VocabSize} tokens" },
                DataType = "text"
            }},
            LargestWeights = largestWeights.Take(20).ToArray(),
            FileSizeBytes = ggufBytes.Length,
        };
    }

    /// <summary>
    /// Check which operators the model needs that we support vs don't support.
    /// <summary>Inspect a SafeTensors file (weights only, no graph).</summary>
    public static InspectionResult InspectSafeTensors(byte[] stBytes)
    {
        var file = SafeTensors.SafeTensorsParser.Parse(stBytes);

        long totalParams = 0;
        long totalBytes = 0;
        var dtypeCounts = file.Tensors
            .GroupBy(t => t.DType)
            .OrderByDescending(g => g.Count())
            .Select(g => new OpUsage { OpType = g.Key, Count = g.Count() })
            .ToArray();

        var largestWeights = new List<WeightInfo>();
        foreach (var tensor in file.Tensors)
        {
            long elems = tensor.Shape.Aggregate(1L, (a, b) => a * b);
            totalParams += elems;
            totalBytes += tensor.DataLength;
            largestWeights.Add(new WeightInfo
            {
                Name = tensor.Name,
                Shape = tensor.Shape,
                Elements = elems,
                SizeBytes = tensor.DataLength,
                DataType = tensor.DType
            });
        }
        largestWeights.Sort((a, b) => b.SizeBytes.CompareTo(a.SizeBytes));

        return new InspectionResult
        {
            GraphName = "SafeTensors (weights only)",
            ProducerName = "HuggingFace SafeTensors",
            NodeCount = 0, // no graph
            InitializerCount = file.Tensors.Length,
            TotalParameters = totalParams,
            TotalWeightBytes = totalBytes,
            Operators = dtypeCounts,
            Inputs = Array.Empty<TensorInfo>(),
            Outputs = Array.Empty<TensorInfo>(),
            LargestWeights = largestWeights.Take(20).ToArray(),
            FileSizeBytes = stBytes.Length,
        };
    }

    // ═══════════════════════════════════════════════════════════
    //  3D Format Inspectors
    // ═══════════════════════════════════════════════════════════

    public static InspectionResult InspectSPZ(byte[] spzBytes)
    {
        var cloud = Formats.SPZParser.Parse(spzBytes);
        return new InspectionResult
        {
            GraphName = $"SPZ Gaussian Splat ({cloud.NumPoints:N0} points)",
            ProducerName = $"SPZ v{cloud.Version}",
            NodeCount = 0,
            InitializerCount = cloud.NumPoints,
            TotalParameters = cloud.NumPoints * 14, // pos(3)+alpha(1)+color(3)+scale(3)+rot(4)
            TotalWeightBytes = spzBytes.Length,
            Inputs = new[] { new TensorInfo { Name = "gaussians", Shape = new[] { cloud.NumPoints.ToString(), "14" } } },
            Outputs = Array.Empty<TensorInfo>(),
            Operators = Array.Empty<OpUsage>(),
            LargestWeights = Array.Empty<WeightInfo>(),
        };
    }

    public static InspectionResult InspectPLY(byte[] plyBytes)
    {
        var ply = Formats.PLYParser.Parse(plyBytes);
        bool isGaussian = ply.Gaussians != null;
        return new InspectionResult
        {
            GraphName = isGaussian ? $"PLY Gaussian Splat ({ply.VertexCount:N0} points)" : $"PLY Mesh ({ply.VertexCount:N0} vertices, {ply.FaceCount} faces)",
            ProducerName = $"PLY {ply.Format}",
            NodeCount = 0,
            InitializerCount = ply.VertexCount,
            TotalParameters = ply.VertexCount * ply.Properties.Length,
            TotalWeightBytes = plyBytes.Length,
            Inputs = new[] { new TensorInfo { Name = "vertices", Shape = new[] { ply.VertexCount.ToString(), ply.Properties.Length.ToString() } } },
            Outputs = Array.Empty<TensorInfo>(),
            Operators = ply.Properties.Select(p => new OpUsage { OpType = p, Count = ply.VertexCount }).ToArray(),
            LargestWeights = Array.Empty<WeightInfo>(),
        };
    }

    public static InspectionResult InspectGLTF(byte[] glbBytes)
    {
        var mesh = Formats.GLTFLoader.LoadGLB(glbBytes);
        return new InspectionResult
        {
            GraphName = $"glTF Mesh ({mesh.VertexCount:N0} vertices, {mesh.TriangleCount:N0} triangles)",
            ProducerName = "glTF 2.0",
            NodeCount = mesh.TriangleCount,
            InitializerCount = mesh.VertexCount,
            TotalParameters = mesh.VertexCount * 3,
            TotalWeightBytes = glbBytes.Length,
            Inputs = new[] { new TensorInfo { Name = "vertices", Shape = new[] { mesh.VertexCount.ToString(), "3" } } },
            Outputs = new[] { new TensorInfo { Name = "triangles", Shape = new[] { mesh.TriangleCount.ToString(), "3" } } },
            Operators = Array.Empty<OpUsage>(),
            LargestWeights = Array.Empty<WeightInfo>(),
        };
    }

    public static InspectionResult InspectOBJ(byte[] objBytes)
    {
        var mesh = Formats.OBJExporter.Load(objBytes);
        return new InspectionResult
        {
            GraphName = $"OBJ Mesh ({mesh.VertexCount:N0} vertices, {mesh.TriangleCount:N0} triangles)",
            ProducerName = "Wavefront OBJ",
            NodeCount = mesh.TriangleCount,
            InitializerCount = mesh.VertexCount,
            TotalParameters = mesh.VertexCount * 3,
            TotalWeightBytes = objBytes.Length,
            Inputs = new[] { new TensorInfo { Name = "vertices", Shape = new[] { mesh.VertexCount.ToString(), "3" } } },
            Outputs = new[] { new TensorInfo { Name = "triangles", Shape = new[] { mesh.TriangleCount.ToString(), "3" } } },
            Operators = Array.Empty<OpUsage>(),
            LargestWeights = Array.Empty<WeightInfo>(),
        };
    }

    /// Answers "Can I run this model?" instantly.
    /// </summary>
    public static CompatibilityResult CheckCompatibility(byte[] onnxBytes, Operators.OperatorRegistry? registry = null)
    {
        var model = OnnxParser.Parse(onnxBytes);
        var graph = model.Graph;

        var opsUsed = graph.Nodes.Select(n => n.OpType).Distinct().OrderBy(o => o).ToArray();

        // Known supported operators (hardcoded fallback if no registry provided)
        var knownSupported = new HashSet<string>
        {
            "Abs", "Add", "ArgMax", "AveragePool", "BatchNormalization", "Cast", "Ceil", "Clip",
            "Concat", "Constant", "ConstantOfShape", "Conv", "ConvTranspose",
            "DepthToSpace", "Div", "Dropout", "Equal", "Erf", "Exp", "Expand",
            "Flatten", "Floor", "Gather", "GatherND", "Gelu", "Gemm", "GlobalAveragePool",
            "Greater", "HardSigmoid", "HardSwish", "Identity", "InstanceNormalization",
            "LayerNormalization", "LeakyRelu", "Less", "Log", "MatMul", "Max", "MaxPool",
            "Min", "Mul", "Neg", "Not", "Pad", "Pow", "Range", "Reciprocal",
            "ReduceMax", "ReduceMean", "ReduceMin", "ReduceSum",
            "Relu", "Reshape", "Resize", "Shape", "Sigmoid", "Sign", "SiLU",
            "Slice", "Softmax", "Split", "Sqrt", "Squeeze", "Sub",
            "Tanh", "TopK", "Transpose", "Unsqueeze", "Upsample", "Where",
        };

        // If registry provided, use its actual supported ops
        HashSet<string> supportedOps;
        if (registry != null)
        {
            supportedOps = new HashSet<string>();
            foreach (var op in opsUsed)
            {
                if (registry.IsSupported(op))
                    supportedOps.Add(op);
            }
        }
        else
        {
            supportedOps = knownSupported;
        }

        var supported = opsUsed.Where(o => supportedOps.Contains(o)).ToArray();
        var unsupported = opsUsed.Where(o => !supportedOps.Contains(o)).ToArray();

        return new CompatibilityResult
        {
            TotalOpsUsed = opsUsed.Length,
            SupportedOps = supported,
            UnsupportedOps = unsupported,
            IsFullySupported = unsupported.Length == 0,
            CompatibilityPercent = opsUsed.Length > 0
                ? (float)supported.Length / opsUsed.Length * 100
                : 100,
        };
    }

    private static string DataTypeName(int dt) => dt switch
    {
        0 => "undefined", 1 => "float32", 2 => "uint8", 3 => "int8",
        4 => "uint16", 5 => "int16", 6 => "int32", 7 => "int64",
        8 => "string", 9 => "bool", 10 => "float16", 11 => "float64",
        12 => "uint32", 13 => "uint64", 16 => "bfloat16",
        _ => $"type_{dt}",
    };

    private static long DataTypeSize(int dt) => dt switch
    {
        1 => 4, 2 => 1, 3 => 1, 4 => 2, 5 => 2, 6 => 4, 7 => 8,
        9 => 1, 10 => 2, 11 => 8, 12 => 4, 13 => 8, 16 => 2,
        _ => 4,
    };
}

public class InspectionResult
{
    public string GraphName { get; set; } = "";
    public string ProducerName { get; set; } = "";
    public string ProducerVersion { get; set; } = "";
    public long IrVersion { get; set; }
    public long OpsetVersion { get; set; }
    public int NodeCount { get; set; }
    public int InitializerCount { get; set; }
    public long TotalParameters { get; set; }
    public long TotalWeightBytes { get; set; }
    public long FileSizeBytes { get; set; }
    public OpUsage[] Operators { get; set; } = Array.Empty<OpUsage>();
    public TensorInfo[] Inputs { get; set; } = Array.Empty<TensorInfo>();
    public TensorInfo[] Outputs { get; set; } = Array.Empty<TensorInfo>();
    public WeightInfo[] LargestWeights { get; set; } = Array.Empty<WeightInfo>();

    public string TotalParametersFormatted => TotalParameters switch
    {
        >= 1_000_000_000 => $"{TotalParameters / 1_000_000_000.0:F1}B",
        >= 1_000_000 => $"{TotalParameters / 1_000_000.0:F1}M",
        >= 1_000 => $"{TotalParameters / 1_000.0:F1}K",
        _ => TotalParameters.ToString(),
    };

    public string TotalWeightMB => $"{TotalWeightBytes / 1024.0 / 1024.0:F1} MB";
    public string FileSizeMB => $"{FileSizeBytes / 1024.0 / 1024.0:F1} MB";
}

public class OpUsage
{
    public string OpType { get; set; } = "";
    public int Count { get; set; }
}

public class TensorInfo
{
    public string Name { get; set; } = "";
    public string[] Shape { get; set; } = Array.Empty<string>();
    public string DataType { get; set; } = "";
    public string ShapeStr => $"[{string.Join(", ", Shape)}]";
}

public class WeightInfo
{
    public string Name { get; set; } = "";
    public int[] Shape { get; set; } = Array.Empty<int>();
    public long Elements { get; set; }
    public long SizeBytes { get; set; }
    public string DataType { get; set; } = "";
    public string ShapeStr => $"[{string.Join(", ", Shape)}]";
    public string SizeFormatted => SizeBytes switch
    {
        >= 1_048_576 => $"{SizeBytes / 1048576.0:F1} MB",
        >= 1024 => $"{SizeBytes / 1024.0:F1} KB",
        _ => $"{SizeBytes} B",
    };
}

/// <summary>Result of checking model compatibility with our engine.</summary>
public class CompatibilityResult
{
    public int TotalOpsUsed { get; set; }
    public string[] SupportedOps { get; set; } = Array.Empty<string>();
    public string[] UnsupportedOps { get; set; } = Array.Empty<string>();
    public bool IsFullySupported { get; set; }
    public float CompatibilityPercent { get; set; }

    public string Summary => IsFullySupported
        ? $"Fully compatible ({TotalOpsUsed} operators supported)"
        : $"{CompatibilityPercent:F0}% compatible ({SupportedOps.Length}/{TotalOpsUsed} operators). Missing: {string.Join(", ", UnsupportedOps)}";
}
