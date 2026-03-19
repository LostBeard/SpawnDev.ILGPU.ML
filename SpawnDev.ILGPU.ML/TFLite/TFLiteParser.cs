namespace SpawnDev.ILGPU.ML.TFLite;

/// <summary>
/// Zero-dependency TFLite model parser.
/// Reads the FlatBuffers binary format and extracts model structure + weights.
/// Produces a TFLiteModel that can be converted to ModelGraph for inference.
///
/// TFLite FlatBuffers layout (root = Model table):
///   Model { version, operator_codes[], subgraphs[], description, buffers[], metadata[] }
///   SubGraph { tensors[], inputs[], outputs[], operators[], name }
///   Tensor { shape[], type, buffer_index, name, quantization }
///   Operator { opcode_index, inputs[], outputs[], builtin_options_type, builtin_options }
///   OperatorCode { deprecated_builtin_code, custom_code, version, builtin_code }
///   Buffer { data[] }
/// </summary>
public static class TFLiteParser
{
    /// <summary>
    /// Parse a TFLite model from raw bytes.
    /// </summary>
    public static TFLiteModel Parse(byte[] data)
    {
        var fb = new FlatBufferReader(data);
        var model = new TFLiteModel();

        // Root table offset
        int rootOffset = fb.GetRootTableOffset();

        // Model fields (field indices from TFLite schema):
        // 0: version (int)
        // 1: operator_codes (vector of OperatorCode tables)
        // 2: subgraphs (vector of SubGraph tables)
        // 3: description (string)
        // 4: buffers (vector of Buffer tables)
        model.Version = fb.ReadFieldInt32(rootOffset, 0);
        model.Description = fb.ReadFieldString(rootOffset, 3);

        // Parse operator codes
        int opCodesVec = fb.ReadFieldOffset(rootOffset, 1);
        int opCodesCount = fb.VectorLength(opCodesVec);
        model.OperatorCodes = new TFLiteOperatorCode[opCodesCount];
        for (int i = 0; i < opCodesCount; i++)
        {
            int ocOffset = fb.VectorTableElement(opCodesVec, i);
            model.OperatorCodes[i] = ParseOperatorCode(fb, ocOffset);
        }

        // Parse buffers
        int buffersVec = fb.ReadFieldOffset(rootOffset, 4);
        int buffersCount = fb.VectorLength(buffersVec);
        model.Buffers = new TFLiteBuffer[buffersCount];
        for (int i = 0; i < buffersCount; i++)
        {
            int bufOffset = fb.VectorTableElement(buffersVec, i);
            // Buffer has one field: data (vector of bytes) at index 0
            int dataVec = fb.ReadFieldOffset(bufOffset, 0);
            model.Buffers[i] = new TFLiteBuffer
            {
                DataOffset = dataVec != 0 ? dataVec + 4 : 0, // skip length prefix
                DataLength = fb.VectorLength(dataVec)
            };
        }

        // Parse subgraphs (usually just one)
        int subgraphsVec = fb.ReadFieldOffset(rootOffset, 2);
        int subgraphsCount = fb.VectorLength(subgraphsVec);
        model.Subgraphs = new TFLiteSubGraph[subgraphsCount];
        for (int i = 0; i < subgraphsCount; i++)
        {
            int sgOffset = fb.VectorTableElement(subgraphsVec, i);
            model.Subgraphs[i] = ParseSubGraph(fb, sgOffset);
        }

        model.RawData = data;
        return model;
    }

    private static TFLiteOperatorCode ParseOperatorCode(FlatBufferReader fb, int offset)
    {
        // OperatorCode fields:
        // 0: deprecated_builtin_code (byte) — old field, use builtin_code (field 4) if available
        // 1: custom_code (string)
        // 2: version (int)
        // 4: builtin_code (int32) — new field since schema v3a
        var oc = new TFLiteOperatorCode();
        byte deprecatedCode = fb.ReadFieldByte(offset, 0);
        oc.CustomCode = fb.ReadFieldString(offset, 1);
        oc.Version = fb.ReadFieldInt32(offset, 2, 1);

        // Try new builtin_code field first (field index 4), fall back to deprecated
        int builtinCode = fb.ReadFieldInt32(offset, 4, -1);
        oc.BuiltinCode = builtinCode >= 0 ? builtinCode : deprecatedCode;

        return oc;
    }

    private static TFLiteSubGraph ParseSubGraph(FlatBufferReader fb, int offset)
    {
        var sg = new TFLiteSubGraph();

        // SubGraph fields:
        // 0: tensors (vector of Tensor tables)
        // 1: inputs (vector of int)
        // 2: outputs (vector of int)
        // 3: operators (vector of Operator tables)
        // 4: name (string)
        sg.Name = fb.ReadFieldString(offset, 4);

        // Parse tensors
        int tensorsVec = fb.ReadFieldOffset(offset, 0);
        int tensorsCount = fb.VectorLength(tensorsVec);
        sg.Tensors = new TFLiteTensor[tensorsCount];
        for (int i = 0; i < tensorsCount; i++)
        {
            int tOffset = fb.VectorTableElement(tensorsVec, i);
            sg.Tensors[i] = ParseTensor(fb, tOffset);
        }

        // Parse inputs (vector of int32)
        int inputsVec = fb.ReadFieldOffset(offset, 1);
        int inputsCount = fb.VectorLength(inputsVec);
        sg.Inputs = new int[inputsCount];
        for (int i = 0; i < inputsCount; i++)
            sg.Inputs[i] = fb.VectorInt32(inputsVec, i);

        // Parse outputs (vector of int32)
        int outputsVec = fb.ReadFieldOffset(offset, 2);
        int outputsCount = fb.VectorLength(outputsVec);
        sg.Outputs = new int[outputsCount];
        for (int i = 0; i < outputsCount; i++)
            sg.Outputs[i] = fb.VectorInt32(outputsVec, i);

        // Parse operators
        int opsVec = fb.ReadFieldOffset(offset, 3);
        int opsCount = fb.VectorLength(opsVec);
        sg.Operators = new TFLiteOperator[opsCount];
        for (int i = 0; i < opsCount; i++)
        {
            int opOffset = fb.VectorTableElement(opsVec, i);
            sg.Operators[i] = ParseOperator(fb, opOffset);
        }

        return sg;
    }

    private static TFLiteTensor ParseTensor(FlatBufferReader fb, int offset)
    {
        // Tensor fields:
        // 0: shape (vector of int)
        // 1: type (TensorType enum: byte)
        // 2: buffer (uint — buffer index)
        // 3: name (string)
        // 4: quantization (QuantizationParameters table)
        var t = new TFLiteTensor();
        t.Name = fb.ReadFieldString(offset, 3);
        t.Type = (TFLiteTensorType)fb.ReadFieldByte(offset, 1);
        t.BufferIndex = fb.ReadFieldInt32(offset, 2);

        // Shape
        int shapeVec = fb.ReadFieldOffset(offset, 0);
        int shapeLen = fb.VectorLength(shapeVec);
        t.Shape = new int[shapeLen];
        for (int i = 0; i < shapeLen; i++)
            t.Shape[i] = fb.VectorInt32(shapeVec, i);

        // Quantization (optional)
        int quantOffset = fb.ReadFieldOffset(offset, 4);
        if (quantOffset != 0)
        {
            t.Quantization = ParseQuantization(fb, quantOffset);
        }

        return t;
    }

    private static TFLiteQuantization ParseQuantization(FlatBufferReader fb, int offset)
    {
        // QuantizationParameters fields:
        // 0: min (vector of float)
        // 1: max (vector of float)
        // 2: scale (vector of float)
        // 3: zero_point (vector of long)
        var q = new TFLiteQuantization();

        int scaleVec = fb.ReadFieldOffset(offset, 2);
        int scaleLen = fb.VectorLength(scaleVec);
        if (scaleLen > 0)
        {
            q.Scale = new float[scaleLen];
            for (int i = 0; i < scaleLen; i++)
                q.Scale[i] = fb.ReadFloat32(fb.VectorScalarOffset(scaleVec, i, 4));
        }

        int zpVec = fb.ReadFieldOffset(offset, 3);
        int zpLen = fb.VectorLength(zpVec);
        if (zpLen > 0)
        {
            q.ZeroPoint = new long[zpLen];
            for (int i = 0; i < zpLen; i++)
                q.ZeroPoint[i] = fb.ReadInt64(fb.VectorScalarOffset(zpVec, i, 8));
        }

        return q;
    }

    private static TFLiteOperator ParseOperator(FlatBufferReader fb, int offset)
    {
        // Operator fields:
        // 0: opcode_index (uint)
        // 1: inputs (vector of int)
        // 2: outputs (vector of int)
        // 3: builtin_options_type (BuiltinOptions enum: byte)
        // 4: builtin_options (union table)
        var op = new TFLiteOperator();
        op.OpcodeIndex = fb.ReadFieldInt32(offset, 0);
        op.BuiltinOptionsType = fb.ReadFieldByte(offset, 3);
        op.BuiltinOptionsOffset = fb.ReadFieldOffset(offset, 4);

        // Inputs
        int inputsVec = fb.ReadFieldOffset(offset, 1);
        int inputsLen = fb.VectorLength(inputsVec);
        op.Inputs = new int[inputsLen];
        for (int i = 0; i < inputsLen; i++)
            op.Inputs[i] = fb.VectorInt32(inputsVec, i);

        // Outputs
        int outputsVec = fb.ReadFieldOffset(offset, 2);
        int outputsLen = fb.VectorLength(outputsVec);
        op.Outputs = new int[outputsLen];
        for (int i = 0; i < outputsLen; i++)
            op.Outputs[i] = fb.VectorInt32(outputsVec, i);

        return op;
    }

    /// <summary>
    /// Get a quick summary of a TFLite model (for logging/display).
    /// </summary>
    public static string GetSummary(TFLiteModel model)
    {
        if (model.Subgraphs.Length == 0) return "Empty model";
        var sg = model.Subgraphs[0];
        var opNames = sg.Operators
            .Select(o => model.GetOperatorName(o.OpcodeIndex))
            .Distinct().OrderBy(n => n);
        return $"TFLite v{model.Version}: {sg.Operators.Length} ops, {sg.Tensors.Length} tensors, " +
               $"operators: {string.Join(", ", opNames)}";
    }
}
