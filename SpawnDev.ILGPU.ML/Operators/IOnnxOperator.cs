using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Tensors;

namespace SpawnDev.ILGPU.ML.Operators;

/// <summary>
/// Interface for ONNX operator implementations.
/// Each operator handles one ONNX op type (e.g., "MatMul", "Relu", "Conv").
/// </summary>
public interface IOnnxOperator
{
    /// <summary>ONNX op type string (e.g., "MatMul", "Relu", "Conv").</summary>
    string OpType { get; }

    /// <summary>
    /// Infer output shapes from input shapes and attributes.
    /// Called during graph compilation (before execution).
    /// </summary>
    int[][] InferOutputShapes(int[][] inputShapes, Dictionary<string, object> attributes);

    /// <summary>
    /// Execute the operator on GPU tensors.
    /// Inputs and outputs are pre-allocated by the graph executor.
    /// </summary>
    void Execute(OnnxOpContext ctx);

    /// <summary>
    /// Async execution path used by <c>GraphExecutor.RunAsync</c>. The default simply runs the
    /// synchronous <see cref="Execute"/> — correct for the vast majority of operators, which
    /// only dispatch GPU kernels (queuing work, never reading it back). Operators that need a
    /// runtime GPU-&gt;CPU readback mid-execution (control flow such as Loop/If/Scan reading a
    /// condition, or Einsum reading dynamic inputs) MUST override this to <c>await</c> the
    /// browser-safe async readback (<c>CopyToHostAsync</c> / <c>SynchronizeAsync</c>) instead
    /// of the synchronous <c>CopyToCPU</c>, which throws on WebGPU/WebGL/Wasm. This is what
    /// gives those operators feature parity on the browser backends.
    /// </summary>
    /// <param name="ctx">The execution context.</param>
    Task ExecuteAsync(OnnxOpContext ctx)
    {
        Execute(ctx);
        return Task.CompletedTask;
    }
}

/// <summary>
/// Execution context passed to each operator. Contains inputs, outputs,
/// attributes, and the buffer pool for temporary allocations.
/// </summary>
public class OnnxOpContext
{
    public required Tensor[] Inputs { get; init; }
    public required Tensor[] Outputs { get; init; }
    public required Dictionary<string, object> Attributes { get; init; }
    public required BufferPool Pool { get; init; }
    /// <summary>Data layout format (NCHW for ONNX, NHWC for TFLite).</summary>
    public DataFormat Format { get; init; } = DataFormat.NCHW;
    /// <summary>Input tensor names (for looking up pre-read constant data).</summary>
    public string[] InputNames { get; init; } = Array.Empty<string>();
    /// <summary>Pre-read constant data from small tensors (avoids GPU→CPU readback at runtime).
    /// Maps tensor name → float[] values. Populated during session creation.</summary>
    public Dictionary<string, float[]>? ConstantValues { get; init; }

    /// <summary>Quantized weight buffers (Q4_0, Q8_0, etc.) stored as raw bytes on GPU.
    /// When a weight tensor name appears here, operators should use fused dequantization
    /// kernels instead of regular float operations. Maps tensor name → byte ArrayView.</summary>
    public Dictionary<string, ArrayView1D<byte, Stride1D.Dense>>? QuantizedWeights { get; init; }

    /// <summary>Operator registry for subgraph execution (If/Loop/Scan).
    /// Allows control flow operators to compile and execute embedded ONNX subgraphs.</summary>
    public OperatorRegistry? Registry { get; init; }

    /// <summary>
    /// Every tensor live in the ENCLOSING graph, for control-flow operators only.
    /// </summary>
    /// <remarks>
    /// ⚠️ ONNX subgraphs may read values from the enclosing scope WITHOUT declaring them as inputs, and an
    /// <c>If</c> node has exactly one input of its own - the condition. So a branch body that references a
    /// parent tensor could neither be compiled (no shape) nor executed (no tensor): it simply was not
    /// passed anything.
    ///
    /// That is invisible while every branch a model actually takes is self-contained. ZipVoice's decoder
    /// made it visible: an utterance longer than its precomputed [1999, 48] positional table takes a
    /// DIFFERENT branch - 156 nodes that read the parent's Gather output - and compiling it crashed with
    /// "shapes=(?; [1])". Short utterances take the other branch, a single Constant, which is why every
    /// test passed until the first realistic chat reply.
    ///
    /// ⚠️ Read ONLY for outer-scope capture. It is the executor's live tensor map, not a general-purpose
    /// back door: an operator reaching into it for anything else is reading state it has no contract with.
    /// </remarks>
    public IReadOnlyDictionary<string, Tensor>? ScopeTensors { get; init; }

    /// <summary>Names of tensors whose ONNX-declared dtype is integer (INT8/16/32/64, UINT8/16,
    /// BOOL). Populated at session-init by walking initializers (with their TensorProto.DataType),
    /// Cast nodes (with their `to` attribute), and known integer-producing ops (ArgMax, Shape,
    /// Size, NonZero, TopK indices). Lets operators honour int-vs-float semantic differences
    /// even though all storage in this pipeline is float32 -- the canonical case is ONNX Div,
    /// which is FLOOR division on integer dtypes but float division on FP dtypes. TF's
    /// `tf.floordiv` exports as Cast(int)+Div(int,int); without this set our Div produced
    /// 18.479 instead of 18 for the (argmax mod 48) keypoint X-coord decode in MoveNet.</summary>
    public HashSet<string>? IntegerTensorNames { get; init; }

    /// <summary>True if all of this op's inputs are declared integer-typed in the ONNX graph.
    /// Returns false when IntegerTensorNames is null or any input is missing / non-integer.</summary>
    public bool AllInputsAreInteger()
    {
        if (IntegerTensorNames == null || InputNames.Length == 0) return false;
        foreach (var name in InputNames)
        {
            if (string.IsNullOrEmpty(name)) return false;
            if (!IntegerTensorNames.Contains(name)) return false;
        }
        return true;
    }

    /// <summary>Try to get pre-read float values for an input tensor (by index).
    /// Returns null if not available (tensor is dynamic, not pre-read).</summary>
    public float[]? TryGetInputValues(int inputIndex)
    {
        if (ConstantValues == null || inputIndex >= InputNames.Length) return null;
        var name = InputNames[inputIndex];
        if (string.IsNullOrEmpty(name)) return null;
        return ConstantValues.TryGetValue(name, out var vals) ? vals : null;
    }

    // ── Typed attribute accessors ──

    public int GetInt(string name, int defaultValue = 0)
        => Attributes.TryGetValue(name, out var v) ? Convert.ToInt32(v) : defaultValue;

    public float GetFloat(string name, float defaultValue = 0f)
        => Attributes.TryGetValue(name, out var v) ? Convert.ToSingle(v) : defaultValue;

    public long[] GetLongs(string name, long[]? defaultValue = null)
        => Attributes.TryGetValue(name, out var v) ? (long[])v : defaultValue ?? Array.Empty<long>();

    public int[] GetInts(string name, int[]? defaultValue = null)
        => Attributes.TryGetValue(name, out var v) && v is int[] ia ? ia
         : Attributes.TryGetValue(name, out var v2) && v2 is long[] la ? la.Select(x => (int)x).ToArray()
         : defaultValue ?? Array.Empty<int>();

    public string GetString(string name, string defaultValue = "")
        => Attributes.TryGetValue(name, out var v) ? v.ToString()! : defaultValue;
}
