using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Tensors;

namespace SpawnDev.ILGPU.ML.Operators;

/// <summary>
/// Additional ONNX operators: DepthToSpace, TopK, Sign.
/// These supplement the 63 operators #1 already built.
/// </summary>

public class DepthToSpaceOperator : IOnnxOperator, IDisposable
{
    private readonly Kernels.MissingElementWiseKernels _kernels;
    public string OpType => "DepthToSpace";
    public DepthToSpaceOperator(Accelerator accelerator) => _kernels = new(accelerator);
    /// <summary>The operator OWNS its kernels; the registry disposes operators that implement this.</summary>
    public void Dispose() => _kernels.Dispose();

    public int[][] InferOutputShapes(int[][] inputShapes, Dictionary<string, object> attributes)
    {
        var shape = inputShapes[0];
        int blockSize = 2;
        if (attributes.TryGetValue("blocksize", out var bs))
            blockSize = bs is long l ? (int)l : (int)bs;
        int outC = shape[1] / (blockSize * blockSize);
        return new[] { new[] { shape[0], outC, shape[2] * blockSize, shape[3] * blockSize } };
    }

    public void Execute(OnnxOpContext ctx)
    {
        var input = ctx.Inputs[0];
        var output = ctx.Outputs[0];
        int blockSize = ctx.GetInt("blocksize", 2);
        int inH = input.Shape[2];
        int inW = input.Shape[3];
        int outC = input.Shape[1] / (blockSize * blockSize);
        // ONNX mode: "DCR" (default) or "CRD"
        var modeStr = ctx.GetString("mode", "DCR");
        int mode = modeStr.Equals("CRD", StringComparison.OrdinalIgnoreCase) ? 1 : 0;
        _kernels.DepthToSpace(input.Data, output.Data, outC, inH, inW, blockSize, mode);
    }
}

/// <summary>
/// ONNX TopK (opset 1/10/11): K from input[1] (opset 10+) or the <c>k</c> attribute (opset 1); <c>axis</c> (default
/// -1), <c>largest</c> (default 1), <c>sorted</c> (default 1). The output is always sorted best-first with the lower
/// index first among equal values - the order ONNX requires when sorted=1 and a valid one when sorted=0. Indices are
/// stored as float like every integer tensor here (exact to 2^24).
/// </summary>
public class TopKOperator : IOnnxOperator, IDisposable
{
    private readonly Kernels.MissingElementWiseKernels _kernels;
    public string OpType => "TopK";
    public TopKOperator(Accelerator accelerator) => _kernels = new(accelerator);
    /// <summary>
    /// The operator OWNS its kernels, including the sort scratch (11 MB on RaCo-ALIKED k3072). Not being
    /// IDisposable, the registry could never release it: it outlived every session (2026-10-03, SpawnScene).
    /// </summary>
    public void Dispose() => _kernels.Dispose();

    static int Axis(int rank, Dictionary<string, object> attributes)
    {
        int axis = attributes.TryGetValue("axis", out var a) ? Convert.ToInt32(a) : -1;
        return axis < 0 ? axis + rank : axis;
    }

    /// <summary>K from the opset-1 <c>k</c> attribute; null when K is input[1] (opset 10+).</summary>
    static int? StaticK(Dictionary<string, object> attributes)
        => attributes.TryGetValue("k", out var kv) ? Convert.ToInt32(kv) : null;

    /// <summary>Output shape for a given K: the input shape with the axis dimension replaced by K. Shared with the
    /// executor's runtime override for a K that is only known at runtime.</summary>
    public static int[] OutputShape(int[] inputShape, Dictionary<string, object> attributes, int k)
    {
        var outShape = (int[])inputShape.Clone();
        outShape[Axis(inputShape.Length, attributes)] = k;
        return outShape;
    }

    public int[][] InferOutputShapes(int[][] inputShapes, Dictionary<string, object> attributes)
    {
        var shape = inputShapes[0];
        // K as input[1] is resolved by the executor (GraphExecutor.TopKRuntimeShapes); until then the axis length is
        // the upper bound (never undersized).
        int k = StaticK(attributes) ?? shape[Axis(shape.Length, attributes)];
        var outShape = OutputShape(shape, attributes, k);
        return new[] { outShape, (int[])outShape.Clone() };
    }

    public void Execute(OnnxOpContext ctx)
    {
        var input = ctx.Inputs[0];
        var outputValues = ctx.Outputs[0];
        int rank = input.Shape.Length;
        int axis = Axis(rank, ctx.Attributes);
        int k = outputValues.Shape[axis];
        var kVals = ctx.Inputs.Length > 1 ? ctx.TryGetInputValues(1) : null;
        if (kVals != null && kVals.Length > 0 && (int)kVals[0] != k)
            throw new InvalidOperationException($"TopK: K = {(int)kVals[0]} but the output was sized for {k} along axis {axis}");
        int axisLen = input.Shape[axis];
        if (k > axisLen) throw new InvalidOperationException($"TopK: K = {k} exceeds the axis length {axisLen}");
        int outer = 1, inner = 1;
        for (int d = 0; d < axis; d++) outer *= input.Shape[d];
        for (int d = axis + 1; d < rank; d++) inner *= input.Shape[d];
        bool largest = !ctx.Attributes.TryGetValue("largest", out var lg) || Convert.ToInt64(lg) != 0;
        // Pass indices output if present; kernel stores indices as float (avoids Wasm Int32Array issues)
        var idxView = ctx.Outputs.Length > 1 && ctx.Outputs[1] != null
            ? ctx.Outputs[1].Data
            : default(ArrayView1D<float, Stride1D.Dense>);
        if (k == 0 || outer * inner == 0) return;
        _kernels.TopK(input.Data, outputValues.Data, idxView, outer, axisLen, inner, k, largest);
    }
}

public class SignOperator : IOnnxOperator, IDisposable
{
    private readonly Kernels.MissingElementWiseKernels _kernels;
    public string OpType => "Sign";
    public SignOperator(Accelerator accelerator) => _kernels = new(accelerator);
    /// <summary>The operator OWNS its kernels; the registry disposes operators that implement this.</summary>
    public void Dispose() => _kernels.Dispose();
    public int[][] InferOutputShapes(int[][] inputShapes, Dictionary<string, object> attributes) => new[] { inputShapes[0] };
    public void Execute(OnnxOpContext ctx) => _kernels.Sign(ctx.Inputs[0].Data, ctx.Outputs[0].Data, ctx.Inputs[0].ElementCount);
}
