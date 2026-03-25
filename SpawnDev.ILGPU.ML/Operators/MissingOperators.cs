using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Tensors;

namespace SpawnDev.ILGPU.ML.Operators;

/// <summary>
/// Additional ONNX operators: DepthToSpace, TopK, Sign.
/// These supplement the 63 operators #1 already built.
/// </summary>

public class DepthToSpaceOperator : IOnnxOperator
{
    private readonly Kernels.MissingElementWiseKernels _kernels;
    public string OpType => "DepthToSpace";
    public DepthToSpaceOperator(Accelerator accelerator) => _kernels = new(accelerator);

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

public class TopKOperator : IOnnxOperator
{
    private readonly Kernels.MissingElementWiseKernels _kernels;
    public string OpType => "TopK";
    public TopKOperator(Accelerator accelerator) => _kernels = new(accelerator);

    public int[][] InferOutputShapes(int[][] inputShapes, Dictionary<string, object> attributes)
    {
        var shape = inputShapes[0];
        int k = 1;
        if (attributes.TryGetValue("k", out var kv))
            k = kv is long l ? (int)l : (int)kv;
        var outShape = (int[])shape.Clone();
        outShape[^1] = k;
        return new[] { outShape, outShape };
    }

    public void Execute(OnnxOpContext ctx)
    {
        var input = ctx.Inputs[0];
        var outputValues = ctx.Outputs[0];
        int lastDim = input.Shape[^1];
        int rows = input.ElementCount / lastDim;
        int k = outputValues.Shape[^1];
        _kernels.TopK(input.Data, outputValues.Data, default, rows, lastDim, k);
    }
}

public class SignOperator : IOnnxOperator
{
    private readonly Kernels.MissingElementWiseKernels _kernels;
    public string OpType => "Sign";
    public SignOperator(Accelerator accelerator) => _kernels = new(accelerator);
    public int[][] InferOutputShapes(int[][] inputShapes, Dictionary<string, object> attributes) => new[] { inputShapes[0] };
    public void Execute(OnnxOpContext ctx) => _kernels.Sign(ctx.Inputs[0].Data, ctx.Outputs[0].Data, ctx.Inputs[0].ElementCount);
}
