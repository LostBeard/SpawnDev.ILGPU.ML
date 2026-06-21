using ILGPU;
using SpawnDev.ILGPU.ML.Kernels;
using SpawnDev.ILGPU.ML.Tensors;

namespace SpawnDev.ILGPU.ML.Operators;

/// <summary>
/// Fused Linear operator: Output = Activation(MatMul(Input, Weights) + Bias)
/// Executes matrix multiplication, bias addition, and activation in a single kernel dispatch.
/// Eliminates 2 out of 3 global memory write cycles.
///
/// Inputs: [input, weights, bias]
/// Outputs: [result]
/// Attributes: activation ("none", "Relu", "Gelu", "Sigmoid", "Tanh", "Clip")
/// </summary>
public class FusedLinearOperator : IOnnxOperator
{
    private readonly OperatorRegistry _registry;
    private FusedLinearKernel? _kernel;

    public FusedLinearOperator(OperatorRegistry registry) => _registry = registry;

    public string OpType => "FusedLinear";

    public int[][] InferOutputShapes(int[][] inputShapes, Dictionary<string, object> attributes)
    {
        // Input: [M, K], Weights: [K, N] or [N, K] (transB) → Output: [M, N]
        if (inputShapes.Length < 2) return new[] { inputShapes[0] };
        int K = inputShapes[0][^1];
        // Detect weight layout: if weights[^1] == K, layout is [N, K] (transB)
        int N = inputShapes[1][^1] == K ? inputShapes[1][0] : inputShapes[1][^1];
        var outShape = inputShapes[0].ToArray();
        outShape[^1] = N;
        return new[] { outShape };
    }

    public void Execute(OnnxOpContext ctx)
    {
        var input = ctx.Inputs[0];  // [..., M, K] (may have leading batch dims)
        var weights = ctx.Inputs[1]; // [K, N]
        var bias = ctx.Inputs[2];    // [N]
        var output = ctx.Outputs[0]; // [..., M, N]

        int K = input.Shape[^1];
        // Detect weight layout: [K, N] or [N, K] (transB from TFLite/Gemm)
        int N = weights.Shape[^1] == K ? weights.Shape[0] : weights.Shape[^1];
        // Flatten all leading dimensions into M — kernel does flat row/col indexing
        int M = input.ElementCount / K;

        // Bounds validation with diagnostic info. NOTE: weights.ElementCount is SHAPE-derived, so it is correct
        // for both a fp32 weight (float Data, length == ElementCount) AND a native low-p weight (data in the
        // typed low-p view; float Data EMPTY). The low-p case is routed below — it must NOT touch weights.Data.
        if (M < 1 || M * K != input.ElementCount || K * N > weights.ElementCount || N > bias.ElementCount || M * N > output.ElementCount)
        {
            throw new InvalidOperationException(
                $"FusedLinear bounds mismatch: M={M} K={K} N={N}, " +
                $"input=[{string.Join(",", input.Shape)}] buf={input.ElementCount}, " +
                $"weights=[{string.Join(",", weights.Shape)}] buf={weights.ElementCount}, " +
                $"bias=[{string.Join(",", bias.Shape)}] buf={bias.ElementCount}, " +
                $"output=[{string.Join(",", output.Shape)}] buf={output.ElementCount}");
        }

        // Determine activation type
        var actStr = ctx.GetString("activation", "none").ToLowerInvariant();
        var activation = actStr switch
        {
            "relu" => FusedActivation.ReLU,
            "gelu" => FusedActivation.GELU,
            "sigmoid" => FusedActivation.Sigmoid,
            "tanh" => FusedActivation.Tanh,
            "silu" => FusedActivation.SiLU,
            "clip" => FusedActivation.ReLU, // RELU6 approximation
            _ => FusedActivation.None
        };

        _kernel ??= new FusedLinearKernel(_registry.Accelerator);

        // Native low-precision weight (bf16/fp16/FP8 — e.g. gpt-oss attn/output projections kept native by the
        // GGUF loader, no f32 upcast): its float Data view is EMPTY, so the fp32 kernel below would read out of
        // bounds. Route to the generic low-p fused kernel, which reads the weight in its native type and converts
        // to float in-register (Rule 4 no-upcast). Input/bias/output are always fp32 activations.
        if (LowPWeightDispatch.IsLowP(weights))
        {
            LowPWeightDispatch.FusedLinear(_kernel,
                input.Data.SubView(0, M * K),
                weights,
                bias.Data.SubView(0, N),
                output.Data.SubView(0, M * N),
                M, K, N, activation);
            return;
        }

        // Quantized weight (Q4_K/Q6_K/…) fused into FusedLinear: its float Data is EMPTY (ShapeOnly), so the
        // FP32 kernel can't read it. Route through the fused dequant matmul, then bias + activation — the
        // correct equivalent of the unfused MatMul→Add→Activation. This is what lets qwen2/qwen3 run: their
        // attention q/k/v projections carry a bias, so GraphOptimizer fuses them into FusedLinear; gemma4's
        // bias-less projections never fuse (they stay MatMul → dequant). Robust to shape-recompiles (runs at
        // execute time off ctx.QuantizedWeights, independent of how the graph was compiled).
        if (weights.Data.Length < (long)K * N && ctx.QuantizedWeights != null
            && weights.Name != null && ctx.QuantizedWeights.TryGetValue(weights.Name, out var qWeightData))
        {
            if (_registry.QuantizedWeightTypes == null
                || !_registry.QuantizedWeightTypes.TryGetValue(weights.Name, out var qType))
                throw new InvalidOperationException(
                    $"FusedLinear: quantized weight '{weights.Name}' has no GGML type registered " +
                    "(OperatorRegistry.QuantizedWeightTypes) — the loader must record the type with the bytes.");
            var outView = output.Data.SubView(0, M * N);
            _registry.FusedDequant.Forward(input.Data.SubView(0, M * K), qWeightData, outView, M, K, N, qType);
            _registry.ElementWise.AddBias(outView, bias.Data.SubView(0, N), M * N, N); // broadcast bias over rows
            int nMN = M * N;
            switch (activation) // mirror the activations the fused FP32 kernel supports
            {
                case FusedActivation.None: break;
                case FusedActivation.ReLU: _registry.ElementWise.ReLUInPlace(outView, nMN); break;
                case FusedActivation.GELU: _registry.ElementWise.GELUInPlace(outView, nMN); break;
                case FusedActivation.Sigmoid: _registry.Activations.SigmoidInPlace(outView, nMN); break;
                case FusedActivation.Tanh: _registry.Activations.TanhInPlace(outView, nMN); break;
                case FusedActivation.SiLU: _registry.Activations.SiLUInPlace(outView, nMN); break;
                default: throw new NotSupportedException($"FusedLinear quantized path: activation {activation} not yet supported.");
            }
            return;
        }

        if (weights.Data.Length < (long)K * N)
            throw new InvalidOperationException(
                $"FusedLinear weight '{weights.Name}' float path but its float Data is empty (len={weights.Data.Length}, " +
                $"need K*N={(long)K * N}, DType={weights.DType}, shape=[{string.Join(",", weights.Shape)}]). A quantized/shape-only " +
                $"weight reached the FP32 linear path but is not in ctx.QuantizedWeights — the quantized byte-view map " +
                "was not wired into this executor.");

        _kernel.Forward(
            input.Data.SubView(0, M * K),
            weights.Data.SubView(0, K * N),
            bias.Data.SubView(0, N),
            output.Data.SubView(0, M * N),
            M, K, N, activation);
    }
}
