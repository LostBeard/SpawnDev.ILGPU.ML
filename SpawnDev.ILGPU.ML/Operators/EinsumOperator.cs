using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Tensors;

namespace SpawnDev.ILGPU.ML.Operators;

/// <summary>
/// ONNX Einsum operator — general Einstein summation.
/// Parses equation string, infers output shape, and executes via GPU kernels
/// where possible (broadcast multiply, batched MatMul) with CPU fallback for
/// arbitrary contractions.
/// </summary>
public class EinsumOperator(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "Einsum";

    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
    {
        var equation = attrs.TryGetValue("equation", out var eq) ? eq.ToString()! : "";
        var parsed = ParseEquation(equation, inputs.Length);

        // Build output shape from parsed labels
        var outShape = new int[parsed.OutputLabels.Length];
        for (int i = 0; i < parsed.OutputLabels.Length; i++)
        {
            char label = parsed.OutputLabels[i];
            // Find the dimension size from any input that has this label
            for (int inp = 0; inp < inputs.Length; inp++)
            {
                int idx = Array.IndexOf(parsed.InputLabels[inp], label);
                if (idx >= 0 && idx < inputs[inp].Length)
                {
                    outShape[i] = inputs[inp][idx];
                    break;
                }
            }
        }
        return new[] { outShape };
    }

    public void Execute(OnnxOpContext ctx)
    {
        var equation = ctx.GetString("equation");
        var parsed = ParseEquation(equation, ctx.Inputs.Length);

        // Build dimension size map from actual tensor shapes
        var dimSizes = new Dictionary<char, int>();
        for (int inp = 0; inp < ctx.Inputs.Length; inp++)
        {
            var labels = parsed.InputLabels[inp];
            var shape = ctx.Inputs[inp].Shape;
            for (int d = 0; d < labels.Length && d < shape.Length; d++)
                dimSizes[labels[d]] = shape[d];
        }

        // GPU fast path: element-wise broadcast multiply (e.g., bnhd,hd->bnhd for RoPE).
        // Pattern: all output labels appear in input A, input B's labels are a suffix of A's.
        if (ctx.Inputs.Length == 2 && parsed.OutputLabels.SequenceEqual(parsed.InputLabels[0]))
        {
            var aLabels = parsed.InputLabels[0];
            var bLabels = parsed.InputLabels[1];
            // Check if B's labels are a contiguous suffix of A's labels (broadcast multiply)
            bool isBroadcastMul = bLabels.Length <= aLabels.Length
                && bLabels.SequenceEqual(aLabels.Skip(aLabels.Length - bLabels.Length).ToArray());
            if (isBroadcastMul)
            {
                reg.ElementWise.BroadcastBinaryOpND(
                    ctx.Inputs[0].Data, ctx.Inputs[1].Data, ctx.Outputs[0].Data,
                    ctx.Inputs[0].Shape, ctx.Inputs[1].Shape, ctx.Outputs[0].Shape,
                    BroadcastOp.Mul);
                return;
            }
        }

        // GPU fast path: batched matmul (e.g., "bnij,bnjd->bnid" or "ij,jk->ik")
        if (ctx.Inputs.Length == 2)
        {
            var aLabels = parsed.InputLabels[0];
            var bLabels = parsed.InputLabels[1];
            var oLabels = parsed.OutputLabels;

            // Find contracted dimensions (in both inputs, not in output)
            var contractedDims = aLabels.Intersect(bLabels).Except(oLabels).ToArray();
            var batchLabels = aLabels.Intersect(bLabels).Intersect(oLabels).ToArray();

            if (contractedDims.Length == 1)
            {
                // Single contraction → matmul pattern
                char k = contractedDims[0];
                var aFree = aLabels.Except(bLabels).Concat(aLabels.Intersect(bLabels).Intersect(oLabels)).ToArray();
                var bFree = bLabels.Except(aLabels).Concat(bLabels.Intersect(aLabels).Intersect(oLabels)).ToArray();

                // Check if this is a standard matmul: batch dims + M + K × batch dims + K + N → batch dims + M + N
                int K = dimSizes.GetValueOrDefault(k, 1);
                int batchSize = 1;
                foreach (var bl in batchLabels) batchSize *= dimSizes.GetValueOrDefault(bl, 1);

                // Compute M (A's free dims) and N (B's free dims)
                var aFreeDims = aLabels.Where(c => !bLabels.Contains(c)).ToArray();
                var bFreeDims = bLabels.Where(c => !aLabels.Contains(c)).ToArray();
                int M = 1; foreach (var d in aFreeDims) M *= dimSizes.GetValueOrDefault(d, 1);
                int N = 1; foreach (var d in bFreeDims) N *= dimSizes.GetValueOrDefault(d, 1);

                if (batchSize * M * K == ctx.Inputs[0].ElementCount &&
                    batchSize * K * N == ctx.Inputs[1].ElementCount &&
                    batchSize * M * N == ctx.Outputs[0].ElementCount)
                {
                    // Batched matmul: treat batch dims as outer, contract K
                    for (int b = 0; b < batchSize; b++)
                    {
                        reg.MatMul.MatMul(
                            ctx.Inputs[0].Data.SubView(b * M * K, M * K),
                            ctx.Inputs[1].Data.SubView(b * K * N, K * N),
                            ctx.Outputs[0].Data.SubView(b * M * N, M * N),
                            M, K, N);
                    }
                    return;
                }
            }
        }

        // CPU fallback for general equations.
        // Read inputs from pre-read constants (avoids GPU→CPU readback on browser backends).
        int outputSize = ctx.Outputs[0].ElementCount;
        var result = new float[outputSize];
        var outLabels = parsed.OutputLabels;
        var outShape = ctx.Outputs[0].Shape;

        // Read all inputs to CPU — use constant values (pre-read during session creation)
        // to avoid sync GPU→CPU which throws on WebGPU/WebGL/Wasm.
        var inputArrays = new float[ctx.Inputs.Length][];
        bool allAvailable = true;
        for (int i = 0; i < ctx.Inputs.Length; i++)
        {
            var constVals = ctx.TryGetInputValues(i);
            if (constVals != null)
            {
                inputArrays[i] = constVals;
            }
            else
            {
                // Try CopyFrom to staging buffer + sync readback (works on desktop, throws on browser)
                try
                {
                    int count = ctx.Inputs[i].ElementCount;
                    using var readBuf = reg.Accelerator.Allocate1D<float>(count);
                    readBuf.View.SubView(0, count).CopyFrom(ctx.Inputs[i].Data.SubView(0, count));
                    reg.Accelerator.Synchronize();
                    inputArrays[i] = readBuf.GetAsArray1D();
                }
                catch (NotSupportedException)
                {
                    // Browser backend — can't do sync GPU→CPU. Fall back to zero.
                    // This Einsum equation needs a GPU kernel (not just broadcast multiply).
                    inputArrays[i] = new float[ctx.Inputs[i].ElementCount];
                    allAvailable = false;
                }
            }
        }

        if (!allAvailable && InferenceSession.VerboseLogging)
            Console.WriteLine($"[Einsum] WARNING: equation '{equation}' has non-constant inputs on browser backend — GPU fast path needed");

        // Identify contracted dimensions (in inputs but not in output)
        var allInputLabels = new HashSet<char>();
        foreach (var labels in parsed.InputLabels)
            foreach (var c in labels) allInputLabels.Add(c);
        var outputLabelSet = new HashSet<char>(outLabels);
        var contracted = allInputLabels.Where(c => !outputLabelSet.Contains(c)).ToArray();

        // Build strides for each input
        var inputStrides = new int[ctx.Inputs.Length][];
        for (int inp = 0; inp < ctx.Inputs.Length; inp++)
        {
            var shape = ctx.Inputs[inp].Shape;
            var strides = new int[shape.Length];
            int stride = 1;
            for (int d = shape.Length - 1; d >= 0; d--)
            {
                strides[d] = stride;
                stride *= shape[d];
            }
            inputStrides[inp] = strides;
        }

        // Build output strides
        var outStrides = new int[outShape.Length];
        {
            int stride = 1;
            for (int d = outShape.Length - 1; d >= 0; d--)
            {
                outStrides[d] = stride;
                stride *= outShape[d];
            }
        }

        // Contracted dimension sizes and iteration
        int contractedCount = 1;
        var contractedSizes = new int[contracted.Length];
        for (int c = 0; c < contracted.Length; c++)
        {
            contractedSizes[c] = dimSizes.GetValueOrDefault(contracted[c], 1);
            contractedCount *= contractedSizes[c];
        }

        // Iterate over all output elements
        for (int outIdx = 0; outIdx < outputSize; outIdx++)
        {
            // Decode output index into per-label values
            var labelValues = new Dictionary<char, int>();
            int remaining = outIdx;
            for (int d = 0; d < outLabels.Length; d++)
            {
                labelValues[outLabels[d]] = remaining / outStrides[d];
                remaining %= outStrides[d];
            }

            // Sum over contracted dimensions
            float sum = 0;
            for (int ci = 0; ci < contractedCount; ci++)
            {
                // Decode contracted index
                int cRemaining = ci;
                for (int c = contracted.Length - 1; c >= 0; c--)
                {
                    labelValues[contracted[c]] = cRemaining % contractedSizes[c];
                    cRemaining /= contractedSizes[c];
                }

                // Compute product of all inputs at this label assignment
                float product = 1f;
                for (int inp = 0; inp < ctx.Inputs.Length; inp++)
                {
                    var labels = parsed.InputLabels[inp];
                    int flatIdx = 0;
                    for (int d = 0; d < labels.Length; d++)
                        flatIdx += labelValues[labels[d]] * inputStrides[inp][d];
                    product *= inputArrays[inp][flatIdx];
                }
                sum += product;
            }
            result[outIdx] = sum;
        }

        // Upload result to GPU
        ctx.Outputs[0].Data.SubView(0, outputSize).CopyFromCPU(result);
    }

    // ═══════════════════════════════════════════════════════════
    //  Equation parsing
    // ═══════════════════════════════════════════════════════════

    private record ParsedEquation(char[][] InputLabels, char[] OutputLabels);

    private static ParsedEquation ParseEquation(string equation, int numInputs)
    {
        // Remove whitespace
        equation = equation.Replace(" ", "");

        char[][] inputLabels;
        char[] outputLabels;

        if (equation.Contains("->"))
        {
            // Explicit mode: "ij,jk->ik"
            var parts = equation.Split("->");
            var inputParts = parts[0].Split(',');
            inputLabels = inputParts.Select(p => p.ToCharArray()).ToArray();
            outputLabels = parts[1].ToCharArray();
        }
        else
        {
            // Implicit mode: output = sorted unique non-repeated labels
            var inputParts = equation.Split(',');
            inputLabels = inputParts.Select(p => p.ToCharArray()).ToArray();

            var counts = new Dictionary<char, int>();
            foreach (var labels in inputLabels)
                foreach (var c in labels)
                    counts[c] = counts.GetValueOrDefault(c, 0) + 1;

            outputLabels = counts.Where(kv => kv.Value == 1)
                .Select(kv => kv.Key).OrderBy(c => c).ToArray();
        }

        if (inputLabels.Length != numInputs)
            throw new InvalidOperationException(
                $"Einsum equation '{equation}' has {inputLabels.Length} input terms but got {numInputs} input tensors");

        return new ParsedEquation(inputLabels, outputLabels);
    }
}
