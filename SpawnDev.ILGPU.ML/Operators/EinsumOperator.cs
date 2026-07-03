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
        var parsed = ParseEquation(ctx.GetString("equation"), ctx.Inputs.Length);
        var dimSizes = BuildDimSizes(ctx, parsed);
        if (TryGpuFastPath(ctx, parsed, dimSizes)) return;
        // Sync CPU fallback (desktop). Dynamic inputs use sync GPU->CPU which throws on
        // browser backends — there they fall back to zeros (see ExecuteAsync for parity).
        var inputArrays = ReadInputsSync(ctx);
        ComputeGeneralContraction(ctx, parsed, dimSizes, inputArrays);
    }

    /// <summary>
    /// Browser-safe async path (GraphExecutor.RunAsync). Identical to <see cref="Execute"/>
    /// except dynamic (non-constant) inputs are read back via the async <c>CopyToHostAsync</c>
    /// instead of the synchronous readback that throws on WebGPU/WebGL/Wasm — giving the general
    /// einsum contraction full feature parity on the browser backends instead of zeros.
    /// </summary>
    public async Task ExecuteAsync(OnnxOpContext ctx)
    {
        var parsed = ParseEquation(ctx.GetString("equation"), ctx.Inputs.Length);
        var dimSizes = BuildDimSizes(ctx, parsed);
        if (TryGpuFastPath(ctx, parsed, dimSizes)) return;
        // CUDA-graph capture pass: the general contraction reads inputs BACK to CPU (illegal mid-capture). This
        // path only runs for small non-GPU-fast einsums (shape/position contractions like RoPE positions⊗inv_freq),
        // whose output is CONSTANT for a fixed shape. The warm pass staged that constant into a stable arena slot
        // (ComputeGeneralContraction below, same cursor position); copy it via a CAPTURED GPU CopyFrom so replay
        // reproduces the write. (Skipping the write leaves the reused pooled output buffer stale on replay.)
        if (SpawnDev.ILGPU.ML.Graph.GraphExecutor.SuppressDrains)
        {
            int outSz = ctx.Outputs[0].ElementCount;
            var slot = Kernels.CaptureParamArena.Shared(reg.Accelerator).RentStableSlotFloat(new float[outSz]);
            ctx.Outputs[0].Data.SubView(0, outSz).CopyFrom(slot);
            return;
        }
        var inputArrays = await ReadInputsAsync(ctx);
        ComputeGeneralContraction(ctx, parsed, dimSizes, inputArrays);
    }

    /// <summary>Builds the einsum dimension-size map from the actual input tensor shapes.</summary>
    private static Dictionary<char, int> BuildDimSizes(OnnxOpContext ctx, ParsedEquation parsed)
    {
        var dimSizes = new Dictionary<char, int>();
        for (int inp = 0; inp < ctx.Inputs.Length; inp++)
        {
            var labels = parsed.InputLabels[inp];
            var shape = ctx.Inputs[inp].Shape;
            for (int d = 0; d < labels.Length && d < shape.Length; d++)
                dimSizes[labels[d]] = shape[d];
        }
        return dimSizes;
    }

    /// <summary>
    /// Attempts the all-GPU fast paths (element-wise broadcast multiply, batched matmul).
    /// Returns true if the equation was handled entirely on the GPU (no CPU readback needed).
    /// </summary>
    private bool TryGpuFastPath(OnnxOpContext ctx, ParsedEquation parsed, Dictionary<char, int> dimSizes)
    {
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
                return true;
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
                    // A NATIVE low-precision operand (Half/bf16/FP8) has an EMPTY float .Data, so the batched
                    // matmul below would read out of bounds (the same cryptic "Index/Extent X out of bounds" that
                    // bit FusedLinear). Einsum has no low-p kernel path; a low-p WEIGHT should reach the graph as a
                    // MatMul/Gemm (which DO consume native low-p via LowPWeightDispatch). Fail loud, not cryptic.
                    if (LowPWeightDispatch.IsLowP(ctx.Inputs[0]) || LowPWeightDispatch.IsLowP(ctx.Inputs[1]))
                        throw new NotSupportedException(
                            "Einsum GPU matmul fast-path: a native low-precision operand (DType " +
                            $"{(LowPWeightDispatch.IsLowP(ctx.Inputs[0]) ? ctx.Inputs[0].DType : ctx.Inputs[1].DType)}) " +
                            "is not supported - its float Data is empty. Consume the low-p weight as MatMul/Gemm " +
                            "(native low-p), or add a low-p einsum kernel.");

                    // Batched matmul: ONE batched dispatch, batch dims as Grid.IdxY. The old per-batch C#
                    // loop issued batchSize separate MatMul dispatches (DAv3 attention: 6 heads × 26 einsum
                    // nodes = 156 dispatches per inference of pure launch churn on every backend, and each
                    // was a skinny per-head GEMM). BatchedMatMul routes register-blocked for large M/N.
                    reg.MatMul.BatchedMatMul(
                        ctx.Inputs[0].Data, ctx.Inputs[1].Data, ctx.Outputs[0].Data,
                        batchSize, M, K, N);
                    return true;
                }
            }
        }

        return false;
    }

    /// <summary>
    /// Sync CPU-fallback input read. Pre-read constants are used directly; dynamic inputs use
    /// a sync GPU-&gt;CPU readback that works on desktop but THROWS on WebGPU/WebGL/Wasm — there
    /// the input falls back to zeros (the async <see cref="ReadInputsAsync"/> path reads it
    /// properly). Returns one float[] per input.
    /// </summary>
    private float[][] ReadInputsSync(OnnxOpContext ctx)
    {
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
                    // This Einsum equation needs the async path (ExecuteAsync) for real data.
                    inputArrays[i] = new float[ctx.Inputs[i].ElementCount];
                    allAvailable = false;
                }
            }
        }

        if (!allAvailable && InferenceSession.VerboseLogging)
            Console.WriteLine($"[Einsum] WARNING: equation '{ctx.GetString("equation")}' has non-constant inputs on a browser backend via the SYNC path — use ExecuteAsync (GraphExecutor.RunAsync) for real readback");

        return inputArrays;
    }

    /// <summary>
    /// Browser-safe async input read. Pre-read constants are used directly; dynamic inputs are
    /// staged GPU-&gt;GPU (<c>CopyFrom</c>, valid on all backends) then read back via the async
    /// <c>CopyToHostAsync</c> (mapAsync / SAB / GL readback) — so the general contraction gets
    /// real input data on WebGPU/WebGL/Wasm instead of zeros. Returns one float[] per input.
    /// </summary>
    private async Task<float[][]> ReadInputsAsync(OnnxOpContext ctx)
    {
        var inputArrays = new float[ctx.Inputs.Length][];
        for (int i = 0; i < ctx.Inputs.Length; i++)
        {
            var constVals = ctx.TryGetInputValues(i);
            if (constVals != null)
            {
                inputArrays[i] = constVals;
                continue;
            }
            int count = ctx.Inputs[i].ElementCount;
            using var readBuf = reg.Accelerator.Allocate1D<float>(count);
            readBuf.View.SubView(0, count).CopyFrom(ctx.Inputs[i].Data.SubView(0, count));
            inputArrays[i] = await readBuf.CopyToHostAsync<float>(0, count);
        }
        return inputArrays;
    }

    /// <summary>
    /// General N-input einsum contraction on the CPU-read inputs, writing the result back to the
    /// GPU output via <c>CopyFromCPU</c> (valid on all backends). Pure compute — no GPU readback.
    /// </summary>
    private void ComputeGeneralContraction(
        OnnxOpContext ctx, ParsedEquation parsed, Dictionary<char, int> dimSizes, float[][] inputArrays)
    {
        int outputSize = ctx.Outputs[0].ElementCount;
        var result = new float[outputSize];
        var outLabels = parsed.OutputLabels;
        var outShape = ctx.Outputs[0].Shape;

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

        // Upload result to GPU. Under CUDA-graph capture-mode this stages the (constant) result into a stable
        // arena slot (at the einsum's deterministic cursor position) + does a captured GPU CopyFrom; the capture
        // pass reads that same slot (see ExecuteAsync). This runs on the WARM pass (capture skips it above).
        if (SpawnDev.ILGPU.ML.Graph.GraphExecutor.UseCaptureParamSlots)
            Kernels.CaptureParamArena.CaptureConstWrite(reg.Accelerator, ctx.Outputs[0].Data.SubView(0, outputSize), result);
        else
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
