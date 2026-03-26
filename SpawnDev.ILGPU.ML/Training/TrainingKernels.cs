using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Training;

/// <summary>
/// GPU kernels for neural network training: backward passes and optimizers.
/// These complement the existing inference kernels to enable end-to-end GPU training.
/// </summary>
public class TrainingKernels
{
    private readonly Accelerator _accelerator;

    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, int, int>? _softmaxCrossEntropyForwardKernel;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, int>? _softmaxCrossEntropyBackwardKernel;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>>? _reluBackwardKernel;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        float>? _sgdUpdateKernel;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>>? _zeroKernel;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        float, float, float, int>? _adamUpdateKernel;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, int, int, int>? _matMulBackwardDataKernel;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, int, int, int>? _matMulBackwardWeightKernel;

    public TrainingKernels(Accelerator accelerator) => _accelerator = accelerator;

    // ═══════════════════════════════════════════════════════════
    //  Loss Functions
    // ═══════════════════════════════════════════════════════════

    /// <summary>
    /// Softmax + Cross-Entropy forward: logits [batch, classes] + targets → loss per sample.
    /// Numerically stable: subtracts max before exp.
    /// Loss = -log(softmax(logits)[target_class]).
    /// </summary>
    public void SoftmaxCrossEntropyForward(
        ArrayView1D<float, Stride1D.Dense> logits,
        ArrayView1D<float, Stride1D.Dense> probs,
        ArrayView1D<float, Stride1D.Dense> loss,
        ArrayView1D<int, Stride1D.Dense> targets,
        int batchSize, int numClasses)
    {
        _softmaxCrossEntropyForwardKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>,
            int, int>(SoftmaxCEForwardImpl);
        _softmaxCrossEntropyForwardKernel(batchSize, logits, probs, loss, targets, batchSize, numClasses);
    }

    private static void SoftmaxCEForwardImpl(Index1D batch,
        ArrayView1D<float, Stride1D.Dense> logits,
        ArrayView1D<float, Stride1D.Dense> probs,
        ArrayView1D<float, Stride1D.Dense> loss,
        ArrayView1D<int, Stride1D.Dense> targets,
        int B, int C)
    {
        int offset = batch * C;

        // Find max for numerical stability
        float maxVal = logits[offset];
        for (int c = 1; c < C; c++)
        {
            float v = logits[offset + c];
            if (v > maxVal) maxVal = v;
        }

        // Compute exp and sum
        float sumExp = 0f;
        for (int c = 0; c < C; c++)
        {
            float e = MathF.Exp(logits[offset + c] - maxVal);
            probs[offset + c] = e;
            sumExp += e;
        }

        // Normalize to get probabilities
        float invSum = 1f / sumExp;
        for (int c = 0; c < C; c++)
            probs[offset + c] *= invSum;

        // Cross-entropy loss: -log(prob[target])
        int target = targets[batch];
        float prob = probs[offset + target];
        loss[batch] = -MathF.Log(prob + 1e-10f); // epsilon to prevent log(0)
    }

    /// <summary>
    /// Softmax + Cross-Entropy backward: grad = softmax(logits) - one_hot(targets).
    /// </summary>
    public void SoftmaxCrossEntropyBackward(
        ArrayView1D<float, Stride1D.Dense> probs,
        ArrayView1D<int, Stride1D.Dense> targets,
        ArrayView1D<float, Stride1D.Dense> gradLogits,
        int batchSize, int numClasses)
    {
        _softmaxCrossEntropyBackwardKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, int>(SoftmaxCEBackwardImpl);
        _softmaxCrossEntropyBackwardKernel(batchSize * numClasses, probs, targets, gradLogits, numClasses);
    }

    private static void SoftmaxCEBackwardImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> probs,
        ArrayView1D<int, Stride1D.Dense> targets,
        ArrayView1D<float, Stride1D.Dense> gradLogits,
        int C)
    {
        int batch = idx / C;
        int cls = idx % C;
        float grad = probs[idx];
        if (cls == targets[batch])
            grad -= 1f;
        gradLogits[idx] = grad;
    }

    // ═══════════════════════════════════════════════════════════
    //  Activation Backward
    // ═══════════════════════════════════════════════════════════

    /// <summary>
    /// ReLU backward: grad_input = grad_output * (input > 0 ? 1 : 0).
    /// </summary>
    public void ReLUBackward(
        ArrayView1D<float, Stride1D.Dense> gradOutput,
        ArrayView1D<float, Stride1D.Dense> forwardInput,
        ArrayView1D<float, Stride1D.Dense> gradInput,
        int count)
    {
        _reluBackwardKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>>(ReLUBackwardImpl);
        _reluBackwardKernel(count, gradOutput, forwardInput, gradInput);
    }

    private static void ReLUBackwardImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> gradOutput,
        ArrayView1D<float, Stride1D.Dense> forwardInput,
        ArrayView1D<float, Stride1D.Dense> gradInput)
    {
        gradInput[idx] = forwardInput[idx] > 0f ? gradOutput[idx] : 0f;
    }

    // ═══════════════════════════════════════════════════════════
    //  Linear Layer Backward (MatMul gradients)
    // ═══════════════════════════════════════════════════════════

    /// <summary>
    /// Linear backward (dInput): gradInput = gradOutput @ weight.
    /// gradOutput [B, outF], weight [outF, inF] → gradInput [B, inF]
    /// </summary>
    public void LinearBackwardData(
        ArrayView1D<float, Stride1D.Dense> gradOutput,
        ArrayView1D<float, Stride1D.Dense> weight,
        ArrayView1D<float, Stride1D.Dense> gradInput,
        int batchSize, int outFeatures, int inFeatures)
    {
        _matMulBackwardDataKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, int, int, int>(LinearBackwardDataImpl);
        _matMulBackwardDataKernel(batchSize * inFeatures, gradOutput, weight, gradInput,
            batchSize, outFeatures, inFeatures);
    }

    private static void LinearBackwardDataImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> gradOutput,
        ArrayView1D<float, Stride1D.Dense> weight,
        ArrayView1D<float, Stride1D.Dense> gradInput,
        int B, int O, int I)
    {
        int b = idx / I;
        int i = idx % I;
        float sum = 0f;
        for (int o = 0; o < O; o++)
            sum += gradOutput[b * O + o] * weight[o * I + i];
        gradInput[idx] = sum;
    }

    /// <summary>
    /// Linear backward (dWeight): gradWeight = gradOutput^T @ input.
    /// gradOutput [B, outF], input [B, inF] → gradWeight [outF, inF]
    /// </summary>
    public void LinearBackwardWeight(
        ArrayView1D<float, Stride1D.Dense> gradOutput,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> gradWeight,
        int batchSize, int outFeatures, int inFeatures)
    {
        _matMulBackwardWeightKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, int, int, int>(LinearBackwardWeightImpl);
        _matMulBackwardWeightKernel(outFeatures * inFeatures, gradOutput, input, gradWeight,
            batchSize, outFeatures, inFeatures);
    }

    private static void LinearBackwardWeightImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> gradOutput,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> gradWeight,
        int B, int O, int I)
    {
        int o = idx / I;
        int i = idx % I;
        float sum = 0f;
        for (int b = 0; b < B; b++)
            sum += gradOutput[b * O + o] * input[b * I + i];
        gradWeight[idx] = sum;
    }

    // ═══════════════════════════════════════════════════════════
    //  Optimizers
    // ═══════════════════════════════════════════════════════════

    /// <summary>
    /// SGD update: weight -= learningRate * gradient.
    /// </summary>
    public void SGDUpdate(
        ArrayView1D<float, Stride1D.Dense> weights,
        ArrayView1D<float, Stride1D.Dense> gradients,
        int count, float learningRate)
    {
        _sgdUpdateKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            float>(SGDUpdateImpl);
        _sgdUpdateKernel(count, weights, gradients, learningRate);
    }

    private static void SGDUpdateImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> weights,
        ArrayView1D<float, Stride1D.Dense> gradients,
        float lr)
    {
        weights[idx] -= lr * gradients[idx];
    }

    /// <summary>
    /// Adam optimizer update with moment estimation and bias correction.
    /// </summary>
    public void AdamUpdate(
        ArrayView1D<float, Stride1D.Dense> weights,
        ArrayView1D<float, Stride1D.Dense> gradients,
        ArrayView1D<float, Stride1D.Dense> m,
        ArrayView1D<float, Stride1D.Dense> v,
        int count, float lr, float beta1, float beta2, int step)
    {
        _adamUpdateKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            float, float, float, int>(AdamUpdateImpl);
        _adamUpdateKernel(count, weights, gradients, m, v, lr, beta1, beta2, step);
    }

    private static void AdamUpdateImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> weights,
        ArrayView1D<float, Stride1D.Dense> gradients,
        ArrayView1D<float, Stride1D.Dense> m,
        ArrayView1D<float, Stride1D.Dense> v,
        float lr, float beta1, float beta2, int step)
    {
        float g = gradients[idx];
        m[idx] = beta1 * m[idx] + (1f - beta1) * g;
        v[idx] = beta2 * v[idx] + (1f - beta2) * g * g;

        float mHat = m[idx] / (1f - MathF.Pow(beta1, step));
        float vHat = v[idx] / (1f - MathF.Pow(beta2, step));

        weights[idx] -= lr * mHat / (MathF.Sqrt(vHat) + 1e-8f);
    }

    // ═══════════════════════════════════════════════════════════
    //  Utilities
    // ═══════════════════════════════════════════════════════════

    /// <summary>Zero-fill a buffer on GPU.</summary>
    public void Zero(ArrayView1D<float, Stride1D.Dense> buffer, int count)
    {
        _zeroKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>>(ZeroImpl);
        _zeroKernel(count, buffer);
    }

    private static void ZeroImpl(Index1D idx, ArrayView1D<float, Stride1D.Dense> buffer)
    {
        buffer[idx] = 0f;
    }
}
