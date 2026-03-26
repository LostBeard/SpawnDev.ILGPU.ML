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
    //  Linear Layer Forward
    // ═══════════════════════════════════════════════════════════

    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, int, int, int>? _linearForwardKernel;

    /// <summary>
    /// Linear forward: output = input @ weight^T.
    /// input [B, inF], weight [outF, inF] → output [B, outF]
    /// output[b, o] = sum_i(input[b, i] * weight[o, i])
    /// </summary>
    public void LinearForward(
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> weight,
        ArrayView1D<float, Stride1D.Dense> output,
        int batchSize, int inFeatures, int outFeatures)
    {
        _linearForwardKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, int, int, int>(LinearForwardImpl);
        _linearForwardKernel(batchSize * outFeatures, input, weight, output,
            batchSize, inFeatures, outFeatures);
    }

    private static void LinearForwardImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> weight,
        ArrayView1D<float, Stride1D.Dense> output,
        int B, int I, int O)
    {
        int b = idx / O;
        int o = idx % O;
        float sum = 0f;
        for (int i = 0; i < I; i++)
            sum += input[b * I + i] * weight[o * I + i];
        output[idx] = sum;
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
    //  Conv2D Backward (for CNN training)
    // ═══════════════════════════════════════════════════════════

    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, int, int, int, int, int, int>? _conv2dBackwardWeightKernel;

    /// <summary>
    /// Conv2D backward (weight gradient): dW[oc,ic,kh,kw] = sum over batch,oh,ow of gradOut[b,oc,oh,ow] * input[b,ic,oh+kh,ow+kw].
    /// </summary>
    public void Conv2DBackwardWeight(
        ArrayView1D<float, Stride1D.Dense> gradOutput,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> gradWeight,
        int batchSize, int inC, int outC, int H, int W,
        int kH, int kW, int outH, int outW)
    {
        int totalWeightElems = outC * inC * kH * kW;
        _conv2dBackwardWeightKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, int, int, int, int, int, int>(Conv2DBackwardWeightImpl);
        _conv2dBackwardWeightKernel(totalWeightElems, gradOutput, input, gradWeight,
            batchSize, inC, H * 1000 + W, outC, kH * 100 + kW, outH * 100 + outW);
    }

    private static void Conv2DBackwardWeightImpl(Index1D wIdx,
        ArrayView1D<float, Stride1D.Dense> gradOutput,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> gradWeight,
        int B, int inC, int HW, int outC, int kHW, int outHW)
    {
        int H = HW / 1000, W = HW % 1000;
        int kH = kHW / 100, kW = kHW % 100;
        int outH = outHW / 100, outW = outHW % 100;

        int oc = wIdx / (inC * kH * kW);
        int rem = wIdx % (inC * kH * kW);
        int ic = rem / (kH * kW);
        rem %= (kH * kW);
        int kh = rem / kW, kw = rem % kW;

        float sum = 0f;
        for (int b = 0; b < B; b++)
            for (int oh = 0; oh < outH; oh++)
                for (int ow = 0; ow < outW; ow++)
                {
                    float go = gradOutput[((b * outC + oc) * outH + oh) * outW + ow];
                    float inp = input[((b * inC + ic) * H + (oh + kh)) * W + (ow + kw)];
                    sum += go * inp;
                }
        gradWeight[wIdx] = sum;
    }

    // ═══════════════════════════════════════════════════════════
    //  MaxPool2D with Indices (for backward pass)
    // ═══════════════════════════════════════════════════════════

    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<int, Stride1D.Dense>, int, int, int>? _maxPoolForwardKernel;

    /// <summary>
    /// MaxPool2D forward saving argmax indices for backward pass.
    /// </summary>
    public void MaxPool2DForward(
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<int, Stride1D.Dense> indices,
        int batchChannels, int H, int W, int poolSize)
    {
        int outH = H / poolSize, outW = W / poolSize;
        _maxPoolForwardKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>, int, int, int>(MaxPool2DForwardImpl);
        _maxPoolForwardKernel(batchChannels * outH * outW, input, output, indices,
            H, W, poolSize);
    }

    private static void MaxPool2DForwardImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<int, Stride1D.Dense> indices,
        int H, int W, int P)
    {
        int outW = W / P, outH = H / P;
        int bc = idx / (outH * outW);
        int rem = idx % (outH * outW);
        int oh = rem / outW, ow = rem % outW;
        int baseOff = bc * H * W;

        float maxVal = float.MinValue;
        int maxIdx = 0;
        for (int ph = 0; ph < P; ph++)
            for (int pw = 0; pw < P; pw++)
            {
                int ii = baseOff + (oh * P + ph) * W + (ow * P + pw);
                if (input[ii] > maxVal) { maxVal = input[ii]; maxIdx = ii; }
            }
        output[idx] = maxVal;
        indices[idx] = maxIdx;
    }

    /// <summary>
    /// MaxPool2D backward: scatter gradients to saved argmax positions.
    /// </summary>
    public void MaxPool2DBackward(
        ArrayView1D<float, Stride1D.Dense> gradOutput,
        ArrayView1D<int, Stride1D.Dense> indices,
        ArrayView1D<float, Stride1D.Dense> gradInput,
        int outputCount, int inputCount)
    {
        Zero(gradInput, inputCount);
        // Scatter — safe for non-overlapping pools (stride == poolSize)
        var kernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>>(
            (Index1D i, ArrayView1D<float, Stride1D.Dense> go, ArrayView1D<int, Stride1D.Dense> idx,
             ArrayView1D<float, Stride1D.Dense> gi) => { gi[idx[i]] = go[i]; });
        kernel(outputCount, gradOutput, indices, gradInput);
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
