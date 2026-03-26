using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Training;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// Tests for the GPU training engine: backward passes, optimizers, and TrainableModel.
/// </summary>
public abstract partial class MLTestBase
{
    // ═══════════════════════════════════════════════════════════
    //  SoftmaxCrossEntropy Tests
    // ═══════════════════════════════════════════════════════════

    [TestMethod]
    public async Task Training_SoftmaxCE_Backward_GradientIsCorrect() => await RunTest(async accelerator =>
    {
        var kernels = new TrainingKernels(accelerator);

        // 2 samples, 3 classes
        // Sample 0: logits [2.0, 1.0, 0.1] → softmax → [0.659, 0.242, 0.099], target=0
        // Sample 1: logits [0.1, 0.5, 3.0] → softmax → [0.035, 0.052, 0.913], target=2
        var logits = new float[] { 2f, 1f, 0.1f, 0.1f, 0.5f, 3f };
        var targets = new int[] { 0, 2 };

        using var logitsBuf = accelerator.Allocate1D(logits);
        using var probsBuf = accelerator.Allocate1D<float>(6);
        using var lossBuf = accelerator.Allocate1D<float>(2);
        using var targetsBuf = accelerator.Allocate1D(targets);
        using var gradBuf = accelerator.Allocate1D<float>(6);

        kernels.SoftmaxCrossEntropyForward(logitsBuf.View, probsBuf.View, lossBuf.View, 2, 3);
        kernels.SoftmaxCrossEntropyBackward(probsBuf.View, targetsBuf.View, gradBuf.View, 2, 3);
        await accelerator.SynchronizeAsync();

        var probs = await probsBuf.CopyToHostAsync<float>(0, 6);
        var grad = await gradBuf.CopyToHostAsync<float>(0, 6);

        // Probs should sum to ~1 per sample
        float sum0 = probs[0] + probs[1] + probs[2];
        float sum1 = probs[3] + probs[4] + probs[5];
        if (MathF.Abs(sum0 - 1f) > 0.01f)
            throw new Exception($"Probs sample 0 sum={sum0}, expected ~1.0");
        if (MathF.Abs(sum1 - 1f) > 0.01f)
            throw new Exception($"Probs sample 1 sum={sum1}, expected ~1.0");

        // Gradient for target class should be negative (prob - 1)
        // Gradient for non-target classes should be positive (prob)
        if (grad[0] >= 0)
            throw new Exception($"Grad for target class should be negative: {grad[0]:F4}");
        if (grad[5] >= 0)
            throw new Exception($"Grad for target class should be negative: {grad[5]:F4}");

        // Sum of gradients per sample should be ~0 (softmax gradient property)
        float gradSum0 = grad[0] + grad[1] + grad[2];
        float gradSum1 = grad[3] + grad[4] + grad[5];
        if (MathF.Abs(gradSum0) > 0.01f)
            throw new Exception($"Gradient sum should be ~0: {gradSum0:F4}");
        if (MathF.Abs(gradSum1) > 0.01f)
            throw new Exception($"Gradient sum should be ~0: {gradSum1:F4}");

        Console.WriteLine($"[Training] SoftmaxCE backward: probs OK, gradients OK");
    });

    // ═══════════════════════════════════════════════════════════
    //  ReLU Backward Tests
    // ═══════════════════════════════════════════════════════════

    [TestMethod]
    public async Task Training_ReLU_Backward_ZerosNegativeGradients() => await RunTest(async accelerator =>
    {
        var kernels = new TrainingKernels(accelerator);

        var gradOutput = new float[] { 1f, 2f, 3f, 4f, 5f };
        var forwardInput = new float[] { 0.5f, -0.3f, 1.2f, -0.1f, 0f };

        using var gradOutBuf = accelerator.Allocate1D(gradOutput);
        using var inputBuf = accelerator.Allocate1D(forwardInput);
        using var gradInBuf = accelerator.Allocate1D<float>(5);

        kernels.ReLUBackward(gradOutBuf.View, inputBuf.View, gradInBuf.View, 5);
        await accelerator.SynchronizeAsync();
        var gradIn = await gradInBuf.CopyToHostAsync<float>(0, 5);

        // Where input > 0, gradient passes through; where input <= 0, gradient is zeroed
        if (MathF.Abs(gradIn[0] - 1f) > 1e-6f) throw new Exception($"grad[0]={gradIn[0]}, expected 1.0");
        if (MathF.Abs(gradIn[1]) > 1e-6f) throw new Exception($"grad[1]={gradIn[1]}, expected 0.0");
        if (MathF.Abs(gradIn[2] - 3f) > 1e-6f) throw new Exception($"grad[2]={gradIn[2]}, expected 3.0");
        if (MathF.Abs(gradIn[3]) > 1e-6f) throw new Exception($"grad[3]={gradIn[3]}, expected 0.0");
        if (MathF.Abs(gradIn[4]) > 1e-6f) throw new Exception($"grad[4]={gradIn[4]}, expected 0.0");

        Console.WriteLine("[Training] ReLU backward: correctly zeros negative gradients");
    });

    // ═══════════════════════════════════════════════════════════
    //  SGD Update Tests
    // ═══════════════════════════════════════════════════════════

    [TestMethod]
    public async Task Training_SGD_UpdatesWeightsCorrectly() => await RunTest(async accelerator =>
    {
        var kernels = new TrainingKernels(accelerator);

        var weights = new float[] { 1.0f, 2.0f, 3.0f, 4.0f };
        var grads = new float[] { 0.1f, -0.2f, 0.3f, -0.4f };
        float lr = 0.5f;

        using var wBuf = accelerator.Allocate1D(weights);
        using var gBuf = accelerator.Allocate1D(grads);

        kernels.SGDUpdate(wBuf.View, gBuf.View, 4, lr);
        await accelerator.SynchronizeAsync();
        var updated = await wBuf.CopyToHostAsync<float>(0, 4);

        // w_new = w - lr * grad
        for (int i = 0; i < 4; i++)
        {
            float expected = weights[i] - lr * grads[i];
            if (MathF.Abs(updated[i] - expected) > 1e-5f)
                throw new Exception($"SGD: w[{i}]={updated[i]:F6}, expected {expected:F6}");
        }

        Console.WriteLine("[Training] SGD: weights updated correctly");
    });

    // ═══════════════════════════════════════════════════════════
    //  Linear Backward Tests
    // ═══════════════════════════════════════════════════════════

    [TestMethod]
    public async Task Training_LinearBackward_GradientsMatchCpu() => await RunTest(async accelerator =>
    {
        var kernels = new TrainingKernels(accelerator);

        // Simple case: 1 batch, 2 input features, 3 output features
        // gradOutput [1, 3], weight [3, 2], input [1, 2]
        var gradOutput = new float[] { 1f, 2f, 3f };
        var weight = new float[] { 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f }; // [3, 2]
        var input = new float[] { 1f, 2f };

        using var gradOutBuf = accelerator.Allocate1D(gradOutput);
        using var weightBuf = accelerator.Allocate1D(weight);
        using var inputBuf = accelerator.Allocate1D(input);
        using var gradInputBuf = accelerator.Allocate1D<float>(2);
        using var gradWeightBuf = accelerator.Allocate1D<float>(6);

        kernels.LinearBackwardData(gradOutBuf.View, weightBuf.View, gradInputBuf.View, 1, 3, 2);
        kernels.LinearBackwardWeight(gradOutBuf.View, inputBuf.View, gradWeightBuf.View, 1, 3, 2);
        await accelerator.SynchronizeAsync();

        var gradInput = await gradInputBuf.CopyToHostAsync<float>(0, 2);
        var gradWeight = await gradWeightBuf.CopyToHostAsync<float>(0, 6);

        // gradInput[i] = sum_o(gradOut[o] * weight[o, i])
        // gradInput[0] = 1*0.1 + 2*0.3 + 3*0.5 = 0.1 + 0.6 + 1.5 = 2.2
        // gradInput[1] = 1*0.2 + 2*0.4 + 3*0.6 = 0.2 + 0.8 + 1.8 = 2.8
        if (MathF.Abs(gradInput[0] - 2.2f) > 0.01f)
            throw new Exception($"gradInput[0]={gradInput[0]:F4}, expected 2.2");
        if (MathF.Abs(gradInput[1] - 2.8f) > 0.01f)
            throw new Exception($"gradInput[1]={gradInput[1]:F4}, expected 2.8");

        // gradWeight[o, i] = gradOut[o] * input[i]
        // gradWeight[0,0] = 1*1=1, gradWeight[0,1] = 1*2=2
        // gradWeight[1,0] = 2*1=2, gradWeight[1,1] = 2*2=4
        // gradWeight[2,0] = 3*1=3, gradWeight[2,1] = 3*2=6
        float[] expectedGW = { 1, 2, 2, 4, 3, 6 };
        for (int i = 0; i < 6; i++)
        {
            if (MathF.Abs(gradWeight[i] - expectedGW[i]) > 0.01f)
                throw new Exception($"gradWeight[{i}]={gradWeight[i]:F4}, expected {expectedGW[i]}");
        }

        Console.WriteLine("[Training] Linear backward: data and weight gradients correct");
    });

    // ═══════════════════════════════════════════════════════════
    //  TrainableModel Integration Test
    // ═══════════════════════════════════════════════════════════

    [TestMethod(Timeout = 30000)]
    public async Task Training_MLP_XOR_LossDecreases() => await RunTest(async accelerator =>
    {
        // Train a 2-layer MLP to learn XOR: (0,0)→0, (0,1)→1, (1,0)→1, (1,1)→0
        var model = new TrainableModel(accelerator);
        model.AddLinear(2, 8);
        model.AddReLU();
        model.AddLinear(8, 2);
        model.Build(maxBatchSize: 4);

        // XOR training data (padded to 2 features)
        var inputs = new float[] { 0, 0, 0, 1, 1, 0, 1, 1 };
        var labels = new int[] { 0, 1, 1, 0 };

        float firstLoss = 0, lastLoss = 0;
        for (int epoch = 0; epoch < 100; epoch++)
        {
            float loss = await model.TrainStepAsync(inputs, labels, 4, learningRate: 0.1f);
            if (epoch == 0) firstLoss = loss;
            if (epoch == 99) lastLoss = loss;
        }

        model.Dispose();

        // Loss should decrease over training
        Console.WriteLine($"[Training] XOR MLP: first_loss={firstLoss:F4}, last_loss={lastLoss:F4}");
        if (lastLoss >= firstLoss)
            throw new Exception($"Loss did not decrease: {firstLoss:F4} → {lastLoss:F4}");
    });
}
