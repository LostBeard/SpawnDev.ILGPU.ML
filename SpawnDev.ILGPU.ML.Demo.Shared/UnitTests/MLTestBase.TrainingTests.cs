using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Training;
using SpawnDev.ILGPU.ML.Demo.Shared.Games.DrawToLearn;
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

        kernels.SoftmaxCrossEntropyForward(logitsBuf.View, probsBuf.View, lossBuf.View, targetsBuf.View, 2, 3);
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
    //  Adam Optimizer Tests
    // ═══════════════════════════════════════════════════════════

    [TestMethod]
    public async Task Training_Adam_UpdatesWeightsAndMoments() => await RunTest(async accelerator =>
    {
        var kernels = new TrainingKernels(accelerator);

        var weights = new float[] { 1.0f, 2.0f, 3.0f };
        var grads = new float[] { 0.1f, -0.2f, 0.3f };

        using var wBuf = accelerator.Allocate1D(weights);
        using var gBuf = accelerator.Allocate1D(grads);
        using var mBuf = accelerator.Allocate1D<float>(3); // first moment (zeros)
        using var vBuf = accelerator.Allocate1D<float>(3); // second moment (zeros)

        // Run 3 Adam steps
        for (int step = 1; step <= 3; step++)
            kernels.AdamUpdate(wBuf.View, gBuf.View, mBuf.View, vBuf.View, 3, 0.001f, 0.9f, 0.999f, step);

        await accelerator.SynchronizeAsync();
        var updated = await wBuf.CopyToHostAsync<float>(0, 3);
        var m = await mBuf.CopyToHostAsync<float>(0, 3);
        var v = await vBuf.CopyToHostAsync<float>(0, 3);

        // Moments should be non-zero after updates
        if (m.All(x => x == 0f))
            throw new Exception("First moment m should be non-zero after Adam steps");
        if (v.All(x => x == 0f))
            throw new Exception("Second moment v should be non-zero after Adam steps");

        // Weights should have changed
        bool changed = false;
        for (int i = 0; i < 3; i++)
            if (MathF.Abs(updated[i] - weights[i]) > 1e-6f) changed = true;
        if (!changed)
            throw new Exception("Weights should change after Adam updates");

        Console.WriteLine($"[Training] Adam: weights updated, moments accumulated, 3 steps OK");
    });

    // ═══════════════════════════════════════════════════════════
    //  Zero Kernel Tests
    // ═══════════════════════════════════════════════════════════

    [TestMethod]
    public async Task Training_Zero_ClearsBuffer() => await RunTest(async accelerator =>
    {
        var kernels = new TrainingKernels(accelerator);

        var data = new float[] { 1f, 2f, 3f, 4f, 5f };
        using var buf = accelerator.Allocate1D(data);

        kernels.Zero(buf.View, 5);
        await accelerator.SynchronizeAsync();
        var result = await buf.CopyToHostAsync<float>(0, 5);

        for (int i = 0; i < 5; i++)
        {
            if (result[i] != 0f)
                throw new Exception($"Zero: buf[{i}]={result[i]}, expected 0.0");
        }

        Console.WriteLine("[Training] Zero: buffer cleared to all zeros");
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

    // ═══════════════════════════════════════════════════════════
    //  Draw to Learn (/train): the page's own code, on hand-drawn-like shapes
    // ═══════════════════════════════════════════════════════════

    /// <summary>
    /// The /train page end to end, minus the mouse: 6 drawings each of a circle, a square and a triangle (wobbly strokes, random place, size,
    /// rotation and aspect, drawn with the page's 12 px round brush on its 280 x 280 canvas), prepared and trained by <c>DrawToLearn</c>
    /// exactly as the page does, then 40 NEW drawings per class classified. Gate: 90% right. Also prints what the page's former recipe
    /// (plain 10 x 10 average downsample, 784-64-N, 50 SGD steps at 0.01) scores on the same drawings.
    /// </summary>
    [TestMethod(Timeout = 120000)]
    public async Task Training_DrawToLearn_ClassifiesNewDrawings() => await RunTest(async accelerator =>
    {
        const int classes = 3, trainPer = 6, testPer = 40, W = 280;
        var rng = new Random(7);
        byte[] Draw(int shape)
        {
            var px = new byte[W * W * 4];
            void Stamp(float x, float y)
            {
                for (int yy = (int)(y - 6); yy <= (int)(y + 6); yy++)
                    for (int xx = (int)(x - 6); xx <= (int)(x + 6); xx++)
                        if ((uint)xx < W && (uint)yy < W && (xx - x) * (xx - x) + (yy - y) * (yy - y) <= 36)
                            for (int ch = 0; ch < 4; ch++) px[(yy * W + xx) * 4 + ch] = 255;
            }
            void Line(float x0, float y0, float x1, float y1)
            {
                int steps = Math.Max(1, (int)(MathF.Sqrt((x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0)) / 2));
                for (int i = 0; i <= steps; i++) Stamp(x0 + (x1 - x0) * i / steps, y0 + (y1 - y0) * i / steps);
            }
            float R(float a, float b) => a + (float)rng.NextDouble() * (b - a);
            float size = R(70, 210), cx = R(size / 2 + 8, W - size / 2 - 8), cy = R(size / 2 + 8, W - size / 2 - 8);
            float aspect = R(0.8f, 1.2f), rot = R(-0.25f, 0.25f), wob = R(0, 6.28f);
            var pts = new List<(float X, float Y)>();
            if (shape == 0)
            {
                float start = R(0, 6.28f), sweep = 6.28f + R(-0.35f, 0.35f);
                for (int i = 0; i <= 48; i++)
                {
                    float t = start + sweep * i / 48, r = size / 2 * (1 + 0.05f * MathF.Sin(3 * t + wob));
                    pts.Add((r * MathF.Cos(t) * aspect, r * MathF.Sin(t) / aspect));
                }
            }
            else
            {
                int n = shape == 1 ? 4 : 3;
                var corners = new List<(float X, float Y)>();
                for (int i = 0; i < n; i++)
                {
                    float t = (shape == 1 ? MathF.PI / 4 : -MathF.PI / 2) + i * 2 * MathF.PI / n;
                    float r = size / 2 * (shape == 1 ? 1.41f : 1.15f) * 0.75f;
                    corners.Add((r * MathF.Cos(t) * aspect + R(-8, 8), r * MathF.Sin(t) / aspect + R(-8, 8)));
                }
                for (int i = 0; i < n; i++)
                {
                    var a = corners[i];
                    var b = corners[(i + 1) % n];
                    for (int j = 0; j < 12; j++) { float f = j / 12f; pts.Add((a.X + (b.X - a.X) * f + R(-2, 2), a.Y + (b.Y - a.Y) * f + R(-2, 2))); }
                }
                pts.Add(corners[0]);
            }
            float cr = MathF.Cos(rot), sr = MathF.Sin(rot);
            for (int i = 0; i + 1 < pts.Count; i++)
                Line(cx + cr * pts[i].X - sr * pts[i].Y, cy + sr * pts[i].X + cr * pts[i].Y,
                     cx + cr * pts[i + 1].X - sr * pts[i + 1].Y, cy + sr * pts[i + 1].X + cr * pts[i + 1].Y);
            return px;
        }
        var train = Enumerable.Range(0, classes).Select(c => Enumerable.Range(0, trainPer).Select(_ => Draw(c)).ToList()).ToList();
        var test = Enumerable.Range(0, classes).Select(c => Enumerable.Range(0, testPer).Select(_ => Draw(c)).ToList()).ToList();
        int nTest = classes * testPer;

        async Task<double> Accuracy(TrainableModel model, Func<byte[], float[]> prep)
        {
            var x = new float[nTest * 784];
            int i = 0;
            for (int c = 0; c < classes; c++)
                foreach (var d in test[c]) Array.Copy(prep(d), 0, x, (i++) * 784, 784);
            var logits = await model.PredictAsync(x, nTest);
            int right = 0;
            var confusion = new int[classes * classes];
            for (int j = 0; j < nTest; j++)
            {
                int best = 0;
                for (int k = 1; k < classes; k++) if (logits[j * classes + k] > logits[j * classes + best]) best = k;
                if (best == j / testPer) right++;
                confusion[j / testPer * classes + best]++;
            }
            Console.WriteLine($"[DrawToLearn] confusion (row = drawn circle / square / triangle): {string.Join(" | ", Enumerable.Range(0, classes).Select(r => string.Join(" ", confusion.Skip(r * classes).Take(classes))))}");
            return right / (double)nTest;
        }

        // the page's former recipe
        float[] Legacy(byte[] px)
        {
            var o = new float[784];
            for (int y = 0; y < 28; y++)
                for (int x = 0; x < 28; x++)
                {
                    float sum = 0;
                    for (int dy = 0; dy < 10; dy++) for (int dx = 0; dx < 10; dx++) sum += px[((y * 10 + dy) * W + x * 10 + dx) * 4];
                    o[y * 28 + x] = sum / (100 * 255f);
                }
            return o;
        }
        double legacy;
        using (var old = new TrainableModel(accelerator))
        {
            old.AddLinear(784, 64); old.AddReLU(); old.AddLinear(64, classes);
            int n = classes * trainPer;
            old.Build(maxBatchSize: Math.Max(n, nTest));
            var x = new float[n * 784];
            var y = new int[n];
            int i = 0;
            for (int c = 0; c < classes; c++) foreach (var d in train[c]) { Array.Copy(Legacy(d), 0, x, i * 784, 784); y[i++] = c; }
            for (int e = 0; e < 50; e++) await old.TrainStepAsync(x, y, n, 0.01f, readLoss: e == 49);
            legacy = await Accuracy(old, Legacy);
        }

        var sw = System.Diagnostics.Stopwatch.StartNew();
        var samples = train.Select(c => (IReadOnlyList<float[]>)c.Select(d => DrawToLearn.Prepare(d, W, W)!).ToList()).ToList();
        using var model = DrawToLearn.CreateModel(accelerator, classes, Math.Max(DrawToLearn.BatchSize(classes * trainPer), nTest));
        float firstLoss = -1;
        float loss = await DrawToLearn.TrainAsync(model, samples, progress: (step, l) => { if (firstLoss < 0) firstLoss = l; return Task.CompletedTask; });
        long trainMs = sw.ElapsedMilliseconds;
        double acc = await Accuracy(model, d => DrawToLearn.Prepare(d, W, W)!);
        Console.WriteLine($"[DrawToLearn] {classes} classes x {trainPer} drawings: loss at step 10 {firstLoss:F3} -> {loss:F4} in {trainMs} ms; held-out accuracy {acc:P1} (former recipe {legacy:P1})");
        if (acc < 0.9)
            throw new Exception($"Draw to Learn classified {acc:P1} of {nTest} new drawings (need 90%); former recipe {legacy:P1}");
    });
}
