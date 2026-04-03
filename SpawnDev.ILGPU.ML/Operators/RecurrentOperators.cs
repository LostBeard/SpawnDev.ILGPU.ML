using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Tensors;

namespace SpawnDev.ILGPU.ML.Operators;

/// <summary>
/// ONNX RNN operator — Simple/Elman recurrent network.
/// Ht = f(Xt * Wi^T + Ht-1 * Ri^T + Wbi + Rbi)
/// Default activation: Tanh. Gate count: 1.
/// Spec: https://onnx.ai/onnx/operators/onnx__RNN.html
/// </summary>
public class RNNOperatorImpl(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "RNN";

    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
    {
        int hiddenSize = attrs.ContainsKey("hidden_size") ? Convert.ToInt32(attrs["hidden_size"]) : inputs[1][1];
        string direction = attrs.ContainsKey("direction") ? attrs["direction"].ToString()! : "forward";
        int numDir = direction == "bidirectional" ? 2 : 1;
        int layout = attrs.ContainsKey("layout") ? Convert.ToInt32(attrs["layout"]) : 0;
        var xShape = inputs[0]; // [seq_len, batch, input_size] or [batch, seq_len, input_size]
        int seqLen = layout == 0 ? xShape[0] : xShape[1];
        int batch = layout == 0 ? xShape[1] : xShape[0];

        var yShape = layout == 0
            ? new[] { seqLen, numDir, batch, hiddenSize }
            : new[] { batch, seqLen, numDir, hiddenSize };
        var yhShape = layout == 0
            ? new[] { numDir, batch, hiddenSize }
            : new[] { batch, numDir, hiddenSize };
        return new[] { yShape, yhShape };
    }

    public void Execute(OnnxOpContext ctx)
    {
        // Inputs: X[0], W[1], R[2], B[3]?, seq_lens[4]?, initial_h[5]?
        var xTensor = ctx.Inputs[0];
        var wVals = ctx.TryGetInputValues(1); // W: [num_dir, hidden_size, input_size]
        var rVals = ctx.TryGetInputValues(2); // R: [num_dir, hidden_size, hidden_size]
        if (wVals == null || rVals == null) return;

        float[]? bVals = ctx.Inputs.Length > 3 ? ctx.TryGetInputValues(3) : null;
        float[]? initHVals = ctx.Inputs.Length > 5 ? ctx.TryGetInputValues(5) : null;

        int hiddenSize = ctx.GetInt("hidden_size", 0);
        string direction = ctx.GetString("direction", "forward");
        int layout = ctx.GetInt("layout", 0);
        int numDir = direction == "bidirectional" ? 2 : 1;

        var xShape = xTensor.Shape;
        int seqLen = layout == 0 ? xShape[0] : xShape[1];
        int batch = layout == 0 ? xShape[1] : xShape[0];
        int inputSize = xShape[2];
        if (hiddenSize == 0) hiddenSize = wVals.Length / (numDir * inputSize);

        // Read X to CPU (recurrent ops are sequential — CPU execution is practical for inference)
        var xVals = ctx.TryGetInputValues(0);
        if (xVals == null) return;

        // Compute bias = Wb + Rb (split B in half, add element-wise)
        var bias = new float[numDir * hiddenSize];
        if (bVals != null)
        {
            for (int d = 0; d < numDir; d++)
            {
                int bOff = d * 2 * hiddenSize;
                for (int h = 0; h < hiddenSize; h++)
                    bias[d * hiddenSize + h] = bVals[bOff + h] + bVals[bOff + hiddenSize + h];
            }
        }

        // Output buffers
        int yTotal = seqLen * numDir * batch * hiddenSize;
        var yData = new float[yTotal];
        var yhData = new float[numDir * batch * hiddenSize];

        for (int dir = 0; dir < numDir; dir++)
        {
            int wOff = dir * hiddenSize * inputSize;
            int rOff = dir * hiddenSize * hiddenSize;
            int bOff = dir * hiddenSize;

            // Initialize hidden state
            var h = new float[batch * hiddenSize];
            if (initHVals != null)
                Array.Copy(initHVals, dir * batch * hiddenSize, h, 0, batch * hiddenSize);

            bool reverse = (dir == 1) || direction == "reverse";

            for (int t = 0; t < seqLen; t++)
            {
                int timeIdx = reverse ? seqLen - 1 - t : t;

                for (int b = 0; b < batch; b++)
                {
                    // Get x for this timestep+batch
                    int xOff = layout == 0
                        ? (timeIdx * batch + b) * inputSize
                        : (b * seqLen + timeIdx) * inputSize;

                    for (int hi = 0; hi < hiddenSize; hi++)
                    {
                        float val = bias[bOff + hi];
                        // Xt * Wi^T
                        for (int xi = 0; xi < inputSize; xi++)
                            val += xVals[xOff + xi] * wVals[wOff + hi * inputSize + xi];
                        // Ht-1 * Ri^T
                        for (int hj = 0; hj < hiddenSize; hj++)
                            val += h[b * hiddenSize + hj] * rVals[rOff + hi * hiddenSize + hj];
                        // Activation (default: Tanh)
                        h[b * hiddenSize + hi] = MathF.Tanh(val);
                    }
                }

                // Store Y for this timestep
                for (int b = 0; b < batch; b++)
                {
                    int yOff = layout == 0
                        ? ((timeIdx * numDir + dir) * batch + b) * hiddenSize
                        : ((b * seqLen + timeIdx) * numDir + dir) * hiddenSize;
                    Array.Copy(h, b * hiddenSize, yData, yOff, hiddenSize);
                }
            }

            // Store Y_h (final hidden state)
            Array.Copy(h, 0, yhData, dir * batch * hiddenSize, batch * hiddenSize);
        }

        // Upload results to GPU
        if (ctx.Outputs.Length > 0 && ctx.Outputs[0] != null)
        {
            using var yBuf = reg.Accelerator.Allocate1D(yData);
            int copyLen = Math.Min(yTotal, ctx.Outputs[0].ElementCount);
            reg.ElementWise.Scale(yBuf.View.SubView(0, copyLen), ctx.Outputs[0].Data.SubView(0, copyLen), copyLen, 1f);
        }
        if (ctx.Outputs.Length > 1 && ctx.Outputs[1] != null)
        {
            using var yhBuf = reg.Accelerator.Allocate1D(yhData);
            int copyLen = Math.Min(yhData.Length, ctx.Outputs[1].ElementCount);
            reg.ElementWise.Scale(yhBuf.View.SubView(0, copyLen), ctx.Outputs[1].Data.SubView(0, copyLen), copyLen, 1f);
        }
    }
}

/// <summary>
/// ONNX LSTM operator — Long Short-Term Memory.
/// Gates: it (input), ft (forget), ct (cell), ot (output)
/// Gate ordering in W/R: [i, o, f, c] (ONNX spec, differs from PyTorch [i, f, g, o])
/// Spec: https://onnx.ai/onnx/operators/onnx__LSTM.html
/// </summary>
public class LSTMOperatorImpl(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "LSTM";

    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
    {
        int hiddenSize = attrs.ContainsKey("hidden_size") ? Convert.ToInt32(attrs["hidden_size"]) : inputs[1][1] / 4;
        string direction = attrs.ContainsKey("direction") ? attrs["direction"].ToString()! : "forward";
        int numDir = direction == "bidirectional" ? 2 : 1;
        int layout = attrs.ContainsKey("layout") ? Convert.ToInt32(attrs["layout"]) : 0;
        var xShape = inputs[0];
        int seqLen = layout == 0 ? xShape[0] : xShape[1];
        int batch = layout == 0 ? xShape[1] : xShape[0];

        var yShape = layout == 0
            ? new[] { seqLen, numDir, batch, hiddenSize }
            : new[] { batch, seqLen, numDir, hiddenSize };
        var yhShape = layout == 0
            ? new[] { numDir, batch, hiddenSize }
            : new[] { batch, numDir, hiddenSize };
        return new[] { yShape, yhShape, yhShape }; // Y, Y_h, Y_c
    }

    public void Execute(OnnxOpContext ctx)
    {
        var wVals = ctx.TryGetInputValues(1); // W: [num_dir, 4*hidden_size, input_size]
        var rVals = ctx.TryGetInputValues(2); // R: [num_dir, 4*hidden_size, hidden_size]
        if (wVals == null || rVals == null) return;

        float[]? bVals = ctx.Inputs.Length > 3 ? ctx.TryGetInputValues(3) : null;
        float[]? initHVals = ctx.Inputs.Length > 5 ? ctx.TryGetInputValues(5) : null;
        float[]? initCVals = ctx.Inputs.Length > 6 ? ctx.TryGetInputValues(6) : null;
        float[]? pVals = ctx.Inputs.Length > 7 ? ctx.TryGetInputValues(7) : null;

        int hiddenSize = ctx.GetInt("hidden_size", 0);
        string direction = ctx.GetString("direction", "forward");
        int layout = ctx.GetInt("layout", 0);
        int numDir = direction == "bidirectional" ? 2 : 1;

        var xShape = ctx.Inputs[0].Shape;
        int seqLen = layout == 0 ? xShape[0] : xShape[1];
        int batch = layout == 0 ? xShape[1] : xShape[0];
        int inputSize = xShape[2];
        if (hiddenSize == 0) hiddenSize = wVals.Length / (numDir * 4 * inputSize);

        var xVals = ctx.TryGetInputValues(0);
        if (xVals == null) return;

        // Combine bias: Wb + Rb (split B in half, add element-wise)
        var bias = new float[numDir * 4 * hiddenSize];
        if (bVals != null)
        {
            for (int d = 0; d < numDir; d++)
            {
                int bOff = d * 8 * hiddenSize;
                for (int g = 0; g < 4 * hiddenSize; g++)
                    bias[d * 4 * hiddenSize + g] = bVals[bOff + g] + bVals[bOff + 4 * hiddenSize + g];
            }
        }

        int yTotal = seqLen * numDir * batch * hiddenSize;
        var yData = new float[yTotal];
        var yhData = new float[numDir * batch * hiddenSize];
        var ycData = new float[numDir * batch * hiddenSize];

        for (int dir = 0; dir < numDir; dir++)
        {
            int H = hiddenSize;
            int wOff = dir * 4 * H * inputSize;
            int rOff = dir * 4 * H * H;
            int bOff = dir * 4 * H;

            var ht = new float[batch * H];
            var ct = new float[batch * H];
            if (initHVals != null) Array.Copy(initHVals, dir * batch * H, ht, 0, batch * H);
            if (initCVals != null) Array.Copy(initCVals, dir * batch * H, ct, 0, batch * H);

            // Peepholes: P = [P_i, P_o, P_f], each of size H
            float[]? pi = null, po = null, pf = null;
            if (pVals != null)
            {
                pi = new float[H]; po = new float[H]; pf = new float[H];
                Array.Copy(pVals, dir * 3 * H, pi, 0, H);
                Array.Copy(pVals, dir * 3 * H + H, po, 0, H);
                Array.Copy(pVals, dir * 3 * H + 2 * H, pf, 0, H);
            }

            bool reverse = (dir == 1) || direction == "reverse";

            for (int t = 0; t < seqLen; t++)
            {
                int timeIdx = reverse ? seqLen - 1 - t : t;

                for (int b = 0; b < batch; b++)
                {
                    int xOff = layout == 0
                        ? (timeIdx * batch + b) * inputSize
                        : (b * seqLen + timeIdx) * inputSize;

                    // Compute 4 gates: [i, o, f, c] — ONNX gate ordering
                    var gates = new float[4 * H];
                    for (int g = 0; g < 4 * H; g++)
                    {
                        float val = bias[bOff + g];
                        // Xt * W^T
                        for (int xi = 0; xi < inputSize; xi++)
                            val += xVals[xOff + xi] * wVals[wOff + g * inputSize + xi];
                        // Ht-1 * R^T
                        for (int hj = 0; hj < H; hj++)
                            val += ht[b * H + hj] * rVals[rOff + g * H + hj];
                        gates[g] = val;
                    }

                    // Split gates: i=[0..H), o=[H..2H), f=[2H..3H), c=[3H..4H)
                    for (int hi = 0; hi < H; hi++)
                    {
                        float gi = gates[hi];          // input gate
                        float go = gates[H + hi];       // output gate
                        float gf = gates[2 * H + hi];   // forget gate
                        float gc = gates[3 * H + hi];   // cell gate

                        // Peephole connections
                        float cPrev = ct[b * H + hi];
                        if (pi != null) gi += pi[hi] * cPrev;
                        if (pf != null) gf += pf[hi] * cPrev;

                        // Activations: i,o,f = sigmoid; c = tanh
                        float it = 1f / (1f + MathF.Exp(-gi));
                        float ft = 1f / (1f + MathF.Exp(-gf));
                        float cellCandidate = MathF.Tanh(gc);

                        // Cell update: Ct = ft * Ct-1 + it * ct
                        float newC = ft * cPrev + it * cellCandidate;
                        ct[b * H + hi] = newC;

                        // Output gate with peephole on NEW cell state
                        if (po != null) go += po[hi] * newC;
                        float ot = 1f / (1f + MathF.Exp(-go));

                        // Hidden state: Ht = ot * tanh(Ct)
                        ht[b * H + hi] = ot * MathF.Tanh(newC);
                    }
                }

                // Store Y
                for (int b = 0; b < batch; b++)
                {
                    int yOff = layout == 0
                        ? ((timeIdx * numDir + dir) * batch + b) * H
                        : ((b * seqLen + timeIdx) * numDir + dir) * H;
                    Array.Copy(ht, b * H, yData, yOff, H);
                }
            }

            Array.Copy(ht, 0, yhData, dir * batch * H, batch * H);
            Array.Copy(ct, 0, ycData, dir * batch * H, batch * H);
        }

        // Upload Y
        if (ctx.Outputs.Length > 0 && ctx.Outputs[0] != null)
        {
            using var buf = reg.Accelerator.Allocate1D(yData);
            int n = Math.Min(yTotal, ctx.Outputs[0].ElementCount);
            reg.ElementWise.Scale(buf.View.SubView(0, n), ctx.Outputs[0].Data.SubView(0, n), n, 1f);
        }
        // Upload Y_h
        if (ctx.Outputs.Length > 1 && ctx.Outputs[1] != null)
        {
            using var buf = reg.Accelerator.Allocate1D(yhData);
            int n = Math.Min(yhData.Length, ctx.Outputs[1].ElementCount);
            reg.ElementWise.Scale(buf.View.SubView(0, n), ctx.Outputs[1].Data.SubView(0, n), n, 1f);
        }
        // Upload Y_c (LSTM only)
        if (ctx.Outputs.Length > 2 && ctx.Outputs[2] != null)
        {
            using var buf = reg.Accelerator.Allocate1D(ycData);
            int n = Math.Min(ycData.Length, ctx.Outputs[2].ElementCount);
            reg.ElementWise.Scale(buf.View.SubView(0, n), ctx.Outputs[2].Data.SubView(0, n), n, 1f);
        }
    }
}

/// <summary>
/// ONNX GRU operator — Gated Recurrent Unit.
/// Gates: zt (update), rt (reset), ht (hidden)
/// Gate ordering in W/R: [z, r, h]
/// linear_before_reset attribute changes how reset gate interacts with bias.
/// Spec: https://onnx.ai/onnx/operators/onnx__GRU.html
/// </summary>
public class GRUOperatorImpl(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "GRU";

    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
    {
        int hiddenSize = attrs.ContainsKey("hidden_size") ? Convert.ToInt32(attrs["hidden_size"]) : inputs[1][1] / 3;
        string direction = attrs.ContainsKey("direction") ? attrs["direction"].ToString()! : "forward";
        int numDir = direction == "bidirectional" ? 2 : 1;
        int layout = attrs.ContainsKey("layout") ? Convert.ToInt32(attrs["layout"]) : 0;
        var xShape = inputs[0];
        int seqLen = layout == 0 ? xShape[0] : xShape[1];
        int batch = layout == 0 ? xShape[1] : xShape[0];

        var yShape = layout == 0
            ? new[] { seqLen, numDir, batch, hiddenSize }
            : new[] { batch, seqLen, numDir, hiddenSize };
        var yhShape = layout == 0
            ? new[] { numDir, batch, hiddenSize }
            : new[] { batch, numDir, hiddenSize };
        return new[] { yShape, yhShape };
    }

    public void Execute(OnnxOpContext ctx)
    {
        var wVals = ctx.TryGetInputValues(1); // W: [num_dir, 3*hidden_size, input_size]
        var rVals = ctx.TryGetInputValues(2); // R: [num_dir, 3*hidden_size, hidden_size]
        if (wVals == null || rVals == null) return;

        float[]? bVals = ctx.Inputs.Length > 3 ? ctx.TryGetInputValues(3) : null;
        float[]? initHVals = ctx.Inputs.Length > 5 ? ctx.TryGetInputValues(5) : null;

        int hiddenSize = ctx.GetInt("hidden_size", 0);
        string direction = ctx.GetString("direction", "forward");
        int layout = ctx.GetInt("layout", 0);
        int linearBeforeReset = ctx.GetInt("linear_before_reset", 0);
        int numDir = direction == "bidirectional" ? 2 : 1;

        var xShape = ctx.Inputs[0].Shape;
        int seqLen = layout == 0 ? xShape[0] : xShape[1];
        int batch = layout == 0 ? xShape[1] : xShape[0];
        int inputSize = xShape[2];
        if (hiddenSize == 0) hiddenSize = wVals.Length / (numDir * 3 * inputSize);

        var xVals = ctx.TryGetInputValues(0);
        if (xVals == null) return;

        // Split bias: B = [Wb_z, Wb_r, Wb_h, Rb_z, Rb_r, Rb_h]
        var wbz = new float[numDir * hiddenSize]; var wbr = new float[numDir * hiddenSize];
        var wbh = new float[numDir * hiddenSize]; var rbz = new float[numDir * hiddenSize];
        var rbr = new float[numDir * hiddenSize]; var rbh = new float[numDir * hiddenSize];
        if (bVals != null)
        {
            int H = hiddenSize;
            for (int d = 0; d < numDir; d++)
            {
                int off = d * 6 * H;
                Array.Copy(bVals, off, wbz, d * H, H);
                Array.Copy(bVals, off + H, wbr, d * H, H);
                Array.Copy(bVals, off + 2 * H, wbh, d * H, H);
                Array.Copy(bVals, off + 3 * H, rbz, d * H, H);
                Array.Copy(bVals, off + 4 * H, rbr, d * H, H);
                Array.Copy(bVals, off + 5 * H, rbh, d * H, H);
            }
        }

        int H2 = hiddenSize;
        int yTotal = seqLen * numDir * batch * H2;
        var yData = new float[yTotal];
        var yhData = new float[numDir * batch * H2];

        for (int dir = 0; dir < numDir; dir++)
        {
            int wOff = dir * 3 * H2 * inputSize;
            int rOff = dir * 3 * H2 * H2;

            var ht = new float[batch * H2];
            if (initHVals != null) Array.Copy(initHVals, dir * batch * H2, ht, 0, batch * H2);

            bool reverse = (dir == 1) || direction == "reverse";

            for (int t = 0; t < seqLen; t++)
            {
                int timeIdx = reverse ? seqLen - 1 - t : t;

                for (int b = 0; b < batch; b++)
                {
                    int xOff = layout == 0
                        ? (timeIdx * batch + b) * inputSize
                        : (b * seqLen + timeIdx) * inputSize;

                    for (int hi = 0; hi < H2; hi++)
                    {
                        // z gate
                        float z = wbz[dir * H2 + hi] + rbz[dir * H2 + hi];
                        for (int xi = 0; xi < inputSize; xi++)
                            z += xVals[xOff + xi] * wVals[wOff + hi * inputSize + xi];
                        for (int hj = 0; hj < H2; hj++)
                            z += ht[b * H2 + hj] * rVals[rOff + hi * H2 + hj];
                        z = 1f / (1f + MathF.Exp(-z)); // sigmoid

                        // r gate
                        float r = wbr[dir * H2 + hi] + rbr[dir * H2 + hi];
                        for (int xi = 0; xi < inputSize; xi++)
                            r += xVals[xOff + xi] * wVals[wOff + (H2 + hi) * inputSize + xi];
                        for (int hj = 0; hj < H2; hj++)
                            r += ht[b * H2 + hj] * rVals[rOff + (H2 + hi) * H2 + hj];
                        r = 1f / (1f + MathF.Exp(-r)); // sigmoid

                        // h gate (depends on linear_before_reset)
                        float h;
                        float xh = wbh[dir * H2 + hi];
                        for (int xi = 0; xi < inputSize; xi++)
                            xh += xVals[xOff + xi] * wVals[wOff + (2 * H2 + hi) * inputSize + xi];

                        if (linearBeforeReset != 0)
                        {
                            // Linear before reset: r * (Ht-1 * Rh^T + Rbh)
                            float rh = rbh[dir * H2 + hi];
                            for (int hj = 0; hj < H2; hj++)
                                rh += ht[b * H2 + hj] * rVals[rOff + (2 * H2 + hi) * H2 + hj];
                            h = MathF.Tanh(xh + r * rh);
                        }
                        else
                        {
                            // Default: (r * Ht-1) * Rh^T + Rbh
                            float rh = rbh[dir * H2 + hi];
                            for (int hj = 0; hj < H2; hj++)
                                rh += (r * ht[b * H2 + hj]) * rVals[rOff + (2 * H2 + hi) * H2 + hj];
                            h = MathF.Tanh(xh + rh);
                        }

                        // Ht = (1 - zt) * ht + zt * Ht-1
                        ht[b * H2 + hi] = (1f - z) * h + z * ht[b * H2 + hi];
                    }
                }

                // Store Y
                for (int b = 0; b < batch; b++)
                {
                    int yOff = layout == 0
                        ? ((timeIdx * numDir + dir) * batch + b) * H2
                        : ((b * seqLen + timeIdx) * numDir + dir) * H2;
                    Array.Copy(ht, b * H2, yData, yOff, H2);
                }
            }

            Array.Copy(ht, 0, yhData, dir * batch * H2, batch * H2);
        }

        if (ctx.Outputs.Length > 0 && ctx.Outputs[0] != null)
        {
            using var buf = reg.Accelerator.Allocate1D(yData);
            int n = Math.Min(yTotal, ctx.Outputs[0].ElementCount);
            reg.ElementWise.Scale(buf.View.SubView(0, n), ctx.Outputs[0].Data.SubView(0, n), n, 1f);
        }
        if (ctx.Outputs.Length > 1 && ctx.Outputs[1] != null)
        {
            using var buf = reg.Accelerator.Allocate1D(yhData);
            int n = Math.Min(yhData.Length, ctx.Outputs[1].ElementCount);
            reg.ElementWise.Scale(buf.View.SubView(0, n), ctx.Outputs[1].Data.SubView(0, n), n, 1f);
        }
    }
}
