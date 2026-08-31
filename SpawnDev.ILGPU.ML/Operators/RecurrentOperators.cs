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

    /// <summary>Cached host copies of the STATIC inputs (W, R, B), keyed by tensor name.</summary>
    private readonly Dictionary<string, float[]> _weightCache = new(StringComparer.Ordinal);

    /// <summary>Weights and bias. Static for the session, so worth caching.</summary>
    private static readonly int[] StaticInputs = { 1, 2, 3 };

    /// <summary>
    /// Per-call inputs: X and <c>initial_h</c>. ⚠️ <c>initial_h</c> must NOT be cached - a model that
    /// threads its recurrent state through graph inputs (Silero VAD does) would have that state frozen at
    /// the first frame.
    /// </summary>
    private static readonly int[] DynamicInputs = { 0, 5 };

    public void Execute(OnnxOpContext ctx) => ExecuteCore(ctx, null);

    /// <summary>Browser-safe path: the sync GPU readback throws on WebGPU/WebGL/Wasm.</summary>
    public async Task ExecuteAsync(OnnxOpContext ctx)
    {
        var pre = new Dictionary<int, float[]>();
        foreach (int i in DynamicInputs)
        {
            var v = await OperatorInputReader.ReadAsync(reg, ctx, i);
            if (v != null) pre[i] = v;
        }
        foreach (int i in StaticInputs)
        {
            var v = await OperatorInputReader.ReadCachedAsync(reg, ctx, i, _weightCache);
            if (v != null) pre[i] = v;
        }
        ExecuteCore(ctx, pre);
    }

    private void ExecuteCore(OnnxOpContext ctx, Dictionary<int, float[]>? pre)
    {
        // ⚠️ ctx.TryGetInputValues returns COMPILE-TIME CONSTANTS ONLY. X is the runtime input, so it read
        // as null on every real inference and this operator returned having computed NOTHING while being
        // advertised in BuiltinOpTypes. See OperatorInputReader.
        float[]? Get(int i) => pre != null && pre.TryGetValue(i, out var p) ? p
            : Array.IndexOf(StaticInputs, i) >= 0
                ? OperatorInputReader.ReadCached(reg, ctx, i, _weightCache)
                : OperatorInputReader.Read(reg, ctx, i);

        // Inputs: X[0], W[1], R[2], B[3]?, seq_lens[4]?, initial_h[5]?
        var xTensor = ctx.Inputs[0];
        var wVals = Get(1); // W: [num_dir, hidden_size, input_size]
        var rVals = Get(2); // R: [num_dir, hidden_size, hidden_size]
        if (wVals == null || rVals == null)
            throw new NotSupportedException(
                "RNN could not read its W/R weights. On a browser backend this operator needs the "
                + "async path (GraphExecutor.RunAsync); the synchronous GPU readback is "
                + "unavailable there. Returning silently is what made this a no-op.");

        float[]? bVals = ctx.Inputs.Length > 3 ? Get(3) : null;
        float[]? initHVals = ctx.Inputs.Length > 5 ? Get(5) : null;

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
        var xVals = Get(0);
        if (xVals == null)
            throw new NotSupportedException(
                "RNN could not read its input X. On a browser backend this operator needs the "
                + "async path (GraphExecutor.RunAsync); the synchronous GPU readback is "
                + "unavailable there. Returning silently is what made this a no-op.");

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

                    // ⚠️ Ht is computed from ALL of Ht-1, so it cannot be written in place: the old
                    // code assigned h[hi] inside this loop while later hi still read h[hj] for the
                    // recurrence, so every unit after the first saw the NEW timestep's values mixed into
                    // the previous state. Measured against onnxruntime: index 0 correct, index 1 onward
                    // wrong (max |d| 2.245E-001 on a 4-unit layer).
                    var hNew = new float[hiddenSize];
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
                        hNew[hi] = MathF.Tanh(val);
                    }
                    Array.Copy(hNew, 0, h, b * hiddenSize, hiddenSize);
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
            int copyLen = Math.Min(yTotal, ctx.Outputs[0].ElementCount);
            ctx.Outputs[0].Data.SubView(0, copyLen).CopyFromCPU(yData.AsSpan(0, copyLen).ToArray());
        }
        if (ctx.Outputs.Length > 1 && ctx.Outputs[1] != null)
        {
            int copyLen = Math.Min(yhData.Length, ctx.Outputs[1].ElementCount);
            ctx.Outputs[1].Data.SubView(0, copyLen).CopyFromCPU(yhData.AsSpan(0, copyLen).ToArray());
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

    /// <summary>Cached host copies of the STATIC inputs (W, R, B, P), keyed by tensor name.</summary>
    private readonly Dictionary<string, float[]> _weightCache = new(StringComparer.Ordinal);

    /// <summary>Static inputs: weights, recurrence, bias, peepholes. Safe to cache for the session.</summary>
    private static readonly int[] StaticInputs = { 1, 2, 3, 7 };

    /// <summary>
    /// Per-call inputs. ⚠️ <c>initial_h</c> (5) and <c>initial_c</c> (6) belong HERE, not in
    /// <see cref="StaticInputs"/>: Silero VAD passes its LSTM state in as graph inputs and expects the new
    /// state back each frame, so caching them would freeze the detector's memory at the first frame.
    /// </summary>
    private static readonly int[] DynamicInputs = { 0, 5, 6 };

    public void Execute(OnnxOpContext ctx) => ExecuteCore(ctx, null);

    /// <summary>
    /// Browser-safe path: reads the dynamic inputs through <c>CopyToHostAsync</c> before computing.
    /// </summary>
    /// <remarks>
    /// The sync readback throws on WebGPU/WebGL/Wasm, so without this the operator would see null for X
    /// there and be right back to producing nothing.
    /// </remarks>
    public async Task ExecuteAsync(OnnxOpContext ctx)
    {
        var pre = new Dictionary<int, float[]>();
        foreach (int i in DynamicInputs)
        {
            var v = await OperatorInputReader.ReadAsync(reg, ctx, i);
            if (v != null) pre[i] = v;
        }
        foreach (int i in StaticInputs)
        {
            var v = await OperatorInputReader.ReadCachedAsync(reg, ctx, i, _weightCache);
            if (v != null) pre[i] = v;
        }
        ExecuteCore(ctx, pre);
    }

    private void ExecuteCore(OnnxOpContext ctx, Dictionary<int, float[]>? pre)
    {
        // ⚠️ These used to be ctx.TryGetInputValues, which returns COMPILE-TIME CONSTANTS ONLY. X is the
        // runtime input, so it was always null and the operator returned having computed NOTHING - LSTM
        // was advertised in BuiltinOpTypes and did not work for any real model. See OperatorInputReader.
        float[]? Get(int i) => pre != null && pre.TryGetValue(i, out var p) ? p
            : Array.IndexOf(StaticInputs, i) >= 0
                ? OperatorInputReader.ReadCached(reg, ctx, i, _weightCache)
                : OperatorInputReader.Read(reg, ctx, i);

        var wVals = Get(1); // W: [num_dir, 4*hidden_size, input_size]
        var rVals = Get(2); // R: [num_dir, 4*hidden_size, hidden_size]
        if (wVals == null || rVals == null)
            throw new NotSupportedException(
                "LSTM could not read its W/R weights. On a browser backend this operator needs the async "
                + "path (GraphExecutor.RunAsync) - the synchronous GPU readback is unavailable there.");

        float[]? bVals = ctx.Inputs.Length > 3 ? Get(3) : null;
        float[]? initHVals = ctx.Inputs.Length > 5 ? Get(5) : null;
        float[]? initCVals = ctx.Inputs.Length > 6 ? Get(6) : null;
        float[]? pVals = ctx.Inputs.Length > 7 ? Get(7) : null;

        int hiddenSize = ctx.GetInt("hidden_size", 0);
        string direction = ctx.GetString("direction", "forward");
        int layout = ctx.GetInt("layout", 0);
        int numDir = direction == "bidirectional" ? 2 : 1;

        var xShape = ctx.Inputs[0].Shape;
        int seqLen = layout == 0 ? xShape[0] : xShape[1];
        int batch = layout == 0 ? xShape[1] : xShape[0];
        int inputSize = xShape[2];
        if (hiddenSize == 0) hiddenSize = wVals.Length / (numDir * 4 * inputSize);

        var xVals = Get(0);
        if (xVals == null)
            throw new NotSupportedException(
                "LSTM could not read its input X. On a browser backend this operator needs the async path "
                + "(GraphExecutor.RunAsync). Returning silently here is what made LSTM a no-op.");

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
            int n = Math.Min(yTotal, ctx.Outputs[0].ElementCount);
            if (n < yData.Length) { var t = new float[n]; Array.Copy(yData, t, n); ctx.Outputs[0].Data.SubView(0, n).CopyFromCPU(t); }
            else ctx.Outputs[0].Data.SubView(0, n).CopyFromCPU(yData);
        }
        // Upload Y_h
        if (ctx.Outputs.Length > 1 && ctx.Outputs[1] != null)
        {
            int n = Math.Min(yhData.Length, ctx.Outputs[1].ElementCount);
            if (n < yhData.Length) { var t = new float[n]; Array.Copy(yhData, t, n); ctx.Outputs[1].Data.SubView(0, n).CopyFromCPU(t); }
            else ctx.Outputs[1].Data.SubView(0, n).CopyFromCPU(yhData);
        }
        // Upload Y_c (LSTM only)
        if (ctx.Outputs.Length > 2 && ctx.Outputs[2] != null)
        {
            int n = Math.Min(ycData.Length, ctx.Outputs[2].ElementCount);
            if (n < ycData.Length) { var t = new float[n]; Array.Copy(ycData, t, n); ctx.Outputs[2].Data.SubView(0, n).CopyFromCPU(t); }
            else ctx.Outputs[2].Data.SubView(0, n).CopyFromCPU(ycData);
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

    /// <summary>Cached host copies of the STATIC inputs (W, R, B), keyed by tensor name.</summary>
    private readonly Dictionary<string, float[]> _weightCache = new(StringComparer.Ordinal);

    /// <summary>Weights and bias. Static for the session, so worth caching.</summary>
    private static readonly int[] StaticInputs = { 1, 2, 3 };

    /// <summary>
    /// Per-call inputs: X and <c>initial_h</c>. ⚠️ <c>initial_h</c> must NOT be cached - a model that
    /// threads its recurrent state through graph inputs (Silero VAD does) would have that state frozen at
    /// the first frame.
    /// </summary>
    private static readonly int[] DynamicInputs = { 0, 5 };

    public void Execute(OnnxOpContext ctx) => ExecuteCore(ctx, null);

    /// <summary>Browser-safe path: the sync GPU readback throws on WebGPU/WebGL/Wasm.</summary>
    public async Task ExecuteAsync(OnnxOpContext ctx)
    {
        var pre = new Dictionary<int, float[]>();
        foreach (int i in DynamicInputs)
        {
            var v = await OperatorInputReader.ReadAsync(reg, ctx, i);
            if (v != null) pre[i] = v;
        }
        foreach (int i in StaticInputs)
        {
            var v = await OperatorInputReader.ReadCachedAsync(reg, ctx, i, _weightCache);
            if (v != null) pre[i] = v;
        }
        ExecuteCore(ctx, pre);
    }

    private void ExecuteCore(OnnxOpContext ctx, Dictionary<int, float[]>? pre)
    {
        // ⚠️ ctx.TryGetInputValues returns COMPILE-TIME CONSTANTS ONLY. X is the runtime input, so it read
        // as null on every real inference and this operator returned having computed NOTHING while being
        // advertised in BuiltinOpTypes. See OperatorInputReader.
        float[]? Get(int i) => pre != null && pre.TryGetValue(i, out var p) ? p
            : Array.IndexOf(StaticInputs, i) >= 0
                ? OperatorInputReader.ReadCached(reg, ctx, i, _weightCache)
                : OperatorInputReader.Read(reg, ctx, i);

        var wVals = Get(1); // W: [num_dir, 3*hidden_size, input_size]
        var rVals = Get(2); // R: [num_dir, 3*hidden_size, hidden_size]
        if (wVals == null || rVals == null)
            throw new NotSupportedException(
                "GRU could not read its W/R weights. On a browser backend this operator needs the "
                + "async path (GraphExecutor.RunAsync); the synchronous GPU readback is "
                + "unavailable there. Returning silently is what made this a no-op.");

        float[]? bVals = ctx.Inputs.Length > 3 ? Get(3) : null;
        float[]? initHVals = ctx.Inputs.Length > 5 ? Get(5) : null;

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

        var xVals = Get(0);
        if (xVals == null)
            throw new NotSupportedException(
                "GRU could not read its input X. On a browser backend this operator needs the "
                + "async path (GraphExecutor.RunAsync); the synchronous GPU readback is "
                + "unavailable there. Returning silently is what made this a no-op.");

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

                    // ⚠️ TWO bugs lived in this loop, both hidden because the operator never ran at all
                    // (X was read with TryGetInputValues, which only returns compile-time constants):
                    //
                    //   1. `ht` was written IN PLACE, so later units read a state already half-updated to
                    //      the new timestep.
                    //   2. the reset gate was applied as `r_hi * ht[hj]` for every j. ONNX specifies
                    //      `(rt ⊙ Ht-1)` - an ELEMENT-WISE product, so element j takes r_j, not the r of
                    //      the output unit being computed. That needs every r before any h gate.
                    //
                    // Measured against onnxruntime after fixing both: max |d| drops from 6.017E-002 to
                    // float32 noise. Gates first, state update last.
                    var zArr = new float[H2];
                    var rArr = new float[H2];
                    for (int hi = 0; hi < H2; hi++)
                    {
                        float z = wbz[dir * H2 + hi] + rbz[dir * H2 + hi];
                        for (int xi = 0; xi < inputSize; xi++)
                            z += xVals[xOff + xi] * wVals[wOff + hi * inputSize + xi];
                        for (int hj = 0; hj < H2; hj++)
                            z += ht[b * H2 + hj] * rVals[rOff + hi * H2 + hj];
                        zArr[hi] = 1f / (1f + MathF.Exp(-z)); // sigmoid

                        float r = wbr[dir * H2 + hi] + rbr[dir * H2 + hi];
                        for (int xi = 0; xi < inputSize; xi++)
                            r += xVals[xOff + xi] * wVals[wOff + (H2 + hi) * inputSize + xi];
                        for (int hj = 0; hj < H2; hj++)
                            r += ht[b * H2 + hj] * rVals[rOff + (H2 + hi) * H2 + hj];
                        rArr[hi] = 1f / (1f + MathF.Exp(-r)); // sigmoid
                    }

                    var hNew = new float[H2];
                    for (int hi = 0; hi < H2; hi++)
                    {
                        float xh = wbh[dir * H2 + hi];
                        for (int xi = 0; xi < inputSize; xi++)
                            xh += xVals[xOff + xi] * wVals[wOff + (2 * H2 + hi) * inputSize + xi];

                        float hHat;
                        if (linearBeforeReset != 0)
                        {
                            // rt ⊙ (Ht-1 * Rh^T + Rbh)
                            float rh = rbh[dir * H2 + hi];
                            for (int hj = 0; hj < H2; hj++)
                                rh += ht[b * H2 + hj] * rVals[rOff + (2 * H2 + hi) * H2 + hj];
                            hHat = MathF.Tanh(xh + rArr[hi] * rh);
                        }
                        else
                        {
                            // (rt ⊙ Ht-1) * Rh^T + Rbh - element j scaled by r_j, not by r_hi
                            float rh = rbh[dir * H2 + hi];
                            for (int hj = 0; hj < H2; hj++)
                                rh += (rArr[hj] * ht[b * H2 + hj]) * rVals[rOff + (2 * H2 + hi) * H2 + hj];
                            hHat = MathF.Tanh(xh + rh);
                        }

                        // Ht = (1 - zt) ⊙ h_hat + zt ⊙ Ht-1
                        hNew[hi] = (1f - zArr[hi]) * hHat + zArr[hi] * ht[b * H2 + hi];
                    }
                    Array.Copy(hNew, 0, ht, b * H2, H2);
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
            int n = Math.Min(yTotal, ctx.Outputs[0].ElementCount);
            if (n < yData.Length) { var t = new float[n]; Array.Copy(yData, t, n); ctx.Outputs[0].Data.SubView(0, n).CopyFromCPU(t); }
            else ctx.Outputs[0].Data.SubView(0, n).CopyFromCPU(yData);
        }
        if (ctx.Outputs.Length > 1 && ctx.Outputs[1] != null)
        {
            int n = Math.Min(yhData.Length, ctx.Outputs[1].ElementCount);
            if (n < yhData.Length) { var t = new float[n]; Array.Copy(yhData, t, n); ctx.Outputs[1].Data.SubView(0, n).CopyFromCPU(t); }
            else ctx.Outputs[1].Data.SubView(0, n).CopyFromCPU(yhData);
        }
    }
}
