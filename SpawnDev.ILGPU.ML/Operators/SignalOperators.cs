using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Tensors;

namespace SpawnDev.ILGPU.ML.Operators;

/// <summary>
/// ONNX DFT operator — Discrete Fourier Transform.
/// Input: [batch, signal_length, 1] (real) or [batch, signal_length, 2] (complex)
/// Output: [batch, signal_length, 2] (complex: real + imaginary)
/// Spec: https://onnx.ai/onnx/operators/onnx__DFT.html
/// </summary>
public class DFTOperatorImpl(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "DFT";

    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
    {
        var shape = inputs[0].ToArray();
        // Output always has 2 components (real + imag) in the last dimension
        shape[^1] = 2;
        // If dft_length is provided, it overrides the signal dimension
        if (inputs.Length > 1 && inputs[1].Length > 0)
        {
            // dft_length input — use it for the signal dimension
        }
        return new[] { shape };
    }

    public void Execute(OnnxOpContext ctx)
    {
        var input = ctx.Inputs[0];
        var shape = input.Shape;
        int inverse = ctx.GetInt("inverse", 0);
        int onesided = ctx.GetInt("onesided", 0);
        int axis = ctx.GetInt("axis", 1);
        int batch = 1;
        for (int i = 0; i < axis; i++) batch *= shape[i];
        int N = shape[axis];
        int isComplex = shape[^1] == 2 ? 1 : 0;
        int dftLength = N;
        if (ctx.Inputs.Length > 1 && ctx.Inputs[1] != null)
        {
            var dftLenVals = ctx.TryGetInputValues(1);
            if (dftLenVals != null && dftLenVals.Length > 0) dftLength = (int)dftLenVals[0];
        }
        int outputN = onesided != 0 ? dftLength / 2 + 1 : dftLength;

        // GPU path: one thread per output frequency bin
        var paramsBuf = reg.Accelerator.Allocate1D(new int[] { N, dftLength, outputN, isComplex, inverse });
        reg.ElementWise.DFT(input.Data, ctx.Outputs[0].Data, paramsBuf.View, batch * outputN);
        reg.Accelerator.Synchronize();
        paramsBuf.Dispose();
    }
}

/// <summary>
/// ONNX STFT operator — Short-Time Fourier Transform.
/// Input signal: [batch, signal_length, 1]
/// Window: [window_length]
/// Output: [batch, num_frames, fft_length/2+1, 2] (complex)
/// Spec: https://onnx.ai/onnx/operators/onnx__STFT.html
/// </summary>
public class STFTOperatorImpl(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "STFT";

    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
    {
        int onesided = attrs.ContainsKey("onesided") ? Convert.ToInt32(attrs["onesided"]) : 1;
        var signalShape = inputs[0]; // [batch, signal_length, 1]
        int signalLength = signalShape[1];

        // frame_step from input[1], frame_length from input[2] or attribute
        int frameLength = 256; // default
        int frameStep = 128; // default
        if (inputs.Length > 2 && inputs[2].Length > 0)
            frameLength = inputs[2][0]; // approximate from shape

        int numFrames = (signalLength - frameLength) / frameStep + 1;
        int fftLength = onesided != 0 ? frameLength / 2 + 1 : frameLength;

        return new[] { new[] { signalShape[0], numFrames, fftLength, 2 } };
    }

    public void Execute(OnnxOpContext ctx)
    {
        var signal = ctx.TryGetInputValues(0);
        if (signal == null) return;

        var signalShape = ctx.Inputs[0].Shape;
        int batch = signalShape[0];
        int signalLength = signalShape[1];

        // frame_step (input 1)
        int frameStep = 128;
        var frameStepVals = ctx.TryGetInputValues(1);
        if (frameStepVals != null && frameStepVals.Length > 0)
            frameStep = (int)frameStepVals[0];

        // window (input 2) — optional
        float[]? window = ctx.Inputs.Length > 2 ? ctx.TryGetInputValues(2) : null;
        int frameLength = window?.Length ?? 256;

        // frame_length (input 3) — optional override
        if (ctx.Inputs.Length > 3)
        {
            var flVals = ctx.TryGetInputValues(3);
            if (flVals != null && flVals.Length > 0)
                frameLength = (int)flVals[0];
        }

        int onesided = ctx.GetInt("onesided", 1);
        int numFrames = Math.Max(1, (signalLength - frameLength) / frameStep + 1);
        int fftOutputLen = onesided != 0 ? frameLength / 2 + 1 : frameLength;

        var result = new float[batch * numFrames * fftOutputLen * 2];

        for (int b = 0; b < batch; b++)
        {
            for (int f = 0; f < numFrames; f++)
            {
                int frameStart = f * frameStep;

                // Apply window and compute DFT for this frame
                for (int k = 0; k < fftOutputLen; k++)
                {
                    float sumReal = 0f, sumImag = 0f;
                    for (int n = 0; n < frameLength; n++)
                    {
                        int sIdx = frameStart + n;
                        float x = sIdx < signalLength ? signal[b * signalLength + sIdx] : 0f;
                        if (window != null && n < window.Length) x *= window[n];

                        float angle = -2f * MathF.PI * k * n / frameLength;
                        sumReal += x * MathF.Cos(angle);
                        sumImag += x * MathF.Sin(angle);
                    }
                    int outIdx = ((b * numFrames + f) * fftOutputLen + k) * 2;
                    result[outIdx] = sumReal;
                    result[outIdx + 1] = sumImag;
                }
            }
        }

        int copyLen = Math.Min(result.Length, ctx.Outputs[0].ElementCount);
        if (copyLen < result.Length) { var t = new float[copyLen]; Array.Copy(result, t, copyLen); ctx.Outputs[0].Data.SubView(0, copyLen).CopyFromCPU(t); }
        else ctx.Outputs[0].Data.SubView(0, copyLen).CopyFromCPU(result);
    }
}

/// <summary>
/// ONNX MelWeightMatrix operator — Mel-scale filter bank.
/// Generates the weight matrix that maps DFT bins to Mel-scale bins.
/// Inputs: num_mel_bins (int), dft_length (int), sample_rate (int),
///         lower_edge_hertz (float), upper_edge_hertz (float)
/// Output: [num_spectrogram_bins, num_mel_bins] — the filter bank matrix
/// Spec: https://onnx.ai/onnx/operators/onnx__MelWeightMatrix.html
/// </summary>
public class MelWeightMatrixOperatorImpl(OperatorRegistry reg) : IOnnxOperator
{
    public string OpType => "MelWeightMatrix";

    public int[][] InferOutputShapes(int[][] inputs, Dictionary<string, object> attrs)
    {
        // inputs: [0]=num_mel_bins, [1]=dft_length, [2]=sample_rate, [3]=lower_edge, [4]=upper_edge
        // Output: [dft_length/2+1, num_mel_bins]
        // We can't know exact values at shape inference time — use estimates
        int numSpecBins = 201; // default for dft_length=400
        int numMelBins = 80;   // common default
        return new[] { new[] { numSpecBins, numMelBins } };
    }

    public void Execute(OnnxOpContext ctx)
    {
        // Read scalar inputs
        var numMelVals = ctx.TryGetInputValues(0);
        var dftLenVals = ctx.TryGetInputValues(1);
        var sampleRateVals = ctx.TryGetInputValues(2);
        var lowerVals = ctx.TryGetInputValues(3);
        var upperVals = ctx.TryGetInputValues(4);

        int numMelBins = numMelVals != null ? (int)numMelVals[0] : 80;
        int dftLength = dftLenVals != null ? (int)dftLenVals[0] : 400;
        int sampleRate = sampleRateVals != null ? (int)sampleRateVals[0] : 16000;
        float lowerEdge = lowerVals != null ? lowerVals[0] : 0f;
        float upperEdge = upperVals != null ? upperVals[0] : sampleRate / 2f;

        int numSpecBins = dftLength / 2 + 1;

        // Generate Mel filter bank
        var matrix = new float[numSpecBins * numMelBins];

        // Mel scale conversion: mel = 2595 * log10(1 + hz / 700)
        float melLower = 2595f * MathF.Log10(1f + lowerEdge / 700f);
        float melUpper = 2595f * MathF.Log10(1f + upperEdge / 700f);

        // Create numMelBins + 2 mel points (including edges)
        var melPoints = new float[numMelBins + 2];
        for (int i = 0; i < numMelBins + 2; i++)
            melPoints[i] = melLower + i * (melUpper - melLower) / (numMelBins + 1);

        // Convert mel points back to Hz
        var hzPoints = new float[numMelBins + 2];
        for (int i = 0; i < numMelBins + 2; i++)
            hzPoints[i] = 700f * (MathF.Pow(10f, melPoints[i] / 2595f) - 1f);

        // Convert Hz to FFT bin indices
        var binPoints = new float[numMelBins + 2];
        for (int i = 0; i < numMelBins + 2; i++)
            binPoints[i] = hzPoints[i] * (dftLength + 1) / sampleRate;

        // Build triangular filters
        for (int m = 0; m < numMelBins; m++)
        {
            float fLeft = binPoints[m];
            float fCenter = binPoints[m + 1];
            float fRight = binPoints[m + 2];

            for (int k = 0; k < numSpecBins; k++)
            {
                float weight = 0f;
                if (k >= fLeft && k <= fCenter && fCenter > fLeft)
                    weight = (k - fLeft) / (fCenter - fLeft);
                else if (k > fCenter && k <= fRight && fRight > fCenter)
                    weight = (fRight - k) / (fRight - fCenter);
                matrix[k * numMelBins + m] = weight;
            }
        }

        // Upload to GPU
        int copyLen = Math.Min(matrix.Length, ctx.Outputs[0].ElementCount);
        if (copyLen < matrix.Length) { var t = new float[copyLen]; Array.Copy(matrix, t, copyLen); ctx.Outputs[0].Data.SubView(0, copyLen).CopyFromCPU(t); }
        else ctx.Outputs[0].Data.SubView(0, copyLen).CopyFromCPU(matrix);
    }
}
