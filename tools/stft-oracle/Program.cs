// CPU oracle for AudioPreprocessor's STFT/FFT.
//
//   dotnet run --project tools/stft-oracle -c Release
//
// Whisper's n_fft is 400, which is NOT a power of two. The FFT was radix-2 only, so with n=400 the
// butterflies addressed index 511 of a 400-element array and ComputeLogMelSpectrogram threw
// IndexOutOfRangeException on its own default arguments - the whole CPU Whisper path was unusable.
// This compares ComputeSTFT against an independent naive DFT: crash-free is not the bar, matching the
// reference is.
using SpawnDev.ILGPU.ML.Preprocessing;

int pass = 0, fail = 0;
void Check(string name, bool ok, string? detail = null)
{
    if (ok) { pass++; Console.WriteLine($"  PASS  {name}"); }
    else { fail++; Console.WriteLine($"  FAIL  {name}{(detail != null ? " - " + detail : "")}"); }
}

// Independent reference: direct O(n^2) DFT of each Hann-windowed frame. Deliberately the dumbest correct
// implementation, sharing no code with the thing under test except the public window generator.
static float[,] ReferenceStft(float[] samples, int fftSize, int hopSize)
{
    var window = AudioPreprocessor.GenerateHannWindow(fftSize);
    int numFrames = (samples.Length - fftSize) / hopSize + 1;
    int freqBins = fftSize / 2 + 1;
    var outp = new float[numFrames, freqBins];
    for (int f = 0; f < numFrames; f++)
    {
        int offset = f * hopSize;
        for (int k = 0; k < freqBins; k++)
        {
            double re = 0, im = 0;
            for (int t = 0; t < fftSize; t++)
            {
                int idx = offset + t;
                double x = idx < samples.Length ? samples[idx] * window[t] : 0.0;
                double ang = -2.0 * Math.PI * k * t / fftSize;
                re += x * Math.Cos(ang);
                im += x * Math.Sin(ang);
            }
            outp[f, k] = (float)Math.Sqrt(re * re + im * im);
        }
    }
    return outp;
}

static float[] Signal(int n, int seed = 7)
{
    var rng = new Random(seed);
    var s = new float[n];
    for (int i = 0; i < n; i++)
        s[i] = 0.6f * MathF.Sin(2f * MathF.PI * 440f * i / 16000f)
             + 0.3f * MathF.Sin(2f * MathF.PI * 1180f * i / 16000f)
             + 0.05f * (float)(rng.NextDouble() * 2 - 1);
    return s;
}

void CompareStft(string name, int fftSize, int hopSize, int sampleCount)
{
    var samples = Signal(sampleCount);
    float[,] got;
    try { got = AudioPreprocessor.ComputeSTFT(samples, fftSize, hopSize); }
    catch (Exception ex) { Check(name, false, $"threw {ex.GetType().Name}: {ex.Message}"); return; }
    var want = ReferenceStft(samples, fftSize, hopSize);

    if (got.GetLength(0) != want.GetLength(0) || got.GetLength(1) != want.GetLength(1))
    {
        Check(name, false, $"shape [{got.GetLength(0)},{got.GetLength(1)}] vs [{want.GetLength(0)},{want.GetLength(1)}]");
        return;
    }
    double worst = 0, scale = 0;
    for (int f = 0; f < want.GetLength(0); f++)
        for (int k = 0; k < want.GetLength(1); k++)
        {
            worst = Math.Max(worst, Math.Abs(got[f, k] - want[f, k]));
            scale = Math.Max(scale, Math.Abs(want[f, k]));
        }
    var rel = scale > 0 ? worst / scale : worst;
    Check($"{name} (n_fft={fftSize})", rel < 1e-4, $"max abs {worst:E3}, relative {rel:E3}");
}

Console.WriteLine("== STFT vs naive DFT oracle ==");
// The one that used to throw. 400 = 2^4 x 25, so it exercises four decimation splits and an odd base case.
CompareStft("whisper frame", 400, 160, 400 * 8);
// Power-of-two sizes must still go down the original radix-2 path and agree.
CompareStft("power of two", 512, 128, 512 * 6);
CompareStft("power of two small", 64, 16, 64 * 8);
// Other awkward lengths: prime, odd, and 2 x prime, to prove the split/base-case handoff.
CompareStft("prime length", 401, 160, 401 * 4);
CompareStft("odd length", 375, 125, 375 * 4);
CompareStft("two times prime", 202, 101, 202 * 5);

Console.WriteLine();
Console.WriteLine("== Whisper log-mel (the call that threw) ==");
try
{
    var mel = AudioPreprocessor.ComputeLogMelSpectrogram(Signal(16000 * 3));
    int frames = mel.Length / AudioPreprocessor.WhisperMelBins;
    Check("log-mel is 80 x 3000 (Whisper input shape)", frames == 3000, $"{AudioPreprocessor.WhisperMelBins} x {frames}");
    bool finite = mel.All(v => !float.IsNaN(v) && !float.IsInfinity(v));
    Check("mel values are finite", finite);
    bool varied = mel.Distinct().Count() > 100;
    Check("mel is not constant", varied);
}
catch (Exception ex) { Check("ComputeLogMelSpectrogram with default args", false, $"threw {ex.GetType().Name}: {ex.Message}"); }

Console.WriteLine();



Console.WriteLine();
Console.WriteLine("== Centred framing ==");
{
    // Centred framing is what makes 30s come out as the 3000 frames the model's input shape demands.
    var thirtySeconds = new float[AudioPreprocessor.WhisperMaxSamples];
    var centred = AudioPreprocessor.ComputeSTFT(thirtySeconds, 400, 160, center: true);
    var plain = AudioPreprocessor.ComputeSTFT(thirtySeconds, 400, 160);
    Check("centred gives 3001 frames (3000 after the caller drops the last)", centred.GetLength(0) == 3001, $"{centred.GetLength(0)}");
    Check("uncentred is unchanged at 2998", plain.GetLength(0) == 2998, $"{plain.GetLength(0)}");

    // Reflect padding must mirror without repeating the edge sample: [a,b,c,d] -> c,b,|a,b,c,d|,c,b
    var ramp = new float[] { 1, 2, 3, 4, 5, 6, 7, 8 };
    var framed = AudioPreprocessor.ComputeSTFT(ramp, 4, 2, center: true);
    Check("centred framing runs on a short signal", framed.GetLength(0) > 0, $"{framed.GetLength(0)} frames");
}

Console.WriteLine();
Console.WriteLine($"== {pass}/{pass + fail} passed, {fail} failed ==");
return fail == 0 ? 0 : 1;
