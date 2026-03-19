namespace SpawnDev.ILGPU.ML.Preprocessing;

/// <summary>
/// Statistics utilities for float tensors. Useful for debugging inference outputs.
/// </summary>
public static class TensorStats
{
    /// <summary>
    /// Compute basic statistics of a float array.
    /// </summary>
    public static Stats Compute(float[] data)
    {
        if (data.Length == 0) return new Stats();

        float min = float.MaxValue, max = float.MinValue;
        double sum = 0;
        int nanCount = 0, infCount = 0;

        for (int i = 0; i < data.Length; i++)
        {
            float v = data[i];
            if (float.IsNaN(v)) { nanCount++; continue; }
            if (float.IsInfinity(v)) { infCount++; continue; }
            if (v < min) min = v;
            if (v > max) max = v;
            sum += v;
        }

        int validCount = data.Length - nanCount - infCount;
        double mean = validCount > 0 ? sum / validCount : 0;

        double variance = 0;
        for (int i = 0; i < data.Length; i++)
        {
            float v = data[i];
            if (float.IsNaN(v) || float.IsInfinity(v)) continue;
            double diff = v - mean;
            variance += diff * diff;
        }
        double std = validCount > 1 ? Math.Sqrt(variance / (validCount - 1)) : 0;

        return new Stats
        {
            Count = data.Length,
            Min = min,
            Max = max,
            Mean = (float)mean,
            Std = (float)std,
            NaNCount = nanCount,
            InfCount = infCount,
        };
    }

    /// <summary>
    /// Compute per-channel statistics for a NCHW tensor.
    /// </summary>
    public static Stats[] ComputePerChannel(float[] nchw, int channels, int height, int width)
    {
        int hw = height * width;
        var stats = new Stats[channels];
        for (int c = 0; c < channels; c++)
        {
            var channelData = new float[hw];
            Array.Copy(nchw, c * hw, channelData, 0, hw);
            stats[c] = Compute(channelData);
        }
        return stats;
    }

    /// <summary>
    /// Format tensor stats as a compact string for logging.
    /// </summary>
    public static string FormatStats(float[] data, string label = "tensor")
    {
        var s = Compute(data);
        var result = $"[{label}] n={s.Count} min={s.Min:G4} max={s.Max:G4} mean={s.Mean:G4} std={s.Std:G4}";
        if (s.NaNCount > 0) result += $" NaN={s.NaNCount}";
        if (s.InfCount > 0) result += $" Inf={s.InfCount}";
        return result;
    }

    /// <summary>
    /// Check if a tensor contains any NaN or Infinity values.
    /// </summary>
    public static bool HasAnomalies(float[] data)
    {
        for (int i = 0; i < data.Length; i++)
        {
            if (float.IsNaN(data[i]) || float.IsInfinity(data[i]))
                return true;
        }
        return false;
    }

    /// <summary>
    /// Compare two tensors element-wise and return error statistics.
    /// Useful for validating inference output against a reference.
    /// </summary>
    public static ComparisonResult Compare(float[] actual, float[] expected)
    {
        int length = Math.Min(actual.Length, expected.Length);
        float maxAbsError = 0;
        double sumAbsError = 0;
        double sumSqError = 0;
        int mismatchCount = 0;

        for (int i = 0; i < length; i++)
        {
            float diff = MathF.Abs(actual[i] - expected[i]);
            if (diff > maxAbsError) maxAbsError = diff;
            sumAbsError += diff;
            sumSqError += diff * diff;
            if (diff > 1e-4f) mismatchCount++;
        }

        return new ComparisonResult
        {
            Length = length,
            MaxAbsoluteError = maxAbsError,
            MeanAbsoluteError = (float)(sumAbsError / length),
            RootMeanSquareError = (float)Math.Sqrt(sumSqError / length),
            MismatchCount = mismatchCount,
            SizeMismatch = actual.Length != expected.Length,
        };
    }

    public class Stats
    {
        public int Count { get; set; }
        public float Min { get; set; }
        public float Max { get; set; }
        public float Mean { get; set; }
        public float Std { get; set; }
        public int NaNCount { get; set; }
        public int InfCount { get; set; }

        public override string ToString() =>
            $"n={Count} min={Min:G4} max={Max:G4} mean={Mean:G4} std={Std:G4}" +
            (NaNCount > 0 ? $" NaN={NaNCount}" : "") +
            (InfCount > 0 ? $" Inf={InfCount}" : "");
    }

    public class ComparisonResult
    {
        public int Length { get; set; }
        public float MaxAbsoluteError { get; set; }
        public float MeanAbsoluteError { get; set; }
        public float RootMeanSquareError { get; set; }
        public int MismatchCount { get; set; }
        public bool SizeMismatch { get; set; }

        public override string ToString() =>
            $"n={Length} maxErr={MaxAbsoluteError:G4} meanErr={MeanAbsoluteError:G4} " +
            $"rmse={RootMeanSquareError:G4} mismatches={MismatchCount}" +
            (SizeMismatch ? " SIZE_MISMATCH" : "");
    }
}
