using System.Reflection;

namespace SpawnDev.ILGPU.ML.Demo.UI.Data;

/// <summary>
/// ImageNet 1000-class labels for classification models (MobileNetV2, EfficientNet, etc.).
/// Loads from embedded text file on first access.
/// </summary>
public static class ImageNetLabels
{
    private static string[]? _labels;

    /// <summary>All 1000 ImageNet class labels.</summary>
    public static string[] Labels => _labels ??= LoadLabels();

    /// <summary>
    /// Get the label for a class index (0-999).
    /// </summary>
    public static string GetLabel(int classIndex)
    {
        var labels = Labels;
        return classIndex >= 0 && classIndex < labels.Length ? labels[classIndex] : $"class_{classIndex}";
    }

    /// <summary>
    /// Get top-K predictions from raw logits.
    /// Applies softmax and returns sorted (label, probability) pairs.
    /// </summary>
    public static (string Label, float Probability)[] TopK(float[] logits, int k = 5)
    {
        var labels = Labels;

        // Softmax
        float max = float.MinValue;
        for (int i = 0; i < logits.Length; i++)
            if (logits[i] > max) max = logits[i];

        float sum = 0;
        var probs = new float[logits.Length];
        for (int i = 0; i < logits.Length; i++)
        {
            probs[i] = MathF.Exp(logits[i] - max);
            sum += probs[i];
        }
        for (int i = 0; i < probs.Length; i++)
            probs[i] /= sum;

        // Top-K via partial sort
        var indices = Enumerable.Range(0, probs.Length)
            .OrderByDescending(i => probs[i])
            .Take(k)
            .ToArray();

        return indices.Select(i => (
            Label: i < labels.Length ? labels[i] : $"class_{i}",
            Probability: probs[i]
        )).ToArray();
    }

    private static string[] LoadLabels()
    {
        // Load from embedded resource
        var assembly = typeof(ImageNetLabels).Assembly;
        var resourceName = assembly.GetManifestResourceNames()
            .FirstOrDefault(n => n.EndsWith("imagenet_classes.txt"));

        if (resourceName != null)
        {
            using var stream = assembly.GetManifestResourceStream(resourceName);
            if (stream != null)
            {
                using var reader = new StreamReader(stream);
                var lines = new List<string>();
                string? line;
                while ((line = reader.ReadLine()) != null)
                    lines.Add(line.Trim());
                if (lines.Count >= 999) // ImageNet has 1000 classes but file may have 999 lines (0-indexed)
                    return lines.ToArray();
            }
        }

        // Fallback: generate placeholder names
        return Enumerable.Range(0, 1000).Select(i => $"class_{i}").ToArray();
    }
}
