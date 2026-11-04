using System.Text.Json;
using System.Text.Json.Serialization;

namespace SpawnDev.ILGPU.ML.Pipelines;

/// <summary>
/// A linear probe on CLAP audio embeddings (<see cref="ClapAudioClassifier.EmbedAudioAsync"/>): class-balanced logistic regression fitted on
/// LABELLED AUDIO instead of text prompts. On in-domain sounds it beats zero-shot by a wide margin: AssistZ (DayZ, labels from the game's logs,
/// 2026-11-03), scored only on recordings it was not trained on: a 150 m automatic burst 30/30 named gunfire (prompts 20/30), no-shot events called
/// gunfire 1/169 (prompts 33/169), real footsteps 0/55, gunfire 618/619 (616/619).
/// </summary>
/// <remarks>
/// <para>
/// Two classes: <c>p = sigmoid(w . x + b)</c> for the SECOND class, fitted by full-batch gradient descent on the class-balanced logistic loss (each
/// class weighs 0.5 in total) with L2 <c>l2 / N</c> on w: exactly AssistZ's Lab/probe_eval.py <c>fit_logistic</c>. More classes: the multinomial
/// (softmax) form of the same loss, each class weighing 1 / K. Host math in double; a fit of a few thousand 512-d embeddings takes milliseconds.
/// </para>
/// <para>
/// Ship a fitted probe with <see cref="ToJson"/> / <see cref="FromJson"/>. A probe is tied to the checkpoint whose embeddings it was fitted on
/// (<see cref="Checkpoint"/>): embeddings of another checkpoint mean nothing to it.
/// </para>
/// </remarks>
public sealed class ClapProbe
{
    /// <summary>The class names, in the order <see cref="Predict"/> returns their probabilities.</summary>
    public string[] Classes { get; init; } = Array.Empty<string>();
    /// <summary>Embedding width the probe was fitted on.</summary>
    public int Dim { get; init; }
    /// <summary>Weights: one row (two classes: the second class's logit) or one row per class.</summary>
    public float[][] Weights { get; init; } = Array.Empty<float[]>();
    /// <summary>Bias per row of <see cref="Weights"/>.</summary>
    public float[] Bias { get; init; } = Array.Empty<float>();
    /// <summary>The CLAP checkpoint the embeddings came from (e.g. <see cref="ClapAudioClassifier.DefaultRepo"/>), or null if not recorded.</summary>
    public string? Checkpoint { get; init; }
    /// <summary>The fit's options and counts, for the record.</summary>
    public ClapProbeFitInfo? Fit { get; init; }

    /// <summary>
    /// Fit a probe. <paramref name="embeddings"/>: one per example (L2-normalized, as <see cref="ClapAudioClassifier.EmbedAudioAsync"/> returns);
    /// <paramref name="labels"/>: the class of each. Classes are ordered by first appearance unless <paramref name="classes"/> fixes the order (for two
    /// classes, the SECOND is the one whose probability the sigmoid models, e.g. { "other", "gunfire" }).
    /// </summary>
    public static ClapProbe FitLogistic(IReadOnlyList<float[]> embeddings, IReadOnlyList<string> labels, IReadOnlyList<string>? classes = null,
        int steps = 800, double learningRate = 0.5, double l2 = 1.0, bool classBalanced = true, string? checkpoint = null)
    {
        ArgumentNullException.ThrowIfNull(embeddings);
        ArgumentNullException.ThrowIfNull(labels);
        if (embeddings.Count == 0 || embeddings.Count != labels.Count) throw new ArgumentException($"{embeddings.Count} embeddings for {labels.Count} labels");
        int n = embeddings.Count, d = embeddings[0].Length;
        foreach (var e in embeddings) if (e.Length != d) throw new ArgumentException($"embeddings of different sizes ({e.Length} vs {d})");
        var order = (classes ?? labels.Distinct().ToArray()).ToArray();
        if (order.Length < 2) throw new ArgumentException("a probe needs at least two classes");
        var index = new Dictionary<string, int>();
        for (int c = 0; c < order.Length; c++) index[order[c]] = c;
        var y = new int[n];
        for (int i = 0; i < n; i++)
            y[i] = index.TryGetValue(labels[i], out int c) ? c : throw new ArgumentException($"label '{labels[i]}' is not one of the classes");
        var count = new int[order.Length];
        foreach (int c in y) count[c]++;
        for (int c = 0; c < order.Length; c++) if (count[c] == 0) throw new ArgumentException($"class '{order[c]}' has no examples");
        // per-example weight: class-balanced (each class sums to 1 / K) or plain mean (1 / N)
        var w = new double[n];
        for (int i = 0; i < n; i++) w[i] = classBalanced ? 1.0 / (order.Length * count[y[i]]) : 1.0 / n;
        var x = new double[n][];
        for (int i = 0; i < n; i++) { x[i] = new double[d]; for (int k = 0; k < d; k++) x[i][k] = embeddings[i][k]; }
        double reg = l2 / n;

        float[][] weights; float[] bias;
        if (order.Length == 2)
        {
            var a = new double[d]; double b = 0;
            var grad = new double[d];
            for (int step = 0; step < steps; step++)
            {
                Array.Clear(grad);
                double gb = 0;
                for (int i = 0; i < n; i++)
                {
                    double z = b; var xi = x[i];
                    for (int k = 0; k < d; k++) z += a[k] * xi[k];
                    double r = w[i] * (1.0 / (1.0 + Math.Exp(-z)) - y[i]);
                    for (int k = 0; k < d; k++) grad[k] += r * xi[k];
                    gb += r;
                }
                for (int k = 0; k < d; k++) a[k] -= learningRate * (grad[k] + reg * a[k]);
                b -= learningRate * gb;
            }
            weights = new[] { a.Select(v => (float)v).ToArray() };
            bias = new[] { (float)b };
        }
        else
        {
            int kc = order.Length;
            var a = new double[kc][]; for (int c = 0; c < kc; c++) a[c] = new double[d];
            var b = new double[kc];
            var grad = new double[kc][]; for (int c = 0; c < kc; c++) grad[c] = new double[d];
            var gb = new double[kc];
            var z = new double[kc];
            for (int step = 0; step < steps; step++)
            {
                for (int c = 0; c < kc; c++) Array.Clear(grad[c]);
                Array.Clear(gb);
                for (int i = 0; i < n; i++)
                {
                    var xi = x[i];
                    double max = double.NegativeInfinity;
                    for (int c = 0; c < kc; c++) { double s = b[c]; for (int k = 0; k < d; k++) s += a[c][k] * xi[k]; z[c] = s; max = Math.Max(max, s); }
                    double sum = 0;
                    for (int c = 0; c < kc; c++) { z[c] = Math.Exp(z[c] - max); sum += z[c]; }
                    for (int c = 0; c < kc; c++)
                    {
                        double r = w[i] * (z[c] / sum - (y[i] == c ? 1 : 0));
                        for (int k = 0; k < d; k++) grad[c][k] += r * xi[k];
                        gb[c] += r;
                    }
                }
                for (int c = 0; c < kc; c++)
                {
                    for (int k = 0; k < d; k++) a[c][k] -= learningRate * (grad[c][k] + reg * a[c][k]);
                    b[c] -= learningRate * gb[c];
                }
            }
            weights = a.Select(row => row.Select(v => (float)v).ToArray()).ToArray();
            bias = b.Select(v => (float)v).ToArray();
        }
        return new ClapProbe
        {
            Classes = order, Dim = d, Weights = weights, Bias = bias, Checkpoint = checkpoint,
            Fit = new ClapProbeFitInfo
            {
                Steps = steps, LearningRate = learningRate, L2 = l2, ClassBalanced = classBalanced,
                ExamplesPerClass = order.Select((c, i) => count[i]).ToArray(),
            },
        };
    }

    /// <summary>Class probabilities for one embedding, in <see cref="Classes"/> order.</summary>
    public float[] Predict(float[] embedding)
    {
        ArgumentNullException.ThrowIfNull(embedding);
        if (embedding.Length != Dim) throw new ArgumentException($"embedding of {embedding.Length} values; this probe was fitted on {Dim}");
        if (Classes.Length == 2)
        {
            double z = Bias[0];
            for (int k = 0; k < Dim; k++) z += (double)Weights[0][k] * embedding[k];
            double p = 1.0 / (1.0 + Math.Exp(-z));
            return new[] { (float)(1 - p), (float)p };
        }
        var s = new double[Classes.Length];
        double max = double.NegativeInfinity;
        for (int c = 0; c < s.Length; c++)
        {
            double v = Bias[c];
            for (int k = 0; k < Dim; k++) v += (double)Weights[c][k] * embedding[k];
            s[c] = v; max = Math.Max(max, v);
        }
        double sum = 0;
        for (int c = 0; c < s.Length; c++) { s[c] = Math.Exp(s[c] - max); sum += s[c]; }
        return s.Select(v => (float)(v / sum)).ToArray();
    }

    /// <summary>The most probable class for one embedding.</summary>
    public string PredictLabel(float[] embedding)
    {
        var p = Predict(embedding);
        int best = 0;
        for (int c = 1; c < p.Length; c++) if (p[c] > p[best]) best = c;
        return Classes[best];
    }

    /// <summary>The probe as JSON (classes, weights, bias, checkpoint, fit record). Source-generated serialization: trim / AOT safe.</summary>
    public string ToJson() => JsonSerializer.Serialize(this, ClapProbeJsonContext.Default.ClapProbe);

    /// <summary>A probe from <see cref="ToJson"/>.</summary>
    public static ClapProbe FromJson(string json)
    {
        var probe = JsonSerializer.Deserialize(json, ClapProbeJsonContext.Default.ClapProbe) ?? throw new ArgumentException("not a ClapProbe");
        int rows = probe.Classes.Length == 2 ? 1 : probe.Classes.Length;
        if (probe.Classes.Length < 2 || probe.Weights.Length != rows || probe.Bias.Length != rows || probe.Weights.Any(r => r.Length != probe.Dim))
            throw new ArgumentException("inconsistent ClapProbe JSON (classes / weights / bias / dim)");
        return probe;
    }
}

/// <summary>How a <see cref="ClapProbe"/> was fitted.</summary>
public sealed class ClapProbeFitInfo
{
    /// <summary>Gradient steps.</summary>
    public int Steps { get; init; }
    /// <summary>Step size.</summary>
    public double LearningRate { get; init; }
    /// <summary>L2 strength (applied as l2 / N).</summary>
    public double L2 { get; init; }
    /// <summary>Class-balanced loss weights.</summary>
    public bool ClassBalanced { get; init; }
    /// <summary>Examples per class, in <see cref="ClapProbe.Classes"/> order.</summary>
    public int[] ExamplesPerClass { get; init; } = Array.Empty<int>();
}

[JsonSourceGenerationOptions(DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull)]
[JsonSerializable(typeof(ClapProbe))]
internal partial class ClapProbeJsonContext : JsonSerializerContext { }
