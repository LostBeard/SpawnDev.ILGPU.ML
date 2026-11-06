using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Training;

namespace SpawnDev.ILGPU.ML.Demo.Shared.Games.DrawToLearn;

/// <summary>
/// The /train page ("Draw to Learn"): a few drawings per class train a small classifier on the GPU, in the browser.
/// Kept here, not in the page, so <c>Training_DrawToLearn_ClassifiesNewDrawings</c> gates the exact code the page runs.
/// </summary>
/// <remarks>
/// MEASURED (<c>Training_DrawToLearn_ClassifiesNewDrawings</c>, CUDA, 3 classes x 6 drawings, 120 new drawings, 7 seeds): 95.0 to 99.2% right,
/// ~0.4 s to train; the page's former recipe (10 x 10 average downsample of the whole canvas, 784-64-N, 50 SGD steps at 0.01) 36 to 50%,
/// near chance (33%). What makes a handful of drawings per class enough:
/// - <see cref="Prepare"/> normalizes a drawing the way MNIST was made: crop to the strokes, scale the longer side to 20 px, center in 28 x 28.
///   Where and how big a shape was drawn then no longer matters, only its form.
/// - A light blur, so a stroke a pixel away from a learned one still overlaps it.
/// - <see cref="Augment"/> trains on shifted, rotated, rescaled copies of every drawing.
/// - Adam (<see cref="TrainableModel.UseAdam"/>) converges in a few hundred steps where SGD at a safe rate barely moves.
/// </remarks>
public static class DrawToLearn
{
    /// <summary>Model input side (28 x 28, one channel).</summary>
    public const int Side = 28;
    /// <summary>Model input length.</summary>
    public const int InputSize = Side * Side;
    /// <summary>Hidden units.</summary>
    public const int Hidden = 128;
    /// <summary>Training copies per drawing: the drawing itself plus augmented ones (<see cref="Augment"/>).</summary>
    public const int CopiesPerDrawing = 48;
    /// <summary>Full-batch Adam steps.</summary>
    public const int Steps = 200;
    /// <summary>Adam step size.</summary>
    public const float LearningRate = 2e-3f;
    /// <summary>Passes of a [1 2 1] / 4 blur over the prepared input (see <see cref="Prepare"/>).</summary>
    public const int BlurPasses = 2;
    /// <summary>Augmentation: rotation up to +- this many degrees.</summary>
    public const float AugmentDegrees = 25f;
    /// <summary>Augmentation: each axis scaled by 1 +- this.</summary>
    public const float AugmentScale = 0.3f;

    /// <summary>
    /// A drawing (RGBA, white strokes on black or transparent; the red channel is read) to a 28 x 28 input in [0, 1]:
    /// cropped to its strokes, longer side scaled to 20 px, centered, box-filtered, blurred (<see cref="BlurPasses"/>: a stroke a pixel off
    /// then still overlaps what was learned), peak scaled to 1. Null for a blank canvas.
    /// </summary>
    public static float[]? Prepare(ReadOnlySpan<byte> rgba, int width, int height)
    {
        int minX = width, minY = height, maxX = -1, maxY = -1;
        for (int y = 0; y < height; y++)
        {
            int row = y * width * 4;
            for (int x = 0; x < width; x++)
                if (rgba[row + x * 4] > 32)
                {
                    if (x < minX) minX = x;
                    if (x > maxX) maxX = x;
                    if (y < minY) minY = y;
                    if (y > maxY) maxY = y;
                }
        }
        if (maxX < 0) return null;
        float cx = (minX + maxX + 1) * 0.5f, cy = (minY + maxY + 1) * 0.5f;
        float scale = Math.Max(Math.Max(maxX - minX + 1, maxY - minY + 1), 20) / 20f;   // canvas px per model px
        var outp = new float[InputSize];
        for (int oy = 0; oy < Side; oy++)
        {
            int y0 = (int)MathF.Floor(cy + (oy - Side / 2) * scale), y1 = Math.Max(y0 + 1, (int)MathF.Floor(cy + (oy + 1 - Side / 2) * scale));
            for (int ox = 0; ox < Side; ox++)
            {
                int x0 = (int)MathF.Floor(cx + (ox - Side / 2) * scale), x1 = Math.Max(x0 + 1, (int)MathF.Floor(cx + (ox + 1 - Side / 2) * scale));
                float sum = 0;
                for (int y = Math.Max(y0, 0); y < Math.Min(y1, height); y++)
                {
                    int row = y * width * 4;
                    for (int x = Math.Max(x0, 0); x < Math.Min(x1, width); x++)
                        sum += rgba[row + x * 4];
                }
                float v = sum / ((y1 - y0) * (x1 - x0) * 255f);
                outp[oy * Side + ox] = v;
            }
        }
        for (int pass = 0; pass < BlurPasses; pass++)
        {
            var t = new float[InputSize];
            for (int y = 0; y < Side; y++)
                for (int x = 0; x < Side; x++)
                    t[y * Side + x] = 0.25f * At(outp, x - 1, y) + 0.5f * At(outp, x, y) + 0.25f * At(outp, x + 1, y);
            for (int y = 0; y < Side; y++)
                for (int x = 0; x < Side; x++)
                    outp[y * Side + x] = 0.25f * At(t, x, y - 1) + 0.5f * At(t, x, y) + 0.25f * At(t, x, y + 1);
        }
        float peak = 0;
        foreach (var v in outp) if (v > peak) peak = v;
        if (peak > 0)
            for (int i = 0; i < outp.Length; i++) outp[i] /= peak;
        return outp;
    }

    /// <summary>A copy of a prepared input: rotated up to +-<see cref="AugmentDegrees"/>, each axis scaled by 1 +-<see cref="AugmentScale"/>, shifted up to 2 px (bilinear).</summary>
    public static float[] Augment(float[] input, Random rng)
    {
        float ang = (float)(rng.NextDouble() * 2 - 1) * (AugmentDegrees * MathF.PI / 180f);
        float sx0 = 1 - AugmentScale + (float)rng.NextDouble() * 2 * AugmentScale, sy0 = 1 - AugmentScale + (float)rng.NextDouble() * 2 * AugmentScale;
        float dx = (float)(rng.NextDouble() * 4 - 2), dy = (float)(rng.NextDouble() * 4 - 2);
        float cos = MathF.Cos(ang), sin = MathF.Sin(ang);
        const float c = (Side - 1) * 0.5f;
        var outp = new float[InputSize];
        for (int y = 0; y < Side; y++)
            for (int x = 0; x < Side; x++)
            {
                // inverse map: output pixel -> source position
                float u = x - c - dx, v = y - c - dy;
                float sx = (cos * u + sin * v) / sx0 + c, sy = (-sin * u + cos * v) / sy0 + c;
                int ix = (int)MathF.Floor(sx), iy = (int)MathF.Floor(sy);
                float fx = sx - ix, fy = sy - iy;
                outp[y * Side + x] = (1 - fy) * ((1 - fx) * At(input, ix, iy) + fx * At(input, ix + 1, iy))
                                   + fy * ((1 - fx) * At(input, ix, iy + 1) + fx * At(input, ix + 1, iy + 1));
            }
        return outp;
    }

    static float At(float[] a, int x, int y) => (uint)x < Side && (uint)y < Side ? a[y * Side + x] : 0f;

    /// <summary>The training batch size for <paramref name="drawings"/> drawings.</summary>
    public static int BatchSize(int drawings) => drawings * CopiesPerDrawing;

    /// <summary>784 -> 128 -> ReLU -> classes, trained with Adam.</summary>
    public static TrainableModel CreateModel(Accelerator accelerator, int classes, int maxBatch)
    {
        var model = new TrainableModel(accelerator);
        model.AddLinear(InputSize, Hidden);
        model.AddReLU();
        model.AddLinear(Hidden, classes);
        model.UseAdam();
        model.Build(maxBatchSize: maxBatch);
        return model;
    }

    /// <summary>
    /// Trains <paramref name="model"/> on the prepared drawings (<paramref name="samples"/>[class] = that class's drawings) and their augmented
    /// copies, full batch. <paramref name="progress"/> gets (step, loss) every 10 steps and after the last; returns the final loss.
    /// </summary>
    public static async Task<float> TrainAsync(TrainableModel model, IReadOnlyList<IReadOnlyList<float[]>> samples, int seed = 1,
        Func<int, float, Task>? progress = null)
    {
        var rng = new Random(seed);
        int drawings = samples.Sum(s => s.Count);
        int n = BatchSize(drawings);
        var x = new float[n * InputSize];
        var labels = new int[n];
        int i = 0;
        for (int c = 0; c < samples.Count; c++)
            foreach (var s in samples[c])
                for (int k = 0; k < CopiesPerDrawing; k++, i++)
                {
                    Array.Copy(k == 0 ? s : Augment(s, rng), 0, x, i * InputSize, InputSize);
                    labels[i] = c;
                }
        float loss = 0;
        for (int step = 1; step <= Steps; step++)
        {
            bool report = step % 10 == 0 || step == Steps;
            if (report)
            {
                loss = await model.TrainStepAsync(x, labels, n, LearningRate);
                if (progress != null) await progress(step, loss);
            }
            else
                model.TrainStep(x, labels, n, LearningRate);
        }
        return loss;
    }
}
