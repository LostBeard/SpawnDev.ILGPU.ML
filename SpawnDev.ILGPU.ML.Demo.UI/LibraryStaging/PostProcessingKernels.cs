using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Kernels;

/// <summary>
/// GPU kernels for inference postprocessing.
/// YOLO box decode, mask application, embedding operations.
/// Keeps inference output on GPU through postprocessing.
/// </summary>
public class PostProcessingKernels
{
    private readonly Accelerator _accelerator;

    public PostProcessingKernels(Accelerator accelerator) => _accelerator = accelerator;

    // ──────────────────────────────────────────────
    //  YOLO output transpose: [1, 84, 8400] → per-detection access
    // ──────────────────────────────────────────────

    /// <summary>
    /// Transpose YOLO output from [channels, detections] to [detections, channels].
    /// Keeps data on GPU for subsequent filtering.
    /// </summary>
    public void TransposeYoloOutput(
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output,
        int channels, int detections)
    {
        var kernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(
            (Index1D idx, ArrayView1D<float, Stride1D.Dense> inp, ArrayView1D<float, Stride1D.Dense> outp) =>
            {
                int d = idx / channels;
                int c = idx % channels;
                outp[idx] = inp[c * detections + d];
            });

        kernel((Index1D)(channels * detections), input, output);
    }

    // ──────────────────────────────────────────────
    //  YOLO box decode: xywh → xyxy
    // ──────────────────────────────────────────────

    /// <summary>
    /// Convert YOLO center-format boxes to corner-format on GPU.
    /// Input/output: [numBoxes, 4] where first 4 values are cx,cy,w,h → x1,y1,x2,y2.
    /// In-place operation on the transposed detection buffer.
    /// </summary>
    public void DecodeYoloBoxes(
        ArrayView1D<float, Stride1D.Dense> detections,
        int numDetections, int stride)
    {
        var kernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>>(
            (Index1D idx, ArrayView1D<float, Stride1D.Dense> det) =>
            {
                int offset = idx * stride;
                float cx = det[offset + 0];
                float cy = det[offset + 1];
                float w = det[offset + 2];
                float h = det[offset + 3];

                det[offset + 0] = cx - w * 0.5f; // x1
                det[offset + 1] = cy - h * 0.5f; // y1
                det[offset + 2] = cx + w * 0.5f; // x2
                det[offset + 3] = cy + h * 0.5f; // y2
            });

        kernel((Index1D)numDetections, detections);
    }

    // ──────────────────────────────────────────────
    //  Confidence filter: mark detections below threshold
    // ──────────────────────────────────────────────

    /// <summary>
    /// Find max class score per detection and mark those below threshold.
    /// Sets the max score to -1 for filtered detections (in-place).
    /// Returns the scores in a separate buffer for CPU-side NMS.
    /// </summary>
    public void FilterByConfidence(
        ArrayView1D<float, Stride1D.Dense> detections,
        ArrayView1D<float, Stride1D.Dense> maxScores,
        ArrayView1D<int, Stride1D.Dense> maxClassIds,
        int numDetections, int numClasses, int stride,
        float threshold)
    {
        var kernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>>(
            (Index1D idx, ArrayView1D<float, Stride1D.Dense> det, ArrayView1D<float, Stride1D.Dense> scores, ArrayView1D<int, Stride1D.Dense> classIds) =>
            {
                int offset = idx * stride + 4; // Skip box coords
                float maxScore = -1f;
                int maxClass = 0;

                for (int c = 0; c < numClasses; c++)
                {
                    float score = det[offset + c];
                    if (score > maxScore)
                    {
                        maxScore = score;
                        maxClass = c;
                    }
                }

                scores[idx] = maxScore >= threshold ? maxScore : -1f;
                classIds[idx] = maxClass;
            });

        kernel((Index1D)numDetections, detections, maxScores, maxClassIds);
    }

    // ──────────────────────────────────────────────
    //  Cosine similarity between embedding vectors
    // ──────────────────────────────────────────────

    /// <summary>
    /// Compute cosine similarity between a query embedding and N candidate embeddings on GPU.
    /// Query: [dim], Candidates: [N, dim], Output: [N] similarity scores.
    /// One thread per candidate.
    /// </summary>
    public void CosineSimilarityBatch(
        ArrayView1D<float, Stride1D.Dense> query,
        ArrayView1D<float, Stride1D.Dense> candidates,
        ArrayView1D<float, Stride1D.Dense> similarities,
        int numCandidates, int dim)
    {
        var kernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(
            (Index1D idx, ArrayView1D<float, Stride1D.Dense> q, ArrayView1D<float, Stride1D.Dense> cands, ArrayView1D<float, Stride1D.Dense> sims) =>
            {
                float dot = 0f, normA = 0f, normB = 0f;
                int offset = idx * dim;

                for (int d = 0; d < dim; d++)
                {
                    float a = q[d];
                    float b = cands[offset + d];
                    dot += a * b;
                    normA += a * a;
                    normB += b * b;
                }

                float denom = MathF.Sqrt(normA) * MathF.Sqrt(normB);
                sims[idx] = denom > 1e-8f ? dot / denom : 0f;
            });

        kernel((Index1D)numCandidates, query, candidates, similarities);
    }

    // ──────────────────────────────────────────────
    //  L2 normalize embeddings (in-place, per-row)
    // ──────────────────────────────────────────────

    /// <summary>
    /// L2 normalize each embedding vector (row) in-place on GPU.
    /// One thread per row. Each row is [dim] elements.
    /// </summary>
    public void L2NormalizeRows(
        ArrayView1D<float, Stride1D.Dense> embeddings,
        int numRows, int dim)
    {
        var kernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>>(
            (Index1D idx, ArrayView1D<float, Stride1D.Dense> emb) =>
            {
                int offset = idx * dim;
                float sumSq = 0f;
                for (int d = 0; d < dim; d++)
                {
                    float v = emb[offset + d];
                    sumSq += v * v;
                }
                float norm = MathF.Sqrt(sumSq);
                if (norm > 1e-10f)
                {
                    float invNorm = 1f / norm;
                    for (int d = 0; d < dim; d++)
                    {
                        emb[offset + d] *= invNorm;
                    }
                }
            });

        kernel((Index1D)numRows, embeddings);
    }

    // ──────────────────────────────────────────────
    //  Softmax on GPU (per-row)
    // ──────────────────────────────────────────────

    /// <summary>
    /// Compute softmax per row on GPU.
    /// One thread per row (sequential within row for numerical stability).
    /// For large vocabularies (text generation), this is faster than CPU.
    /// </summary>
    public void SoftmaxRows(
        ArrayView1D<float, Stride1D.Dense> data,
        int numRows, int rowSize)
    {
        var kernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>>(
            (Index1D idx, ArrayView1D<float, Stride1D.Dense> d) =>
            {
                int offset = idx * rowSize;

                // Find max
                float maxVal = d[offset];
                for (int i = 1; i < rowSize; i++)
                {
                    float v = d[offset + i];
                    if (v > maxVal) maxVal = v;
                }

                // Exp and sum
                float sum = 0f;
                for (int i = 0; i < rowSize; i++)
                {
                    float e = MathF.Exp(d[offset + i] - maxVal);
                    d[offset + i] = e;
                    sum += e;
                }

                // Normalize
                float invSum = 1f / sum;
                for (int i = 0; i < rowSize; i++)
                {
                    d[offset + i] *= invSum;
                }
            });

        kernel((Index1D)numRows, data);
    }

    // ──────────────────────────────────────────────
    //  Segmentation mask resize + threshold
    // ──────────────────────────────────────────────

    /// <summary>
    /// Resize a segmentation mask and apply threshold on GPU.
    /// Input: float mask [maskH, maskW]. Output: float mask [dstH, dstW] with values 0 or 1.
    /// Uses bilinear interpolation + threshold in one pass.
    /// </summary>
    public void ResizeAndThresholdMask(
        ArrayView1D<float, Stride1D.Dense> mask,
        ArrayView1D<float, Stride1D.Dense> output,
        int maskW, int maskH, int dstW, int dstH,
        float threshold = 0.5f)
    {
        var kernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(
            (Index1D idx, ArrayView1D<float, Stride1D.Dense> m, ArrayView1D<float, Stride1D.Dense> o) =>
            {
                int dy = idx / dstW;
                int dx = idx % dstW;

                float srcX = (dx + 0.5f) * maskW / dstW - 0.5f;
                float srcY = (dy + 0.5f) * maskH / dstH - 0.5f;

                int x0 = (int)srcX; int x1 = x0 + 1;
                int y0 = (int)srcY; int y1 = y0 + 1;
                float fx = srcX - x0; float fy = srcY - y0;

                x0 = x0 < 0 ? 0 : (x0 >= maskW ? maskW - 1 : x0);
                x1 = x1 < 0 ? 0 : (x1 >= maskW ? maskW - 1 : x1);
                y0 = y0 < 0 ? 0 : (y0 >= maskH ? maskH - 1 : y0);
                y1 = y1 < 0 ? 0 : (y1 >= maskH ? maskH - 1 : y1);

                float v00 = m[y0 * maskW + x0], v10 = m[y0 * maskW + x1];
                float v01 = m[y1 * maskW + x0], v11 = m[y1 * maskW + x1];

                float val = v00 * (1f - fy) * (1f - fx) + v10 * (1f - fy) * fx
                          + v01 * fy * (1f - fx) + v11 * fy * fx;

                o[idx] = val >= threshold ? 1f : 0f;
            });

        kernel((Index1D)(dstW * dstH), mask, output);
    }
}
