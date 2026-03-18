using SpawnDev.ILGPU.ML.Data;

namespace SpawnDev.ILGPU.ML.Preprocessing;

/// <summary>
/// CPU-side postprocessing for YOLOv8 detection output.
/// Handles transposing, confidence filtering, box conversion, and NMS.
/// </summary>
public static class YoloPostProcessor
{
    /// <summary>
    /// Process raw YOLOv8 output tensor into a list of detections.
    /// </summary>
    /// <param name="output">Raw output: shape [1, 84, 8400] flattened row-major</param>
    /// <param name="numClasses">Number of classes (80 for COCO)</param>
    /// <param name="numDetections">Number of detection candidates (8400 for YOLOv8)</param>
    /// <param name="confThreshold">Minimum class confidence (default 0.25)</param>
    /// <param name="iouThreshold">NMS IoU threshold (default 0.45)</param>
    /// <param name="inputWidth">Model input width (640)</param>
    /// <param name="inputHeight">Model input height (640)</param>
    /// <param name="originalWidth">Original image width (for rescaling boxes)</param>
    /// <param name="originalHeight">Original image height (for rescaling boxes)</param>
    public static List<Detection> Process(
        float[] output,
        int numClasses = 80,
        int numDetections = 8400,
        float confThreshold = 0.25f,
        float iouThreshold = 0.45f,
        int inputWidth = 640,
        int inputHeight = 640,
        int originalWidth = 640,
        int originalHeight = 480)
    {
        int channels = 4 + numClasses; // 84 for COCO

        // Transpose from [1, 84, 8400] to per-detection access
        // output[c * numDetections + d] gives channel c of detection d
        var candidates = new List<Detection>();

        for (int d = 0; d < numDetections; d++)
        {
            // Find best class
            float maxScore = 0;
            int bestClass = 0;
            for (int c = 0; c < numClasses; c++)
            {
                float score = output[(4 + c) * numDetections + d];
                if (score > maxScore)
                {
                    maxScore = score;
                    bestClass = c;
                }
            }

            if (maxScore < confThreshold) continue;

            // Extract box (cx, cy, w, h)
            float cx = output[0 * numDetections + d];
            float cy = output[1 * numDetections + d];
            float w = output[2 * numDetections + d];
            float h = output[3 * numDetections + d];

            // Convert to xyxy
            float x1 = cx - w / 2;
            float y1 = cy - h / 2;
            float x2 = cx + w / 2;
            float y2 = cy + h / 2;

            candidates.Add(new Detection
            {
                X1 = x1, Y1 = y1, X2 = x2, Y2 = y2,
                Confidence = maxScore,
                ClassId = bestClass,
                Label = CocoLabels.GetLabel(bestClass)
            });
        }

        // NMS per class
        var results = new List<Detection>();
        var byClass = candidates.GroupBy(d => d.ClassId);
        foreach (var group in byClass)
        {
            var sorted = group.OrderByDescending(d => d.Confidence).ToList();
            var kept = Nms(sorted, iouThreshold);
            results.AddRange(kept);
        }

        // Rescale boxes from model input coords to original image coords
        // Assumes letterbox padding (aspect-preserving resize)
        float scale = Math.Min((float)inputWidth / originalWidth, (float)inputHeight / originalHeight);
        float padX = (inputWidth - originalWidth * scale) / 2;
        float padY = (inputHeight - originalHeight * scale) / 2;

        foreach (var det in results)
        {
            det.X1 = (det.X1 - padX) / scale;
            det.Y1 = (det.Y1 - padY) / scale;
            det.X2 = (det.X2 - padX) / scale;
            det.Y2 = (det.Y2 - padY) / scale;

            // Clamp to image bounds
            det.X1 = Math.Max(0, det.X1);
            det.Y1 = Math.Max(0, det.Y1);
            det.X2 = Math.Min(originalWidth, det.X2);
            det.Y2 = Math.Min(originalHeight, det.Y2);
        }

        return results.OrderByDescending(d => d.Confidence).ToList();
    }

    private static List<Detection> Nms(List<Detection> detections, float iouThreshold)
    {
        var kept = new List<Detection>();
        var suppressed = new bool[detections.Count];

        for (int i = 0; i < detections.Count; i++)
        {
            if (suppressed[i]) continue;
            kept.Add(detections[i]);

            for (int j = i + 1; j < detections.Count; j++)
            {
                if (suppressed[j]) continue;
                if (IoU(detections[i], detections[j]) > iouThreshold)
                    suppressed[j] = true;
            }
        }

        return kept;
    }

    private static float IoU(Detection a, Detection b)
    {
        float interX1 = Math.Max(a.X1, b.X1);
        float interY1 = Math.Max(a.Y1, b.Y1);
        float interX2 = Math.Min(a.X2, b.X2);
        float interY2 = Math.Min(a.Y2, b.Y2);

        float interArea = Math.Max(0, interX2 - interX1) * Math.Max(0, interY2 - interY1);
        float aArea = (a.X2 - a.X1) * (a.Y2 - a.Y1);
        float bArea = (b.X2 - b.X1) * (b.Y2 - b.Y1);

        return interArea / (aArea + bArea - interArea + 1e-6f);
    }

    public class Detection
    {
        public float X1 { get; set; }
        public float Y1 { get; set; }
        public float X2 { get; set; }
        public float Y2 { get; set; }
        public float Confidence { get; set; }
        public int ClassId { get; set; }
        public string Label { get; set; } = "";

        public float Width => X2 - X1;
        public float Height => Y2 - Y1;
    }
}
