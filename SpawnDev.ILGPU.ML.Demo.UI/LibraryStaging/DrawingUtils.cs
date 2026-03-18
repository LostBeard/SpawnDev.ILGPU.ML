namespace SpawnDev.ILGPU.ML.Preprocessing;

/// <summary>
/// CPU-side drawing utilities for rendering inference results onto RGBA images.
/// Bounding boxes, keypoint skeletons, text labels, and masks.
/// For headless or non-browser rendering (e.g., desktop console, unit tests, image export).
/// </summary>
public static class DrawingUtils
{
    /// <summary>
    /// Draw a rectangle outline on an RGBA image.
    /// </summary>
    public static void DrawRect(byte[] rgba, int width, int height,
        int x1, int y1, int x2, int y2, byte r, byte g, byte b, int thickness = 2)
    {
        x1 = Math.Clamp(x1, 0, width - 1);
        y1 = Math.Clamp(y1, 0, height - 1);
        x2 = Math.Clamp(x2, 0, width - 1);
        y2 = Math.Clamp(y2, 0, height - 1);

        // Top and bottom edges
        for (int t = 0; t < thickness; t++)
        {
            int topY = Math.Clamp(y1 + t, 0, height - 1);
            int botY = Math.Clamp(y2 - t, 0, height - 1);
            for (int x = x1; x <= x2; x++)
            {
                SetPixel(rgba, width, x, topY, r, g, b);
                SetPixel(rgba, width, x, botY, r, g, b);
            }
        }

        // Left and right edges
        for (int t = 0; t < thickness; t++)
        {
            int leftX = Math.Clamp(x1 + t, 0, width - 1);
            int rightX = Math.Clamp(x2 - t, 0, width - 1);
            for (int y = y1; y <= y2; y++)
            {
                SetPixel(rgba, width, leftX, y, r, g, b);
                SetPixel(rgba, width, rightX, y, r, g, b);
            }
        }
    }

    /// <summary>
    /// Draw a line between two points on an RGBA image using Bresenham's algorithm.
    /// </summary>
    public static void DrawLine(byte[] rgba, int width, int height,
        int x1, int y1, int x2, int y2, byte r, byte g, byte b, int thickness = 1)
    {
        int dx = Math.Abs(x2 - x1);
        int dy = Math.Abs(y2 - y1);
        int sx = x1 < x2 ? 1 : -1;
        int sy = y1 < y2 ? 1 : -1;
        int err = dx - dy;

        while (true)
        {
            // Draw thick pixel
            for (int ty = -thickness / 2; ty <= thickness / 2; ty++)
            {
                for (int tx = -thickness / 2; tx <= thickness / 2; tx++)
                {
                    int px = x1 + tx;
                    int py = y1 + ty;
                    if (px >= 0 && px < width && py >= 0 && py < height)
                        SetPixel(rgba, width, px, py, r, g, b);
                }
            }

            if (x1 == x2 && y1 == y2) break;
            int e2 = 2 * err;
            if (e2 > -dy) { err -= dy; x1 += sx; }
            if (e2 < dx) { err += dx; y1 += sy; }
        }
    }

    /// <summary>
    /// Draw a filled circle at the specified position.
    /// </summary>
    public static void DrawCircle(byte[] rgba, int width, int height,
        int cx, int cy, int radius, byte r, byte g, byte b)
    {
        for (int dy = -radius; dy <= radius; dy++)
        {
            for (int dx = -radius; dx <= radius; dx++)
            {
                if (dx * dx + dy * dy <= radius * radius)
                {
                    int px = cx + dx;
                    int py = cy + dy;
                    if (px >= 0 && px < width && py >= 0 && py < height)
                        SetPixel(rgba, width, px, py, r, g, b);
                }
            }
        }
    }

    /// <summary>
    /// Draw YOLO detection results onto an RGBA image.
    /// </summary>
    public static void DrawDetections(byte[] rgba, int width, int height,
        List<YoloPostProcessor.Detection> detections, int thickness = 2)
    {
        foreach (var det in detections)
        {
            var (r, g, b) = GetClassColor(det.ClassId);
            int x1 = (int)det.X1, y1 = (int)det.Y1;
            int x2 = (int)det.X2, y2 = (int)det.Y2;

            DrawRect(rgba, width, height, x1, y1, x2, y2, r, g, b, thickness);

            // Draw label background
            int labelH = 16;
            int labelY = Math.Max(y1 - labelH, 0);
            FillRect(rgba, width, height, x1, labelY, x2, labelY + labelH, r, g, b);
        }
    }

    /// <summary>
    /// Draw pose skeleton onto an RGBA image.
    /// </summary>
    public static void DrawSkeleton(byte[] rgba, int width, int height,
        PoseSkeleton.Keypoint[] keypoints, float confidenceThreshold = 0.3f,
        int lineThickness = 2, int pointRadius = 4)
    {
        // Draw bones
        foreach (var (from, to) in PoseSkeleton.Bones)
        {
            if (from >= keypoints.Length || to >= keypoints.Length) continue;
            var kpA = keypoints[from];
            var kpB = keypoints[to];
            if (kpA.Confidence < confidenceThreshold || kpB.Confidence < confidenceThreshold) continue;

            DrawLine(rgba, width, height,
                (int)kpA.X, (int)kpA.Y, (int)kpB.X, (int)kpB.Y,
                139, 92, 246, lineThickness); // Purple
        }

        // Draw keypoints
        foreach (var kp in keypoints)
        {
            if (kp.Confidence < confidenceThreshold) continue;
            DrawCircle(rgba, width, height,
                (int)kp.X, (int)kp.Y, pointRadius,
                52, 211, 153); // Green
        }
    }

    /// <summary>
    /// Apply an alpha mask as a colored overlay on an image.
    /// </summary>
    public static void DrawMaskOverlay(byte[] rgba, int width, int height,
        float[] mask, int maskW, int maskH,
        byte overlayR, byte overlayG, byte overlayB, float opacity = 0.4f)
    {
        float scaleX = (float)maskW / width;
        float scaleY = (float)maskH / height;

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int mx = Math.Clamp((int)(x * scaleX), 0, maskW - 1);
                int my = Math.Clamp((int)(y * scaleY), 0, maskH - 1);
                float maskVal = Math.Clamp(mask[my * maskW + mx], 0f, 1f);
                float alpha = maskVal * opacity;

                int idx = (y * width + x) * 4;
                rgba[idx + 0] = (byte)(rgba[idx + 0] * (1 - alpha) + overlayR * alpha);
                rgba[idx + 1] = (byte)(rgba[idx + 1] * (1 - alpha) + overlayG * alpha);
                rgba[idx + 2] = (byte)(rgba[idx + 2] * (1 - alpha) + overlayB * alpha);
            }
        }
    }

    /// <summary>
    /// Fill a rectangle on an RGBA image.
    /// </summary>
    public static void FillRect(byte[] rgba, int width, int height,
        int x1, int y1, int x2, int y2, byte r, byte g, byte b, byte a = 200)
    {
        x1 = Math.Clamp(x1, 0, width - 1);
        y1 = Math.Clamp(y1, 0, height - 1);
        x2 = Math.Clamp(x2, 0, width - 1);
        y2 = Math.Clamp(y2, 0, height - 1);

        float alpha = a / 255f;
        for (int y = y1; y <= y2; y++)
        {
            for (int x = x1; x <= x2; x++)
            {
                int idx = (y * width + x) * 4;
                rgba[idx + 0] = (byte)(rgba[idx + 0] * (1 - alpha) + r * alpha);
                rgba[idx + 1] = (byte)(rgba[idx + 1] * (1 - alpha) + g * alpha);
                rgba[idx + 2] = (byte)(rgba[idx + 2] * (1 - alpha) + b * alpha);
            }
        }
    }

    private static void SetPixel(byte[] rgba, int width, int x, int y, byte r, byte g, byte b)
    {
        int idx = (y * width + x) * 4;
        rgba[idx + 0] = r;
        rgba[idx + 1] = g;
        rgba[idx + 2] = b;
        rgba[idx + 3] = 255;
    }

    /// <summary>20 distinct colors for detection class visualization.</summary>
    private static readonly (byte R, byte G, byte B)[] ClassColors =
    {
        (59, 130, 246), (239, 68, 68), (34, 197, 94), (245, 158, 11), (139, 92, 246),
        (236, 72, 153), (20, 184, 166), (249, 115, 22), (6, 182, 212), (132, 204, 22),
        (168, 85, 247), (225, 29, 72), (14, 165, 233), (217, 70, 239), (16, 185, 129),
        (234, 179, 8), (99, 102, 241), (244, 63, 94), (45, 212, 191), (251, 146, 60),
    };

    /// <summary>Get a consistent color for a class ID.</summary>
    public static (byte R, byte G, byte B) GetClassColor(int classId) =>
        ClassColors[classId % ClassColors.Length];
}
