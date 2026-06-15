namespace SpawnDev.ILGPU.ML.Preprocessing;

/// <summary>
/// Image preprocessing for Gemma 4 12B "Unified" vision (projector type <c>gemma4uv</c>). Turns a decoded
/// RGB image into the <c>[nPatches, 6912]</c> patch matrix the vision projector consumes, matching
/// llama.cpp <c>tools/mtmd</c> exactly (file:line citations in Plans/gemma4-llamacpp-reference-snippets.md):
///
///  1. <b>smart_resize</b> (<c>calc_size_preserved_ratio</c>): aspect-preserving target, both dims rounded
///     to multiples of the effective 48px patch (away-from-zero), total pixels clamped to [92160, 645120].
///  2. <b>PAD_CEIL</b>: bilinear-resize keeping aspect (scale = min of the two axis scales), then
///     center-composite onto a black canvas of the smart-resize target size.
///  3. <b>im2col</b> (48×48 patches, stride 48): each patch flattened channel-outermost, row-major within
///     channel — <c>[c][ky][kx]</c> → index <c>c*48*48 + ky*48 + kx</c>. Patch order p = gy*nCols + gx.
///
/// Normalization is <c>pixel/255</c> only (gemma4 mmproj image_mean=0, image_std=1).
/// </summary>
public static class Gemma4ImagePreprocessor
{
    public const int PatchPx = 48;        // effective patch (stored 16 × n_merge 3)
    public const int Channels = 3;
    public const int PatchLen = PatchPx * PatchPx * Channels; // 6912
    public const int MinPixels = 40 * PatchPx * PatchPx;      // 92160
    public const int MaxPixels = 280 * PatchPx * PatchPx;     // 645120

    /// <summary>
    /// Preprocess an RGB image (interleaved HWC, 8-bit, <paramref name="srcPixels"/> length = srcW*srcH*3)
    /// into the patch matrix. Returns the flat <c>[nPatches, 6912]</c> patches plus the 48px grid dims.
    /// </summary>
    public static (float[] patches, int nCols, int nRows) Preprocess(byte[] srcPixels, int srcW, int srcH)
    {
        if (srcPixels.Length < (long)srcW * srcH * 3)
            throw new ArgumentException($"srcPixels length {srcPixels.Length} < {srcW}x{srcH}x3.");

        var (tgtW, tgtH) = SmartResize(srcW, srcH);

        // PAD_CEIL: aspect-preserving bilinear resize (scale = min axis scale), then center on black canvas.
        float scale = Math.Min((float)tgtW / srcW, (float)tgtH / srcH);
        int newW = Math.Min((int)Math.Ceiling(srcW * (double)scale), tgtW);
        int newH = Math.Min((int)Math.Ceiling(srcH * (double)scale), tgtH);
        var resized = ResizeBilinear(srcPixels, srcW, srcH, newW, newH); // float RGB HWC, /255

        // Black canvas (tgtW×tgtH), composite resized centered.
        var canvas = new float[(long)tgtW * tgtH * 3]; // zero = black
        int offX = (tgtW - newW) / 2, offY = (tgtH - newH) / 2;
        for (int y = 0; y < newH; y++)
        {
            int srcRow = y * newW * 3;
            int dstRow = ((offY + y) * tgtW + offX) * 3;
            Array.Copy(resized, srcRow, canvas, dstRow, newW * 3);
        }

        // im2col: 48×48 stride-48 patches, channel-outermost row-major flatten.
        int nCols = tgtW / PatchPx, nRows = tgtH / PatchPx;
        int nPatches = nCols * nRows;
        var patches = new float[(long)nPatches * PatchLen];
        for (int gy = 0; gy < nRows; gy++)
        for (int gx = 0; gx < nCols; gx++)
        {
            int p = gy * nCols + gx;
            long pBase = (long)p * PatchLen;
            int baseY = gy * PatchPx, baseX = gx * PatchPx;
            for (int c = 0; c < Channels; c++)
            {
                long cBase = pBase + (long)c * PatchPx * PatchPx;
                for (int ky = 0; ky < PatchPx; ky++)
                {
                    int canRow = ((baseY + ky) * tgtW + baseX) * 3 + c;
                    long outRow = cBase + (long)ky * PatchPx;
                    for (int kx = 0; kx < PatchPx; kx++)
                        patches[outRow + kx] = canvas[canRow + kx * 3];
                }
            }
        }
        return (patches, nCols, nRows);
    }

    /// <summary>llama.cpp <c>calc_size_preserved_ratio</c> (align=48, min/max pixels). Returns (w, h).</summary>
    public static (int w, int h) SmartResize(int w, int h, int f = PatchPx, int minPix = MinPixels, int maxPix = MaxPixels)
    {
        int RoundF(double x) => (int)Math.Round(x / f, MidpointRounding.AwayFromZero) * f;
        int CeilF(double x) => (int)Math.Ceiling(x / f) * f;
        int FloorF(double x) => (int)Math.Floor(x / f) * f;

        int hBar = Math.Max(f, RoundF(h));
        int wBar = Math.Max(f, RoundF(w));
        if ((long)hBar * wBar > maxPix)
        {
            double beta = Math.Sqrt((double)h * w / maxPix);
            hBar = Math.Max(f, FloorF(h / beta));
            wBar = Math.Max(f, FloorF(w / beta));
        }
        else if ((long)hBar * wBar < minPix)
        {
            double beta = Math.Sqrt((double)minPix / ((double)h * w));
            hBar = CeilF(h * beta);
            wBar = CeilF(w * beta);
        }
        return (wBar, hBar);
    }

    /// <summary>Bilinear resize 8-bit RGB HWC → float RGB HWC normalized to [0,1] (/255). Half-pixel centers.</summary>
    private static float[] ResizeBilinear(byte[] src, int sw, int sh, int dw, int dh)
    {
        var dst = new float[(long)dw * dh * 3];
        float sxRatio = (float)sw / dw, syRatio = (float)sh / dh;
        for (int dy = 0; dy < dh; dy++)
        {
            float sy = (dy + 0.5f) * syRatio - 0.5f;
            int y0 = (int)MathF.Floor(sy);
            float fy = sy - y0;
            int y0c = Math.Clamp(y0, 0, sh - 1), y1c = Math.Clamp(y0 + 1, 0, sh - 1);
            for (int dx = 0; dx < dw; dx++)
            {
                float sx = (dx + 0.5f) * sxRatio - 0.5f;
                int x0 = (int)MathF.Floor(sx);
                float fx = sx - x0;
                int x0c = Math.Clamp(x0, 0, sw - 1), x1c = Math.Clamp(x0 + 1, 0, sw - 1);
                int dOff = (dy * dw + dx) * 3;
                for (int c = 0; c < 3; c++)
                {
                    float v00 = src[(y0c * sw + x0c) * 3 + c], v01 = src[(y0c * sw + x1c) * 3 + c];
                    float v10 = src[(y1c * sw + x0c) * 3 + c], v11 = src[(y1c * sw + x1c) * 3 + c];
                    float top = v00 + (v01 - v00) * fx;
                    float bot = v10 + (v11 - v10) * fx;
                    dst[dOff + c] = (top + (bot - top) * fy) / 255f;
                }
            }
        }
        return dst;
    }
}
