namespace SpawnDev.ILGPU.ML.StableDiffusion;

/// <summary>
/// The host-side pieces of ComfyUI's guidance patches that work on the sampler's latents (which the caller keeps), for one latent (batch 1).
/// </summary>
public static class SdGuidance
{
    /// <summary>
    /// ComfyUI's Self-Attention Guidance blur map (nodes_sag.py create_blur_map): <paramref name="x0"/> [C, lh, lw] (the unconditional x0 prediction) blurred
    /// (gaussian_blur_2d: 9 taps, <paramref name="sigma"/>, reflect padding) where the middle-block attention mass (<see cref="SdUNet.ReadMiddleAttentionAsync"/>)
    /// is above <paramref name="threshold"/>, unchanged elsewhere. The mass is laid out on ComfyUI's grid: x rows from round(sqrt(lh / lw * tokens)), searched
    /// outward for a divisor, then upsampled (nearest) to the latent. ComfyUI's SAG then runs the unconditional pass on <c>degraded + x - x0</c> and adds
    /// <c>(degraded - sag) * scale</c> to the CFG result. Defaults: ComfyUI's node (blur_sigma 2, threshold 1).
    /// </summary>
    public static float[] SagBlur(float[] x0, int C, int lh, int lw, float[] mass, float sigma = 2f, float threshold = 1f)
    {
        if (x0.Length != (long)C * lh * lw) throw new ArgumentException($"x0 must be [{C}, {lh}, {lw}]", nameof(x0));
        int total = mass.Length;
        // the grid of the mask (create_blur_map's search)
        int x = (int)Math.Round(Math.Sqrt((double)lh / lw * total), MidpointRounding.ToEven);
        int? xx = null;
        for (int i = 0; i < (int)Math.Floor(Math.Sqrt(total) / 2) && xx == null; i++)
            foreach (var j in new[] { x + i, Math.Max(1, x - i) })
                if (total % j == 0) { xx = j; break; }
        if (xx == null) throw new InvalidOperationException($"no grid for {total} attention tokens");   // ComfyUI fails here as well (x = None)
        int gx = xx.Value, gy = total / gx;
        // nearest upsampling (torch's legacy "nearest": src = floor(dst * in / out) in fp32)
        var mask = new float[lh * lw];
        float sy = (float)gx / lh, sx = (float)gy / lw;
        for (int r = 0; r < lh; r++)
            for (int c = 0; c < lw; c++)
            {
                int my = Math.Min((int)MathF.Floor(r * sy), gx - 1), mx = Math.Min((int)MathF.Floor(c * sx), gy - 1);
                mask[r * lw + c] = mass[my * gy + mx] > threshold ? 1f : 0f;
            }
        // gaussian_blur_2d(x0, 9, sigma): separable kernel normalised in fp32, reflect padding of 4
        const int K = 9, half = K / 2;
        var k1 = new float[K];
        float ks = 0f;
        for (int t = 0; t < K; t++) { float v = -4f + t; k1[t] = MathF.Exp(-0.5f * (v / sigma) * (v / sigma)); ks += k1[t]; }
        for (int t = 0; t < K; t++) k1[t] /= ks;
        if (lh <= half || lw <= half) throw new ArgumentException($"reflect padding of {half} needs latent sides above {half}");
        static int Reflect(int i, int n) => i < 0 ? -i : i >= n ? 2 * (n - 1) - i : i;
        var outp = new float[x0.Length];
        for (int ch = 0; ch < C; ch++)
        {
            int b0 = ch * lh * lw;
            for (int r = 0; r < lh; r++)
                for (int c = 0; c < lw; c++)
                {
                    float acc = 0f;
                    for (int a = 0; a < K; a++)
                    {
                        int rr = Reflect(r + a - half, lh);
                        for (int bb = 0; bb < K; bb++) acc += k1[a] * k1[bb] * x0[b0 + rr * lw + Reflect(c + bb - half, lw)];
                    }
                    float m = mask[r * lw + c];
                    outp[b0 + r * lw + c] = acc * m + x0[b0 + r * lw + c] * (1f - m);
                }
        }
        return outp;
    }
}
