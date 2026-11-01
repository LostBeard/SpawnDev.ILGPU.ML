using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Kernels;

using FV = ArrayView1D<float, Stride1D.Dense>;
using IV = ArrayView1D<int, Stride1D.Dense>;

/// <summary>
/// Exact top-<see cref="K"/> of each row of values on the device (GgufBatchDecoder's candidates, DeviceVectorIndex's search): each chunk's best, the
/// row's K best chunk maxima (the K-th is a floor at least K values reach), the values at or above it gathered (atomic slots: not WebGL), then sorted.
/// Output row o (stride S >= 2K + 1 + chunks): [0, K) values best first, [K, 2K) their indices as floats (exact below 2^24), 2K the count that reached
/// the floor (more than <see cref="SelectCap"/>: the gather overflowed and the row must be ranked another way), 2K + 1 + chunk the chunk's softmax mass
/// at T (T > 0). Better = larger, or equal with the lower index. -Infinity never reaches the floor (it is below float.MinValue): a row with fewer than K
/// finite values ends in float.MinValue / -1 slots. Choose the chunk so the row has at least K chunks, or the floor drops to float.MinValue and every
/// finite value is gathered.
/// </summary>
internal static class TopKKernels
{
    public const int K = 64;
    public const int SelectCap = 4096;

    // each chunk's largest value and its first index
    public static void ChunkMaxImpl(Index1D i, FV logits, FV maxV, IV maxI, int V, int nch, int chunk)
    {
        int ii = i, r = ii / nch, ch = ii % nch, end = Math.Min(V, (ch + 1) * chunk), bi = ch * chunk;
        float best = logits[(long)r * V + bi];
        for (int idx = bi + 1; idx < end; idx++) { float v = logits[(long)r * V + idx]; if (v > best) { best = v; bi = idx; } }
        maxV[ii] = best; maxI[ii] = bi;
    }
    // the row's K best chunk maxima into o's value / id slots (o[0] is then the row's best value); the K-th of them is the floor: those K values all reach it.
    // The gather count is zeroed here, before the gather runs.
    public static void FloorImpl(Index1D row, FV maxV, IV maxI, FV o, IV selN, int nch, int S)
    {
        int r = row;
        long ob = (long)r * S;
        for (int j = 0; j < K; j++) { o[ob + j] = float.MinValue; o[ob + K + j] = -1f; }
        for (int ch = 0; ch < nch; ch++)
        {
            float v = maxV[(long)r * nch + ch];
            if (v <= o[ob + K - 1]) continue;
            int pos = K - 1;
            while (pos > 0 && o[ob + pos - 1] < v) { o[ob + pos] = o[ob + pos - 1]; o[ob + K + pos] = o[ob + K + pos - 1]; pos--; }
            o[ob + pos] = v; o[ob + K + pos] = maxI[(long)r * nch + ch];
        }
        selN[r] = 0;
    }
    // each chunk's values at or above the floor appended to the row's list (in any order), and (T > 0) the chunk's softmax mass at T over the values within
    // e^20 of the best (the set GgufBatchDecoder.SampleLogits draws from)
    public static void SelectImpl(Index1D i, FV logits, FV o, IV selN, FV selV, IV selI, int V, int nch, int S, float T, int chunk)
    {
        int ii = i, r = ii / nch, ch = ii % nch, end = Math.Min(V, (ch + 1) * chunk);
        long ob = (long)r * S;
        float floorK = o[ob + K - 1], max = o[ob], floorT = max - 20f * T, sum = 0f;
        for (int idx = ch * chunk; idx < end; idx++)
        {
            float v = logits[(long)r * V + idx];
            if (v >= floorK)
            {
                int slot = Atomic.Add(ref selN[r], 1);
                if (slot < SelectCap) { selV[(long)r * SelectCap + slot] = v; selI[(long)r * SelectCap + slot] = idx; }
            }
            if (T > 0f && v >= floorT) sum += MathF.Exp((v - max) / T);
        }
        o[ob + 2 * K + 1 + ch] = sum;
    }
    // the row's gathered values sorted into its K best (skipped when more than SelectCap reached the floor)
    public static void FinalImpl(Index1D row, FV selV, IV selI, IV selN, FV o, int S)
    {
        int r = row, n = selN[r];
        long ob = (long)r * S, sb = (long)r * SelectCap;
        o[ob + 2 * K] = n;
        if (n > SelectCap) return;
        for (int j = 0; j < K; j++) { o[ob + j] = float.MinValue; o[ob + K + j] = -1f; }
        for (int s = 0; s < n; s++)
        {
            float v = selV[sb + s]; int id = selI[sb + s];
            float lv = o[ob + K - 1];
            if (v < lv || (v == lv && o[ob + 2 * K - 1] >= 0f && id > (int)o[ob + 2 * K - 1])) continue;
            int pos = K - 1;
            while (pos > 0)
            {
                float pv = o[ob + pos - 1];
                if (pv > v || (pv == v && (int)o[ob + K + pos - 1] < id)) break;
                o[ob + pos] = pv; o[ob + K + pos] = o[ob + K + pos - 1]; pos--;
            }
            o[ob + pos] = v; o[ob + K + pos] = id;
        }
    }
}
