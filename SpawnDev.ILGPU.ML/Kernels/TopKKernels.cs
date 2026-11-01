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
    /// <summary>Threads for <see cref="FloorImpl"/>: one per chunk maximum of each row (at least <see cref="K"/>, so every slot is written).</summary>
    public static int FloorThreads(int rows, int nch) => rows * Math.Max(nch, K);

    // the row's K best chunk maxima into o's value / id slots (o[0] is then the row's best value); the K-th of them is the floor: those K values all reach it.
    // One thread per chunk (FloorThreads): its maximum's rank among the row's (better = larger, or equal from an earlier chunk) is its slot when below K;
    // thread j also fills slot j when fewer than j + 1 maxima are above float.MinValue. The gather count is zeroed here, before the gather runs. (It was one thread per row
    // insertion-sorting all the maxima through device memory: about a millisecond per draw on an RTX 4070, measured 2026-11-01.)
    public static void FloorImpl(Index1D i, FV maxV, IV maxI, FV o, IV selN, int nch, int S)
    {
        int w = Math.Max(nch, K), ii = i, r = ii / w, ch = ii % w;
        long ob = (long)r * S, mb = (long)r * nch;
        if (ch == 0) selN[r] = 0;
        // only maxima above float.MinValue take a slot (-Infinity never reaches the floor): with fewer than K of them the floor stays float.MinValue
        bool mine = ch < nch;
        float v = mine ? maxV[mb + ch] : float.MinValue;
        mine &= v > float.MinValue;
        int rank = 0, valid = 0;
        for (int c = 0; c < nch; c++)
        {
            float u = maxV[mb + c];
            bool counts = u > float.MinValue;
            if (counts) valid++;
            bool better = counts & ((u > v) | ((u == v) & (c < ch)));
            if (better) rank++;
        }
        if (mine & (rank < K)) { o[ob + rank] = v; o[ob + K + rank] = maxI[mb + ch]; }
        // the slots no maximum takes
        if ((ch < K) & (ch >= valid)) { o[ob + ch] = float.MinValue; o[ob + K + ch] = -1f; }
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
    /// <summary>Threads for <see cref="FinalImpl"/>: one per gather slot of each row.</summary>
    public static int FinalThreads(int rows) => rows * SelectCap;

    // the row's gathered values sorted into its K best (skipped when more than SelectCap reached the floor). One thread per gathered value (FinalThreads):
    // its rank among the gathered (better = larger, or equal with the lower index; indices are distinct) is its slot when below K; the slots none
    // takes already hold FloorImpl's fill. (It was one thread per row insertion-sorting the gathered values through device memory.)
    public static void FinalImpl(Index1D i, FV selV, IV selI, IV selN, FV o, int S)
    {
        int ii = i, r = ii / SelectCap, s = ii % SelectCap, n = selN[r];
        long ob = (long)r * S, sb = (long)r * SelectCap;
        if (s == 0) o[ob + 2 * K] = n;
        if (n > SelectCap) return;
        if (s >= n) return;                 // the slots past the gathered keep FloorImpl's float.MinValue / -1 fill (n is at least its count of valid maxima)
        float v = selV[sb + s]; int id = selI[sb + s];
        int rank = 0;
        for (int t = 0; t < n; t++)
        {
            float u = selV[sb + t];
            bool better = (u > v) | ((u == v) & (selI[sb + t] < id));
            if (better) rank++;
        }
        if (rank < K) { o[ob + rank] = v; o[ob + K + rank] = id; }
    }
}
