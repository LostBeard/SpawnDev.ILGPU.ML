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
    /// <summary>The most chunk maxima a row may have for <see cref="FloorGroupImpl"/> (they are cached in shared memory).</summary>
    public const int FloorGroupMaxChunks = 1024;

    /// <summary>Threads a group of <see cref="FloorGroupImpl"/>; launch <see cref="FloorGroups"/> groups of it.</summary>
    public const int FloorGroupThreads = 64;
    /// <summary>Groups for <see cref="FloorGroupImpl"/>: each row's max(nch, K) chunk slots in blocks of <see cref="FloorGroupThreads"/>.</summary>
    public static int FloorGroups(int rows, int nch) => rows * ((Math.Max(nch, K) + FloorGroupThreads - 1) / FloorGroupThreads);

    /// <summary>
    /// <see cref="FloorImpl"/> with the row's chunk maxima cached in shared memory (each group loads them once) and each thread's rank count split over
    /// four independent counters (the comparisons are independent, so the count is the same): a thread per chunk slot as FloorImpl, the same slots,
    /// values and fill. Launch <see cref="FloorGroups"/> groups of <see cref="FloorGroupThreads"/>. Needs nch &lt;= <see cref="FloorGroupMaxChunks"/>.
    /// MEASURED 2026-11-09 (RTX 4070, 594 chunks): FloorImpl, every thread re-reading every maximum from device memory in one dependent chain, 28.8 us; a
    /// group per row with the same chain from shared memory, 26.0 us.
    /// </summary>
    public static void FloorGroupImpl(FV maxV, IV maxI, FV o, IV selN, int nch, int S)
    {
        int w = Math.Max(nch, K), nb = (w + FloorGroupThreads - 1) / FloorGroupThreads;
        int g = Grid.IdxX, t = Group.IdxX, D = Group.DimX, r = g / nb, ch = (g % nb) * D + t;
        long ob = (long)r * S, mb = (long)r * nch;
        var mv = SharedMemory.Allocate<float>(FloorGroupMaxChunks);
        for (int c = t; c < nch; c += D) mv[c] = maxV[mb + c];
        if (ch == 0) selN[r] = 0;
        Group.Barrier();
        if (ch >= w) return;
        bool mine = ch < nch;
        float v = mine ? mv[ch] : float.MinValue;
        mine &= v > float.MinValue;
        int r0 = 0, r1 = 0, r2 = 0, r3 = 0, v0 = 0, v1 = 0, v2 = 0, v3 = 0, c4 = 0;
        for (; c4 + 4 <= nch; c4 += 4)
        {
            float u0 = mv[c4], u1 = mv[c4 + 1], u2 = mv[c4 + 2], u3 = mv[c4 + 3];
            bool k0 = u0 > float.MinValue, k1 = u1 > float.MinValue, k2 = u2 > float.MinValue, k3 = u3 > float.MinValue;
            if (k0) v0++; if (k1) v1++; if (k2) v2++; if (k3) v3++;
            if (k0 & ((u0 > v) | ((u0 == v) & (c4 < ch)))) r0++;
            if (k1 & ((u1 > v) | ((u1 == v) & (c4 + 1 < ch)))) r1++;
            if (k2 & ((u2 > v) | ((u2 == v) & (c4 + 2 < ch)))) r2++;
            if (k3 & ((u3 > v) | ((u3 == v) & (c4 + 3 < ch)))) r3++;
        }
        for (; c4 < nch; c4++)
        {
            float u = mv[c4];
            bool k = u > float.MinValue;
            if (k) v0++;
            if (k & ((u > v) | ((u == v) & (c4 < ch)))) r0++;
        }
        int rank = r0 + r1 + r2 + r3, valid = v0 + v1 + v2 + v3;
        if (mine & (rank < K)) { o[ob + rank] = v; o[ob + K + rank] = maxI[mb + ch]; }
        if ((ch < K) & (ch >= valid)) { o[ob + ch] = float.MinValue; o[ob + K + ch] = -1f; }
    }

    /// <summary>The group size of <see cref="ChunkMaxGroupImpl"/> / <see cref="SelectGroupImpl"/>: one thread per value of a chunk, so the chunk is this long.</summary>
    public const int GroupChunk = 256;

    /// <summary>
    /// <see cref="ChunkMaxImpl"/> with a group per chunk and a thread per value (launch rows x nch groups of <see cref="GroupChunk"/>): the chunk read
    /// coalesced, its best (larger, or equal with the lower index) found by a shared-memory tree. The same maximum and index as the serial scan, which
    /// keeps the first of equal values (a thread past the row's end holds -Infinity at index int.MaxValue: it never beats a value). MEASURED
    /// 2026-11-09 (RTX 4070, 151,936 logits, 594 chunks): the serial scan took 61.8 us, its threads 1 KB apart.
    /// </summary>
    public static void ChunkMaxGroupImpl(FV logits, FV maxV, IV maxI, int V, int nch)
    {
        int g = Grid.IdxX, t = Group.IdxX, chunk = Group.DimX, r = g / nch, ch = g % nch, idx = ch * chunk + t;
        var sv = SharedMemory.Allocate<float>(GroupChunk);
        var si = SharedMemory.Allocate<int>(GroupChunk);
        bool inRow = idx < V;
        sv[t] = inRow ? logits[(long)r * V + idx] : float.NegativeInfinity;
        si[t] = inRow ? idx : int.MaxValue;
        Group.Barrier();
        for (int w = chunk / 2; w > 0; w >>= 1)
        {
            if (t < w)
            {
                float u = sv[t + w], v = sv[t];
                int ui = si[t + w];
                if ((u > v) | ((u == v) & (ui < si[t]))) { sv[t] = u; si[t] = ui; }
            }
            Group.Barrier();
        }
        if (t == 0) { maxV[g] = sv[0]; maxI[g] = si[0]; }
    }

    /// <summary>
    /// <see cref="SelectImpl"/> with a group per chunk and a thread per value (launch rows x nch groups of <see cref="GroupChunk"/>): the same values
    /// gathered (the gather's order was never defined: <see cref="FinalImpl"/> ranks them), and the chunk's softmax mass summed by a shared-memory tree
    /// (pairwise: the serial loop's terms in another order, no less accurate). MEASURED 2026-11-09 (RTX 4070, 151,936 logits): the serial loop took 76.6 us.
    /// </summary>
    public static void SelectGroupImpl(FV logits, FV o, IV selN, FV selV, IV selI, int V, int nch, int S, float T)
    {
        int g = Grid.IdxX, t = Group.IdxX, chunk = Group.DimX, r = g / nch, ch = g % nch, idx = ch * chunk + t;
        var red = SharedMemory.Allocate<float>(GroupChunk);
        long ob = (long)r * S;
        float floorK = o[ob + K - 1], max = o[ob], floorT = max - 20f * T, e = 0f;
        if (idx < V)
        {
            float v = logits[(long)r * V + idx];
            if (v >= floorK)
            {
                int slot = Atomic.Add(ref selN[r], 1);
                if (slot < SelectCap) { selV[(long)r * SelectCap + slot] = v; selI[(long)r * SelectCap + slot] = idx; }
            }
            if (T > 0f && v >= floorT) e = MathF.Exp((v - max) / T);
        }
        red[t] = e;
        Group.Barrier();
        for (int w = chunk / 2; w > 0; w >>= 1)
        {
            if (t < w) red[t] += red[t + w];
            Group.Barrier();
        }
        if (t == 0) o[ob + 2 * K + 1 + ch] = red[0];
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
