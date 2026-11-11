using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.GGUF;
using SpawnDev.ILGPU.ML.Kernels;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    /// <summary>
    /// The tensor-core K-quant GEMMs (<see cref="MmaKQuantGemm"/>: a prefill's M &gt; 1 rows of Q4_K and Q6_K on CUDA sm_75+) against the CPU dequant oracle: shapes
    /// whose M and N do not fill the 128 x 128 tiles (M = 200 / N = 300, M = 16 / N = 130, M = 1000 / N = 257) over K = 4096, 1024, 2048 (several super-blocks:
    /// every Q4_K sub-block scale form j &lt; 4 and j &gt;= 4, every Q6_K half and quarter). The operands are rounded to f16, so each output's error is bounded by
    /// 2e-3 x sum |a_k w_k| (f16 keeps 11 significant bits; the bound is ~4x the two roundings). It must be the kernel that ran; a 16-byte-misaligned input must
    /// take the SIMT kernels and still match. Both variants: the sm_80+ kernels (m16n8k16) where the card has them, and the Turing kernels (sm_75: two m16n8k8 per
    /// 16 k, scalar f16 packs) on any sm_75+ card (<see cref="FusedDequantMatMul.ForceSm75Mma"/> runs them on a newer one). Added 2026-11-02 (Q4_K), Q6_K the
    /// same day.
    /// </summary>
    /// <summary>
    /// FusedDequantMatMul.ForwardGateUpSwiGLUDown (GateUpSwiGLUDown: SwiGLU in the gate | up GEMM's epilogue, its f16 output straight into the
    /// down GEMM) against the separate path (ForwardGroup gate | up, ActivationKernels.SwiGLU, Forward down): bitwise, Q4_K and Q6_K mixes, a
    /// partial row tile. CUDA sm_80+. Added 2026-11-05.
    /// </summary>
    [TestMethod]
    public async Task Gemm_MmaKQuant_SwiGLUDownMatchesSeparate() => await RunTest(async accelerator =>
    {
        if (!CudaCaps.TensorCores(accelerator)) throw new UnsupportedTestException("the fused FFN GEMM runs on CUDA sm_80+");
        using var fused = new FusedDequantMatMul(accelerator);
        var act = new ActivationKernels(accelerator);
        var notes = new List<string>();
        int scrMinSaved = FusedDequantMatMul.MmaScratchMinRows;
        bool forceSaved = FusedDequantMatMul.ForceSm75Mma, t75Saved = FusedDequantMatMul.EnableMmaSm75V5;
        FusedDequantMatMul.MmaScratchMinRows = 1;
        FusedDequantMatMul.EnableMmaSm75V5 = true;   // the Turing form exists on the Turing V5 kernels
        try
        {
            foreach (bool turing in new[] { false, true })
            foreach (var (M, K, Nff, Nout, tg, tu, td) in new[] { (300, 512, 768, 264, GGMLType.Q4_K, GGMLType.Q4_K, GGMLType.Q6_K),
                (77, 256, 512, 512, GGMLType.Q6_K, GGMLType.Q4_K, GGMLType.Q4_K) })
            {
                FusedDequantMatMul.ForceSm75Mma = turing;
                var rng = new Random(301 + M + K);
                var input = new float[M * K];
                // small inputs: the random blocks' scales are large, and SwiGLU of O(100) values overflows f16 (the comparison must see finite values)
                for (int i = 0; i < input.Length; i++) input[i] = (float)(rng.NextDouble() * 2 - 1) * 0.01f;
                byte[] Weight(GGMLType t, int rows, int cols)
                {
                    int rb = RowBytes(t, cols);
                    var b = new byte[rows * rb];
                    for (int n = 0; n < rows; n++) Buffer.BlockCopy(MakeBlocks(t, cols, rng), 0, b, n * rb, rb);
                    return b;
                }
                using var inBuf = accelerator.Allocate1D(input);
                using var gw = AllocatePadded(accelerator, Weight(tg, Nff, K));
                using var uw = AllocatePadded(accelerator, Weight(tu, Nff, K));
                using var dw = AllocatePadded(accelerator, Weight(td, Nout, Nff));
                using var g = accelerator.Allocate1D<float>((long)M * Nff); using var u = accelerator.Allocate1D<float>((long)M * Nff);
                using var sv = accelerator.Allocate1D<float>((long)M * Nff);
                using var want = accelerator.Allocate1D<float>((long)M * Nout); using var got = accelerator.Allocate1D<float>((long)M * Nout);
                if (!fused.ForwardGroup(inBuf.View, new[] { gw.View, uw.View }, new[] { tg, tu }, new[] { Nff, Nff }, new[] { g.View, u.View }, M, K))
                    throw new Exception("the grouped GEMM was not taken");
                act.SwiGLU(g.View, u.View, sv.View, M * Nff);
                fused.Forward(sv.View, dw.View, want.View, M, Nff, Nout, td);
                if (!fused.ForwardGateUpSwiGLUDown(inBuf.View, gw.View, tg, uw.View, tu, dw.View, td, got.View, M, K, Nff, Nout) || fused.LastGemmKernel != "mma-scratch-swiglu" + (turing ? "-sm75" : ""))
                    throw new Exception($"M={M} K={K} Nff={Nff}: the fused FFN GEMM was not taken ('{fused.LastGemmKernel}')");
                await accelerator.SynchronizeAsync();
                var a = await want.CopyToHostAsync<float>(0, (long)M * Nout);
                var b2 = await got.CopyToHostAsync<float>(0, (long)M * Nout);
                for (int e = 0; e < a.Length; e++)
                    if (BitConverter.SingleToInt32Bits(a[e]) != BitConverter.SingleToInt32Bits(b2[e]))
                        throw new Exception($"M={M} K={K} Nff={Nff} Nout={Nout}: output {e} = {b2[e]:R}, separate {a[e]:R}");
                int finite = a.Count(v => float.IsFinite(v) && v != 0f);
                if (finite < a.Length * 9 / 10) throw new Exception($"only {finite} of {a.Length} outputs finite and nonzero: the comparison proves little");
                notes.Add($"{(turing ? "sm75 " : "")}M={M} K={K} Nff={Nff} Nout={Nout} {tg}/{tu}/{td}: bitwise ({finite} finite nonzero, max |y| {a.Max(Math.Abs):G3})");
            }
        }
        finally { FusedDequantMatMul.MmaScratchMinRows = scrMinSaved; FusedDequantMatMul.ForceSm75Mma = forceSaved; FusedDequantMatMul.EnableMmaSm75V5 = t75Saved; }
        return "ForwardGateUpSwiGLUDown vs group + SwiGLU + down: " + string.Join("; ", notes);
    });

    /// <summary>
    /// FusedDequantMatMul.ForwardGroup (MatMulGroup: sibling projections of one input as one scratch GEMM with the output columns split per
    /// tensor) against separate <see cref="FusedDequantMatMul.Forward"/> calls: bitwise, for a q | k | v-like triple (Q4_K, Q6_K, Q4_K with
    /// different widths) and a gate | up-like pair; and it must be taken (LastGemmKernel). CUDA sm_80+ (the V6 split kernel). Added 2026-11-05.
    /// </summary>
    [TestMethod]
    public async Task Gemm_MmaKQuant_GroupMatchesSeparate() => await RunTest(async accelerator =>
    {
        if (!CudaCaps.TensorCores(accelerator)) throw new UnsupportedTestException("the grouped scratch GEMM runs on CUDA sm_80+");
        using var fused = new FusedDequantMatMul(accelerator);
        var notes = new List<string>();
        int scrMinSaved = FusedDequantMatMul.MmaScratchMinRows;
        bool forceSaved = FusedDequantMatMul.ForceSm75Mma, t75Saved = FusedDequantMatMul.EnableMmaSm75V5;
        FusedDequantMatMul.MmaScratchMinRows = 1;
        FusedDequantMatMul.EnableMmaSm75V5 = true;
        try
        {
            foreach (bool turing in new[] { false, true })
            foreach (var (M, K, parts) in new[] { (600, 1024, new[] { (GGMLType.Q4_K, 256), (GGMLType.Q6_K, 128), (GGMLType.Q4_K, 136) }),
                (130, 512, new[] { (GGMLType.Q4_K, 264), (GGMLType.Q4_K, 264) }) })
            {
                FusedDequantMatMul.ForceSm75Mma = turing;
                var rng = new Random(91 + M + K);
                var input = new float[M * K];
                for (int i = 0; i < input.Length; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);
                using var inBuf = accelerator.Allocate1D(input);
                var bufs = new List<MemoryBuffer1D<byte, Stride1D.Dense>>();
                var outs = new List<MemoryBuffer1D<float, Stride1D.Dense>>();
                var sep = new List<float[]>();
                try
                {
                    foreach (var (type, N) in parts)
                    {
                        int rb = RowBytes(type, K);
                        var bytes = new byte[N * rb];
                        for (int n = 0; n < N; n++) Buffer.BlockCopy(MakeBlocks(type, K, rng), 0, bytes, n * rb, rb);
                        var wb = AllocatePadded(accelerator, bytes);
                        bufs.Add(wb);
                        var ob = accelerator.Allocate1D<float>((long)M * N);
                        outs.Add(ob);
                        fused.Forward(inBuf.View, wb.View, ob.View, M, K, N, type);
                        await accelerator.SynchronizeAsync();
                        sep.Add(await ob.CopyToHostAsync<float>(0, (long)M * N));
                        ob.MemSetToZero();
                    }
                    bool taken = fused.ForwardGroup(inBuf.View, bufs.Select(b => b.View).ToList(), parts.Select(x => x.Item1).ToList(), parts.Select(x => x.Item2).ToList(),
                        outs.Select(o => o.View).ToList(), M, K);
                    if (!taken || fused.LastGemmKernel != "mma-scratch-group" + parts.Length + (turing ? "-sm75" : ""))
                        throw new Exception($"M={M} K={K}: the grouped GEMM was not taken (returned {taken}, ran '{fused.LastGemmKernel}')");
                    await accelerator.SynchronizeAsync();
                    for (int i = 0; i < parts.Length; i++)
                    {
                        var got = await outs[i].CopyToHostAsync<float>(0, (long)M * parts[i].Item2);
                        for (int e = 0; e < got.Length; e++)
                            if (BitConverter.SingleToInt32Bits(got[e]) != BitConverter.SingleToInt32Bits(sep[i][e]))
                                throw new Exception($"M={M} K={K} part {i} ({parts[i].Item1}, N={parts[i].Item2}): output {e} = {got[e]:R}, separate {sep[i][e]:R}");
                    }
                    notes.Add($"{(turing ? "sm75 " : "")}M={M} K={K} [{string.Join(", ", parts.Select(x => $"{x.Item1} {x.Item2}"))}]: bitwise");
                }
                finally { foreach (var b in bufs) b.Dispose(); foreach (var o in outs) o.Dispose(); }
            }
        }
        finally { FusedDequantMatMul.MmaScratchMinRows = scrMinSaved; FusedDequantMatMul.ForceSm75Mma = forceSaved; FusedDequantMatMul.EnableMmaSm75V5 = t75Saved; }
        return "ForwardGroup vs separate scratch GEMMs: " + string.Join("; ", notes);
    });

    [TestMethod]
    public async Task Gemm_MmaKQuant_MatchesOracle_AndIsTaken() => await RunTest(async accelerator =>
    {
        if (!CudaCaps.TuringTensorCores(accelerator)) throw new UnsupportedTestException("the tensor-core GEMMs run on CUDA sm_75+");
        var forces = CudaCaps.TensorCores(accelerator) ? new[] { false, true } : new[] { true };
        bool forceSaved = FusedDequantMatMul.ForceSm75Mma, v2Saved = FusedDequantMatMul.EnableMmaV2, v3Saved = FusedDequantMatMul.EnableMmaV3; bool v5Saved = FusedDequantMatMul.EnableMmaV5, v5t75Saved = FusedDequantMatMul.EnableMmaSm75V5, v6Saved = FusedDequantMatMul.EnableMmaV6;
        bool scrSaved = FusedDequantMatMul.EnableMmaScratch;
        int v3MinSaved = FusedDequantMatMul.MmaV3MinRows, scrMinSaved = FusedDequantMatMul.MmaScratchMinRows;
        FusedDequantMatMul.MmaV3MinRows = 1;   // every shape on V3 too
        FusedDequantMatMul.MmaScratchMinRows = 1;   // and on the scratch-operand GEMM
        using var fused = new FusedDequantMatMul(accelerator);
        var notes = new List<string>();
        try
        {
        foreach (var type in new[] { GGMLType.Q4_K, GGMLType.Q6_K })
        foreach (bool force in forces)
        foreach (var (M, K, N) in new[] { (200, 4096, 300), (16, 1024, 130), (1000, 2048, 257) })
        {
            FusedDequantMatMul.ForceSm75Mma = force;
            string want = (type == GGMLType.Q4_K ? "mma-q4k" : "mma-q6k") + (force ? "-sm75" : "");
            var rng = new Random(13 + M + K + N + (int)type);
            int bytesPerRow = RowBytes(type, K);
            var weightBytes = new byte[N * bytesPerRow];
            var wRows = new float[N][];
            for (int n = 0; n < N; n++)
            {
                var rowBytes = MakeBlocks(type, K, rng);
                Buffer.BlockCopy(rowBytes, 0, weightBytes, n * bytesPerRow, bytesPerRow);
                wRows[n] = ReferenceDequant(type, rowBytes, K);
            }
            var input = new float[M * K + 4];
            for (int i = 0; i < input.Length; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);
            (float[] Want, float[] Bound) Oracle(int off)
            {
                var o = new float[M * N]; var bound = new float[M * N];
                Parallel.For(0, M, m =>
                {
                    for (int n = 0; n < N; n++)
                    {
                        double s = 0, a = 0;
                        for (int k = 0; k < K; k++) { double p = input[off + m * K + k] * wRows[n][k]; s += p; a += Math.Abs(p); }
                        o[m * N + n] = (float)s; bound[m * N + n] = (float)(2e-3 * a + 1e-5);
                    }
                });
                return (o, bound);
            }
            static int Bad(float[] got, (float[] Want, float[] Bound) o, out float worst)
            {
                int bad = 0; worst = 0;
                for (int i = 0; i < got.Length; i++)
                {
                    float e = Math.Abs(got[i] - o.Want[i]);
                    if (!(e <= o.Bound[i])) bad++;
                    worst = Math.Max(worst, e / o.Bound[i]);
                }
                return bad;
            }
            using var weightBuf = AllocatePadded(accelerator, weightBytes);
            using var inputBuf = accelerator.Allocate1D(input);
            using var outBuf = accelerator.Allocate1D<float>((long)M * N);
            // aligned: the tensor-core kernel, V1, V2 (double-buffered, 64-bit fragment loads) and V3 (256-row tiles): the same rounding and MMA
            // order per output, so V2 and V3 must be bit-identical to V1
            float[] got = Array.Empty<float>(); string route = "";
            foreach (var (v2, v3, scr, v5, v6, sfx0) in new[] { (false, false, false, false, false, ""), (true, false, false, false, false, "-v2"),
                (true, true, false, false, false, "-v3"), (true, true, true, false, false, "-scratch"), (true, true, true, true, false, "-scratch-v5"),
                (true, true, true, true, true, "-scratch-v6") })
            {
                // V5 (64 x 64 warp tiles): the sm_80 cp.async kernel, or the Turing register-staged one on the forced sm_75 path
                // V6 is sm_80 only: the forced sm_75 path runs its V5 there
                string sfx = v6 && want.Contains("sm75") ? "-scratch-v5" : sfx0;
                FusedDequantMatMul.EnableMmaV6 = v6;
                FusedDequantMatMul.KqSplitK = false;   // the bit-identity ladder: one K order per output (split-K is its own arm below)
                FusedDequantMatMul.EnableMmaV2 = v2;
                FusedDequantMatMul.EnableMmaV3 = v3;
                FusedDequantMatMul.EnableMmaScratch = scr;
                FusedDequantMatMul.EnableMmaV5 = v5;
                FusedDequantMatMul.EnableMmaSm75V5 = v5;
                fused.Forward(inputBuf.View.SubView(0, (long)M * K), weightBuf.View, outBuf.View, M, K, N, type);
                string r = fused.LastGemmKernel;
                await accelerator.SynchronizeAsync();
                var g = await outBuf.CopyToHostAsync<float>(0, (long)M * N);
                if (r != want + sfx) throw new Exception($"{type} M={M} K={K} N={N}: an aligned GEMM ran on '{r}', not {want}{sfx}");
                if (sfx != "")
                {
                    int diff = 0; for (int i = 0; i < g.Length; i++) if (BitConverter.SingleToInt32Bits(g[i]) != BitConverter.SingleToInt32Bits(got[i])) diff++;
                    if (diff > 0) throw new Exception($"{type} M={M} K={K} N={N} {want}{sfx}: differs from V1 in {diff} of {g.Length} outputs");
                }
                else got = g;
                route = r;
            }
            int bad = Bad(got, Oracle(0), out float worst);
            if (bad > 0) throw new Exception($"{type} M={M} K={K} N={N} {want}: {bad} of {got.Length} outputs past 2e-3 x sum|a w| (worst {worst:F2}x the bound)");
            // split-K (sm_80 V2, a grid smaller than the SM count: every shape here): its own K order, so held to the oracle's bound, not to V1's bits
            string splitNote = "";
            if (!force)
            {
                FusedDequantMatMul.KqSplitK = true; FusedDequantMatMul.EnableMmaV2 = true; FusedDequantMatMul.EnableMmaV3 = false; FusedDequantMatMul.EnableMmaScratch = false;
                fused.Forward(inputBuf.View.SubView(0, (long)M * K), weightBuf.View, outBuf.View, M, K, N, type);
                string rs = fused.LastGemmKernel;
                await accelerator.SynchronizeAsync();
                var gs = await outBuf.CopyToHostAsync<float>(0, (long)M * N);
                bool small = ((M + 127) / 128) * ((N + 127) / 128) * 2 <= accelerator.NumMultiprocessors;   // split only when one wave holds 2+ slices per tile
                string wantS = want + "-v2" + (small ? "-splitk" : "");
                if (rs != wantS) throw new Exception($"{type} M={M} K={K} N={N}: ran '{rs}', not {wantS}");
                int badS = Bad(gs, Oracle(0), out float worstS);
                if (badS > 0) throw new Exception($"{type} M={M} K={K} N={N} {rs}: {badS} of {gs.Length} outputs past the bound (worst {worstS:F2}x)");
                splitNote = $", {rs} {worstS:F3}x";
            }
            // misaligned by one float: the SIMT kernels, the same answer
            FusedDequantMatMul.EnableMmaScratch = false;
            fused.Forward(inputBuf.View.SubView(1, (long)M * K), weightBuf.View, outBuf.View, M, K, N, type);
            string route2 = fused.LastGemmKernel;
            await accelerator.SynchronizeAsync();
            var got2 = await outBuf.CopyToHostAsync<float>(0, (long)M * N);
            if (route2.StartsWith("mma")) throw new Exception($"{type} M={M} K={K} N={N}: a misaligned input took the tensor-core kernel (its loads assume 16-byte alignment)");
            int bad2 = Bad(got2, Oracle(1), out float worst2);
            if (bad2 > 0) throw new Exception($"{type} M={M} K={K} N={N}: SIMT GEMM (misaligned input), {bad2} of {got2.Length} outputs past the bound (worst {worst2:F2}x)");
            notes.Add($"{type} M={M} K={K} N={N}: {route} worst {worst:F3}x the bound{splitNote}, {route2} {worst2:F3}x");
        }
        }
        finally
        {
            FusedDequantMatMul.ForceSm75Mma = forceSaved; FusedDequantMatMul.EnableMmaV2 = v2Saved; FusedDequantMatMul.EnableMmaV3 = v3Saved; FusedDequantMatMul.KqSplitK = true;
            FusedDequantMatMul.MmaV3MinRows = v3MinSaved; FusedDequantMatMul.EnableMmaScratch = scrSaved; FusedDequantMatMul.MmaScratchMinRows = scrMinSaved;
            FusedDequantMatMul.EnableMmaV5 = v5Saved; FusedDequantMatMul.EnableMmaSm75V5 = v5t75Saved; FusedDequantMatMul.EnableMmaV6 = v6Saved;
        }
        return string.Join("; ", notes);
    });

    /// <summary>
    /// A prefill's Q8_0 rows on the tensor cores (CUDA sm_75+: rows dequantized to f16, then the 16-bit GEMM) against the CPU dequant oracle, on the sm_80
    /// kernels and the forced Turing ones; the route must be the Q8_0 scratch GEMM. Each output within 2e-3 of the sum of |x w| (f16 rounding of x and of
    /// d x q). Added 2026-11-05 with the route (qwen3:1.7b Q8_0 prefill: 1,265 tok/s on the SIMT GEMM, Ollama 16,141).
    /// </summary>
    [TestMethod]
    public async Task Gemm_MmaQ8_0_MatchesOracle_AndIsTaken() => await RunTest(async accelerator =>
    {
        if (!CudaCaps.TuringTensorCores(accelerator)) throw new UnsupportedTestException("the tensor-core GEMMs run on CUDA sm_75+");
        var forces = CudaCaps.TensorCores(accelerator) ? new[] { false, true } : new[] { true };
        bool forceSaved = FusedDequantMatMul.ForceSm75Mma;
        int scrMinSaved = FusedDequantMatMul.MmaScratchMinRows;
        FusedDequantMatMul.MmaScratchMinRows = 1;
        using var fused = new FusedDequantMatMul(accelerator);
        var notes = new List<string>();
        var type = GGMLType.Q8_0;
        try
        {
            foreach (bool force in forces)
            foreach (var (M, K, N) in new[] { (200, 2048, 300), (16, 1024, 130), (1000, 6144, 257) })
            {
                FusedDequantMatMul.ForceSm75Mma = force;
                var rng = new Random(29 + M + K + N);
                int bytesPerRow = RowBytes(type, K);
                var weightBytes = new byte[N * bytesPerRow];
                var wRows = new float[N][];
                for (int n = 0; n < N; n++)
                {
                    var rowBytes = MakeBlocks(type, K, rng);
                    Buffer.BlockCopy(rowBytes, 0, weightBytes, n * bytesPerRow, bytesPerRow);
                    wRows[n] = ReferenceDequant(type, rowBytes, K);
                }
                var input = new float[M * K];
                for (int i = 0; i < input.Length; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);
                var want = new float[M * N]; var bound = new float[M * N];
                Parallel.For(0, M, m =>
                {
                    for (int n = 0; n < N; n++)
                    {
                        double sum = 0, a = 0;
                        for (int k = 0; k < K; k++) { double p = input[m * K + k] * wRows[n][k]; sum += p; a += Math.Abs(p); }
                        want[m * N + n] = (float)sum; bound[m * N + n] = (float)(2e-3 * a + 1e-5);
                    }
                });
                using var weightBuf = AllocatePadded(accelerator, weightBytes);
                using var inputBuf = accelerator.Allocate1D(input);
                using var outBuf = accelerator.Allocate1D<float>((long)M * N);
                fused.Forward(inputBuf.View, weightBuf.View, outBuf.View, M, K, N, type);
                string route = fused.LastGemmKernel;
                await accelerator.SynchronizeAsync();
                var got = await outBuf.CopyToHostAsync<float>(0, (long)M * N);
                string wantRoute = "mma-q8_0" + (force ? "-sm75" : "") + "-scratch";
                if (!route.StartsWith(wantRoute)) throw new Exception($"M={M} K={K} N={N}: ran on '{route}', not {wantRoute}*");
                int bad = 0; float worst = 0;
                for (int i = 0; i < got.Length; i++)
                {
                    float e = Math.Abs(got[i] - want[i]);
                    if (!(e <= bound[i])) bad++;
                    worst = Math.Max(worst, e / bound[i]);
                }
                if (bad > 0) throw new Exception($"{route} M={M} K={K} N={N}: {bad} of {got.Length} outputs outside 2e-3 sum|xw| (worst {worst:F2} of the bound)");
                notes.Add($"{route} M={M} K={K} N={N}: worst {worst:F3} of the bound");
            }
        }
        finally { FusedDequantMatMul.ForceSm75Mma = forceSaved; FusedDequantMatMul.MmaScratchMinRows = scrMinSaved; }
        return string.Join("; ", notes);
    });
}
