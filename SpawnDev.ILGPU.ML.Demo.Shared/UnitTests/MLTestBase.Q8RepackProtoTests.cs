using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.GGUF;
using SpawnDev.ILGPU.ML.Kernels;
using SpawnDev.UnitTesting;
using System.Diagnostics;
using W16 = SpawnDev.ILGPU.ML.Kernels.FusedDequantMatMul.W16;
using F4 = SpawnDev.ILGPU.ML.Kernels.Vec4LoadMatMul.F4;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    /// <summary>
    /// "Measure twice" for a repacked Q8_0 layout (2026-11-06, TJ: go for it). WebGPU decode is 79% Q8_0 GEMV at ~355 GB/s (Bench_Qwen3_1_7B_WebGPU_DecodeStepProfile):
    /// the 34-byte blocks are 2-byte aligned, so every read is a 32-bit word recombined across a block boundary. Repacked: a matrix's int8 quants row-major
    /// (K bytes a row, 16-byte aligned) and its fp16 block scales in their own region, two to a word; same 1.0625 bytes a weight. Two kernels on it, A (a
    /// group of 64 per output, one 16-byte load per 16 values) and B (a warp per 4 outputs sharing its inputs), against today's one-row kernel on the native
    /// layout, on decode shapes. Every launch reads a fresh slice of a 65,536 x 2,048 matrix (~140 MB per layout, past the 4070's 36 MB L2), so the numbers are
    /// DRAM-bound as in a real decode token. A prototype: it decides whether the layout is worth building into the library.
    /// </summary>
    [TestMethod(Timeout = 900000)]
    public async Task Bench_Q8RepackedLayout_Gemv() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is not (AcceleratorType.Cuda or AcceleratorType.WebGPU or AcceleratorType.OpenCL))
            throw new UnsupportedTestException("a measurement for the fast lanes");
        const int K = 2048, Rows = 65536, Distinct = 64;
        int rb = RowBytes(GGMLType.Q8_0, K);                 // 2176
        var rng = new Random(17);
        // 64 distinct random rows, native and repacked
        var native = new byte[Distinct * rb];
        var quants = new byte[Distinct * K];
        var scales = new ushort[Distinct * (K / 32)];
        for (int r = 0; r < Distinct; r++)
        {
            var row = MakeBlocks(GGMLType.Q8_0, K, rng);
            Buffer.BlockCopy(row, 0, native, r * rb, rb);
            for (int b = 0; b < K / 32; b++)
            {
                scales[r * (K / 32) + b] = (ushort)(row[b * 34] | (row[b * 34 + 1] << 8));
                Buffer.BlockCopy(row, b * 34 + 2, quants, r * K + b * 32, 32);
            }
        }
        // grouped: per 8 blocks (272 bytes, the same span the native blocks occupy) the 8 fp16 scales, then the 256 int8 quants
        var grouped = new byte[native.Length];
        for (int r = 0; r < Distinct; r++)
            for (int g = 0; g < K / 256; g++)
            {
                int gb = r * rb + g * 272;
                for (int j = 0; j < 8; j++)
                {
                    grouped[gb + 2 * j] = native[gb + 34 * j]; grouped[gb + 2 * j + 1] = native[gb + 34 * j + 1];
                    Buffer.BlockCopy(native, gb + 34 * j + 2, grouped, gb + 16 + 32 * j, 32);
                }
            }
        var scaleInts = new int[scales.Length / 2];
        for (int i = 0; i < scaleInts.Length; i++) scaleInts[i] = scales[2 * i] | (scales[2 * i + 1] << 16);
        var quantInts = new int[quants.Length / 4];
        Buffer.BlockCopy(quants, 0, quantInts, 0, quants.Length);

        long nativeBytes = (long)Rows * rb, qInts = (long)Rows * K / 4, sInts = (long)Rows * (K / 32) / 2;
        using var nat = accelerator.Allocate1D<byte>(nativeBytes);
        using var rep = accelerator.Allocate1D<int>(qInts + sInts);
        // the 64 distinct rows in small source buffers, copied device-to-device across the big ones (row r = distinct row r % 64; WebGPU cannot copy a
        // buffer onto itself)
        using var natSrc = accelerator.Allocate1D<byte>(native.Length);
        using var grp = accelerator.Allocate1D<byte>(nativeBytes);
        using var grpSrc = accelerator.Allocate1D<byte>(grouped.Length);
        grpSrc.View.CopyFromCPU(grouped);
        using var qSrc = accelerator.Allocate1D(quantInts);
        using var sSrc = accelerator.Allocate1D(scaleInts);
        natSrc.View.CopyFromCPU(native);
        for (long r0 = 0; r0 < Rows; r0 += Distinct)
        {
            nat.View.SubView(r0 * rb, (long)Distinct * rb).CopyFrom(natSrc.View);
            grp.View.SubView(r0 * rb, (long)Distinct * rb).CopyFrom(grpSrc.View);
            rep.View.SubView(r0 * K / 4, (long)Distinct * K / 4).CopyFrom(qSrc.View);
            rep.View.SubView(qInts + r0 * (K / 32) / 2, (long)Distinct * (K / 32) / 2).CopyFrom(sSrc.View);
            if ((r0 / Distinct) % 128 == 127) await accelerator.SynchronizeAsync();   // keep the browser queue short
        }
        var x = new float[K];
        for (int i = 0; i < K; i++) x[i] = (float)(rng.NextDouble() * 2 - 1);
        using var xBuf = accelerator.Allocate1D(x);
        using var outBuf = accelerator.Allocate1D<float>(Rows);
        await accelerator.SynchronizeAsync();

        using var fused = new FusedDequantMatMul(accelerator);
        var kA = accelerator.LoadStreamKernel<ArrayView<F4>, ArrayView<W16>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int>(RepackGemvA);
        var kB = accelerator.LoadStreamKernel<ArrayView<F4>, ArrayView<W16>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int>(RepackGemvB);
        var kC = accelerator.LoadStreamKernel<ArrayView<F4>, ArrayView<W16>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int>(RepackGemvC);
        var kG = accelerator.LoadStreamKernel<ArrayView<F4>, ArrayView<W16>, ArrayView1D<float, Stride1D.Dense>, int, int>(RepackGemvG);
        var kD = accelerator.LoadStreamKernel<ArrayView<F4>, ArrayView<W16>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int>(RepackGemvD);
        var x4 = xBuf.View.BaseView.Cast<F4>();
        void Launch(int which, int r0, int N)
        {
            var o = outBuf.View.SubView(0, N);
            if (which == 0) { fused.ForwardRows(xBuf.View, nat.View.SubView((long)r0 * rb, (long)N * rb), o, 1, K, N, GGMLType.Q8_0); return; }
            var q16 = rep.View.SubView((long)r0 * K / 4, (long)N * K / 4).BaseView.Cast<W16>();
            var sc = rep.View.SubView(qInts + (long)r0 * (K / 32) / 2, (long)N * (K / 32) / 2);
            if (which == 1) kA(new KernelConfig(N, 64), x4, q16, sc, o, K, N);
            else if (which == 2) kB(new KernelConfig((N + 7) / 8, 64), x4, q16, sc, o, K, N);
            else if (which == 3) kC(new KernelConfig((N + 15) / 16, 64), x4, q16, sc, o, K, N);
            else if (which == 4) kD(new KernelConfig((N + 7) / 8, 64), x4, q16, sc, o, K, N);
            else
            {
                var gv = grp.View.SubView((long)r0 * rb, (long)N * rb);
                kG(new KernelConfig((N + 7) / 8, 64), x4, gv.BaseView.Cast<W16>(), o, K, N);
            }
        }

        // correctness: A and B against today's kernel on one slice
        const int W = 6;
        var outs = new float[W][];
        for (int w = 0; w < W; w++) { Launch(w, 0, 2048); await accelerator.SynchronizeAsync(); outs[w] = await outBuf.View.SubView(0, 2048).CopyToHostAsync(); }
        for (int w = 1; w < W; w++) AssertCloseQuant(outs[w], outs[0], 1e-3f, $"repacked {"xABCDG"[w]} vs native");

        var lines = new List<string>();
        string baseName = fused.LastRowsKernel;
        foreach (var (label, N) in new[] { ("o-proj", 2048), ("q|k", 3072), ("gate|up", 12288), ("down K=2048 stand-in", 6144) })
        {
            double bytes = (double)N * K * 1.0625;
            var ms = new double[W];
            int slices = Math.Max(1, (Rows - N) / N);
            for (int w = 0; w < W; w++)
            {
                for (int i = 0; i < 3; i++) Launch(w, (i % slices) * N, N);   // warm
                await accelerator.SynchronizeAsync();
                const int R = 40;
                if (accelerator is SpawnDev.ILGPU.WebGPU.WebGPUAccelerator wacc)
                {
                    // the browser's per-dispatch host cost (~130 us uncaptured) would swamp the GPU time: record the R launches into a plan and read the
                    // GPU time per kernel from a timestamped replay (PMT's Chrome has --enable-webgpu-developer-features: not quantized)
                    bool caching = SpawnDev.ILGPU.WebGPU.Backend.WebGPUBackend.EnableBindGroupCaching;
                    SpawnDev.ILGPU.WebGPU.Backend.WebGPUBackend.EnableBindGroupCaching = false;
                    SpawnDev.ILGPU.WebGPU.WebGPUDispatchPlan plan;
                    try
                    {
                        plan = wacc.BeginDispatchCapture();
                        for (int i = 0; i < R; i++) Launch(w, (i % slices) * N, N);
                        wacc.EndDispatchCapture();
                    }
                    finally { SpawnDev.ILGPU.WebGPU.Backend.WebGPUBackend.EnableBindGroupCaching = caching; }
                    using (plan)
                    {
                        await plan.ReplayAsync(); await accelerator.SynchronizeAsync();
                        using var doc = System.Text.Json.JsonDocument.Parse(await plan.ReplayTimedAsync());
                        double sum = 0; int cnt = 0;
                        foreach (var k in doc.RootElement.GetProperty("kernels").EnumerateArray())
                            if (k.GetProperty("label").GetString()!.Contains("Gemv")) { sum += k.GetProperty("ms").GetDouble(); cnt += k.GetProperty("count").GetInt32(); }
                        ms[w] = cnt > 0 ? sum / cnt : double.NaN;
                    }
                }
                else
                {
                    var sw = Stopwatch.StartNew();
                    for (int i = 0; i < R; i++) Launch(w, (i % slices) * N, N);
                    await accelerator.SynchronizeAsync();
                    ms[w] = sw.Elapsed.TotalMilliseconds / R;
                }
            }
            string Gb(double m) => $"{bytes / (m * 1e6):F0} GB/s";
            var line = $"{accelerator.AcceleratorType} {label} N={N}: native {baseName} {ms[0] * 1000:F1} us ({Gb(ms[0])}), A {Gb(ms[1])}, B {Gb(ms[2])}, C(8 rows) {Gb(ms[3])}, D(2 chunks) {Gb(ms[4])}, G(grouped in place) {Gb(ms[5])}";
            Console.WriteLine("[Benchmark] Q8Repack " + line);
            lines.Add(line);
        }
        return string.Join("; ", lines);
    });

    // fp16 bits -> float (normal / subnormal; block scales are never inf / nan)
    private static float RpHalf(int h)
    {
        int exp = (h >> 10) & 0x1F, man = h & 0x3FF;
        float r = exp == 0 ? man * (1f / 16777216f) : Interop.IntAsFloat((uint)(((exp + 112) << 23) | (man << 13)));
        return (h & 0x8000) != 0 ? -r : r;
    }
    private static float RpI8x4(int q, F4 x) => x.A * ((q << 24) >> 24) + x.B * ((q << 16) >> 24) + x.C * ((q << 8) >> 24) + x.D * (q >> 24);
    // 16 values of row n at chunk c (16-byte load), times its block scale
    private static float RpChunk(ArrayView<W16> q, ArrayView1D<int, Stride1D.Dense> sc, int n, int c, int K, F4 x0, F4 x1, F4 x2, F4 x3)
    {
        W16 w = q.AsAligned16()[(long)n * (K >> 4) + c];
        int b = n * (K >> 5) + (c >> 1);
        int sw = sc[b >> 1];
        float d = RpHalf((b & 1) != 0 ? (sw >> 16) & 0xFFFF : sw & 0xFFFF);
        return d * (RpI8x4(w.a, x0) + RpI8x4(w.b, x1) + RpI8x4(w.c, x2) + RpI8x4(w.d, x3));
    }

    // A: a group of 64 per output row, each thread one 16-value chunk per step, a shared-memory tree at the end
    private static void RepackGemvA(ArrayView<F4> x4, ArrayView<W16> q, ArrayView1D<int, Stride1D.Dense> sc, ArrayView1D<float, Stride1D.Dense> o, int K, int N)
    {
        int n = Grid.IdxX, tid = Group.IdxX;
        float acc = 0f;
        int nC = K >> 4;
        for (int c = tid; c < nC; c += 64)
            acc += RpChunk(q, sc, n, c, K, x4.AsAligned16()[4 * c], x4.AsAligned16()[4 * c + 1], x4.AsAligned16()[4 * c + 2], x4.AsAligned16()[4 * c + 3]);
        var sh = SharedMemory.Allocate<float>(64);
        sh[tid] = acc;
        Group.Barrier();
        for (int s = 32; s > 0; s >>= 1) { if (tid < s) sh[tid] += sh[tid + s]; Group.Barrier(); }
        if (tid == 0 && n < N) o[n] = sh[0];
    }

    // B: a warp per 4 output rows (2 warps a group), a lane per 16-value chunk, its inputs loaded once for the 4 rows
    private static void RepackGemvB(ArrayView<F4> x4, ArrayView<W16> q, ArrayView1D<int, Stride1D.Dense> sc, ArrayView1D<float, Stride1D.Dense> o, int K, int N)
    {
        int tid = Group.IdxX, lane = tid & 31, n0 = (Grid.IdxX * 2 + (tid >> 5)) * 4;
        float p0 = 0f, p1 = 0f, p2 = 0f, p3 = 0f;
        int nC = K >> 4;
        bool r0 = n0 < N, r1 = n0 + 1 < N, r2 = n0 + 2 < N, r3 = n0 + 3 < N;
        if (r0)
            for (int c = lane; c < nC; c += 32)
            {
                var x0 = x4.AsAligned16()[4 * c]; var x1 = x4.AsAligned16()[4 * c + 1]; var x2 = x4.AsAligned16()[4 * c + 2]; var x3 = x4.AsAligned16()[4 * c + 3];
                p0 += RpChunk(q, sc, n0, c, K, x0, x1, x2, x3);
                if (r1) p1 += RpChunk(q, sc, n0 + 1, c, K, x0, x1, x2, x3);
                if (r2) p2 += RpChunk(q, sc, n0 + 2, c, K, x0, x1, x2, x3);
                if (r3) p3 += RpChunk(q, sc, n0 + 3, c, K, x0, x1, x2, x3);
            }
        var sh = SharedMemory.Allocate<float>(2 * 32 * 4);
        int wb = (tid >> 5) * 128;
        sh[wb + lane] = p0; sh[wb + 32 + lane] = p1; sh[wb + 64 + lane] = p2; sh[wb + 96 + lane] = p3;
        Group.Barrier();
        for (int s = 16; s > 0; s >>= 1)
        {
            if (lane < s) for (int r = 0; r < 4; r++) sh[wb + 32 * r + lane] += sh[wb + 32 * r + lane + s];
            Group.Barrier();
        }
        if (lane < 4 && n0 + lane < N) o[n0 + lane] = sh[wb + 32 * lane];
    }

    // C: B with 8 rows a warp (inputs shared twice as widely)
    private static void RepackGemvC(ArrayView<F4> x4, ArrayView<W16> q, ArrayView1D<int, Stride1D.Dense> sc, ArrayView1D<float, Stride1D.Dense> o, int K, int N)
    {
        int tid = Group.IdxX, lane = tid & 31, n0 = (Grid.IdxX * 2 + (tid >> 5)) * 8;
        float p0 = 0f, p1 = 0f, p2 = 0f, p3 = 0f, p4 = 0f, p5 = 0f, p6 = 0f, p7 = 0f;
        int nC = K >> 4;
        if (n0 < N)
            for (int c = lane; c < nC; c += 32)
            {
                var x0 = x4.AsAligned16()[4 * c]; var x1 = x4.AsAligned16()[4 * c + 1]; var x2 = x4.AsAligned16()[4 * c + 2]; var x3 = x4.AsAligned16()[4 * c + 3];
                p0 += RpChunk(q, sc, n0, c, K, x0, x1, x2, x3);
                if (n0 + 1 < N) p1 += RpChunk(q, sc, n0 + 1, c, K, x0, x1, x2, x3);
                if (n0 + 2 < N) p2 += RpChunk(q, sc, n0 + 2, c, K, x0, x1, x2, x3);
                if (n0 + 3 < N) p3 += RpChunk(q, sc, n0 + 3, c, K, x0, x1, x2, x3);
                if (n0 + 4 < N) p4 += RpChunk(q, sc, n0 + 4, c, K, x0, x1, x2, x3);
                if (n0 + 5 < N) p5 += RpChunk(q, sc, n0 + 5, c, K, x0, x1, x2, x3);
                if (n0 + 6 < N) p6 += RpChunk(q, sc, n0 + 6, c, K, x0, x1, x2, x3);
                if (n0 + 7 < N) p7 += RpChunk(q, sc, n0 + 7, c, K, x0, x1, x2, x3);
            }
        var sh = SharedMemory.Allocate<float>(2 * 32 * 8);
        int wb = (tid >> 5) * 256;
        sh[wb + lane] = p0; sh[wb + 32 + lane] = p1; sh[wb + 64 + lane] = p2; sh[wb + 96 + lane] = p3;
        sh[wb + 128 + lane] = p4; sh[wb + 160 + lane] = p5; sh[wb + 192 + lane] = p6; sh[wb + 224 + lane] = p7;
        Group.Barrier();
        for (int s = 16; s > 0; s >>= 1)
        {
            if (lane < s) for (int r = 0; r < 8; r++) sh[wb + 32 * r + lane] += sh[wb + 32 * r + lane + s];
            Group.Barrier();
        }
        if (lane < 8 && n0 + lane < N) o[n0 + lane] = sh[wb + 32 * lane];
    }

    // D: B with two 16-value chunks a lane per step (c and c + 32: more loads in flight)
    private static void RepackGemvD(ArrayView<F4> x4, ArrayView<W16> q, ArrayView1D<int, Stride1D.Dense> sc, ArrayView1D<float, Stride1D.Dense> o, int K, int N)
    {
        int tid = Group.IdxX, lane = tid & 31, n0 = (Grid.IdxX * 2 + (tid >> 5)) * 4;
        float p0 = 0f, p1 = 0f, p2 = 0f, p3 = 0f;
        int nC = K >> 4;
        bool r1 = n0 + 1 < N, r2 = n0 + 2 < N, r3 = n0 + 3 < N;
        if (n0 < N)
            for (int c = lane; c < nC; c += 64)
            {
                bool two = c + 32 < nC;
                int c2 = two ? c + 32 : c;
                var x0 = x4.AsAligned16()[4 * c]; var x1 = x4.AsAligned16()[4 * c + 1]; var x2 = x4.AsAligned16()[4 * c + 2]; var x3 = x4.AsAligned16()[4 * c + 3];
                var y0 = x4.AsAligned16()[4 * c2]; var y1 = x4.AsAligned16()[4 * c2 + 1]; var y2 = x4.AsAligned16()[4 * c2 + 2]; var y3 = x4.AsAligned16()[4 * c2 + 3];
                float m = two ? 1f : 0f;
                p0 += RpChunk(q, sc, n0, c, K, x0, x1, x2, x3) + m * RpChunk(q, sc, n0, c2, K, y0, y1, y2, y3);
                if (r1) p1 += RpChunk(q, sc, n0 + 1, c, K, x0, x1, x2, x3) + m * RpChunk(q, sc, n0 + 1, c2, K, y0, y1, y2, y3);
                if (r2) p2 += RpChunk(q, sc, n0 + 2, c, K, x0, x1, x2, x3) + m * RpChunk(q, sc, n0 + 2, c2, K, y0, y1, y2, y3);
                if (r3) p3 += RpChunk(q, sc, n0 + 3, c, K, x0, x1, x2, x3) + m * RpChunk(q, sc, n0 + 3, c2, K, y0, y1, y2, y3);
            }
        var sh = SharedMemory.Allocate<float>(2 * 32 * 4);
        int wb = (tid >> 5) * 128;
        sh[wb + lane] = p0; sh[wb + 32 + lane] = p1; sh[wb + 64 + lane] = p2; sh[wb + 96 + lane] = p3;
        Group.Barrier();
        for (int s = 16; s > 0; s >>= 1)
        {
            if (lane < s) for (int r = 0; r < 4; r++) sh[wb + 32 * r + lane] += sh[wb + 32 * r + lane + s];
            Group.Barrier();
        }
        if (lane < 4 && n0 + lane < N) o[n0 + lane] = sh[wb + 32 * lane];
    }

    // G: B on the grouped in-place layout (per 256 values: 8 fp16 scales in a 16-byte header, then 256 int8; a row is K / 256 * 272 bytes)
    private static void RepackGemvG(ArrayView<F4> x4, ArrayView<W16> w16, ArrayView1D<float, Stride1D.Dense> o, int K, int N)
    {
        int tid = Group.IdxX, lane = tid & 31, n0 = (Grid.IdxX * 2 + (tid >> 5)) * 4;
        float p0 = 0f, p1 = 0f, p2 = 0f, p3 = 0f;
        int nC = K >> 4, rowW = (K >> 8) * 17;                         // 16-value chunks a row; 16-byte units a row
        bool r1 = n0 + 1 < N, r2 = n0 + 2 < N, r3 = n0 + 3 < N;
        if (n0 < N)
            for (int c = lane; c < nC; c += 32)
            {
                var x0 = x4.AsAligned16()[4 * c]; var x1 = x4.AsAligned16()[4 * c + 1]; var x2 = x4.AsAligned16()[4 * c + 2]; var x3 = x4.AsAligned16()[4 * c + 3];
                int hu = (c >> 4) * 17, j = c & 15;                    // the group's header unit; the chunk within the group (block j >> 1)
                p0 += RpG(w16, n0 * rowW + hu, j, x0, x1, x2, x3);
                if (r1) p1 += RpG(w16, (n0 + 1) * rowW + hu, j, x0, x1, x2, x3);
                if (r2) p2 += RpG(w16, (n0 + 2) * rowW + hu, j, x0, x1, x2, x3);
                if (r3) p3 += RpG(w16, (n0 + 3) * rowW + hu, j, x0, x1, x2, x3);
            }
        var sh = SharedMemory.Allocate<float>(2 * 32 * 4);
        int wb = (tid >> 5) * 128;
        sh[wb + lane] = p0; sh[wb + 32 + lane] = p1; sh[wb + 64 + lane] = p2; sh[wb + 96 + lane] = p3;
        Group.Barrier();
        for (int s = 16; s > 0; s >>= 1)
        {
            if (lane < s) for (int r = 0; r < 4; r++) sh[wb + 32 * r + lane] += sh[wb + 32 * r + lane + s];
            Group.Barrier();
        }
        if (lane < 4 && n0 + lane < N) o[n0 + lane] = sh[wb + 32 * lane];
    }
    // chunk j (0..15) of the group whose header is 16-byte unit gu: its quants at unit gu + 1 + j, its block scale (block j >> 1) in the header
    private static float RpG(ArrayView<W16> w16, int gu, int j, F4 x0, F4 x1, F4 x2, F4 x3)
    {
        W16 h = w16.AsAligned16()[gu];
        W16 q = w16.AsAligned16()[gu + 1 + j];
        int sel = j >> 2, sw = sel == 0 ? h.a : sel == 1 ? h.b : sel == 2 ? h.c : h.d;
        float d = RpHalf(((j >> 1) & 1) != 0 ? (sw >> 16) & 0xFFFF : sw & 0xFFFF);
        return d * (RpI8x4(q.a, x0) + RpI8x4(q.b, x1) + RpI8x4(q.c, x2) + RpI8x4(q.d, x3));
    }
}
