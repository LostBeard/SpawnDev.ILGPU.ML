using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.ILGPU.ML.Trellis2;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// The portable TRELLIS.2 kernels (register-tiled GEMM, gathered sparse convolution, flash attention) on EVERY backend against a host double
/// reference. These are the kernels the browser runs (no tensor cores there), so the WebGPU lane is the one that matters most: the
/// generated kernels are large and fully unrolled, and the WGSL transpiler has to get all of it right. Shapes are deliberately ragged
/// (not multiples of the 128 tile, of the 64-query group, or of the key tile) so every bounds path runs.
/// </summary>
public abstract partial class MLTestBase
{
    private static float Bf16Round(float f, out ushort bits)
    {
        uint u = BitConverter.SingleToUInt32Bits(f);
        bits = (ushort)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
        return BitConverter.UInt32BitsToSingle((uint)bits << 16);
    }

    private static float Silu(double v) => (float)(v / (1 + Math.Exp(-v)));

    /// <summary>
    /// TRELLIS.2 targets WebGPU in the browser and every desktop backend. WebGL has no atomics (the voxel hash) and no shared memory; Wasm frees a
    /// buffer that its worker pool may still be reading once a synchronous stage boundary only flushes (no synchronous drain exists there).
    /// </summary>
    private static void RequireTrellisBackend(Accelerator a)
    {
        if (a.AcceleratorType is AcceleratorType.WebGL or AcceleratorType.Wasm)
            throw new UnsupportedTestException("TRELLIS.2 runs on WebGPU and the desktop backends (not WebGL / Wasm)");
    }

    [TestMethod]
    public async Task Trellis2_TiledLinear_MatchesHostReference() => await RunTest(async accelerator =>
    {
        RequireTrellisBackend(accelerator);
        const int M = 200, K = 72, N = 136;     // ragged on purpose: K is a multiple of 8 but not of the tensor-core 32, so CUDA takes the tiled kernel too
        var rng = new Random(7);
        var x = new float[M * K]; for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 2 - 1);
        var wf = new float[N * K]; var wb = new BFloat16[N * K];
        for (int i = 0; i < wf.Length; i++) { wf[i] = Bf16Round((float)(rng.NextDouble() - 0.5), out var b); wb[i] = BFloat16.FromRawBits(b); }
        var bf = new float[N]; var bb = new BFloat16[N];
        for (int i = 0; i < N; i++) { bf[i] = Bf16Round((float)(rng.NextDouble() - 0.5), out var b); bb[i] = BFloat16.FromRawBits(b); }
        using var xb = accelerator.Allocate1D(x);
        using var wbuf = accelerator.Allocate1D(wb);
        using var bbuf = accelerator.Allocate1D(bb);
        using var o = accelerator.Allocate1D<float>(M * N);
        new DiTKernels(accelerator).Linear(xb.View, wbuf.View, bbuf.View, o.View, M, K, N, DiTKernels.Act.Silu);
        var got = await o.View.CopyToHostAsync();
        double maxErr = 0;
        for (int m = 0; m < M; m++)
            for (int n = 0; n < N; n++)
            {
                double acc = bf[n];
                for (int k = 0; k < K; k++) acc += (double)x[m * K + k] * wf[n * K + k];
                maxErr = Math.Max(maxErr, Math.Abs(got[m * N + n] - Silu(acc)));
            }
        if (maxErr > 2e-4) throw new Exception($"tiled linear max |err| {maxErr:E2} on {accelerator.AcceleratorType}");
        return $"tiled linear {M}x{K}x{N} max |err| {maxErr:E2}";
    });

    [TestMethod]
    public async Task Trellis2_TiledSparseConv_MatchesHostReference() => await RunTest(async accelerator =>
    {
        RequireTrellisBackend(accelerator);
        const int G = 10, cin = 8, cout = 72;    // 27 * 8 = 216 inputs per output: the gathered tiled kernel on every backend
        var rng = new Random(11);
        var cells = new List<int>();
        for (int i = 0; i < G * G * G; i++) if (rng.NextDouble() < 0.35) cells.Add(i);
        int n = cells.Count;
        var coords = new int[n * 3];
        for (int i = 0; i < n; i++) { int c = cells[i]; coords[i * 3] = c / (G * G); coords[i * 3 + 1] = (c / G) % G; coords[i * 3 + 2] = c % G; }
        var x = new float[n * cin]; for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 2 - 1);
        int K = 27 * cin;
        var wf = new float[cout * K]; var wh = new global::ILGPU.Half[cout * K];
        for (int i = 0; i < wf.Length; i++) { wh[i] = (global::ILGPU.Half)(float)(rng.NextDouble() - 0.5); wf[i] = (float)wh[i]; }
        var bf = new float[cout]; var bh = new global::ILGPU.Half[cout];
        for (int i = 0; i < cout; i++) { bh[i] = (global::ILGPU.Half)(float)(rng.NextDouble() - 0.5); bf[i] = (float)bh[i]; }
        var sk = new SparseKernels(accelerator);
        using var cb = accelerator.Allocate1D(coords);
        using var nbr = accelerator.Allocate1D<int>((long)n * 27);
        sk.BuildNeighbors(cb.View, n, G, nbr.View);
        using var xb = accelerator.Allocate1D(x);
        using var wbuf = accelerator.Allocate1D(wh);
        using var bbuf = accelerator.Allocate1D(bh);
        using var o = accelerator.Allocate1D<float>((long)n * cout);
        sk.Conv3(xb.View, nbr.View, wbuf.View, bbuf.View, o.View, n, cin, cout);
        var got = await o.View.CopyToHostAsync();
        var nb = await nbr.View.CopyToHostAsync();
        double maxErr = 0;
        for (int m = 0; m < n; m++)
            for (int co = 0; co < cout; co++)
            {
                double acc = bf[co];
                for (int tap = 0; tap < 27; tap++)
                {
                    int src = nb[m * 27 + tap];
                    if (src < 0) continue;
                    for (int ci = 0; ci < cin; ci++) acc += (double)wf[co * K + tap * cin + ci] * x[src * cin + ci];
                }
                maxErr = Math.Max(maxErr, Math.Abs(got[m * cout + co] - acc));
            }
        if (maxErr > 2e-4) throw new Exception($"tiled sparse conv max |err| {maxErr:E2} on {accelerator.AcceleratorType}");
        return $"tiled sparse conv {n} voxels {cin}->{cout} max |err| {maxErr:E2}";
    });

    /// <summary>
    /// Device mesh clustering (float atomics are CAS loops on WebGPU) on a finely tessellated sphere: the mesh must shrink a lot, every
    /// output vertex must stay on the surface (curvature makes the coarse levels fail the error test, so this also proves cells are refused),
    /// the area must survive and every face must be valid and non-degenerate.
    /// </summary>
    [TestMethod]
    public async Task Trellis2_MeshClusterer_SphereStaysOnSurface() => await RunTest(async accelerator =>
    {
        RequireTrellisBackend(accelerator);
        const int grid = 128, seg = 240, rings = 120;
        const double R = 0.4;                                   // 51.2 voxels
        var pos = new List<float>(); var tri = new List<int>();
        pos.AddRange(new[] { 0f, (float)R, 0f });
        for (int r = 1; r < rings; r++)
        {
            double th = Math.PI * r / rings;
            for (int s = 0; s < seg; s++) { double ph = 2 * Math.PI * s / seg; pos.AddRange(new[] { (float)(R * Math.Sin(th) * Math.Cos(ph)), (float)(R * Math.Cos(th)), (float)(R * Math.Sin(th) * Math.Sin(ph)) }); }
        }
        pos.AddRange(new[] { 0f, (float)-R, 0f });
        int south = pos.Count / 3 - 1;
        int V(int r, int s) => 1 + (r - 1) * seg + (s % seg);
        for (int s = 0; s < seg; s++) tri.AddRange(new[] { 0, V(1, s + 1), V(1, s) });
        for (int r = 1; r < rings - 1; r++)
            for (int s = 0; s < seg; s++) { tri.AddRange(new[] { V(r, s), V(r, s + 1), V(r + 1, s) }); tri.AddRange(new[] { V(r, s + 1), V(r + 1, s + 1), V(r + 1, s) }); }
        for (int s = 0; s < seg; s++) tri.AddRange(new[] { south, V(rings - 1, s), V(rings - 1, s + 1) });
        int nv = pos.Count / 3, nf = tri.Count / 3;
        using var vb = accelerator.Allocate1D(pos.ToArray());
        using var fb = accelerator.Allocate1D(tri.ToArray());
        var (p, f) = await new MeshClusterer(accelerator, new SparseKernels(accelerator)).ClusterAsync(vb.View, nv, fb.View, nf, grid);
        int outV = p.Length / 3, outF = f.Length / 3;
        if (outF == 0 || outF > nf / 4) throw new Exception($"{nf} -> {outF} faces: expected at least a 4x reduction on {accelerator.AcceleratorType}");
        double maxDev = 0;
        for (int v = 0; v < outV; v++)
        {
            double r = Math.Sqrt(p[v * 3] * p[v * 3] + p[v * 3 + 1] * p[v * 3 + 1] + p[v * 3 + 2] * p[v * 3 + 2]);
            if (double.IsNaN(r)) throw new Exception($"vertex {v} is NaN");
            maxDev = Math.Max(maxDev, Math.Abs(r - R) * grid);
        }
        if (maxDev > 1.0) throw new Exception($"a clustered vertex is {maxDev:F2} voxels off the sphere on {accelerator.AcceleratorType}");
        double area = 0;
        for (int t = 0; t < outF; t++)
        {
            int a = f[t * 3], b = f[t * 3 + 1], c = f[t * 3 + 2];
            if (a < 0 || b < 0 || c < 0 || a >= outV || b >= outV || c >= outV || a == b || b == c || a == c) throw new Exception($"face {t} invalid ({a},{b},{c})");
            double ux = p[b * 3] - p[a * 3], uy = p[b * 3 + 1] - p[a * 3 + 1], uz = p[b * 3 + 2] - p[a * 3 + 2];
            double wx = p[c * 3] - p[a * 3], wy = p[c * 3 + 1] - p[a * 3 + 1], wz = p[c * 3 + 2] - p[a * 3 + 2];
            double cx = uy * wz - uz * wy, cy = uz * wx - ux * wz, cz = ux * wy - uy * wx;
            area += 0.5 * Math.Sqrt(cx * cx + cy * cy + cz * cz);
        }
        double want = 4 * Math.PI * R * R;
        if (Math.Abs(area - want) / want > 0.03) throw new Exception($"area {area:F5} vs sphere {want:F5} on {accelerator.AcceleratorType}");
        return $"sphere {nf} -> {outF} faces, {outV} vertices, max deviation {maxDev:F3} voxels, area {area / want:P2}";
    });

    [TestMethod]
    public async Task Trellis2_FlashAttention_MatchesHostReference() => await RunTest(async accelerator =>
    {
        RequireTrellisBackend(accelerator);
        var dk = new DiTKernels(accelerator);
        var report = new List<string>();
        foreach (int D in new[] { 128, 64 })
        {
            if (!dk.UsesFlashAttention(D)) throw new UnsupportedTestException($"no flash attention for D={D} on this backend");
            foreach (bool packed in new[] { false, true })
            {
                const int Nq = 70, Nk = 50, H = 2;
                int C = H * D;
                var rng = new Random(D + (packed ? 1 : 0));
                float[] Rand(int count) { var a = new float[count]; for (int i = 0; i < count; i++) a[i] = (float)(rng.NextDouble() * 4 - 2); return a; }
                // packed: q, k, v are slices of one [tokens, 3C] buffer (self attention; tokens = max(Nq, Nk)); else q [Nq, C] and kv [Nk, 2C]
                int qStride = packed ? 3 * C : C, kvStride = packed ? 3 * C : 2 * C, kOff = packed ? C : 0, vOff = packed ? 2 * C : C;
                var qh = Rand((packed ? Math.Max(Nq, Nk) : Nq) * qStride);
                var kvh = packed ? qh : Rand(Nk * kvStride);
                using var qb = accelerator.Allocate1D(qh);
                using var kvb = packed ? null : accelerator.Allocate1D(kvh);
                using var o = accelerator.Allocate1D<float>((long)Nq * C);
                dk.Attention(qb.View, qStride, 0, packed ? qb.View : kvb!.View, kvStride, kOff, vOff, default, o.View, Nq, Nk, H, D);
                var got = await o.View.CopyToHostAsync();
                double maxErr = 0, scale = 1 / Math.Sqrt(D);
                var p = new double[Nk];
                for (int h = 0; h < H; h++)
                    for (int i = 0; i < Nq; i++)
                    {
                        double mx = double.MinValue;
                        for (int j = 0; j < Nk; j++)
                        {
                            double s = 0;
                            for (int d = 0; d < D; d++) s += (double)qh[i * qStride + h * D + d] * kvh[j * kvStride + kOff + h * D + d];
                            p[j] = s * scale; mx = Math.Max(mx, p[j]);
                        }
                        double sum = 0; for (int j = 0; j < Nk; j++) { p[j] = Math.Exp(p[j] - mx); sum += p[j]; }
                        for (int d = 0; d < D; d++)
                        {
                            double acc = 0; for (int j = 0; j < Nk; j++) acc += p[j] * kvh[j * kvStride + vOff + h * D + d];
                            maxErr = Math.Max(maxErr, Math.Abs(got[i * C + h * D + d] - acc / sum));
                        }
                    }
                // CUDA at D = 128 runs the bf16 tensor-core kernel (inputs rounded to bf16); every other case is the fp32 portable kernel
                double tol = accelerator.AcceleratorType == AcceleratorType.Cuda && D == 128 && DiTKernels.EnableTensorCores ? 3e-2 : 1e-4;
                if (maxErr > tol) throw new Exception($"flash attention D={D} {(packed ? "self" : "cross")} max |err| {maxErr:E2} on {accelerator.AcceleratorType}");
                report.Add($"D{D} {(packed ? "self" : "cross")} {maxErr:E1}");
            }
        }
        return "flash attention: " + string.Join(", ", report);
    });

    /// <summary>
    /// Throughput of the DiT's kernels at the real 1.3B shapes (width 1536, 12 heads x 128, MLP 6144, 4096 tokens = the sparse-structure
    /// flow): how far each backend is from the silicon. Prints "[Benchmark] Trellis2 ..." lines (PMT_CONSOLE_LOG=Trellis2 shows them for browser lanes).
    /// </summary>
    [TestMethod]
    public async Task Trellis2_DiTKernels_Benchmark() => await RunTest(async accelerator =>
    {
        RequireTrellisBackend(accelerator);
        const int T = 4096, C = 1536, H = 12, D = 128, Mlp = 6144;
        var dk = new DiTKernels(accelerator);
        using var x = accelerator.Allocate1D(Enumerable.Range(0, T * Mlp).Select(i => (float)Math.Sin(i * 0.37)).ToArray());
        using var w = accelerator.Allocate1D(Enumerable.Range(0, Mlp * C).Select(i => BFloat16.FromRawBits((ushort)(0x3C00 + (i % 255)))).ToArray());
        using var o = accelerator.Allocate1D<float>((long)T * Math.Max(Mlp, 3 * C));
        var report = new List<string>();
        bool slow = accelerator.AcceleratorType == AcceleratorType.CPU;
        async Task<double> Time(Action op, int reps = 10)
        {
            if (slow) reps = 1;
            for (int i = 0; i < (slow ? 1 : 3); i++) { op(); await accelerator.SynchronizeAsync(); }   // compile + warm (and let the clocks ramp)
            var sw = System.Diagnostics.Stopwatch.StartNew();
            for (int r = 0; r < reps; r++) op();
            await accelerator.SynchronizeAsync();
            return sw.Elapsed.TotalSeconds / reps;
        }
        // "k8": one k step over the mlp-up tile count = the per-tile fixed cost (prologue + epilogue) alone
        for (int pass = 0; pass < (slow ? 1 : 2); pass++)                   // report the second pass: everything compiled, clocks up
        foreach (var (name, K, N) in new[] { ("qkv", C, 3 * C), ("mlp-up", C, Mlp), ("mlp-down", Mlp, C), ("k8", 8, Mlp) })
        {
            if (pass == 0 && !slow) { await Time(() => dk.Linear(x.View.SubView(0, (long)T * K), w.View.SubView(0, (long)N * K), null, o.View.SubView(0, (long)T * N), T, K, N), 2); continue; }
            double s = await Time(() => dk.Linear(x.View.SubView(0, (long)T * K), w.View.SubView(0, (long)N * K), null, o.View.SubView(0, (long)T * N), T, K, N));
            double tf = 2.0 * T * K * N / s / 1e12;
            report.Add($"{name} {s * 1000:F1} ms {tf:F2} TFLOPS");
        }
        if (dk.UsesFlashAttention(D))
        {
            using var qkv = accelerator.Allocate1D(Enumerable.Range(0, T * 3 * C).Select(i => (float)Math.Cos(i * 0.11) * 0.5f).ToArray());
            double s = await Time(() => dk.Attention(qkv.View, 3 * C, 0, qkv.View, 3 * C, C, 2 * C, default, o.View.SubView(0, (long)T * C), T, T, H, D));
            double tf = 4.0 * T * T * H * D / s / 1e12;
            report.Add($"self-attention {s * 1000:F1} ms {tf:F2} TFLOPS ({dk.TiledAttentionKind})");
        }
        var line = $"[Benchmark] Trellis2 DiT kernels [{accelerator.AcceleratorType}] " + string.Join(" | ", report);
        Console.WriteLine(line);
        return line;
    });
}
