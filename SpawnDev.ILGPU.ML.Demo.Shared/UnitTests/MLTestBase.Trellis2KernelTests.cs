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
}
