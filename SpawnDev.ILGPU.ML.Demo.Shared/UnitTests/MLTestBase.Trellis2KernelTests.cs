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

    [TestMethod(Timeout = 120000)]
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

    [TestMethod(Timeout = 120000)]
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

    /// <summary>An icosphere (radius 1, 4 subdivisions: 2562 vertices) whose vertices are pushed along the radius by up to +-jitter: voxel-scale bumps on a smooth curved surface. Outward winding.</summary>
    private static (float[] Positions, int[] Faces) NoisySphere(double jitter, int seed)
    {
        double g = (1 + Math.Sqrt(5)) / 2;
        var v = new List<double[]>
        {
            new[] { -1, g, 0 }, new[] { 1, g, 0 }, new[] { -1, -g, 0 }, new[] { 1, -g, 0 }, new[] { 0, -1, g }, new[] { 0, 1, g },
            new[] { 0, -1, -g }, new[] { 0, 1, -g }, new[] { g, 0, -1 }, new[] { g, 0, 1 }, new[] { -g, 0, -1 }, new[] { -g, 0, 1 },
        };
        var f = new List<int[]>
        {
            new[] { 0, 11, 5 }, new[] { 0, 5, 1 }, new[] { 0, 1, 7 }, new[] { 0, 7, 10 }, new[] { 0, 10, 11 }, new[] { 1, 5, 9 }, new[] { 5, 11, 4 }, new[] { 11, 10, 2 }, new[] { 10, 7, 6 }, new[] { 7, 1, 8 },
            new[] { 3, 9, 4 }, new[] { 3, 4, 2 }, new[] { 3, 2, 6 }, new[] { 3, 6, 8 }, new[] { 3, 8, 9 }, new[] { 4, 9, 5 }, new[] { 2, 4, 11 }, new[] { 6, 2, 10 }, new[] { 8, 6, 7 }, new[] { 9, 8, 1 },
        };
        for (int s = 0; s < 4; s++)
        {
            var mid = new Dictionary<long, int>(); var nf = new List<int[]>();
            int Mid(int a, int b)
            {
                long key = a < b ? ((long)a << 32) | (uint)b : ((long)b << 32) | (uint)a;
                if (mid.TryGetValue(key, out int m)) return m;
                v.Add(new[] { (v[a][0] + v[b][0]) / 2, (v[a][1] + v[b][1]) / 2, (v[a][2] + v[b][2]) / 2 }); return mid[key] = v.Count - 1;
            }
            foreach (var tri in f) { int a = Mid(tri[0], tri[1]), b = Mid(tri[1], tri[2]), c = Mid(tri[2], tri[0]); nf.Add(new[] { tri[0], a, c }); nf.Add(new[] { tri[1], b, a }); nf.Add(new[] { tri[2], c, b }); nf.Add(new[] { a, b, c }); }
            f = nf;
        }
        var rng = new Random(seed); var pos = new float[v.Count * 3];
        for (int i = 0; i < v.Count; i++)
        {
            double l = Math.Sqrt(v[i][0] * v[i][0] + v[i][1] * v[i][1] + v[i][2] * v[i][2]), r = 1 + jitter * (rng.NextDouble() * 2 - 1);
            for (int k = 0; k < 3; k++) pos[i * 3 + k] = (float)(v[i][k] / l * r);
        }
        return (pos, f.SelectMany(x => x).ToArray());
    }

    /// <summary>A cube [-1,1]^3 with n x n quads per side, welded along the edges, outward winding: sharp 90 degree edges between flat sides.</summary>
    private static (float[] Positions, int[] Faces) GridCube(int n)
    {
        var index = new Dictionary<(int, int, int), int>(); var pos = new List<float>(); var faces = new List<int>();
        int Vert(int x, int y, int z)
        {
            if (index.TryGetValue((x, y, z), out int i)) return i;
            index[(x, y, z)] = pos.Count / 3; pos.Add(-1f + 2f * x / n); pos.Add(-1f + 2f * y / n); pos.Add(-1f + 2f * z / n); return index[(x, y, z)];
        }
        for (int axis = 0; axis < 3; axis++)
            foreach (int side in new[] { 0, n })
                for (int a = 0; a < n; a++)
                    for (int b = 0; b < n; b++)
                    {
                        int[] P(int da, int db) { var c = new int[3]; c[axis] = side; c[(axis + 1) % 3] = a + da; c[(axis + 2) % 3] = b + db; return c; }
                        var q = new[] { P(0, 0), P(1, 0), P(1, 1), P(0, 1) }.Select(c => Vert(c[0], c[1], c[2])).ToArray();
                        faces.AddRange(new[] { q[0], q[1], q[2], q[0], q[2], q[3] });
                    }
        var p = pos.ToArray(); var fa = faces.ToArray();
        for (int f = 0; f < fa.Length / 3; f++)      // outward: the face normal must point away from the centre
        {
            int a = fa[f * 3], b = fa[f * 3 + 1], c = fa[f * 3 + 2];
            double ux = p[b * 3] - p[a * 3], uy = p[b * 3 + 1] - p[a * 3 + 1], uz = p[b * 3 + 2] - p[a * 3 + 2], vx = p[c * 3] - p[a * 3], vy = p[c * 3 + 1] - p[a * 3 + 1], vz = p[c * 3 + 2] - p[a * 3 + 2];
            double nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
            double mx = (p[a * 3] + p[b * 3] + p[c * 3]) / 3, my = (p[a * 3 + 1] + p[b * 3 + 1] + p[c * 3 + 1]) / 3, mz = (p[a * 3 + 2] + p[b * 3 + 2] + p[c * 3 + 2]) / 3;
            if (nx * mx + ny * my + nz * mz < 0) (fa[f * 3 + 1], fa[f * 3 + 2]) = (fa[f * 3 + 2], fa[f * 3 + 1]);
        }
        return (p, fa);
    }

    /// <summary>Mean angle (degrees) between each vertex normal and the exact normal of the sphere it should lie on (the radial direction of its position).</summary>
    private static double MeanAngleToRadialDegrees(float[] positions, float[] normals)
    {
        double sum = 0; int nv = positions.Length / 3;
        for (int v = 0; v < nv; v++)
        {
            double l = Math.Sqrt(positions[v * 3] * positions[v * 3] + positions[v * 3 + 1] * positions[v * 3 + 1] + positions[v * 3 + 2] * positions[v * 3 + 2]);
            double d = (normals[v * 3] * positions[v * 3] + normals[v * 3 + 1] * positions[v * 3 + 1] + normals[v * 3 + 2] * positions[v * 3 + 2]) / l;
            sum += Math.Acos(Math.Clamp(d, -1, 1)) * 180 / Math.PI;
        }
        return sum / nv;
    }

    /// <summary>
    /// Feature-preserving vertex-normal smoothing (the gold-visor facets): on a bumpy sphere the normals come back to the exact sphere normals (2x closer or better), on a cube the sharp 90 degree
    /// edges stay sharp (flat sides keep their exact normal; the edge vertices keep theirs), and the device kernel gives the host's numbers.
    /// </summary>
    [TestMethod(Timeout = 120000)]
    public async Task Trellis2_NormalSmoothing_RemovesNoiseKeepsEdges() => await RunTest(async accelerator =>
    {
        RequireTrellisBackend(accelerator);
        var atlas = new DeviceUvAtlas(accelerator, new SparseKernels(accelerator));
        // 1. a bumpy sphere: +-2 % radius noise on 0.075-long edges (the true normals change 4.3 degrees per edge; the bumps add far more)
        var (sp, sf) = NoisySphere(0.02, 7);
        var n0 = PbrGlbWriter.VertexNormals(sp, sf);
        var n1 = PbrGlbWriter.BilateralVertexNormals(sp, sf, Trellis2Pipeline.NormalSmoothingIterations, Trellis2Pipeline.NormalSmoothingDegrees);
        double j0 = MeanAngleToRadialDegrees(sp, n0), j1 = MeanAngleToRadialDegrees(sp, n1);
        if (!(j0 > 4)) throw new Exception($"the test sphere is not bumpy enough: its normals are {j0:F1} degrees off the exact ones");
        if (!(j1 < j0 / 2 && j1 < 3)) throw new Exception($"smoothing did not calm the bumps: normals {j0:F1} -> {j1:F1} degrees off the exact sphere normals");
        // 2. a cube: flat sides keep their normal exactly, the edge vertices keep theirs
        var (cp, cf) = GridCube(8);
        var c0 = PbrGlbWriter.VertexNormals(cp, cf);
        var c1 = PbrGlbWriter.BilateralVertexNormals(cp, cf, Trellis2Pipeline.NormalSmoothingIterations, Trellis2Pipeline.NormalSmoothingDegrees);
        int flat = 0, edge = 0; double worstFlat = 0, worstEdge = 0;
        for (int v = 0; v < cp.Length / 3; v++)
        {
            int onFace = 0; for (int k = 0; k < 3; k++) if (Math.Abs(Math.Abs(cp[v * 3 + k]) - 1) < 1e-4) onFace++;
            double d = c0[v * 3] * c1[v * 3] + c0[v * 3 + 1] * c1[v * 3 + 1] + c0[v * 3 + 2] * c1[v * 3 + 2], deg = Math.Acos(Math.Clamp(d, -1, 1)) * 180 / Math.PI;
            if (onFace == 1) { flat++; worstFlat = Math.Max(worstFlat, deg); } else { edge++; worstEdge = Math.Max(worstEdge, deg); }
        }
        if (worstFlat > 2.0) throw new Exception($"a flat side's normal moved by {worstFlat:F2} degrees: the smoothing crossed a sharp edge ({flat} flat vertices)");
        if (worstEdge > 3) throw new Exception($"an edge vertex's normal moved by {worstEdge:F2} degrees ({edge} edge/corner vertices)");
        // 3. the device kernel is the same arithmetic
        foreach (var (name, pp, ff, host) in new[] { ("sphere", sp, sf, n1), ("cube", cp, cf, c1) })
        {
            var dev = await atlas.SmoothedVertexNormalsAsync(pp, ff, Trellis2Pipeline.NormalSmoothingIterations, Trellis2Pipeline.NormalSmoothingDegrees);
            double worst = 0; for (int i = 0; i < host.Length; i++) worst = Math.Max(worst, Math.Abs(dev[i] - host[i]));
            if (worst > 2e-3) throw new Exception($"device smoothed normals of the {name} differ from the host's by {worst:E2} on {accelerator.AcceleratorType}");
        }
        return $"bumpy sphere: normals {j0:F1} -> {j1:F1} degrees off the exact ones; cube: {flat} flat vertices moved at most {worstFlat:F3}, {edge} edge vertices at most {worstEdge:F2} degrees; device == host";
    });

    /// <summary>
    /// Diagnostic: the WGSL the WebGPU backend generates for the DiT's two big kernels (portable flash attention at head width 128, the bf16 tiled GEMM), printed as "WGSL_DUMP name" + source
    /// so PlaywrightMultiTest writes them to PlaywrightMultiTest/WGSLDumps/browser_console.log (PMT_CONSOLE_LOG=WGSL_DUMP). For reading the shader, not a gate.
    /// </summary>
    [TestMethod(Timeout = 300000, Category = "HeavyModel")]
    public async Task Trellis2_DumpWgsl_AttentionAndGemm() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType != AcceleratorType.WebGPU) throw new UnsupportedTestException("the WGSL is generated by the WebGPU backend");
        var dumped = new List<(string Name, string Source)>();
        var prev = global::SpawnDev.ILGPU.WebGPU.Backend.WebGPUBackend.OnShaderCompiled;
        global::SpawnDev.ILGPU.WebGPU.Backend.WebGPUBackend.OnShaderCompiled = (name, src, e) => { prev?.Invoke(name, src, e); dumped.Add((name, src)); };
        try
        {
            var dk = new DiTKernels(accelerator);
            const int T = 256, C = 1536, H = 12, D = 128;
            using var qkv = accelerator.Allocate1D(Enumerable.Range(0, T * 3 * C).Select(i => (float)Math.Cos(i * 0.11) * 0.5f).ToArray());
            using var o = accelerator.Allocate1D<float>((long)T * Math.Max(C, 3 * C));
            using var x = accelerator.Allocate1D(Enumerable.Range(0, T * C).Select(i => (float)Math.Sin(i * 0.37)).ToArray());
            using var w = accelerator.Allocate1D(Enumerable.Range(0, 3 * C * C).Select(i => BFloat16.FromRawBits((ushort)(0x3C00 + (i % 255)))).ToArray());
            dk.Attention(qkv.View, 3 * C, 0, qkv.View, 3 * C, C, 2 * C, default, o.View.SubView(0, (long)T * C), T, T, H, D);
            dk.Linear(x.View, w.View.SubView(0, 3L * C * C), null, o.View, T, C, 3 * C);
            await accelerator.SynchronizeAsync();
        }
        finally { global::SpawnDev.ILGPU.WebGPU.Backend.WebGPUBackend.OnShaderCompiled = prev; }
        // PlaywrightMultiTest turns [Wasm_DUMP_START] / [Wasm_DUMP]<base64 chunk> / [Wasm_DUMP_END] console lines into a file (wasm_dump_<HHmmss>.wasm: here the shader's UTF-8 text); one second apart so the names differ
        var summary = new List<string>();
        foreach (var (name, src) in dumped.Where(d => d.Source.Length > 5000).OrderBy(d => d.Source.Length))
        {
            var b64 = Convert.ToBase64String(System.Text.Encoding.UTF8.GetBytes(src));
            Console.WriteLine($"WGSL_DUMP {name} {src.Length} chars -> the next wasm_dump file");
            Console.WriteLine("[Wasm_DUMP_START]");
            for (int i = 0; i < b64.Length; i += 40000) Console.WriteLine("[Wasm_DUMP]" + b64.Substring(i, Math.Min(40000, b64.Length - i)));
            Console.WriteLine("[Wasm_DUMP_END]");
            summary.Add($"{name} {src.Length}");
            await Task.Delay(1100);
        }
        return "dumped: " + string.Join(", ", summary) + " | all shaders: " + string.Join(", ", dumped.Select(d => $"{d.Name} {d.Source.Length}"));
    });

    private static void SamplerTestModelImpl(Index1D i, ArrayView1D<float, Stride1D.Dense> x, ArrayView1D<float, Stride1D.Dense> v, float a, float b) => v[i] = a * x[i] + b;

    /// <summary>
    /// <see cref="FlowSampler.SampleAsync"/> (a real device wait after every step: what keeps a 25 s-per-call flow from queueing all its steps in the
    /// browser) runs the same steps as <see cref="FlowSampler.Sample"/>: identical numbers, guidance + rescale included, and the sample really moved.
    /// </summary>
    [TestMethod(Timeout = 120000)]
    public async Task Trellis2_FlowSampler_AsyncMatchesSync() => await RunTest(async accelerator =>
    {
        RequireTrellisBackend(accelerator);
        const int Count = 5000;
        var sampler = new FlowSampler(accelerator);
        var model = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, float, float>(SamplerTestModelImpl);
        FlowSampler.Model m = (x, t1000, useCond, v) => model(Count, x, v, useCond ? 0.7f : 0.2f, t1000 * (useCond ? 1e-3f : -1e-3f));
        var o = new FlowSampler.Options(Steps: 6, GuidanceStrength: 3f, GuidanceRescale: 0.5f, IntervalLo: 0.1, IntervalHi: 0.9, RescaleT: 3.0);
        using var noise = accelerator.Allocate1D<float>(Count);
        sampler.Randn(noise.View, Count, 1234UL);
        using var a = accelerator.Allocate1D<float>(Count); a.View.CopyFrom(noise.View);
        using var b = accelerator.Allocate1D<float>(Count); b.View.CopyFrom(noise.View);
        int stepsA = 0, stepsB = 0;
        sampler.Sample(m, a.View, Count, o, (s, n) => stepsA = s);
        await accelerator.SynchronizeAsync();
        await sampler.SampleAsync(m, b.View, Count, o, (s, n) => stepsB = s);
        var ha = await a.View.CopyToHostAsync(); var hb = await b.View.CopyToHostAsync(); var h0 = await noise.View.CopyToHostAsync();
        if (stepsA != 6 || stepsB != 6) throw new Exception($"steps reported {stepsA} / {stepsB}, want 6");
        double moved = 0;
        for (int i = 0; i < Count; i++)
        {
            if (!float.IsFinite(hb[i])) throw new Exception($"non-finite sample at {i}");
            if (ha[i] != hb[i]) throw new Exception($"async sampler differs at {i}: {hb[i]} vs {ha[i]} on {accelerator.AcceleratorType}");
            moved += Math.Abs(hb[i] - h0[i]);
        }
        if (moved / Count < 1e-3) throw new Exception($"the sample did not move ({moved / Count:E2} per element)");
        return $"{Count} elements, 6 steps with guidance and rescale: async == sync, mean move {moved / Count:F3}";
    });

    [TestMethod(Timeout = 120000)]
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
                // The same attention split into one-group dispatches (what a 20k-token attention does so no single dispatch runs into the GPU watchdog): the same numbers, bit for bit.
                if (!(accelerator.AcceleratorType == AcceleratorType.Cuda && D == 128 && DiTKernels.EnableTensorCores))     // the tensor-core kernel is not chunked
                {
                    long was = DiTKernels.MaxAttentionPairsPerDispatch;
                    DiTKernels.MaxAttentionPairsPerDispatch = 1;
                    try
                    {
                        using var o2 = accelerator.Allocate1D<float>((long)Nq * C);
                        dk.Attention(qb.View, qStride, 0, packed ? qb.View : kvb!.View, kvStride, kOff, vOff, default, o2.View, Nq, Nk, H, D);
                        var got2 = await o2.View.CopyToHostAsync();
                        for (int i = 0; i < got.Length; i++) if (got[i] != got2[i]) throw new Exception($"chunked flash attention D={D} {(packed ? "self" : "cross")} differs at {i}: {got2[i]} vs {got[i]} on {accelerator.AcceleratorType}");
                    }
                    finally { DiTKernels.MaxAttentionPairsPerDispatch = was; }
                }
                report.Add($"D{D} {(packed ? "self" : "cross")} {maxErr:E1}");
            }
        }
        return "flash attention: " + string.Join(", ", report);
    });

    /// <summary>
    /// Where a cold session's kernel cost goes, for the two largest generated kernels: LoadStreamKernel (IL -> IR -> backend code; on the
    /// browser this runs in the interpreted .NET runtime) versus the first launch (WebGPU: shader module + pipeline creation) minus a warm launch.
    /// Prints "[Benchmark] Trellis2 compile ..." (PMT_CONSOLE_LOG=Trellis2).
    /// </summary>
    [TestMethod(Timeout = 300000, Category = "HeavyModel")]   // a measurement, not a correctness gate
    public async Task Trellis2_KernelCompile_Benchmark() => await RunTest(async accelerator =>
    {
        RequireTrellisBackend(accelerator);
        if (accelerator.MaxNumThreadsPerGroup < TiledGemm.Threads)
            throw new UnsupportedTestException($"the tiled kernels need {TiledGemm.Threads}-thread groups (this accelerator: {accelerator.MaxNumThreadsPerGroup})");
        const int M = 256, K = 256, N = 256;
        using var x = accelerator.Allocate1D<float>(M * K);
        using var w = accelerator.Allocate1D<uint>(N * K / 2);
        using var b = accelerator.Allocate1D<uint>(N / 2);
        using var o = accelerator.Allocate1D<float>(M * N);
        using var r = accelerator.Allocate1D<float>(1);
        using var nb = accelerator.Allocate1D<int>(1);
        var report = new List<string>();
        var sizes = new Dictionary<string, int>();
        var prevHook = global::SpawnDev.ILGPU.WebGPU.Backend.WebGPUBackend.OnShaderCompiled;
        if (accelerator.AcceleratorType == AcceleratorType.WebGPU)
            global::SpawnDev.ILGPU.WebGPU.Backend.WebGPUBackend.OnShaderCompiled = (name, src, e) => { prevHook?.Invoke(name, src, e); sizes[name] = src.Length; };
        try {
        var sw = System.Diagnostics.Stopwatch.StartNew();
        var gemm = accelerator.LoadStreamKernel<ArrayView1D<float, Stride1D.Dense>, ArrayView1D<uint, Stride1D.Dense>, ArrayView1D<uint, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, int, int, int, int, int, int, int, int>(TiledGemm.For(accelerator));
        double load = sw.Elapsed.TotalMilliseconds; sw.Restart();
        var cfg = new KernelConfig(4, TiledGemm.Threads);
        gemm(cfg, x.View, w.View, b.View, o.View, r.View, nb.View, M, K, N, 0, 0, 0, 0, TiledGemm.ModeBf16); await accelerator.SynchronizeAsync();
        double first = sw.Elapsed.TotalMilliseconds; sw.Restart();
        gemm(cfg, x.View, w.View, b.View, o.View, r.View, nb.View, M, K, N, 0, 0, 0, 0, TiledGemm.ModeBf16); await accelerator.SynchronizeAsync();
        double warm = sw.Elapsed.TotalMilliseconds;
        report.Add($"gemm load {load:F0} ms, first launch {first - warm:F0} ms");

        const int T = 256, H = 2, D = 128;
        using var qkv = accelerator.Allocate1D<float>(T * 3 * H * D);
        using var ao = accelerator.Allocate1D<float>(T * H * D);
        sw.Restart();
        var attn = accelerator.LoadStreamKernel<ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int, int, int, int, int, int, int>(TiledAttention.SelfD128Smem);
        load = sw.Elapsed.TotalMilliseconds; sw.Restart();
        var acfg = new KernelConfig((T / TiledAttention.QueriesPerGroupSmem) * H, TiledAttention.Threads);
        attn(acfg, qkv.View, ao.View, T, T, H, 3 * H * D, 0, 3 * H * D, H * D, 2 * H * D); await accelerator.SynchronizeAsync();
        first = sw.Elapsed.TotalMilliseconds; sw.Restart();
        attn(acfg, qkv.View, ao.View, T, T, H, 3 * H * D, 0, 3 * H * D, H * D, 2 * H * D); await accelerator.SynchronizeAsync();
        warm = sw.Elapsed.TotalMilliseconds;
        report.Add($"attention(smem) load {load:F0} ms, first launch {first - warm:F0} ms");
        foreach (var (n, len) in sizes) report.Add($"{n} WGSL {len / 1024} KB");
        var line = $"[Benchmark] Trellis2 compile [{accelerator.AcceleratorType}] " + string.Join(" | ", report);
        Console.WriteLine(line);
        return line;
        }
        finally { global::SpawnDev.ILGPU.WebGPU.Backend.WebGPUBackend.OnShaderCompiled = prevHook; }
    });

    /// <summary>
    /// The 1024-resolution flow runs ~20k tokens: how long each op of one block takes at that size, and a whole block / several blocks queued without a
    /// sync (the browser lost the device with DXGI_ERROR_DEVICE_HUNG, a GPU watchdog kill, on the first forward). Prints "[Benchmark] Trellis2 20k ...".
    /// </summary>
    [TestMethod(Timeout = 600000, Category = "HeavyModel")]
    public async Task Trellis2_Flow20kTokens_OpTimes() => await RunTest(async accelerator =>
    {
        RequireTrellisBackend(accelerator);
        const int T = 19786, C = 1536, H = 12, D = 128, Mlp = 6144;
        var dk = new DiTKernels(accelerator);
        using var x = accelerator.Allocate1D(Enumerable.Range(0, T * C).Select(i => (float)Math.Sin(i * 0.37) * 0.5f).ToArray());
        using var w = accelerator.Allocate1D(Enumerable.Range(0, Mlp * C).Select(i => BFloat16.FromRawBits((ushort)(0x3C00 + (i % 255)))).ToArray());
        using var qkv = accelerator.Allocate1D(Enumerable.Range(0, T * 3 * C).Select(i => (float)Math.Cos(i * 0.11) * 0.5f).ToArray());
        using var att = accelerator.Allocate1D<float>((long)T * C);
        using var mlp = accelerator.Allocate1D<float>((long)T * Mlp);
        using var outq = accelerator.Allocate1D<float>((long)T * 3 * C);
        var report = new List<string>();
        async Task<double> Time(string name, Action op)
        {
            op(); await accelerator.SynchronizeAsync();                       // compile
            var sw = System.Diagnostics.Stopwatch.StartNew();
            op(); await accelerator.SynchronizeAsync();
            double ms = sw.Elapsed.TotalMilliseconds; report.Add($"{name} {ms:F0} ms"); Console.WriteLine($"[Benchmark] Trellis2 20k {name} {ms:F0} ms"); return ms;
        }
        Action linQkv = () => dk.Linear(x.View, w.View.SubView(0, 3L * C * C), null, outq.View, T, C, 3 * C);
        Action self = () => dk.Attention(qkv.View, 3 * C, 0, qkv.View, 3 * C, C, 2 * C, default, att.View, T, T, H, D);
        Action mlp1 = () => dk.Linear(x.View, w.View, null, mlp.View, T, C, Mlp);
        Action mlp2 = () => dk.Linear(mlp.View, w.View, null, att.View, T, Mlp, C);
        const int CondTokens = 4101;                                           // the 1024 px DINOv3 features: 64 x 64 patches + 5 register / class tokens
        using var kv = accelerator.Allocate1D(Enumerable.Range(0, CondTokens * 2 * C).Select(i => (float)Math.Sin(i * 0.07) * 0.5f).ToArray());
        Action cross = () => dk.Attention(x.View, C, 0, kv.View, 2 * C, 0, C, default, att.View, T, CondTokens, H, D);
        // element-wise ops over N * C = 30M elements: 118k workgroups of 256, more than the 65535 a WebGPU dispatch has per dimension
        using var modv = accelerator.Allocate1D(Enumerable.Range(0, 6 * C).Select(i => 0.5f).ToArray());
        using var resid = accelerator.Allocate1D<float>((long)T * C);
        await Time("gate residual (30M elements)", () => dk.GateResidual(resid.View, x.View, modv.View, 2 * C, T * C, C));
        await Time("silu (30M elements)", () => dk.Silu(x.View, resid.View, T * C));
        await Time("layer norm modulate", () => dk.LayerNormModulate(x.View, resid.View, modv.View, 0, C, T, C, 1e-6f));
        {   // the residual really was updated at the far end of the range (a dispatch that covers only part of it would leave that tail untouched)
            using var probe = accelerator.Allocate1D<float>((long)T * C);
            dk.GateResidual(probe.View, x.View, modv.View, 2 * C, T * C, C); await accelerator.SynchronizeAsync();
            var tail = await probe.View.SubView((long)T * C - 4, 4).CopyToHostAsync();
            var xt = await x.View.SubView((long)T * C - 4, 4).CopyToHostAsync();
            for (int i = 0; i < 4; i++) if (Math.Abs(tail[i] - 0.5f * xt[i]) > 1e-5) throw new Exception($"gate residual left the tail of 30M elements wrong: [{i}] {tail[i]} vs {0.5f * xt[i]}");
        }
        await Time("cross attention", cross);
        // the remaining GEMM shapes of the flow (each one is a different tile / K path)
        await Time("input layer 32 -> 1536", () => dk.Linear(x.View.SubView(0, (long)T * 32), w.View.SubView(0, 32L * C), null, outq.View.SubView(0, (long)T * C), T, 32, C));
        await Time("input layer 64 -> 1536", () => dk.Linear(x.View.SubView(0, (long)T * 64), w.View.SubView(0, 64L * C), null, outq.View.SubView(0, (long)T * C), T, 64, C));
        await Time("linear q / out 1536 -> 1536", () => dk.Linear(x.View, w.View.SubView(0, (long)C * C), null, outq.View.SubView(0, (long)T * C), T, C, C));
        await Time("linear kv 1024 -> 3072 (4101 tokens)", () => dk.Linear(x.View.SubView(0, 4101L * 1024), w.View.SubView(0, 1024L * 2 * C), null, outq.View.SubView(0, 4101L * 2 * C), 4101, 1024, 2 * C));
        await Time("out layer 1536 -> 32", () => dk.Linear(x.View, w.View.SubView(0, 32L * C), null, outq.View.SubView(0, (long)T * 32), T, C, 32));
        await Time("linear qkv", linQkv);
        await Time("self attention", self);
        await Time("linear mlp1", mlp1);
        await Time("linear mlp2", mlp2);
        await Time("one block queued", () => { linQkv(); self(); mlp1(); mlp2(); });
        await Time("five blocks queued", () => { for (int b = 0; b < 5; b++) { linQkv(); self(); mlp1(); mlp2(); } });
        await Time("thirty blocks queued (one flow forward)", () => { for (int b = 0; b < 30; b++) { linQkv(); self(); cross(); mlp1(); mlp2(); } });
        // a sampler step: forwards flushed one after another with no wait between them (FlowDiT.Forward ends with Fence = Flush on the browser), the way the real flow runs
        await Time("six forwards flushed back to back, one wait", () => { for (int f = 0; f < 6; f++) { for (int b = 0; b < 30; b++) { linQkv(); self(); cross(); mlp1(); mlp2(); } accelerator.Flush(); } });
        return "20k tokens: " + string.Join(", ", report);
    });

    /// <summary>
    /// Throughput of the DiT's kernels at the real 1.3B shapes (width 1536, 12 heads x 128, MLP 6144, 4096 tokens = the sparse-structure
    /// flow): how far each backend is from the silicon. Prints "[Benchmark] Trellis2 ..." lines (PMT_CONSOLE_LOG=Trellis2 shows them for browser lanes).
    /// </summary>
    [TestMethod(Timeout = 300000, Category = "HeavyModel")]   // a measurement, not a correctness gate
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
