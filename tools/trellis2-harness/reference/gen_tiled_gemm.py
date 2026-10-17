"""Generates SpawnDev.ILGPU.ML/Trellis2/TiledGemm.Generated.cs.

EDIT THIS FILE, NOT THE OUTPUT.

Portable register-tiled GEMM (no tensor cores, no inline assembly): runs on every ILGPU backend (WebGPU, OpenCL, CUDA, CPU...).
  C[m, n] = act( bias[n] + sum_k A[m, k] * W[n, k] )        A fp32 [M, K] (or gathered rows), W bf16 or fp16 [N, K] (nn.Linear layout)
Group = 256 threads (16 x 16), tile 128 x 128, k step 8; each thread owns an 8 x 8 block of outputs at rows ty + 16 i, cols tx + 16 j
(strided, so the shared-memory reads are broadcasts / consecutive banks). Shared memory: As[8][132], Bs[8][132] floats = 8.4 KB (inside
WebGPU's guaranteed 16 KB). The next k step's global words are fetched into registers while the current one computes.
Weights are read as 32-bit words and decoded in registers (bf16: shift; fp16: bit rebuild), never through sub-word element loads
(on WebGPU those are atomic word loads).
Gather variant (implicit-GEMM sparse conv): A[m, k] = x[nbr[m * 27 + k / cin] * cin + k % cin] (0 where the neighbour is absent).
Requires K % 8 == 0 (and K even for the word reads); M and N are arbitrary.
Epilogue: v = act(acc + bias); hasRes 1 adds Res[m, n], hasRes 2 adds the value already in C (an in-place residual without binding the
same buffer twice, which WebGPU forbids).
"""
import sys

BM = BN = 128
BK = 8
TH = 256
LD = 132          # shared row stride (floats): 132 % 32 = 4 -> the transposed stores are conflict free

out = []
w = out.append

def kernel(name, wtype, gather):
    # ONE kernel for all three variants (bf16 / f16 weights, dense / gathered rows), chosen by the uniform `mode` argument: on WebGPU each
    # kernel costs ~7 s of IL -> WGSL compile in the browser's interpreted .NET runtime plus a shader compile, MEASURED per variant.
    w(f"    /// <summary>mode bit 0: f16 weights (else bf16); bit 1: gathered rows (sparse conv, A[m,k] = x[nbr[m*27 + k/cin]*cin + k%cin]). grid = ceil(M/128) * ceil(N/128), group = {TH}.</summary>")
    w(f"    public static void {name}(FV A, UV W, UV Bias, FV C, FV Res, IV nbr, int M, int K, int N, int cin, int hasBias, int act, int hasRes, int mode)")
    w("    {")
    w("        bool f16 = (mode & 1) != 0, gather = (mode & 2) != 0;")
    w(f"        int tilesN = (N + {BN - 1}) / {BN};")
    w("        int tm = Grid.IdxX / tilesN, tn = Grid.IdxX - tm * tilesN;")
    w("        int tid = Group.IdxX, tx = tid & 15, ty = tid >> 4;")
    w(f"        var As = SharedMemory.Allocate<float>({BK * LD});")
    w(f"        var Bs = SharedMemory.Allocate<float>({BK * LD});")
    w(f"        int mBase = tm * {BM}, nBase = tn * {BN};")
    # A loads: 128 rows x 8 k = 1024 floats, 4 per thread; idx = tid + 256 r -> row = idx >> 3, kk = idx & 7
    for r in range(4):
        w(f"        int aRow{r} = ({r * TH} + tid) >> 3, aK{r} = ({r * TH} + tid) & 7, aM{r} = mBase + aRow{r};")
    # B loads: 128 n x 4 words, 2 per thread; idx = tid + 256 r -> nrow = idx >> 2, kw = idx & 3
    for r in range(2):
        w(f"        int bRow{r} = ({r * TH} + tid) >> 2, bW{r} = ({r * TH} + tid) & 3, bN{r} = nBase + bRow{r};")
    # Row bases hoisted out of the k loop, in int: the host guarantees M*K, M*N, N*K < 2^31 (TiledGemm.Fits). (long) index math is
    # EMULATED on WebGPU (vec2<u32>): a 64-bit multiply + adds on every global load of every k step.
    for r in range(4):
        w(f"        int aOff{r} = aM{r} * K + aK{r};")
    for r in range(2):
        w(f"        int bOff{r} = ((bN{r} * K) >> 1) + bW{r};")
    for i in range(8):
        w("        float " + ", ".join(f"c{i}{j} = 0f" for j in range(8)) + ";")
    # The k loop is emitted once per variant actually used (bf16 dense, f16 dense, f16 gathered), each free of mode branches; the epilogue
    # is shared. MEASURED on WebGPU: mode branches inside one shared loop cost ~7% of GEMM throughput.
    def kloop(f16v, gatherv, indent):
        sp = " " * indent
        lo, hi = ("F16Lo", "F16Hi") if f16v else ("Bf16Lo", "Bf16Hi")
        def loadA(r, k0):
            if gatherv:
                return f"aM{r} < M ? GatherA(A, nbr, aM{r}, {k0} + aK{r}, cin) : 0f"
            return f"aM{r} < M ? A[aOff{r} + {k0}] : 0f"
        def loadB(r, k0):
            return f"bN{r} < N ? W[bOff{r} + ({k0} >> 1)] : 0u"
        for r in range(4):
            w(f"{sp}float pa{r} = {loadA(r, '0')};")
        for r in range(2):
            w(f"{sp}uint pb{r} = {loadB(r, '0')};")
        w(f"{sp}for (int k0 = 0; k0 < K; k0 += {BK})")
        w(f"{sp}{{")
        for r in range(4):
            w(f"{sp}    As[aK{r} * {LD} + aRow{r}] = pa{r};")
        for r in range(2):
            w(f"{sp}    Bs[(2 * bW{r}) * {LD} + bRow{r}] = {lo}(pb{r}); Bs[(2 * bW{r} + 1) * {LD} + bRow{r}] = {hi}(pb{r});")
        w(f"{sp}    Group.Barrier();")
        w(f"{sp}    int kn = k0 + {BK};")
        w(f"{sp}    if (kn < K)")
        w(f"{sp}    {{")
        for r in range(4):
            w(f"{sp}        pa{r} = {loadA(r, 'kn')};")
        for r in range(2):
            w(f"{sp}        pb{r} = {loadB(r, 'kn')};")
        w(f"{sp}    }}")
        w(f"{sp}    for (int kk = 0; kk < {BK}; kk++)")
        w(f"{sp}    {{")
        w(f"{sp}        int ab = kk * {LD} + ty, bb = kk * {LD} + tx;")
        for i in range(8):
            w(f"{sp}        float a{i} = As[ab + {16 * i}];")
        for j in range(8):
            w(f"{sp}        float b{j} = Bs[bb + {16 * j}];")
        for i in range(8):
            w(f"{sp}        " + " ".join(f"c{i}{j} += a{i} * b{j};" for j in range(8)))
        w(f"{sp}    }}")
        w(f"{sp}    Group.Barrier();")
        w(f"{sp}}}")
    w("        if (gather)")
    w("        {")
    kloop(True, True, 12)
    w("        }")
    w("        else if (f16)")
    w("        {")
    kloop(True, False, 12)
    w("        }")
    w("        else")
    w("        {")
    kloop(False, False, 12)
    w("        }")
    for j in range(8):
        w(f"        int n{j} = nBase + tx + {16 * j};")
        w(f"        float bias{j} = hasBias != 0 && n{j} < N ? ((n{j} & 1) == 0 ? Lo(Bias[n{j} >> 1], f16) : Hi(Bias[n{j} >> 1], f16)) : 0f;")
    for i in range(8):
        w(f"        int m{i} = mBase + ty + {16 * i};")
        w(f"        if (m{i} < M)")
        w("        {")
        w(f"            int rb = m{i} * N;")
        for j in range(8):
            w(f"            if (n{j} < N) {{ float v = DiTKernels.Activate(c{i}{j} + bias{j}, act); if (hasRes == 1) v += Res[rb + n{j}]; else if (hasRes == 2) v += C[rb + n{j}]; C[rb + n{j}] = v; }}")
        w("        }")
    w("    }")
    w("")

w("// <auto-generated> by tools/trellis2-harness/reference/gen_tiled_gemm.py - edit the generator, not this file. </auto-generated>")
w("using ILGPU;")
w("using ILGPU.Runtime;")
w("")
w("namespace SpawnDev.ILGPU.ML.Trellis2;")
w("")
w("using FV = ArrayView1D<float, Stride1D.Dense>;")
w("using UV = ArrayView1D<uint, Stride1D.Dense>;")
w("using IV = ArrayView1D<int, Stride1D.Dense>;")
w("")
w("/// <summary>Portable register-tiled GEMM for every backend (see the generator for the layout).</summary>")
w("internal static class TiledGemm")
w("{")
w(f"    public const int BM = {BM}, BN = {BN}, BK = {BK}, Threads = {TH};")
w("")
w("    /// <summary>The kernels index in int (64-bit index math is emulated on WebGPU): every flat index of A, C, W and the neighbour table must fit.</summary>")
w("    public static bool Fits(long M, long K, long N, long rowsA = -1) => M * K < int.MaxValue && M * N < int.MaxValue && N * K < int.MaxValue && M * 27 < int.MaxValue && (rowsA < 0 || rowsA * K < int.MaxValue);")
w("")
w("    private static float Bf16Lo(uint u) => Interop.IntAsFloat(u << 16);")
w("    private static float Bf16Hi(uint u) => Interop.IntAsFloat(u & 0xFFFF0000u);")
w("    private static float HalfBits(uint h)")
w("    {")
w("        uint sign = (h & 0x8000u) << 16, e = (h >> 10) & 31u, m = h & 1023u;")
w("        if (e == 0u) { float v = m * 5.9604644775390625e-8f; return sign != 0u ? -v : v; }   // zero / subnormal: m * 2^-24")
w("        if (e == 31u) return Interop.IntAsFloat(sign | 0x7F800000u | (m << 13));               // inf / nan")
w("        return Interop.IntAsFloat(sign | ((e + 112u) << 23) | (m << 13));")
w("    }")
w("    private static float F16Lo(uint u) => HalfBits(u & 0xFFFFu);")
w("    private static float F16Hi(uint u) => HalfBits(u >> 16);")
w("    private static float Lo(uint u, bool f16) => f16 ? F16Lo(u) : Bf16Lo(u);")
w("    private static float Hi(uint u, bool f16) => f16 ? F16Hi(u) : Bf16Hi(u);")
w("    /// <summary>The mode argument of <see cref=\"Gemm\"/>.</summary>")
w("    public const int ModeBf16 = 0, ModeF16 = 1, ModeGather = 2;")
w("    private static float GatherA(FV A, IV nbr, int m, int k, int cin)")
w("    {")
w("        int tap = k / cin, ci = k - tap * cin;")
w("        int src = nbr[m * 27 + tap];")
w("        return src >= 0 ? A[src * cin + ci] : 0f;")
w("    }")
w("")
kernel("Gemm", "any", None)
w("}")

open(sys.argv[1], "w", encoding="utf-8", newline="\n").write("\n".join(out) + "\n")
print("wrote", sys.argv[1], len(out), "lines")
