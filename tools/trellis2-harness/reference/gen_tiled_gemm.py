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
import os
import sys

BM = BN = 128
BK = 8
KU = int(os.environ.get("GEMM_KU", "1"))   # inner k steps emitted per loop iteration (1 = the plain 8-trip loop, 8 = fully unrolled)
TH = 256
LD = 132          # shared row stride (floats): 132 % 32 = 4 -> the transposed stores are conflict free

out = []
w = out.append

def kernel(name, BK):
    # ONE kernel for all three variants (bf16 / f16 weights, dense / gathered rows), chosen by the uniform `mode` argument: on WebGPU each
    # kernel costs ~7 s of IL -> WGSL compile in the browser's interpreted .NET runtime plus a shader compile, MEASURED per variant.
    NA = BM * BK // TH              # A floats per thread per k step
    NB = BN * (BK // 2) // TH       # B words per thread per k step
    sa = BK.bit_length() - 1
    sb = (BK // 2).bit_length() - 1
    tail = BK > 8                   # K is a multiple of 8, not necessarily of BK: the last step is masked
    w(f"    /// <summary>k step {BK}. mode bit 0: f16 weights (else bf16); bit 1: gathered rows (sparse conv, A[m,k] = x[nbr[m*27 + k/cin]*cin + k%cin]). grid = ceil(M/128) * ceil(N/128), group = {TH}, shared {2 * BK * LD * 4} bytes.</summary>")
    w(f"    public static void {name}(FV A, UV W, UV Bias, FV C, FV Res, IV nbr, int M, int K, int N, int cin, int hasBias, int act, int hasRes, int mode)")
    w("    {")
    w("        bool f16 = (mode & 1) != 0, gather = (mode & 2) != 0;")
    w(f"        int tilesN = (N + {BN - 1}) / {BN};")
    w("        int tm = Grid.IdxX / tilesN, tn = Grid.IdxX - tm * tilesN;")
    w("        int tid = Group.IdxX, tx = tid & 15, ty = tid >> 4;")
    w(f"        var As = SharedMemory.Allocate<float>({BK * LD});")
    w(f"        var Bs = SharedMemory.Allocate<float>({BK * LD});")
    w(f"        int mBase = tm * {BM}, nBase = tn * {BN};")
    # A loads: 128 rows x BK k; idx = tid + 256 r -> row = idx >> log2(BK), kk = idx & (BK-1)
    for r in range(NA):
        w(f"        int aRow{r} = ({r * TH} + tid) >> {sa}, aK{r} = ({r * TH} + tid) & {BK - 1}, aM{r} = mBase + aRow{r};")
    # B loads: 128 n x BK/2 words; idx = tid + 256 r -> nrow = idx >> log2(BK/2), kw = idx & (BK/2-1)
    for r in range(NB):
        w(f"        int bRow{r} = ({r * TH} + tid) >> {sb}, bW{r} = ({r * TH} + tid) & {BK // 2 - 1}, bN{r} = nBase + bRow{r};")
    # Row bases hoisted out of the k loop, in int: the host guarantees M*K, M*N, N*K < 2^31 (TiledGemm.Fits). (long) index math is
    # EMULATED on WebGPU (vec2<u32>): a 64-bit multiply + adds on every global load of every k step.
    for r in range(NA):
        w(f"        int aOff{r} = aM{r} * K + aK{r};")
    for r in range(NB):
        w(f"        int bOff{r} = ((bN{r} * K) >> 1) + bW{r};")
    for i in range(8):
        w("        float " + ", ".join(f"c{i}{j} = 0f" for j in range(8)) + ";")
    # The k loop is emitted once per variant actually used (bf16 dense, f16 dense, f16 gathered), each free of mode branches; the epilogue
    # is shared. MEASURED on WebGPU: mode branches inside one shared loop cost ~7% of GEMM throughput.
    def kloop(f16v, gatherv, indent):
        sp = " " * indent
        lo, hi = ("F16Lo", "F16Hi") if f16v else ("Bf16Lo", "Bf16Hi")
        def loadA(r, k0):
            kOk = f" && {k0} + aK{r} < K" if tail else ""
            if gatherv:
                return f"aM{r} < M{kOk} ? GatherA(A, nbr, aM{r}, {k0} + aK{r}, cin) : 0f"
            return f"aM{r} < M{kOk} ? A[aOff{r} + {k0}] : 0f"
        def loadB(r, k0):
            kOk = f" && {k0} + 2 * bW{r} < K" if tail else ""
            return f"bN{r} < N{kOk} ? W[bOff{r} + ({k0} >> 1)] : 0u"
        for r in range(NA):
            w(f"{sp}float pa{r} = {loadA(r, '0')};")
        for r in range(NB):
            w(f"{sp}uint pb{r} = {loadB(r, '0')};")
        w(f"{sp}for (int k0 = 0; k0 < K; k0 += {BK})")
        w(f"{sp}{{")
        for r in range(NA):
            w(f"{sp}    As[aK{r} * {LD} + aRow{r}] = pa{r};")
        for r in range(NB):
            w(f"{sp}    Bs[(2 * bW{r}) * {LD} + bRow{r}] = {lo}(pb{r}); Bs[(2 * bW{r} + 1) * {LD} + bRow{r}] = {hi}(pb{r});")
        w(f"{sp}    Group.Barrier();")
        w(f"{sp}    int kn = k0 + {BK};")
        w(f"{sp}    if (kn < K)")
        w(f"{sp}    {{")
        for r in range(NA):
            w(f"{sp}        pa{r} = {loadA(r, 'kn')};")
        for r in range(NB):
            w(f"{sp}        pb{r} = {loadB(r, 'kn')};")
        w(f"{sp}    }}")
        # KU = k steps per iteration of the inner loop, emitted with constant offsets (GEMM_KU=8 removes the loop). MEASURED 2026-10-21 on WebGPU
        # (RTX 4070, Chrome 156, SpawnDev.ILGPU 5.4.1-local.2, 4096x1536 -> 6144): NO GAIN from source-level unrolling, so the default stays 1.
        #   KU   qkv / mlp-up / mlp-down TFLOPS   GEMM WGSL   cold first launch
        #    1   7.9 / 8.0 / 7.4                  331 KB      1.25 s
        #    2   8.0 / 7.8 / 7.5                  370 KB      2.2 s
        #    4   8.2 / 8.2 / 7.8 (inside the +-0.4 noise)  447 KB   2.1 s
        #    8   7.3 / 7.2 / 7.1 (about 9 % SLOWER)        559 KB   2.5 s
        # (Data measured +12 % with ILGPU's own LoopUnrolling raised to unroll this loop in a copy of this kernel; the two results are not
        # reconciled. This kernel holds three k-loops (gather / f16 / bf16) in one shader, so unrolling triples the unrolled code.)
        ku = min(KU, BK)
        body_indent = sp + "    "
        if ku < BK:
            w(f"{sp}    for (int kk = 0; kk < {BK}; kk{'++' if ku == 1 else f' += {ku}'})")
            w(f"{sp}    {{")
            w(f"{sp}        int ab = kk * {LD} + ty, bb = kk * {LD} + tx;")
            body_indent = sp + "        "
        else:
            w(f"{sp}    int ab = ty, bb = tx;")
        for s in range(ku):
            if ku > 1:
                w(f"{body_indent}{{")
            ind = body_indent + ("    " if ku > 1 else "")
            for i in range(8):
                w(f"{ind}float a{i} = As[ab + {s * LD + 16 * i}];")
            for j in range(8):
                w(f"{ind}float b{j} = Bs[bb + {s * LD + 16 * j}];")
            for i in range(8):
                w(f"{ind}" + " ".join(f"c{i}{j} += a{i} * b{j};" for j in range(8)))
            if ku > 1:
                w(f"{body_indent}}}")
        if ku < BK:
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
w(f"    /// <summary>BK: the k step and the multiple K must be.</summary>")
w(f"    public const int BM = {BM}, BN = {BN}, BK = {BK}, Threads = {TH};")
w("")
w("    /// <summary>The GEMM kernel every caller loads (one place to choose a variant per accelerator). MEASURED: a k step of 16 (generator")
w("    /// kernel(name, 16), 17 KB shared, masked k tail) was SLOWER than 8 on WebGPU (7.0-7.4 vs 7.6-8.2 TFLOPS) and on OpenCL.</summary>")
w("    public static Action<FV, UV, UV, FV, FV, IV, int, int, int, int, int, int, int, int> For(Accelerator accelerator) => Gemm;")
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
kernel("Gemm", BK)
w("}")

open(sys.argv[1], "w", encoding="utf-8", newline="\n").write("\n".join(out) + "\n")
print("wrote", sys.argv[1], len(out), "lines")
