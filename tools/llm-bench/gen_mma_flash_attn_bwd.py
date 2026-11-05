"""Generates SpawnDev.ILGPU.ML/Kernels/MmaFlashAttnBwd.Generated.cs.

EDIT THIS FILE, NOT THE OUTPUT:
    py -3.13 tools/llm-bench/gen_mma_flash_attn_bwd.py SpawnDev.ILGPU.ML/Kernels/MmaFlashAttnBwd.Generated.cs

Fused (flash) attention backward on tensor cores, head dim 128, grouped-query, causal / sliding window / kv offset (2026-11-05). The materialised
backward (BackwardKernels.AttentionTensorCore) runs five GEMMs between ten memory passes over [heads, Sq, Skv] fp32 score matrices: at 1,600 tokens
on llama3.1:8b that is ~150 GB of traffic per step and 548 ms of a 2.82 s step on an RTX 4070. Here the scores live only in registers
(FlashAttention-2's backward, deterministic: no atomics on the gradients).

Inputs are 16-bit copies in the forward's natural layouts (Q16 [Sq, H, 128], O16 = dO [Sq, H, 128] times its scale on Turing, K16 / V16
[Skv, KvH, 128], read as 32-bit words of two values). The same numerics as the materialised path: S = Q K^T and dP = dO V^T with 16-bit operands and
fp32 accumulation, P = exp(S scale - lse) in fp32, dS = P (dP - D) scale in fp32, then dQ = dS K, dK = dS^T Q, dV = P^T dO with P and dS rounded to 16
bits. sm_80+: bf16 operands (m16n8k16). Turing (Sm75): f16 operands as two m16n8k8 each, dO scaled by a device power of two (as before) and dS by a
power of two from its max: per query row for dQ (finer than before), over the whole tensor for dK (as before, per head chunk).

KernelQ  : group = 4 warps, 64 query rows of one head (a warp 16). Pass 1: the log-sum-exp per row (online over key tiles of 32), D = dO . O; writes
           both to stats. Turing pass 2: the row max of |dS| (and the global max, Atomic.Max on the float bits). Last pass: dQ += dS K.
           Q and dO fragments stay in registers; K and V tiles are staged in shared ([key][dim] words, rows of 68 words: conflict-free ldmatrix).
           S / dP B fragments by ldmatrix, dQ's K operand by ldmatrix.trans (the same [key][dim] tile read as [k][n]).
KernelKV : group = 4 warps, 32 keys of one kv head; loops over its G query heads and the query tiles that can see the keys. Warp (kg, hf):
           S^T = K Q^T and dP^T = V dO^T for keys 16 kg.. and query half hf (K and V fragments in registers), P^T and dS^T written to shared as
           16-bit words; then dV += P^T dO and dK += dS^T Q for keys 16 kg.. and dims 64 hf.. over the whole query tile (dO / Q by ldmatrix.trans).
           Query tile 64 on sm_80 (44.5 KB shared), 32 on Turing (22.5 KB: two groups per SM within Turing's 64 KB).
Fragment layouts: gen_mma_kquant_gemm.py. ldmatrix addresses from Unsafe.AsPointer (SpawnDev.ILGPU 5.4.16).
"""
import sys

D = 128
WR = D // 2            # words per row
RS = WR + 4            # shared row stride (words) for [row][dim] tiles: 68
QTQ = 64               # KernelQ query rows per group
KTQ = 32               # KernelQ key tile
KTK = 32               # KernelKV keys per group

out = []
w = out.append
MMA16 = "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
MMA8 = "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};"
LDM = "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
LDMT = "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
SM75 = False


def mma(c, a, b0, b1, ind):
    """c += A B: a = four A-fragment expressions, b0 / b1 the k16 B registers."""
    if not SM75:
        w(f"{ind}{{")
        w(f'{ind}CudaAsm.Emit("{MMA16}",')
        w(f"{ind}    out float d0, out float d1, out float d2, out float d3, {a[0]}, {a[1]}, {a[2]}, {a[3]}, {b0}, {b1}, {c}0, {c}1, {c}2, {c}3);")
        w(f"{ind}{c}0 = d0; {c}1 = d1; {c}2 = d2; {c}3 = d3;")
        w(f"{ind}}}")
        return
    for (x0, x1, bb) in ((0, 1, b0), (2, 3, b1)):
        w(f"{ind}{{")
        w(f'{ind}CudaAsm.Emit("{MMA8}",')
        w(f"{ind}    out float d0, out float d1, out float d2, out float d3, {a[x0]}, {a[x1]}, {bb}, {c}0, {c}1, {c}2, {c}3);")
        w(f"{ind}{c}0 = d0; {c}1 = d1; {c}2 = d2; {c}3 = d3;")
        w(f"{ind}}}")


def ldm(outs, arr, idx, trans, ind):
    w(f'{ind}CudaAsm.Emit("{LDMT if trans else LDM}", out uint {outs[0]}, out uint {outs[1]}, out uint {outs[2]}, out uint {outs[3]}, '
      f'(ulong)Unsafe.AsPointer(ref {arr}[{idx}]));')


def pk(lo, hi):
    return f"Pack{SFX}({lo}, {hi})"


def frags(name, view, base0, base1, ok0, ok1, ind):
    """A fragments of a 16 x 128 row block from a 16-bit [row][dim] global view (words): {name}{s}{0..3}, s = 16-dim step."""
    for s in range(D // 16):
        w(f"{ind}uint {name}{s}0 = {ok0} ? {view}[{base0} + {8 * s} + t] : 0u, {name}{s}1 = {ok1} ? {view}[{base1} + {8 * s} + t] : 0u;")
        w(f"{ind}uint {name}{s}2 = {ok0} ? {view}[{base0} + {8 * s + 4} + t] : 0u, {name}{s}3 = {ok1} ? {view}[{base1} + {8 * s + 4} + t] : 0u;")


def scores(c, a, tile, row_base_expr, nj, ind):
    """c{j} (j < nj n-tiles of 8 columns) = A (fragments a{s}{x}) times the [n][dim] shared tile rows row_base.. (ldmatrix, non-transposed)."""
    for j in range(nj):
        w(f"{ind}float {c}{j}0 = 0f, {c}{j}1 = 0f, {c}{j}2 = 0f, {c}{j}3 = 0f;")
    for s16 in range(D // 16):
        w(f"{ind}{{")
        for npair in range(nj // 2):
            ldm([f"b{2 * npair}0", f"b{2 * npair}1", f"b{2 * npair + 1}0", f"b{2 * npair + 1}1"], tile,
                f"({row_base_expr} + {npair * 16}) * {RS} + lB + {s16 * 8}", False, ind + "    ")
        for j in range(nj):
            mma(f"{c}{j}", [f"{a}{s16}{x}" for x in range(4)], f"b{j}0", f"b{j}1", ind + "    ")
        w(f"{ind}}}")


def emit_header():
    w("// <auto-generated> by tools/llm-bench/gen_mma_flash_attn_bwd.py - edit the generator, not this file. </auto-generated>")
    w("using ILGPU;")
    w("using ILGPU.Runtime;")
    w("using ILGPU.Runtime.Cuda;")
    w("using System.Runtime.CompilerServices;")
    w("")
    w("namespace SpawnDev.ILGPU.ML.Kernels;")
    w("")
    w("using FV = ArrayView1D<float, Stride1D.Dense>;")
    w("using IV = ArrayView1D<int, Stride1D.Dense>;")
    w("using UV = ArrayView<uint>;")
    w("using AttnDims = SpawnDev.ILGPU.ML.Training.BackwardKernels.AttnDims;")
    w("")
    w("/// <summary>")
    w("/// Fused (flash) attention backward on tensor cores, head dim 128: dQ / dK / dV without materialised score matrices. Generated by")
    w("/// tools/llm-bench/gen_mma_flash_attn_bwd.py (its docstring has the layout).")
    w("/// </summary>")
    w("internal static class MmaFlashAttnBwd")
    w("{")
    w(f"    public const int HeadDim = {D}, QRows = {QTQ}, KeyRows = {KTK}, Threads = 128;")
    w("")
    w("    private static uint Pack(float lo, float hi)")
    w("    {")
    w('        CudaAsm.Emit("cvt.rn.bf16x2.f32 %0, %2, %1;", out uint packed, lo, hi);   // %1 -> low half, %2 -> high half')
    w("        return packed;")
    w("    }")
    w("")
    w("    private static uint PackSm75(float lo, float hi)")
    w("    {")
    w('        CudaAsm.Emit("{ .reg .b16 plo, phi; cvt.rn.f16.f32 plo, %1; cvt.rn.f16.f32 phi, %2; mov.b32 %0, {plo, phi}; }", out uint packed, lo, hi);')
    w("        return packed;")
    w("    }")
    w("")
    w("    /// <summary>Whether query row i may attend key j (BackwardKernels.Allowed plus the bounds).</summary>")
    w("    private static bool Ok(int i, int j, AttnDims d)")
    w("        => i < d.SeqQ && j < d.SeqKV && (d.Causal != 1 || j <= i + d.KvOffset) && (d.Window <= 0 || i + d.KvOffset - j < d.Window);")
    w("")
    w("    /// <summary>The power-of-two exponent e with max * 2^e in (2^13, 2^14] (BackwardKernels.ScaleFromMaxImpl), from the float bits.</summary>")
    w("    private static int ScaleExp(float mx)")
    w("    {")
    w("        int bits = (int)Interop.FloatAsInt(mx);")
    w("        int e = 0;")
    w("        int be = (bits >> 23) & 0xFF;")
    w("        if (bits > 0 && be != 0xFF) e = 14 - ((be - 127) + ((bits & 0x7FFFFF) != 0 ? 1 : 0));")
    w("        if (e < -20) e = -20; if (e > 40) e = 40;")
    w("        return e;")
    w("    }")
    w("")
    w("    private static float Pow2(int e) => Interop.IntAsFloat((uint)((e + 127) << 23));")
    w("")


def kernel_q():
    sfx = SFX
    ind = "        "
    w(f"    /// <summary>grid = ceil(SeqQ / {QTQ}) * Heads, group = 128. Writes stats (lse [H, Sq], then D [H, Sq]) and dq [Sq, H, 128]; Turing also")
    w("    /// Atomic.Max's the float bits of max |dS| into maxBits[0] (zeroed by the caller). sc = { dO scale, 1 / dO scale } ({ 1, 1 } on sm_80).</summary>")
    w(f"    public static unsafe void KernelQ{sfx}(UV Q16, UV O16, UV K16, UV V16, FV dout, FV outp, FV stats, FV sc, IV maxBits, FV dq, AttnDims d)")
    w("    {")
    w(f"{ind}int H = d.Heads, qTile = Grid.IdxX / H, h = Grid.IdxX % H, hk = h / (H / d.KvHeads);")
    w(f"{ind}int tid = Group.IdxX, warp = tid >> 5, lane = tid & 31;")
    w(f"{ind}int g = lane >> 2, t = lane & 3;")
    w(f"{ind}var ks = SharedMemory.Allocate<uint>({KTQ * RS});")
    w(f"{ind}var vs = SharedMemory.Allocate<uint>({KTQ * RS});")
    w(f"{ind}int row0 = qTile * {QTQ} + warp * 16 + g, row1 = row0 + 8;")
    w(f"{ind}bool r0ok = row0 < d.SeqQ, r1ok = row1 < d.SeqQ;")
    w(f"{ind}long qw0 = ((long)row0 * H + h) * {WR}, qw1 = ((long)row1 * H + h) * {WR};")
    w(f"{ind}int lB = ((lane & 7) + ((lane >> 4) << 3)) * {RS} + ((lane >> 3) & 1) * 4;     // ldmatrix B pair: n 0-7 / 8-15 at k 0 / 8")
    w(f"{ind}int lT = ((lane & 7) + ((lane >> 3) & 1) * 8) * {RS} + (lane >> 4) * 4;        // ldmatrix.trans B pair: k 0-7 / 8-15 at n 0 / 8")
    w(f"{ind}float sc1 = sc[1];")
    frags("qa", "Q16", "qw0", "qw1", "r0ok", "r1ok", ind)
    frags("oa", "O16", "qw0", "qw1", "r0ok", "r1ok", ind)
    w("")
    w(f"{ind}// key range: the keys some row of this tile may see")
    w(f"{ind}int qLo = qTile * {QTQ}, qHi = Math.Min(d.SeqQ, qLo + {QTQ}) - 1;")
    w(f"{ind}int kEnd = d.SeqKV;")
    w(f"{ind}if (d.Causal == 1) kEnd = Math.Min(kEnd, qHi + d.KvOffset + 1);")
    w(f"{ind}int kBeg = 0;")
    w(f"{ind}if (d.Window > 0) kBeg = Math.Max(0, qLo + d.KvOffset - d.Window + 1) / {KTQ} * {KTQ};")
    w("")

    NW = KTQ * WR // 128   # words per thread per tile

    def decl_kv(do_v, ind2):
        w(f"{ind2}uint " + ", ".join(f"kr{i} = 0u" for i in range(NW)) + ";")
        if do_v:
            w(f"{ind2}uint " + ", ".join(f"vr{i} = 0u" for i in range(NW)) + ";")

    def load_kv(do_v, kb_expr, ind2):
        # the tile's words into registers (issued before the previous tile's MMAs: the loads overlap them)
        for i in range(NW):
            w(f"{ind2}{{ int idx = {i * 128} + tid; int gk = {kb_expr} + (idx >> 6); long src = ((long)gk * d.KvHeads + hk) * {WR} + (idx & 63);")
            w(f"{ind2}  kr{i} = gk < d.SeqKV ? K16[src] : 0u;" + (f" vr{i} = gk < d.SeqKV ? V16[src] : 0u; }}" if do_v else " }"))

    def store_kv(do_v, ind2):
        for i in range(NW):
            w(f"{ind2}{{ int idx = {i * 128} + tid; int sidx = (idx >> 6) * {RS} + (idx & 63); ks[sidx] = kr{i};" + (f" vs[sidx] = vr{i}; }}" if do_v else " }"))

    def stage_kv(do_v, ind2):
        store_kv(do_v, ind2)

    NJ = KTQ // 8
    # pass 1: lse
    w(f"{ind}float m0 = -1e30f, m1 = -1e30f, l0 = 0f, l1 = 0f;")
    i2 = ind + "    "
    w(f"{ind}{{")
    decl_kv(False, i2)
    load_kv(False, "kBeg", i2)
    w(f"{i2}for (int kb = kBeg; kb < kEnd; kb += {KTQ})")
    w(f"{i2}{{")
    i2o = i2
    i2 = i2 + "    "
    stage_kv(False, i2)
    w(f"{i2}Group.Barrier();")
    w(f"{i2}if (kb + {KTQ} < kEnd)")
    w(f"{i2}{{")
    load_kv(False, f"kb + {KTQ}", i2 + "    ")
    w(f"{i2}}}")
    scores("s", "qa", "ks", f"0", NJ, i2)
    for j in range(NJ):
        w(f"{i2}int kk{j} = kb + {8 * j} + 2 * t;")
        w(f"{i2}float x{j}0 = Ok(row0, kk{j}, d) ? s{j}0 * d.Scale : -1e30f, x{j}1 = Ok(row0, kk{j} + 1, d) ? s{j}1 * d.Scale : -1e30f;")
        w(f"{i2}float x{j}2 = Ok(row1, kk{j}, d) ? s{j}2 * d.Scale : -1e30f, x{j}3 = Ok(row1, kk{j} + 1, d) ? s{j}3 * d.Scale : -1e30f;")
    mx0 = "MathF.Max(" * (2 * NJ - 1) + "x00" + "".join([f", x{j}{c})" for j in range(NJ) for c in (0, 1)][1:])
    mx1 = "MathF.Max(" * (2 * NJ - 1) + "x02" + "".join([f", x{j}{c})" for j in range(NJ) for c in (2, 3)][1:])
    w(f"{i2}float mn0 = MathF.Max(m0, {mx0}), mn1 = MathF.Max(m1, {mx1});")
    e0 = " + ".join([f"(x{j}{c} <= -1e29f ? 0f : MathF.Exp(x{j}{c} - mn0))" for j in range(NJ) for c in (0, 1)])
    e1 = " + ".join([f"(x{j}{c} <= -1e29f ? 0f : MathF.Exp(x{j}{c} - mn1))" for j in range(NJ) for c in (2, 3)])
    w(f"{i2}l0 = l0 * MathF.Exp(m0 - mn0) + {e0};")
    w(f"{i2}l1 = l1 * MathF.Exp(m1 - mn1) + {e1};")
    w(f"{i2}m0 = mn0; m1 = mn1;")
    w(f"{i2}Group.Barrier();")
    w(f"{i2o}}}")
    w(f"{ind}}}")
    i2 = ind + "    "
    w(f"{ind}float M0 = MathF.Max(m0, Warp.ShuffleXor(m0, 1)); M0 = MathF.Max(M0, Warp.ShuffleXor(M0, 2));")
    w(f"{ind}float M1 = MathF.Max(m1, Warp.ShuffleXor(m1, 1)); M1 = MathF.Max(M1, Warp.ShuffleXor(M1, 2));")
    w(f"{ind}float L0 = l0 * MathF.Exp(m0 - M0); L0 += Warp.ShuffleXor(L0, 1); L0 += Warp.ShuffleXor(L0, 2);")
    w(f"{ind}float L1 = l1 * MathF.Exp(m1 - M1); L1 += Warp.ShuffleXor(L1, 1); L1 += Warp.ShuffleXor(L1, 2);")
    w(f"{ind}float lse0 = M0 <= -1e29f ? 0f : M0 + MathF.Log(L0), lse1 = M1 <= -1e29f ? 0f : M1 + MathF.Log(L1);")
    w(f"{ind}// D = dO . O per row (fp32): lane t sums dims 2t, 2t + 1 (mod 8), the quad adds")
    w(f"{ind}float dd0 = 0f, dd1 = 0f;")
    w(f"{ind}long f0 = ((long)row0 * H + h) * {D}, f1 = ((long)row1 * H + h) * {D};")
    w(f"{ind}if (r0ok) for (int e = 2 * t; e < {D}; e += 8) dd0 += dout[f0 + e] * outp[f0 + e] + dout[f0 + e + 1] * outp[f0 + e + 1];")
    w(f"{ind}if (r1ok) for (int e = 2 * t; e < {D}; e += 8) dd1 += dout[f1 + e] * outp[f1 + e] + dout[f1 + e + 1] * outp[f1 + e + 1];")
    w(f"{ind}dd0 += Warp.ShuffleXor(dd0, 1); dd0 += Warp.ShuffleXor(dd0, 2);")
    w(f"{ind}dd1 += Warp.ShuffleXor(dd1, 1); dd1 += Warp.ShuffleXor(dd1, 2);")
    w(f"{ind}if (t == 0)")
    w(f"{ind}{{")
    w(f"{ind}    if (r0ok) {{ stats[h * d.SeqQ + row0] = lse0; stats[(H + h) * d.SeqQ + row0] = dd0; }}")
    w(f"{ind}    if (r1ok) {{ stats[h * d.SeqQ + row1] = lse1; stats[(H + h) * d.SeqQ + row1] = dd1; }}")
    w(f"{ind}}}")
    w("")

    def pass2(mode):
        # mode "max": Turing row max |dS|; "dq": dQ += dS K
        w(f"{ind}{{")
        decl_kv(True, i2)
        load_kv(True, "kBeg", i2)
        w(f"{i2}for (int kb = kBeg; kb < kEnd; kb += {KTQ})")
        w(f"{i2}{{")
        i3 = i2 + "    "
        stage_kv(True, i3)
        w(f"{i3}Group.Barrier();")
        w(f"{i3}if (kb + {KTQ} < kEnd)")
        w(f"{i3}{{")
        load_kv(True, f"kb + {KTQ}", i3 + "    ")
        w(f"{i3}}}")
        scores("s", "qa", "ks", "0", NJ, i2)
        scores("p", "oa", "vs", "0", NJ, i2)
        for j in range(NJ):
            w(f"{i2}int kk{j} = kb + {8 * j} + 2 * t;")
            for c, (row, kk, lse, dd) in enumerate((("row0", f"kk{j}", "lse0", "dd0"), ("row0", f"kk{j} + 1", "lse0", "dd0"),
                                                     ("row1", f"kk{j}", "lse1", "dd1"), ("row1", f"kk{j} + 1", "lse1", "dd1"))):
                w(f"{i2}float g{j}{c} = Ok({row}, {kk}, d) ? MathF.Exp(s{j}{c} * d.Scale - {lse}) * (p{j}{c} * sc1 - {dd}) * d.Scale : 0f;")
        if mode == "max":
            w(f"{i2}rm0 = MathF.Max(rm0, " + "MathF.Max(" * (2 * NJ - 1) + "MathF.Abs(g00)" + "".join(
                [f", MathF.Abs(g{j}{c}))" for j in range(NJ) for c in (0, 1)][1:]) + ");")
            w(f"{i2}rm1 = MathF.Max(rm1, " + "MathF.Max(" * (2 * NJ - 1) + "MathF.Abs(g02)" + "".join(
                [f", MathF.Abs(g{j}{c}))" for j in range(NJ) for c in (2, 3)][1:]) + ");")
        else:
            m0s = " * rs0" if SM75 else ""
            m1s = " * rs1" if SM75 else ""
            for k2 in range(KTQ // 16):
                w(f"{i2}{{")
                a = [pk(f"g{2 * k2}0{m0s}", f"g{2 * k2}1{m0s}"), pk(f"g{2 * k2}2{m1s}", f"g{2 * k2}3{m1s}"),
                     pk(f"g{2 * k2 + 1}0{m0s}", f"g{2 * k2 + 1}1{m0s}"), pk(f"g{2 * k2 + 1}2{m1s}", f"g{2 * k2 + 1}3{m1s}")]
                w(f"{i2}    uint da0 = {a[0]}, da1 = {a[1]}, da2 = {a[2]}, da3 = {a[3]};")
                for npair in range(D // 16):
                    w(f"{i2}    {{")
                    ldm(["b00", "b01", "b10", "b11"], "ks", f"lT + {k2 * 16 * RS + npair * 8}", True, i2 + "        ")
                    mma(f"dq{2 * npair}", ["da0", "da1", "da2", "da3"], "b00", "b01", i2 + "        ")
                    mma(f"dq{2 * npair + 1}", ["da0", "da1", "da2", "da3"], "b10", "b11", i2 + "        ")
                    w(f"{i2}    }}")
                w(f"{i2}}}")
        w(f"{i2}Group.Barrier();")
        w(f"{i2}}}")
        w(f"{ind}}}")

    if SM75:
        w(f"{ind}// Turing: the row max of |dS| first (f16 needs a scale; a power of two per row, exact to undo)")
        w(f"{ind}float rm0 = 0f, rm1 = 0f;")
        pass2("max")
        w(f"{ind}rm0 = MathF.Max(rm0, Warp.ShuffleXor(rm0, 1)); rm0 = MathF.Max(rm0, Warp.ShuffleXor(rm0, 2));")
        w(f"{ind}rm1 = MathF.Max(rm1, Warp.ShuffleXor(rm1, 1)); rm1 = MathF.Max(rm1, Warp.ShuffleXor(rm1, 2));")
        w(f"{ind}if (t == 0) {{ Atomic.Max(ref maxBits[0], (int)Interop.FloatAsInt(rm0)); Atomic.Max(ref maxBits[0], (int)Interop.FloatAsInt(rm1)); }}")
        w(f"{ind}int re0 = ScaleExp(rm0), re1 = ScaleExp(rm1);")
        w(f"{ind}float rs0 = Pow2(re0), rs1 = Pow2(re1), irs0 = Pow2(-re0), irs1 = Pow2(-re1);")
    for j in range(D // 8):
        w(f"{ind}float dq{j}0 = 0f, dq{j}1 = 0f, dq{j}2 = 0f, dq{j}3 = 0f;")
    pass2("dq")
    i0 = " * irs0" if SM75 else ""
    i1 = " * irs1" if SM75 else ""
    for j in range(D // 8):
        w(f"{ind}if (r0ok) {{ dq[f0 + {8 * j} + 2 * t] = dq{j}0{i0}; dq[f0 + {8 * j} + 2 * t + 1] = dq{j}1{i0}; }}")
        w(f"{ind}if (r1ok) {{ dq[f1 + {8 * j} + 2 * t] = dq{j}2{i1}; dq[f1 + {8 * j} + 2 * t + 1] = dq{j}3{i1}; }}")
    w("    }")
    w("")


def kernel_kv():
    sfx = SFX
    QT = 32 if SM75 else 64
    PW = QT // 2 + 4
    NJ = QT // 2 // 8        # n-tiles per warp in S^T (its half of the query tile)
    ind = "        "
    i2 = ind + "    "
    i3 = i2 + "    "
    w(f"    /// <summary>grid = ceil(SeqKV / {KTK}) * KvHeads, group = 128. dk / dv [Skv, KvH, 128] (the sum over the kv head's G query heads). stats from")
    w("    /// KernelQ; sc = { dO scale, 1 / dO scale }; scDs = { dS scale, 1 / dS scale } (Turing; ignored on sm_80).</summary>")
    w(f"    public static unsafe void KernelKV{sfx}(UV Q16, UV O16, UV K16, UV V16, FV stats, FV sc, FV scDs, FV dk, FV dv, AttnDims d)")
    w("    {")
    w(f"{ind}int H = d.Heads, KvH = d.KvHeads, G = H / KvH;")
    w(f"{ind}int kTile = Grid.IdxX / KvH, hk = Grid.IdxX % KvH;")
    w(f"{ind}int tid = Group.IdxX, warp = tid >> 5, lane = tid & 31;")
    w(f"{ind}int g = lane >> 2, t = lane & 3, kg = warp >> 1, hf = warp & 1;")
    w(f"{ind}var qs = SharedMemory.Allocate<uint>({QT * RS});")
    w(f"{ind}var os = SharedMemory.Allocate<uint>({QT * RS});")
    w(f"{ind}var pt = SharedMemory.Allocate<uint>({KTK * PW});")
    w(f"{ind}var dt = SharedMemory.Allocate<uint>({KTK * PW});")
    w(f"{ind}var lseS = SharedMemory.Allocate<float>({QT});")
    w(f"{ind}var ddS = SharedMemory.Allocate<float>({QT});")
    w(f"{ind}int key0 = kTile * {KTK} + kg * 16 + g, key1 = key0 + 8;")
    w(f"{ind}bool k0ok = key0 < d.SeqKV, k1ok = key1 < d.SeqKV;")
    w(f"{ind}long kw0 = ((long)key0 * KvH + hk) * {WR}, kw1 = ((long)key1 * KvH + hk) * {WR};")
    w(f"{ind}int lB = ((lane & 7) + ((lane >> 4) << 3)) * {RS} + ((lane >> 3) & 1) * 4;")
    w(f"{ind}int lT = ((lane & 7) + ((lane >> 3) & 1) * 8) * {RS} + (lane >> 4) * 4;")
    w(f"{ind}int lA = (kg * 16 + (lane & 15)) * {PW} + (lane >> 4) * 4;                    // ldmatrix A: rows 0-15 at k 0 / 8")
    w(f"{ind}float sc1 = sc[1];")
    if SM75:
        w(f"{ind}float sDs = scDs[0], iDs = scDs[1];")
    frags("ka", "K16", "kw0", "kw1", "k0ok", "k1ok", ind)
    frags("va", "V16", "kw0", "kw1", "k0ok", "k1ok", ind)
    for j in range(8):
        w(f"{ind}float dk{j}0 = 0f, dk{j}1 = 0f, dk{j}2 = 0f, dk{j}3 = 0f, dv{j}0 = 0f, dv{j}1 = 0f, dv{j}2 = 0f, dv{j}3 = 0f;")
    w(f"{ind}int kLo = kTile * {KTK}, kHi = Math.Min(d.SeqKV, kLo + {KTK}) - 1;")
    w(f"{ind}int qBeg = 0;")
    w(f"{ind}if (d.Causal == 1) qBeg = Math.Max(0, kLo - d.KvOffset) / {QT} * {QT};")
    w(f"{ind}int qEnd = d.SeqQ;")
    w(f"{ind}if (d.Window > 0) qEnd = Math.Min(qEnd, kHi - d.KvOffset + d.Window);")
    w(f"{ind}for (int hh = 0; hh < G; hh++)")
    w(f"{ind}{{")
    w(f"{i2}int h = hk * G + hh;")
    w(f"{i2}for (int qb = qBeg; qb < qEnd; qb += {QT})")
    w(f"{i2}{{")
    for i in range(QT * WR // 128):
        w(f"{i3}{{ int idx = {i * 128} + tid; int r = idx >> 6, c = idx & 63; int gq = qb + r; long src = ((long)gq * H + h) * {WR} + c;")
        w(f"{i3}  qs[r * {RS} + c] = gq < d.SeqQ ? Q16[src] : 0u; os[r * {RS} + c] = gq < d.SeqQ ? O16[src] : 0u; }}")
    w(f"{i3}if (tid < {QT}) {{ int gq = qb + tid; lseS[tid] = gq < d.SeqQ ? stats[h * d.SeqQ + gq] : 0f; ddS[tid] = gq < d.SeqQ ? stats[(H + h) * d.SeqQ + gq] : 0f; }}")
    w(f"{i3}Group.Barrier();")
    scores("s", "ka", "qs", f"hf * {QT // 2}", NJ, i3)
    scores("p", "va", "os", f"hf * {QT // 2}", NJ, i3)
    sds = " * sDs" if SM75 else ""
    for j in range(NJ):
        w(f"{i3}int li{j} = hf * {QT // 2} + {8 * j} + 2 * t;")
        for c, (key, liq) in enumerate((("key0", f"li{j}"), ("key0", f"li{j} + 1"), ("key1", f"li{j}"), ("key1", f"li{j} + 1"))):
            w(f"{i3}float e{j}{c} = Ok(qb + {liq}, {key}, d) ? MathF.Exp(s{j}{c} * d.Scale - lseS[{liq}]) : 0f;")
            w(f"{i3}float f{j}{c} = e{j}{c} * (p{j}{c} * sc1 - ddS[{liq}]) * d.Scale{sds};")
        w(f"{i3}pt[(kg * 16 + g) * {PW} + hf * {QT // 4} + {4 * j} + t] = {pk(f'e{j}0', f'e{j}1')};")
        w(f"{i3}pt[(kg * 16 + g + 8) * {PW} + hf * {QT // 4} + {4 * j} + t] = {pk(f'e{j}2', f'e{j}3')};")
        w(f"{i3}dt[(kg * 16 + g) * {PW} + hf * {QT // 4} + {4 * j} + t] = {pk(f'f{j}0', f'f{j}1')};")
        w(f"{i3}dt[(kg * 16 + g + 8) * {PW} + hf * {QT // 4} + {4 * j} + t] = {pk(f'f{j}2', f'f{j}3')};")
    w(f"{i3}Group.Barrier();")
    w(f"{i3}// dV += P^T dO, dK += dS^T Q: keys 16 kg.., dims 64 hf.., k = the query tile")
    for k2 in range(QT // 16):
        w(f"{i3}{{")
        i4 = i3 + "    "
        ldm(["pa0", "pa1", "pa2", "pa3"], "pt", f"lA + {k2 * 8}", False, i4)
        ldm(["da0", "da1", "da2", "da3"], "dt", f"lA + {k2 * 8}", False, i4)
        for npair in range(4):
            w(f"{i4}{{")
            ldm(["ob00", "ob01", "ob10", "ob11"], "os", f"lT + {k2 * 16 * RS} + hf * 32 + {npair * 8}", True, i4 + "    ")
            mma(f"dv{2 * npair}", ["pa0", "pa1", "pa2", "pa3"], "ob00", "ob01", i4 + "    ")
            mma(f"dv{2 * npair + 1}", ["pa0", "pa1", "pa2", "pa3"], "ob10", "ob11", i4 + "    ")
            ldm(["qb00", "qb01", "qb10", "qb11"], "qs", f"lT + {k2 * 16 * RS} + hf * 32 + {npair * 8}", True, i4 + "    ")
            mma(f"dk{2 * npair}", ["da0", "da1", "da2", "da3"], "qb00", "qb01", i4 + "    ")
            mma(f"dk{2 * npair + 1}", ["da0", "da1", "da2", "da3"], "qb10", "qb11", i4 + "    ")
            w(f"{i4}}}")
        w(f"{i3}}}")
    w(f"{i3}Group.Barrier();")
    w(f"{i2}}}")
    w(f"{ind}}}")
    w(f"{ind}long o0 = ((long)key0 * KvH + hk) * {D} + hf * 64 + 2 * t, o1 = ((long)key1 * KvH + hk) * {D} + hf * 64 + 2 * t;")
    ik = " * iDs" if SM75 else ""
    for j in range(8):
        w(f"{ind}if (k0ok) {{ dk[o0 + {8 * j}] = dk{j}0{ik}; dk[o0 + {8 * j + 1}] = dk{j}1{ik}; dv[o0 + {8 * j}] = dv{j}0 * sc1; dv[o0 + {8 * j + 1}] = dv{j}1 * sc1; }}")
        w(f"{ind}if (k1ok) {{ dk[o1 + {8 * j}] = dk{j}2{ik}; dk[o1 + {8 * j + 1}] = dk{j}3{ik}; dv[o1 + {8 * j}] = dv{j}2 * sc1; dv[o1 + {8 * j + 1}] = dv{j}3 * sc1; }}")
    w("    }")
    w("")


emit_header()
for SM75 in (False, True):
    SFX = "Sm75" if SM75 else ""
    kernel_q()
    kernel_kv()
w("}")

open(sys.argv[1], "w", encoding="utf-8", newline="\n").write("\n".join(out) + "\n")
print("wrote", sys.argv[1], len(out), "lines")
