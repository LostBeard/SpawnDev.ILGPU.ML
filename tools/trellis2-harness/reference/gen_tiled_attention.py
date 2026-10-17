"""Generates SpawnDev.ILGPU.ML/Trellis2/TiledAttention.Generated.cs.

EDIT THIS FILE, NOT THE OUTPUT.

Portable flash attention (no tensor cores, no inline assembly; fp32 throughout) for every ILGPU backend. Non-causal, no mask.
  O[i, h*D + d] = sum_j softmax_j(q_i . k_j / sqrt(D)) v_j[d]
Group = 256 threads owns QPG queries of one head; P = 4 consecutive lanes share a query, lane p owning the dims d = p + 4 t (interleaved,
so the four lanes of a quad read four consecutive banks). Keys go through shared memory in tiles of BKT; for each tile the quad computes
BKT scores (partial dots + two Warp.ShuffleXor steps), rescales its running output once per tile (online softmax), then accumulates
P V. Shared memory: Ks + Vs = 2 * BKT * D floats (16 KB at D = 128, BKT = 16: WebGPU's guaranteed workgroup storage).
Rows >= Nq load zeros and are not stored; keys >= Nk score -inf and stage zeros.
Two entry points: Cross (Q and KV in different buffers) and Self (one buffer holds both: WebGPU forbids binding a buffer twice).
*Smem variants: for backends without warp shuffles (NVIDIA's OpenCL, browsers without the subgroups feature) the quad's partial dots meet in
shared memory instead (red[key][lane][query]: 8 KB; key tiles of 8 so Ks + Vs + red stay within 16 KB).
"""
import sys

P = 4
TH = 256
QPG = TH // P      # 64 queries per group

out = []
w = out.append

import os
# Shuffle variant shape: SP lanes share a query, each thread owns SQT queries (every shared K/V load is reused SQT times; a query costs
# log2(SP) shuffles per key). ATT_CFG="P,QT,BKT128,BKT64" overrides for experiments.
_cfg = [int(v) for v in os.environ.get("ATT_CFG", "8,4,8,16").split(",")]
SP, SQT, SBKT128, SBKT64 = _cfg

def kernel_shuffle(D, self_attn):
    P_, QT = SP, SQT
    BKT = SBKT128 if D == 128 else SBKT64
    DT = D // P_
    LANES = TH // P_
    QG = LANES * QT                  # queries per group
    lg = P_.bit_length() - 1
    name = f"{'Self' if self_attn else 'Cross'}D{D}"
    if self_attn:
        w(f"    /// <summary>D = {D}, Q and K/V in one buffer. grid = ceil(Nq / {QG}) * H, group = {TH}.</summary>")
        w(f"    public static void {name}(FV S, FV O, int Nq, int Nk, int H, int qStride, int qOff, int kvStride, int kOff, int vOff)")
        Q = KV = "S"
    else:
        w(f"    /// <summary>D = {D}, Q and K/V in separate buffers. grid = ceil(Nq / {QG}) * H, group = {TH}.</summary>")
        w(f"    public static void {name}(FV Q, FV KV, FV O, int Nq, int Nk, int H, int qStride, int qOff, int kvStride, int kOff, int vOff)")
        Q, KV = "Q", "KV"
    w("    {")
    w("        int qTile = Grid.IdxX / H, h = Grid.IdxX - qTile * H;")
    w(f"        int tid = Group.IdxX, p = tid & {P_ - 1}, ql = tid >> {lg};")
    w(f"        var ks = SharedMemory.Allocate<float>({BKT * D});")
    w(f"        var vs = SharedMemory.Allocate<float>({BKT * D});")
    w(f"        const float scale = {1.0 / (D ** 0.5)!r}f;")
    for r in range(QT):
        w(f"        int row{r} = qTile * {QG} + ql + {r * LANES};")
        w(f"        bool ok{r} = row{r} < Nq;")
        w(f"        int qb{r} = row{r} * qStride + qOff + h * {D} + p;")
        for t in range(DT):
            w(f"        float q{r}_{t} = ok{r} ? {Q}[qb{r} + {P_ * t}] * scale : 0f;")
        w("        " + " ".join(f"float o{r}_{t} = 0f;" for t in range(DT)))
        w(f"        float m{r} = -1e30f, l{r} = 0f;")
    loads = BKT * D // TH
    w(f"        for (int kb = 0; kb < Nk; kb += {BKT})")
    w("        {")
    for r in range(loads):
        w(f"            {{ int idx = {r * TH} + tid; int key = idx / {D}, d = idx - key * {D}; int gk = kb + key;")
        w(f"              int kvb = gk * kvStride + h * {D} + d;")
        w(f"              ks[idx] = gk < Nk ? {KV}[kvb + kOff] : 0f; vs[idx] = gk < Nk ? {KV}[kvb + vOff] : 0f; }}")
    w("            Group.Barrier();")
    for r in range(QT):
        w("            float " + ", ".join(f"s{r}_{j}" for j in range(BKT)) + ";")
    for j in range(BKT):
        w("            { " + " ".join(f"float k{t} = ks[{j * D + P_ * t} + p];" for t in range(DT)))
        for r in range(QT):
            w(f"              s{r}_{j} = " + " + ".join(f"q{r}_{t} * k{t}" for t in range(DT)) + ";")
            for st in range(lg):
                w(f"              s{r}_{j} += Warp.ShuffleXor(s{r}_{j}, {1 << st});")
            w(f"              if (kb + {j} >= Nk) s{r}_{j} = -1e30f;")
        w("            }")
    for r in range(QT):
        mx = f"s{r}_0"
        for j in range(1, BKT):
            mx = f"MathF.Max({mx}, s{r}_{j})"
        w(f"            float mn{r} = MathF.Max(m{r}, {mx});")
        w(f"            float corr{r} = MathF.Exp(m{r} - mn{r});")
        w(f"            m{r} = mn{r};")
        for j in range(BKT):
            w(f"            float p{r}_{j} = MathF.Exp(s{r}_{j} - mn{r});")
        w(f"            l{r} = l{r} * corr{r} + (" + " + ".join(f"p{r}_{j}" for j in range(BKT)) + ");")
        w("            " + " ".join(f"o{r}_{t} *= corr{r};" for t in range(DT)))
    for j in range(BKT):
        w("            { " + " ".join(f"float v{t} = vs[{j * D + P_ * t} + p];" for t in range(DT)))
        for r in range(QT):
            w("              " + " ".join(f"o{r}_{t} += p{r}_{j} * v{t};" for t in range(DT)))
        w("            }")
    w("            Group.Barrier();")
    w("        }")
    for r in range(QT):
        w(f"        if (ok{r})")
        w("        {")
        w(f"            float inv = 1f / l{r};")
        w(f"            int ob = row{r} * H * {D} + h * {D} + p;")
        w("            " + " ".join(f"O[ob + {P_ * t}] = o{r}_{t} * inv;" for t in range(DT)))
        w("        }")
    w("    }")
    w("")

def kernel(D, self_attn, shuffle):
    if shuffle:
        return kernel_shuffle(D, self_attn)
    BKT = 8
    DT = D // P                 # dims per lane
    name = f"{'Self' if self_attn else 'Cross'}D{D}{'' if shuffle else 'Smem'}"
    if self_attn:
        w(f"    /// <summary>D = {D}, Q and K/V in one buffer. grid = ceil(Nq / {QPG}) * H, group = {TH}.</summary>")
        w(f"    public static void {name}(FV S, FV O, int Nq, int Nk, int H, int qStride, int qOff, int kvStride, int kOff, int vOff)")
        Q = KV = "S"
    else:
        w(f"    /// <summary>D = {D}, Q and K/V in separate buffers. grid = ceil(Nq / {QPG}) * H, group = {TH}.</summary>")
        w(f"    public static void {name}(FV Q, FV KV, FV O, int Nq, int Nk, int H, int qStride, int qOff, int kvStride, int kOff, int vOff)")
        Q, KV = "Q", "KV"
    w("    {")
    w("        int qTile = Grid.IdxX / H, h = Grid.IdxX - qTile * H;")
    w(f"        int tid = Group.IdxX, p = tid & {P - 1}, ql = tid >> 2;")
    w(f"        int row = qTile * {QPG} + ql;")
    w("        bool ok = row < Nq;")
    w(f"        var ks = SharedMemory.Allocate<float>({BKT * D});")
    w(f"        var vs = SharedMemory.Allocate<float>({BKT * D});")
    if not shuffle:
        w(f"        var red = SharedMemory.Allocate<float>({BKT * P * QPG});   // partial dots [key][lane][query]")
    w(f"        long qb = (long)row * qStride + qOff + h * {D} + p;")
    w(f"        const float scale = {1.0 / (D ** 0.5)!r}f;")
    for t in range(DT):
        w(f"        float q{t} = ok ? {Q}[qb + {P * t}] * scale : 0f;")
    for t in range(DT):
        w(f"        float o{t} = 0f;")
    w("        float m = -1e30f, l = 0f;")
    loads = BKT * D // TH
    w(f"        for (int kb = 0; kb < Nk; kb += {BKT})")
    w("        {")
    for r in range(loads):
        w(f"            {{ int idx = {r * TH} + tid; int key = idx / {D}, d = idx - key * {D}; int gk = kb + key;")
        w(f"              long kvb = (long)gk * kvStride + h * {D} + d;")
        w(f"              ks[idx] = gk < Nk ? {KV}[kvb + kOff] : 0f; vs[idx] = gk < Nk ? {KV}[kvb + vOff] : 0f; }}")
    w("            Group.Barrier();")
    # scores
    for j in range(BKT):
        terms = " + ".join(f"q{t} * ks[{j * D} + {P * t} + p]" for t in range(DT))
        w(f"            float s{j} = {terms};")
        if shuffle:
            w(f"            s{j} += Warp.ShuffleXor(s{j}, 1); s{j} += Warp.ShuffleXor(s{j}, 2);")
        else:
            w(f"            red[ql + {QPG} * (p + {P * j})] = s{j};")
    if not shuffle:
        w("            Group.Barrier();")
        for j in range(BKT):
            w(f"            s{j} = " + " + ".join(f"red[ql + {QPG * (P * j + k)}]" for k in range(P)) + ";")
    for j in range(BKT):
        w(f"            if (kb + {j} >= Nk) s{j} = -1e30f;")
    mx = "s0"
    for j in range(1, BKT):
        mx = f"MathF.Max({mx}, s{j})"
    w(f"            float mn = MathF.Max(m, {mx});")
    w("            float corr = MathF.Exp(m - mn);")
    w("            m = mn;")
    for j in range(BKT):
        w(f"            float p{j} = MathF.Exp(s{j} - mn);")
    w("            l = l * corr + (" + " + ".join(f"p{j}" for j in range(BKT)) + ");")
    for t in range(DT):
        terms = " + ".join(f"p{j} * vs[{j * D} + {P * t} + p]" for j in range(BKT))
        w(f"            o{t} = o{t} * corr + {terms};")
    w("            Group.Barrier();")
    w("        }")
    w("        if (ok)")
    w("        {")
    w("            float inv = 1f / l;")
    w(f"            long ob = (long)row * H * {D} + h * {D} + p;")
    for t in range(DT):
        w(f"            O[ob + {P * t}] = o{t} * inv;")
    w("        }")
    w("    }")
    w("")

w("// <auto-generated> by tools/trellis2-harness/reference/gen_tiled_attention.py - edit the generator, not this file. </auto-generated>")
w("using ILGPU;")
w("using ILGPU.Runtime;")
w("")
w("namespace SpawnDev.ILGPU.ML.Trellis2;")
w("")
w("using FV = ArrayView1D<float, Stride1D.Dense>;")
w("")
w("/// <summary>Portable flash attention for every backend (see the generator for the layout).</summary>")
w("internal static class TiledAttention")
w("{")
w(f"    /// <summary>Queries per group: shuffle kernels ({SP} lanes x {SQT} queries per thread), *Smem kernels.</summary>")
w(f"    public const int QueriesPerGroup = {(TH // SP) * SQT}, QueriesPerGroupSmem = {QPG}, Threads = {TH}, LanesPerQuery = {SP};")
w("")
for D in (128, 64):
    for sh in (True, False):
        kernel(D, False, sh)
        kernel(D, True, sh)
w("}")

open(sys.argv[1], "w", encoding="utf-8", newline="\n").write("\n".join(out) + "\n")
print("wrote", sys.argv[1], len(out), "lines")
