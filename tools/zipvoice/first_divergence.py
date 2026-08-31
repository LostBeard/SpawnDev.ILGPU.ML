# Find the FIRST node where our engine disagrees with onnxruntime.
#
#   1) python tools/zipvoice/ort_intermediates.py "<substring>" > ort.txt
#   2) ML_DUMP_TENSORS="<substring>" dotnet run --project tools/zipvoice-harness -c Release -- compare > ours.txt
#   3) python tools/zipvoice/first_divergence.py ort.txt ours.txt
#
# Everything downstream of the first disagreement is corrupted by it, so only the first one is a lead.
# Comparing min/max/mean rather than full tensors is deliberate: it is what both sides already print, it
# needs no aligned element ordering, and a real op bug moves at least one of the three.
#
# ⚠️ A node present in ORT but absent from our dump is NOT automatically a bug - our executor elides
# shape-only nodes and evaluates them in the interpreter instead (those print as `[interp]`). Missing
# entries are reported separately rather than counted as mismatches.
import sys, re

if len(sys.argv) < 3:
    print(__doc__)
    sys.exit(2)

ort_path, ours_path = sys.argv[1], sys.argv[2]
tol = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-3

# ORT:  "  MatMul   [13, 1, 272]   -0.708038  0.793441  0.002164  /name"
ort_re = re.compile(r"^\s{2}(\S+)\s+(\[[^\]]*\])\s+(-?[\d.]+|-)\s+(-?[\d.]+|-)\s+(-?[\d.]+|-)\s+(\S+)\s*$")
# ours: "[dump]   35 Slice   /name shape=[13,1,128] min=-0.8053 max=0.8914 mean=0.01709 nonfinite=0 ..."
our_re = re.compile(r"^\[dump\]\s+(\d+)\s+(\S+)\s+(\S+)\s+shape=(\[[^\]]*\])\s+min=(\S+)\s+max=(\S+)\s+mean=(\S+)")

def fnum(s):
    try: return float(s)
    except Exception: return None

ort = {}
order = []
for line in open(ort_path, encoding="utf-8", errors="replace"):
    m = ort_re.match(line.rstrip("\n"))
    if m:
        op, shape, lo, hi, mean, name = m.groups()
        ort[name] = (op, shape, fnum(lo), fnum(hi), fnum(mean))
        order.append(name)

ours = {}
for line in open(ours_path, encoding="utf-8", errors="replace"):
    m = our_re.match(line.rstrip("\n"))
    if m:
        _, op, name, shape, lo, hi, mean = m.groups()
        ours[name] = (op, shape, fnum(lo), fnum(hi), fnum(mean))

print(f"ORT tensors: {len(ort)}   ours: {len(ours)}   tolerance: {tol}")

def rel(a, b):
    if a is None or b is None: return None
    scale = max(abs(a), abs(b), 1e-6)
    return abs(a - b) / scale

compared = mismatches = 0
missing = []
first = None
for name in order:
    if name not in ours:
        missing.append(name)
        continue
    o_op, o_shape, o_lo, o_hi, o_mean = ort[name]
    m_op, m_shape, m_lo, m_hi, m_mean = ours[name]
    compared += 1
    ds = [rel(o_lo, m_lo), rel(o_hi, m_hi), rel(o_mean, m_mean)]
    worst = max((d for d in ds if d is not None), default=0.0)
    if worst > tol:
        mismatches += 1
        if first is None:
            first = (name, o_op, o_shape, m_shape, (o_lo, o_hi, o_mean), (m_lo, m_hi, m_mean), worst)

print(f"compared: {compared}   mismatching: {mismatches}   in ORT but not dumped by us: {len(missing)}")

if first:
    name, op, o_shape, m_shape, o, m, worst = first
    print("\n=== FIRST DIVERGENCE ===")
    print(f"  node   : {name}")
    print(f"  op     : {op}")
    print(f"  shape  : ORT {o_shape}   ours {m_shape}" + ("   <-- SHAPE DIFFERS" if o_shape.replace(' ','') != m_shape.replace(' ','') else ""))
    print(f"  min    : ORT {o[0]!s:>14}   ours {m[0]!s:>14}")
    print(f"  max    : ORT {o[1]!s:>14}   ours {m[1]!s:>14}")
    print(f"  mean   : ORT {o[2]!s:>14}   ours {m[2]!s:>14}")
    print(f"  worst relative difference: {worst:.4%}")
    print("\nEverything after this node is downstream of the error. Fix this one first.")
else:
    print("\nNo divergence above tolerance in the compared set.")
    if missing:
        print("Nodes ORT produced that we never dumped (shape-only nodes are expected here):")
        for n in missing[:15]:
            print(f"   {ort[n][0]:16s} {n}")
