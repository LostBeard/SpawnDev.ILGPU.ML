# Find the FIRST node where our engine disagrees with onnxruntime.
#
#   1) python tools/zipvoice/ort_intermediates.py "<substring>" > ort.txt      (ZipVoice encoder)
#      python tools/ort_node_reference.py <model> <fixture> "<substring>" > ort.txt   (any model)
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
#
# ⚠️ AND a node PRESENT in our dump is not automatically comparable either. A shape-lane tensor prints
# TWICE: an `[interp]` line carrying the value the graph actually consumed, and a `[dump]` line reading the
# GPU buffer registered under that name - which for an interpreter-evaluated node is a stale allocation,
# usually all zeros. Comparing the `[dump]` line then reports a 100% divergence on a node that is entirely
# correct. MEASURED on Silero VAD: `/feature_extractor/Slice_output_0` dumped as zeros while its `[interp]`
# line held [0,0,0,0,96,96] - matching ORT exactly - and the Pad consuming it agreed with ORT to six
# decimal places. That false lead is why this handling exists.
#
# So: where an `[interp]` line exists it WINS over the `[dump]` line for the same tensor. A TRUNCATED
# interp line (one printing "(+56)") has no honest min/max, so it is EXCLUDED rather than compared on a
# prefix - reporting nothing beats reporting a number computed from the first eight elements.
import sys, re

if len(sys.argv) < 3:
    print(__doc__)
    sys.exit(2)

ort_path, ours_path = sys.argv[1], sys.argv[2]
tol = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-3

# ORT:  "  MatMul   [13, 1, 272]   -0.708038  0.793441  0.002164  /name"
NUM = r"(-?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|-?inf|nan|-)"
ort_re = re.compile(r"^\s{2}(\S+)\s+(\[[^\]]*\])\s+" + NUM + r"\s+" + NUM + r"\s+" + NUM + r"\s+(\S+)\s*$")
# ours: "[dump]   35 Slice   /name shape=[13,1,128] min=-0.8053 max=0.8914 mean=0.01709 nonfinite=0 ..."
our_re = re.compile(r"^\[dump\]\s+(\d+)\s+(\S+)\s+(\S+)\s+shape=(\[[^\]]*\])\s+min=(\S+)\s+max=(\S+)\s+mean=(\S+)")
# interp: "[interp]    8 Slice   /name = [0,0,0,0,96,96] <- [...]"  or "... = [0.324,0.343] (+56) <- [...]"
interp_re = re.compile(r"^\[interp\]\s+(\d+)\s+(\S+)\s+(\S+)\s+=\s+\[([^\]]*)\](\s+\(\+\d+\))?")


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
interp_full, interp_truncated = {}, set()
for line in open(ours_path, encoding="utf-8", errors="replace"):
    line = line.rstrip("\n")
    m = our_re.match(line)
    if m:
        _, op, name, shape, lo, hi, mean = m.groups()
        ours[name] = (op, shape, fnum(lo), fnum(hi), fnum(mean))
        continue
    m = interp_re.match(line)
    if m:
        _, op, name, body, more = m.groups()
        if more:
            interp_truncated.add(name)
            continue
        vals = [v for v in (fnum(t) for t in body.split(",") if t.strip()) if v is not None]
        if vals:
            interp_full[name] = (op, min(vals), max(vals), sum(vals) / len(vals))

# The interpreter value is what the graph consumed, so it wins wherever we have a complete one.
shadowed = sum(1 for n in interp_full if n in ours)
for name, (op, lo, hi, mean) in interp_full.items():
    shape = ours[name][1] if name in ours else "[interp]"
    ours[name] = (op, shape, lo, hi, mean)

print(f"ORT tensors: {len(ort)}   ours: {len(ours)}   tolerance: {tol}")
if shadowed:
    print(f"interp-lane tensors compared on the INTERPRETER value, not the stale GPU buffer: {shadowed}")
if interp_truncated:
    print(f"interp-lane tensors excluded (truncated value list, no honest min/max): {len(interp_truncated)}")


def rel(a, b):
    if a is None or b is None: return None
    scale = max(abs(a), abs(b), 1e-6)
    return abs(a - b) / scale


compared = mismatches = 0
missing = []
excluded = []
first = None
for name in order:
    if name in interp_truncated and name not in interp_full:
        excluded.append(name)
        continue
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

print(f"compared: {compared}   mismatching: {mismatches}   in ORT but not dumped by us: {len(missing)}"
      + (f"   excluded (truncated interp): {len(excluded)}" if excluded else ""))

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
