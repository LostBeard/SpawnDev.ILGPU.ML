"""Compare slot-packing strategies for BufferPool planned reuse offline, on recordings dumped with ML_PLANNED_REUSE_DUMP=<dir> (plan-NNN.csv:
seq, bucket elements, rent clock, return clock or -1). Prints, per recording, the live peak (the floor: the most fp32 bytes live at one instant) and the
total slot bytes of each strategy:
  size-first   : greedy by size, largest rent first, into the FIRST slot with no overlapping interval (what BufferPool builds, 2026-11-03)
  size-best    : greedy by size, into the SMALLEST fitting slot with no overlap (all existing slots are >= the rent, so: the smallest slot)
  breadth      : TFLite GPU's greedy by breadth: visit the instants in order of decreasing live bytes, and at each place its live rents (largest first)
                 into the smallest free slot that fits, else the largest free slot (grown to fit), else a new slot
Usage: py -3.13 tools/planned_reuse_sim.py <dir or csv> [...]
"""
import csv, glob, os, sys

MiB = 1048576.0


def load(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append((int(r["bucket"]), int(r["rent"]), int(r["ret"])))
    return [(b * 4, a, e) for b, a, e in rows if e >= 0]   # bytes; only rents returned within the run are planned


def overlap(a, b):
    return a[1] < b[2] and b[1] < a[2]


def live_peak(rs):
    events = []
    for size, a, e in rs:
        events.append((a, size))
        events.append((e, -size))
    events.sort()
    cur = peak = 0
    for _, d in events:
        cur += d
        peak = max(peak, cur)
    return peak


def greedy_size(rs, best):
    order = sorted(rs, key=lambda r: (-r[0], r[1]))
    slots = []   # [size, [intervals]]
    for r in order:
        fits = [s for s in slots if not any(overlap(r, o) for o in s[1])]
        if fits:
            s = min(fits, key=lambda s: s[0]) if best else fits[0]
        else:
            s = [r[0], []]
            slots.append(s)
        s[1].append(r)
    return sum(s[0] for s in slots)


def greedy_breadth(rs):
    clock = sorted({a for _, a, _ in rs} | {e for _, _, e in rs})
    breadth = []
    for t in clock:
        live = [i for i, (s, a, e) in enumerate(rs) if a <= t < e]
        breadth.append((sum(rs[i][0] for i in live), t, live))
    breadth.sort(key=lambda x: -x[0])
    slots = []   # [size, [rent indices]]
    assigned = {}
    for _, _, live in breadth:
        for i in sorted(live, key=lambda i: -rs[i][0]):
            if i in assigned: continue
            r = rs[i]
            free = [s for s in slots if not any(overlap(r, rs[j]) for j in s[1])]
            fit = [s for s in free if s[0] >= r[0]]
            if fit: s = min(fit, key=lambda s: s[0])
            elif free:
                s = max(free, key=lambda s: s[0]); s[0] = r[0]
            else:
                s = [r[0], []]; slots.append(s)
            s[1].append(i); assigned[i] = s
    return sum(s[0] for s in slots)


def main(args):
    files = []
    for a in args:
        files += sorted(glob.glob(os.path.join(a, "plan-*.csv"))) if os.path.isdir(a) else [a]
    for f in files:
        rs = load(f)
        if not rs: continue
        print(f"{os.path.basename(f)}: {len(rs)} planned rents | live peak {live_peak(rs) / MiB:8.1f} MiB | size-first {greedy_size(rs, False) / MiB:8.1f}"
              f" | size-best {greedy_size(rs, True) / MiB:8.1f} | breadth {greedy_breadth(rs) / MiB:8.1f} MiB")


if __name__ == "__main__":
    main(sys.argv[1:])
