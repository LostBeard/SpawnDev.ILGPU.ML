"""Print check_results.json as a Markdown table (for ANIMA-SPEC.md section 5).
Order: DiT stages, DiT end to end, adapter stages, adapter end to end."""
import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else r'D:\users\tj\Projects\SpawnDev.ILGPU.ML\SpawnDev.ILGPU.ML\_scratch\anima-ref\check_results.json'
rows = json.load(open(path))


def group(r):
    s = r['stage']
    adapter = 'adapter' in s
    e2e = 'END-TO-END' in s or 'chained' in s or 'call_' in s or 'padded to 512' in s
    return (1 if adapter else 0) * 2 + (1 if e2e else 0)


rows = sorted(rows, key=group)  # stable: keeps the run order inside each group
print('| stage | max abs | relRMS | note |')
print('|---|---|---|---|')
for r in rows:
    note = r['note'].replace('|', '\\|')
    print(f"| {r['stage'].strip()} | {r['max_abs']:.3e} | {r['rel_rms']:.3e} | {note} |")
