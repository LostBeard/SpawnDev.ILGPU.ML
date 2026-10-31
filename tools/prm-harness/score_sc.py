"""Score sc_eval.cs's output the way pilot_search.py scores PyTorch's: math_verify against MATH-500's answer.

  python score_sc.py RESULTS.jsonl

Reported: greedy, the C# vote (SelfConsistency: normalised boxed strings; right when the sample it took is right), the pilot's vote (math_verify's parse as the key),
pass@N, and how often the two votes disagree.
"""
import json, sys
from collections import defaultdict
from math_verify import parse, verify


def correct(answer, gold):
    try:
        return bool(verify(parse("$" + gold + "$"), parse(answer)))
    except Exception:
        return False


def key(answer):
    try:
        p = parse(answer)
        return str(p[-1]) if p else ""
    except Exception:
        return ""


recs = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")]
n = len(recs)
greedy = ours = pilots = anyok = disagree = 0
secs = toks = 0.0
for r in recs:
    ok = [correct(s, r["gold"]) for s in r["samples"]]
    greedy += correct(r["greedy"], r["gold"])
    ours += r["vote_member"] >= 0 and ok[r["vote_member"]]
    votes = defaultdict(int)
    keys = [key(s) for s in r["samples"]]
    for k in keys:
        if k: votes[k] += 1
    if votes:
        top = max(votes, key=votes.get)
        pilot_ok = any(o for o, k in zip(ok, keys) if k == top)
        pilots += pilot_ok
        disagree += pilot_ok != (r["vote_member"] >= 0 and ok[r["vote_member"]])
    anyok += any(ok)
    secs += r["greedy_seconds"] + r["sample_seconds"]; toks += r["sample_tokens"]
ns = len(recs[0]["samples"]) if recs else 0
print(f"{n} problems, {ns} samples each")
print(f"greedy                    {100.0 * greedy / n:5.1f}")
print(f"majority@{ns} (C# vote)      {100.0 * ours / n:5.1f}")
print(f"majority@{ns} (math_verify)  {100.0 * pilots / n:5.1f}   (the two votes differ on {disagree})")
print(f"pass@{ns}                    {100.0 * anyok / n:5.1f}")
print(f"time {secs / 60:.1f} min, {toks / max(1e-9, sum(r['sample_seconds'] for r in recs)):.0f} sampled tok/s")
