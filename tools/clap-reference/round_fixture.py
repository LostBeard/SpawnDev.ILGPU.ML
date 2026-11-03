"""Round a make_fixtures.py JSON to 5 decimals for the demo's wwwroot (tests hold the towers to cosine / 0.01-probability tolerances, far above
the rounding). Usage: python round_fixture.py full.json rounded.json"""
import json, sys


def rnd(v):
    if isinstance(v, float): return round(v, 5)
    if isinstance(v, list): return [rnd(x) for x in v]
    if isinstance(v, dict): return {k: rnd(x) for k, x in v.items()}
    return v


json.dump(rnd(json.load(open(sys.argv[1]))), open(sys.argv[2], "w"), separators=(",", ":"))
