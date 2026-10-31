"""Pilot: does PRM-guided selection help a small math model enough to build search into ML?

  python pilot_search.py ROOT [N_PROBLEMS] [N_SAMPLES]      (ROOT holds Qwen2.5-Math-1.5B-Instruct/, Skywork-o1-Open-PRM-Qwen-2.5-1.5B/, MATH-500/test.jsonl)

Policy Qwen2.5-Math-1.5B-Instruct (Apache-2.0) with its own CoT prompt; per problem one greedy answer and N samples (temperature 0.7, top-p 0.8). Each sample is scored by
Skywork-o1-Open-PRM-Qwen-2.5-1.5B (Skywork Community License): Skywork's prepare_input (step token "\\n"), sigmoid of the value head at each step's last token. Answers are
checked with math_verify against MATH-500's answer. Reported: greedy accuracy, majority@N, PRM best-of-N (mean / min / last step reward), PRM-weighted majority, pass@N.
Results append to ROOT/pilot_results.jsonl one problem per line (a rerun skips the problems already there).
"""
import json, math, random, sys, time
from collections import defaultdict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Qwen2Config, Qwen2Model
from math_verify import parse, verify

root = sys.argv[1]
n_problems = int(sys.argv[2]) if len(sys.argv) > 2 else 50
n_samples = int(sys.argv[3]) if len(sys.argv) > 3 else 8
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
dev = "cuda"

problems = [json.loads(l) for l in open(f"{root}/MATH-500/test.jsonl", encoding="utf-8")]
random.Random(0).shuffle(problems)
problems = problems[:n_problems]
out_path = f"{root}/pilot_results.jsonl"
done = set()
try:
    for l in open(out_path, encoding="utf-8"):
        done.add(json.loads(l)["unique_id"])
except FileNotFoundError:
    pass

ptok = AutoTokenizer.from_pretrained(f"{root}/Qwen2.5-Math-1.5B-Instruct")
ptok.padding_side = "left"
policy = AutoModelForCausalLM.from_pretrained(f"{root}/Qwen2.5-Math-1.5B-Instruct", dtype=torch.bfloat16).to(dev).eval()

# the PRM: a plain Qwen2 trunk + Skywork's ValueHead (Linear(hidden, 1)); dropout is identity at eval
prm_dir = f"{root}/Skywork-o1-Open-PRM-Qwen-2.5-1.5B"
rtok = AutoTokenizer.from_pretrained(prm_dir)
cfg = Qwen2Config.from_pretrained(prm_dir)
sd = torch.load(f"{prm_dir}/pytorch_model.bin", map_location="cpu", weights_only=True, mmap=True)     # mapped, not read: WSL has 7 GB
torch.set_default_dtype(torch.bfloat16)
with torch.device(dev):
    trunk = Qwen2Model(cfg)                     # built on the GPU in bf16 (on the CPU in fp32 it alone is 6 GB)
torch.set_default_dtype(torch.float32)
missing, unexpected = trunk.load_state_dict({k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}, strict=False)
assert not missing, missing
trunk = trunk.eval()
v_head = torch.nn.Linear(cfg.hidden_size, 1).to(dev)
v_head.weight.data = sd["v_head.summary.weight"].float().to(dev)
v_head.bias.data = sd["v_head.summary.bias"].float().to(dev)
del sd


def prepare_input(problem, response, step_token="\n"):        # Skywork's model_utils/io_utils.py (Apache-2.0), verbatim logic
    prompt_ids = rtok.encode(rtok.bos_token + problem + "\n")
    response_ids, flags = [], [0] * len(prompt_ids)
    step_token_id = rtok.encode(step_token)[-1]
    for step in response.split(step_token):
        step_ids = rtok.encode(step) if step != "" else []
        step_ids += [step_token_id]
        f = [0] * len(step_ids); f[-1] = 1
        response_ids.extend(step_ids); flags.extend(f)
    return prompt_ids + response_ids, flags


@torch.no_grad()
def step_rewards(problem, response):
    ids, flags = prepare_input(problem, response)
    ids = ids[:4096]; flags = flags[:4096]
    h = trunk(input_ids=torch.tensor([ids], device=dev)).last_hidden_state[0].float()
    v = torch.sigmoid(v_head(h).squeeze(-1)).cpu().tolist()
    return [v[i] for i, f in enumerate(flags) if f == 1]


def chat(problem):
    msgs = [{"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."}, {"role": "user", "content": problem}]
    return ptok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def generate(problem, n, greedy):
    enc = ptok([chat(problem)] * n, return_tensors="pt", padding=True).to(dev)
    kw = dict(max_new_tokens=1024, pad_token_id=ptok.eos_token_id)
    kw.update(dict(do_sample=False) if greedy else dict(do_sample=True, temperature=0.7, top_p=0.8))
    outs = policy.generate(**enc, **kw)
    return [ptok.decode(o[enc.input_ids.shape[1]:], skip_special_tokens=True) for o in outs]


def correct(answer, gold):
    try:
        return bool(verify(parse("$" + gold + "$"), parse(answer)))
    except Exception:
        return False


def key(answer):
    p = parse(answer)
    return str(p[-1]) if p else ""


torch.manual_seed(0)
t0 = time.time()
with open(out_path, "a", encoding="utf-8") as fo:
    for k, pr in enumerate(problems):
        if pr["unique_id"] in done:
            continue
        g = generate(pr["problem"], 1, True)[0]
        samples = generate(pr["problem"], n_samples, False)
        rec = {"unique_id": pr["unique_id"], "level": pr["level"], "gold": pr["answer"], "greedy_ok": correct(g, pr["answer"]), "samples": []}
        for s in samples:
            r = step_rewards(pr["problem"], s)
            rec["samples"].append({"ok": correct(s, pr["answer"]), "key": key(s), "mean": sum(r) / len(r) if r else 0, "min": min(r) if r else 0, "last": r[-1] if r else 0, "steps": len(r)})
        fo.write(json.dumps(rec) + "\n"); fo.flush()
        print(f"{k + 1}/{len(problems)} level {pr['level']} greedy {rec['greedy_ok']} samples {sum(s['ok'] for s in rec['samples'])}/{n_samples} ({time.time() - t0:.0f} s)", flush=True)

# the report over every recorded problem
recs = [json.loads(l) for l in open(out_path, encoding="utf-8")]
n = len(recs)
def acc(pick): return 100.0 * sum(pick(r) for r in recs) / n
def best(r, f): return max(r["samples"], key=lambda s: s[f])["ok"]
def maj(r, weight):
    votes = defaultdict(float)
    for s in r["samples"]:
        if s["key"]: votes[s["key"]] += weight(s)
    if not votes: return False
    top = max(votes, key=votes.get)
    return any(s["ok"] for s in r["samples"] if s["key"] == top)
print(f"\n{n} problems, {n_samples} samples each")
print(f"greedy                 {acc(lambda r: r['greedy_ok']):5.1f}")
print(f"majority@{n_samples}             {acc(lambda r: maj(r, lambda s: 1.0)):5.1f}")
print(f"PRM best-of-{n_samples} (mean)   {acc(lambda r: best(r, 'mean')):5.1f}")
print(f"PRM best-of-{n_samples} (min)    {acc(lambda r: best(r, 'min')):5.1f}")
print(f"PRM best-of-{n_samples} (last)   {acc(lambda r: best(r, 'last')):5.1f}")
print(f"PRM-weighted majority  {acc(lambda r: maj(r, lambda s: s['mean'])):5.1f}")
print(f"pass@{n_samples} (oracle)        {acc(lambda r: any(s['ok'] for s in r['samples'])):5.1f}")
