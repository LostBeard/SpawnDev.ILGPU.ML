# Build the fixture that gates the SUBGRAPH PLAN CACHE, and its onnxruntime reference outputs.
#
#   python tools/gen_subgraph_plan_cache_reference.py
#
# WHY THIS EXISTS
# ---------------
# SubgraphRunner caches each compiled If/Loop branch on the session's OperatorRegistry, keyed on THE
# SUBGRAPH'S OWN INPUT SHAPES. An If's only explicit input is its condition - a scalar - so that key is
# IDENTICAL at every outer shape, and one cached entry served every shape the session ever ran. The entry
# holds a SHAPE-SPECIALISED GraphExecutor with Constants allocated from one particular executor's
# BufferPool, so a plan built while running one shape was handed to a different one.
#
# 5.2.10 rejected such a plan when its pool had been DISPOSED. 5.2.11 rejects it unless the pool is the
# CALLER'S pool, which is the commoner case: alive, but belonging to a different still-cached executor
# (InferenceSession.MaxShapeExecutors = 3).
#
# Found through ZipVoice, where a synthesis's audio depended on WHAT HAD BEEN SYNTHESISED BEFORE IT. That
# proof needed a model download, a browser and Whisper. This fixture is the same defect in a few kilobytes.
#
# ⚠️ THE BRANCH MUST READ AN OUTER-SCOPE TENSOR WHOSE SHAPE VARIES. That is the whole point: the cache key
# covers only the subgraph's EXPLICIT inputs, so a branch that depends on nothing but its condition would
# be legitimately reusable and would prove nothing. `X` is captured from the enclosing graph and is the
# thing the cache key cannot see.
#
# 🔴 READ THIS BEFORE TRUSTING THIS FIXTURE TO GATE THE PLAN CACHE - IT DOES NOT.
#
# MEASURED 2026-09-06: with the 5.2.11 fix DISABLED
# (`if (!ReferenceEquals(candidate.ConstantsPool, ctx.Pool)) continue;` commented out), the test built on
# this fixture still passed 8 of 8. The premise was that the plan-cache key covers only the subgraph's
# DECLARED inputs, so a branch reading a dynamic outer-scope tensor would collide across outer shapes.
# That is wrong: IfOperator.ExecuteAsync calls OuterScope.Add - "every tensor the subgraph references but
# does not itself produce" - BEFORE SubgraphRunner.ExecuteAsync, so the captured tensor's SHAPE is part of
# the signature and the plan is correctly rebuilt per shape. A branch capturing a varying-shape tensor can
# never collide. This fixture is a genuine If/outer-scope/eviction correctness case and nothing more.
#
# ── THE FIXTURE THAT WOULD ACTUALLY REPRODUCE IT (not yet built) ──────────────────────────────────
# The signature must stay CONSTANT while EXECUTORS churn, which is the opposite of what is below: the
# branch's captures must be shape-INVARIANT while the OUTER graph's shape varies. ZipVoice's
# relative-position If captures [1] scalars, which is exactly why one entry served every utterance length.
#
#   inputs : cond bool[1], S float[1], X float["N", 4]
#   branch : then_out = Mul(table_const[4], S)        # captures ONLY S -> sig is {cond:[1], S:[1]}, constant
#   outer  : Y = ReduceSum(Mul(X, then_out), axes=[1]) # ["N"] - the branch result reaches the output
#   run    : five distinct N so four evictions occur, repeating an earlier N at the end
#
# Then the cached plan is built under ONE executor's pool and handed to every other N. Drive the red check
# against BOTH guards independently - disable the 5.2.10 `ConstantsPool.IsDisposed` check and the 5.2.11
# `ReferenceEquals` check in separate runs - and report which one it actually catches. It may only catch
# the disposed-pool case (constants become freed memory); an alive-but-foreign pool still holds correct
# values, so that half may not be observable this way at all. Say which, rather than assuming both.
#
# ── THE OTHER ROUTE, AND PROBABLY THE BETTER ONE: A REAL-MODEL DESKTOP GUARD ──────────────────────
# The synthetic fixture above is FAST but UNPROVEN - I could not make it go red. The ZipVoice route is
# the reverse: slower, needs the model, but the mechanism is already PROVEN to reproduce (it is what the
# SpawnDev.AI voice gate detects by audio hash). And it can live in THIS repo, with no browser:
#
#   `SpawnDev.ILGPU.ML/Pipelines/IlgpuZipVoiceGraphs.cs` holds OUR InferenceSession objects
#   (`InferenceSession.CreateFromFile(accelerator, ...)`), and the plan cache lives on the session's
#   OperatorRegistry - so ONE IlgpuZipVoiceGraphs shared across several synthesises is exactly the
#   precondition. A ZipVoicePipeline per utterance is fine; the SESSIONS are what must be shared.
#
#   Shape of the test (HeavyModel, desktop, alongside the existing Pipeline_ZipVoice_* rows):
#     1. build ONE IlgpuZipVoiceGraphs
#     2. synthesise utterances at >= 5 DISTINCT lengths through it, NoiseSeed pinned
#     3. FNV-hash each result, then repeat an earlier length
#     4. assert the repeat hashes IDENTICALLY to its first reading
#
#   That is the SpawnDev.AI voice gate's mechanism (296 chars -> 0b57ba1c1373a9fb at both positions after
#   the fix; two different hashes before it) moved into the repo where the fix actually lives.
#
# ⚠️ `tools/zipvoice-harness`'s `endtoend`/`verify` modes look like they would do this - they share one
# graphs object across many fixtures - but they use `OrtZipVoiceGraphs`, which constructs ONNX RUNTIME's
# InferenceSession from a path. They never touch our plan cache. Do not mistake a green `endtoend` for
# coverage of this defect.
#
# ⚠️ FOUR distinct NON-BASE shapes, minimum - which means FIVE distinct shapes in the sequence.
# InferenceSession.ResolveExecutor returns the BASE executor whenever the run's shapes match the ones the
# session was CREATED with, and "the base executor is never in this cache" (InferenceSession.cs:245). So the
# creation shape costs nothing against MaxShapeExecutors = 3, and a naive [2,3,5,7,2] fills the LRU to
# exactly 3 and NEVER EVICTS - the fixture would run clean against the bug it exists for.
#
# An eviction is what leaves a cached subgraph plan pointing at an executor that is no longer the caller's,
# and eviction also DISPOSES the evicted executor's pool, so both the 5.2.10 shape of the defect (pool
# disposed) and the 5.2.11 shape (pool alive but owned by someone else) are reachable from here.
import json, os
import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                   "SpawnDev.ILGPU.ML.Demo", "wwwroot", "references", "controlflow"))
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(20260906)

NAME = "if_outer_scope_dynamic"
COLS = 4

# The rows each case runs, in order. Walking the LRU (base = 2, cap = 3):
#
#   2 -> base            3 -> [3]         5 -> [3,5]        7 -> [3,5,7]
#   9 -> [5,7,9]  EVICTS 3     <- the first eviction, which is where the ZipVoice failure appeared
#   3 -> [7,9,3]  EVICTS 5     <- the first REBUILD after an eviction
#   5 -> [9,3,5]  EVICTS 7
#   2 -> base                  <- back to the base executor, after four evictions
#
# 3, 5 and 2 are each run TWICE, and every repeat is compared to its own first reading bit for bit - the
# comparison the ZipVoice audio hashes made, and the one word overlap was too coarse to make.
ROWS = [2, 3, 5, 7, 9, 3, 5, 2]

# A Constant inside each branch, so the plan really does allocate Constants from a pool - the buffers whose
# ownership the fix is about. Distinct per branch so a wrong branch is unmistakable.
THEN_SCALE = rng.standard_normal(COLS).astype(np.float32)
ELSE_SCALE = (THEN_SCALE * -3.0).astype(np.float32)


def branch(tag, scale, out_name):
    """
    A branch that READS OUTER-SCOPE `X` - the dependency the plan cache key cannot see.

    Mul broadcasts [N, COLS] against [COLS] and ReduceSum collapses to [N], so both the computation AND
    the output size follow the outer-scope shape. Nodes are emitted in dependency order because
    make_graph does not sort them and ReduceSum's axes input must exist first.
    """
    return helper.make_graph(
        [
            helper.make_node("Constant", [], [f"{tag}_scale"],
                             value=helper.make_tensor(f"{tag}_s", TensorProto.FLOAT, [COLS], scale)),
            helper.make_node("Constant", [], [f"{tag}_axes"],
                             value=helper.make_tensor(f"{tag}_a", TensorProto.INT64, [1], [1])),
            helper.make_node("Mul", ["X", f"{tag}_scale"], [f"{tag}_scaled"]),
            helper.make_node("ReduceSum", [f"{tag}_scaled", f"{tag}_axes"], [out_name], keepdims=0),
        ],
        tag, [], [helper.make_tensor_value_info(out_name, TensorProto.FLOAT, ["N"])])


if_node = helper.make_node("If", ["cond"], ["Y"],
                           then_branch=branch("then", THEN_SCALE, "then_out"),
                           else_branch=branch("else", ELSE_SCALE, "else_out"))

graph = helper.make_graph(
    [if_node], NAME,
    [helper.make_tensor_value_info("cond", TensorProto.BOOL, [1]),
     helper.make_tensor_value_info("X", TensorProto.FLOAT, ["N", COLS])],
    [helper.make_tensor_value_info("Y", TensorProto.FLOAT, ["N"])])

model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
model.ir_version = 9
onnx.checker.check_model(model)
onnx.save(model, os.path.join(OUT, f"{NAME}.onnx"))

sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
cases = []
for i, n in enumerate(ROWS):
    # Deterministic per ROW COUNT, not per case, so the repeat of 2 feeds byte-identical input and any
    # difference in the output is the engine's alone.
    #
    # ⚠️ EVERY ROW'S VALUES MUST DEPEND ON N. A plain arange makes row i identical at every N, so a stale
    # plan that read the leading rows of a LARGER shape's buffer would return exactly the right numbers
    # and the gate would pass against the very bug it exists for - the same trap the control-flow fixtures
    # avoid by making each output larger than inputs[0]. The (n + 1) factor makes any cross-shape read
    # numerically obvious.
    x = (np.arange(n * COLS, dtype=np.float32).reshape(n, COLS) / 10.0 + 1.0) * (n + 1)
    feeds = {"cond": np.array([True]), "X": x}
    y = sess.run(["Y"], feeds)[0]
    cases.append({
        "index": i,
        "n": int(n),
        "inputs": {k: dict(shape=list(np.asarray(v).shape), data=np.asarray(v).ravel().tolist())
                   for k, v in feeds.items()},
        "outputs": {"Y": dict(shape=list(y.shape), data=y.ravel().tolist())},
    })
    print(f"  case {i}: N={n} -> Y{list(y.shape)} {np.round(y, 4).tolist()}")

with open(os.path.join(OUT, f"{NAME}.json"), "w", encoding="utf-8") as f:
    json.dump({"rows": ROWS, "cols": COLS, "cases": cases}, f)

print(f"{NAME} -> {len(cases)} cases, rows {ROWS}")
