# System One — local decision heads

System One is a **typed decision API** over a float state vector: choice / score / noul answers with Softmax probabilities. It is inspired by the Jev/Laya "System One" category (structured decisions, not text generation). This library does **not** port Laya/Jev weights or a text encoder.

Backed by a tiny GPU MLP via [`TrainableModel`](../SpawnDev.ILGPU.ML/Training/TrainableModel.cs). Same kernels run on CUDA, OpenCL, CPU, WebGPU, WebGL, and Wasm.

| Piece | Where |
|-------|--------|
| Library API | `SpawnDev.ILGPU.ML.SystemOne` |
| Training MLP | `SpawnDev.ILGPU.ML.Training.TrainableModel` |
| Classic Snake demo | `/snake` (train on-device, play human or GPU) |
| Tests | `PMT_FILTER=SystemOne` |

---

## Basics

### Create a head

```csharp
using SpawnDev.ILGPU.ML.SystemOne;

// Generic — any stateDim / numOptions
using var head = new SystemOneDecisionHead(
    accelerator,
    stateDim: 16,
    numOptions: 4,
    hidden: 64,
    maxBatchSize: 64);

// Classic Snake demo helper (Demo.Shared — not in the NuGet package):
// using var snakeHead = SnakeSystemOneSpec.CreateHead(accelerator);
```

Snake dims live in demo-only [`SnakeSystemOneSpec`](../SpawnDev.ILGPU.ML.Demo.Shared/Games/Snake/SnakeSystemOneSpec.cs) (`StateDim=40`, `NumActions=4`, `Hidden=128`). The library ships only the generic head.
### Ask questions

```csharp
float[] state = /* your features, length == head.StateDim */;

var moveQ = new ChoiceQuestion("next move", "up", "down", "left", "right");
var choice = await head.ChooseAsync(state, moveQ);
// choice.Choice, choice.Confidence, choice.Probabilities

var scoreQ = new ScoreQuestion("urgency", ["low", "med", "high", "crit"]);
var score = await head.ScoreAsync(state, scoreQ);
// score.Score = expected class index in [0, NumOptions)

var noulQ = new NoulQuestion("safe meal?"); // needs NumOptions == 2
var noul = await head.NoulAsync(state, noulQ);
// noul.Noul = P(true)

// Bundle several questions on one state
var resp = await head.DecideAsync(state, new Dictionary<string, SystemOneQuestion>
{
    ["move"] = moveQ,
    ["urgency"] = scoreQ,
});
// resp.Answers["move"], resp.DecisionLatencyMs
```

### Legal-action mask

Environments often forbid options (wall, reverse into body). Pass a `bool[]` of length `NumOptions`:

```csharp
bool[] allowed = [true, false, true, true]; // down illegal
var choice = await head.ChooseAsync(state, moveQ, allowed);
// Illegal probs → 0, remaining renormalized; new argmax among survivors.
```

`DecideAsync(..., allowedMask)` applies the same mask to every `ChoiceQuestion` in the batch. Score/noul answers ignore the mask. If every option is disallowed, the original answer is returned unchanged.

### Train (behavioral clone or any Softmax-CE labels)

```csharp
// Hot loop: no per-step host loss (dominant cost on WebGPU/Wasm)
head.TrainStep(batchInputs, batchLabels, batchSize, learningRate: 0.05f);
await head.FlushAsync();                    // every N batches
float loss = await head.ReadLastLossAsync(batchSize); // once per epoch
```

Or `TrainStepAsync(..., readLoss: false)` then `ReadLastLossAsync` at epoch end.

Labels are class indices in `[0, NumOptions)`.

### Save / load weights

```csharp
byte[] blob = await head.ExportWeightsAsync(); // ~23 KB for Snake (5764 floats + headers)
head.ImportWeights(blob);                      // dims must match this head
```

Format: `S1DH` header (stateDim / numOptions / hidden) + `TMPL` MLP blob (arch + FP32 params). Round-trip is gated by `SystemOne_Weights_RoundTrip_PreservesProbs`.

**Persistence tips**

| Sink | When |
|------|------|
| **localStorage** (base64) | Tiny heads (Snake ~31 KB string). Demo uses this. |
| **OPFS / file** | Multi-MB blobs, or when you already stream models that way. |

Bump [`SnakeSystemOneSpec.WeightsCacheVersion`](../SpawnDev.ILGPU.ML.Demo.Shared/Games/Snake/SnakeSystemOneSpec.cs) when encoder/teacher semantics change so demo caches invalidate.

---

## Classic Snake (demo reference)

Live at **`/snake`**. Flow:

1. Pick backend (WebGPU / WebGL / Wasm in browser).
2. **Train head** — behavioral clone of a flood-fill **safe teacher** (food only if path to tail remains; hunger / anti-stall at long length).
3. Head is **cached in localStorage**; reload restores without retrain.
4. Switch to **System One**, Start — each tick: encode → masked choose → step.

Demo helpers (not in the NuGet package; live under `Demo.Shared`):

| Type | Role |
|------|------|
| `SnakeSystemOneSpec` | Demo-only dims + `CreateHead` (not in NuGet) |
| `SnakeGame` | Grid sim |
| `SnakeStateEncoder` | 40-float features |
| `SnakeTeacher` | Safe heuristic policy |
| `SnakeSystemOneTrainer` | Collect + BC train + agreement / score eval |
| `SnakeSystemOnePolicy` | Encode + legal mask + play-time shields |

```csharp
using var head = SnakeSystemOneSpec.CreateHead(accelerator);
await SnakeSystemOneTrainer.TrainAsync(head); // defaults: 4096 samples, 80 epochs

var (action, answer, ms) = await SnakeSystemOnePolicy.DecideAsync(head, game);
game.SetAction(action);
```

**Play shields** (policy, not the MLP): refuse illegal moves; prefer safe food when hunger is high; avoid pure tail-orbit stalls. The teacher alone already clears a teacher score floor in tests; the clone is gated on held-out agreement and mean policy score.

---

## Advanced

### Architecture

```
state[StateDim]
  → Linear(StateDim, Hidden) + bias
  → ReLU
  → Linear(Hidden, NumOptions) + bias
  → Softmax  → Choice / Score / Noul views
```

No shared text encoder. State is **your** features. Keep encoding deterministic and documented (`SnakeSystemOneSpec` comments list the Snake layout).

### Building a non-Snake head

1. Define `StateDim` and option keys.
2. `new SystemOneDecisionHead(acc, stateDim, numOptions, hidden)`.
3. Collect `(state, label)` with a teacher, oracle, or human.
4. Train with `TrainStep` / epoch loss readback.
5. At inference, always pass an `allowedMask` if the environment can forbid options.
6. Export weights; version your cache key when the encoder changes.

### Training performance (browser)

- Reuse one head / `TrainableModel` — scratch buffers allocate in `Build`, not per step.
- Prefer `TrainStep` + `readLoss: false`; one `ReadLastLossAsync` per epoch.
- Flush every few batches (`SnakeSystemOneTrainer.FlushEveryBatches = 8`).
- Staging arrays for `CopyFromCPU` must be **exact length** (`batch * InputSize`); `SubView` + `Span` is not supported.

### Masking details

`SystemOneChoiceMask.Apply` zeros disallowed options, renormalizes, and if the raw argmax was illegal picks the best allowed survivor. If every option is false, behavior is defined in code (do not ship an all-false mask).

### Per-step stateful games (future)

Snake KV-style caches are position-addressed and idempotent. If you add a **shift-register** or recurrent state for another game, follow the three contracts in the library CLAUDE.md (stable bindings, snapshot around capture, veto prefix reuse). Snake today has no such cache.

### Out of scope (Phase 2+)

- Laya/Jev text encoder or weight import
- Multi-head dialogue / natural language questions
- RL fine-tune beyond behavioral cloning (possible on the same MLP; not shipped)

---

## Tests

```bash
# Fast lanes while iterating (desktop GPU + browser WebGPU)
PMT_FILTER=SystemOne PMT_LANES=Cuda,OpenCL,WebGPU PMT_PARALLEL=off `
  dotnet test PlaywrightMultiTest/PlaywrightMultiTest.csproj -c Release
```

Notable cases:

| Test | Asserts |
|------|---------|
| `SystemOne_SnakeTeacher_*` | Legality, score floor, no tail-orbit stall |
| `SystemOne_Choice*` / `ScoreAndNoul` / `Decide` | API + mask + latency |
| `SystemOne_Snake_BehavioralClone_AgreesWithTeacher` | BC agreement ≥ 85%, policy mean ≥ 12 |
| `SystemOne_Weights_RoundTrip_PreservesProbs` | Export/import bit-stable Softmax |

---

## Ownership / dispose

- Dispose the **head** (disposes its `TrainableModel`).
- Never dispose the **Accelerator** from library/demo helper code — the app owns it (same rule as `InferenceSession`).
