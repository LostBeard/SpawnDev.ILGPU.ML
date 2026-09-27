# SpawnDev.ILGPU.ML Documentation

Native GPU neural-network inference for .NET / Blazor WebAssembly, built on SpawnDev.ILGPU. Runs ONNX (and other) models as ILGPU compute kernels across all six backends (CUDA, OpenCL, CPU, WebGPU, WebGL, Wasm) — no ONNX Runtime dependency.

## Start here

| Doc | What it covers |
|-----|----------------|
| [**DEMO_AND_MODEL_STATUS.md**](DEMO_AND_MODEL_STATUS.md) | **Source of truth for what actually works.** Per-demo VERIFIED / PARTIAL / WIP status with the test that proves each one. Read this before trusting any "it works" claim elsewhere. |
| [**operators.md**](operators.md) | **Source of truth for ONNX op support.** Points at `OperatorRegistry.BuiltinOpTypes` — never hardcode counts or paste the op list elsewhere. |
| [**system-one.md**](system-one.md) | **System One decision heads** — package API (choice/score/noul, masks, weights) + Classic Snake **demo** reference (`SnakeSystemOneSpec` is not NuGet) |
| [`../README.md`](../README.md) | Project overview, quick start, API, supported backends |
| [`../CHANGELOG.md`](../CHANGELOG.md) | Release-accurate fixes per version (often the most candid record) |
| [`../Plans/`](../Plans/) | Engineering roadmaps and design notes |
| In-app `/getting-started` | Install + first-run walkthrough (runs in the demo app) |

## Documentation policy (so the docs stop drifting from the code)

This folder exists because marketing copy had drifted ahead of reality (e.g. Home.razor claimed 71 operators when the registry had ~194). The rules:

1. **Operator support has one code SSO and one human doc.** Code: `OperatorRegistry.BuiltinOpTypes`. Human: [operators.md](operators.md) (includes implementation tiers — registered ≠ real math). Count = `BuiltinOpTypes.Count` (Home page renders it live). Don't hardcode the count or paste the op list anywhere else.
2. **"Works / verified / live" requires evidence.** A demo is only ✅ VERIFIED in `DEMO_AND_MODEL_STATUS.md` when a passing end-to-end test is cited for it. Adding a page ≠ verifying it. Unverified or stubbed demos are marked 🚧 WIP — honestly.
3. **"N tests passing" must cite a PMT run**, not a remembered number. The canonical pass/fail is the latest `PlaywrightMultiTest` results JSON.
4. **Loaders ≠ inference.** "We can load format X" is not "we run model X end-to-end." The status doc separates the two.

## Planned topic docs (not written yet — listed so they aren't claimed prematurely)

- `architecture.md` — multi-format engine, graph compiler, executor, fixed-shape decode
- `backends.md` — per-backend support matrix, sync-vs-async rules (see SpawnDev.ILGPU `Docs/async.md`)
- `weight-loading.md` — streaming load, OPFS/disk caching, hub model delivery (HTTP; torrent optional)

If you add one of these, link it here and delete it from this "planned" list — same rule as the demo status.
