# SpawnDev.ILGPU.ML Documentation

Native GPU neural-network inference for .NET / Blazor WebAssembly, built on SpawnDev.ILGPU. Runs ONNX (and other) models as ILGPU compute kernels across all six backends (CUDA, OpenCL, CPU, WebGPU, WebGL, Wasm) — no ONNX Runtime dependency.

## Start here

| Doc | What it covers |
|-----|----------------|
| [**DEMO_AND_MODEL_STATUS.md**](DEMO_AND_MODEL_STATUS.md) | **Source of truth for what actually works.** Per-demo VERIFIED / PARTIAL / WIP status with the test that proves each one. Read this before trusting any "it works" claim elsewhere. |
| [`../README.md`](../README.md) | Project overview, quick start, API, supported backends |
| [`../CHANGELOG.md`](../CHANGELOG.md) | Release-accurate fixes per version (often the most candid record) |
| [`../Plans/`](../Plans/) | Engineering roadmaps and design notes |
| In-app `/getting-started` | Install + first-run walkthrough (runs in the demo app) |

## Documentation policy (so the docs stop drifting from the code)

This folder exists because marketing copy had drifted ahead of reality (e.g. Home.razor claimed 71 operators when the registry has ~194). The rules:

1. **Counts come from code, not memory.** Operator count = `OperatorRegistry.BuiltinOpTypes.Count` (rendered live on the Home page; it's the registry's documented single source of truth). Don't hardcode it anywhere.
2. **"Works / verified / live" requires evidence.** A demo is only ✅ VERIFIED in `DEMO_AND_MODEL_STATUS.md` when a passing end-to-end test is cited for it. Adding a page ≠ verifying it. Unverified or stubbed demos are marked 🚧 WIP — honestly.
3. **"N tests passing" must cite a PMT run**, not a remembered number. The canonical pass/fail is the latest `PlaywrightMultiTest` results JSON.
4. **Loaders ≠ inference.** "We can load format X" is not "we run model X end-to-end." The status doc separates the two.

## Planned topic docs (not written yet — listed so they aren't claimed prematurely)

- `architecture.md` — multi-format engine, graph compiler, executor, fixed-shape decode
- `operators.md` — registered vs. reference-tested vs. pass-through ops; the `BuiltinOpTypes` policy
- `backends.md` — per-backend support matrix, sync-vs-async rules (see SpawnDev.ILGPU `Docs/async.md`)
- `weight-loading.md` — streaming load, OPFS caching, hub/torrent model delivery

If you add one of these, link it here and delete it from this "planned" list — same rule as the demo status.
