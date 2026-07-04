# Wasm backend: multi-GB models via fp16-resident weights + OPFS weight paging

**Status: TRACKED GAP** (2026-07-03). `SDTurbo_Generate_E2E` skips the Wasm lane until one of these
lands. Captain's framing: "there is no reason why you can't load multi-GB data - you just don't
load it all into memory at once."

## The problem

The Wasm backend's "device memory" is WASM-heap SharedArrayBuffer, bounded by the wasm32 4GB
address space (and the app's Emscripten heap cap below that). Weights are expanded to fp32 on
load, so SD-Turbo's 2.5GB of fp16 needs ~5GB resident - the .NET runtime exits with code 1
mid-load (observed 2026-07-03, stream loading active; the stream is fine, the destination is
the ceiling).

## Fix (a): fp16-resident weights - the cheaper half

Store initializers at their shipped precision on the Wasm backend (the mixed-precision activation
work already established fp16 storage + convert-around-node in the executor). 2.5GB fp16 stays
2.5GB; with the heap cap raised toward 4GB (`EmccMaximumHeapSize`) SD-Turbo-class models fit.
Bounded win: models beyond ~3.5GB still cannot fit.

## Fix (b): OPFS-backed per-layer weight paging - the general answer

Weights live in OPFS (they are already streamed through it by the hub); the GraphExecutor pages a
layer's weights into the heap right before its nodes execute and releases them after, bounding
residency to the largest layer + activations regardless of model size. Notes:
- The executor already has a deferred-release machinery (`MaxPendingReleaseBytes`) and per-node
  execution ordering - the paging hook sits where weights are resolved per node.
- Layer-granularity read from OPFS per forward = the cost; the Wasm lane is the correctness/
  fallback lane, not the speed lane, so paging overhead is acceptable there.
- Same machinery would serve any future memory-constrained target.

## Order

(a) first - it unlocks SD-Turbo on the lane with far less surface; (b) when a model actually
exceeds it or when the paging hook is wanted for another backend.
