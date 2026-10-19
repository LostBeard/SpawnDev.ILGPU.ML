# Main-thread C# and managed-heap work (TJ, 2026-10-19)

Rules (TJ): model weights never enter the .NET heap in the browser (JS -> GPU streams only); no C# processing that takes more than a few ms on the main thread (the GPU does the work); do not cut output quality to gain speed before the rest of the pipeline is tight.
Audits (static reads, estimates, marked INFERRED where not read): [weight loading](audit-weight-loading-2026-10-19.md), [CPU work and copies](audit-cpu-and-copies-2026-10-19.md).
Measured so far (browser, WebGPU, real hosted weights): FLUX 3 text encoder 2157 MiB and transformer 3311 MiB go JS -> GPU in 2.2 s and 3.3 s with 0.00 MiB through the .NET heap (`Flux3_Browser_WeightsStayOutOfTheManagedHeap`).

## Order of work
1. Weights: V5 ZipVoice vocoder + V6 YOLOv8n/BlazeFace (HttpClient byte[]), V10 inspector local file, V12 lazy 16 MiB scratch, V13 duplicates (Whisper decoders, VAE encoder per reference, pages' own ModelHub), V7 ILGPU 4-byte gate (aligned prefix), V1 GGUF hydrate, V2/V3 ONNX loaders, V8 depth, V14 a census assertion per model.
2. FLUX host loops: FLUX.2 prompt embeds read back and re-uploaded per step, noise, ToVaeInput, tiled VAE blend (FLUX 3 per-step traffic is done).
3. TRELLIS.2 mesh chain on the GPU (remesh topology + flag read-back, simplify, repair, UV atlas, texture readbacks, PNG/GLB); until then yield points + peak-memory trimming.
4. Image pages: Remove Background and the pages that read a photo into an int[] (`data.Read<int>()`), GPU compositing and blur.
5. GGUF: tokenizer built twice, top-p sampler on the CPU, last-position logits (done for the caption writer and chat).

## Status (2026-10-19, evening)
Done on main, not released: item 3 mesh chain on the device: remesh (`DeviceRemesher`), simplify (`GpuMeshSimplifier`, five matching passes per round: crown 4.67 M -> 298k faces in 31 rounds), winding + vertex normals + UV atlas (`DeviceUvAtlas`), bake from the atlas buffers. The host repair of the simplified mesh is off by default. Browser (interpreted dev build, WebGPU), crown: 235 s -> 187 s; winding + normals + atlas 17.0 s -> 5.7 s. Item 4 (this page): the dropped picture goes from its `Uint8ClampedArray` to the device (`RgbaUpload.FromTypedArray`), the opaque test and the border-touch measure are kernels (`RgbaStats`), the cut-out applies the RMBG mask on the device (`BackgroundRemovalPipeline.ApplyMaskAsync` / `RemoveBackgroundToDeviceAsync`: range check, sigmoid, bilinear resize, alpha), and `Image3DPage` hands device buffers to TRELLIS.2; no pixel loop remains on the page.
Found on the way: ILGPU 5.3.8 loop unrolling lost the result of a loop inside an `if` (fixed in the fork, 3721a9c, unreleased; see CHANGELOG there). Old ML kernels with a loop inside an if and a value used after it may have been silently wrong: audit once the fixed package is in the local feed.
Still host-side in the TRELLIS.2 tail: PNG encode and GLB write (PngEncoder / PbrGlbWriter on arrays read back from the device), `RemoveBgPage` and the other image pages' compositing, FLUX.2 host loops (item 2), GGUF tokenizer built twice and the CPU top-p sampler (item 5), weight-loading items V1-V3, V7, V8, V13, V14 (item 1).
