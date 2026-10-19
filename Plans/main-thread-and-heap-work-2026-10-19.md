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
