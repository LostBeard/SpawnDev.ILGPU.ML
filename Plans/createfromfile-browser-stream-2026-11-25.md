# CreateFromFileAsync in the browser should not hold the model in .NET (Bones, 2026-11-25)

Rule 4 (device data stays on the device; make the wrong thing unreachable): `InferenceSession.CreateFromFileAsync(accelerator, http, url)` is the call a newcomer finds first, and in the browser it
downloads the whole model into a `byte[]`.

MEASURED by Bones (published trimmed Blazor app, ML 5.8.31, MobileNetV2 14.2 MB, GC.GetTotalAllocatedBytes around session creation, WebGL on SwiftShader): `CreateFromFileAsync` 96.7 and 81.8 MB
allocated (about 6x the model); `ModelHub.OpenStreamFromUrlAsync` + `CreateFromStreamAsync` 2.9 MB, both runs. Allocated bytes, not peak heap.

Proposal (not built, after the model residency work):
1. In the browser, for ONNX and GGUF: `ModelsHost.ProbeSizeAsync`, then `CreateFromStreamAsync` over a `FetchRangeStream` (JS-side, stores nothing). Not OPFS: writing the model to the visitor's
   disk is a side effect the simple call should not have; `ModelHub.OpenStreamFromUrlAsync` stays the cached way.
2. Today's path when the host refuses ranges (200 to a Range) or gives no size, and for the formats that drain into `byte[]` anyway (TFLite, SafeTensors, PyTorch, CoreML, TF GraphDef).
3. Costs to measure first: a ranged load makes many requests (16 MiB read-ahead) where the plain one makes one; a compressed host loses its compression (ranges ask for identity), so a 13.1 MB
   download becomes 14.2 MB. Compare the wall time on GitHub Pages and models.spawndev.com before switching.
4. `CreateFromOnnxAsync` / `CreateFromTFLiteAsync` / the GGUF URL overloads use the same `DownloadBytesChunkedAsync`: same decision for the two stream-capable formats.

Related: `streamThreshold` (CreateFromStreamAsync, CreateFromOnnxStreamAsync, OnnxParser.ParseFromStreamAsync) has done nothing since July 2026 (OnnxParser.Stream.cs discards it); a parameter cannot be
marked [Obsolete], so its doc says so. Remove it at the next breaking release.
