# Compiler warning audit - 2026-10-30 (Tuvok)

A clean Release build of SpawnDev.ILGPU.ML (on SpawnDev.ILGPU 5.4.12) prints 3534 warnings. Most are documentation debt
(CS1591 missing XML comments, CS1573 missing param tags, CA1416 browser-only API on desktop TFMs). The kinds that can hide a
defect were read site by site. Verdicts:

## Fixed
- [x] CS4014 `Preprocessing/MediaStreamCapture.cs` StartWebcamAsync: `_video.Play()` (a JS promise) was not awaited and a
      fixed 100 ms delay stood in for it: a rejected play (autoplay policy, device lost) was an unobserved exception and capture
      started on a dead video. Now awaited (a rejection returns false through the existing catch). `StartFromStream` (sync)
      observes the promise and stops the capture on rejection.
- [x] CS8602 `Graph/GraphOptimizer.cs` FuseGroupNorm: `inorm.Attributes.TryGetValue("epsilon")` unguarded, and
      `GraphNode.Attributes` is null for nodes the TFLite loader builds: NullReferenceException out of Optimize = model load
      failure. Test `Fusion_GroupNorm_InstanceNormWithoutAttributes_UsesDefaultEpsilon` (red on the old line, CPU lane).
- [x] CS1570 malformed XML: `Hub/ModelHub.cs` (unclosed para, remarks closed with /summary), `Pipelines/IlgpuZipVoiceGraphs.cs` (unclosed para).
- [x] CS1574 stale crefs to `SdControlNet.RunAsync` (the `into` parameter added in a58ece28): SdControlNet.cs, SdUNet.cs.
- [x] CS0219 / CS0169 dead locals / fields: `ElementWiseKernels.cs` CenterCropPad `val`, `Trellis2/GpuWeights.cs` `tAlloc`, `SdKernels._noBias`, and CS8618 `Kernels/MarchingCubesKernel.cs` `_accelerator` (never assigned, never read: the class's work is a static method).

## Read, not defects (intentional or flow-analysis false positives)
- CS1718 `ElementWiseKernels.cs` 1172 / 2575, `Kernels/TensorStatsKernel.cs` 71 / 137: `v != v` is the NaN test inside kernels (IEEE; intended).
- CS8601 `Graph/GraphExecutor.cs` 3990: an ONNX optional input is null by design (operators check); 4976: `decodeAttrs ?? node.Attributes`, both set by every ONNX path.
- CS8602 `Graph/GraphOptimizer.cs` 1645: null-checked in the same condition (the compiler does not track an indexer element).
- CS8602 `InferenceSession.cs` 2759: `Layers[i].Bias` null-checked one line earlier (indexer element again).
- CS8602 `Pipelines/ImageGenerator.cs` 138 / 145 / 162: `_flux3` / `_flux2` assigned by the `LoadAsync(engine)` awaited just before.
- CS8602 `SoftmaxKernel.cs` 131, `Training/TrainingKernels.cs` 88, `UniRig/UniRigEncoder.cs` 99, `Trellis2/MeshClusterer.cs` 246-308,
  `Pipelines/IlgpuZipVoiceGraphs.cs` 259: workspace buffers / kernels assigned by `??=`, an Ensure* call or `Load()` right before use.

## Open
- [ ] CS9113 (54), CS0419 (56), CS1734 (38), CS1572 / CS1735 / CS1587: unread parameters and doc-reference noise - read in a second pass.
- [ ] Documentation debt (CS1591 x ~2800 unique): not defects; document the public API surface over time.
