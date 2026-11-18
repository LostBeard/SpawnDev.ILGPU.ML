# Workarounds for bugs outside this library

SpawnDev.ILGPU.ML does not work around its own bugs or those of the other SpawnDev libraries: those get fixed at the source. A workaround is only allowed where the
real fix is in code we do not own, and then it has to be visible. Every one is listed here, marked `// WORKAROUND:` in the source with a link to its public
tracking issue, and carries a test that reports whether it is still needed.

| # | What is wrong, and where | Since | The workaround | Tracking | How we know when to remove it |
|---|--------------------------|-------|----------------|----------|-------------------------------|
| 1 | **`Vector128.Narrow` saturates in Blazor WebAssembly once its loop is hot** (.NET 10 browser runtime). It must truncate, and does on desktop and while the code is cold. | 5.8.21 | `Ltx2Fp8Rounding.Low8` masks each word to its low byte before narrowing, so truncating and saturating give the same byte. | [Issue #3](https://github.com/LostBeard/SpawnDev.ILGPU.ML/issues/3); report to dotnet/runtime not yet filed | `Ltx2_Fp8CompatStream_MatchesPlainFormAndUploads` prints `unmasked output mix, 300000 hot rounds of 16 words: equals the plain mix` on the WebGPU, WebGL and Wasm lanes |

## 1. `Vector128.Narrow` saturates in Blazor WebAssembly once the loop is hot

**Found** 2026-11-17, the first time the LTX-2 kernel tests ran on the browser lanes. **Approved** by the maintainer 2026-11-18. **Shipped** in 5.8.21.

**What we measured** (.NET SDK 10.0.403, an interpreted, trimmed Blazor WebAssembly publish, Chrome; all three browser lanes share this runtime):

- A loop that takes the low byte of 32-bit words with `Vector128.Narrow` (32 to 16 bits, then 16 to 8) returns the right bytes for its first 342 rounds.
- From round 342 on, 4,756,832 of the remaining bytes are `FF` (300,000 rounds of 16 words): every word above 255 comes back saturated.
- The same code on desktop .NET 10 is right for all 300,000 rounds.
- Run once, each narrowing is right in the browser too (32 to 16 bits, 16 to 8 bits, and the nested form). Three one-shot probes said "truncates" while the hot
  loop was wrong: a probe of this has to run hot.

Right while cold and wrong once hot suggests the runtime's hot-path compiler emits WebAssembly's saturating narrow. That is our reading of the measurements, not
something we confirmed in the runtime's source.

**Where it bit us.** `Ltx2Fp8Rounding.Low8`: the vectorized MT19937 byte stream of the opt-in ComfyUI-compatible LoRA mode
(`Ltx2Transformer.ComfyCompatibleFp8Lora`). In the browser it returned `FF` for nearly every draw. The test that compares it with the plain generator
(`Low8Scalar`) caught it; a test that used the vectorized stream on both sides of its comparison did not.

**The workaround** (`SpawnDev.ILGPU.ML/Ltx2/Ltx2Fp8Rounding.cs`):

```csharp
v0 &= low8; v1 &= low8;      // every word is now 0..255: truncating and saturating give the same byte
Vector128.Narrow(Vector128.Narrow(v0.GetLower(), v0.GetUpper()), Vector128.Narrow(v1.GetLower(), v1.GetUpper()))...
```

Two AND operations per 16 draws. The vectorized stream equals the plain one word for word on all seven test lanes.

**If you write SIMD code that runs in Blazor WebAssembly:** mask to the narrow type's range before `Vector128.Narrow`, and test the result against a scalar
form in a loop that runs hundreds of thousands of times. The issue has a snippet you can paste into an app to check your own runtime.

**Still to do:** a standalone repro project and the report to dotnet/runtime (both will be linked from the issue), then removal of the mask once the canary reads
`equals the plain mix` in the browser.
