# WebGPU: plain access to an atomic buffer through a helper function's pointer is invalid WGSL

Found 2026-10-19 with the device remesher's union-find (`Trellis2/DeviceRemesher.cs`, `Find` / `Link`).

A kernel that uses `Atomic.CompareExchange(ref parent[a], ...)` makes `parent` an `array<atomic<i32>>` in the shader. A plain `parent[x]` read inside a HELPER method (`static int Find(IV parent, int x)`) is emitted as `v = *ptr;`, which Chrome rejects:

```
Error while parsing WGSL: cannot assign 'atomic<i32>' to 'i32'
            v_48 = *v_215;
```

`WGSLCodeGenerator.GenerateCode(Load)` / `GenerateCode(Store)` choose `atomicLoad` / `atomicStore` from `IsAtomicPointer`, which looks at the pointer's element type; a helper's pointer parameter is typed `ptr<storage, i32>` at that point, so the check misses buffers that are atomic only because another site in the entry kernel is. (The same plain access written directly in the kernel body is handled.)

**Workaround in use:** every access to such a buffer in the helpers is an explicit atomic op (`Atomic.Add(ref x, 0)` for the load, `Atomic.CompareExchange` for the store). **Real fix (SpawnDev.ILGPU, WGSL emitter):** carry the "this buffer is atomic" fact on the helper's pointer parameter (or declare helper pointer params with the atomic element type) so plain loads/stores through it become `atomicLoad` / `atomicStore`. Add a regression test with a helper that reads and writes an atomic buffer plainly.
