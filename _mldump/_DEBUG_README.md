# SpawnDev.ILGPU Debug Output Folder

This folder receives auto-dumped shaders and Wasm binaries from SpawnDev.ILGPU
every time a kernel is compiled. Set via the 'Set Debug Folder' button on the /tests page.

## Folder Structure

```
debugfolder/
├── _DEBUG_README.md    (this file)
├── wgsl/               (WebGPU WGSL shaders)
│   ├── 000_KernelName.wgsl
│   └── ...
├── glsl/               (WebGL GLSL shaders)
│   ├── 000_KernelName.glsl
│   └── ...
└── wasm/               (Wasm backend binaries + info)
    ├── 000_KernelName.wasm
    ├── 000_KernelName.txt
    └── ...
```

## Files Written Automatically

- `wgsl/NNN_KernelName.wgsl` — WGSL shaders (WebGPU) with metadata headers
- `glsl/NNN_KernelName.glsl` — GLSL shaders (WebGL) with metadata headers
- `wasm/NNN_KernelName.wasm` — Wasm binary (disassemble: `wasm2wat --enable-threads file.wasm`)
- `wasm/NNN_KernelName.txt` — Wasm kernel compilation info (params, locals, shared mem, etc.)

## For AI Agents Debugging

If you are a Claude, Gemini, or other AI agent examining this folder:

1. **WGSL files** (`wgsl/`) contain the generated GPU shader code. Look for the `@workgroup_size`
   annotation and the kernel entry point (`@compute @workgroup_size(X) fn main()`). Check for:
   - Correct shared memory declarations (`var<workgroup>`)
   - Barrier placement (`workgroupBarrier()`)
   - PHI variable merge blocks (variables set in multiple branches)
   - Loop structure (loop/continuing/break_if patterns)

2. **GLSL files** (`glsl/`) contain WebGL compute shaders. Check for:
   - Transform feedback output declarations
   - Uniform/varying bindings
   - Precision qualifiers

3. **Wasm files** (`wasm/`) are WebAssembly binaries. Disassemble with:
   ```
   wasm2wat --enable-threads kernel.wasm > kernel.wat
   ```
   Look for:
   - Function signatures (params, locals)
   - `memory.atomic.wait32` / `memory.atomic.notify` for barriers
   - Block/loop/br_table structure for the state machine
   - Shared memory access patterns

## Persistence

The folder handle is saved in IndexedDB. On next browser visit, click 'allow' when
prompted for filesystem permission and the folder is automatically reconnected.
