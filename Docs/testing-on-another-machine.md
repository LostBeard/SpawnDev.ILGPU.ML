# Running the ML tests on another machine (a tester session)

For a person or a Claude Code session on a different PC (another GPU, another driver, another OS version) who runs `SpawnDev.ILGPU.ML`'s tests and reports what differs. The tests are the only
way to run the library's checks: `PlaywrightMultiTest` (PMT) discovers every project, publishes the Blazor demo, and runs each test on every backend ("lane") it is scoped to.

## Prerequisites

- .NET 10 SDK, git, a clone of the repo (`LostBeard/SpawnDev.ILGPU.ML`) and the packages it restores (SpawnDev.ILGPU 5.4.2 or later from nuget.org).
- A recent Chrome for the browser lanes (PMT drives Playwright's Chromium; the first run downloads it). Browser lanes need a GPU the browser can use for WebGPU.
- Disk: the heavy real-model tests download GBs of weights into the model cache (`FileModelStore.Default()` on the desktop, OPFS in the browser lanes). A laptop with an integrated GPU should not run them.

## The one command

```bash
PMT_LANES=opencl,cpu,velocity PMT_FILTER=Operator dotnet test PlaywrightMultiTest/PlaywrightMultiTest.csproj -c Release
```

`dotnet test --filter` only limits what is REPORTED (every phase still runs), so scope with the environment variables:

| Variable | Meaning |
|----------|---------|
| `PMT_LANES` | comma list of `cuda`, `opencl`, `webgpu`, `webgl`, `wasm`, `cpu`, `velocity`. `cpu` is the plain CPU accelerator, `velocity` the SIMD CPU backend (the default on 64-bit desktop). Pick the lanes the machine has: an Intel iGPU is `opencl` (if its OpenCL runtime is installed) plus `webgpu`/`webgl`/`wasm` in the browser plus the CPU ones |
| `PMT_FILTER` | substring of the test name; comma = OR. List first (`PMT_FILTER=Foo` with a wrong name runs nothing and says so) |
| `PMT_EXCLUDE_CATEGORIES` | unset = the default gate (the heavy model tests are left out). Set to the EMPTY string (`PMT_EXCLUDE_CATEGORIES=`, a bash prefix) to include the `HeavyModel` tests: they stream 2 to 15 GB of weights and need 8 GB or more of GPU memory |
| `PMT_CONSOLE_LOG` | keep browser console lines containing this text (a browser lane otherwise reports only `Console: N error(s)`; numbers printed by browser tests only appear with this) |
| `PMT_CONSOLE_TIMEOUT_MS` | the outer cap of a heavy desktop test (default 10 minutes) |

Run ONE PMT process at a time (two share the ports, the browser profile and the GPU, and the Wasm lane's timings break under any other CPU load). Do not build during a run. A long run goes in the
background with its output in a file; kill a process by its PID, never by image name.

## What a useful report holds

- The machine: GPU and driver version (`Device.PrintInformation`/the lane's own log lines print them), OS, RAM, Chrome version.
- Counts per lane and the NAMES of failing tests with the first lines of the error (the end-of-run listing prints them; the console shows them only at the end).
- For a failure: whether it is the same on another lane of the same machine (then it is the library or the test), or only on this GPU (then it is a driver, a limit such as a small shared-memory size or
  `maxStorageBufferBindingSize`, or a missing extension: say which limit the error names).
- Timings only with the lane, the size and whether the run was cold or warm.
- Put the evidence in a file (`_DevComms/YYYY-MM-DD-HHMM-<from>-TO-<to>-<subject>.md`, written under a temp name and renamed) and send a short message with the path, as the project's rules say.

## Known light-GPU facts

- WebGL and Wasm are the slow lanes; their failures matter after the fast ones (CUDA/OpenCL and WebGPU) are green.
- A GPU without fp16 or with a small workgroup memory limit skips or fails the tiled GEMM kernels with an error that names the limit: report it, do not work around it.
- The 3D / video / large image generation tests (`Trellis2_`, `Flux`, `Wan_`) are HeavyModel: leave them out on a laptop.
