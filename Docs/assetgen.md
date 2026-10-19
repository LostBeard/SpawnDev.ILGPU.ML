# Asset generation on the desktop (`AssetGenerator`, examples 07 and 08)

The browser demo's text-to-character chain as a library class, a command-line tool and a WPF app, on CUDA or OpenCL (WebGPU too). Nothing leaves your machine except the one-time model downloads.

```
prompt --FLUX.2 klein--> pictures --you pick / best fit--> RMBG-1.4 cut-out --TRELLIS.2--> textured PBR GLB --SkinTokens--> rigged GLB --Kimodo clips--> animated GLB
```

## The class

```csharp
using var gen = new AssetGenerator(accelerator, source, motionLoader) { ImageModelBits = 4, BridgeHops = 4 };
var pictures = await gen.DrawAsync("a knight in plate armor, full body", new[] { 11, 12, 13, 14 });          // FLUX.2 klein, 512 px; Size = 1024 works too
var (cutOut, touch) = await gen.CutOutAsync(pictures[0]);                                                      // touch: how much of the frame's edge the subject covers
var glb = await gen.BuildModelAsync(cutOut, pictures[0].Width, pictures[0].Height);                          // TRELLIS.2
var rigged = await gen.RigAsync(glb);                                                                         // SkinTokens + RigCleanup
var animated = await gen.AnimateAsync(rigged, new[] { "walk", "dance", "wave" }, armClearanceDegrees: 8);    // Kimodo clips
// or everything at once:
var asset = await gen.GenerateAsync("a knight", seed: 11, pictures: 4, moves: motionIds);
```

`source` is any `IModelSource`: on a desktop `new HttpClientModelSource(http, new FileModelStore(folder))` (resumable downloads, cached on disk), in a browser the OPFS source. `AssetGenerator.MotionsFromDirectory(dir)` loads the bundled `*.bin` Kimodo clips
(the examples copy them from the demo's `wwwroot/motions`, see `NOTICE.md` there). `KeepImageModelResident = true` keeps FLUX.2's transformer on the GPU between draws (about 2 s a picture after the first on a desktop GPU); it is released before the 3D stage. `ImageModelBits = 8` selects the near-lossless FLUX.2 weights.
`ImageModel.SdTurbo` draws with SD-Turbo instead.

## The console tool (`Examples/07.AssetGen.Console`)

```
AssetGen.Console "<prompt>" [--out name] [--seed N] [--pictures N] [--pick] [--size px] [--image-model flux2|sd] [--quality 4|8]
                           [--faces N] [--texture N] [--no-rig] [--no-animate] [--moves a,b,c] [--arm-clearance deg] [--bridge-hops N] [--cache dir] [--ci]
```

It draws `--pictures` seeds (default 2), keeps the one whose cut-out fits its frame best (or asks, with `--pick`), and writes `<name>.png`, `<name>.glb`, `<name>_rigged.glb` and `<name>_animated.glb`. Per-stage times print as it goes.
Measured on an RTX 4070 (CUDA), models cached: a knight in 77 s end to end (draw 2.8 s a picture, cut-out 0.8 s, TRELLIS.2 ~45 s, rig 6 s). First run, including 17 GB of downloads: 479 s.

## The WPF app (`Examples/08.AssetGen.Wpf`)

A prompt box, an image-model / pictures / seed / size / FLUX-precision row, and a thumbnail grid: pictures appear as they finish, **Make 3D from this** builds the one you pick, then **Rig this model**, **Animate** (all 22 moves, an arm-clearance setting) and **Save .glb**.
The preview is the browser demo's own WebGL viewer (GPU skinning, glTF animation) hosted in WebView2, so what you see is what the .glb does in any engine. `ASSETGEN_CACHE` (or the Model folder button) puts the 17 GB of models on a big drive.
