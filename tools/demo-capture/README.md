# Demo capture tools

Scripts that drive a real Chrome (own profile, own CDP port, killed by PID) against a locally served demo (`dotnet run -c Release --no-build --no-launch-profile --urls http://localhost:5088`)
to make the README media. Paths and ports are arguments; none of this ships.

- `t2i-candidates.mjs <profile> <outDir> '<json [[prompt, seed], ...]>'` draws pictures with SD Turbo in the text-to-3d page and saves the one the page chose for each, so a good subject can be picked before the slow 3D stage.
  It also logs the page console and WebGPU `device.lost` reasons.
- `make-gif-frames.mjs <profile> <outDir> <prompt> <seed> <picture.png>` runs the real flow on the Image-to-3D page (generate, Rig this model, Animate) and captures frames at exact times:
  the viewer is frozen with `glbViewer.pose(canvas, animation, seconds, yaw)` (and `glbViewer.skeleton(canvas, true)` for the skeleton overlay), one viewport screenshot per frame, plus `rect.json` (the canvas rectangle).
- `tools/unirig-harness/reference/make_gif.py <framesDir> <out.gif> "<prompt>"` crops, captions and assembles the GIF (one shared palette).
