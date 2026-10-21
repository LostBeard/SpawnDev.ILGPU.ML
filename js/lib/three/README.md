# three.js (vendored)

three.js r170 (MIT, see LICENSE), from `npm pack three@0.170.0`: `build/three.module.min.js`, and from `examples/jsm` only `loaders/GLTFLoader.js`, `utils/BufferGeometryUtils.js`
and `environments/RoomEnvironment.js`. The only change: the bare `from 'three'` imports are rewritten to the relative path of `three.module.min.js` so the files load
with no import map (the demo, the WPF example's WebView2 and the render tools all load them the same way). `../glb-viewer.js` is the code of ours that uses it.
