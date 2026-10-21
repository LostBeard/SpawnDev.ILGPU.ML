// The demo's 3D preview: a GLB (bytes handed over by the page, they never leave JS) drawn by three.js (vendored in lib/three, MIT): PBR materials, image-based lighting
// from three's RoomEnvironment, ACES tone mapping, a shadow-casting key light, GPU skinning and glTF animation. The previous version of this file was a hand-written WebGL
// shader that drew the base colour only; the API below is the same, so the pages (and the WPF example's WebView2) did not change.
//
//   glbViewer.show(canvas, glbBytes)  -> Promise<{ vertices, triangles, animations: [names] }>      load and show (drag to orbit, wheel to zoom, slow auto-rotate until you drag)
//   glbViewer.play(canvas, which, inPlace)     play an animation by index or name (null / -1 = the bind pose); inPlace (default true) drops the hips' travel along the ground
//   glbViewer.pose(canvas, which, seconds, yaw, inPlace, dist)    freeze at one pose for capturing frames: animation `which` at `seconds`, the model turned `yaw` radians, the camera `dist` away (optional; default 1.8)
//   glbViewer.resume(canvas)  real-time playback and auto-rotation again        glbViewer.skeleton(canvas, on)  draw the bones        glbViewer.animations(canvas)  names
//   glbViewer.dispose(canvas)
(function () {
    "use strict";
    const BASE = document.currentScript ? document.currentScript.src : location.href;
    let lib = null;
    function load() {
        if (!lib) {
            const u = p => new URL("lib/three/" + p, BASE).href;
            lib = Promise.all([import(u("three.module.min.js")), import(u("addons/loaders/GLTFLoader.js")), import(u("addons/environments/RoomEnvironment.js"))])
                .then(([THREE, g, e]) => ({ THREE, GLTFLoader: g.GLTFLoader, RoomEnvironment: e.RoomEnvironment }));
        }
        return lib;
    }

    const viewers = new Map();      // canvas -> state of the model being shown
    const stages = new WeakMap();   // canvas -> { renderer, env }: kept for the canvas's life (a canvas keeps its WebGL context)

    function stageFor(T, canvas) {
        let st = stages.get(canvas);
        if (st) return st;
        const { THREE, RoomEnvironment } = T;
        const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: "high-performance" });
        renderer.toneMapping = THREE.ACESFilmicToneMapping;
        renderer.toneMappingExposure = 1.05;
        renderer.shadowMap.enabled = true;
        renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        const pmrem = new THREE.PMREMGenerator(renderer);
        const env = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
        pmrem.dispose();
        st = { renderer, env };
        stages.set(canvas, st);
        return st;
    }

    // The same clip with the root joint's travel along the ground removed (x and z held at the rest pose, the bob kept), as the previous viewer did.
    function inPlaceClip(T, clip, root, cache) {
        let c = cache.get(clip);
        if (c) return c;
        const tracks = clip.tracks.map(tr => {
            const dot = tr.name.lastIndexOf(".");
            if (tr.name.slice(dot + 1) !== "position") return tr;
            const node = root.getObjectByName(tr.name.slice(0, dot));
            if (!node || !node.userData.rest) return tr;
            const v = tr.values.slice(), r = node.userData.rest;
            for (let i = 0; i < v.length; i += 3) { v[i] = r.x; v[i + 2] = r.z; }
            return new T.THREE.VectorKeyframeTrack(tr.name, tr.times.slice(), v);
        });
        c = new T.THREE.AnimationClip(clip.name, clip.duration, tracks);
        cache.set(clip, c);
        return c;
    }

    async function show(canvas, glbBytes) {
        if (typeof canvas === "string") canvas = document.getElementById(canvas);
        dispose(canvas);
        const T = await load(), { THREE } = T;
        const bytes = glbBytes instanceof Uint8Array ? glbBytes : new Uint8Array(glbBytes);
        const gltf = await new T.GLTFLoader().parseAsync(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength), "");
        const { renderer, env } = stageFor(T, canvas);

        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x0f141f);
        scene.environment = env;

        const model = gltf.scene;
        let vertices = 0, triangles = 0;
        model.traverse(o => {
            if (o.isBone || o.isObject3D) o.userData.rest = o.position.clone();                      // the rest translation, for in-place playback
            if (!o.isMesh) return;
            o.castShadow = true; o.receiveShadow = true; o.frustumCulled = false;                    // skinned meshes move outside their bind-pose bounds
            const g = o.geometry; vertices += g.attributes.position.count; triangles += (g.index ? g.index.count : g.attributes.position.count) / 3;
        });
        // centre on the origin and scale to a unit box, so the camera distance means the same for every model
        const box = new THREE.Box3().setFromObject(model);
        const size = box.getSize(new THREE.Vector3()), centre = box.getCenter(new THREE.Vector3());
        const scale = 1 / (Math.max(size.x, size.y, size.z) || 1);
        const pivot = new THREE.Group();              // turned by yaw
        const holder = new THREE.Group();             // scaled and centred
        holder.scale.setScalar(scale);
        holder.position.copy(centre).multiplyScalar(-scale);
        holder.add(model); pivot.add(holder); scene.add(pivot);

        const floorY = -size.y * scale / 2;
        const floor = new THREE.Mesh(new THREE.PlaneGeometry(12, 12), new THREE.ShadowMaterial({ opacity: 0.38 }));
        floor.rotation.x = -Math.PI / 2; floor.position.y = floorY - 0.002; floor.receiveShadow = true; scene.add(floor);
        const key = new THREE.DirectionalLight(0xffffff, 2.3);
        key.position.set(1.4, 2.6, 1.8); key.castShadow = true;
        key.shadow.mapSize.set(2048, 2048); key.shadow.bias = -0.0004; key.shadow.normalBias = 0.004;
        Object.assign(key.shadow.camera, { left: -1.2, right: 1.2, top: 1.2, bottom: -1.2, near: 0.1, far: 8 });
        scene.add(key);
        const camera = new THREE.PerspectiveCamera(46, 1, 0.05, 60);

        let mixer = null; const clips = gltf.animations || [], cache = new Map();
        if (clips.length) mixer = new THREE.AnimationMixer(model);
        const state = { T, scene, camera, renderer, pivot, model, mixer, clips, cache, floor, key, raf: 0, last: performance.now(), yaw: 0.6, pitch: 0.35, dist: 1.8, auto: true, drag: null,
                        anim: -1, inPlace: true, fixedT: null, helper: null, w: 0, h: 0 };

        const onDown = e => { state.drag = { x: e.clientX, y: e.clientY }; state.auto = false; canvas.setPointerCapture(e.pointerId); };
        const onMove = e => { if (!state.drag) return; state.yaw += (e.clientX - state.drag.x) * 0.01; state.pitch = Math.max(-1.5, Math.min(1.5, state.pitch + (e.clientY - state.drag.y) * 0.01)); state.drag = { x: e.clientX, y: e.clientY }; };
        const onUp = () => { state.drag = null; };
        const onWheel = e => { e.preventDefault(); state.dist = Math.max(0.8, Math.min(8, state.dist * (e.deltaY > 0 ? 1.1 : 0.9))); };
        canvas.addEventListener("pointerdown", onDown); canvas.addEventListener("pointermove", onMove);
        canvas.addEventListener("pointerup", onUp); canvas.addEventListener("wheel", onWheel, { passive: false });
        state.cleanup = () => { canvas.removeEventListener("pointerdown", onDown); canvas.removeEventListener("pointermove", onMove); canvas.removeEventListener("pointerup", onUp); canvas.removeEventListener("wheel", onWheel); };

        const frame = () => {
            const now = performance.now(), dt = Math.min(0.1, (now - state.last) / 1000); state.last = now;
            const w = canvas.clientWidth | 0, h = canvas.clientHeight | 0;
            if (w > 0 && h > 0 && (w !== state.w || h !== state.h)) {
                state.w = w; state.h = h;
                renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2)); renderer.setSize(w, h, false);
                camera.aspect = w / h; camera.updateProjectionMatrix();
            }
            if (state.auto) state.yaw += 0.004;
            if (mixer) mixer.update(state.fixedT == null ? dt : 0);
            pivot.rotation.y = state.yaw;
            const cp = Math.cos(state.pitch), sp = Math.sin(state.pitch);
            camera.position.set(0, state.dist * sp, state.dist * cp); camera.lookAt(0, 0, 0);
            renderer.render(scene, camera);
            state.raf = requestAnimationFrame(frame);
        };
        state.raf = requestAnimationFrame(frame);
        viewers.set(canvas, state);
        return { vertices, triangles, animations: clips.map(c => c.name) };
    }

    function select(s, which, inPlace) {
        if (!s.mixer) return false;
        const i = typeof which === "string" ? s.clips.findIndex(a => a.name === which) : (which === null || which === undefined ? -1 : which);
        s.mixer.stopAllAction();
        if (i < 0 || i >= s.clips.length) { s.anim = -1; s.model.traverse(o => { if (o.isSkinnedMesh) o.skeleton.pose(); }); s.mixer.update(0); return false; }
        s.anim = i; s.inPlace = inPlace !== false;
        const clip = s.inPlace ? inPlaceClip(s.T, s.clips[i], s.model, s.cache) : s.clips[i];
        const action = s.mixer.clipAction(clip); action.reset().play();
        return true;
    }

    // Play one of the file's animations (by index or name; null/-1 = the bind pose). inPlace (default true) drops the hips' travel along the ground so the character stays in frame.
    function play(canvas, which, inPlace) {
        if (typeof canvas === "string") canvas = document.getElementById(canvas);
        const s = viewers.get(canvas); if (!s || !s.mixer) return false;
        s.fixedT = null;
        return select(s, which, inPlace);
    }
    // Freeze the viewer at one pose, for capturing frames at exact times: animation `which` (index, name or -1 for the bind pose) at `seconds`, the model turned `yaw` radians.
    // `resume(canvas)` goes back to real-time playback and auto-rotation.
    function pose(canvas, which, seconds, yaw, inPlace, dist) {
        if (typeof canvas === "string") canvas = document.getElementById(canvas);
        const s = viewers.get(canvas); if (!s) return false;
        s.auto = false; s.yaw = yaw; s.fixedT = seconds; if (dist > 0) s.dist = dist;
        if (select(s, which, inPlace)) { const a = s.mixer.clipAction(s.inPlace ? inPlaceClip(s.T, s.clips[s.anim], s.model, s.cache) : s.clips[s.anim]); a.time = seconds; a.paused = true; s.mixer.update(0); }
        return true;
    }
    function resume(canvas) {
        if (typeof canvas === "string") canvas = document.getElementById(canvas);
        const s = viewers.get(canvas); if (!s) return false;
        s.fixedT = null; s.auto = true; s.last = performance.now();
        if (s.anim >= 0) select(s, s.anim, s.inPlace);
        return true;
    }
    function skeleton(canvas, on) {
        if (typeof canvas === "string") canvas = document.getElementById(canvas);
        const s = viewers.get(canvas); if (!s) return false;
        if (on && !s.helper) { s.helper = new s.T.THREE.SkeletonHelper(s.model); s.helper.material.depthTest = false; s.helper.material.linewidth = 2; s.helper.renderOrder = 10; s.scene.add(s.helper); }
        if (!on && s.helper) { s.scene.remove(s.helper); s.helper.dispose(); s.helper = null; }
        return true;
    }
    function animations(canvas) {
        if (typeof canvas === "string") canvas = document.getElementById(canvas);
        const s = viewers.get(canvas); return s && s.clips ? s.clips.map(a => a.name) : [];
    }

    function dispose(canvas) {
        if (typeof canvas === "string") canvas = document.getElementById(canvas);
        if (!canvas) return;
        const s = viewers.get(canvas);
        if (!s) return;
        cancelAnimationFrame(s.raf); s.cleanup && s.cleanup();
        if (s.helper) s.helper.dispose();
        s.mixer && s.mixer.stopAllAction();
        s.scene.traverse(o => {
            if (o.geometry) o.geometry.dispose();
            const ms = o.material ? (Array.isArray(o.material) ? o.material : [o.material]) : [];
            for (const m of ms) { for (const k in m) if (m[k] && m[k].isTexture) m[k].dispose(); m.dispose(); }
        });
        s.key.shadow.map && s.key.shadow.map.dispose();
        viewers.delete(canvas);      // (the renderer stays: the same canvas gets its WebGL context back on the next show)
    }

    window.glbViewer = { show, dispose, play, animations, pose, resume, skeleton };
})();
