// Minimal GLB viewer for the Text/Image to 3D demo: WebGL2, one textured mesh (POSITION, NORMAL, TEXCOORD_0, uint32 indices,
// baseColorTexture), orbit with the mouse (drag to rotate, wheel to zoom), slow auto-rotation until the first drag.
// A rigged GLB (JOINTS_0 / WEIGHTS_0, a skin, glTF animations) is skinned on the GPU: joint matrices go to the shader through a float texture,
// glbViewer.animations(canvas) lists the clips and glbViewer.play(canvas, indexOrName, inPlace) plays one (in place by default: the hips' travel is dropped so a walk stays in frame).
// Same-origin script (the demo is cross-origin isolated, so a CDN viewer would need CORP headers). No dependencies.
(function () {
    "use strict";
    const viewers = new WeakMap();

    function parseGlb(bytes) {
        const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
        if (dv.getUint32(0, true) !== 0x46546C67) throw new Error("not a GLB");
        const jsonLen = dv.getUint32(12, true);
        const json = JSON.parse(new TextDecoder().decode(bytes.subarray(20, 20 + jsonLen)));
        const binStart = 20 + jsonLen + 8;
        const bin = bytes.subarray(binStart);
        const view = (accIndex, Type, comps) => {
            const acc = json.accessors[accIndex], bv = json.bufferViews[acc.bufferView];
            const off = (bv.byteOffset || 0) + (acc.byteOffset || 0);
            const copy = bin.slice(off, off + acc.count * comps * Type.BYTES_PER_ELEMENT);
            return new Type(copy.buffer);
        };
        const prim = json.meshes[0].primitives[0];
        const mesh = {
            positions: view(prim.attributes.POSITION, Float32Array, 3),
            normals: prim.attributes.NORMAL !== undefined ? view(prim.attributes.NORMAL, Float32Array, 3) : null,
            uvs: prim.attributes.TEXCOORD_0 !== undefined ? view(prim.attributes.TEXCOORD_0, Float32Array, 2) : null,
            indices: view(prim.indices, Uint32Array, 1),
            image: null,
        };
        if (prim.attributes.JOINTS_0 !== undefined && prim.attributes.WEIGHTS_0 !== undefined && json.skins && json.skins.length) {
            const jAcc = json.accessors[prim.attributes.JOINTS_0];
            mesh.joints = view(prim.attributes.JOINTS_0, jAcc.componentType === 5121 ? Uint8Array : Uint16Array, 4);
            mesh.jointsType = jAcc.componentType === 5121 ? "u8" : "u16";
            mesh.weights = view(prim.attributes.WEIGHTS_0, Float32Array, 4);
            const sk = json.skins[0], J = sk.joints.length, index = new Map(sk.joints.map((n, i) => [n, i]));
            const parent = new Int32Array(J).fill(-1);
            sk.joints.forEach((n, i) => (json.nodes[n].children || []).forEach(c => { if (index.has(c)) parent[index.get(c)] = i; }));
            const restT = new Float32Array(J * 3);
            sk.joints.forEach((n, i) => { const t = json.nodes[n].translation || [0, 0, 0]; restT.set(t, i * 3); });
            mesh.skin = { J, parent, restT, ibm: view(sk.inverseBindMatrices, Float32Array, 16) };
            mesh.animations = (json.animations || []).map((a, k) => {
                const tracks = [];
                for (const ch of a.channels) {
                    const j = index.get(ch.target.node); if (j === undefined) continue;
                    const smp = a.samplers[ch.sampler];
                    const times = view(smp.input, Float32Array, 1), comps = ch.target.path === "rotation" ? 4 : 3;
                    tracks.push({ joint: j, path: ch.target.path, times, values: view(smp.output, Float32Array, comps), comps });
                }
                const duration = tracks.reduce((m, t) => Math.max(m, t.times[t.times.length - 1]), 0);
                return { name: a.name || ("animation " + k), tracks, duration };
            });
        }
        const mat = json.materials && json.materials[prim.material || 0];
        const texInfo = mat && mat.pbrMetallicRoughness && mat.pbrMetallicRoughness.baseColorTexture;
        if (texInfo) {
            const img = json.images[json.textures[texInfo.index].source], bv = json.bufferViews[img.bufferView];
            mesh.image = new Blob([bin.slice(bv.byteOffset || 0, (bv.byteOffset || 0) + bv.byteLength)], { type: img.mimeType || "image/png" });
        }
        return mesh;
    }

    const VS = `#version 300 es
        in vec3 aPos; in vec3 aNrm; in vec2 aUv; in vec4 aJoints; in vec4 aWeights;
        uniform mat4 uMvp; uniform mat4 uModel; uniform bool uSkinned; uniform highp sampler2D uJointTex;
        out vec3 vNrm; out vec2 vUv;
        mat4 jointMat(float j) { int r = int(j + 0.5); return mat4(texelFetch(uJointTex, ivec2(0, r), 0), texelFetch(uJointTex, ivec2(1, r), 0), texelFetch(uJointTex, ivec2(2, r), 0), texelFetch(uJointTex, ivec2(3, r), 0)); }
        void main() {
            vec4 p = vec4(aPos, 1.0); vec3 n = aNrm;
            if (uSkinned) {
                mat4 sk = aWeights.x * jointMat(aJoints.x) + aWeights.y * jointMat(aJoints.y) + aWeights.z * jointMat(aJoints.z) + aWeights.w * jointMat(aJoints.w);
                p = sk * p; n = mat3(sk) * aNrm;
            }
            vNrm = mat3(uModel) * n; vUv = aUv; gl_Position = uMvp * p;
        }`;
    const FS = `#version 300 es
        precision highp float;
        in vec3 vNrm; in vec2 vUv;
        uniform sampler2D uTex; uniform bool uHasTex;
        out vec4 frag;
        void main() {
            vec3 base = uHasTex ? texture(uTex, vUv).rgb : vec3(0.75);
            vec3 n = normalize(vNrm);
            if (!gl_FrontFacing) n = -n;
            float key = max(dot(n, normalize(vec3(0.4, 0.8, 0.6))), 0.0);
            float fill = max(dot(n, normalize(vec3(-0.6, 0.2, -0.5))), 0.0);
            frag = vec4(base * (0.35 + 0.75 * key + 0.25 * fill), 1.0);
        }`;

    function compile(gl, type, src) {
        const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
        if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
        return s;
    }

    function mul(a, b) {
        const o = new Float32Array(16);
        for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) {
            let s = 0; for (let k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k];
            o[c * 4 + r] = s;
        }
        return o;
    }
    function perspective(fovy, aspect, near, far) {
        const f = 1 / Math.tan(fovy / 2), nf = 1 / (near - far);
        return new Float32Array([f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, (far + near) * nf, -1, 0, 0, 2 * far * near * nf, 0]);
    }
    function rotY(a) { const c = Math.cos(a), s = Math.sin(a); return new Float32Array([c, 0, -s, 0, 0, 1, 0, 0, s, 0, c, 0, 0, 0, 0, 1]); }
    function rotX(a) { const c = Math.cos(a), s = Math.sin(a); return new Float32Array([1, 0, 0, 0, 0, c, s, 0, 0, -s, c, 0, 0, 0, 0, 1]); }
    function translate(x, y, z) { return new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, y, z, 1]); }

    // joint matrices of a skin at time t of an animation: local T (rest, or the animated root translation) and R (animated or identity), parents first, times the inverse bind matrices
    function slerpQuat(out, a, ao, b, bo, u) {
        let ax = a[ao], ay = a[ao + 1], az = a[ao + 2], aw = a[ao + 3], bx = b[bo], by = b[bo + 1], bz = b[bo + 2], bw = b[bo + 3];
        let d = ax * bx + ay * by + az * bz + aw * bw; if (d < 0) { bx = -bx; by = -by; bz = -bz; bw = -bw; d = -d; }
        let ka, kb;
        if (d > 0.9995) { ka = 1 - u; kb = u; } else { const th = Math.acos(d), s = Math.sin(th); ka = Math.sin((1 - u) * th) / s; kb = Math.sin(u * th) / s; }
        const x = ka * ax + kb * bx, y = ka * ay + kb * by, z = ka * az + kb * bz, w = ka * aw + kb * bw, n = 1 / Math.hypot(x, y, z, w);
        out[0] = x * n; out[1] = y * n; out[2] = z * n; out[3] = w * n;
    }
    function sampleTrack(tr, t, out) {
        const times = tr.times, n = times.length;
        let k = 0; if (t >= times[n - 1]) k = n - 2; else { while (k < n - 2 && times[k + 1] <= t) k++; }
        k = Math.max(0, k);
        const span = times[k + 1] - times[k], u = n < 2 || span <= 0 ? 0 : Math.min(1, Math.max(0, (t - times[k]) / span));
        if (n < 2) { for (let c = 0; c < tr.comps; c++) out[c] = tr.values[c]; return; }
        if (tr.comps === 4) slerpQuat(out, tr.values, k * 4, tr.values, (k + 1) * 4, u);
        else for (let c = 0; c < 3; c++) out[c] = tr.values[k * 3 + c] + u * (tr.values[(k + 1) * 3 + c] - tr.values[k * 3 + c]);
    }
    function poseMatrices(skin, anim, t, inPlace, outTex, outPos) {
        const J = skin.J, world = new Array(J), q = new Float32Array(4), tmp = new Float32Array(3);
        const rot = new Float32Array(J * 4), trans = Float32Array.from(skin.restT);
        for (let j = 0; j < J; j++) rot[j * 4 + 3] = 1;
        if (anim) for (const tr of anim.tracks) {
            if (tr.path === "rotation") { sampleTrack(tr, t, q); rot.set(q, tr.joint * 4); }
            else if (tr.path === "translation") {
                sampleTrack(tr, t, tmp);
                if (inPlace) { tmp[0] = skin.restT[tr.joint * 3]; tmp[2] = skin.restT[tr.joint * 3 + 2]; }            // drop the hips' travel along the ground, keep their bob
                trans.set(tmp, tr.joint * 3);
            }
        }
        for (let j = 0; j < J; j++) {
            const x = rot[j * 4], y = rot[j * 4 + 1], z = rot[j * 4 + 2], w = rot[j * 4 + 3];
            const L = new Float32Array([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y), 0, 2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x), 0,
                                        2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y), 0, trans[j * 3], trans[j * 3 + 1], trans[j * 3 + 2], 1]);
            world[j] = skin.parent[j] >= 0 ? mul(world[skin.parent[j]], L) : L;
            outTex.set(mul(world[j], skin.ibm.subarray(j * 16, j * 16 + 16)), j * 16);
            if (outPos) { outPos[j * 3] = world[j][12]; outPos[j * 3 + 1] = world[j][13]; outPos[j * 3 + 2] = world[j][14]; }
        }
    }

    async function show(canvas, glbBytes) {
        if (typeof canvas === "string") canvas = document.getElementById(canvas);
        dispose(canvas);
        const mesh = parseGlb(glbBytes instanceof Uint8Array ? glbBytes : new Uint8Array(glbBytes));
        const gl = canvas.getContext("webgl2", { antialias: true });
        if (!gl) throw new Error("WebGL2 is not available for the preview");
        const prog = gl.createProgram();
        gl.attachShader(prog, compile(gl, gl.VERTEX_SHADER, VS));
        gl.attachShader(prog, compile(gl, gl.FRAGMENT_SHADER, FS));
        gl.linkProgram(prog);
        const vao = gl.createVertexArray(); gl.bindVertexArray(vao);
        const attr = (name, data, size) => {
            const loc = gl.getAttribLocation(prog, name); if (loc < 0 || !data) return;
            const b = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, b); gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
            gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, size, gl.FLOAT, false, 0, 0);
        };
        attr("aPos", mesh.positions, 3); attr("aNrm", mesh.normals, 3); attr("aUv", mesh.uvs, 2);
        const skinned = !!mesh.skin;
        if (skinned) {
            const jl = gl.getAttribLocation(prog, "aJoints"), wl = gl.getAttribLocation(prog, "aWeights");
            const jb = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, jb); gl.bufferData(gl.ARRAY_BUFFER, mesh.joints, gl.STATIC_DRAW);
            gl.enableVertexAttribArray(jl); gl.vertexAttribPointer(jl, 4, mesh.jointsType === "u8" ? gl.UNSIGNED_BYTE : gl.UNSIGNED_SHORT, false, 0, 0);
            const wb = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, wb); gl.bufferData(gl.ARRAY_BUFFER, mesh.weights, gl.STATIC_DRAW);
            gl.enableVertexAttribArray(wl); gl.vertexAttribPointer(wl, 4, gl.FLOAT, false, 0, 0);
        }
        // skeleton overlay (glbViewer.skeleton(canvas, true)): the bones as lines and the joints as dots, drawn over the model
        const lineProg = gl.createProgram();
        gl.attachShader(lineProg, compile(gl, gl.VERTEX_SHADER, `#version 300 es
            in vec3 aPos; uniform mat4 uMvp; void main() { gl_Position = uMvp * vec4(aPos, 1.0); gl_PointSize = 7.0; }`));
        gl.attachShader(lineProg, compile(gl, gl.FRAGMENT_SHADER, `#version 300 es
            precision mediump float; uniform vec3 uColor; out vec4 frag; void main() { frag = vec4(uColor, 1.0); }`));
        gl.linkProgram(lineProg);
        const lineVao = gl.createVertexArray(), lineBuf = gl.createBuffer();
        let jointPos = null;
        let jointTex = null, jointData = null;
        if (skinned) {
            jointData = new Float32Array(mesh.skin.J * 16);
            jointTex = gl.createTexture(); gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, jointTex);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
            jointPos = new Float32Array(mesh.skin.J * 3);
            poseMatrices(mesh.skin, null, 0, true, jointData, jointPos);                                  // the bind pose: identity joint matrices
            gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, 4, mesh.skin.J, 0, gl.RGBA, gl.FLOAT, jointData);
            gl.activeTexture(gl.TEXTURE0);
        }
        const ib = gl.createBuffer(); gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ib); gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, mesh.indices, gl.STATIC_DRAW);
        let hasTex = false;
        if (mesh.image) {
            const bmp = await createImageBitmap(mesh.image);
            const tex = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, tex);
            gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, bmp);
            gl.generateMipmap(gl.TEXTURE_2D);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
            hasTex = true;
        }
        // centre and scale to the unit box
        let mn = [Infinity, Infinity, Infinity], mx = [-Infinity, -Infinity, -Infinity];
        for (let i = 0; i < mesh.positions.length; i += 3) for (let k = 0; k < 3; k++) { const v = mesh.positions[i + k]; if (v < mn[k]) mn[k] = v; if (v > mx[k]) mx[k] = v; }
        const ctr = [0, 1, 2].map(k => (mn[k] + mx[k]) / 2), ext = Math.max(mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2]) || 1;
        const state = { yaw: 0.6, pitch: 0.35, dist: 2.2, auto: true, drag: null, raf: 0, gl, count: mesh.indices.length, mesh, anim: -1, t0: 0, inPlace: true, fixedT: null, skeleton: false };
        const uMvp = gl.getUniformLocation(prog, "uMvp"), uModel = gl.getUniformLocation(prog, "uModel");
        const uTex = gl.getUniformLocation(prog, "uTex"), uHasTex = gl.getUniformLocation(prog, "uHasTex");
        const onDown = e => { state.drag = { x: e.clientX, y: e.clientY }; state.auto = false; canvas.setPointerCapture(e.pointerId); };
        const onMove = e => { if (!state.drag) return; state.yaw += (e.clientX - state.drag.x) * 0.01; state.pitch = Math.max(-1.5, Math.min(1.5, state.pitch + (e.clientY - state.drag.y) * 0.01)); state.drag = { x: e.clientX, y: e.clientY }; };
        const onUp = () => { state.drag = null; };
        const onWheel = e => { e.preventDefault(); state.dist = Math.max(0.8, Math.min(8, state.dist * (e.deltaY > 0 ? 1.1 : 0.9))); };
        canvas.addEventListener("pointerdown", onDown); canvas.addEventListener("pointermove", onMove);
        canvas.addEventListener("pointerup", onUp); canvas.addEventListener("wheel", onWheel, { passive: false });
        state.cleanup = () => { canvas.removeEventListener("pointerdown", onDown); canvas.removeEventListener("pointermove", onMove); canvas.removeEventListener("pointerup", onUp); canvas.removeEventListener("wheel", onWheel); };
        const frame = () => {
            const w = canvas.clientWidth | 0, h = canvas.clientHeight | 0;
            if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
            if (state.auto) state.yaw += 0.004;
            gl.viewport(0, 0, w, h); gl.clearColor(0.06, 0.08, 0.12, 1); gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
            gl.enable(gl.DEPTH_TEST); gl.useProgram(prog); gl.bindVertexArray(vao);
            const model = mul(rotX(state.pitch), mul(rotY(state.yaw), mul(new Float32Array([1 / ext, 0, 0, 0, 0, 1 / ext, 0, 0, 0, 0, 1 / ext, 0, 0, 0, 0, 1]), translate(-ctr[0], -ctr[1], -ctr[2]))));
            const mvp = mul(perspective(0.8, w / Math.max(h, 1), 0.05, 50), mul(translate(0, 0, -state.dist), model));
            gl.uniformMatrix4fv(uMvp, false, mvp); gl.uniformMatrix4fv(uModel, false, model);
            gl.uniform1i(uTex, 0); gl.uniform1i(uHasTex, hasTex ? 1 : 0);
            gl.uniform1i(gl.getUniformLocation(prog, "uSkinned"), skinned ? 1 : 0); gl.uniform1i(gl.getUniformLocation(prog, "uJointTex"), 1);
            if (skinned) {
                const anim = state.anim >= 0 ? mesh.animations[state.anim] : null;
                const t = anim && anim.duration > 0 ? (state.fixedT != null ? state.fixedT : (performance.now() - state.t0) / 1000) % anim.duration : 0;
                poseMatrices(mesh.skin, anim, t, state.inPlace, jointData, jointPos);
                gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, jointTex);
                gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, 4, mesh.skin.J, 0, gl.RGBA, gl.FLOAT, jointData);
                gl.activeTexture(gl.TEXTURE0);
            }
            gl.drawElements(gl.TRIANGLES, state.count, gl.UNSIGNED_INT, 0);
            if (state.skeleton && skinned) {
                const J = mesh.skin.J, lines = [];
                for (let j = 0; j < J; j++) { const pa = mesh.skin.parent[j]; if (pa >= 0) lines.push(jointPos[pa * 3], jointPos[pa * 3 + 1], jointPos[pa * 3 + 2], jointPos[j * 3], jointPos[j * 3 + 1], jointPos[j * 3 + 2]); }
                gl.useProgram(lineProg); gl.bindVertexArray(lineVao); gl.disable(gl.DEPTH_TEST);
                gl.bindBuffer(gl.ARRAY_BUFFER, lineBuf); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(lines), gl.DYNAMIC_DRAW);
                const al = gl.getAttribLocation(lineProg, "aPos"); gl.enableVertexAttribArray(al); gl.vertexAttribPointer(al, 3, gl.FLOAT, false, 0, 0);
                gl.uniformMatrix4fv(gl.getUniformLocation(lineProg, "uMvp"), false, mvp); gl.uniform3f(gl.getUniformLocation(lineProg, "uColor"), 1.0, 0.62, 0.1);
                gl.drawArrays(gl.LINES, 0, lines.length / 3); gl.uniform3f(gl.getUniformLocation(lineProg, "uColor"), 1.0, 0.9, 0.5); gl.drawArrays(gl.POINTS, 0, lines.length / 3);
                gl.enable(gl.DEPTH_TEST);
            }
            state.raf = requestAnimationFrame(frame);
        };
        state.raf = requestAnimationFrame(frame);
        viewers.set(canvas, state);
        return { vertices: mesh.positions.length / 3, triangles: mesh.indices.length / 3, animations: (mesh.animations || []).map(a => a.name) };
    }

    // Play one of the file's animations (by index or name; null/-1 = the bind pose). inPlace (default true) drops the hips' travel so the character stays in frame.
    function play(canvas, which, inPlace) {
        if (typeof canvas === "string") canvas = document.getElementById(canvas);
        const s = viewers.get(canvas); if (!s || !s.mesh.animations) return false;
        let i = typeof which === "string" ? s.mesh.animations.findIndex(a => a.name === which) : (which === null || which === undefined ? -1 : which);
        s.anim = i; s.t0 = performance.now(); s.inPlace = inPlace !== false;
        return i >= 0;
    }
    // Freeze the viewer at one pose, for capturing frames at exact times: animation `which` (index, name or -1 for the bind pose) at `seconds`, the model turned `yaw` radians.
    // `resume(canvas)` goes back to real-time playback and auto-rotation.
    function pose(canvas, which, seconds, yaw, inPlace) {
        if (typeof canvas === "string") canvas = document.getElementById(canvas);
        const s = viewers.get(canvas); if (!s) return false;
        s.anim = typeof which === "string" ? (s.mesh.animations || []).findIndex(a => a.name === which) : which;
        s.fixedT = seconds; s.yaw = yaw; s.auto = false; s.inPlace = inPlace !== false;
        return true;
    }
    function resume(canvas) {
        if (typeof canvas === "string") canvas = document.getElementById(canvas);
        const s = viewers.get(canvas); if (!s) return false;
        s.fixedT = null; s.auto = true; s.t0 = performance.now(); return true;
    }
    function skeleton(canvas, on) {
        if (typeof canvas === "string") canvas = document.getElementById(canvas);
        const s = viewers.get(canvas); if (!s) return false; s.skeleton = !!on; return true;
    }
    function animations(canvas) {
        if (typeof canvas === "string") canvas = document.getElementById(canvas);
        const s = viewers.get(canvas); return s && s.mesh.animations ? s.mesh.animations.map(a => a.name) : [];
    }

    function dispose(canvas) {
        if (typeof canvas === "string") canvas = document.getElementById(canvas);
        if (!canvas) return;
        const s = viewers.get(canvas);
        if (!s) return;
        cancelAnimationFrame(s.raf); s.cleanup && s.cleanup();
        viewers.delete(canvas);   // (the context stays: the same canvas gets it back on the next show)
    }

    window.glbViewer = { show, dispose, play, animations, pose, resume, skeleton };
})();
