// Minimal GLB viewer for the Text/Image to 3D demo: WebGL2, one textured mesh (POSITION, NORMAL, TEXCOORD_0, uint32 indices,
// baseColorTexture), orbit with the mouse (drag to rotate, wheel to zoom), slow auto-rotation until the first drag.
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
        const mat = json.materials && json.materials[prim.material || 0];
        const texInfo = mat && mat.pbrMetallicRoughness && mat.pbrMetallicRoughness.baseColorTexture;
        if (texInfo) {
            const img = json.images[json.textures[texInfo.index].source], bv = json.bufferViews[img.bufferView];
            mesh.image = new Blob([bin.slice(bv.byteOffset || 0, (bv.byteOffset || 0) + bv.byteLength)], { type: img.mimeType || "image/png" });
        }
        return mesh;
    }

    const VS = `#version 300 es
        in vec3 aPos; in vec3 aNrm; in vec2 aUv;
        uniform mat4 uMvp; uniform mat4 uModel;
        out vec3 vNrm; out vec2 vUv;
        void main() { vNrm = mat3(uModel) * aNrm; vUv = aUv; gl_Position = uMvp * vec4(aPos, 1.0); }`;
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

    async function show(canvas, glbBytes) {
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
        const state = { yaw: 0.6, pitch: 0.35, dist: 2.2, auto: true, drag: null, raf: 0, gl, count: mesh.indices.length };
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
            gl.drawElements(gl.TRIANGLES, state.count, gl.UNSIGNED_INT, 0);
            state.raf = requestAnimationFrame(frame);
        };
        state.raf = requestAnimationFrame(frame);
        viewers.set(canvas, state);
        return { vertices: mesh.positions.length / 3, triangles: mesh.indices.length / 3 };
    }

    function dispose(canvas) {
        const s = viewers.get(canvas);
        if (!s) return;
        cancelAnimationFrame(s.raf); s.cleanup && s.cleanup();
        viewers.delete(canvas);   // (the context stays: the same canvas gets it back on the next show)
    }

    window.glbViewer = { show, dispose };
})();
