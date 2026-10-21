// Render GLB files with the demo's own WebGL viewer (wwwroot/js/glb-viewer.js) in a real Chrome and save PNGs: what a user sees in the demo, from fixed angles, for comparing generations.
//
//   node tools/render-glb.mjs <outDir> <a.glb> [b.glb ...]           one PNG per file and angle: <outDir>/<name>_yaw<deg>.png
//   env: RENDER_YAWS=20,110,200,290 (degrees)   RENDER_SIZE=1200 (px, square)   RENDER_ZOOM=2.5 (magnify the centre: the viewer frames a model small)   CHROME_EXE   RENDER_CDP_PORT (default 9361)
//
// Own Chrome profile + port (never the user's browser); the Chrome is killed by PID at the end.
import { spawn, execFileSync } from 'node:child_process';
import { createServer } from 'node:http';
import { readFileSync, writeFileSync, mkdirSync, rmSync, existsSync } from 'node:fs';
import { basename, dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import os from 'node:os';

const here = dirname(fileURLToPath(import.meta.url));
const [outDirArg, ...glbArgs] = process.argv.slice(2);
if (!outDirArg || glbArgs.length === 0) { console.error('usage: node tools/render-glb.mjs <outDir> <a.glb> [b.glb ...]'); process.exit(2); }
const OUT = resolve(outDirArg); mkdirSync(OUT, { recursive: true });
const glbs = glbArgs.map(p => resolve(p));
for (const g of glbs) if (!existsSync(g)) { console.error('no such file: ' + g); process.exit(2); }
const YAWS = (process.env.RENDER_YAWS ?? '20,110,200,290').split(',').map(Number);
const SIZE = Number(process.env.RENDER_SIZE || 1200);
const ZOOM = Number(process.env.RENDER_ZOOM || 1);
const ANIM = process.env.RENDER_ANIM === undefined ? '-1' : (isNaN(Number(process.env.RENDER_ANIM)) ? JSON.stringify(process.env.RENDER_ANIM) : process.env.RENDER_ANIM), ANIM_T = Number(process.env.RENDER_T || 0);   // RENDER_ANIM=<index|name> RENDER_T=<seconds>: a pose of an animation instead of the bind pose
const PORT = Number(process.env.RENDER_CDP_PORT || 9361);
const CHROME = process.env.CHROME_EXE || 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const PROFILE = join(os.tmpdir(), 'spawndev-ml-render-profile');
const viewerHtml = readFileSync(join(here, '..', 'Examples', '08.AssetGen.Wpf', 'viewer', 'viewer.html'), 'utf8');
const jsRoot = join(here, '..', 'SpawnDev.ILGPU.ML.Demo', 'wwwroot', 'js');      // glb-viewer.js and its lib/three

const server = createServer((req, res) => {
  const url = decodeURIComponent(req.url.split('?')[0]);
  if (url === '/' || url === '/viewer.html') { res.writeHead(200, { 'content-type': 'text/html' }); res.end(viewerHtml); }
  else if (url === '/glb-viewer.js' || url.startsWith('/lib/')) { const f = join(jsRoot, url); if (!f.startsWith(jsRoot) || !existsSync(f)) { res.writeHead(404); res.end(); return; } res.writeHead(200, { 'content-type': f.endsWith('.js') ? 'text/javascript' : 'text/plain' }); res.end(readFileSync(f)); }
  else if (url.startsWith('/glb/')) { const i = Number(url.slice(5)); res.writeHead(200, { 'content-type': 'model/gltf-binary' }); res.end(readFileSync(glbs[i])); }
  else { res.writeHead(404); res.end(); }
});
await new Promise(r => server.listen(0, '127.0.0.1', r));
const base = `http://127.0.0.1:${server.address().port}`;

rmSync(PROFILE, { recursive: true, force: true });
const chrome = spawn(CHROME, [`--user-data-dir=${PROFILE}`, `--remote-debugging-port=${PORT}`, '--no-first-run', '--no-default-browser-check', `--window-size=${SIZE + 40},${SIZE + 140}`, 'about:blank'], { stdio: 'ignore' });
const kill = () => { try { execFileSync('taskkill', ['/PID', String(chrome.pid), '/T', '/F'], { stdio: 'ignore' }); } catch { /* gone */ } };
process.on('exit', kill);
const sleep = ms => new Promise(r => setTimeout(r, ms));

let targets;
for (let i = 0; i < 60; i++) { try { targets = await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json(); if (targets.some(t => t.type === 'page')) break; } catch { /* starting */ } await sleep(500); }
const page = targets.find(t => t.type === 'page');
const ws = new WebSocket(page.webSocketDebuggerUrl);
await new Promise(r => ws.onopen = r);
let nextId = 0; const pending = new Map();
ws.onmessage = e => { const m = JSON.parse(e.data); if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); } };
const send = (method, params = {}) => new Promise(r => { const id = ++nextId; pending.set(id, r); ws.send(JSON.stringify({ id, method, params })); });
await send('Page.enable'); await send('Runtime.enable');
await send('Emulation.setDeviceMetricsOverride', { width: SIZE, height: SIZE, deviceScaleFactor: 1, mobile: false });

for (let i = 0; i < glbs.length; i++) {
  const name = basename(glbs[i]).replace(/\.glb$/i, '');
  await send('Page.navigate', { url: base + '/viewer.html' });
  await sleep(1500);
  const r = await send('Runtime.evaluate', { awaitPromise: true, returnByValue: true, expression: `(async()=>{const bytes=new Uint8Array(await (await fetch('/glb/${i}')).arrayBuffer());return JSON.stringify(await glbViewer.show('c',bytes));})()` });
  if (r.result?.exceptionDetails) { console.error(name + ': ' + JSON.stringify(r.result.exceptionDetails).slice(0, 300)); continue; }
  console.log(name, r.result?.result?.value);
  for (const yaw of YAWS) {
    await send('Runtime.evaluate', { expression: `glbViewer.pose('c', ${ANIM}, ${ANIM_T}, ${(yaw * Math.PI / 180).toFixed(4)}, true)` });
    await sleep(700);
    const side = SIZE / ZOOM, s = await send('Page.captureScreenshot', { format: 'png', clip: { x: (SIZE - side) / 2, y: (SIZE - side) / 2, width: side, height: side, scale: ZOOM } });
    writeFileSync(join(OUT, `${name}_yaw${yaw}.png`), Buffer.from(s.result.data, 'base64'));
  }
}
ws.close(); server.close(); kill();
