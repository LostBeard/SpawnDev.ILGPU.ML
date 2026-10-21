// Serve the three.js viewer with a list of GLB files, and either keep serving (open the URL in your own browser) or take screenshots in a throwaway Chrome.
//
//   node tools/threejs-viewer/run.mjs <a.glb> [b.glb ...]                               serve on http://127.0.0.1:5180/ (Ctrl+C to stop)
//   node tools/threejs-viewer/run.mjs --shots <outDir> <a.glb> [b.glb ...]              PNGs <outDir>/<name>_yaw<deg>.png, then exit
//     env: SHOT_YAWS=20,110 (degrees)  SHOT_PITCH=8  SHOT_SIZE=1200  PORT=5180  CHROME_EXE  SHOT_CDP_PORT (default 9362)
import { spawn, execFileSync } from 'node:child_process';
import { createServer } from 'node:http';
import { readFileSync, writeFileSync, mkdirSync, rmSync, existsSync, statSync } from 'node:fs';
import { basename, dirname, extname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import os from 'node:os';

const here = dirname(fileURLToPath(import.meta.url));
let args = process.argv.slice(2), shotsDir = null;
if (args[0] === '--shots') { shotsDir = resolve(args[1]); args = args.slice(2); }
const glbs = args.map(p => resolve(p));
if (glbs.length === 0) { console.error('usage: node tools/threejs-viewer/run.mjs [--shots <outDir>] <a.glb> [b.glb ...]'); process.exit(2); }
for (const g of glbs) if (!existsSync(g)) { console.error('no such file: ' + g); process.exit(2); }
const PORT = Number(process.env.PORT || 5180);
const types = { '.html': 'text/html', '.js': 'text/javascript', '.json': 'application/json', '.glb': 'model/gltf-binary' };
const models = glbs.map((g, i) => ({ name: basename(g), url: `/glb/${i}/${encodeURIComponent(basename(g))}` }));

const server = createServer((req, res) => {
  const url = decodeURIComponent(req.url.split('?')[0]);
  if (url === '/models.json') { res.writeHead(200, { 'content-type': 'application/json' }); res.end(JSON.stringify(models)); return; }
  const m = url.match(/^\/glb\/(\d+)\//);
  const file = m ? glbs[Number(m[1])] : join(here, url === '/' ? 'index.html' : url.replace(/^\//, ''));
  if (!file || !existsSync(file) || !statSync(file).isFile() || (!m && !file.startsWith(here))) { res.writeHead(404); res.end(); return; }
  res.writeHead(200, { 'content-type': types[extname(file)] || 'application/octet-stream' }); res.end(readFileSync(file));
});
await new Promise(r => server.listen(PORT, '127.0.0.1', r));
console.log(`three.js viewer: http://127.0.0.1:${PORT}/   (${models.length} model(s))`);
if (!shotsDir) { await new Promise(() => {}); }

// ---- screenshot mode ----
mkdirSync(shotsDir, { recursive: true });
const YAWS = (process.env.SHOT_YAWS ?? '20,110').split(',').map(Number), PITCH = Number(process.env.SHOT_PITCH ?? 8), SIZE = Number(process.env.SHOT_SIZE || 1200);
const CDP = Number(process.env.SHOT_CDP_PORT || 9362);
const CHROME = process.env.CHROME_EXE || 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const PROFILE = join(os.tmpdir(), 'spawndev-ml-three-profile'); rmSync(PROFILE, { recursive: true, force: true });
const chrome = spawn(CHROME, [`--user-data-dir=${PROFILE}`, `--remote-debugging-port=${CDP}`, '--no-first-run', '--no-default-browser-check', '--ignore-gpu-blocklist', `--window-size=${SIZE + 40},${SIZE + 140}`, 'about:blank'], { stdio: 'ignore' });
const kill = () => { try { execFileSync('taskkill', ['/PID', String(chrome.pid), '/T', '/F'], { stdio: 'ignore' }); } catch { /* gone */ } };
process.on('exit', kill);
const sleep = ms => new Promise(r => setTimeout(r, ms));
let tabs; for (let i = 0; i < 60; i++) { try { tabs = await (await fetch(`http://127.0.0.1:${CDP}/json/list`)).json(); if (tabs.some(t => t.type === 'page')) break; } catch { /* starting */ } await sleep(500); }
const ws = new WebSocket(tabs.find(t => t.type === 'page').webSocketDebuggerUrl); await new Promise(r => ws.onopen = r);
let id = 0; const pend = new Map(); ws.onmessage = e => { const m = JSON.parse(e.data); if (m.id && pend.has(m.id)) { pend.get(m.id)(m); pend.delete(m.id); } };
const send = (method, params = {}) => new Promise(r => { const i = ++id; pend.set(i, r); ws.send(JSON.stringify({ id: i, method, params })); });
await send('Page.enable'); await send('Runtime.enable');
await send('Emulation.setDeviceMetricsOverride', { width: SIZE, height: SIZE, deviceScaleFactor: 1, mobile: false });
for (const m of models) for (const yaw of YAWS) {
  await send('Page.navigate', { url: `http://127.0.0.1:${PORT}/?shot=1&glb=${encodeURIComponent(m.url)}&yaw=${yaw}&pitch=${PITCH}` });
  let ready = false; const t0 = Date.now();
  while (!ready && Date.now() - t0 < 120000) { await sleep(400); const r = await send('Runtime.evaluate', { expression: 'window.__ready === true', returnByValue: true }); ready = r.result?.result?.value === true; }
  if (!ready) { console.error(`${m.name} yaw ${yaw}: not ready in 120 s`); continue; }
  await sleep(500);
  const s = await send('Page.captureScreenshot', { format: 'png' });
  const out = join(shotsDir, `${m.name.replace(/\.glb$/i, '')}_yaw${yaw}.png`); writeFileSync(out, Buffer.from(s.result.data, 'base64')); console.log('saved', out);
}
ws.close(); server.close(); kill(); process.exit(0);
