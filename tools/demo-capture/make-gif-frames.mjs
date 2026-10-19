// Frames for the README GIF, captured from the real demo page: prompt -> 3D model (turntable) -> rig (skeleton overlay, turntable) -> animated (dance, wave).
// Deterministic: the viewer is frozen at exact (animation time, yaw) with glbViewer.pose(), one screenshot per frame.
// usage: node make-gif-frames.mjs <profileDir> <outDir> <prompt> <seed>
import { spawn } from 'node:child_process';
import { writeFileSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';

const PROFILE = process.argv[2], OUT = process.argv[3], PROMPT = process.argv[4], SEED = process.argv[5];
mkdirSync(OUT, { recursive: true });
const PORT = 9339, URL_ = 'http://localhost:5088/image-to-3d';
const chrome = spawn('C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
  [`--user-data-dir=${PROFILE}`, `--remote-debugging-port=${PORT}`, '--enable-unsafe-webgpu', '--enable-features=Vulkan', '--no-first-run', '--no-default-browser-check', '--window-size=1400,1000', 'about:blank'], { stdio: 'ignore' });
const sleep = ms => new Promise(r => setTimeout(r, ms));
const t0 = Date.now(); const T = () => ((Date.now() - t0) / 1000).toFixed(0).padStart(5) + 's';
try {
  for (let i = 0; i < 60; i++) { try { await (await fetch(`http://127.0.0.1:${PORT}/json/version`)).json(); break; } catch { await sleep(500); } }
  const tgt = await (await fetch(`http://127.0.0.1:${PORT}/json/new?about:blank`, { method: 'PUT' })).json();
  const ws = new WebSocket(tgt.webSocketDebuggerUrl); await new Promise(r => ws.addEventListener('open', r));
  let id = 0; const pending = new Map();
  ws.addEventListener('message', e => { const m = JSON.parse(e.data); if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); } });
  const send = (method, params = {}) => new Promise(r => { const i = ++id; pending.set(i, r); ws.send(JSON.stringify({ id: i, method, params })); });
  const ev = async expr => { const r = await send('Runtime.evaluate', { expression: expr, awaitPromise: true, returnByValue: true, timeout: 120000 }); return r.result?.exceptionDetails ? 'EXC ' + JSON.stringify(r.result.exceptionDetails.exception?.description) : r.result?.result?.value; };
  await send('Page.enable'); await send('Runtime.enable'); await send('DOM.enable');
  await send('Page.navigate', { url: URL_ });
  for (let i = 0; i < 120; i++) { if (await ev(`!!document.querySelector('.t3d-prompt')`) === true) break; await sleep(1000); }
  // image mode: hand the page the picture SD Turbo drew for this prompt (cand_06, seed 23)
  for (let i = 0; i < 180; i++) { if (await ev(`!!document.querySelector('input[type=file]')`) === true) break; await sleep(1000); }
  const doc = (await send('DOM.getDocument')).result; const node = (await send('DOM.querySelector', { nodeId: doc.root.nodeId, selector: 'input[type=file]' })).result.nodeId;
  await send('DOM.setFileInputFiles', { files: [process.argv[6]], nodeId: node });
  writeFileSync(join(OUT, 'picture.png'), (await import('node:fs')).readFileSync(process.argv[6]));
  for (let i = 0; i < 60; i++) { if (await ev(`!![...document.querySelectorAll('.t3d-btn')].find(b=>/Generate/.test(b.textContent)&&!b.disabled)`) === true) break; await sleep(500); }
  await ev(`[...document.querySelectorAll('.t3d-btn')].find(b=>/Generate/.test(b.textContent)).click()`);
  const waitFor = async (what, cond, maxS) => { for (let i = 0; i < maxS / 3; i++) { await sleep(3000); const v = await ev(cond); if (v && v !== 'false') return v; const err = await ev(`document.querySelector('.error-banner')?.innerText ?? ''`); if (err) { console.log('ERROR BANNER:', err.slice(0, 500)); throw new Error('page error during ' + what); } } throw new Error('timeout: ' + what); };
  console.log(T(), 'model:', await waitFor('generation', `(document.querySelector('.t3d-result-info')?.innerText ?? '').includes('GLB') && document.querySelector('.t3d-result-info').innerText`, 1800));
  await ev(`document.getElementById('t3d-canvas').scrollIntoView({block:'center'})`); await sleep(800);
  const rect = JSON.parse(await ev(`JSON.stringify(document.getElementById('t3d-canvas').getBoundingClientRect())`));
  console.log('canvas', JSON.stringify(rect)); writeFileSync(join(OUT, 'rect.json'), JSON.stringify(rect));
  const frame = async name => {
    await sleep(110);                                                          // a few animation frames so the new pose is on screen
    const r = await send('Page.captureScreenshot', { format: 'png' });                  // the whole viewport; make_gif.py crops it with rect.json (clip coordinates are page, not viewport, coordinates)
    writeFileSync(join(OUT, name), Buffer.from(r.result.data, 'base64'));
  };
  const pose = (which, t, yaw, skel) => ev(`(()=>{glbViewer.skeleton('t3d-canvas', ${skel});return glbViewer.pose('t3d-canvas', ${JSON.stringify(which)}, ${t}, ${yaw}, true)})()`);
  const TAU = Math.PI * 2;
  for (let k = 0; k < 36; k++) { await pose(-1, 0, 0.5 + TAU * k / 36, false); await frame(`A_${String(k).padStart(3, '0')}.png`); }       // stage 1: the generated model, a full turn
  console.log(T(), 'stage 1 frames done');
  console.log(T(), 'rig click', await ev(`(()=>{const b=[...document.querySelectorAll('button')].find(b=>/Rig this model/.test(b.textContent));b.click();return b.textContent})()`));
  console.log(T(), 'rig:', await waitFor('rig', `(()=>{const t=[...document.querySelectorAll('.t3d-hint')].map(h=>h.innerText).join(' | ');return /Rigged:/.test(t)&&t.slice(t.indexOf('Rigged:'), t.indexOf('Rigged:') + 70)})()`, 900));
  console.log(T(), 'animate click', await ev(`(()=>{const b=[...document.querySelectorAll('button')].find(b=>/Animate/.test(b.textContent));b.click();return b.textContent})()`));
  console.log(T(), 'animate:', await waitFor('animate', `(()=>{const t=[...document.querySelectorAll('.t3d-hint')].map(h=>h.innerText).join(' | ');return /moves on/.test(t)&&t.slice(t.indexOf('moves on') - 3, t.indexOf('moves on') + 40)})()`, 300));
  await ev(`document.getElementById('t3d-canvas').scrollIntoView({block:'center'})`); await sleep(800);
  for (let k = 0; k < 30; k++) { await pose(-1, 0, 0.5 + TAU * k / 36 * 0.8, true); await frame(`B_${String(k).padStart(3, '0')}.png`); }          // stage 2: the skeleton over the model in bind pose
  console.log(T(), 'stage 2 frames done');
  const dur = JSON.parse(await ev(`JSON.stringify(glbViewer.animations('t3d-canvas'))`)); console.log('moves', dur.length);
  const clip = async (name, seconds, fps, tag, yaw0, yaw1) => { const n = Math.round(seconds * fps); for (let k = 0; k < n; k++) { await pose(name, k / fps, yaw0 + (yaw1 - yaw0) * k / n, false); await frame(`${tag}_${String(k).padStart(3, '0')}.png`); } };
  await clip('dance', 3.0, 15, 'C', 0.4, 0.9);
  await clip('wave', 2.0, 15, 'D', 0.5, 0.5);
  await clip('walk', 2.0, 15, 'E', 0.3, 1.2);
  console.log(T(), 'stage 3 frames done');
  ws.close();
} finally { try { process.kill(chrome.pid); } catch {} }
