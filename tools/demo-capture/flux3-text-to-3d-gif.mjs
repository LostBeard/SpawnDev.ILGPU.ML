// Text to 3D with FLUX 3 and the Qwen3 caption writer, end to end on a page (the live site or a local one), and the frames for the README GIF:
// prompt -> written caption -> N FLUX 3 pictures -> the picture picked -> 3D model (turntable) -> rig (skeleton) -> animated (dance, wave, walk).
// The viewer is frozen at exact (animation time, yaw) with glbViewer.pose(), one viewport screenshot per frame; tools/unirig-harness/reference/make_gif.py assembles them.
// usage: node flux3-text-to-3d-gif.mjs <profileDir> <outDir> <prompt> <seed> <pictures> <pick (0-based, or 'wait': read it from <outDir>/pick.txt once you have looked at cand_*.png)> [url=https://lostbeard.github.io/SpawnDev.ILGPU.ML/text-to-3d]
import { spawn } from 'node:child_process';
import { writeFileSync, mkdirSync, existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
const [PROFILE, OUT, PROMPT, SEED, PICS, PICK = '0', URL_ = 'https://lostbeard.github.io/SpawnDev.ILGPU.ML/text-to-3d'] = process.argv.slice(2);
mkdirSync(OUT, { recursive: true });
const PORT = 9345;
const chrome = spawn(process.env.CHROME_EXE || 'C:/Program Files/Google/Chrome/Application/chrome.exe',
  [`--user-data-dir=${PROFILE}`, `--remote-debugging-port=${PORT}`, '--enable-unsafe-webgpu', '--no-first-run', '--no-default-browser-check', '--hide-crash-restore-bubble', '--window-size=1400,1100', 'about:blank'], { stdio: 'ignore' });
const sleep = ms => new Promise(r => setTimeout(r, ms));
const t0 = Date.now(); const T = () => ((Date.now() - t0) / 1000).toFixed(0).padStart(5) + 's';
try {
  for (let i = 0; i < 60; i++) { try { await (await fetch(`http://127.0.0.1:${PORT}/json/version`)).json(); break; } catch { await sleep(500); } }
  const tgt = await (await fetch(`http://127.0.0.1:${PORT}/json/new?about:blank`, { method: 'PUT' })).json();
  // the profile restores the last session's tabs (each a WebGPU page of its own, all sharing the card): close every page but ours, so the run is the only thing on the GPU
  for (const t of await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json()) if (t.type === 'page' && t.id !== tgt.id) await fetch(`http://127.0.0.1:${PORT}/json/close/${t.id}`);
  console.log('pages open:', (await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json()).filter(t => t.type === 'page').length);
  const ws = new WebSocket(tgt.webSocketDebuggerUrl); await new Promise(r => ws.addEventListener('open', r));
  let id = 0; const pending = new Map();
  ws.addEventListener('message', e => {
    const m = JSON.parse(e.data);
    if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); }
    else if (m.method === 'Runtime.consoleAPICalled') { const t = m.params.args.map(a => a.value ?? a.description ?? '').join(' '); if (/error|lost|OOM|out of memory|[VRAM]/i.test(t)) console.log('  [console]', t.slice(0, 220)); }
  });
  const send = (method, params = {}) => new Promise(r => { const i = ++id; pending.set(i, r); ws.send(JSON.stringify({ id: i, method, params })); });
  const ev = async expr => { const r = await send('Runtime.evaluate', { expression: expr, awaitPromise: true, returnByValue: true, timeout: 120000 }); return r.result?.exceptionDetails ? 'EXC ' + JSON.stringify(r.result.exceptionDetails.exception?.description ?? r.result.exceptionDetails.text) : r.result?.result?.value; };
  await send('Page.enable'); await send('Runtime.enable');
  // sharp frames: the page is laid out at 1400 x 1100 but rendered at GIF_DSF x the pixels (the viewer's renderer follows devicePixelRatio, up to 2), and each frame is a whole-viewport screenshot (a clip of the canvas comes out blank: the WebGL canvas is not captured through a clip) that make_gif.py crops by rect.json
  const DSF = Number(process.env.GIF_DSF ?? 2), DIST = Number(process.env.GIF_DIST ?? 1.45);   // GIF_DIST: the camera distance (the viewer's default 1.8 leaves a standing figure at half the frame)
  await send('Emulation.setDeviceMetricsOverride', { width: 1400, height: 1100, deviceScaleFactor: DSF, mobile: false });
  // log WebGPU device-lost reasons (a lost device shows up in the page only as "A valid external Instance reference no longer exists")
  await send('Page.addScriptToEvaluateOnNewDocument', { source: `(()=>{const orig=GPUAdapter.prototype.requestDevice;GPUAdapter.prototype.requestDevice=async function(...a){const d=await orig.apply(this,a);d.lost.then(i=>console.error('[device.lost] reason='+i.reason+' message='+i.message));d.addEventListener('uncapturederror',e=>console.error('[uncapturederror] '+(e.error&&e.error.message)));return d;};})()` });
  await send('Page.navigate', { url: URL_ });
  for (let i = 0; i < 180; i++) { if (await ev(`!!document.querySelector('.t3d-prompt')`) === true) break; await sleep(1000); }
  const shot = async name => { const r = await send('Page.captureScreenshot', { format: 'png' }); writeFileSync(join(OUT, name), Buffer.from(r.result.data, 'base64')); };
  const setSel = (valueMatch, v) => ev(`(()=>{const s=[...document.querySelectorAll('select.t3d-select')].find(x=>[...x.options].some(o=>o.value==='${valueMatch}'));if(!s)return 'nosel';s.value='${v}';s.dispatchEvent(new Event('change',{bubbles:true}));return s.value})()`);
  console.log('image model', await setSel('flux3', 'flux3'), '| pictures', await setSel('8', PICS));
  // the 3D quality (GIF_DETAIL 512 | 1024 | 1024c | 1536c, GIF_FACES, GIF_TEXTURE; defaults: the best the browser runs: the 1024 cascade, 1M triangles, a 4096 texture)
  console.log('detail', await setSel('1024c', process.env.GIF_DETAIL ?? '1024c'), '| triangles', await setSel('1000000', process.env.GIF_FACES ?? '1000000'), '| texture', await setSel('4096', process.env.GIF_TEXTURE ?? '4096'));
  await ev(`(()=>{const p=document.querySelector('.t3d-prompt');p.value=${JSON.stringify(PROMPT)};p.dispatchEvent(new Event('input',{bubbles:true}));const s=document.querySelector('.t3d-num');s.value='${SEED}';s.dispatchEvent(new Event('input',{bubbles:true}));s.dispatchEvent(new Event('change',{bubbles:true}))})()`);
  // the caption writer: make sure its box is ticked
  console.log('caption writer ticked:', await ev(`(()=>{const c=[...document.querySelectorAll('label')].find(l=>/full caption/.test(l.textContent))?.querySelector('input[type=checkbox]');if(!c)return 'nobox';if(!c.checked)c.click();return c.checked})()`));
  await sleep(800);
  await shot('S0_prompt.png');
  await ev(`[...document.querySelectorAll('.t3d-btn')].find(b=>/Generate/.test(b.textContent)).click()`);
  const want = Number(PICS);
  let lastStage = '', shots = 0;
  for (let i = 0; i < 2400; i++) {
    await sleep(3000);
    const st = JSON.parse(await ev(`JSON.stringify({n:document.querySelectorAll('.t3d-pick-item').length,stage:document.querySelector('.t3d-stage.running')?.innerText?.replace(/\\s+/g,' ')??'',err:document.querySelector('.error-banner')?.innerText??''})`));
    if (st.stage !== lastStage) { console.log(T(), 'stage:', st.stage, '| pictures', st.n); lastStage = st.stage; if (shots < 6) await shot(`S1_stage_${shots++}.png`); }
    if (st.err) { console.log('ERROR', st.err.slice(0, 600)); break; }
    if (st.n >= want && !st.stage) { console.log(T(), 'all pictures drawn'); break; }
  }
  const caption = await ev(`document.querySelector('details.t3d-hint p')?.innerText ?? ''`);
  writeFileSync(join(OUT, 'caption.txt'), caption); console.log('caption:', caption.slice(0, 300));
  await ev(`document.querySelector('details.t3d-hint summary')?.click()`); await sleep(400);
  const urls = JSON.parse(await ev(`JSON.stringify([...document.querySelectorAll('.t3d-pick-item img')].map(i=>i.src))`));
  urls.forEach((u, k) => { if (u.startsWith('data:')) writeFileSync(join(OUT, `cand_${k}.png`), Buffer.from(u.split(',')[1], 'base64')); });
  await ev(`document.querySelector('.t3d-pick')?.scrollIntoView({block:'center'})`); await sleep(600);
  await shot('S2_picker.png');
  let pick = PICK;
  if (PICK === 'wait') { console.log(T(), 'pictures saved as cand_*.png; write the index to', join(OUT, 'pick.txt')); while (!existsSync(join(OUT, 'pick.txt'))) await sleep(2000); pick = readFileSync(join(OUT, 'pick.txt'), 'utf8').trim(); }
  writeFileSync(join(OUT, 'picture.png'), Buffer.from(urls[Number(pick)].split(',')[1], 'base64'));
  // build the picked picture in 3D
  console.log('pick', pick, await ev(`(()=>{const b=document.querySelectorAll('.t3d-pick-item button')[${Number(pick)}];b.click();return b.textContent})()`));
  for (let i = 0; i < 2400; i++) {
    await sleep(3000);
    const st = JSON.parse(await ev(`JSON.stringify({stage:document.querySelector('.t3d-stage.running')?.innerText?.replace(/\\s+/g,' ')??'',err:document.querySelector('.error-banner')?.innerText??'',glb:(document.querySelector('.t3d-result-info')?.innerText??'').includes('GLB')})`));
    if (st.stage !== lastStage) { console.log(T(), 'stage:', st.stage); lastStage = st.stage; }
    if (st.err) { console.log('ERROR', st.err.slice(0, 600)); break; }
    if (st.glb) { console.log(T(), 'GLB ready:', await ev(`document.querySelector('.t3d-result-info').innerText`)); break; }
  }
  await ev(`document.getElementById('t3d-canvas').scrollIntoView({block:'center'})`); await sleep(1200);
  const rect = JSON.parse(await ev(`JSON.stringify(document.getElementById('t3d-canvas').getBoundingClientRect())`));
  writeFileSync(join(OUT, 'rect.json'), JSON.stringify({ x: Math.round(rect.x * DSF), y: Math.round(rect.y * DSF), width: Math.round(rect.width * DSF), height: Math.round(rect.height * DSF) })); console.log('canvas', JSON.stringify(rect));   // in the pixels of the (DSF x) screenshots
  await shot('S3_model.png');
  const frame = async name => { await sleep(110); await shot(name); };
  const pose = (which, t, yaw, skel) => ev(`(()=>{glbViewer.skeleton('t3d-canvas', ${skel});return glbViewer.pose('t3d-canvas', ${JSON.stringify(which)}, ${t}, ${yaw}, true, ${DIST})})()`);
  const TAU = Math.PI * 2;
  const waitFor = async (what, cond, maxS) => { for (let i = 0; i < maxS / 3; i++) { await sleep(3000); const v = await ev(cond); if (v && v !== 'false') return v; const err = await ev(`document.querySelector('.error-banner')?.innerText ?? ''`); if (err) { console.log('ERROR while waiting for', what, err.slice(0, 400)); return null; } } console.log('timeout waiting for', what); return null; };
  for (let k = 0; k < 36; k++) { await pose(-1, 0, 0.5 + TAU * k / 36, false); await frame(`A_${String(k).padStart(3, '0')}.png`); }
  console.log(T(), 'stage 1 frames done');
  console.log(T(), 'rig click', await ev(`(()=>{const b=[...document.querySelectorAll('button')].find(b=>/Rig this model/.test(b.textContent));if(!b)return 'no button';b.click();return b.textContent})()`));
  console.log(T(), 'rig:', await waitFor('rig', `(()=>{const t=[...document.querySelectorAll('.t3d-hint')].map(h=>h.innerText).join(' | ');return /Rigged:/.test(t)&&t.slice(t.indexOf('Rigged:'), t.indexOf('Rigged:') + 90)})()`, 900));
  console.log(T(), 'animate click', await ev(`(()=>{const b=[...document.querySelectorAll('button')].find(b=>/Animate/.test(b.textContent));if(!b)return 'no button';b.click();return b.textContent})()`));
  console.log(T(), 'animate:', await waitFor('animate', `(()=>{const t=[...document.querySelectorAll('.t3d-hint')].map(h=>h.innerText).join(' | ');return /moves on/.test(t)&&t.slice(t.indexOf('moves on') - 3, t.indexOf('moves on') + 60)})()`, 1200));
  await ev(`document.getElementById('t3d-canvas').scrollIntoView({block:'center'})`); await sleep(800);
  for (let k = 0; k < 30; k++) { await pose(-1, 0, 0.5 + TAU * k / 36 * 0.8, true); await frame(`B_${String(k).padStart(3, '0')}.png`); }
  console.log(T(), 'stage 2 frames done');
  const clip = async (name, seconds, fps, tag, yaw0, yaw1) => { const n = Math.round(seconds * fps); for (let k = 0; k < n; k++) { await pose(name, k / fps, yaw0 + (yaw1 - yaw0) * k / n, false); await frame(`${tag}_${String(k).padStart(3, '0')}.png`); } };
  await clip('dance', 3.0, 15, 'C', 0.4, 0.9);
  await clip('wave', 2.0, 15, 'D', 0.5, 0.5);
  await clip('walk', 2.0, 15, 'E', 0.3, 1.2);
  console.log(T(), 'stage 3 frames done');
  ws.close();
} finally { try { process.kill(chrome.pid); } catch {} }
