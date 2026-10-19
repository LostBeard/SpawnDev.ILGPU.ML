// Drive the text-to-3d page: FLUX.2 klein, draw N, screenshot the picker, optionally build the first (or given) picture in 3D.
// usage: node flux-pick.mjs <profile> <outDir> <prompt> <seed> <pictures> [make=0|1] [model=flux2|sd]
import { spawn } from 'node:child_process';
import { writeFileSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';
const [PROFILE, OUT, PROMPT, SEED, PICS, MAKE = '0', MODEL = 'flux2'] = process.argv.slice(2);
mkdirSync(OUT, { recursive: true });
const PORT = 9343;
const chrome = spawn(process.env.CHROME_EXE || 'C:/Program Files/Google/Chrome/Application/chrome.exe',
  [`--user-data-dir=${PROFILE}`, `--remote-debugging-port=${PORT}`, '--enable-unsafe-webgpu', '--no-first-run', '--no-default-browser-check', '--window-size=1400,1100', 'about:blank'], { stdio: 'ignore' });
const sleep = ms => new Promise(r => setTimeout(r, ms));
const t0 = Date.now(); const T = () => ((Date.now() - t0) / 1000).toFixed(0).padStart(5) + 's';
try {
  for (let i = 0; i < 60; i++) { try { await (await fetch(`http://127.0.0.1:${PORT}/json/version`)).json(); break; } catch { await sleep(500); } }
  const tgt = await (await fetch(`http://127.0.0.1:${PORT}/json/new?about:blank`, { method: 'PUT' })).json();
  const ws = new WebSocket(tgt.webSocketDebuggerUrl); await new Promise(r => ws.addEventListener('open', r));
  let id = 0; const pending = new Map();
  ws.addEventListener('message', e => { const m = JSON.parse(e.data); if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); } else if (m.method === 'Runtime.consoleAPICalled') { const t = m.params.args.map(a => a.value ?? a.description ?? '').join(' '); if (/Flux2|FLUX|DEVICE|error|Error|VAEDIAG|\[Generate/.test(t)) console.log('  [console.' + m.params.type + ']', t.slice(0, 300)); } else if (m.method === 'Runtime.exceptionThrown') console.log('  [exception]', (m.params.exceptionDetails.exception?.description ?? m.params.exceptionDetails.text).slice(0, 500)); });
  const send = (method, params = {}) => new Promise(r => { const i = ++id; pending.set(i, r); ws.send(JSON.stringify({ id: i, method, params })); });
  const ev = async expr => { const r = await send('Runtime.evaluate', { expression: expr, awaitPromise: true, returnByValue: true, timeout: 120000 }); return r.result?.exceptionDetails ? 'EXC ' + JSON.stringify(r.result.exceptionDetails.exception?.description) : r.result?.result?.value; };
  await send('Page.enable'); await send('Runtime.enable');
  await send('Page.addScriptToEvaluateOnNewDocument', { source: `(()=>{const orig=GPUAdapter.prototype.requestDevice;GPUAdapter.prototype.requestDevice=async function(...a){const d=await orig.apply(this,a);d.lost.then(i=>console.log('DEVICE LOST reason='+i.reason+' message='+i.message));return d;}})()` });
  await send('Page.navigate', { url: 'http://localhost:5088/text-to-3d' });
  for (let i = 0; i < 120; i++) { if (await ev(`!!document.querySelector('.t3d-prompt')`) === true) break; await sleep(1000); }
  const setSel = (valueMatch, v) => ev(`(()=>{const s=[...document.querySelectorAll('select.t3d-select')].find(x=>[...x.options].some(o=>o.value==='${valueMatch}'));if(!s)return 'nosel';s.value='${v}';s.dispatchEvent(new Event('change',{bubbles:true}));return s.value})()`);
  console.log('model', await setSel('flux2', MODEL), 'pictures', await setSel('8', PICS));
  await ev(`(()=>{const p=document.querySelector('.t3d-prompt');p.value=${JSON.stringify(PROMPT)};p.dispatchEvent(new Event('input',{bubbles:true}));const s=document.querySelector('.t3d-num');s.value='${SEED}';s.dispatchEvent(new Event('input',{bubbles:true}));s.dispatchEvent(new Event('change',{bubbles:true}));return 1})()`);
  await sleep(500);
  await ev(`[...document.querySelectorAll('.t3d-btn')].find(b=>/Generate/.test(b.textContent)).click()`);
  const want = Number(PICS) > 1 ? Number(PICS) : 1;
  let lastStage = '';
  for (let i = 0; i < 1200; i++) {
    await sleep(3000);
    const st = JSON.parse(await ev(`JSON.stringify({n:document.querySelectorAll('.t3d-pick-item').length,stage:document.querySelector('.t3d-stage.running')?.innerText?.replace(/\s+/g,' ')??'',err:document.querySelector('.error-banner')?.innerText??'',glb:!!(document.querySelector('.t3d-result-info')?.innerText??'').includes('GLB')})`));
    if (st.stage !== lastStage) { console.log(T(), 'stage:', st.stage, '| pictures', st.n); lastStage = st.stage; }
    if (st.err) { console.log('ERROR', st.err.slice(0, 600)); break; }
    if (want > 1 && st.n >= want && !st.stage) { console.log(T(), 'all pictures drawn'); break; }
    if (want === 1 && st.glb) break;
  }
  const shot = async name => { const r = await send('Page.captureScreenshot', { format: 'png' }); writeFileSync(join(OUT, name), Buffer.from(r.result.data, 'base64')); };
  const urls = JSON.parse(await ev(`JSON.stringify([...document.querySelectorAll('.t3d-pick-item img')].map(i=>i.src))`));
  urls.forEach((u, k) => { if (u.startsWith('data:')) writeFileSync(join(OUT, `cand_${k}.png`), Buffer.from(u.split(',')[1], 'base64')); });
  await ev(`document.querySelector('.t3d-pick')?.scrollIntoView({block:'center'})`); await sleep(500);
  await shot('picker.png');
  if (MAKE === '1' && want > 1) {
    await ev(`document.querySelector('.t3d-pick-item button').click()`);
    for (let i = 0; i < 1200; i++) {
      await sleep(3000);
      const st = JSON.parse(await ev(`JSON.stringify({stage:document.querySelector('.t3d-stage.running')?.innerText?.replace(/\s+/g,' ')??'',err:document.querySelector('.error-banner')?.innerText??'',glb:(document.querySelector('.t3d-result-info')?.innerText??'').includes('GLB')})`));
      if (st.stage !== lastStage) { console.log(T(), 'stage:', st.stage); lastStage = st.stage; }
      if (st.err) { console.log('ERROR', st.err.slice(0, 600)); break; }
      if (st.glb) { console.log(T(), 'GLB ready'); break; }
    }
    await ev(`document.getElementById('t3d-canvas').scrollIntoView({block:'center'})`); await sleep(1500);
    await shot('result.png');
  }
  ws.close();
} finally { try { process.kill(chrome.pid); } catch {} }
