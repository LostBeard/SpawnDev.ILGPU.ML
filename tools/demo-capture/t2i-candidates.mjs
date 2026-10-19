// For each prompt x seed: open the text-to-3d page, draw the picture with SD Turbo, save the source picture the page chose, then reload (skip the slow 3D stage).
// usage: node t2i-candidates.mjs <profileDir> <outDir> <json: [["prompt", seed], ...]>
import { spawn } from 'node:child_process';
import { writeFileSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';

const PROFILE = process.argv[2], OUT = process.argv[3], LIST = JSON.parse(process.argv[4]);
mkdirSync(OUT, { recursive: true });
const PORT = 9338, URL_ = 'http://localhost:5088/text-to-3d';
const chrome = spawn('C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
  [`--user-data-dir=${PROFILE}`, `--remote-debugging-port=${PORT}`, '--enable-unsafe-webgpu', '--enable-features=Vulkan', '--no-first-run', '--no-default-browser-check', 'about:blank'], { stdio: 'ignore' });
const sleep = ms => new Promise(r => setTimeout(r, ms));
const t0 = Date.now(); const T = () => ((Date.now() - t0) / 1000).toFixed(0).padStart(5) + 's';
try {
  for (let i = 0; i < 60; i++) { try { await (await fetch(`http://127.0.0.1:${PORT}/json/version`)).json(); break; } catch { await sleep(500); } }
  const tgt = await (await fetch(`http://127.0.0.1:${PORT}/json/new?about:blank`, { method: 'PUT' })).json();
  const ws = new WebSocket(tgt.webSocketDebuggerUrl); await new Promise(r => ws.addEventListener('open', r));
  let id = 0; const pending = new Map();
  ws.addEventListener('message', e => { const m = JSON.parse(e.data); if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); } else if (m.method === 'Runtime.consoleAPICalled') console.log('  [console.' + m.params.type + ']', m.params.args.map(a => a.value ?? a.description ?? '').join(' ').slice(0, 400)); else if (m.method === 'Runtime.exceptionThrown') console.log('  [exception]', (m.params.exceptionDetails.exception?.description ?? m.params.exceptionDetails.text).slice(0, 600)); });
  const send = (method, params = {}) => new Promise(r => { const i = ++id; pending.set(i, r); ws.send(JSON.stringify({ id: i, method, params })); });
  const ev = async expr => { const r = await send('Runtime.evaluate', { expression: expr, awaitPromise: true, returnByValue: true, timeout: 120000 }); return r.result?.exceptionDetails ? 'EXC ' + JSON.stringify(r.result.exceptionDetails.exception?.description) : r.result?.result?.value; };
  await send('Page.enable'); await send('Runtime.enable');
  await send('Page.addScriptToEvaluateOnNewDocument', { source: `(()=>{const orig=GPUAdapter.prototype.requestDevice;GPUAdapter.prototype.requestDevice=async function(...a){const d=await orig.apply(this,a);d.lost.then(i=>console.log('DEVICE LOST reason='+i.reason+' message='+i.message));d.addEventListener('uncapturederror',e=>console.log('GPU UNCAPTURED ERROR '+e.error.message));console.log('device limits maxBufferSize='+d.limits.maxBufferSize+' maxStorageBufferBindingSize='+d.limits.maxStorageBufferBindingSize);return d;}})()` });
  let n = 0;
  for (const [prompt, seed] of LIST) {
    await send('Page.navigate', { url: URL_ });
    for (let i = 0; i < 120; i++) { if (await ev(`!!document.querySelector('.t3d-prompt')`) === true) break; await sleep(1000); }
    await ev(`(()=>{const p=document.querySelector('.t3d-prompt');p.value=${JSON.stringify(prompt)};p.dispatchEvent(new Event('input',{bubbles:true}));const s=document.querySelector('.t3d-num');s.value='${seed}';s.dispatchEvent(new Event('input',{bubbles:true}));s.dispatchEvent(new Event('change',{bubbles:true}));return 1})()`);
    await sleep(400);
    await ev(`[...document.querySelectorAll('.t3d-btn')].find(b=>/Generate/.test(b.textContent)).click()`);
    let data = null;
    for (let i = 0; i < 400; i++) {                                           // SD Turbo first load downloads 2.5 GB, later draws take seconds
      await sleep(3000);
      data = await ev(`(()=>{const im=document.querySelector('img.t3d-source');const st=(document.querySelector('.t3d-stages')?.innerText??'');return im&&im.src.startsWith('data:')&&/image prep|dinov3|DINOv3/.test(st)?im.src:null})()`);
      if (data) break;
      const err = await ev(`document.querySelector('.error-banner')?.innerText ?? ''`); if (err) { console.log('ERROR', err.slice(0, 200)); break; }
    }
    if (data) {
      const b64 = data.startsWith('data:') ? data.split(',')[1] : await ev(`(async()=>{const b=await (await fetch(document.querySelector('img.t3d-source').src)).blob();const buf=new Uint8Array(await b.arrayBuffer());let s='';for(let i=0;i<buf.length;i+=0x8000)s+=String.fromCharCode.apply(null,buf.subarray(i,i+0x8000));return btoa(s)})()`);
      const name = `cand_${String(++n).padStart(2, '0')}_seed${seed}.png`; writeFileSync(join(OUT, name), Buffer.from(b64, 'base64'));
      console.log(T(), name, '<-', prompt);
    } else console.log(T(), 'no picture for', prompt, seed);
  }
  ws.close();
} finally { try { process.kill(chrome.pid); } catch {} }
