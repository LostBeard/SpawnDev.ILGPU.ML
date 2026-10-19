// Generate page: draw the same prompt at several sizes in ONE session, save every picture. usage: node gen-sizes.mjs <profile> <outDir> <prompt> <seed> <size,size,...>
import { spawn } from 'node:child_process';
import { writeFileSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';
const [PROFILE, OUT, PROMPT, SEED, SIZES] = process.argv.slice(2);
mkdirSync(OUT, { recursive: true });
const PORT = 9348;
const chrome = spawn('C:/Program Files/Google/Chrome/Application/chrome.exe', [`--user-data-dir=${PROFILE}`, `--remote-debugging-port=${PORT}`, '--enable-unsafe-webgpu', '--no-first-run', '--no-default-browser-check', 'about:blank'], { stdio: 'ignore' });
const sleep = ms => new Promise(r => setTimeout(r, ms));
const t0 = Date.now(); const T = () => ((Date.now() - t0) / 1000).toFixed(0).padStart(5) + 's';
try {
  for (let i = 0; i < 60; i++) { try { await (await fetch(`http://127.0.0.1:${PORT}/json/version`)).json(); break; } catch { await sleep(500); } }
  const tgt = await (await fetch(`http://127.0.0.1:${PORT}/json/new?about:blank`, { method: 'PUT' })).json();
  const ws = new WebSocket(tgt.webSocketDebuggerUrl); await new Promise(r => ws.addEventListener('open', r));
  let id = 0; const pending = new Map();
  ws.addEventListener('message', e => { const m = JSON.parse(e.data); if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); } else if (m.method === 'Runtime.consoleAPICalled') { const t = m.params.args.map(a => a.value ?? a.description ?? '').join(' '); if (/DEVICE|Generate\] Error|rror/.test(t)) console.log('  [console]', t.slice(0, 300)); } });
  const send = (method, params = {}) => new Promise(r => { const i = ++id; pending.set(i, r); ws.send(JSON.stringify({ id: i, method, params })); });
  const ev = async expr => { const r = await send('Runtime.evaluate', { expression: expr, awaitPromise: true, returnByValue: true, timeout: 120000 }); return r.result?.exceptionDetails ? 'EXC ' + JSON.stringify(r.result.exceptionDetails.exception?.description) : r.result?.result?.value; };
  await send('Page.enable'); await send('Runtime.enable');
  await send('Page.addScriptToEvaluateOnNewDocument', { source: `(()=>{const orig=GPUAdapter.prototype.requestDevice;GPUAdapter.prototype.requestDevice=async function(...a){const d=await orig.apply(this,a);d.lost.then(i=>console.log('DEVICE LOST reason='+i.reason+' message='+i.message));return d;}})()` });
  await send('Page.navigate', { url: 'http://localhost:5088/generate' });
  for (let i = 0; i < 120; i++) { if (await ev(`!!document.querySelector('.gen-prompt-input')`) === true) break; await sleep(1000); }
  await ev(`(()=>{const p=document.querySelector('.gen-prompt-input');p.value=${JSON.stringify(PROMPT)};p.dispatchEvent(new Event('input',{bubbles:true}));const s=[...document.querySelectorAll('input.gen-setting-input')].find(x=>x.placeholder==='random');s.value='${SEED}';s.dispatchEvent(new Event('input',{bubbles:true}));s.dispatchEvent(new Event('change',{bubbles:true}));return 1})()`);
  for (const size of SIZES.split(',')) {
    await ev(`(()=>{const s=[...document.querySelectorAll('select.gen-setting-select')].find(x=>[...x.options].some(o=>o.value==='1024'));s.value='${size}';s.dispatchEvent(new Event('change',{bubbles:true}));return s.value})()`);
    await sleep(500);
    await ev(`(()=>{const i=document.querySelector('.gen-result-img'); if(i) i.remove(); return 1})()`);
    const ts = Date.now();
    await ev(`document.querySelector('.gen-btn').click()`);
    let src = null;
    for (let i = 0; i < 300; i++) {
      await sleep(2500);
      src = await ev(`(()=>{const i=document.querySelector('.gen-result-img'); const b=document.querySelector('.gen-btn'); return i && i.src.startsWith('data:') && !b.disabled ? i.src : null})()`);
      if (src) break;
      const err = await ev(`document.querySelector('.error-banner')?.innerText ?? ''`); if (err) { console.log('ERROR', err.slice(0, 300)); break; }
    }
    if (src) { writeFileSync(join(OUT, `fox_${size}.png`), Buffer.from(src.split(',')[1], 'base64')); console.log(T(), size, 'ok in', ((Date.now() - ts) / 1000).toFixed(0), 's'); }
    else console.log(T(), size, 'NO IMAGE');
  }
  ws.close();
} finally { try { process.kill(chrome.pid); } catch {} }
