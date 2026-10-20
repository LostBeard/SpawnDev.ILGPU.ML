// Audit every demo route against a served build: does it work, and does EVERY model request go to the models host?
//
//   node tools/serve-demo.mjs <publish/wwwroot> 5071            (static server with COOP/COEP, as PMT uses)
//   node tools/audit-demo-models.mjs <baseUrl> <outDir> [route,route,...]
//   e.g. node tools/audit-demo-models.mjs http://127.0.0.1:5071 _scratch/demo-audit /sentiment,/embeddings
//
// Each route starts with the site's storage and the HTTP cache CLEARED (Storage.clearDataForOrigin): empty OPFS, so every model really
// downloads and every request is visible. (Not an isolated browser context: that has an incognito-sized ~1 GB quota.) A real Chrome (own profile, own CDP port, killed by PID) with real WebGPU.
// What is recorded per route: every network request (host, status, bytes, protocol, redirects), every failed request (CORS, blocked, reset),
// console errors, uncaught exceptions, a screenshot, and the scenario's own evidence (the rendered result).
//
// A route PASSES only when ALL hold:
//   1. its scenario produced a real result (a scenario that returns nothing, or throws, FAILS: a page that merely mounts proves nothing),
//   2. zero requests to hub.spawndev.com (the hub no longer serves models),
//   3. zero requests to huggingface.co beyond a route's explicit allowance (a request there is the FALLBACK working, i.e. a file is missing from
//      the models host: it is reported by URL so it can be mirrored), and
//   4. zero failed requests and zero console errors/exceptions that the scenario did not declare as expected.
//
// Scenarios live in tools/audit-scenarios/*.mjs: { '/route': { run: async (h) => evidence, allowHuggingFace?: bool, expectConsole?: RegExp[] } }.
// A route with no scenario only gets the load check (mounts, no errors, no hub requests) and is reported as LOAD-ONLY, never as a pass of the demo.
import { spawn, execFileSync } from 'node:child_process';
import { mkdirSync, writeFileSync, existsSync, rmSync, readdirSync } from 'node:fs';
import { join, resolve } from 'node:path';
import os from 'node:os';
import { pathToFileURL } from 'node:url';

const [BASE_RAW, OUT_RAW, ROUTES_ARG] = process.argv.slice(2);
if (!BASE_RAW || !OUT_RAW) { console.error('usage: node tools/audit-demo-models.mjs <baseUrl> <outDir> [route,route,...]'); process.exit(2); }
const BASE = BASE_RAW.replace(/\/$/, '');
const OUT = resolve(OUT_RAW);
mkdirSync(OUT, { recursive: true });
const PORT = Number(process.env.AUDIT_CDP_PORT || 9355);
const CHROME = process.env.CHROME_EXE || 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const PROFILE = process.env.AUDIT_PROFILE || join(os.tmpdir(), 'spawndev-ml-audit-profile');
const sleep = ms => new Promise(r => setTimeout(r, ms));

// Scenarios: every tools/audit-scenarios/*.mjs (one file per group of demos; _routes.mjs holds the route list and the contract), or the one file
// named by AUDIT_SCENARIOS (used to red-check this driver).
async function loadScenarios() {
  if (process.env.AUDIT_SCENARIOS) { const m = await import(pathToFileURL(resolve(process.env.AUDIT_SCENARIOS)).href); return { scenarios: m.scenarios, defaultRoutes: m.defaultRoutes }; }
  const dir = new URL('./audit-scenarios/', import.meta.url);
  const all = {}; let defaultRoutes = [];
  for (const f of readdirSync(dir).filter(f => f.endsWith('.mjs')).sort()) {
    const m = await import(new URL(f, dir).href);
    if (f === '_routes.mjs') defaultRoutes = m.defaultRoutes; else Object.assign(all, m.scenarios);
  }
  return { scenarios: all, defaultRoutes };
}
const { scenarios, defaultRoutes } = await loadScenarios();
// Git Bash rewrites an argument that starts with "/" into a Windows path; strip it back to the route.
const routes = (ROUTES_ARG ? ROUTES_ARG.split(',') : defaultRoutes)
  .map(r => r.trim()).filter(Boolean)
  .map(r => '/' + r.replace(/\\/g, '/').replace(/^.*?\/(?=[^/]*$)/, '').replace(/^\//, ''))
  .map(r => r === '/' || r === '/home' ? '/' : r);

rmSync(PROFILE, { recursive: true, force: true });
const chrome = spawn(CHROME, [`--user-data-dir=${PROFILE}`, `--remote-debugging-port=${PORT}`, '--enable-unsafe-webgpu', '--no-first-run', '--no-default-browser-check',
  '--hide-crash-restore-bubble', '--window-size=1500,1000',
  // microphone / webcam demos: AUDIT_CHROME_ARGS='--use-fake-device-for-media-stream --use-fake-ui-for-media-stream --use-file-for-fake-audio-capture=C:/x/clip.wav'
  ...(process.env.AUDIT_CHROME_ARGS ? process.env.AUDIT_CHROME_ARGS.split(' ').filter(Boolean) : []),
  'about:blank'], { stdio: 'ignore' });
console.log('chrome pid', chrome.pid, '(killed by PID at the end)');
const killChrome = () => { try { execFileSync('taskkill', ['/PID', String(chrome.pid), '/T', '/F'], { stdio: 'ignore' }); } catch { /* already gone */ } };
process.on('exit', killChrome);

async function cdpJson(path, init) { const r = await fetch(`http://127.0.0.1:${PORT}${path}`, init); return r.json(); }
for (let i = 0; i < 60; i++) { try { await cdpJson('/json/version'); break; } catch { await sleep(500); } }
const version = await cdpJson('/json/version');
console.log('browser', version.Browser);

function connect(wsUrl) {
  const ws = new WebSocket(wsUrl);
  const pending = new Map(); const listeners = []; let id = 0;
  const opened = new Promise(r => ws.addEventListener('open', r));
  ws.addEventListener('message', e => {
    const m = JSON.parse(e.data);
    if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); } else listeners.forEach(l => l(m));
  });
  const send = (method, params = {}) => new Promise((res, rej) => { const i = ++id; pending.set(i, m => m.error ? rej(new Error(`${method}: ${m.error.message}`)) : res(m.result)); ws.send(JSON.stringify({ id: i, method, params })); });
  return { opened, send, on: l => listeners.push(l), close: () => ws.close() };
}

const hostOf = u => { try { return new URL(u).host; } catch { return ''; } };
const summary = [];

for (const route of routes) {
  const spec = scenarios[route] ?? null;
  const t0 = Date.now();
  const res = { route, scenario: spec ? 'yes' : 'LOAD-ONLY', status: 'ERROR', evidence: null, error: null, seconds: 0 };
  const requests = new Map(); const failed = []; const consoleErrors = []; const exceptions = []; const info = [];
  let ctxId = null, targetId = null, page = null, browser = null;
  try {
    browser = connect(version.webSocketDebuggerUrl); await browser.opened;
    // ⚠️ The DEFAULT browser context, not Target.createBrowserContext: an isolated context behaves like an incognito window and gets a small
    // storage quota (~1 GB), so a multi-GB model fails with "would exceed its storage quota" (found 2026-10-20 on /generate). A clean slate is
    // guaranteed instead by clearing the site's storage and HTTP cache before the route (below).
    targetId = (await browser.send('Target.createTarget', { url: 'about:blank' })).targetId;
    page = connect(`ws://127.0.0.1:${PORT}/devtools/page/${targetId}`); await page.opened;
    page.on(m => {
      const p = m.params;
      if (m.method === 'Network.requestWillBeSent') {
        const r = requests.get(p.requestId) ?? { urls: [], redirects: 0 };
        if (p.redirectResponse) r.redirects++;
        r.urls.push(p.request.url); r.url = p.request.url; r.method = p.request.method; r.type = p.type; requests.set(p.requestId, r);
      } else if (m.method === 'Network.responseReceived') {
        const r = requests.get(p.requestId); if (r) { r.status = p.response.status; r.protocol = p.response.protocol; r.fromCache = p.response.fromDiskCache || p.response.fromServiceWorker; r.mime = p.response.mimeType; }
      } else if (m.method === 'Network.loadingFinished') {
        const r = requests.get(p.requestId); if (r) { r.bytes = p.encodedDataLength; r.done = true; }
      } else if (m.method === 'Network.loadingFailed') {
        const r = requests.get(p.requestId);
        if (!p.canceled) failed.push({ url: r?.url ?? '?', error: p.errorText, cors: p.corsErrorStatus?.corsError, blocked: p.blockedReason, type: p.type });
      } else if (m.method === 'Runtime.consoleAPICalled') {
        const text = p.args.map(a => a.value ?? a.description ?? '').join(' ');
        if (p.type === 'error') consoleErrors.push(text.slice(0, 400)); else if (/\[VRAM\]|device lost|fallback/i.test(text)) info.push(text.slice(0, 300));
      } else if (m.method === 'Runtime.exceptionThrown') {
        exceptions.push((p.exceptionDetails.exception?.description ?? p.exceptionDetails.text ?? '').slice(0, 500));
      } else if (m.method === 'Log.entryAdded' && p.entry.level === 'error') {
        consoleErrors.push(`[${p.entry.source}] ${p.entry.text} ${p.entry.url ?? ''}`.slice(0, 400));
      }
    });
    await page.send('Storage.clearDataForOrigin', { origin: new URL(BASE).origin, storageTypes: 'all' });   // OPFS, IndexedDB, caches, service workers: every model downloads
    await page.send('Network.enable'); await page.send('Network.clearBrowserCache'); await page.send('Page.enable'); await page.send('Runtime.enable'); await page.send('Log.enable'); await page.send('DOM.enable');
    // a lost WebGPU device shows up in the page only as a vague error: log the reason
    await page.send('Page.addScriptToEvaluateOnNewDocument', { source: `(()=>{const o=GPUAdapter.prototype.requestDevice;GPUAdapter.prototype.requestDevice=async function(...a){const d=await o.apply(this,a);d.lost.then(l=>console.log('device lost: '+l.reason+' '+l.message));return d;};})();` });

    const ev = async expr => {
      const r = await page.send('Runtime.evaluate', { expression: expr, awaitPromise: true, returnByValue: true, timeout: 120000 });
      if (r.exceptionDetails) throw new Error('page eval failed: ' + (r.exceptionDetails.exception?.description ?? r.exceptionDetails.text).slice(0, 300));
      return r.result.value;
    };
    const h = {
      ev, sleep, base: BASE, info,
      log: m => console.log('   |', m),
      async waitFor(expr, timeoutMs = 120000, label = expr) {
        const end = Date.now() + timeoutMs; let last;
        while (Date.now() < end) { try { last = await ev(expr); if (last) return last; } catch (e) { last = e.message; } await sleep(500); }
        throw new Error(`timed out (${timeoutMs / 1000}s) waiting for: ${label}`);
      },
      async click(sel) { const ok = await ev(`(()=>{const e=document.querySelector(${JSON.stringify(sel)});if(!e)return false;e.click();return true;})()`); if (!ok) throw new Error('no element: ' + sel); },
      async clickText(sel, re) { const ok = await ev(`(()=>{const e=[...document.querySelectorAll(${JSON.stringify(sel)})].find(x=>${re}.test(x.textContent));if(!e)return false;e.click();return true;})()`); if (!ok) throw new Error(`no ${sel} matching ${re}`); },
      async type(sel, text) { const ok = await ev(`(()=>{const e=document.querySelector(${JSON.stringify(sel)});if(!e)return false;e.focus();e.value=${JSON.stringify(text)};e.dispatchEvent(new Event('input',{bubbles:true}));e.dispatchEvent(new Event('change',{bubbles:true}));return true;})()`); if (!ok) throw new Error('no input: ' + sel); },
      // Sample the page every intervalMs until untilExpr is truthy (returns every sample): this is how a scenario proves something HAPPENED OVER TIME
      // (a bar that moved, a caption that grew), not just that the end state is right. An .error-banner on the page throws at once.
      async observe(sampleExpr, untilExpr, timeoutMs = 600000, intervalMs = 300) {
        const samples = []; const end = Date.now() + timeoutMs;
        while (Date.now() < end) {
          const err = await ev(`document.querySelector('.error-banner')?.innerText ?? ''`);
          if (err) throw new Error('the page showed an error: ' + err.slice(0, 400));
          const v = await ev(`(${sampleExpr})`); samples.push(typeof v === 'string' ? JSON.parse(v) : v);
          if (await ev(`!!(${untilExpr})`)) return samples;
          await sleep(intervalMs);
        }
        throw new Error(`timed out (${timeoutMs / 1000}s) observing until: ${untilExpr}`);
      },
      async selectValue(sel, value) { const ok = await ev(`(()=>{const s=[...document.querySelectorAll(${JSON.stringify(sel)})].find(x=>[...x.options].some(o=>o.value===${JSON.stringify(value)}));if(!s)return false;s.value=${JSON.stringify(value)};s.dispatchEvent(new Event('change',{bubbles:true}));return true;})()`); if (!ok) throw new Error(`no <select> ${sel} with option ${value}`); },
      async typeNth(sel, n, text) { const ok = await ev(`(()=>{const e=document.querySelectorAll(${JSON.stringify(sel)})[${n}];if(!e)return false;e.focus();e.value=${JSON.stringify(text)};e.dispatchEvent(new Event('input',{bubbles:true}));e.dispatchEvent(new Event('change',{bubbles:true}));return true;})()`); if (!ok) throw new Error(`no input #${n}: ${sel}`); },
      async count(sel) { return ev(`document.querySelectorAll(${JSON.stringify(sel)}).length`); },
      async text(sel) { return ev(`(document.querySelector(${JSON.stringify(sel)})?.innerText ?? '').replace(/\\s+/g,' ').trim()`); },
      async upload(sel, file) { const doc = await page.send('DOM.getDocument', { depth: 0 }); const q = await page.send('DOM.querySelector', { nodeId: doc.root.nodeId, selector: sel }); if (!q.nodeId) throw new Error('no file input: ' + sel); await page.send('DOM.setFileInputFiles', { nodeId: q.nodeId, files: [resolve(file)] }); },
      async shot(name) { const r = await page.send('Page.captureScreenshot', { format: 'png' }); const f = join(OUT, `${route.replace(/\W+/g, '_') || 'home'}_${name}.png`); writeFileSync(f, Buffer.from(r.data, 'base64')); return f; },
      async mounted() { await h.waitFor(`(document.querySelector('#app')?.children.length ?? 0) > 0 && document.body.innerText.trim().length > 40`, 120000, 'Blazor mounted'); },
    };

    console.log(`\n--- ${route} ${spec ? '' : '(load-only: no scenario yet)'}`);
    await page.send('Page.navigate', { url: BASE + route.split('#')[0] });   // '/generate#flux3' = the /generate page, the 'flux3' scenario variant
    await h.mounted();
    await sleep(1500);
    if (spec?.run) res.evidence = await spec.run(h);
    else res.evidence = (await h.text('body')).slice(0, 200);
    res.shot = await h.shot('end');
    res.status = 'RAN';
  } catch (e) {
    res.error = String(e.message ?? e).slice(0, 600); res.status = 'FAIL';
    try { res.shot = await (async () => { const r = await page.send('Page.captureScreenshot', { format: 'png' }); const f = join(OUT, `${route.replace(/\W+/g, '_') || 'home'}_fail.png`); writeFileSync(f, Buffer.from(r.data, 'base64')); return f; })(); } catch { /* page gone */ }
  }

  // ---- the audit of the traffic ----
  const byHost = {};
  for (const r of requests.values()) {
    if (!r.url || /^(data|blob|chrome-extension|about):/.test(r.url)) continue;
    const host = hostOf(r.url); if (!host) continue;
    const k = (byHost[host] ??= { requests: 0, bytes: 0, statuses: {}, redirected: 0 });
    k.requests++; k.bytes += r.bytes ?? 0; k.statuses[r.status ?? '?'] = (k.statuses[r.status ?? '?'] ?? 0) + 1; k.redirected += r.redirects ?? 0;
  }
  const to = h => [...requests.values()].filter(r => r.url && hostOf(r.url) === h || (r.urls ?? []).some(u => hostOf(u) === h));
  const hubReqs = to('hub.spawndev.com:44365').concat(to('hub.spawndev.com'));
  const hfReqs = to('huggingface.co').concat([...requests.values()].filter(r => hostOf(r.url ?? '').endsWith('.hf.co') && !to('huggingface.co').includes(r)));
  const modelsReqs = to('models.spawndev.com');
  res.hosts = byHost;
  res.modelsRequests = modelsReqs.length; res.modelsBytes = modelsReqs.reduce((a, r) => a + (r.bytes ?? 0), 0);
  res.hubRequests = hubReqs.map(r => `${r.status ?? '?'} ${r.url}`);
  res.huggingFaceRequests = [...new Set(hfReqs.map(r => `${r.status ?? '?'} ${r.url}`))];
  res.failedRequests = failed;
  const expected = spec?.expectConsole ?? [];
  const isExpected = t => expected.some(re => re.test(t));
  res.consoleErrors = consoleErrors.filter(t => !isExpected(t)); res.exceptions = exceptions.filter(t => !isExpected(t));
  res.info = info.length <= 24 ? info : [...info.slice(0, 8), `... ${info.length - 20} more ...`, ...info.slice(-12)];   // the END matters: a 'device lost: <reason>' line comes last

  const problems = [];
  if (!spec) problems.push('no scenario: only the load was checked');
  if (res.status === 'FAIL') problems.push('scenario failed: ' + res.error);
  if (res.hubRequests.length) problems.push(`${res.hubRequests.length} request(s) to hub.spawndev.com`);
  if (res.huggingFaceRequests.length && !spec?.allowHuggingFace) problems.push(`${res.huggingFaceRequests.length} request(s) to huggingface.co (a file missing from the models host, served by the fallback)`);
  if (res.failedRequests.length) problems.push(`${res.failedRequests.length} failed request(s)`);
  if (res.consoleErrors.length) problems.push(`${res.consoleErrors.length} console error(s)`);
  if (res.exceptions.length) problems.push(`${res.exceptions.length} exception(s)`);
  res.problems = problems;
  res.verdict = !spec ? 'LOAD-ONLY' : problems.length === 0 ? 'PASS' : 'FAIL';
  res.seconds = Math.round((Date.now() - t0) / 1000);
  summary.push(res);
  console.log(`   ${res.verdict}  ${res.seconds}s  models.spawndev.com: ${res.modelsRequests} req ${(res.modelsBytes / 1048576).toFixed(1)} MB | hub ${res.hubRequests.length} | huggingface.co ${res.huggingFaceRequests.length} | failed ${res.failedRequests.length} | console errors ${res.consoleErrors.length}`);
  if (res.evidence) console.log('   evidence:', typeof res.evidence === 'string' ? res.evidence.slice(0, 220) : JSON.stringify(res.evidence).slice(0, 220));
  for (const p of problems) console.log('   !', p.slice(0, 300));
  res.huggingFaceRequests.slice(0, 6).forEach(u => console.log('     hf  ', u));
  res.failedRequests.slice(0, 4).forEach(f => console.log('     fail', JSON.stringify(f).slice(0, 220)));
  res.consoleErrors.slice(0, 3).forEach(t => console.log('     err ', t.slice(0, 220)));
  res.exceptions.slice(0, 2).forEach(t => console.log('     exc ', t.slice(0, 220)));

  writeFileSync(join(OUT, `${route.replace(/\W+/g, '_') || 'home'}.json`), JSON.stringify(res, null, 1));
  try { await page?.send('Storage.clearDataForOrigin', { origin: new URL(BASE).origin, storageTypes: 'all' }); } catch { /* page gone */ }   // free the disk the models took
  try { page?.close(); await browser?.send('Target.closeTarget', { targetId }); } catch { /* target already gone */ }
  browser?.close();
}

writeFileSync(join(OUT, '_summary.json'), JSON.stringify(summary.map(s => ({ route: s.route, verdict: s.verdict, seconds: s.seconds, modelsRequests: s.modelsRequests, modelsMB: +(s.modelsBytes / 1048576).toFixed(1), hub: s.hubRequests.length, huggingFace: s.huggingFaceRequests.length, failed: s.failedRequests.length, consoleErrors: s.consoleErrors.length, problems: s.problems })), null, 1));
console.log('\n=== summary ===');
for (const s of summary) console.log(`${s.verdict.padEnd(9)} ${s.route.padEnd(16)} ${String(s.seconds).padStart(4)}s  models ${String(s.modelsRequests).padStart(4)} req ${(s.modelsBytes / 1048576).toFixed(1).padStart(8)} MB  hub ${s.hubRequests.length}  hf ${s.huggingFaceRequests.length}  failed ${s.failedRequests.length}`);
const bad = summary.filter(s => s.verdict === 'FAIL').length, loadOnly = summary.filter(s => s.verdict === 'LOAD-ONLY').length;
console.log(`${summary.length} routes: ${summary.filter(s => s.verdict === 'PASS').length} PASS, ${bad} FAIL, ${loadOnly} LOAD-ONLY (a load-only route is NOT a pass)`);
killChrome();
process.exit(bad);
