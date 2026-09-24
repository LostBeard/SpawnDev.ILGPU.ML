// DAv3 through Transformers.js in real Chrome, on the ORT-CPU reference inputs.
//
//   node tools/dav3/dav3-tjs.mjs [--device webgpu|wasm] [--cases a,b,...] [--tjs 4.3.0]
//
// Serves (loopback, COOP/COEP so the wasm EP gets threads):
//   /            tools/dav3/ (dav3-tjs.html)
//   /models/     the SAME model bytes as the hub + the ORT reference (_mldump/models, sha-checked by
//                dav3_reference.py) laid out the way Transformers.js' localModelPath expects
//   /refs/       test-refs/dav3 (inputs + manifest from dav3_reference.py)
//   POST /out/   -> _mldump/test-out/dav3/<path>  (read by dav3_compare.py)
// Launches SYSTEM Chrome (channel 'chrome'): Playwright's bundled Chromium only exposes the SwiftShader
// software adapter, and the page logs the adapter so a wrong one is visible in the results.
import { chromium } from 'playwright';
import http from 'http';
import { readFile, writeFile, mkdir, stat } from 'fs/promises';
import { createReadStream } from 'fs';
import { join, extname, dirname, resolve, sep } from 'path';
import { fileURLToPath } from 'url';

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO = resolve(HERE, '..', '..');
const MODEL_DIR = resolve(REPO, '_mldump', 'models', 'depth-anything-v3-small');
const REFS = join(REPO, 'SpawnDev.ILGPU.ML.Demo', 'wwwroot', 'test-refs', 'dav3');
const OUT = resolve(REPO, '_mldump', 'test-out', 'dav3');
const PORT = 9311;

const arg = (k, d) => { const i = process.argv.indexOf(k); return i > 0 ? process.argv[i + 1] : d; };
const device = arg('--device', 'webgpu');
const tjs = arg('--tjs', '4.3.0');
const manifest = JSON.parse(await readFile(join(REFS, 'manifest.json'), 'utf8'));
const cases = arg('--cases', Object.keys(manifest.cases).join(','));

// The export's config.json (model_type depth_anything + use_external_data_format). Written next to the
// weights so the model folder is self-contained for Transformers.js.
const CONFIG = { model_type: 'depth_anything', 'transformers.js_config': { dtype: 'fp32', use_external_data_format: true } };
await writeFile(join(MODEL_DIR, 'config.json'), JSON.stringify(CONFIG, null, 2));

const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.json': 'application/json' };
function route(url) {
    const p = decodeURIComponent(url.split('?')[0]);
    const m = p.match(/^\/models\/onnx-community\/depth-anything-v3-small\/(.+)$/);
    if (m) return join(MODEL_DIR, m[1].replace(/^onnx\//, ''));
    if (p.startsWith('/refs/')) return join(REFS, p.slice(6));
    return join(HERE, p === '/' ? 'dav3-tjs.html' : p.slice(1));
}

const server = http.createServer(async (req, res) => {
    const headers = { 'Cross-Origin-Opener-Policy': 'same-origin', 'Cross-Origin-Embedder-Policy': 'require-corp' };
    try {
        if (req.method === 'POST' && req.url.startsWith('/out/')) {
            const dest = resolve(OUT, decodeURIComponent(req.url.slice(5)));
            if (!dest.startsWith(OUT + sep)) { res.writeHead(400, headers); return res.end('escapes out'); }
            const chunks = [];
            for await (const c of req) chunks.push(c);
            await mkdir(dirname(dest), { recursive: true });
            await writeFile(dest, Buffer.concat(chunks));
            res.writeHead(200, headers); return res.end('ok');
        }
        const file = route(req.url);
        const st = await stat(file);
        res.writeHead(200, { ...headers, 'Content-Type': MIME[extname(file)] || 'application/octet-stream', 'Content-Length': st.size });
        createReadStream(file).pipe(res);
    } catch {
        res.writeHead(404, headers); res.end('not found');
    }
});
await new Promise(r => server.listen(PORT, '127.0.0.1', r));

const browser = await chromium.launch({
    channel: 'chrome', headless: false,
    args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan', `--remote-debugging-port=0`],
});
const page = await browser.newPage();
page.on('console', m => { if (m.text().startsWith('[TJS]')) console.log(m.text()); });
page.on('pageerror', e => console.log('[PAGE ERROR]', e.message));
await page.goto(`http://127.0.0.1:${PORT}/?device=${device}&tjs=${tjs}&cases=${cases}`);
await page.waitForFunction(() => window.__done === true, null, { timeout: 60 * 60 * 1000 });
const results = await page.evaluate(() => ({ adapter: window.__adapter, error: window.__error, results: window.__results }));
await mkdir(OUT, { recursive: true });
await writeFile(join(OUT, `tjs-${device}.json`), JSON.stringify({ tjs, device, ...results }, null, 1));
console.log(`wrote ${join(OUT, `tjs-${device}.json`)}`);
await browser.close();
server.close();
