// DAv3-Small ORT-Web baseline: serves the demo wwwroot, drives ort-comparison.html,
// captures load/cold/warm inference numbers for the WebGPU and Wasm EPs.
// Usage: node dav3-ort-baseline.mjs

import { chromium } from 'playwright';
import http from 'http';
import { readFile } from 'fs/promises';
import { join, extname } from 'path';
import { fileURLToPath } from 'url';

const ROOT = join(fileURLToPath(new URL('.', import.meta.url)), 'SpawnDev.ILGPU.ML.Demo', 'wwwroot');
const PORT = 9999;

const MIME = {
    '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript',
    '.css': 'text/css', '.json': 'application/json', '.wasm': 'application/wasm',
};

const server = http.createServer(async (req, res) => {
    try {
        const path = req.url.split('?')[0];
        const file = join(ROOT, path === '/' ? 'index.html' : path);
        const body = await readFile(file);
        res.writeHead(200, {
            'Content-Type': MIME[extname(file)] || 'application/octet-stream',
            // Cross-origin isolation so the wasm EP gets SharedArrayBuffer threads
            'Cross-Origin-Opener-Policy': 'same-origin',
            'Cross-Origin-Embedder-Policy': 'require-corp',
        });
        res.end(body);
    } catch {
        res.writeHead(404);
        res.end('not found');
    }
});

async function runEP(page, ep, timeoutMs) {
    console.log(`\n=== DAv3-Small (${ep}) ===`);
    // Reset the result div so waitForFunction sees the fresh outcome
    await page.evaluate(() => {
        const el = document.getElementById('dav3-result');
        el.className = 'result';
        el.textContent = 'Waiting...';
    });
    await page.click(`button:has-text("Run DAv3 (${ep === 'webgpu' ? 'WebGPU' : 'Wasm'})")`);
    try {
        await page.waitForFunction(() => {
            const el = document.getElementById('dav3-result');
            return el && (el.classList.contains('pass') || el.classList.contains('fail'));
        }, { timeout: timeoutMs });
    } catch {
        console.log('TIMEOUT waiting for result');
    }
    const text = await page.textContent('#dav3-result');
    console.log(text);
    return text;
}

async function run() {
    await new Promise(r => server.listen(PORT, r));
    console.log(`Serving ${ROOT} at http://localhost:${PORT} (COOP/COEP on)`);

    const browser = await chromium.launch({
        headless: false, // GPU access
        args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan'],
    });
    const page = await browser.newPage();
    page.on('console', msg => {
        if (msg.type() === 'error') console.log('  [BROWSER ERROR]', msg.text());
    });
    page.on('pageerror', e => console.log('  [PAGE ERROR]', e.message));

    await page.goto(`http://localhost:${PORT}/ort-comparison.html`, { waitUntil: 'networkidle' });

    const isolated = await page.evaluate(() => crossOriginIsolated);
    console.log(`crossOriginIsolated: ${isolated}`);

    await runEP(page, 'webgpu', 300000);
    await runEP(page, 'wasm', 300000);

    // Report wasm thread count for context
    const threads = await page.evaluate(() => globalThis.ort?.env?.wasm?.numThreads);
    console.log(`\nort.env.wasm.numThreads: ${threads}`);

    await browser.close();
    server.close();
    console.log('\n=== Done ===');
}

run().catch(e => { console.error(e); process.exit(1); });
