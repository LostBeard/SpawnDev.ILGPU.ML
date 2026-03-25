// Automated browser test: ONNX Runtime Web vs our engine
// Launches Chrome, loads the comparison page, clicks each test, captures results

import { chromium } from 'playwright';

const BASE_URL = 'http://localhost:9999';

async function run() {
    console.log('Launching Chrome with WebGPU...');
    const browser = await chromium.launch({
        headless: false, // Need GPU access
        args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan'],
    });

    const page = await browser.newPage();

    // Capture console output
    page.on('console', msg => {
        if (msg.type() === 'error') console.log('  [BROWSER ERROR]', msg.text());
    });

    console.log('Loading ORT comparison page...');
    await page.goto(`${BASE_URL}/ort-comparison.html`, { waitUntil: 'networkidle' });
    await page.waitForTimeout(2000);

    // Test 1: SqueezeNet (small, should always work)
    console.log('\n=== SqueezeNet (5MB) ===');
    await page.click('button:has-text("Run SqueezeNet")');
    await page.waitForFunction(() => {
        const el = document.getElementById('squeezenet-result');
        return el && (el.classList.contains('pass') || el.classList.contains('fail'));
    }, { timeout: 60000 });
    const sqResult = await page.textContent('#squeezenet-result');
    console.log(sqResult);

    // Test 2: Whisper Encoder (75MB)
    console.log('\n=== Whisper Encoder (75MB) ===');
    await page.click('button:has-text("Run Whisper Encoder")');
    await page.waitForFunction(() => {
        const el = document.getElementById('whisper-result');
        return el && (el.classList.contains('pass') || el.classList.contains('fail'));
    }, { timeout: 120000 });
    const wResult = await page.textContent('#whisper-result');
    console.log(wResult);

    // Test 3: DistilBERT (256MB) — the big one
    console.log('\n=== DistilBERT-SST2 (256MB) ===');
    await page.click('button:has-text("Run DistilBERT")');
    try {
        await page.waitForFunction(() => {
            const el = document.getElementById('distilbert-result');
            return el && (el.classList.contains('pass') || el.classList.contains('fail'));
        }, { timeout: 300000 }); // 5 min timeout for large model
        const dbResult = await page.textContent('#distilbert-result');
        console.log(dbResult);
    } catch (e) {
        const dbResult = await page.textContent('#distilbert-result');
        console.log('TIMEOUT or ERROR:', dbResult);
    }

    console.log('\n=== Done ===');
    await browser.close();
}

run().catch(e => { console.error(e); process.exit(1); });
