// Scenarios for the text, audio and tooling demos: /text-gen /whisper /tts /ai-chat /inspector /models /cache /benchmark /train /snake /tests
// and the doc pages / /getting-started /pipelines (see _routes.mjs for the scenario contract).
//
// ⚠️ Every scenario asserts on the RENDERED RESULT (generated text, a transcription, the status line's audio duration, a reply bubble), never on a
// console line. The four doc/meta pages (/, /getting-started, /pipelines, /tests) are LOAD-ONLY by design (they have no model and no action): they
// assert their key content renders, and the audit report must call them LOAD-ONLY rather than a pass of a demo.
//
// The microphone routes (/whisper, /tts clone) need Chrome's fake audio device; run the driver with
//   AUDIT_CHROME_ARGS='--use-fake-device-for-media-stream --use-fake-ui-for-media-stream --use-file-for-fake-audio-capture=<abs path to librivox-public-domain.wav>'
// (the clip is 4 s, 16 kHz mono, and loops; it says "All LibriVox recordings are in the public domain").

import { fileURLToPath } from 'node:url';

const CLIP = fileURLToPath(new URL('../../SpawnDev.ILGPU.ML.Demo/wwwroot/test-audio/librivox-public-domain.wav', import.meta.url));
const sleep = ms => new Promise(r => setTimeout(r, ms));

// A synthetic pointer stroke on a canvas (Blazor's @onmousedown/@onmousemove/@onmouseup read offsetX/offsetY, which Chrome derives from clientX/clientY).
// `points` are canvas-relative CSS pixels.
const strokeJs = (sel, points) => `(async()=>{
  const c=document.querySelector(${JSON.stringify(sel)}); if(!c) return false;
  const r=c.getBoundingClientRect(); const pts=${JSON.stringify(points)};
  const ev=(t,p)=>c.dispatchEvent(new MouseEvent(t,{bubbles:true,cancelable:true,clientX:r.left+p[0],clientY:r.top+p[1],buttons:1,view:window}));
  ev('mousedown',pts[0]); for(const p of pts){ev('mousemove',p); await new Promise(x=>setTimeout(x,4));} ev('mouseup',pts[pts.length-1]); return true;})()`;
const circlePts = (cx, cy, rad) => Array.from({ length: 41 }, (_, i) => [cx + rad * Math.cos(i / 40 * 2 * Math.PI), cy + rad * Math.sin(i / 40 * 2 * Math.PI)]);
const squarePts = (cx, cy, h) => [[cx - h, cy - h], [cx + h, cy - h], [cx + h, cy + h], [cx - h, cy + h], [cx - h, cy - h]]
  .flatMap((p, i, a) => i === 0 ? [p] : Array.from({ length: 10 }, (_, k) => [a[i - 1][0] + (p[0] - a[i - 1][0]) * (k + 1) / 10, a[i - 1][1] + (p[1] - a[i - 1][1]) * (k + 1) / 10]));

// Chrome's fake mic loops a 4 s clip: record long enough for at least one whole sentence to be inside the recording.
// Blazor paints the page a moment after #app has content: wait for the page's own key element before driving it.
const ready = (h, sel) => h.waitFor(`!!document.querySelector(${JSON.stringify(sel)})`, 120000, 'page element ' + sel);

const FAKE_MIC_NOTE = 'needs AUDIT_CHROME_ARGS with --use-file-for-fake-audio-capture=librivox-public-domain.wav';

export const scenarios = {
  // ---------------------------------------------------------------- TEXT
  '/text-gen': {
    // DistilGPT-2 (onnx/decoder_model.onnx + tokenizer.json) through the models host; greedy so the result is deterministic.
    run: async h => {
      await ready(h, '.textgen-textarea');
      await h.type('.textgen-textarea', 'The capital of France is');
      await h.ev(`(()=>{const s=document.querySelector('.setting-select'); s.value='greedy'; s.dispatchEvent(new Event('change',{bubbles:true})); return s.value;})()`);
      await h.type('.setting-input', '30');   // first .setting-input = Max tokens
      await h.click('.run-btn');
      // the text streams in token by token: wait for the run to FINISH (button back to "Generate", enabled) with output present, not for the first token
      const text = await h.waitFor(`(()=>{const b=document.querySelector('.run-btn'); return b && !b.disabled && b.innerText.trim()==='Generate' ? document.querySelector('.output-generated')?.innerText?.trim() : '';})()`, 600000, 'finished generation (.output-generated + enabled Generate)');
      const stats = await h.text('.output-stats');
      if (text.length < 10 || !/[a-z]{3}/i.test(text)) throw new Error('no real generated text rendered: ' + JSON.stringify(text));
      if (!/[1-9]\d* tokens/.test(stats)) throw new Error('no token count rendered: ' + stats);
      return `generated: ${JSON.stringify(text.slice(0, 120))} | ${stats}`;
    },
  },

  // ---------------------------------------------------------------- AUDIO
  '/whisper': {
    // Microphone -> Whisper Tiny. The fake mic plays the LibriVox clip; the transcription must contain "public domain".
    // If the page cannot open the microphone (a product bug, see the thrown message) the scenario STILL exercises the Upload Audio path with the same
    // clip so the model hosting is verified, but the route FAILS: the mic path is what the demo advertises.
    run: async h => {
      await ready(h, '.mode-btn');
      await h.clickText('.mode-btn', /Microphone/);
      await h.waitFor(`!!document.querySelector('.mic-btn')`, 15000, 'mic button');
      await h.click('.mic-btn');                                     // Start Recording (the model downloads WHILE recording)
      const started = await h.waitFor(`document.querySelector('.mic-btn.recording') ? 'recording' : (/^Error/.test(document.querySelector('.transcription-text')?.innerText||'') ? document.querySelector('.transcription-text').innerText : '')`, 30000, `recording started (${FAKE_MIC_NOTE})`);
      if (started !== 'recording') {
        // fallback: same clip through Upload Audio, so the models-host evidence is still collected
        await h.clickText('.mode-btn', /Upload Audio/);
        await h.waitFor(`!!document.querySelector('input[type=file]')`, 15000, 'file input');
        await h.upload('input[type=file]', CLIP);
        const t = await h.waitFor(`(()=>{const x=document.querySelector('.transcription-text')?.innerText?.trim()||''; return /could not open the microphone/.test(x) ? '' : x;})()`, 600000, 'upload transcription');
        throw new Error(`MIC PATH BROKEN: page says "${started}". (Upload Audio path with the same clip transcribed: ${JSON.stringify(t)})`);
      }
      await sleep(9000);                                              // > 2 loops of the 4 s clip
      await h.click('.mic-btn.recording');                           // Stop -> transcribe (waits for the model if it is still loading)
      const t = await h.waitFor(`document.querySelector('.transcription-text')?.innerText?.trim()`, 600000, '.transcription-text');
      if (/^Error/i.test(t)) throw new Error('the page reported: ' + t);
      if (!/public\s+domain/i.test(t)) throw new Error('transcription does not contain "public domain": ' + JSON.stringify(t));
      const stats = await h.text('.transcription-stats');
      return `transcription: ${JSON.stringify(t)} | ${stats}`;
    },
  },

  '/tts': {
    // (1) Kokoro built-in voice, (2) ZipVoice one-shot clone: reference recorded from the fake mic, saved, then spoken. The clone path loads
    // vocos_24khz.onnx from the models host (the mirrored sherpa-onnx release member): that request is checked below.
    run: async h => {
      const evidence = [];
      await ready(h, '.tts-textarea');
      // AUDIT_TTS_ONLY=clone skips the Kokoro half (used to reach the ZipVoice vocoder when Kokoro synthesis stalls); the result is then not a full pass of the route.
      // The vocoder fetch is observed from the page: its URL and status are added to the evidence.
      await h.ev(`(()=>{window.__voc=[]; const o=window.fetch; window.fetch=async function(i,init){const u=String(i?.url??i); const r=await o.apply(this,arguments); if(/vocos|sherpa-onnx/.test(u)) window.__voc.push(r.status+' '+u.slice(0,140)+' len='+r.headers.get('content-length')); return r;}; return 1;})()`);
      if (process.env.AUDIT_TTS_ONLY !== 'clone') {
        // ---- Kokoro
        await h.type('.tts-textarea', 'Hello from the audit. This is a test of the voice.');
        await h.click('.speak-btn:not(.secondary)');
        var status = await h.waitFor(`(()=>{const e=document.querySelector('.tts-error'); if(e) return 'ERR '+e.innerText; const s=document.querySelector('.tts-status')?.innerText||''; return /\\d+(\\.\\d+)?s audio in/.test(s) ? s : '';})()`, 1800000, 'Kokoro status "Ns audio in"');
        if (status.startsWith('ERR ')) throw new Error('Kokoro: ' + status);
        const kSec = parseFloat(status);
        if (!(kSec > 1)) throw new Error('Kokoro produced no real audio: ' + status);
        evidence.push('kokoro: ' + status.trim());

      } else { var status = ''; evidence.push('kokoro: SKIPPED (AUDIT_TTS_ONLY=clone)'); }

      // ---- ZipVoice clone
      await h.clickText('.tts-mode-toggle .mode-btn', /Clone a voice/);
      await h.waitFor(`!!document.querySelector('.clone-add')`, 15000, 'clone panel');
      await h.ev(`(()=>{const d=document.querySelector('.clone-add'); d.open=true; return d.open;})()`);
      await h.type('.clone-add .tts-input', 'Audit voice');
      await h.typeNth('.clone-add .tts-input', 1, 'All LibriVox recordings are in the public domain.');
      await h.waitFor(`!!document.querySelector('.clone-add .mic-btn')`, 15000, 'Record reference button');
      await h.click('.clone-add .mic-btn');                          // auto-stops at 8 s
      const rec = await h.waitFor(`document.querySelector('.ref-ready') ? 'ok' : (document.querySelector('.tts-error')?.innerText || '')`, 60000, `reference recorded (${FAKE_MIC_NOTE})`);
      let micBroken = null;
      if (rec !== 'ok') {
        // fallback so the ZipVoice models (incl. the vocoder) are still exercised: upload the same clip; the route is still reported FAIL below
        micBroken = rec;
        await h.clickText('.clone-add .mode-btn', /Upload WAV/);
        await h.waitFor(`!!document.querySelector('.clone-add input[type=file]')`, 15000, 'file input');
        await h.upload('.clone-add input[type=file]', CLIP);
        await h.waitFor(`!!document.querySelector('.ref-ready')`, 60000, 'reference loaded from file');
      }
      evidence.push('reference: ' + await h.text('.ref-ready'));
      await h.clickText('.clone-add .speak-btn', /Save voice/);
      await h.waitFor(`(document.querySelector('.tts-status')?.innerText||'').startsWith('Saved')`, 60000, 'voice saved');
      await h.type('.tts-textarea', 'The quick brown fox jumps over the lazy dog.');
      await h.waitFor(`(()=>{const b=document.querySelector('.tts-actions .speak-btn:not(.secondary)'); return b && !b.disabled;})()`, 30000, 'Speak enabled');
      await h.click('.tts-actions .speak-btn:not(.secondary)');
      status = await h.waitFor(`(()=>{const e=document.querySelector('.tts-error'); if(e) return 'ERR '+e.innerText; const s=document.querySelector('.tts-status')?.innerText||''; return /clone '/.test(s) && /\\d+(\\.\\d+)?s audio in/.test(s) ? s : '';})()`, 1800000, 'clone status "Ns audio in ... clone"');
      if (status.startsWith('ERR ')) throw new Error('ZipVoice clone: ' + status);
      const cSec = parseFloat(status);
      if (!(cSec > 1)) throw new Error('ZipVoice produced no real audio: ' + status);
      evidence.push('clone: ' + status.trim());
      evidence.push('vocoder fetches: ' + JSON.stringify(await h.ev('window.__voc')));
      if (micBroken) throw new Error(`MIC PATH BROKEN: page says "${micBroken}". (With the clip uploaded instead, everything else worked: ${evidence.join(' || ')})`);
      return evidence.join(' || ');
    },
  },

  '/ai-chat': {
    // SmolLM2-360M Instruct Q8_0 (smallest "verified" option after Gemma-270M): pick it, send a message, wait for a non-empty reply bubble.
    run: async h => {
      await ready(h, 'li.selectable');
      await h.clickText('li.selectable', /SmolLM2 360M/);
      await h.waitFor(`!!document.querySelector('.chat .composer textarea') || !!document.querySelector('.error pre')`, 900000, 'model loaded (chat view)');
      if (await h.count('.error pre')) throw new Error('model load failed: ' + await h.text('.error pre'));
      await h.type('.composer textarea', 'What is the capital of France? Answer in one short sentence.');
      await h.click('.composer .primary:not(.stop)');
      // the reply is final when a completed assistant message exists and the typing bubble is gone
      const reply = await h.waitFor(`(()=>{if(document.querySelector('.msg.assistant .typing')) return ''; const m=[...document.querySelectorAll('.msg.assistant .text')]; return m.length ? m[m.length-1].innerText.trim() : '';})()`, 600000, 'completed assistant reply');
      if (reply.length < 3) throw new Error('empty assistant reply');
      return `reply: ${JSON.stringify(reply.slice(0, 200))}`;
    },
  },

  // ---------------------------------------------------------------- TOOLING
  '/inspector': {
    // Inspect from the models host by repo + file: an architecture summary must render, and only a small fraction of the file may be fetched.
    run: async h => {
      await ready(h, '.hub-input');
      await h.type('.hub-input:not(.hub-input-path)', 'onnx-community/mobilenetv3_small_100.lamb_in1k');
      await h.type('.hub-input-path', 'onnx/model.onnx');
      await h.waitFor(`(()=>{const b=document.querySelector('.hub-btn'); return b && !b.disabled;})()`, 15000, 'Inspect button enabled');
      // observe the page's own fetches to the models host (Resource Timing reports 0 bytes cross-origin): sum the Content-Length of each response
      await h.ev(`(()=>{window.__mh={n:0,bytes:0,ranged:0}; const o=window.fetch; window.fetch=async function(i,init){const u=String(i?.url??i); const r=await o.apply(this,arguments);
        if(/models\.spawndev\.com/.test(u)&&/\.onnx/.test(u)){window.__mh.n++; window.__mh.bytes+=Number(r.headers.get('content-length')||0); if(r.status===206) window.__mh.ranged++;} return r;}; return true;})()`);
      await h.click('.hub-btn');
      await h.waitFor(`!!document.querySelector('.result-grid') || !!document.querySelector('.error-box')`, 180000, 'inspection result');
      const nodes = await h.ev(`[...document.querySelectorAll('.stat-row')].find(r=>/Nodes/.test(r.innerText))?.querySelector('.stat-value')?.innerText ?? ''`);
      const params = await h.ev(`[...document.querySelectorAll('.stat-row')].find(r=>/Parameters/.test(r.innerText))?.querySelector('.stat-value')?.innerText ?? ''`);
      const fileSize = await h.ev(`[...document.querySelectorAll('.stat-row')].find(r=>/File Size/.test(r.innerText))?.querySelector('.stat-value')?.innerText ?? ''`);
      const ops = await h.text('.op-list');
      if (!(parseInt(nodes) > 10)) throw new Error('no node count rendered: ' + nodes);
      if (!/Conv/.test(ops)) throw new Error('operator list has no Conv: ' + ops.slice(0, 100));
      // "only a small fraction of the file": bytes of the .onnx actually fetched (from the observed responses) vs the file size the page reports
      const mh = await h.ev('window.__mh');
      const sizeMB = parseFloat(fileSize);
      const fetchedMB = mh.bytes / 1048576;
      if (mh.n === 0) throw new Error('could not observe the .onnx fetches (fetch not on the main thread?)');
      if (!(sizeMB > 0 && fetchedMB < sizeMB * 0.5)) throw new Error(`fetched ${fetchedMB.toFixed(2)} MB of a ${sizeMB} MB file: not a small fraction`);
      return `nodes ${nodes}, params ${params}, file ${fileSize}, ops ${ops.slice(0, 80)} | fetched ${fetchedMB.toFixed(2)} MB of the .onnx in ${mh.n} responses (${mh.ranged} ranged 206)`;
    },
  },

  '/models': {
    // Despite being described as an HF browser, the page is a static catalogue + "Load Custom Model", which inspects structure through the models host.
    // No huggingface.co request is expected (the HF badges are plain links), so allowHuggingFace stays OFF: any request there is a finding.
    run: async h => {
      await ready(h, '.model-card');
      const cards = await h.count('.model-card');
      if (cards < 10) throw new Error('model catalogue did not render: ' + cards + ' cards');
      await h.type('.custom-input:not(.custom-input-file)', 'onnx-community/mobilenetv3_small_100.lamb_in1k');
      await h.type('.custom-input-file', 'onnx/model.onnx');
      await h.click('.load-btn');
      await h.waitFor(`!!document.querySelector('.inspect-summary') || !!document.querySelector('.custom-error')`, 180000, 'custom model inspection');
      if (await h.count('.custom-error')) throw new Error('custom model: ' + await h.text('.custom-error'));
      const summary = await h.text('.custom-result');
      const ops = await h.text('.inspect-summary');
      if (!/nodes/.test(summary)) throw new Error('no summary rendered: ' + summary);
      return `${cards} catalogue cards | ${summary} | ${ops.slice(0, 160)}`;
    },
  },

  '/cache': {
    // Download something first (/sentiment, DistilBERT) in the SAME browser context, then /cache must list it.
    run: async h => {
      await ready(h, '.cache-page'); await h.sleep(1500);
      const empty = await h.text('.cache-page');
      const emptyCount = await h.text('.cache-num');
      // cause a real download through the sentiment demo, then come back (full navigations: OPFS survives within the context)
      await h.ev(`setTimeout(()=>{location.href='/sentiment'},0); true`);
      await h.sleep(1500); await h.mounted(); await h.sleep(1500);
      await h.type('.sentiment-textarea', 'I absolutely love this movie.');
      await h.click('.run-btn');
      await h.waitFor(`document.querySelector('.sentiment-verdict')?.innerText?.trim()`, 600000, 'sentiment verdict (model downloaded)');
      await h.ev(`setTimeout(()=>{location.href='/cache'},0); true`);
      await h.sleep(1500); await h.mounted(); await h.sleep(2000);
      await h.waitFor(`document.querySelectorAll('.cache-table .cache-key').length > 0`, 60000, 'cache table lists a model');
      const rows = await h.ev(`[...document.querySelectorAll('.cache-table')][0] ? [...document.querySelectorAll('.cache-table')][0].innerText.replace(/\\s+/g,' ').trim() : ''`);
      const summary = await h.text('.cache-summary');
      if (/partial/.test(rows)) throw new Error('cache lists a PARTIAL entry after a completed load: ' + rows);
      return `before: ${emptyCount} | after: ${summary} | ${rows.slice(0, 260)}`;
    },
  },

  '/benchmark': {
    // WebGPU MatMul GFLOPS + the three small-model benches (SqueezeNet 4.8 MB, Mosaic 6.5 MB, ESPCN 0.2 MB, all from the models host).
    run: async h => {
      await ready(h, '.bench-run-btn');
      await h.click('.bench-run-btn');
      await h.waitFor(`document.querySelectorAll('.result-group').length >= 4 && !document.querySelector('.bench-progress')`, 600000, 'benchmark results');
      const rows = await h.ev(`[...document.querySelectorAll('.result-group')].map(g=>g.querySelector('.group-title').innerText+': '+[...g.querySelectorAll('.result-bar-row')].map(r=>r.innerText.replace(/\\s+/g,' ').trim()).join(' / '))`);
      const failed = await h.count('.result-bar-row.failed');
      if (failed) throw new Error(failed + ' benchmark row(s) failed: ' + rows.join(' | '));
      if (!rows.every(r => /\d+(\.\d+)? (ms|GFLOPS)/.test(r) || /\d/.test(r))) throw new Error('a group has no number: ' + rows.join(' | '));
      return rows.join(' | ');
    },
  },

  '/train': {
    // Draw 3 circles + 3 squares, train on the GPU, draw a new circle: the live classification must render per-class confidences.
    run: async h => {
      const sel = '.draw-surface';
      await ready(h, sel); await h.sleep(2000);   // the page creates its GPU accelerator after first render
      const draw = async (cls, pts) => {
        await h.clickText('.class-select-btn', new RegExp(cls));
        await h.ev(strokeJs(sel, pts));
        await h.sleep(150);
        await h.clickText('.canvas-actions .canvas-btn', /^\s*Save to/);
        await h.sleep(150);
      };
      for (const [cx, cy, r] of [[140, 140, 80], [130, 150, 60], [150, 130, 95]]) await draw('Circle', circlePts(cx, cy, r));
      for (const [cx, cy, hh] of [[140, 140, 80], [130, 150, 60], [150, 130, 90]]) await draw('Square', squarePts(cx, cy, hh));
      const counts = await h.ev(`[...document.querySelectorAll('.class-count')].map(e=>e.innerText).join(', ')`);
      await h.waitFor(`[...document.querySelectorAll('.run-btn')].some(b=>/Train on GPU/.test(b.innerText))`, 30000, `Train button (samples: ${counts})`);
      await h.clickText('.run-btn', /Train on GPU/);
      await h.waitFor(`/Prediction/.test(document.querySelector('.results-panel')?.innerText||'')`, 120000, 'training finished (Prediction panel)');
      const loss = await h.ev(`document.body.innerText.match(/Loss: [\\d.]+/)?.[0] ?? ''`);
      await h.ev(strokeJs(sel, circlePts(140, 140, 85)));
      const bars = await h.waitFor(`(()=>{const r=[...document.querySelectorAll('.cb-row')].map(e=>e.innerText.replace(/\\s+/g,' ').trim()); return r.length>=2 ? r.join(' / ') : '';})()`, 30000, 'confidence bars after drawing');
      return `samples ${counts} | trained | circle drawn -> ${bars}`;
    },
  },

  '/snake': {
    // Train the (tiny) System One decision head on the GPU, switch to System One, play: the score/length stats and decision latency must render.
    run: async h => {
      await h.waitFor(`[...document.querySelectorAll('.run-btn')].some(b=>/Train head|Retrain/.test(b.innerText)&&!b.disabled)`, 120000, 'Train head enabled');
      await h.clickText('.run-btn', /Train head|Retrain/);
      const st = await h.waitFor(`(()=>{const s=document.querySelector('.snake-status')?.innerText||''; return /^Head ready/.test(s) ? s : (/failed|No .* device/i.test(s) ? 'ERR '+s : '');})()`, 600000, 'Head ready');
      if (st.startsWith('ERR')) throw new Error(st);
      await h.clickText('.mode-btn', /System One/);
      await h.clickText('.run-btn', /^\s*Start/);
      await h.waitFor(`parseFloat(document.querySelector('.latency-line strong')?.innerText||'0') > 0`, 60000, 'a System One decision was made');
      await h.sleep(5000);
      const stats = await h.text('.snake-stats');
      const lat = await h.text('.latency-line');
      return `${st} | ${stats} | ${lat}`;
    },
  },

  // ---------------------------------------------------------------- DOC / META pages: LOAD-ONLY (they render content, there is no model or action)
  '/': { run: async h => { await ready(h, '.hero-title'); const t = await h.text('body'); if (!/How It Works/.test(t) || !/WebGPU/.test(t)) throw new Error('home: key sections missing'); return 'LOAD-ONLY: ' + (await h.text('.hero-title')) + ' | sections: How It Works, backends'; } },
  '/getting-started': { run: async h => { await ready(h, '.gs-title'); const t = await h.text('body'); if (!/Install/.test(t) || !/Blazor WASM setup/.test(t)) throw new Error('getting-started: key sections missing'); return 'LOAD-ONLY: ' + (await h.text('.gs-title')) + ' | sections: Install ... Blazor WASM setup, Try a demo'; } },
  '/pipelines': { run: async h => { await ready(h, '.pipelines-title'); const t = await h.text('body'); if (t.length < 500) throw new Error('pipelines: page is nearly empty'); return 'LOAD-ONLY: ' + (await h.text('.pipelines-title')) + ' | ' + t.slice(0, 120); } },
  '/tests': { run: async h => { await h.waitFor(`/Unit Tests/.test(document.body.innerText)`, 60000, 'Unit Tests heading'); const t = await h.text('body'); return 'LOAD-ONLY: ' + t.slice(0, 160); } },
};
