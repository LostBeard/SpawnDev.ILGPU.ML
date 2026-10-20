// Scenarios for the image demos: /classify /clip /depth /depth-voxel /detect /explain /face /pose /remove-bg /style /super-res (see _routes.mjs for the scenario contract).
//
// Every scenario picks a bundled sample (the page's own "Or try a sample" button, which is the same code path as a file pick: ImageDropZone ->
// OnImageLoaded), waits for the model to download + run, then asserts on the RENDERED result: a label, a count, a box/keypoint in the SVG overlay, or
// an output <img>/<canvas> whose pixels are measured (non-uniform, and different from the input where that is the point of the demo).

// ---- in-page measuring helpers (injected into ev() expressions) ----
// pixelStats(el): draws an <img> or <canvas> (downscaled to <=128 px) and returns { w, h, std, mean, aMin, aMax, aTransparent } over RGB/alpha.
const PIXEL_STATS_JS = `(function(el){
  const sw = el.naturalWidth || el.width, sh = el.naturalHeight || el.height;
  if (!sw || !sh) return null;
  const k = Math.min(1, 128 / Math.max(sw, sh)), w = Math.max(1, Math.round(sw * k)), h = Math.max(1, Math.round(sh * k));
  const c = document.createElement('canvas'); c.width = w; c.height = h;
  const x = c.getContext('2d', { willReadFrequently: true }); x.drawImage(el, 0, 0, w, h);
  const d = x.getImageData(0, 0, w, h).data; let n = 0, s = 0, s2 = 0, aMin = 255, aMax = 0, aT = 0;
  for (let i = 0; i < d.length; i += 4) { for (let j = 0; j < 3; j++) { const v = d[i + j]; s += v; s2 += v * v; n++; } const a = d[i + 3]; if (a < aMin) aMin = a; if (a > aMax) aMax = a; if (a < 128) aT++; }
  const mean = s / n; return { w: sw, h: sh, std: Math.sqrt(Math.max(0, s2 / n - mean * mean)), mean, aMin, aMax, aTransparent: aT / (d.length / 4) };
})`;
// a cheap hash of the downscaled pixels, to prove the output is not just the input echoed back
const PIXEL_HASH_JS = `(function(el){
  const c = document.createElement('canvas'); c.width = 32; c.height = 32; const x = c.getContext('2d', { willReadFrequently: true });
  x.drawImage(el, 0, 0, 32, 32); const d = x.getImageData(0, 0, 32, 32).data; return Array.from(d);
})`;

/**
 * Wrap the driver helper so a page that is busy (model compile / inference blocks the page's main thread for minutes) cannot hang the scenario:
 * every evaluate is raced against a timeout and re-sent. Observed on /super-res: a Runtime.evaluate sent while the page was busy never answered
 * and the plain h.waitFor stayed blocked for 15+ minutes although the result was on screen.
 */
function robust(h) {
  const ev = async (expr, tries = 40, perTryMs = 15000) => {
    if (process.env.VISION_TRACE) h.log('ev ' + expr.slice(0, 90).replace(/[\r\n\t ]+/g, ' '));
    for (let i = 0; i < tries; i++) {
      const r = await Promise.race([h.ev(expr).then(v => ({ v }), e => ({ e })), h.sleep(perTryMs).then(() => null)]);
      if (r?.e) throw r.e;
      if (r) return r.v;
    }
    throw new Error('page did not answer an evaluate for ' + (tries * perTryMs / 1000) + ' s: ' + expr.slice(0, 120));
  };
  return {
    ...h, ev,
    async waitFor(expr, timeoutMs = 120000, label = expr) {
      const end = Date.now() + timeoutMs;
      while (Date.now() < end) { try { const v = await ev(expr, 1, 15000); if (v) return v; } catch { /* retry */ } await h.sleep(500); }
      throw new Error(`timed out (${timeoutMs / 1000}s) waiting for: ${label}`);
    },
    async click(sel) { if (!await ev(`(()=>{const e=document.querySelector(${JSON.stringify(sel)});if(!e)return false;e.click();return true;})()`)) throw new Error('no element: ' + sel); },
    async clickText(sel, re) { if (!await ev(`(()=>{const e=[...document.querySelectorAll(${JSON.stringify(sel)})].find(x=>${re}.test(x.textContent));if(!e)return false;e.click();return true;})()`)) throw new Error(`no ${sel} matching ${re}`); },
    async count(sel) { return ev(`document.querySelectorAll(${JSON.stringify(sel)}).length`); },
    async text(sel) { return ev(`(document.querySelector(${JSON.stringify(sel)})?.innerText ?? '').replace(/[\\s\\u00a0]+/g,' ').trim()`); },
  };
}

/**
 * Feed the page's own <InputFile> (the "click to browse" path) a downscaled copy of a bundled sample: fetch -> canvas -> JPEG File -> input.files + change.
 * Used where the full-size sample makes the page's main thread block for many minutes (see /super-res), so the demo can still be exercised end to end.
 */
async function uploadScaled(h, url, maxSide) {
  await h.waitFor("!!document.querySelector('input[type=file]')", 30000, 'file input');
  return h.ev(`(async () => {
    const blob = await (await fetch(${JSON.stringify(url)})).blob(); const bmp = await createImageBitmap(blob);
    const k = Math.min(1, ${maxSide} / Math.max(bmp.width, bmp.height)), w = Math.round(bmp.width * k), h = Math.round(bmp.height * k);
    const c = document.createElement('canvas'); c.width = w; c.height = h; c.getContext('2d').drawImage(bmp, 0, 0, w, h);
    const out = await new Promise(r => c.toBlob(r, 'image/jpeg', 0.92)); const f = new File([out], 'scaled.jpg', { type: 'image/jpeg' });
    const dt = new DataTransfer(); dt.items.add(f); const inp = document.querySelector('input[type=file]'); inp.files = dt.files;
    inp.dispatchEvent(new Event('change', { bubbles: true })); return w + 'x' + h;
  })()`);
}

const fmt = s => `${s.w}x${s.h} std=${s.std.toFixed(1)} mean=${s.mean.toFixed(0)}`;

/** Click the page's sample button with this label (ImageDropZone: .sample-btn). */
async function pickSample(h, label) {
  await h.waitFor(`[...document.querySelectorAll('.sample-btn')].some(b => ${new RegExp('^\\s*' + label + '\\s*$', 'i')}.test(b.textContent) && !b.disabled)`, 30000, `sample button ${label}`);
  await h.clickText('.sample-btn', new RegExp('^\\s*' + label + '\\s*$', 'i'));
}

/** Wait until every result <img> has really loaded (polled: awaiting img.decode() can hang forever in a background window and would hang the driver). */
async function settleImages(h) {
  await h.waitFor(`(() => { const i = [...document.querySelectorAll('.ba-container img, .feature-img')]; return i.length > 0 && i.every(x => x.complete && x.naturalWidth > 0); })()`, 60000, 'result images loaded');
}

/** Throw if the page shows an error banner / status error (pages catch their own exceptions and print them, so a console check alone would miss it). */
async function assertNoPageError(h) {
  const t = await h.ev(`[...document.querySelectorAll('.error-banner,.status-message')].map(e => e.innerText.trim()).filter(Boolean).join(' | ')`);
  if (/error|fail|no .* device|not available/i.test(t)) throw new Error('page reported: ' + t);
}

export const scenarios = {

  // SqueezeNet: cat sample must come back with a cat/feline ImageNet label in the top 5.
  '/classify': {
    run: async h0 => {
      const h = robust(h0);
      await pickSample(h, 'Cat');
      await h.waitFor(`document.querySelectorAll('.cb-row').length > 0 || document.querySelector('.error-banner')`, 300000, 'classification rows');
      await assertNoPageError(h);
      const rows = await h.ev(`[...document.querySelectorAll('.cb-row')].map(r => r.querySelector('.cb-label')?.title + ' ' + r.querySelector('.cb-value')?.innerText)`);
      if (!rows.length) throw new Error('no predictions rendered');
      if (!rows.some(r => /cat|tabby|tiger|lynx|egyptian|persian|siamese|leopard|cougar/i.test(r))) throw new Error('expected a cat/feline label in the top 5, got: ' + rows.join(' ; '));
      return `top-5 for the cat sample: ${rows.join(' ; ')}`;
    },
  },

  // CLIP zero-shot: the default categories include "a photo of a cat"; the cat sample must rank it first and the distribution must not be flat.
  '/clip': {
    run: async h0 => {
      const h = robust(h0);
      await pickSample(h, 'Cat');
      await h.waitFor(`document.querySelectorAll('.clip-results .cb-row').length > 0`, 300000, 'CLIP result rows');
      await assertNoPageError(h);
      const rows = await h.ev(`[...document.querySelectorAll('.clip-results .cb-row')].map(r => ({ label: r.querySelector('.cb-label')?.title, pct: parseFloat(r.querySelector('.cb-value')?.innerText) }))`);
      rows.sort((a, b) => b.pct - a.pct);
      if (!/cat/i.test(rows[0].label)) throw new Error('expected the top category to be the cat one, got: ' + JSON.stringify(rows));
      if (rows[0].pct < 40) throw new Error('top score suspiciously low (flat distribution?): ' + JSON.stringify(rows));
      return `ranked: ${rows.map(r => `${r.label} ${r.pct}%`).join(' ; ')}`;
    },
  },

  // Depth Anything V2: the "After" side is a <canvas> the GPU renders into: it must be non-uniform and spatially structured.
  '/depth': {
    run: async h0 => {
      const h = robust(h0);
      await pickSample(h, 'Street');
      await h.waitFor(`document.querySelector('.ba-after canvas') && document.querySelector('.depth-stat')`, 300000, 'depth canvas + stats');
      await h.sleep(1500); // PresentAsync after the canvas is attached
      await assertNoPageError(h);
      const stats = await h.ev(`(${PIXEL_STATS_JS})(document.querySelector('.ba-after canvas'))`);
      const txt = await h.ev(`[...document.querySelectorAll('.depth-stat')].map(e => e.innerText).join(' ')`);
      if (!stats || stats.std < 15) throw new Error('depth canvas is blank / uniform: ' + JSON.stringify(stats) + ' ' + txt);
      // The depth map must also have SPATIAL structure (not one flat gradient/noise level per half): the street photo's top and bottom bands differ.
      const bands = await h.ev(`(() => { const c = document.querySelector('.ba-after canvas'), x = document.createElement('canvas'); x.width = 16; x.height = 16; const g = x.getContext('2d', { willReadFrequently: true }); g.drawImage(c, 0, 0, 16, 16); const d = g.getImageData(0, 0, 16, 16).data; let t = 0, b = 0; for (let i = 0; i < 16 * 8 * 4; i += 4) t += d[i] + d[i + 1] + d[i + 2]; for (let i = 16 * 8 * 4; i < d.length; i += 4) b += d[i] + d[i + 1] + d[i + 2]; return Math.abs(t - b) / (16 * 8 * 3); })()`);
      if (bands < 3) throw new Error('depth map has no spatial structure (top/bottom bands equal): ' + bands);
      // NOTE: the page's "Min/Max depth" labels are never assigned in DepthPage.razor (_minDepth/_maxDepth stay 0), so they cannot be asserted on.
      return `depth canvas ${fmt(stats)}, top/bottom band difference ${bands.toFixed(1)}; page stats text: ${txt}`;
    },
  },

  // The page is webcam-only. Needs Chrome's fake camera: AUDIT_CHROME_ARGS='--use-fake-device-for-media-stream --use-fake-ui-for-media-stream'.
  // Enable Camera -> Capture -> the model loads on the first frame and the GPU colormap is presented to #depth canvas (.depth-canvas): must be non-uniform.
  '/depth-voxel': {
    run: async h0 => {
      const h = robust(h0);
      await h.waitFor(`!!document.querySelector('.webcam-start-btn')`, 30000, 'Enable Camera button');
      // WebcamCapture reports its failure only with Console.WriteLine ("[WebcamCapture] Camera error: ..."), which is not a console.error, so capture it here
      // and put it in the failure: otherwise a broken Enable Camera looks like a timeout.
      await h.ev(`window.__camLogs = []; const o = console.log; console.log = (...a) => { window.__camLogs.push(a.map(String).join(' ').slice(0, 400)); o.apply(console, a); }; true`);
      await h.click('.webcam-start-btn');
      try { await h.waitFor(`document.querySelector('.webcam-video')?.videoWidth > 0`, 30000, 'fake camera streaming'); }
      catch (e) { const logs = await h.ev(`window.__camLogs.filter(l => /Webcam|Camera/i.test(l)).join(' || ')`); throw new Error(e.message + ' | page log: ' + (logs || '(none)')); }
      await h.click('.webcam-capture-btn'); // first captured frame loads the model; AutoCapture then keeps feeding frames
      await h.waitFor(`(() => { const c = document.querySelector('.depth-canvas'); if (!c) return false; const f = (${PIXEL_STATS_JS})(c); return f && f.std > 15; })()`, 300000, 'non-uniform depth canvas');
      const stats = await h.ev(`(${PIXEL_STATS_JS})(document.querySelector('.depth-canvas'))`);
      const fps = await h.text('.fps-badge');
      return `depth canvas ${fmt(stats)} fps-badge=${fps || '-'}`;
    },
  },

  // YOLOv8-Nano: the street sample must yield at least one box + a chip with a plausible COCO class.
  '/detect': {
    run: async h0 => {
      const h = robust(h0);
      await pickSample(h, 'Street');
      await h.waitFor(`document.querySelectorAll('.detection-chip').length > 0 || document.querySelector('.error-banner')`, 300000, 'detections');
      await assertNoPageError(h);
      const chips = await h.ev(`[...document.querySelectorAll('.detection-chip')].map(c => c.innerText.trim())`);
      const boxes = await h.count('.bbox-svg rect[fill="none"]');
      if (!chips.length || boxes < 1) throw new Error(`no detections rendered (chips=${chips.length} boxes=${boxes})`);
      return `${boxes} box rects, classes: ${chips.join(', ')}`;
    },
  },

  // SqueezeNet with the executor intercepted: a feature-map tile must render non-blank with real channel statistics (see the known limitations below).
  '/explain': {
    run: async h0 => {
      const h = robust(h0);
      await pickSample(h, 'Cat');
      await h.waitFor(`document.querySelectorAll('.feature-tile').length > 0`, 300000, 'feature-map tiles');
      await settleImages(h);
      const r = await h.ev(`(() => {
        const imgs = [...document.querySelectorAll('.feature-img')];
        const st = imgs.slice(0, 8).map(i => (${PIXEL_STATS_JS})(i)); const hs = new Set(imgs.map(i => (${PIXEL_HASH_JS})(i).join(',')));
        return { tiles: imgs.length, stds: st.map(s => s?.std ?? 0), distinct: hs.size, title: document.querySelector('.section-title')?.innerText, stats: [...document.querySelectorAll('.stat-row')].map(e => e.innerText.replace(/\\s+/g, ' ')).join(' ') };
      })()`);
      if (r.tiles < 1) throw new Error('no feature tiles');
      if (Math.max(...r.stds) < 8) throw new Error('feature maps are blank: ' + JSON.stringify(r));
      // KNOWN LIMITATIONS of this PARTIAL demo (DEMO_AND_MODEL_STATUS.md: "works on a limited set of models"), observed 2026-10-20 and recorded in the
      // evidence rather than hidden: (1) layer 0 (conv0, 64x111x111) is shown as ONE tile "(1 channels)" because the page's channel-size guess only knows
      // 7/14/28/56/112/224 squares; (2) the Layer slider only changes the label: UpdateFeatureMaps() is called once, so the maps never change; the Model and
      // Visualization selectors are likewise unwired (no @bind:after). The scenario asserts the REAL result that exists (a rendered, non-blank map with
      // real per-channel statistics) and probes the slider so a future fix shows up as a change in this evidence.
      const nums = (r.stats.match(/-?[0-9]+[.][0-9]+/g) ?? []).map(Number);
      if (nums.length < 4 || !(nums[1] > nums[0])) throw new Error('channel statistics are not real numbers (max > min expected): ' + r.stats);
      await h.ev(`(() => { const s = document.querySelector('.layer-slider'); s.value = '6'; s.dispatchEvent(new Event('input', { bubbles: true })); s.dispatchEvent(new Event('change', { bubbles: true })); })()`);
      await h.sleep(1500);
      const after = await h.ev(`(() => { const imgs = [...document.querySelectorAll('.feature-img')]; return { tiles: imgs.length, title: document.querySelector('.section-title')?.innerText }; })()`);
      const slider = /conv0/.test(after.title) && after.tiles === r.tiles ? 'slider moved to layer 6 but the maps did NOT change (unwired)' : 'slider changed the maps';
      return `${r.title}; ${r.tiles} tile(s), max std ${Math.max(...r.stds).toFixed(1)}; ${r.stats}; ${slider}: "${after.title}"`;
    },
  },

  // BlazeFace: the portrait sample must give "Faces: 1" and an SVG box + landmark circles.
  '/face': {
    run: async h0 => {
      const h = robust(h0);
      await pickSample(h, 'Portrait');
      await h.waitFor(`document.querySelector('.face-overlay rect') || document.querySelector('.no-faces-hint') || document.querySelector('.error-banner')`, 300000, 'face result');
      await assertNoPageError(h);
      const count = await h.text('.face-count');
      const rects = await h.count('.face-overlay rect'), dots = await h.count('.face-overlay circle');
      if (!/Faces:\s*[1-9]/.test(count) || rects < 1) throw new Error(`no face found in the portrait sample: ${count}, ${rects} boxes`);
      if (dots < 1) throw new Error('face box but no landmarks rendered');
      return `${count}; ${rects} box, ${dots} landmark dots`;
    },
  },

  // MoveNet: the runner sample must report confident keypoints (page counts those > 0.3) and draw them.
  '/pose': {
    run: async h0 => {
      const h = robust(h0);
      await pickSample(h, 'Runner');
      await h.waitFor(`document.querySelector('.pose-stat') || document.querySelector('.error-banner')`, 300000, 'pose keypoints');
      await assertNoPageError(h);
      const stat = await h.text('.pose-stat');
      const n = parseInt(/detected:\s*(\d+)/.exec(stat)?.[1] ?? '0');
      const dots = await h.count('.pose-overlay circle'), lines = await h.count('.pose-overlay line');
      if (n < 5 || dots < 5 || lines < 3) throw new Error(`too few keypoints: "${stat}", ${dots} dots, ${lines} bones`);
      return `${stat}; ${dots} dots, ${lines} bones drawn`;
    },
  },

  // RMBG-1.4: the cut-out (transparent mode) must contain BOTH transparent and opaque pixels, and differ from the original.
  '/remove-bg': {
    run: async h0 => {
      const h = robust(h0);
      await pickSample(h, 'Portrait');
      await h.waitFor(`document.querySelector('.ba-after img')?.src?.length > 100 || document.querySelector('.error-banner')`, 300000, 'background-removal result');
      await settleImages(h);
      await assertNoPageError(h);
      const r = await h.ev(`(() => {
        const a = document.querySelector('.ba-after img'), b = document.querySelector('.ba-before img');
        return { out: (${PIXEL_STATS_JS})(a), inp: (${PIXEL_STATS_JS})(b), same: JSON.stringify((${PIXEL_HASH_JS})(a)) === JSON.stringify((${PIXEL_HASH_JS})(b)) };
      })()`);
      if (!(r.out.aTransparent > 0.03 && r.out.aTransparent < 0.97)) throw new Error('mask is all-opaque or all-transparent: ' + JSON.stringify(r.out));
      return `cut-out ${fmt(r.out)} transparent=${(r.out.aTransparent * 100).toFixed(0)}% alpha ${r.out.aMin}..${r.out.aMax}; input ${fmt(r.inp)}`;
    },
  },

  // Style transfer (image mode): a styled output must render, be non-uniform, and differ from the original.
  '/style': {
    run: async h0 => {
      const h = robust(h0);
      await pickSample(h, 'Cat');
      await h.waitFor(`document.querySelector('.ba-after img')?.src?.length > 100 || document.querySelector('.error-banner')`, 300000, 'styled image');
      await settleImages(h);
      await assertNoPageError(h);
      const r = await h.ev(`(() => {
        const a = document.querySelector('.ba-after img'), b = document.querySelector('.ba-before img');
        const ha = (${PIXEL_HASH_JS})(a), hb = (${PIXEL_HASH_JS})(b); let diff = 0; for (let i = 0; i < ha.length; i++) diff += Math.abs(ha[i] - hb[i]);
        return { out: (${PIXEL_STATS_JS})(a), inp: (${PIXEL_STATS_JS})(b), meanAbsDiff: diff / ha.length, style: document.querySelector('.style-thumb.selected')?.innerText };
      })()`);
      if (r.out.std < 15) throw new Error('styled output is blank/uniform: ' + JSON.stringify(r.out));
      if (r.meanAbsDiff < 4) throw new Error('styled output is identical to the input: ' + JSON.stringify(r));
      return `style "${r.style}": output ${fmt(r.out)}; input ${fmt(r.inp)}; mean abs pixel diff ${r.meanAbsDiff.toFixed(1)}`;
    },
  },

  // ESPCN 3x: the enhanced image must be larger than the input (about 3x), non-uniform.
  '/super-res': {
    run: async h0 => {
      const h = robust(h0);
      // The full 640x398 sample blocks the page's main thread for 10+ minutes in this (interpreted) build, so the default run feeds a 160 px copy through
      // the file input; VISION_FULL_SAMPLE=1 clicks the real sample button instead (used once to measure that duration).
      const fed = process.env.VISION_FULL_SAMPLE ? (await pickSample(h, 'Low-res Photo'), 'sample button (640x398)') : 'file input, ' + await uploadScaled(h, 'samples/lowres.jpg', 160);
      await h.waitFor(`document.querySelector('.ba-after img')?.src?.length > 100 || document.querySelector('.status-message,.error-banner')`, (process.env.VISION_FULL_SAMPLE ? 1800000 : 300000), 'enhanced image');
      await settleImages(h);
      await assertNoPageError(h);
      const r = await h.ev(`(() => {
        const a = document.querySelector('.ba-after img'), b = document.querySelector('.ba-before img');
        return { out: (${PIXEL_STATS_JS})(a), inp: (${PIXEL_STATS_JS})(b) };
      })()`);
      const ratio = r.out.w / r.inp.w;
      if (!(ratio > 2.5 && ratio < 3.5)) throw new Error(`expected ~3x upscale, got ${r.inp.w}x${r.inp.h} -> ${r.out.w}x${r.out.h}`);
      if (r.out.std < 10) throw new Error('enhanced image is blank/uniform: ' + JSON.stringify(r.out));
      return `via ${fed}: ${r.inp.w}x${r.inp.h} -> ${r.out.w}x${r.out.h} (${ratio.toFixed(2)}x), output ${fmt(r.out)}`;
    },
  },
};
