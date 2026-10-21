// Scenarios for the generative demos: /generate (SD Turbo, FLUX.2 klein, FLUX 3 + the Qwen3 caption writer) and /text-to-3d, /image-to-3d
// (a picture -> TRELLIS.2 -> a GLB -> SkinTokens rig -> Kimodo animation). See _routes.mjs for the scenario contract.
// A '#variant' on a route is the same page with a different scenario: '/generate#sd' runs /generate with SD Turbo.
//
// These are the HEAVY demos (GBs of weights, minutes of GPU work). Run them alone on the GPU: the driver runs routes one at a time.

// Pixel statistics of an <img> (drawn into a 64x64 canvas): a blank or constant picture has std ~ 0, a drawn one much more.
const imageStats = sel => `(()=>{const img=document.querySelector(${JSON.stringify(sel)});if(!img||!img.naturalWidth)return null;
  const c=document.createElement('canvas');c.width=64;c.height=64;const x=c.getContext('2d');x.drawImage(img,0,0,64,64);const d=x.getImageData(0,0,64,64).data;
  let s=0,s2=0,n=0;for(let i=0;i<d.length;i+=4){const v=(d[i]+d[i+1]+d[i+2])/3;s+=v;s2+=v*v;n++;}const m=s/n;return {w:img.naturalWidth,h:img.naturalHeight,mean:+m.toFixed(1),std:+Math.sqrt(Math.max(0,s2/n-m*m)).toFixed(1)};})()`;

// The header progress bar (DemoLayout > ModelLoadProgress) and the live caption, sampled together.
const BAR = (captionSel = 'details.gen-caption p, details.t3d-hint p') => `JSON.stringify({
  label: document.querySelector('.mp-label')?.innerText ?? '',
  status: document.querySelector('.mp-status')?.innerText ?? '',
  pct: parseFloat(document.querySelector('.mp-fill')?.style.width) || 0,
  cap: (document.querySelector(${JSON.stringify(captionSel)})?.innerText ?? '').length,
  stage: (document.querySelector('.t3d-stage.running')?.innerText ?? '').replace(/\\s+/g,' ')
})`;

/** The Qwen3 caption writer must (1) move the header bar while the model downloads and loads, (2) show the caption growing as it is written. */
function assertCaptionExperience(samples) {
  const qwen = samples.filter(s => /Qwen3/i.test(s.label));
  const pcts = [...new Set(qwen.map(s => Math.round(s.pct)).filter(p => p > 0))];
  if (qwen.length === 0) throw new Error('the header never showed the Qwen3 caption model');
  if (pcts.length < 3) throw new Error(`the Qwen3 progress bar did not move: distinct non-zero values ${JSON.stringify(pcts)} over ${qwen.length} samples`);
  const caps = samples.map(s => s.cap).filter(n => n > 0);
  const distinct = [...new Set(caps)];
  if (distinct.length < 3) throw new Error(`the caption did not stream: only ${distinct.length} distinct length(s) seen ${JSON.stringify(distinct.slice(0, 8))}`);
  if (distinct[distinct.length - 1] < distinct[0]) throw new Error('the caption shrank while it was written');
  return { qwenBarSteps: pcts.length, qwenBarMaxPct: Math.max(...pcts), captionPieces: distinct.length, captionFirst: distinct[0], captionLast: distinct[distinct.length - 1] };
}

async function assertPicture(h, sel, what) {
  const st = await h.ev(imageStats(sel));
  if (!st) throw new Error(`${what}: no loaded <img> at ${sel}`);
  if (st.std < 8) throw new Error(`${what}: the picture is blank or constant (std ${st.std}, mean ${st.mean})`);
  return st;
}

export const scenarios = {
  '/generate#sd': {
    run: async h => {
      await h.selectValue('select.gen-setting-select', 'sd');
      await h.type('.gen-prompt-input', 'a red fox standing in fresh snow at sunrise, photograph');
      await h.click('.gen-btn');
      const samples = await h.observe(BAR(), `document.querySelector('.gen-result-img, .gen-grid-item img')`, 25 * 60e3);
      const st = await assertPicture(h, '.gen-result-img, .gen-grid-item img', 'SD Turbo picture');
      return { picture: st, barMaxPct: Math.max(...samples.map(s => s.pct)) };
    },
  },

  '/generate#flux2': {
    run: async h => {
      await h.selectValue('select.gen-setting-select', 'flux2');
      await h.type('.gen-prompt-input', 'a red fox standing in fresh snow at sunrise, photograph');
      await h.click('.gen-btn');
      const samples = await h.observe(BAR(), `document.querySelector('.gen-result-img, .gen-grid-item img')`, 40 * 60e3);
      const st = await assertPicture(h, '.gen-result-img, .gen-grid-item img', 'FLUX.2 klein picture');
      return { picture: st, barMaxPct: Math.max(...samples.map(s => s.pct)) };
    },
  },

  '/generate#flux3': {
    run: async h => {
      await h.selectValue('select.gen-setting-select', 'flux3');
      await h.type('.gen-prompt-input', 'a red fox standing in fresh snow at sunrise');
      await h.ev(`(()=>{const c=[...document.querySelectorAll('label')].find(l=>/full caption/.test(l.textContent))?.querySelector('input[type=checkbox]');if(c&&!c.checked)c.click();return !!c;})()`);
      await h.click('.gen-btn');
      const samples = await h.observe(BAR(), `document.querySelector('.gen-result-img, .gen-grid-item img')`, 60 * 60e3);
      const st = await assertPicture(h, '.gen-result-img, .gen-grid-item img', 'FLUX 3 picture');
      // textContent, not innerText: the 'Caption used' <details> is collapsed, and innerText of collapsed content is ''
      const caption = (await h.ev(`document.querySelector('.gen-result details.gen-caption p')?.textContent ?? ''`)).trim();
      if (caption.length < 40) throw new Error('no written caption is shown with the picture: ' + JSON.stringify(caption));
      return { picture: st, caption: caption.slice(0, 120) + '...', ...assertCaptionExperience(samples) };
    },
  },

  // Text -> 4 pictures with FLUX 3 + the Qwen3 caption -> pick one -> TRELLIS.2 GLB -> SkinTokens rig -> Kimodo animation: the whole chain.
  '/text-to-3d#flux3': {
    run: async h => {
      await h.clickText('.t3d-mode', /Text to 3D/);
      await h.selectValue('select.t3d-select', 'flux3');
      await h.selectValue('select.t3d-select', '4');
      // a standing humanoid: the rig and animate steps need one (a chest rigs, but Animate refuses it: "does not look like a standing humanoid")
      await h.type('.t3d-prompt', 'a knight in plate armor, full body, standing in a relaxed A-pose');
      await h.ev(`(()=>{const c=[...document.querySelectorAll('label')].find(l=>/full caption/.test(l.textContent))?.querySelector('input[type=checkbox]');if(c&&!c.checked)c.click();return !!c;})()`);
      await h.click('.t3d-btn');
      const pics = await h.observe(BAR(), `document.querySelectorAll('.t3d-pick-item').length >= 4 && !document.querySelector('.t3d-stage.running')`, 70 * 60e3);
      const exp = assertCaptionExperience(pics);
      const stats = [];
      for (let i = 0; i < 4; i++) stats.push(await assertPicture(h, `.t3d-pick-item:nth-child(${i + 1}) img`, `picture ${i + 1}`));
      h.log(`4 pictures drawn, caption experience ${JSON.stringify(exp)}`);

      await h.ev(`document.querySelector('.t3d-pick-item button').click()`);
      const bar3d = await h.observe(BAR(), `(document.querySelector('.t3d-result-info')?.innerText ?? '').includes('GLB')`, 60 * 60e3);
      const info = await h.text('.t3d-result-info');
      const canvas = await h.ev(`(()=>{const c=document.getElementById('t3d-canvas');return c?{w:c.width,h:c.height}:null})()`);
      if (!canvas || canvas.w < 100) throw new Error('no 3D canvas: ' + JSON.stringify(canvas));

      await h.ev(`[...document.querySelectorAll('button')].find(b=>/Rig this model/.test(b.textContent)).click()`);
      await h.observe(BAR(), `/Rigged:/.test([...document.querySelectorAll('.t3d-hint')].map(x=>x.innerText).join(' | '))`, 30 * 60e3);
      const rigged = await h.ev(`(()=>{const t=[...document.querySelectorAll('.t3d-hint')].map(h=>h.innerText).join(' | ');const i=t.indexOf('Rigged:');return t.slice(i,i+100)})()`);

      await h.ev(`[...document.querySelectorAll('button')].find(b=>/Animate/.test(b.textContent)).click()`);
      await h.observe(BAR(), `/moves on/.test([...document.querySelectorAll('.t3d-hint')].map(x=>x.innerText).join(' | '))`, 30 * 60e3);
      const animated = await h.ev(`(()=>{const t=[...document.querySelectorAll('.t3d-hint')].map(h=>h.innerText).join(' | ');const i=t.indexOf('moves on');return t.slice(Math.max(0,i-25),i+60)})()`);
      return { caption: exp, pictures: stats.map(s => `${s.w}x${s.h} std ${s.std}`), glb: info.slice(0, 140), rigged, animated };
    },
  },

  // A photo -> TRELLIS.2 -> GLB. (The rig and animation steps are covered once, by /text-to-3d#flux3.)
  '/image-to-3d': {
    run: async h => {
      await h.clickText('.t3d-mode', /Image to 3D/);
      await h.clickText('.sample-btn', /Crown/);
      await h.waitFor(`!!document.querySelector('.t3d-btn')`, 60000, 'the Generate button after choosing the sample');
      await h.click('.t3d-btn');
      await h.observe(BAR(), `(document.querySelector('.t3d-result-info')?.innerText ?? '').includes('GLB')`, 60 * 60e3);
      const info = await h.text('.t3d-result-info');
      const canvas = await h.ev(`(()=>{const c=document.getElementById('t3d-canvas');return c?{w:c.width,h:c.height}:null})()`);
      if (!canvas || canvas.w < 100) throw new Error('no 3D canvas: ' + JSON.stringify(canvas));
      return { glb: info.slice(0, 160), canvas };
    },
  },
};

// Several pictures in a row on /generate (the transformer stays resident between them): the third picture is where a lost WebGPU device showed up (FLUX 3, 10-20).
function severalPictures(model, caption) {
  return {
    run: async h => {
      await h.selectValue('select.gen-setting-select', model);
      await h.selectValue('select.gen-setting-select', '4');
      await h.type('.gen-prompt-input', 'a red fox standing in fresh snow at sunrise');
      if (caption) await h.ev(`(()=>{const c=[...document.querySelectorAll('label')].find(l=>/full caption/.test(l.textContent))?.querySelector('input[type=checkbox]');if(c&&!c.checked)c.click();return !!c;})()`);
      await h.click('.gen-btn');
      const samples = await h.observe(BAR(), `document.querySelectorAll('.gen-grid-item img').length >= 4 && document.querySelector('.gen-btn')?.innerText.trim() === 'Generate'`, 60 * 60e3);
      const stats = [];
      for (let i = 0; i < 4; i++) stats.push(await assertPicture(h, `.gen-grid-item:nth-child(${i + 1}) img`, `picture ${i + 1}`));
      return { pictures: stats.map(s => `${s.w}x${s.h} std ${s.std}`), barMaxPct: Math.max(...samples.map(s => s.pct)) };
    },
  };
}
scenarios['/generate#flux3x4'] = severalPictures('flux3', true);
scenarios['/generate#flux2x4'] = severalPictures('flux2', false);

// "Change this picture" in the Text to 3D picker (FLUX reference-image editing): draw 4 pictures, tell the model to change the first one, and the edited picture
// must appear next to it, labelled, still a real picture, and actually redder (the instruction asks for red paint). FLUX.2 klein: the quick one of the two models.
const redness = sel => `(()=>{const img=document.querySelector(${JSON.stringify(sel)});if(!img||!img.naturalWidth)return null;const c=document.createElement('canvas');c.width=64;c.height=64;const x=c.getContext('2d');x.drawImage(img,0,0,64,64);
  const d=x.getImageData(0,0,64,64).data;let r=0,o=0,n=0;for(let i=0;i<d.length;i+=4){r+=d[i]-(d[i+1]+d[i+2])/2;n++;}return +(r/n).toFixed(2);})()`;
function editPicture(model, expand) {
  return {
  run: async h => {
    await h.clickText('.t3d-mode', /Text to 3D/);
    await h.selectValue('select.t3d-select', model);
    await h.selectValue('select.t3d-select', '4');
    await h.type('.t3d-prompt', 'a wooden treasure chest with iron bands on a plain white background');
    if (expand) await h.ev(`(()=>{const c=[...document.querySelectorAll('label')].find(l=>/full caption/.test(l.textContent))?.querySelector('input[type=checkbox]');if(c&&!c.checked)c.click();return !!c;})()`);   // FLUX 3: the Qwen3 caption writer draws the first pictures
    await h.click('.t3d-btn');
    await h.observe(BAR(), `document.querySelectorAll('.t3d-pick-item').length >= 4 && !document.querySelector('.t3d-stage.running')`, 60 * 60e3);
    const before = await h.ev(redness('.t3d-pick-item:nth-child(1) img'));
    await assertPicture(h, '.t3d-pick-item:nth-child(1) img', 'picture 1');
    await h.type('.pic-edit-input', 'paint the whole chest bright red');
    await h.waitFor(`!![...document.querySelectorAll('.t3d-pick-item button')].find(b=>/Change this picture/.test(b.textContent) && !b.disabled)`, 15000, 'the Change this picture button enabled by the typed text');
    await h.clickText('.t3d-pick-item button', /Change this picture/);
    await h.observe(BAR(), `document.querySelectorAll('.t3d-pick-item').length >= 5 && !document.querySelector('.t3d-stage.running')`, 30 * 60e3);
    const edited = await h.ev(redness('.t3d-pick-item:nth-child(2) img'));
    await assertPicture(h, '.t3d-pick-item:nth-child(2) img', 'the changed picture');
    const note = await h.ev(`document.querySelector('.t3d-pick-item:nth-child(2) .t3d-hint')?.innerText ?? ''`);
    if (!/changed from seed/.test(note)) throw new Error('the changed picture is not labelled: ' + JSON.stringify(note));
    if (!(edited > before + 8)) throw new Error(`"paint it red" did not make the picture redder: red excess ${before} -> ${edited}`);
    return { note, redExcessBefore: before, redExcessAfter: edited, pictures: await h.ev(`document.querySelectorAll('.t3d-pick-item').length`) };
  },
  };
}
scenarios['/text-to-3d#editpicture'] = editPicture('flux2', false);
scenarios['/text-to-3d#editpicture-flux3'] = editPicture('flux3', true);

// "Change this picture" on the Image Generation page: one picture with FLUX.2 klein, then the edit box under it; the changed picture joins it in the grid, labelled and redder.
scenarios['/generate#editpicture'] = {
  run: async h => {
    await h.selectValue('select.gen-setting-select', 'flux2');
    await h.type('.gen-prompt-input', 'a wooden treasure chest with iron bands on a plain white background');
    await h.click('.gen-btn');
    await h.observe(BAR(), `!!document.querySelector('.gen-result-img') && !!document.querySelector('.pic-edit-input')`, 40 * 60e3);
    const before = await h.ev(redness('.gen-result-img'));
    await assertPicture(h, '.gen-result-img', 'the first picture');
    await h.type('.pic-edit-input', 'paint the whole chest bright red');
    await h.waitFor(`!!document.querySelector('.pic-edit-btn') && !document.querySelector('.pic-edit-btn').disabled`, 15000, 'the Change this picture button enabled');
    await h.click('.pic-edit-btn');
    await h.observe(BAR(), `!![...document.querySelectorAll('.gen-grid-item .gen-info-item')].find(x => /changed from seed/.test(x.innerText))`, 40 * 60e3);   // the changed picture is added when the edit is done
    const edited = await h.ev(redness('.gen-grid-item:nth-child(2) img'));
    await assertPicture(h, '.gen-grid-item:nth-child(2) img', 'the changed picture');
    const note = await h.ev(`document.querySelector('.gen-grid-item:nth-child(2) .gen-info-item')?.innerText ?? ''`);
    if (!/changed from seed/.test(note)) throw new Error('the changed picture is not labelled: ' + JSON.stringify(note));
    if (!(edited > before + 8)) throw new Error(`"paint it red" did not make the picture redder: red excess ${before} -> ${edited}`);
    return { note, redExcessBefore: before, redExcessAfter: edited, pictures: await h.ev(`document.querySelectorAll('.gen-grid-item').length`) };
  },
};

// "Change this picture" in Image to 3D: a sample picture is dropped, changed before it is built (the dropped picture is replaced, Restore brings the original back).
scenarios['/image-to-3d#editpicture'] = {
  run: async h => {
    await h.clickText('.t3d-mode', /Image to 3D/);
    await h.clickText('.sample-btn', /Crown/);
    await h.waitFor(`!!document.querySelector('.pic-edit-input') && !!document.querySelector('.image-dropzone img')`, 120000, 'the sample picture loaded');
    const src0 = await h.ev(`document.querySelector('.image-dropzone img').src.length`);
    const before = await h.ev(redness('.image-dropzone img'));
    await h.type('.pic-edit-input', 'make the whole crown bright red');
    await h.waitFor(`!!document.querySelector('.pic-edit-btn') && !document.querySelector('.pic-edit-btn').disabled`, 15000, 'the Change this picture button enabled');
    await h.click('.pic-edit-btn');
    await h.waitFor(`!![...document.querySelectorAll('button')].find(b=>/Restore the original picture/.test(b.textContent))`, 40 * 60e3, 'the changed picture (the Restore button appears with it)');
    await h.waitFor(`!document.querySelector('.pic-edit-btn').disabled`, 60000, 'the edit to finish');
    const edited = await h.ev(redness('.image-dropzone img'));
    await assertPicture(h, '.image-dropzone img', 'the changed picture');
    if (!(edited > before + 8)) throw new Error(`"make it red" did not make the picture redder: red excess ${before} -> ${edited}`);
    await h.clickText('button', /Restore the original picture/);
    await h.waitFor(`!![...document.querySelectorAll('button')].every(b=>!/Restore the original picture/.test(b.textContent))`, 30000, 'the original restored');
    const restored = await h.ev(redness('.image-dropzone img'));
    if (Math.abs(restored - before) > 1) throw new Error(`Restore did not bring the original back: red excess ${before} -> ${edited} -> ${restored}`);
    return { redExcessBefore: before, redExcessChanged: edited, redExcessRestored: restored, srcLen: src0 };
  },
};
