// One picture through /image-to-3d, and the GLB the page builds saved to disk, for comparing generations (deployed site vs a local build vs the desktop).
//   AUDIT_SCENARIOS=tools/audit-scenarios/trellis-compare.mjs AUDIT_IMAGE=C:/x/pic.png AUDIT_OUT_GLB=C:/x/out.glb node tools/audit-demo-models.mjs <baseUrl> <outDir> /image-to-3d
//   optional: AUDIT_RESOLUTION=512|1024|1024c|1536c  AUDIT_FACES=50000|100000|150000|300000|500000|1000000  AUDIT_TEXTURE=1024|2048  AUDIT_SEED=42
const BAR = `JSON.stringify({ label: document.querySelector('.mp-label')?.innerText ?? '', pct: parseFloat(document.querySelector('.mp-fill')?.style.width) || 0,
  stage: (document.querySelector('.t3d-stage.running')?.innerText ?? '').replace(/\s+/g, ' ') })`;

export const scenarios = {
  '/image-to-3d': {
    allowHuggingFace: true,
    run: async h => {
      const image = process.env.AUDIT_IMAGE, out = process.env.AUDIT_OUT_GLB;
      if (!image || !out) throw new Error('set AUDIT_IMAGE and AUDIT_OUT_GLB');
      await h.clickText('.t3d-mode', /Image to 3D/);
      await h.upload('input.dropzone-input', image);
      await h.waitFor(`!!document.querySelector('.t3d-btn')`, 600000, 'the Generate button after choosing the picture');
      if (process.env.AUDIT_RESOLUTION) await h.selectValue('select.t3d-select', process.env.AUDIT_RESOLUTION);   // the Detail select (512 | 1024 | 1024c | 1536c); the first select with that value
      if (process.env.AUDIT_FACES) await h.selectValue('select.t3d-select', process.env.AUDIT_FACES);
      if (process.env.AUDIT_TEXTURE) await h.selectValue('select.t3d-select', process.env.AUDIT_TEXTURE);
      if (process.env.AUDIT_SEED) await h.ev(`(()=>{const e=document.querySelector('input.t3d-num');e.value=${JSON.stringify(process.env.AUDIT_SEED)};e.dispatchEvent(new Event('input',{bubbles:true}));e.dispatchEvent(new Event('change',{bubbles:true}));})()`);
      await h.click('.t3d-btn');
      await h.observe(BAR, `(document.querySelector('.t3d-result-info')?.innerText ?? '').includes('GLB')`, 60 * 60e3);
      const info = await h.text('.t3d-result-info');
      await h.saveDownload('button', /Download \.glb/, out);
      return { glb: info.slice(0, 160), saved: out };
    },
  },
};
