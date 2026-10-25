// Scenario for the video demo: /video (Wan 2.2 TI2V-5B text to video). See _routes.mjs for the scenario contract.
//
// The Wan files are not on models.spawndev.com yet, so the page's "Model server" field points at a local file server laid out like the models host:
//   node tools/unirig-harness/serve-files.mjs D:/Projects/wan-serve 8768        (D:/Projects/wan-serve/LostBeard/Wan2.2-TI2V-5B-ILGPU/resolve/main = junction to D:/Projects/wan)
// Env: WAN_MODEL_SERVER (default http://127.0.0.1:8768). This is the heaviest demo (13 GB of weights into OPFS, minutes of GPU work): run it alone on the GPU.

// Pixel statistics of an <img> (drawn into a 64x64 canvas): a blank or constant picture has std ~ 0, a drawn one much more.
const imageStats = sel => `(()=>{const img=document.querySelector(${JSON.stringify(sel)});if(!img||!img.naturalWidth)return null;
  const c=document.createElement('canvas');c.width=64;c.height=64;const x=c.getContext('2d');x.drawImage(img,0,0,64,64);const d=x.getImageData(0,0,64,64).data;
  let s=0,s2=0,n=0;for(let i=0;i<d.length;i+=4){const v=(d[i]+d[i+1]+d[i+2])/3;s+=v;s2+=v*v;n++;}const m=s/n;return {w:img.naturalWidth,h:img.naturalHeight,mean:+m.toFixed(1),std:+Math.sqrt(Math.max(0,s2/n-m*m)).toFixed(1)};})()`;

// Set the nth <select class="vid-select"> to a value the way a user does (Blazor listens to "change").
const setSelect = (n, value) => `(()=>{const s=document.querySelectorAll('select.vid-select')[${n}];s.value=${JSON.stringify(value)};s.dispatchEvent(new Event('change',{bubbles:true}));return s.value;})()`;

export const scenarios = {
  '/video': {
    expectConsole: [/status of 404.*\/video/],                  // a deep link on GitHub Pages is answered by 404.html (the SPA redirect); the local static server has no such response
    run: async h => {
      const ltx = process.env.VIDEO_MODEL === 'ltx';                                  // VIDEO_MODEL=ltx: LTX-Video 2B (served from LTX_MODEL_SERVER, default http://127.0.0.1:8769) instead of Wan
      const server = (ltx ? process.env.LTX_MODEL_SERVER : process.env.WAN_MODEL_SERVER) || (ltx ? 'http://127.0.0.1:8769' : 'http://127.0.0.1:8768');   // =default: leave the field empty = the real models host (what the live site does)
      await h.waitFor(`document.querySelectorAll('select.vid-select').length >= 4`, 180e3, 'the video page renders');   // GitHub Pages serves a deep link through its 404.html and the WASM app boots after
      // selects: 0 = model, 1 = size, 2 = frames (then steps, transformer for Wan only)
      if (ltx) { await h.ev(setSelect(0, 'ltx')); await h.waitFor(`document.querySelectorAll('select.vid-select').length === 3`, 30e3, 'the LTX settings render'); await h.ev(setSelect(1, '480x320')); await h.ev(setSelect(2, '17')); }
      else { await h.ev(setSelect(1, '512x288')); await h.ev(setSelect(2, '9')); }      // the Wan defaults are the Turbo transformer: 4 steps, guidance 1
      if (server !== 'default') {
        await h.ev(`(()=>{document.querySelector('details.vid-advanced').open=true;})()`);
        await h.type('input[placeholder="http://127.0.0.1:8767"]', server);
      }
      await h.click('.vid-btn');
      await h.waitFor(`(()=>{const r=document.querySelector('.vid-result');return !!r&&r.style.display!=='none'&&document.querySelector('#vid-player')?.naturalWidth>0;})()`, 90 * 60e3, 'the video player shows a frame');
      const st = await h.ev(imageStats('#vid-player'));
      if (!st) throw new Error('no frame in the video player');
      if (st.std < 8) throw new Error(`the video frame is blank or constant (std ${st.std}, mean ${st.mean})`);
      await h.shot('video-frame');
      return { frame: st, info: await h.text('.vid-info-item') };
    },
  },
};
