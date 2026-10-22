// The demo's video preview: the frames of a generated clip arrive as image data URLs (addFrame), are played in an <img> at the clip's frame rate (play),
// and can be encoded to a WebM file with the browser's own MediaRecorder (download). Nothing leaves the page.
//
//   videoPlayer.clear(id)                 forget the frames of the <img id>
//   videoPlayer.addFrame(id, dataUrl)     append one frame (decoded ahead of time so playback does not stall)
//   videoPlayer.play(id, fps)             loop the clip
//   videoPlayer.stop(id)                  stop the loop
//   videoPlayer.download(id, fps, name)   draw the frames to a canvas, record it with MediaRecorder at real time, save <name> (WebM)
(function () {
    "use strict";
    const clips = {};
    const clip = id => clips[id] || (clips[id] = { frames: [], timer: 0, index: 0 });

    function decode(url) {
        const img = new Image();
        img.src = url;
        return img.decode ? img.decode().then(() => img).catch(() => img) : Promise.resolve(img);
    }

    window.videoPlayer = {
        clear(id) { this.stop(id); clips[id] = { frames: [], timer: 0, index: 0 }; },
        addFrame(id, url) { clip(id).frames.push({ url, ready: decode(url) }); },
        play(id, fps) {
            this.stop(id);
            const c = clip(id), el = document.getElementById(id);
            if (!el || c.frames.length === 0) return;
            c.index = 0;
            el.src = c.frames[0].url;
            c.timer = setInterval(() => {
                c.index = (c.index + 1) % c.frames.length;
                el.src = c.frames[c.index].url;
            }, 1000 / (fps || 24));
        },
        stop(id) { const c = clips[id]; if (c && c.timer) { clearInterval(c.timer); c.timer = 0; } },
        async download(id, fps, name) {
            const c = clip(id);
            if (c.frames.length === 0) return;
            const first = await c.frames[0].ready;
            const canvas = document.createElement("canvas");
            canvas.width = first.naturalWidth; canvas.height = first.naturalHeight;
            const ctx = canvas.getContext("2d");
            const stream = canvas.captureStream(fps || 24);
            const type = ["video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm"].find(t => MediaRecorder.isTypeSupported(t));
            const rec = new MediaRecorder(stream, { mimeType: type, videoBitsPerSecond: 12000000 });
            const parts = [];
            rec.ondataavailable = e => { if (e.data.size) parts.push(e.data); };
            const done = new Promise(resolve => { rec.onstop = resolve; });
            rec.start();
            const delay = 1000 / (fps || 24);
            for (let loop = 0; loop < 2; loop++)           // twice: the clip is short, and the first frames of a recording are sometimes dropped
                for (const f of c.frames) { ctx.drawImage(await f.ready, 0, 0); await new Promise(r => setTimeout(r, delay)); }
            rec.stop();
            await done;
            const a = document.createElement("a");
            a.href = URL.createObjectURL(new Blob(parts, { type: "video/webm" }));
            a.download = name || "video.webm";
            document.body.appendChild(a); a.click(); a.remove();
            setTimeout(() => URL.revokeObjectURL(a.href), 60000);
        },
    };
})();
