// The demo's video preview: the frames of a generated clip arrive as image data URLs (addFrame), are played in an <img> at the clip's frame rate (play),
// and can be encoded to a WebM file with the browser's own MediaRecorder (download). Nothing leaves the page.
//
//   videoPlayer.clear(id)                   forget the frames (and the sound) of the <img id>
//   videoPlayer.addFrame(id, dataUrl)       append one frame (decoded ahead of time so playback does not stall)
//   videoPlayer.setAudio(id, samples, rate) a mono sound track (floats in [-1, 1]) that starts with frame 0; null removes it
//   videoPlayer.audioSeconds(id)            the sound track's length (0 without one)
//   videoPlayer.play(id, fps)               loop the clip; the sound plays once, from the first frame (play again, e.g. Replay, to hear it again)
//   videoPlayer.stop(id)                    stop the loop and the sound
//   videoPlayer.download(id, fps, name)     draw the frames to a canvas, record it (and the sound) with MediaRecorder at real time, save <name> (WebM)
(function () {
    "use strict";
    const clips = {};
    const clip = id => clips[id] || (clips[id] = { frames: [], timer: 0, index: 0, audio: null, source: null });
    let audioCtx = null;
    const ctxAudio = () => audioCtx || (audioCtx = new AudioContext());

    function decode(url) {
        const img = new Image();
        img.src = url;
        return img.decode ? img.decode().then(() => img).catch(() => img) : Promise.resolve(img);
    }

    // start the clip's sound now, into the speakers or a recording destination
    function startAudio(c, destination) {
        if (!c.audio) return null;
        const ac = ctxAudio();
        if (ac.state === "suspended") ac.resume();
        const src = ac.createBufferSource();
        src.buffer = c.audio;
        src.connect(destination || ac.destination);
        src.start();
        return src;
    }
    function stopAudio(c) { if (c.source) { try { c.source.stop(); } catch { } c.source = null; } }

    window.videoPlayer = {
        clear(id) { this.stop(id); clips[id] = { frames: [], timer: 0, index: 0, audio: null, source: null }; },
        addFrame(id, url) { clip(id).frames.push({ url, ready: decode(url) }); },
        setAudio(id, samples, rate) {
            const c = clip(id);
            stopAudio(c);
            if (!samples || samples.length === 0) { c.audio = null; return; }
            const data = samples instanceof Float32Array ? samples : Float32Array.from(samples);
            const buf = ctxAudio().createBuffer(1, data.length, rate || 24000);
            buf.copyToChannel(data, 0);
            c.audio = buf;
        },
        audioSeconds(id) { const c = clips[id]; return c && c.audio ? c.audio.duration : 0; },
        play(id, fps) {
            this.stop(id);
            const c = clip(id), el = document.getElementById(id);
            if (!el || c.frames.length === 0) return;
            c.index = 0;
            el.src = c.frames[0].url;
            c.source = startAudio(c);
            c.timer = setInterval(() => {
                c.index = (c.index + 1) % c.frames.length;
                el.src = c.frames[c.index].url;
            }, 1000 / (fps || 24));
        },
        stop(id) { const c = clips[id]; if (c) { if (c.timer) { clearInterval(c.timer); c.timer = 0; } stopAudio(c); } },
        async download(id, fps, name) {
            const c = clip(id);
            if (c.frames.length === 0) return;
            const first = await c.frames[0].ready;
            const canvas = document.createElement("canvas");
            canvas.width = first.naturalWidth; canvas.height = first.naturalHeight;
            const ctx = canvas.getContext("2d");
            const stream = canvas.captureStream(fps || 24);
            let dest = null;
            if (c.audio) { dest = ctxAudio().createMediaStreamDestination(); dest.stream.getAudioTracks().forEach(t => stream.addTrack(t)); }
            const types = c.audio ? ["video/webm;codecs=vp9,opus", "video/webm;codecs=vp8,opus", "video/webm"] : ["video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm"];
            const type = types.find(t => MediaRecorder.isTypeSupported(t));
            const rec = new MediaRecorder(stream, { mimeType: type, videoBitsPerSecond: 12000000 });
            const parts = [];
            rec.ondataavailable = e => { if (e.data.size) parts.push(e.data); };
            const done = new Promise(resolve => { rec.onstop = resolve; });
            rec.start();
            const delay = 1000 / (fps || 24);
            // the first frames of a recording are sometimes dropped: without sound the clip is recorded twice; with sound once, after a short hold on the first frame (picture and sound start together)
            const loops = c.audio ? 1 : 2;
            if (c.audio) { ctx.drawImage(first, 0, 0); await new Promise(r => setTimeout(r, 300)); }
            const src = dest ? startAudio(c, dest) : null;
            for (let loop = 0; loop < loops; loop++)
                for (const f of c.frames) { ctx.drawImage(await f.ready, 0, 0); await new Promise(r => setTimeout(r, delay)); }
            if (src) { try { src.stop(); } catch { } }
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
