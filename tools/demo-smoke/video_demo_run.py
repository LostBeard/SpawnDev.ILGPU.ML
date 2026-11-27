"""Drive a deployed /video demo exactly as a visitor does (defaults, click Generate) in a separate Chrome profile and record what it costs.

Why this exists (2026-11-27): the live /video demo was released without being driven. TJ ran it with the defaults: it showed no download progress,
and every interrupted attempt downloaded its multi-GB file again from byte 0 while the earlier attempt's bytes stayed on disk, until his OS drive was
nearly full. No unit test runs a demo page's own default path, and the cache's resume test "interrupted" a download with a clean close.

Run it after every GitHub Pages deploy of the demo (and against a local publish before one):

    python video_demo_run.py --profile F:/_ml_demo_profile --url https://lostbeard.github.io/SpawnDev.ILGPU.ML/video --log run.jsonl --kill-after-gb 3

The profile goes on a drive with room, NEVER the OS drive (Chrome's own profile lives there). The script does not delete the profile folder; it empties
the site's storage through the page before the run (unless --reuse) and after it (unless --keep).

Every --interval seconds it records: the page's progress text and bar, the download bar's text, bytes requested per model file and how often a file was
started from byte 0, the storage entries, and the profile folder's real size on disk.

--kill-after-gb N ends Chrome the hard way (as a crash or a closed browser does) once N GB were requested, starts it again on the same profile and
clicks Generate again. Checks, all must hold for exit code 0:
  - a video is shown;
  - download progress was visible while bytes were arriving (the #active-downloads bar), and the page's own progress line said "loading" then;
  - no file is started from byte 0 twice within a session (one-byte ranges are size probes and do not count; a range may ask past the end of
    the file, so a file's requested bytes can exceed its size by up to one segment);
  - across the kill, each file's requested bytes stay within its size plus --slack-mb (the interrupted attempt's bytes are continued, not fetched again);
  - the profile never grows by more than the models' total size plus --slack-mb (plus 0.3 GB for the browser's own files).
"""
import argparse, asyncio, json, os, shutil, subprocess, sys, time, urllib.request

import websockets

CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

FETCH_HOOK = r"""
(() => {
  if (window.__fx) return;
  window.__fx = { t0: Date.now(), byUrl: {}, log: [] };
  const orig = window.fetch.bind(window);
  window.fetch = async function (input, init) {
    const url = typeof input === 'string' ? input : input.url;
    let range = null;
    try { const h = init && init.headers; range = h ? (h instanceof Headers ? h.get('Range') : (h.Range || h.range)) : null; } catch (e) {}
    let host = ''; try { host = new URL(url, location.href).host; } catch (e) {}
    const name = url.split('?')[0].split('/').slice(-1)[0];
    const big = /\.(safetensors|onnx|gguf|bin|json)$/.test(name) && host !== location.host;
    const rec = big ? (window.__fx.byUrl[name] || (window.__fx.byUrl[name] = { n: 0, bytes: 0, starts: [], fails: 0 })) : null;
    if (rec) {
      rec.n++;
      const m = range && /bytes=(\d+)-(\d*)/.exec(range);
      if (m) {
        const s = +m[1], e = m[2] ? +m[2] : -1;
        if (e === s) return orig(input, init);                 // a one-byte range is a size probe, not a download
        if (e >= 0) rec.bytes += e - s + 1;
        if (s === 0) rec.starts.push(Math.round((Date.now() - window.__fx.t0) / 1000));
      }
    }
    try {
      const r = await orig(input, init);
      if (rec && !r.ok) { rec.fails++; window.__fx.log.push([Math.round((Date.now() - window.__fx.t0) / 1000), name, r.status, range]); }
      return r;
    } catch (e) {
      if (rec) { rec.fails++; window.__fx.log.push([Math.round((Date.now() - window.__fx.t0) / 1000), name, String(e), range]); }
      throw e;
    }
  };
})();
"""

STATUS = r"""
(async () => {
  const files = [];
  async function walk(dir, path, depth) {
    for await (const [name, h] of dir.entries()) {
      if (h.kind === 'directory') { if (depth < 3) await walk(h, path + name + '/', depth + 1); continue; }
      try {
        const f = await h.getFile();
        if (name.endsWith('.meta')) {
          const t = await f.text(); const g = k => (new RegExp('^' + k + '=(.*)$', 'm').exec(t) || [])[1];
          files.push({ name: path + name, total: +g('total'), received: +g('received'), complete: g('complete') === '1' });
        } else files.push({ name: path + name, size: f.size });
      } catch (e) { files.push({ name: path + name, error: String(e) }); }
    }
  }
  try { await walk(await navigator.storage.getDirectory(), '', 0); } catch (e) { files.push({ error: String(e) }); }
  const q = s => document.querySelector(s);
  return JSON.stringify({
    fx: window.__fx ? window.__fx.byUrl : null, fxlog: window.__fx ? window.__fx.log.slice(-10) : null,
    text: (q('.vid-progress-text') || {}).textContent || '', bar: q('.vid-progress-fill') ? q('.vid-progress-fill').style.width : null,
    downloads: (q('#active-downloads') || {}).innerText || '',
    error: (q('.vid-error') || {}).textContent || null,
    running: !!q('.vid-progress'),
    hasVideo: !!(q('.vid-result') && q('.vid-result').style.display !== 'none' && q('#vid-player') && q('#vid-player').naturalWidth > 0),
    info: (q('.vid-info-item') || {}).textContent || '', files });
})()
"""

CLEAR = r"""
(async () => {
  const root = await navigator.storage.getDirectory();
  const names = []; for await (const n of root.keys()) names.push(n);
  let failed = 0;
  for (const n of names) { try { await root.removeEntry(n, { recursive: true }); } catch (e) { failed++; } }
  return names.length + ' entries removed, ' + failed + ' failed';
})()
"""


def folder_bytes(path):
    total = 0
    for root, _, names in os.walk(path):
        for n in names:
            try: total += os.path.getsize(os.path.join(root, n))
            except OSError: pass
    return total


class Cdp:
    def __init__(self, ws): self.ws, self.id, self.console = ws, 0, []

    async def call(self, method, **params):
        self.id += 1; mid = self.id
        await self.ws.send(json.dumps({"id": mid, "method": method, "params": params}))
        while True:
            msg = json.loads(await self.ws.recv())
            if msg.get("id") == mid:
                if "error" in msg: raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result", {})
            if msg.get("method") == "Runtime.consoleAPICalled":
                self.console.append(" ".join(str(a.get("value", a.get("description", ""))) for a in msg["params"]["args"])[:600])
            elif msg.get("method") == "Runtime.exceptionThrown":
                self.console.append("EXCEPTION " + json.dumps(msg["params"]["exceptionDetails"])[:800])

    async def eval(self, expr):
        r = await self.call("Runtime.evaluate", expression=expr, awaitPromise=True, returnByValue=True)
        if "exceptionDetails" in r: raise RuntimeError(json.dumps(r["exceptionDetails"])[:500])
        return r["result"].get("value")


async def session(args, log, state, clear_first, kill_after_gb):
    """One browser session: open the page, click Generate, sample until a video, an error, the kill point or the time limit. Returns the verdict."""
    chrome = subprocess.Popen([CHROME, f"--user-data-dir={args.profile}", f"--remote-debugging-port={args.port}", "--remote-allow-origins=*",
                               "--no-first-run", "--no-default-browser-check", "--window-size=1300,1000", "about:blank"])
    print(f"session {state['session']}: chrome pid {chrome.pid}", flush=True)
    verdict, killed = "timeout", False
    try:
        target = None
        for _ in range(60):
            try:
                tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{args.port}/json", timeout=2))
                target = next(t for t in tabs if t["type"] == "page"); break
            except Exception: await asyncio.sleep(0.5)
        if target is None: return "Chrome's debugging port did not answer"
        async with websockets.connect(target["webSocketDebuggerUrl"], max_size=None) as ws:
            cdp = Cdp(ws)
            await cdp.call("Runtime.enable"); await cdp.call("Page.enable")
            await cdp.call("Page.addScriptToEvaluateOnNewDocument", source=FETCH_HOOK)
            await cdp.call("Page.navigate", url=args.url)
            for _ in range(240):
                await asyncio.sleep(0.5)
                if await cdp.eval("!!document.querySelector('button.vid-btn') && !document.querySelector('button.vid-btn').disabled"): break
            else: return "the page never showed an enabled Generate button"
            if clear_first:
                print("storage before the run:", await cdp.eval(CLEAR), flush=True)
                await cdp.call("Page.navigate", url=args.url)
                for _ in range(240):
                    await asyncio.sleep(0.5)
                    if await cdp.eval("!!document.querySelector('button.vid-btn') && !document.querySelector('button.vid-btn').disabled"): break
            print("gpu:", await cdp.eval("navigator.gpu ? navigator.gpu.requestAdapter().then(a => a && a.info ? a.info.vendor + ' ' + a.info.architecture : String(a)) : 'no navigator.gpu'"), flush=True)
            if args.setup: await cdp.eval(args.setup)
            if state["base_disk"] is None: state["base_disk"] = folder_bytes(args.profile)       # what the profile held before this run's first click
            await cdp.eval("window.__fx.t0 = Date.now(); document.querySelector('button.vid-btn').click(); 1")
            t0 = time.time(); last_down = 0
            while time.time() - t0 < args.minutes * 60:
                await asyncio.sleep(args.interval)
                st = json.loads(await cdp.eval(STATUS))
                disk = folder_bytes(args.profile); state["peak_disk"] = max(state["peak_disk"], disk)
                st["session"] = state["session"]; st["sec"] = round(time.time() - t0); st["diskGB"] = round(disk / 1e9, 2); st["console"] = cdp.console[-8:]; cdp.console.clear()
                log.write(json.dumps(st) + "\n"); log.flush()
                fx = st["fx"] or {}
                down = sum(v["bytes"] for v in fx.values())
                for k, v in fx.items():
                    state["requested"].setdefault(k, {})[state["session"]] = v["bytes"]
                    if len(v["starts"]) > 1: state["problems"].add(f"session {state['session']}: {k} was started from byte 0 {len(v['starts'])} times (at {v['starts']} s)")
                for f in st["files"]:
                    if f.get("total", 0) > 0 and f["name"].endswith(".meta"): state["totals"][f["name"].split("_main_")[-1][:-5]] = f["total"]
                arriving = down - last_down > 4e6; last_down = down
                if arriving:
                    state["samples_downloading"] += 1
                    if st["downloads"].strip(): state["samples_with_bar"] += 1
                    # the page's own line must say it is loading while a file downloads (5.8.36 said "encoding the prompt" through the 6.4 GB umT5
                    # download and "decoding the video" through the VAE's: opening a file waits for its whole download)
                    if "loading" in (st["text"] or "").lower(): state["samples_saying_loading"] += 1
                print(f"{st['sec']:5d}s disk {st['diskGB']:6.2f} GB requested {down / 1e9:6.2f} GB | page: {st['bar']} '{st['text'].strip()[:44]}' | downloads: '{' / '.join(st['downloads'].split(chr(10)))[:110]}'", flush=True)
                if st["error"]: verdict = "page error: " + st["error"]; break
                if st["hasVideo"] and not st["running"]: verdict = "video shown: " + st["info"]; break
                if kill_after_gb and down >= kill_after_gb * 1e9:
                    chrome.kill(); killed = True; verdict = f"killed at {down / 1e9:.2f} GB requested"; break
            if not killed:
                if not args.keep: print("storage after the run:", await cdp.eval(CLEAR), flush=True)
                try: await cdp.call("Browser.close")
                except Exception: pass
    except (websockets.ConnectionClosed, ConnectionError) as ex:
        if not killed: verdict = f"the browser went away: {ex}"
    finally:
        try: chrome.wait(timeout=20)
        except Exception: chrome.kill()
    return verdict


async def run(args):
    drive = os.path.splitdrive(os.path.abspath(args.profile))[0].upper()
    if drive == os.environ.get("SystemDrive", "C:").upper(): sys.exit("the profile must not be on the OS drive")
    free = shutil.disk_usage(drive + "\\").free
    if free < args.need_gb * 1e9: sys.exit(f"{drive} has {free / 1e9:.1f} GB free, the run needs {args.need_gb} GB")
    os.makedirs(args.profile, exist_ok=True)
    state = {"session": 1, "base_disk": None, "peak_disk": 0, "requested": {}, "totals": {}, "problems": set(), "samples_downloading": 0, "samples_with_bar": 0, "samples_saying_loading": 0}
    with open(args.log, "w", encoding="utf-8") as log:
        verdict = await session(args, log, state, clear_first=not args.reuse, kill_after_gb=args.kill_after_gb)
        print("SESSION 1:", verdict, flush=True)
        if verdict.startswith("killed"):
            await asyncio.sleep(5)
            state["session"] = 2
            verdict = await session(args, log, state, clear_first=False, kill_after_gb=0)
            print("SESSION 2:", verdict, flush=True)
    slack = args.slack_mb * 1e6
    models = sum(state["totals"].values())
    print("\nfile: size, bytes requested per session")
    for name, per in sorted(state["requested"].items()):
        total = state["totals"].get(name)
        asked = sum(per.values())
        print(f"  {name}: {total} bytes; requested {per} = {asked}")
        if total and asked > total + slack: state["problems"].add(f"{name}: {asked / 1e9:.2f} GB requested for a {total / 1e9:.2f} GB file (allowed: its size + {args.slack_mb} MB)")
    grew = state["peak_disk"] - (state["base_disk"] or 0)
    print(f"models: {models / 1e9:.2f} GB; the profile grew by at most {grew / 1e9:.2f} GB during the run (it held {(state['base_disk'] or 0) / 1e9:.2f} GB before)")
    if models and grew > models + slack + 0.3e9: state["problems"].add(f"the profile grew by {grew / 1e9:.2f} GB for {models / 1e9:.2f} GB of models")
    print(f"samples with bytes arriving: {state['samples_downloading']}, of which the page showed the download bar: {state['samples_with_bar']}")
    if state["samples_downloading"] and state["samples_with_bar"] < 0.8 * state["samples_downloading"]:
        state["problems"].add(f"download progress was visible in {state['samples_with_bar']} of {state['samples_downloading']} samples taken while bytes were arriving")
    print(f"samples with bytes arriving: {state['samples_downloading']}, of which the page's own line said it was loading: {state['samples_saying_loading']}")
    if state["samples_downloading"] and state["samples_saying_loading"] < 0.8 * state["samples_downloading"]:
        state["problems"].add(f"the page's line said it was loading in {state['samples_saying_loading']} of {state['samples_downloading']} samples taken while bytes were arriving")
    print("VERDICT:", verdict)
    for p in sorted(state["problems"]): print("PROBLEM:", p)
    ok = verdict.startswith("video shown") and not state["problems"]
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")      # the page text has characters a Windows console code page cannot encode
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True); ap.add_argument("--url", required=True); ap.add_argument("--log", required=True)
    ap.add_argument("--port", type=int, default=9377); ap.add_argument("--minutes", type=float, default=45); ap.add_argument("--interval", type=float, default=10)
    ap.add_argument("--need-gb", type=float, default=40, help="free space the profile's drive must have")
    ap.add_argument("--slack-mb", type=float, default=400, help="bytes allowed beyond a file's size: one part (128 MiB) plus the segments in flight, twice")
    ap.add_argument("--kill-after-gb", type=float, default=0, help="end Chrome the hard way once this many GB were requested, then run again on the same profile")
    ap.add_argument("--setup", default="", help="JavaScript run before the click (select a model, a size)")
    ap.add_argument("--keep", action="store_true", help="leave the site's storage as the run left it"); ap.add_argument("--reuse", action="store_true", help="start from the storage as it is")
    sys.exit(asyncio.run(run(ap.parse_args())))
