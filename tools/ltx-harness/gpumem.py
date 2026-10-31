"""Peak GPU memory per pipeline stage (NVIDIA, desktop): samples nvidia-smi every 100 ms (with its own timestamps) while an Examples/11.VideoGen.Console run executes, and
attributes the samples to the console's stage lines by the console's own clock ("[  12 s] denoise: ..."; the piped lines arrive late by an irregular delay, so the clock is
anchored on the least-delayed line: stage boundaries +-1 s). The baseline (memory in use before the run) is subtracted. Use --no-frames so saving the frames is not counted.
  py -3.13 gpumem.py OUT_PREFIX -- <VideoGen.Console.exe> <args...>
Writes OUT_PREFIX.log (the console output), OUT_PREFIX.mem.txt (the table) and OUT_PREFIX.samples.json.
"""
import re
import subprocess
import sys
import time

prefix = sys.argv[1]
cmd = sys.argv[sys.argv.index("--") + 1:]
base = int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip())
smi = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,memory.used", "--format=csv,noheader,nounits", "-lms", "100"], stdout=subprocess.PIPE, text=True)
samples = []
import datetime
import threading


def reader():
    # the sample's own clock (nvidia-smi's timestamp): a piped reader receives the lines late, in bursts
    for line in smi.stdout:
        try:
            ts, v = line.strip().split(",")
            t = datetime.datetime.strptime(ts.strip(), "%Y/%m/%d %H:%M:%S.%f").timestamp()
            samples.append((t, int(v)))
        except ValueError:
            pass


threading.Thread(target=reader, daemon=True).start()
t0 = time.time()
events = []  # (time, stage text)
p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=0)
buf = ""
last = None
log = open(prefix + ".log", "w", encoding="utf-8")
while True:
    ch = p.stdout.read(1)
    if not ch:
        break
    if ch in "\r\n":
        line = buf.strip()
        buf = ""
        if line:
            log.write(line + "\n")
            m = re.match(r"\[\s*(\d+) s\] ([a-z]+): (.*)", line)
            if m:
                key = m.group(2) + (" (second pass)" if m.group(3).startswith("second pass") else "") + (" load" if "loading" in m.group(3) else "")
                if key != last:
                    events.append((time.time(), key, int(m.group(1))))
                    print(f"event {key} at {time.time() - t0:.1f} s (console says {m.group(1)} s)", flush=True)
                    last = key
    else:
        buf += ch
p.wait()
time.sleep(0.5)
smi.terminate()
log.close()
# the lines reach this reader late by an irregular delay; the console stamps each with its own clock (whole seconds). Anchor its clock on the least-delayed line.
anchor = min(t - cs for t, _, cs in events)
end = time.time()
events = [(anchor + cs, name) for _, name, cs in events] + [(end, "end")]
import json
json.dump(samples, open(prefix + ".samples.json", "w"))
out = [f"baseline {base} MiB; total peak {max(v for _, v in samples) - base} MiB over baseline; run {end - t0:.0f} s (stage boundaries +-1 s)"]
for (ta, name), (tb, _) in zip(events, events[1:]):
    vals = [v for t, v in samples if ta <= t < tb]
    if vals:
        out.append(f"{name:28s} {ta - t0:6.1f}-{tb - t0:6.1f} s  peak {max(vals) - base:6d} MiB over baseline")
open(prefix + ".mem.txt", "w").write("\n".join(out) + "\n")
print("\n".join(out))
