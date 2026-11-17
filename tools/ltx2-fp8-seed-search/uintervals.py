import numpy as np, json, struct, glob, re
CK = 'D:/spawndev-ltx2/lightricks/ltx-2-19b-dev-fp8.safetensors'
LO = 'D:/spawndev-ltx2/lightricks/ltx-2-19b-distilled-lora-384.safetensors'
D = 'F:/spawndev-ltx2/comfyui-dumps/lora/fp8_rounding/'
import os; OUT = os.path.dirname(os.path.abspath(__file__)) + '/u_intervals.npy'
ROWS, N = 4, 4096          # the first ROWS rows of attn1.to_q (row-major element order)

def opener(p):
    f = open(p, 'rb'); n = struct.unpack('<Q', f.read(8))[0]; h = json.loads(f.read(n)); return f, h, 8 + n
def raw(fhb, k, start=0, count=None):
    f, h, base = fhb; a, b = h[k]['data_offsets']; f.seek(base + a + start); return f.read((b - a - start) if count is None else count), h[k]
codes = np.arange(256); e = (codes >> 3) & 15; m = codes & 7
val = np.where(e == 0, m / 8 * 2.0 ** -6, (1 + m / 8) * 2.0 ** (e - 7)); val = np.where(codes & 0x80, -val, val); val[(codes & 0x7F) == 0x7F] = np.nan
pos = np.unique(val[codes < 0x7F])

ck, lo = opener(CK), opener(LO)
key = 'model.diffusion_model.transformer_blocks.1.attn1.to_q'
W = val[np.frombuffer(raw(ck, key + '.weight', 0, ROWS * N)[0], np.uint8)].reshape(ROWS, N).astype(np.float32)
def bf(k, start, count, shape):
    r, _ = raw(lo, k, start * 2, count * 2); return (np.frombuffer(r, np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(shape)
A = bf('diffusion_model.transformer_blocks.1.attn1.to_q.lora_A.weight', 0, 384 * N, (384, N))
B = bf('diffusion_model.transformer_blocks.1.attn1.to_q.lora_B.weight', 0, ROWS * 384, (ROWS, 384))
BA = (B @ A).astype(np.float32)
lo_u = np.zeros(ROWS * N); hi_u = np.ones(ROWS * N); used = []
for path in sorted(glob.glob(D + 'fp8r_q_s*_n10_*weight_raw_float8_e4m3fn.bin')):
    tag = re.search(r'fp8r_q_s([0-9]+)_n10', path).group(1)
    s = float('0.' + tag[1:]) if tag.startswith('0') else float(tag)
    scale = np.fromfile(path.replace('weight_raw_float8_e4m3fn.bin', 'weight_scale_raw_float32.bin'), np.float32)[0]
    stored = np.fromfile(path, np.uint8, count=ROWS * N)
    exact = (W + np.float32(s) * BA).astype(np.float32).ravel()
    y = np.abs(exact / scale).astype(np.float64)
    idx = np.clip(np.searchsorted(pos, y, side='right') - 1, 0, len(pos) - 2)
    l, h = pos[idx], pos[idx + 1]; frac = (y - l) / (h - l)
    sv = np.abs(val[stored]); up = sv == h; down = sv == l
    # u < frac -> up: up => u in [0, frac); down => u in [frac, 1)
    lo_u = np.where(down, np.maximum(lo_u, frac), lo_u); hi_u = np.where(up, np.minimum(hi_u, frac), hi_u)
    used.append(s)
w = hi_u - lo_u
print('strengths', sorted(used), '| first', ROWS * N, 'elements: conflicts', int((w <= 0).sum()), 'width median', np.median(w), 'mean', w.mean())
print('bits of information in the first 64 elements:', float(-np.log2(np.clip(w[:64], 1e-9, 1)).sum()))
np.save(OUT, np.stack([lo_u, hi_u]))
print('first 8 intervals:', [(round(a, 3), round(b, 3)) for a, b in zip(lo_u[:8], hi_u[:8])])
