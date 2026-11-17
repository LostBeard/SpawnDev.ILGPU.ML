# As uintervals.py, for the COLUMN-major hypothesis: the first COUNT elements of column 0 (rows 0..COUNT-1) of attn1.to_q, in that order.
# Writes real_col.bin (lo[COUNT] then hi[COUNT], float64): the harness input for "the uniform stream runs down the columns".
import numpy as np, json, struct, glob, re, os
CK = 'D:/spawndev-ltx2/lightricks/ltx-2-19b-dev-fp8.safetensors'
LO = 'D:/spawndev-ltx2/lightricks/ltx-2-19b-distilled-lora-384.safetensors'
D = 'F:/spawndev-ltx2/comfyui-dumps/lora/fp8_rounding/'
OUT = os.path.dirname(os.path.abspath(__file__)) + '/real_col.bin'
COUNT, N = 200, 4096

def opener(p):
    f = open(p, 'rb'); n = struct.unpack('<Q', f.read(8))[0]; h = json.loads(f.read(n)); return f, h, 8 + n
def raw(fhb, k, start, count):
    f, h, base = fhb; a, b = h[k]['data_offsets']; f.seek(base + a + start); return f.read(count)
codes = np.arange(256); e = (codes >> 3) & 15; m = codes & 7
val = np.where(e == 0, m / 8 * 2.0 ** -6, (1 + m / 8) * 2.0 ** (e - 7)); val = np.where(codes & 0x80, -val, val); val[(codes & 0x7F) == 0x7F] = np.nan
pos = np.unique(val[codes < 0x7F])

ck, lo = opener(CK), opener(LO)
key = 'model.diffusion_model.transformer_blocks.1.attn1.to_q'
W = np.array([val[raw(ck, key + '.weight', r * N, 1)[0]] for r in range(COUNT)], np.float32)                 # column 0 of rows 0..COUNT-1
def bf(k, start, count, shape):
    return (np.frombuffer(raw(lo, k, start * 2, count * 2), np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(shape)
A = bf('diffusion_model.transformer_blocks.1.attn1.to_q.lora_A.weight', 0, 384 * N, (384, N))
B = bf('diffusion_model.transformer_blocks.1.attn1.to_q.lora_B.weight', 0, COUNT * 384, (COUNT, 384))
BA = (B @ A).astype(np.float32)[:, 0]                                                                        # the same float32 product as uintervals.py, column 0
lo_u = np.zeros(COUNT); hi_u = np.ones(COUNT); used = []
for path in sorted(glob.glob(D + 'fp8r_q_s*_n10_*weight_raw_float8_e4m3fn.bin')):
    tag = re.search(r'fp8r_q_s([0-9]+)_n10', path).group(1)
    s = float('0.' + tag[1:]) if tag.startswith('0') else float(tag)
    scale = np.fromfile(path.replace('weight_raw_float8_e4m3fn.bin', 'weight_scale_raw_float32.bin'), np.float32)[0]
    with open(path, 'rb') as f:
        stored = np.array([(f.seek(r * N), f.read(1)[0])[1] for r in range(COUNT)], np.uint8)
    exact = (W + np.float32(s) * BA).astype(np.float32)
    y = np.abs(exact / scale).astype(np.float64)
    idx = np.clip(np.searchsorted(pos, y, side='right') - 1, 0, len(pos) - 2)
    l, h = pos[idx], pos[idx + 1]; frac = (y - l) / (h - l)
    sv = np.abs(val[stored]); up = sv == h; down = sv == l
    lo_u = np.where(down, np.maximum(lo_u, frac), lo_u); hi_u = np.where(up, np.minimum(hi_u, frac), hi_u)
    used.append(s)
w = hi_u - lo_u
print('strengths', sorted(used), '| column 0, rows 0..%d: conflicts' % (COUNT - 1), int((w <= 0).sum()), 'width median', np.median(w), '| bits in the first 48:', float(-np.log2(np.clip(w[:48], 1e-9, 1)).sum()))
# cross-check against the row-major file: element (0, 0) is the first element of both orders
rm = np.fromfile(os.path.dirname(os.path.abspath(__file__)) + '/real.bin', np.float64)
print('element (0,0): col file', (lo_u[0], hi_u[0]), ' row file', (rm[0], rm[len(rm) // 2]))
np.concatenate([lo_u, hi_u]).astype(np.float64).tofile(OUT)
print('wrote', OUT)
