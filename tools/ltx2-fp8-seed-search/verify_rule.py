# The black-box rule the seed search found, checked on EVERY element of a tensor at every dumped LoRA strength:
#   seed = crc32(key), key = "diffusion_model.<tensor>.weight" (UTF-8);  generator = MT19937 init_genrand(seed) (torch's CPU default);
#   u_i = (word_i & 0xFF) / 256 for element i in row-major order (one 32-bit word per element: torch.rand in bfloat16);
#   w = W + s B A (float32), scale = max|w| / 448, y = float16(|w| / scale) (the scaled weight passes through HALF precision), between the e4m3
#   neighbours l <= y < h, frac = (y - l) / (h - l);  stored = sign(w) (h if frac + u >= 1 else l).
# MEASURED 2026-11-17 on attn1.to_q (16,777,216 elements x 10 strengths): 27 to 70 elements differ (the last float32 bit of B A, by summation order);
# without the float16 step 0.19% differ (every even 8-bit draw one step early), with ">" instead of ">=" 0.39%.
# usage: python verify_rule.py [q|k]      (attn1.to_q: ten strengths; attn1.to_k: strength 1)
import numpy as np, json, struct, glob, re, sys, zlib
CK = 'D:/spawndev-ltx2/lightricks/ltx-2-19b-dev-fp8.safetensors'
LO = 'D:/spawndev-ltx2/lightricks/ltx-2-19b-distilled-lora-384.safetensors'
D = 'F:/spawndev-ltx2/comfyui-dumps/lora/fp8_rounding/'
which = sys.argv[1] if len(sys.argv) > 1 else 'q'
name = 'transformer_blocks.1.attn1.to_' + which
N = 4096
def opener(p):
    f = open(p, 'rb'); n = struct.unpack('<Q', f.read(8))[0]; h = json.loads(f.read(n)); return f, h, 8 + n
def raw(fhb, k):
    f, h, base = fhb; a, b = h[k]['data_offsets']; f.seek(base + a); return f.read(b - a), h[k]
codes = np.arange(256); e = (codes >> 3) & 15; m = codes & 7
val = np.where(e == 0, m / 8 * 2.0 ** -6, (1 + m / 8) * 2.0 ** (e - 7)); val = np.where(codes & 0x80, -val, val); val[(codes & 0x7F) == 0x7F] = np.nan
pos = np.unique(val[codes < 0x7F])
ck, lo = opener(CK), opener(LO)
wb, wi = raw(ck, 'model.diffusion_model.' + name + '.weight'); assert wi['dtype'] == 'F8_E4M3' and wi['shape'] == [N, N], wi
W = val[np.frombuffer(wb, np.uint8)].reshape(N, N).astype(np.float32)
def bf(k):
    r, i = raw(lo, k); return (np.frombuffer(r, np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(i['shape'])
BA = (bf('diffusion_model.' + name + '.lora_B.weight') @ bf('diffusion_model.' + name + '.lora_A.weight')).astype(np.float32)
key = 'diffusion_model.' + name + '.weight'
seed = zlib.crc32(key.encode('utf-8'))
bg = np.random.MT19937(); bg._legacy_seeding(seed)                       # init_genrand(seed), as torch.manual_seed
words = bg.random_raw(N * N).astype(np.uint64)
u = ((words & 0xFF).astype(np.float64) / 256.0)
print('key', key, 'seed', seed, 'first uniforms', u[:4])
files = sorted(glob.glob(D + 'fp8r_%s_s*_n10_*weight_raw_float8_e4m3fn.bin' % which))
for path in files:
    tag = re.search(r'fp8r_%s_s([0-9]+)_n10' % which, path).group(1)
    s = float('0.' + tag[1:]) if tag.startswith('0') else float(tag)
    scale = np.fromfile(path.replace('weight_raw_float8_e4m3fn.bin', 'weight_scale_raw_float32.bin'), np.float32)[0]
    stored = np.fromfile(path, np.uint8)
    exact = (W + np.float32(s) * BA).astype(np.float32).ravel()
    sv = np.abs(val[stored])
    line = 'strength %-6g scale %.6g:' % (s, scale)
    y32 = np.abs(exact) / scale
    for nm, yv, ge in (('float16(y), frac + u >= 1', y32.astype(np.float16).astype(np.float64), True), ('float32 y, frac + u >= 1', y32.astype(np.float64), True), ('float16(y), frac + u > 1', y32.astype(np.float16).astype(np.float64), False)):
        idx = np.clip(np.searchsorted(pos, yv, side='right') - 1, 0, len(pos) - 2)
        l, h = pos[idx], pos[idx + 1]; frac = (yv - l) / (h - l)
        up = (frac + u >= 1) if ge else (frac + u > 1)
        bad = np.where(up, h, l) != sv
        line += '  [%s] %d wrong of %d (%.5f%%)' % (nm, int(bad.sum()), bad.size, 100.0 * bad.mean())
    # sign and scale sanity
    sgn_bad = int((np.signbit(val[stored]) != np.signbit(exact))[sv != 0].sum())
    print(line, ' sign mismatches (non-zero)', sgn_bad, ' scale == max|exact|/448:', bool(np.float32(np.abs(exact).max() / np.float32(448)) == scale), flush=True)
