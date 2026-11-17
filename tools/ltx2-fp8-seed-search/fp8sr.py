import numpy as np, json, struct
CK = 'D:/spawndev-ltx2/lightricks/ltx-2-19b-dev-fp8.safetensors'
LO = 'D:/spawndev-ltx2/lightricks/ltx-2-19b-distilled-lora-384.safetensors'
TAP = 'F:/spawndev-ltx2/comfyui-dumps/lora/fp8_block1_repeat/ltx2_lora8_lin_diffusion_model_transformer_blocks_1_'

def opener(p):
    f = open(p, 'rb'); n = struct.unpack('<Q', f.read(8))[0]; h = json.loads(f.read(n)); return f, h, 8 + n
def raw(fhb, k):
    f, h, base = fhb; a, b = h[k]['data_offsets']; f.seek(base + a); return f.read(b - a), h[k]
def bf16(fhb, k):
    r, t = raw(fhb, k); return (np.frombuffer(r, np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(t['shape'])

# e4m3fn decode table (codes 0..255)
codes = np.arange(256)
e = (codes >> 3) & 15; m = codes & 7
val = np.where(e == 0, m / 8 * 2.0 ** -6, (1 + m / 8) * 2.0 ** (e - 7))
val = np.where(codes & 0x80, -val, val)
val[(codes & 0x7F) == 0x7F] = np.nan                       # NaN codes
pos = np.unique(val[(codes < 0x7F)])                       # non-negative finite grid, ascending (0 .. 448)

ck = opener(CK); lo = opener(LO)
for name, n in [('attn1.to_q', 4096), ('audio_attn1.to_q', 2048)]:
    key = 'model.diffusion_model.transformer_blocks.1.' + name
    r, _ = raw(ck, key + '.weight'); W = val[np.frombuffer(r, np.uint8)].reshape(n, n).astype(np.float32)
    ws = np.frombuffer(raw(ck, key + '.weight_scale')[0], np.float32)[0] if (key + '.weight_scale') in ck[1] else 1.0
    A = bf16(lo, 'diffusion_model.transformer_blocks.1.' + name + '.lora_A.weight'); B = bf16(lo, 'diffusion_model.transformer_blocks.1.' + name + '.lora_B.weight')
    exact = (W * np.float32(ws) + (B @ A).astype(np.float32)).astype(np.float32)
    tn = TAP + name.replace('.', '_')
    stored = np.fromfile(tn + '_weight_raw_float8_e4m3fn.bin', np.uint8).reshape(n, n)
    s2 = np.fromfile(tn + '_weight_scale_raw_float32.bin', np.float32)[0]
    print(f'{name}: ws {ws}, new scale {s2:.10g}, max|exact|/448 {np.abs(exact).max()/448:.10g}')
    y = (exact / np.float32(s2)).astype(np.float64)
    ay = np.abs(y)
    idx = np.clip(np.searchsorted(pos, ay, side='right') - 1, 0, len(pos) - 2)
    lo_v, hi_v = pos[idx], pos[idx + 1]
    sv = np.abs(val[stored])
    exactgrid = ay == lo_v
    up = sv == hi_v; down = sv == lo_v
    frac = (ay - lo_v) / (hi_v - lo_v)
    other = ~(up | down)
    print(f'  on-grid {exactgrid.mean():.4f}, up {up.mean():.4f}, down {down.mean():.4f}, neither {other.mean():.6f}, sign mismatches {(np.sign(val[stored]) * np.sign(y) < 0).mean():.6f}')
    ok = ~exactgrid & ~other
    bins = np.linspace(0, 1, 11)
    which = np.digitize(frac[ok], bins) - 1
    print('  P(up | frac bin):', ' '.join(f'{np.mean(up[ok][which == b]):.3f}' for b in range(10)))
    # deterministic structure? fraction of 'up' vs frac threshold with a single uniform u per element: up iff u < frac
    import os; np.save(os.path.dirname(os.path.abspath(__file__)) + f'/sr_{name}.npy', np.stack([frac, up.astype(np.float64), ok.astype(np.float64)]))
