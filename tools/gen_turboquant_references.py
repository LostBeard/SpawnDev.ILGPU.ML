#!/usr/bin/env python3
"""Generate TurboQuant reference data for ILGPU.ML unit tests.

Generates:
- FWHT reference values (known inputs → known outputs)
- Lloyd-Max codebook for Beta distribution
- Quantize/dequantize round-trip test vectors
- Bit-pack/unpack test vectors

Requires: numpy, scipy
"""

import os
import json
import numpy as np
from scipy.special import gamma as gamma_func
from scipy.linalg import hadamard

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, '..', 'SpawnDev.ILGPU.ML')
REFERENCES_DIR = os.path.join(PROJECT_ROOT, 'SpawnDev.ILGPU.ML.Demo', 'wwwroot', 'references', 'turboquant')
os.makedirs(REFERENCES_DIR, exist_ok=True)


def save_metadata(path, data):
    def convert(obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (list, tuple)): return [convert(x) for x in obj]
        return obj
    with open(path, 'w') as f:
        json.dump({k: convert(v) for k, v in data.items()}, f, indent=2)


# ============================================================
# 1. FWHT Reference Values
# ============================================================
print('=== FWHT Reference Values ===')

def fwht_reference(x):
    """Reference FWHT via Hadamard matrix multiply."""
    d = len(x)
    H = hadamard(d).astype(np.float64) / np.sqrt(d)
    return H @ x

# Test case 1: impulse
d8_impulse = np.zeros(8, dtype=np.float64)
d8_impulse[0] = 1.0
d8_impulse_out = fwht_reference(d8_impulse)
print(f'  d=8 impulse: {d8_impulse_out.tolist()}')

# Test case 2: all ones
d8_ones = np.ones(8, dtype=np.float64)
d8_ones_out = fwht_reference(d8_ones)
print(f'  d=8 ones: {d8_ones_out.tolist()}')

# Test case 3: seeded random d=128 (full head_dim)
rng = np.random.RandomState(42)
d128_random = rng.randn(128).astype(np.float64)
d128_random_out = fwht_reference(d128_random)
print(f'  d=128 random: first5={d128_random_out[:5].tolist()}')

# Test case 4: round-trip (FWHT is self-inverse up to scaling)
d128_roundtrip = fwht_reference(d128_random_out)
roundtrip_error = np.max(np.abs(d128_roundtrip - d128_random))
print(f'  d=128 round-trip max error: {roundtrip_error:.2e}')

# In-place butterfly reference
def fwht_inplace(a):
    """In-place butterfly FWHT (the GPU kernel algorithm)."""
    a = a.copy().astype(np.float64)
    d = len(a)
    h = 1
    while h < d:
        for i in range(0, d, h * 2):
            for j in range(i, i + h):
                x = a[j]
                y = a[j + h]
                a[j] = x + y
                a[j + h] = x - y
        h *= 2
    a /= np.sqrt(d)
    return a

# Verify butterfly matches Hadamard matrix multiply
d128_butterfly = fwht_inplace(d128_random)
butterfly_error = np.max(np.abs(d128_butterfly - d128_random_out))
print(f'  butterfly vs matrix max error: {butterfly_error:.2e}')

fwht_tests = {
    'd8_impulse': {
        'input': d8_impulse.tolist(),
        'expected': d8_impulse_out.tolist(),
    },
    'd8_ones': {
        'input': d8_ones.tolist(),
        'expected': d8_ones_out.tolist(),
    },
    'd128_random': {
        'input': d128_random.tolist(),
        'expected': d128_random_out.tolist(),
        'seed': 42,
    },
    'd128_roundtrip_max_error': roundtrip_error,
}


# ============================================================
# 2. Beta Distribution PDF and Lloyd-Max Codebook
# ============================================================
print('\n=== Lloyd-Max Codebook ===')

def beta_pdf(x, d):
    """PDF of a coordinate after random rotation on S^(d-1)."""
    if abs(x) >= 1.0:
        return 0.0
    coeff = gamma_func(d / 2.0) / (np.sqrt(np.pi) * gamma_func((d - 1) / 2.0))
    return coeff * (1.0 - x * x) ** ((d - 3) / 2.0)

def lloyd_max(d, bits, n_grid=50000, n_iter=100):
    """Compute Lloyd-Max codebook for the Beta distribution."""
    n_levels = 2 ** bits

    # Dense grid focused on the distribution support
    sigma = 1.0 / np.sqrt(d)
    x_max = min(1.0 - 1e-10, 6 * sigma)
    grid = np.linspace(-x_max, x_max, n_grid)
    pdf = np.array([beta_pdf(x, d) for x in grid])
    pdf /= pdf.sum()  # normalize

    # Initialize centroids uniformly
    centroids = np.linspace(-x_max, x_max, n_levels)

    for iteration in range(n_iter):
        # Assign each grid point to nearest centroid
        assignments = np.argmin(np.abs(grid[:, None] - centroids[None, :]), axis=1)

        # Update centroids as weighted mean within each partition
        new_centroids = np.zeros(n_levels)
        for k in range(n_levels):
            mask = assignments == k
            if mask.any():
                new_centroids[k] = np.sum(grid[mask] * pdf[mask]) / np.sum(pdf[mask])
            else:
                new_centroids[k] = centroids[k]

        if np.max(np.abs(new_centroids - centroids)) < 1e-10:
            break
        centroids = new_centroids

    return np.sort(centroids)

# Generate codebooks for common configurations
codebooks = {}
for bits in [2, 3, 4]:
    for d in [64, 128]:
        key = f'{bits}bit_d{d}'
        cb = lloyd_max(d, bits)
        codebooks[key] = cb.tolist()
        print(f'  {key}: {cb.tolist()[:4]}...{cb.tolist()[-2:]}')

# Compute theoretical MSE for 4-bit d=128
d = 128
bits = 4
cb = np.array(codebooks['4bit_d128'])
sigma = 1.0 / np.sqrt(d)
grid = np.linspace(-6*sigma, 6*sigma, 50000)
pdf = np.array([beta_pdf(x, d) for x in grid])
pdf /= pdf.sum()
assignments = np.argmin(np.abs(grid[:, None] - cb[None, :]), axis=1)
mse = np.sum(pdf * (grid - cb[assignments]) ** 2)
print(f'  4-bit d=128 MSE per coordinate: {mse:.6f}')


# ============================================================
# 3. Sign Flip Vectors
# ============================================================
print('\n=== Sign Flip Vectors ===')

rng_sign = np.random.RandomState(123)
sign_d128 = rng_sign.choice([-1.0, 1.0], size=128)
print(f'  d=128 seed=123: first10={sign_d128[:10].tolist()}')


# ============================================================
# 4. Full Quantize/Dequantize Round-Trip
# ============================================================
print('\n=== Quantize/Dequantize Round-Trip ===')

def quantize_vector(x, sign_flip, codebook):
    """Full TurboQuant quantize pipeline."""
    norm = np.linalg.norm(x)
    if norm < 1e-10:
        return norm, np.zeros(len(x), dtype=int)
    x_unit = x / norm
    y = x_unit * sign_flip  # sign flip
    y = fwht_inplace(y)  # FWHT
    # Nearest centroid
    indices = np.argmin(np.abs(y[:, None] - codebook[None, :]), axis=1)
    return norm, indices

def dequantize_vector(norm, indices, sign_flip, codebook):
    """Full TurboQuant dequantize pipeline."""
    y_hat = codebook[indices]
    x_hat = fwht_inplace(y_hat)  # inverse FWHT
    x_hat = x_hat * sign_flip  # reverse sign flip
    x_hat = x_hat * norm
    return x_hat

# Test vector: seeded random d=128
rng_vec = np.random.RandomState(99)
test_vec = rng_vec.randn(128).astype(np.float64)
cb_4bit = np.array(codebooks['4bit_d128'])

norm, indices = quantize_vector(test_vec, sign_d128, cb_4bit)
reconstructed = dequantize_vector(norm, indices, sign_d128, cb_4bit)

mse_total = np.mean((test_vec - reconstructed) ** 2)
relative_error = np.linalg.norm(test_vec - reconstructed) / np.linalg.norm(test_vec)
dot_product = np.dot(test_vec, reconstructed) / (np.linalg.norm(test_vec) * np.linalg.norm(reconstructed))

print(f'  4-bit d=128: MSE={mse_total:.6f}, relative_err={relative_error:.6f}, cosine={dot_product:.6f}')
print(f'  norm={norm:.4f}, first5_indices={indices[:5].tolist()}')

roundtrip_test = {
    'input': test_vec.tolist(),
    'sign_flip_seed': 123,
    'codebook_key': '4bit_d128',
    'norm': float(norm),
    'indices': indices.tolist(),
    'reconstructed': reconstructed.tolist(),
    'mse': float(mse_total),
    'relative_error': float(relative_error),
    'cosine_similarity': float(dot_product),
}


# ============================================================
# 5. Bit-Pack/Unpack Test Vectors
# ============================================================
print('\n=== Bit Pack/Unpack ===')

def pack_4bit(indices):
    """Pack array of 4-bit values into bytes (2 per byte)."""
    packed = []
    for i in range(0, len(indices), 2):
        lo = indices[i] & 0xF
        hi = (indices[i + 1] & 0xF) if i + 1 < len(indices) else 0
        packed.append(lo | (hi << 4))
    return packed

def unpack_4bit(packed, count):
    """Unpack bytes into 4-bit values."""
    values = []
    for byte in packed:
        values.append(byte & 0xF)
        values.append((byte >> 4) & 0xF)
    return values[:count]

test_indices_16 = [0, 3, 7, 15, 1, 2, 4, 8, 9, 10, 11, 12, 13, 14, 5, 6]
packed_16 = pack_4bit(test_indices_16)
unpacked_16 = unpack_4bit(packed_16, 16)
assert unpacked_16 == test_indices_16, "4-bit pack/unpack failed!"
print(f'  4-bit: {test_indices_16} -> packed {packed_16} -> unpacked {unpacked_16}')

def pack_3bit(indices):
    """Pack array of 3-bit values into bytes (8 values per 3 bytes)."""
    bits = 0
    bit_count = 0
    packed = []
    for idx in indices:
        bits |= (idx & 0x7) << bit_count
        bit_count += 3
        while bit_count >= 8:
            packed.append(bits & 0xFF)
            bits >>= 8
            bit_count -= 8
    if bit_count > 0:
        packed.append(bits & 0xFF)
    return packed

test_indices_8_3bit = [0, 1, 2, 3, 4, 5, 6, 7]
packed_3bit = pack_3bit(test_indices_8_3bit)
print(f'  3-bit: {test_indices_8_3bit} -> packed {packed_3bit} ({len(packed_3bit)} bytes)')

bitpack_tests = {
    '4bit_16values': {
        'indices': test_indices_16,
        'packed': packed_16,
    },
    '3bit_8values': {
        'indices': test_indices_8_3bit,
        'packed': packed_3bit,
    },
}


# ============================================================
# 6. Attention with Quantized KV (Orthogonality Property)
# ============================================================
print('\n=== Attention Orthogonality Property ===')

# The key property: <q, Pi^T y> == <Pi q, y>
# So we pre-rotate the query instead of dequantizing keys
q = rng_vec.randn(128).astype(np.float64)

# Method 1: dequantize key, then dot
key_original = test_vec
key_reconstructed = reconstructed
dot_method1 = np.dot(q, key_reconstructed)

# Method 2: rotate query, dot with centroid values
q_flipped = q * sign_d128
q_rotated = fwht_inplace(q_flipped)
y_hat = cb_4bit[indices]
dot_method2 = np.dot(q_rotated, y_hat) * norm

print(f'  Method 1 (dequant key): {dot_method1:.6f}')
print(f'  Method 2 (rotate query): {dot_method2:.6f}')
print(f'  Difference: {abs(dot_method1 - dot_method2):.2e}')

attention_test = {
    'query': q.tolist(),
    'dot_via_dequant': float(dot_method1),
    'dot_via_query_rotation': float(dot_method2),
    'max_difference': float(abs(dot_method1 - dot_method2)),
}


# ============================================================
# SAVE
# ============================================================
all_references = {
    'fwht_tests': fwht_tests,
    'codebooks': codebooks,
    'sign_flip': {
        'seed': 123,
        'd128': sign_d128.tolist(),
    },
    'roundtrip_test': roundtrip_test,
    'bitpack_tests': bitpack_tests,
    'attention_test': attention_test,
    'mse_4bit_d128_per_coord': float(mse),
}

save_metadata(os.path.join(REFERENCES_DIR, 'turboquant_test_cases.json'), all_references)

# Also save binary files for large vectors
test_vec.astype(np.float32).tofile(os.path.join(REFERENCES_DIR, 'test_vector_d128.bin'))
d128_random.astype(np.float32).tofile(os.path.join(REFERENCES_DIR, 'fwht_input_d128.bin'))
d128_random_out.astype(np.float32).tofile(os.path.join(REFERENCES_DIR, 'fwht_output_d128.bin'))

print(f'\nSaved to {REFERENCES_DIR}')
print('Done.')
