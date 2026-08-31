#!/usr/bin/env python3
"""Generate expanded operator-level test data for untested ONNX operators.

Adds test cases for ~35 operators that have implementations but no test coverage.
Output is merged into operator_test_cases.json alongside existing tests.
"""

import numpy as np
import json
import os
import math

out_dir = r'D:\users\tj\Projects\SpawnDev.ILGPU.ML\SpawnDev.ILGPU.ML\SpawnDev.ILGPU.ML.Demo\wwwroot\test-models'

# Load existing tests to merge with
existing_path = os.path.join(out_dir, 'operator_test_cases.json')
with open(existing_path, 'r') as f:
    tests = json.load(f)

print(f'Loaded {len(tests)} existing test cases')
print('=' * 60)

# ============================================
# MATMUL — fundamental compute kernel
# ============================================
print('=== MatMul ===')

# 2D x 2D
a_mm = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)  # [2,3]
b_mm = np.array([[7, 8], [9, 10], [11, 12]], dtype=np.float32)  # [3,2]
result_mm = a_mm @ b_mm
tests['matmul_2d'] = {
    'op': 'MatMul', 'description': '[2,3] @ [3,2] -> [2,2]',
    'input_a': {'shape': [2, 3], 'data': a_mm.flatten().tolist()},
    'input_b': {'shape': [3, 2], 'data': b_mm.flatten().tolist()},
    'expected': {'shape': list(result_mm.shape), 'data': result_mm.flatten().tolist()}
}
print(f'  2D: {result_mm.flatten().tolist()}')

# Batched 3D x 3D
a_mm3 = np.random.seed(42)
a_mm3 = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
b_mm3 = np.arange(16, dtype=np.float32).reshape(2, 4, 2)
result_mm3 = a_mm3 @ b_mm3
tests['matmul_batched_3d'] = {
    'op': 'MatMul', 'description': '[2,3,4] @ [2,4,2] -> [2,3,2] batched',
    'input_a': {'shape': [2, 3, 4], 'data': a_mm3.flatten().tolist()},
    'input_b': {'shape': [2, 4, 2], 'data': b_mm3.flatten().tolist()},
    'expected': {'shape': list(result_mm3.shape), 'data': result_mm3.flatten().tolist()}
}
print(f'  Batched 3D: shape={list(result_mm3.shape)}, first 4={result_mm3.flatten()[:4].tolist()}')

# ============================================
# GEMM — General Matrix Multiply (alpha*A@B + beta*C)
# ============================================
print('=== Gemm ===')

a_g = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)  # [3,2]
b_g = np.array([[7, 8, 9], [10, 11, 12]], dtype=np.float32)  # [2,3]
c_g = np.array([1, 2, 3], dtype=np.float32)  # [3] bias
alpha, beta = 1.0, 1.0
result_g = alpha * (a_g @ b_g) + beta * c_g
tests['gemm_basic'] = {
    'op': 'Gemm', 'description': '[3,2]@[2,3]+[3] alpha=1 beta=1',
    'input_a': {'shape': [3, 2], 'data': a_g.flatten().tolist()},
    'input_b': {'shape': [2, 3], 'data': b_g.flatten().tolist()},
    'input_c': {'shape': [3], 'data': c_g.tolist()},
    'attributes': {'alpha': alpha, 'beta': beta, 'transA': 0, 'transB': 0},
    'expected': {'shape': list(result_g.shape), 'data': result_g.flatten().tolist()}
}
print(f'  basic: {result_g.flatten().tolist()}')

# Gemm with transB
b_gt = b_g.T  # [3,2] -> transposed to act as [2,3]
result_gt = alpha * (a_g @ b_gt.T) + beta * c_g
tests['gemm_transB'] = {
    'op': 'Gemm', 'description': '[3,2]@[3,2]^T+[3] transB=1',
    'input_a': {'shape': [3, 2], 'data': a_g.flatten().tolist()},
    'input_b': {'shape': [3, 2], 'data': b_gt.flatten().tolist()},
    'input_c': {'shape': [3], 'data': c_g.tolist()},
    'attributes': {'alpha': alpha, 'beta': beta, 'transA': 0, 'transB': 1},
    'expected': {'shape': list(result_gt.shape), 'data': result_gt.flatten().tolist()}
}
print(f'  transB: {result_gt.flatten().tolist()}')

# ============================================
# LAYERNORMALIZATION
# ============================================
print('=== LayerNormalization ===')

x_ln = np.array([[1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0]], dtype=np.float32)  # [2,4]
scale_ln = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
bias_ln = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
eps = 1e-5
# Normalize over last axis
mean = x_ln.mean(axis=-1, keepdims=True)
var = x_ln.var(axis=-1, keepdims=True)
result_ln = (x_ln - mean) / np.sqrt(var + eps) * scale_ln + bias_ln
tests['layernorm_basic'] = {
    'op': 'LayerNormalization', 'description': '[2,4] axis=-1 eps=1e-5',
    'input': {'shape': [2, 4], 'data': x_ln.flatten().tolist()},
    'scale': {'shape': [4], 'data': scale_ln.tolist()},
    'bias': {'shape': [4], 'data': bias_ln.tolist()},
    'attributes': {'axis': -1, 'epsilon': eps},
    'expected': {'shape': [2, 4], 'data': result_ln.flatten().tolist()}
}
print(f'  basic: first 4={result_ln.flatten()[:4].tolist()}')

# LayerNorm with scale and bias
scale_ln2 = np.array([2.0, 0.5, 1.0, 3.0], dtype=np.float32)
bias_ln2 = np.array([0.1, -0.1, 0.0, 0.5], dtype=np.float32)
result_ln2 = (x_ln - mean) / np.sqrt(var + eps) * scale_ln2 + bias_ln2
tests['layernorm_scale_bias'] = {
    'op': 'LayerNormalization', 'description': '[2,4] with scale=[2,0.5,1,3] bias=[0.1,-0.1,0,0.5]',
    'input': {'shape': [2, 4], 'data': x_ln.flatten().tolist()},
    'scale': {'shape': [4], 'data': scale_ln2.tolist()},
    'bias': {'shape': [4], 'data': bias_ln2.tolist()},
    'attributes': {'axis': -1, 'epsilon': eps},
    'expected': {'shape': [2, 4], 'data': result_ln2.flatten().tolist()}
}
print(f'  scale+bias: first 4={result_ln2.flatten()[:4].tolist()}')

# ============================================
# BATCHNORMALIZATION
# ============================================
print('=== BatchNormalization ===')

x_bn = np.arange(1, 13, dtype=np.float32).reshape(1, 3, 2, 2)  # [1,3,2,2]
bn_scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)
bn_bias = np.array([0.0, 0.0, 0.0], dtype=np.float32)
bn_mean = np.array([2.5, 6.5, 10.5], dtype=np.float32)
bn_var = np.array([1.25, 1.25, 1.25], dtype=np.float32)
bn_eps = 1e-5
# BN: (x - mean) / sqrt(var + eps) * scale + bias, per channel
result_bn = np.zeros_like(x_bn)
for c in range(3):
    result_bn[0, c] = (x_bn[0, c] - bn_mean[c]) / np.sqrt(bn_var[c] + bn_eps) * bn_scale[c] + bn_bias[c]
tests['batchnorm_basic'] = {
    'op': 'BatchNormalization', 'description': '[1,3,2,2] per-channel normalization',
    'input': {'shape': [1, 3, 2, 2], 'data': x_bn.flatten().tolist()},
    'scale': {'shape': [3], 'data': bn_scale.tolist()},
    'bias': {'shape': [3], 'data': bn_bias.tolist()},
    'mean': {'shape': [3], 'data': bn_mean.tolist()},
    'var': {'shape': [3], 'data': bn_var.tolist()},
    'attributes': {'epsilon': bn_eps},
    'expected': {'shape': [1, 3, 2, 2], 'data': result_bn.flatten().tolist()}
}
print(f'  basic: first 4={result_bn.flatten()[:4].tolist()}')

# ============================================
# INSTANCENORMALIZATION
# ============================================
print('=== InstanceNormalization ===')

x_in = np.arange(1, 13, dtype=np.float32).reshape(1, 3, 2, 2)
in_scale = np.array([1.0, 2.0, 0.5], dtype=np.float32)
in_bias = np.array([0.0, 1.0, -1.0], dtype=np.float32)
in_eps = 1e-5
result_in = np.zeros_like(x_in)
for c in range(3):
    m = x_in[0, c].mean()
    v = x_in[0, c].var()
    result_in[0, c] = (x_in[0, c] - m) / np.sqrt(v + in_eps) * in_scale[c] + in_bias[c]
tests['instancenorm_basic'] = {
    'op': 'InstanceNormalization', 'description': '[1,3,2,2] per-instance normalization',
    'input': {'shape': [1, 3, 2, 2], 'data': x_in.flatten().tolist()},
    'scale': {'shape': [3], 'data': in_scale.tolist()},
    'bias': {'shape': [3], 'data': in_bias.tolist()},
    'attributes': {'epsilon': in_eps},
    'expected': {'shape': [1, 3, 2, 2], 'data': result_in.flatten().tolist()}
}
print(f'  basic: first 4={result_in.flatten()[:4].tolist()}')

# ============================================
# MAXPOOL
# ============================================
print('=== MaxPool ===')

x_mp = np.arange(1, 17, dtype=np.float32).reshape(1, 1, 4, 4)
out_mp = np.zeros((1, 1, 2, 2), dtype=np.float32)
for i in range(2):
    for j in range(2):
        out_mp[0, 0, i, j] = x_mp[0, 0, i*2:i*2+2, j*2:j*2+2].max()
tests['maxpool_2x2_s2'] = {
    'op': 'MaxPool', 'description': '[1,1,4,4] k=2x2 s=2 -> [1,1,2,2]',
    'input': {'shape': [1, 1, 4, 4], 'data': x_mp.flatten().tolist()},
    'attributes': {'kernel_shape': [2, 2], 'strides': [2, 2], 'pads': [0, 0, 0, 0]},
    'expected': {'shape': [1, 1, 2, 2], 'data': out_mp.flatten().tolist()}
}
print(f'  2x2 s2: {out_mp.flatten().tolist()}')

# 3x3 stride=1
x_mp2 = np.arange(1, 26, dtype=np.float32).reshape(1, 1, 5, 5)
out_mp2 = np.zeros((1, 1, 3, 3), dtype=np.float32)
for i in range(3):
    for j in range(3):
        out_mp2[0, 0, i, j] = x_mp2[0, 0, i:i+3, j:j+3].max()
tests['maxpool_3x3_s1'] = {
    'op': 'MaxPool', 'description': '[1,1,5,5] k=3x3 s=1 -> [1,1,3,3]',
    'input': {'shape': [1, 1, 5, 5], 'data': x_mp2.flatten().tolist()},
    'attributes': {'kernel_shape': [3, 3], 'strides': [1, 1], 'pads': [0, 0, 0, 0]},
    'expected': {'shape': [1, 1, 3, 3], 'data': out_mp2.flatten().tolist()}
}
print(f'  3x3 s1: {out_mp2.flatten().tolist()}')

# ============================================
# GLOBALAVERAGEPOOL
# ============================================
print('=== GlobalAveragePool ===')

x_gap = np.arange(1, 13, dtype=np.float32).reshape(1, 3, 2, 2)
result_gap = x_gap.mean(axis=(2, 3), keepdims=True)
tests['globalavgpool_basic'] = {
    'op': 'GlobalAveragePool', 'description': '[1,3,2,2] -> [1,3,1,1]',
    'input': {'shape': [1, 3, 2, 2], 'data': x_gap.flatten().tolist()},
    'expected': {'shape': list(result_gap.shape), 'data': result_gap.flatten().tolist()}
}
print(f'  basic: {result_gap.flatten().tolist()}')

# ============================================
# CONVTRANSPOSE (deconvolution)
# ============================================
print('=== ConvTranspose ===')

# Simple 1-channel 2x2 input, 3x3 kernel, stride=2 -> 5x5 output (no padding)
x_ct = np.array([[[[1, 2], [3, 4]]]], dtype=np.float32)  # [1,1,2,2]
w_ct = np.ones((1, 1, 3, 3), dtype=np.float32)  # [inC,outC,kH,kW]
# Manual conv transpose: each input pixel generates a 3x3 patch at stride-2 locations
out_ct = np.zeros((1, 1, 5, 5), dtype=np.float32)
for iy in range(2):
    for ix in range(2):
        for ky in range(3):
            for kx in range(3):
                out_ct[0, 0, iy*2+ky, ix*2+kx] += x_ct[0, 0, iy, ix] * w_ct[0, 0, ky, kx]
tests['convtranspose_basic'] = {
    'op': 'ConvTranspose', 'description': '[1,1,2,2] k=3x3 s=2 -> [1,1,5,5]',
    'input': {'shape': [1, 1, 2, 2], 'data': x_ct.flatten().tolist()},
    'weight': {'shape': [1, 1, 3, 3], 'data': w_ct.flatten().tolist()},
    'bias': {'shape': [1], 'data': [0.0]},
    'attributes': {'kernel_shape': [3, 3], 'strides': [2, 2], 'pads': [0, 0, 0, 0]},
    'expected': {'shape': [1, 1, 5, 5], 'data': out_ct.flatten().tolist()}
}
print(f'  basic: shape=[1,1,5,5], first 5={out_ct.flatten()[:5].tolist()}')

# ============================================
# REDUCEMEAN
# ============================================
print('=== ReduceMean ===')

x_rmean = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
result_rmean = x_rmean.mean(axis=1, keepdims=True)
tests['reducemean_axis1'] = {
    'op': 'ReduceMean', 'description': '[3,3] reducemean axis=1 keepdims -> [3,1]',
    'input': {'shape': [3, 3], 'data': x_rmean.flatten().tolist()},
    'attributes': {'axes': [1], 'keepdims': 1},
    'expected': {'shape': list(result_rmean.shape), 'data': result_rmean.flatten().tolist()}
}
print(f'  axis=1: {result_rmean.flatten().tolist()}')

# ReduceMean over multiple axes (spatial dims - common in transformers)
x_rmean2 = np.arange(24, dtype=np.float32).reshape(1, 2, 3, 4)
result_rmean2 = x_rmean2.mean(axis=(2, 3), keepdims=True)
tests['reducemean_spatial'] = {
    'op': 'ReduceMean', 'description': '[1,2,3,4] reducemean axes=[2,3] -> [1,2,1,1]',
    'input': {'shape': [1, 2, 3, 4], 'data': x_rmean2.flatten().tolist()},
    'attributes': {'axes': [2, 3], 'keepdims': 1},
    'expected': {'shape': list(result_rmean2.shape), 'data': result_rmean2.flatten().tolist()}
}
print(f'  spatial: {result_rmean2.flatten().tolist()}')

# ============================================
# REDUCESUM
# ============================================
print('=== ReduceSum ===')

x_rsum = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
result_rsum = x_rsum.sum(axis=1, keepdims=True)
tests['reducesum_axis1'] = {
    'op': 'ReduceSum', 'description': '[2,3] reducesum axis=1 keepdims -> [2,1]',
    'input': {'shape': [2, 3], 'data': x_rsum.flatten().tolist()},
    'attributes': {'axes': [1], 'keepdims': 1},
    'expected': {'shape': list(result_rsum.shape), 'data': result_rsum.flatten().tolist()}
}
print(f'  axis=1: {result_rsum.flatten().tolist()}')

# ============================================
# GATHER
# ============================================
print('=== Gather ===')

# Embedding lookup pattern (common in NLP)
x_gather = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)  # [4,2] vocab
indices = np.array([0, 2, 3], dtype=np.int64)
result_gather = x_gather[indices]
tests['gather_axis0'] = {
    'op': 'Gather', 'description': '[4,2] gather indices=[0,2,3] axis=0 -> [3,2]',
    'input': {'shape': [4, 2], 'data': x_gather.flatten().tolist()},
    'indices': {'shape': [3], 'data': indices.tolist()},
    'attributes': {'axis': 0},
    'expected': {'shape': list(result_gather.shape), 'data': result_gather.flatten().tolist()}
}
print(f'  axis=0: {result_gather.flatten().tolist()}')

# Gather axis=1 (column select)
x_gather2 = np.arange(12, dtype=np.float32).reshape(3, 4)
indices2 = np.array([1, 3], dtype=np.int64)
result_gather2 = np.take(x_gather2, indices2, axis=1)
tests['gather_axis1'] = {
    'op': 'Gather', 'description': '[3,4] gather indices=[1,3] axis=1 -> [3,2]',
    'input': {'shape': [3, 4], 'data': x_gather2.flatten().tolist()},
    'indices': {'shape': [2], 'data': indices2.tolist()},
    'attributes': {'axis': 1},
    'expected': {'shape': list(result_gather2.shape), 'data': result_gather2.flatten().tolist()}
}
print(f'  axis=1: {result_gather2.flatten().tolist()}')

# ============================================
# SPLIT
# ============================================
print('=== Split ===')

x_split = np.arange(12, dtype=np.float32).reshape(1, 6, 2)
splits = [2, 2, 2]
results_split = np.split(x_split, [2, 4], axis=1)
for i, s in enumerate(results_split):
    tests[f'split_axis1_part{i}'] = {
        'op': 'Split', 'description': f'[1,6,2] split axis=1 into [2,2,2] part {i}',
        'input': {'shape': [1, 6, 2], 'data': x_split.flatten().tolist()},
        'attributes': {'axis': 1, 'split': splits, 'output_index': i},
        'expected': {'shape': list(s.shape), 'data': s.flatten().tolist()}
    }
print(f'  3-way: shapes={[list(s.shape) for s in results_split]}')

# ============================================
# EXPAND (broadcast to shape)
# ============================================
print('=== Expand ===')

x_exp = np.array([[[1], [2], [3]]], dtype=np.float32)  # [1,3,1]
shape_exp = [2, 3, 4]
result_exp = np.broadcast_to(x_exp, shape_exp)
tests['expand_broadcast'] = {
    'op': 'Expand', 'description': '[1,3,1] expand to [2,3,4]',
    'input': {'shape': [1, 3, 1], 'data': x_exp.flatten().tolist()},
    'attributes': {'shape': shape_exp},
    'expected': {'shape': shape_exp, 'data': result_exp.flatten().tolist()}
}
print(f'  [1,3,1]->[2,3,4]: first 8={result_exp.flatten()[:8].tolist()}')

# Attention mask expand pattern: [1,1,1,6] -> [1,1,6,6]
x_exp2 = np.array([[[[1, 1, 1, 0, 0, 0]]]], dtype=np.float32)  # [1,1,1,6]
shape_exp2 = [1, 1, 6, 6]
result_exp2 = np.broadcast_to(x_exp2, shape_exp2)
tests['expand_attention_mask'] = {
    'op': 'Expand', 'description': '[1,1,1,6] expand to [1,1,6,6] attention mask',
    'input': {'shape': [1, 1, 1, 6], 'data': x_exp2.flatten().tolist()},
    'attributes': {'shape': shape_exp2},
    'expected': {'shape': shape_exp2, 'data': result_exp2.flatten().tolist()}
}
print(f'  attention mask: [1,1,1,6]->[1,1,6,6]')

# ============================================
# UNSQUEEZE
# ============================================
print('=== Unsqueeze ===')

x_us = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)  # [2,3]
result_us0 = np.expand_dims(x_us, axis=0)  # [1,2,3]
tests['unsqueeze_axis0'] = {
    'op': 'Unsqueeze', 'description': '[2,3] unsqueeze axis=0 -> [1,2,3]',
    'input': {'shape': [2, 3], 'data': x_us.flatten().tolist()},
    'attributes': {'axes': [0]},
    'expected': {'shape': list(result_us0.shape), 'data': x_us.flatten().tolist()}
}
result_us2 = np.expand_dims(x_us, axis=2)  # [2,3,1]
tests['unsqueeze_axis2'] = {
    'op': 'Unsqueeze', 'description': '[2,3] unsqueeze axis=2 -> [2,3,1]',
    'input': {'shape': [2, 3], 'data': x_us.flatten().tolist()},
    'attributes': {'axes': [2]},
    'expected': {'shape': list(result_us2.shape), 'data': x_us.flatten().tolist()}
}
# Multiple axes (common in DistilBERT: [1,6] -> [1,1,1,6])
result_us_multi = x_us[np.newaxis, np.newaxis, :, :]  # [1,1,2,3]
tests['unsqueeze_multi_axes'] = {
    'op': 'Unsqueeze', 'description': '[2,3] unsqueeze axes=[0,1] -> [1,1,2,3]',
    'input': {'shape': [2, 3], 'data': x_us.flatten().tolist()},
    'attributes': {'axes': [0, 1]},
    'expected': {'shape': list(result_us_multi.shape), 'data': x_us.flatten().tolist()}
}
print(f'  axis=0: [2,3]->[1,2,3], axis=2: [2,3]->[2,3,1], multi: [2,3]->[1,1,2,3]')

# ============================================
# SQUEEZE
# ============================================
print('=== Squeeze ===')

x_sq = np.array([[[1, 2, 3]]], dtype=np.float32)  # [1,1,3]
result_sq = np.squeeze(x_sq, axis=0)  # [1,3]
tests['squeeze_axis0'] = {
    'op': 'Squeeze', 'description': '[1,1,3] squeeze axis=0 -> [1,3]',
    'input': {'shape': [1, 1, 3], 'data': x_sq.flatten().tolist()},
    'attributes': {'axes': [0]},
    'expected': {'shape': list(result_sq.shape), 'data': x_sq.flatten().tolist()}
}
result_sq_both = np.squeeze(x_sq)  # [3]
tests['squeeze_all'] = {
    'op': 'Squeeze', 'description': '[1,1,3] squeeze all -> [3]',
    'input': {'shape': [1, 1, 3], 'data': x_sq.flatten().tolist()},
    'attributes': {'axes': [0, 1]},
    'expected': {'shape': list(result_sq_both.shape), 'data': x_sq.flatten().tolist()}
}
print(f'  axis=0: [1,1,3]->[1,3], all: [1,1,3]->[3]')

# ============================================
# FLATTEN
# ============================================
print('=== Flatten ===')

x_fl = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
result_fl = x_fl.reshape(2, 12)  # axis=1 (default)
tests['flatten_axis1'] = {
    'op': 'Flatten', 'description': '[2,3,4] flatten axis=1 -> [2,12]',
    'input': {'shape': [2, 3, 4], 'data': x_fl.flatten().tolist()},
    'attributes': {'axis': 1},
    'expected': {'shape': [2, 12], 'data': x_fl.flatten().tolist()}
}
print(f'  axis=1: [2,3,4]->[2,12]')

# ============================================
# CAST (type conversion)
# ============================================
print('=== Cast ===')

x_cast = np.array([1.7, 2.3, -0.5, 3.9], dtype=np.float32)
result_cast_int = x_cast.astype(np.int64)
tests['cast_float_to_int'] = {
    'op': 'Cast', 'description': '[4] float32 -> int64',
    'input': {'shape': [4], 'data': x_cast.tolist()},
    'attributes': {'to': 7},  # ONNX TensorProto.INT64 = 7
    'expected': {'shape': [4], 'data': result_cast_int.tolist()}
}
print(f'  float->int: {result_cast_int.tolist()}')

# ============================================
# PAD
# ============================================
print('=== Pad ===')

x_pad = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)  # [2,3]
# Constant pad: 1 top, 1 bottom, 2 left, 2 right
result_pad = np.pad(x_pad, ((1, 1), (2, 2)), mode='constant', constant_values=0)
tests['pad_constant'] = {
    'op': 'Pad', 'description': '[2,3] pad constant [1,2,1,2] -> [4,7]',
    'input': {'shape': [2, 3], 'data': x_pad.flatten().tolist()},
    'attributes': {'pads': [1, 2, 1, 2], 'mode': 'constant', 'value': 0.0},
    'expected': {'shape': list(result_pad.shape), 'data': result_pad.flatten().tolist()}
}
print(f'  constant: [2,3]->[4,7]')

# Reflect pad (used in style transfer)
x_pad2 = np.arange(1, 10, dtype=np.float32).reshape(1, 1, 3, 3)
result_pad2 = np.pad(x_pad2, ((0, 0), (0, 0), (1, 1), (1, 1)), mode='reflect')
tests['pad_reflect'] = {
    'op': 'Pad', 'description': '[1,1,3,3] reflect pad [1,1,1,1] -> [1,1,5,5]',
    'input': {'shape': [1, 1, 3, 3], 'data': x_pad2.flatten().tolist()},
    'attributes': {'pads': [0, 0, 1, 1, 0, 0, 1, 1], 'mode': 'reflect'},
    'expected': {'shape': list(result_pad2.shape), 'data': result_pad2.flatten().tolist()}
}
print(f'  reflect: [1,1,3,3]->[1,1,5,5]')

# ============================================
# RESIZE (bilinear upscale)
# ============================================
print('=== Resize ===')

# Simple 2x upscale via scales — half_pixel coordinate transform
x_resize = np.array([[[[1, 2], [3, 4]]]], dtype=np.float32)  # [1,1,2,2]
# Manual half_pixel bilinear: src = (dst + 0.5) / scale - 0.5
in_h, in_w = 2, 2
out_h, out_w = 4, 4
scale_h, scale_w = out_h / in_h, out_w / in_w
result_resize = np.zeros((1, 1, out_h, out_w), dtype=np.float32)
for oy in range(out_h):
    for ox in range(out_w):
        sy = max(0, min((oy + 0.5) / scale_h - 0.5, in_h - 1))
        sx = max(0, min((ox + 0.5) / scale_w - 0.5, in_w - 1))
        y0, x0 = int(sy), int(sx)
        y1, x1 = min(y0 + 1, in_h - 1), min(x0 + 1, in_w - 1)
        fy, fx = sy - y0, sx - x0
        result_resize[0, 0, oy, ox] = (
            x_resize[0, 0, y0, x0] * (1-fy) * (1-fx) +
            x_resize[0, 0, y0, x1] * (1-fy) * fx +
            x_resize[0, 0, y1, x0] * fy * (1-fx) +
            x_resize[0, 0, y1, x1] * fy * fx)
tests['resize_bilinear_2x'] = {
    'op': 'Resize', 'description': '[1,1,2,2] bilinear 2x -> [1,1,4,4]',
    'input': {'shape': [1, 1, 2, 2], 'data': x_resize.flatten().tolist()},
    'attributes': {
        'mode': 'linear',
        'scales': [1.0, 1.0, 2.0, 2.0],
        'coordinate_transformation_mode': 'half_pixel'
    },
    'expected': {'shape': list(result_resize.shape), 'data': result_resize.flatten().tolist()}
}
print(f'  bilinear 2x: [1,1,2,2]->[1,1,4,4]')

# ============================================
# ACTIVATION FUNCTIONS
# ============================================
print('=== Activations ===')

x_act = np.array([-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0], dtype=np.float32)

# Sigmoid
result_sig = 1.0 / (1.0 + np.exp(-x_act))
tests['sigmoid'] = {
    'op': 'Sigmoid', 'description': 'sigmoid [-3..3]',
    'input': {'shape': [7], 'data': x_act.tolist()},
    'expected': {'shape': [7], 'data': result_sig.tolist()}
}
print(f'  sigmoid: {[round(x, 4) for x in result_sig.tolist()]}')

# Tanh
result_tanh = np.tanh(x_act)
tests['tanh'] = {
    'op': 'Tanh', 'description': 'tanh [-3..3]',
    'input': {'shape': [7], 'data': x_act.tolist()},
    'expected': {'shape': [7], 'data': result_tanh.tolist()}
}
print(f'  tanh: {[round(x, 4) for x in result_tanh.tolist()]}')

# GELU (approximate — erf version)
result_gelu = 0.5 * x_act * (1.0 + np.vectorize(math.erf)(x_act / math.sqrt(2.0)))
tests['gelu'] = {
    'op': 'Gelu', 'description': 'gelu [-3..3]',
    'input': {'shape': [7], 'data': x_act.tolist()},
    'expected': {'shape': [7], 'data': result_gelu.astype(np.float32).tolist()}
}
print(f'  gelu: {[round(x, 4) for x in result_gelu.tolist()]}')

# Clip (min=0, max=6 — ReLU6 pattern)
result_clip = np.clip(x_act, 0, 6)
tests['clip_relu6'] = {
    'op': 'Clip', 'description': 'clip min=0 max=6 (ReLU6)',
    'input': {'shape': [7], 'data': x_act.tolist()},
    'attributes': {'min': 0.0, 'max': 6.0},
    'expected': {'shape': [7], 'data': result_clip.tolist()}
}
print(f'  clip: {result_clip.tolist()}')

# HardSigmoid: max(0, min(1, alpha*x + beta)), alpha=0.2, beta=0.5
alpha_hs = 0.2
beta_hs = 0.5
result_hs = np.clip(alpha_hs * x_act + beta_hs, 0, 1)
tests['hardsigmoid'] = {
    'op': 'HardSigmoid', 'description': 'hardsigmoid alpha=0.2 beta=0.5',
    'input': {'shape': [7], 'data': x_act.tolist()},
    'attributes': {'alpha': alpha_hs, 'beta': beta_hs},
    'expected': {'shape': [7], 'data': result_hs.astype(np.float32).tolist()}
}
print(f'  hardsigmoid: {[round(x, 4) for x in result_hs.tolist()]}')

# HardSwish: x * HardSigmoid(x) with alpha=1/6, beta=0.5
result_hswish = x_act * np.clip(x_act / 6.0 + 0.5, 0, 1)
tests['hardswish'] = {
    'op': 'HardSwish', 'description': 'hardswish',
    'input': {'shape': [7], 'data': x_act.tolist()},
    'expected': {'shape': [7], 'data': result_hswish.astype(np.float32).tolist()}
}
print(f'  hardswish: {[round(x, 4) for x in result_hswish.tolist()]}')

# SiLU (Swish): x * sigmoid(x)
result_silu = x_act * result_sig
tests['silu'] = {
    'op': 'SiLU', 'description': 'silu (swish) [-3..3]',
    'input': {'shape': [7], 'data': x_act.tolist()},
    'expected': {'shape': [7], 'data': result_silu.tolist()}
}
print(f'  silu: {[round(x, 4) for x in result_silu.tolist()]}')

# ============================================
# ELEMENT-WISE MATH
# ============================================
print('=== Element-wise Math ===')

x_math = np.array([0.25, 1.0, 4.0, 9.0, 16.0], dtype=np.float32)

# Sqrt
result_sqrt = np.sqrt(x_math)
tests['sqrt'] = {
    'op': 'Sqrt', 'description': 'sqrt [0.25,1,4,9,16]',
    'input': {'shape': [5], 'data': x_math.tolist()},
    'expected': {'shape': [5], 'data': result_sqrt.tolist()}
}
print(f'  sqrt: {result_sqrt.tolist()}')

# Exp
x_exp_m = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
result_exp_m = np.exp(x_exp_m)
tests['exp'] = {
    'op': 'Exp', 'description': 'exp [-2..2]',
    'input': {'shape': [5], 'data': x_exp_m.tolist()},
    'expected': {'shape': [5], 'data': result_exp_m.tolist()}
}
print(f'  exp: {[round(x, 4) for x in result_exp_m.tolist()]}')

# Log
x_log = np.array([0.5, 1.0, 2.0, math.e, 10.0], dtype=np.float32)
result_log = np.log(x_log)
tests['log'] = {
    'op': 'Log', 'description': 'log [0.5,1,2,e,10]',
    'input': {'shape': [5], 'data': x_log.tolist()},
    'expected': {'shape': [5], 'data': result_log.tolist()}
}
print(f'  log: {[round(x, 4) for x in result_log.tolist()]}')

# Neg
x_neg = np.array([-3.0, -1.0, 0.0, 2.0, 5.0], dtype=np.float32)
result_neg = -x_neg
tests['neg'] = {
    'op': 'Neg', 'description': 'neg [-3,-1,0,2,5]',
    'input': {'shape': [5], 'data': x_neg.tolist()},
    'expected': {'shape': [5], 'data': result_neg.tolist()}
}
print(f'  neg: {result_neg.tolist()}')

# Abs
x_abs = np.array([-3.0, -1.5, 0.0, 1.5, 3.0], dtype=np.float32)
result_abs = np.abs(x_abs)
tests['abs'] = {
    'op': 'Abs', 'description': 'abs [-3,-1.5,0,1.5,3]',
    'input': {'shape': [5], 'data': x_abs.tolist()},
    'expected': {'shape': [5], 'data': result_abs.tolist()}
}
print(f'  abs: {result_abs.tolist()}')

# Reciprocal
x_recip = np.array([0.5, 1.0, 2.0, 4.0, 10.0], dtype=np.float32)
result_recip = 1.0 / x_recip
tests['reciprocal'] = {
    'op': 'Reciprocal', 'description': 'reciprocal [0.5,1,2,4,10]',
    'input': {'shape': [5], 'data': x_recip.tolist()},
    'expected': {'shape': [5], 'data': result_recip.tolist()}
}
print(f'  reciprocal: {result_recip.tolist()}')

# Erf
x_erf = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
result_erf = np.vectorize(math.erf)(x_erf).astype(np.float32)
tests['erf'] = {
    'op': 'Erf', 'description': 'erf [-2..2]',
    'input': {'shape': [5], 'data': x_erf.tolist()},
    'expected': {'shape': [5], 'data': result_erf.tolist()}
}
print(f'  erf: {[round(x, 4) for x in result_erf.tolist()]}')

# Pow
x_pow = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
exp_pow = np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32)
result_pow = np.power(x_pow, exp_pow)
tests['pow_square'] = {
    'op': 'Pow', 'description': 'pow [1,2,3,4]^2',
    'input_a': {'shape': [4], 'data': x_pow.tolist()},
    'input_b': {'shape': [4], 'data': exp_pow.tolist()},
    'expected': {'shape': [4], 'data': result_pow.tolist()}
}
print(f'  pow^2: {result_pow.tolist()}')

# Div
x_div_a = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
x_div_b = np.array([2.0, 5.0, 3.0, 8.0], dtype=np.float32)
result_div = x_div_a / x_div_b
tests['div'] = {
    'op': 'Div', 'description': '[10,20,30,40] / [2,5,3,8]',
    'input_a': {'shape': [4], 'data': x_div_a.tolist()},
    'input_b': {'shape': [4], 'data': x_div_b.tolist()},
    'expected': {'shape': [4], 'data': result_div.tolist()}
}
print(f'  div: {result_div.tolist()}')

# Div broadcast (scalar divisor)
x_div_s = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
divisor = np.array([2.0], dtype=np.float32)
result_div_s = x_div_s / divisor
tests['div_scalar_broadcast'] = {
    'op': 'Div', 'description': '[2,3] / [1] scalar broadcast',
    'input_a': {'shape': [2, 3], 'data': x_div_s.flatten().tolist()},
    'input_b': {'shape': [1], 'data': divisor.tolist()},
    'expected': {'shape': [2, 3], 'data': result_div_s.flatten().tolist()}
}
print(f'  div scalar: {result_div_s.flatten().tolist()}')

# Floor and Ceil
x_floor = np.array([-1.7, -0.3, 0.0, 0.7, 2.5], dtype=np.float32)
tests['floor'] = {
    'op': 'Floor', 'description': 'floor [-1.7,-0.3,0,0.7,2.5]',
    'input': {'shape': [5], 'data': x_floor.tolist()},
    'expected': {'shape': [5], 'data': np.floor(x_floor).tolist()}
}
tests['ceil'] = {
    'op': 'Ceil', 'description': 'ceil [-1.7,-0.3,0,0.7,2.5]',
    'input': {'shape': [5], 'data': x_floor.tolist()},
    'expected': {'shape': [5], 'data': np.ceil(x_floor).tolist()}
}
print(f'  floor: {np.floor(x_floor).tolist()}')
print(f'  ceil: {np.ceil(x_floor).tolist()}')

# ============================================
# COMPARISON / LOGIC
# ============================================
print('=== Comparison ===')

x_cmp_a = np.array([1, 3, 5, 2, 4], dtype=np.float32)
x_cmp_b = np.array([2, 3, 1, 4, 4], dtype=np.float32)

tests['equal'] = {
    'op': 'Equal', 'description': 'equal element-wise',
    'input_a': {'shape': [5], 'data': x_cmp_a.tolist()},
    'input_b': {'shape': [5], 'data': x_cmp_b.tolist()},
    'expected': {'shape': [5], 'data': (x_cmp_a == x_cmp_b).astype(np.int32).tolist()}
}
tests['greater'] = {
    'op': 'Greater', 'description': 'greater element-wise',
    'input_a': {'shape': [5], 'data': x_cmp_a.tolist()},
    'input_b': {'shape': [5], 'data': x_cmp_b.tolist()},
    'expected': {'shape': [5], 'data': (x_cmp_a > x_cmp_b).astype(np.int32).tolist()}
}
tests['less'] = {
    'op': 'Less', 'description': 'less element-wise',
    'input_a': {'shape': [5], 'data': x_cmp_a.tolist()},
    'input_b': {'shape': [5], 'data': x_cmp_b.tolist()},
    'expected': {'shape': [5], 'data': (x_cmp_a < x_cmp_b).astype(np.int32).tolist()}
}
tests['less_or_equal'] = {
    'op': 'LessOrEqual', 'description': 'less_or_equal element-wise',
    'input_a': {'shape': [5], 'data': x_cmp_a.tolist()},
    'input_b': {'shape': [5], 'data': x_cmp_b.tolist()},
    'expected': {'shape': [5], 'data': (x_cmp_a <= x_cmp_b).astype(np.int32).tolist()}
}
print(f'  equal: {(x_cmp_a == x_cmp_b).astype(int).tolist()}')
print(f'  greater: {(x_cmp_a > x_cmp_b).astype(int).tolist()}')
print(f'  less: {(x_cmp_a < x_cmp_b).astype(int).tolist()}')

# Not (logical)
x_not = np.array([1, 0, 1, 0, 1], dtype=np.int32)
tests['not_logical'] = {
    'op': 'Not', 'description': 'logical not',
    'input': {'shape': [5], 'data': x_not.tolist()},
    'expected': {'shape': [5], 'data': (1 - x_not).tolist()}
}
print(f'  not: {(1 - x_not).tolist()}')

# ============================================
# SOFTMAX (additional axes)
# ============================================
print('=== Softmax ===')

# Softmax axis=0 (column-wise)
x_sm = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
e_sm = np.exp(x_sm - x_sm.max(axis=0, keepdims=True))
result_sm = e_sm / e_sm.sum(axis=0, keepdims=True)
tests['softmax_axis0'] = {
    'op': 'Softmax', 'description': '[3,2] softmax axis=0 (column-wise)',
    'input': {'shape': [3, 2], 'data': x_sm.flatten().tolist()},
    'attributes': {'axis': 0},
    'expected': {'shape': [3, 2], 'data': result_sm.flatten().tolist()}
}
print(f'  axis=0: first col sums to {result_sm[:, 0].sum():.4f}')

# Softmax on 3D (attention scores pattern)
x_sm3 = np.array([[[1, 2, 3, 4], [5, 6, 7, 8]]], dtype=np.float32)  # [1,2,4]
e_sm3 = np.exp(x_sm3 - x_sm3.max(axis=-1, keepdims=True))
result_sm3 = e_sm3 / e_sm3.sum(axis=-1, keepdims=True)
tests['softmax_3d_last_axis'] = {
    'op': 'Softmax', 'description': '[1,2,4] softmax axis=-1 (attention pattern)',
    'input': {'shape': [1, 2, 4], 'data': x_sm3.flatten().tolist()},
    'attributes': {'axis': -1},
    'expected': {'shape': [1, 2, 4], 'data': result_sm3.flatten().tolist()}
}
print(f'  3D last axis: row sums to {result_sm3[0, 0].sum():.4f}')

# ============================================
# CONCAT (additional patterns)
# ============================================
print('=== Concat (additional) ===')

# axis=1 channel concat (SqueezeNet pattern)
a_cc1 = np.arange(1, 13, dtype=np.float32).reshape(1, 3, 2, 2)
b_cc1 = np.arange(13, 21, dtype=np.float32).reshape(1, 2, 2, 2)
result_cc1 = np.concatenate([a_cc1, b_cc1], axis=1)
tests['concat_axis1_channel'] = {
    'op': 'Concat', 'description': '[1,3,2,2]+[1,2,2,2] axis=1 -> [1,5,2,2] channel concat',
    'input_a': {'shape': [1, 3, 2, 2], 'data': a_cc1.flatten().tolist()},
    'input_b': {'shape': [1, 2, 2, 2], 'data': b_cc1.flatten().tolist()},
    'attributes': {'axis': 1},
    'expected': {'shape': list(result_cc1.shape), 'data': result_cc1.flatten().tolist()}
}
print(f'  axis=1 channel: [1,3,2,2]+[1,2,2,2]->[1,5,2,2]')

# 3-input concat
c_cc = np.arange(21, 29, dtype=np.float32).reshape(1, 2, 2, 2)
result_cc3 = np.concatenate([a_cc1, b_cc1, c_cc], axis=1)
tests['concat_3_inputs'] = {
    'op': 'Concat', 'description': '3-input concat axis=1 -> [1,7,2,2]',
    'input_a': {'shape': [1, 3, 2, 2], 'data': a_cc1.flatten().tolist()},
    'input_b': {'shape': [1, 2, 2, 2], 'data': b_cc1.flatten().tolist()},
    'input_c': {'shape': [1, 2, 2, 2], 'data': c_cc.flatten().tolist()},
    'attributes': {'axis': 1},
    'expected': {'shape': list(result_cc3.shape), 'data': result_cc3.flatten().tolist()}
}
print(f'  3-input: [1,3+2+2,2,2]->[1,7,2,2]')

# ============================================
# TRANSPOSE (additional permutations)
# ============================================
print('=== Transpose (additional) ===')

# NCHW -> NHWC (common for TFLite models)
x_tr = np.arange(24, dtype=np.float32).reshape(1, 3, 2, 4)
result_tr = x_tr.transpose(0, 2, 3, 1)
tests['transpose_nchw_to_nhwc'] = {
    'op': 'Transpose', 'description': '[1,3,2,4] NCHW->NHWC perm=[0,2,3,1] -> [1,2,4,3]',
    'input': {'shape': [1, 3, 2, 4], 'data': x_tr.flatten().tolist()},
    'attributes': {'perm': [0, 2, 3, 1]},
    'expected': {'shape': list(result_tr.shape), 'data': result_tr.flatten().tolist()}
}
print(f'  NCHW->NHWC: [1,3,2,4]->[1,2,4,3]')

# Attention transpose: [B,H,S,D] -> [B,S,H,D]
x_tr2 = np.arange(48, dtype=np.float32).reshape(1, 2, 6, 4)  # [B,H,S,D]
result_tr2 = x_tr2.transpose(0, 2, 1, 3)  # [B,S,H,D]
tests['transpose_attention'] = {
    'op': 'Transpose', 'description': '[1,2,6,4] attention perm=[0,2,1,3] -> [1,6,2,4]',
    'input': {'shape': [1, 2, 6, 4], 'data': x_tr2.flatten().tolist()},
    'attributes': {'perm': [0, 2, 1, 3]},
    'expected': {'shape': list(result_tr2.shape), 'data': result_tr2.flatten().tolist()}
}
print(f'  attention: [1,2,6,4]->[1,6,2,4]')

# ============================================
# MIN / MAX element-wise
# ============================================
print('=== Min/Max element-wise ===')

x_mm_a = np.array([1, 5, 3, 7, 2], dtype=np.float32)
x_mm_b = np.array([4, 2, 6, 1, 8], dtype=np.float32)

tests['min_elementwise'] = {
    'op': 'Min', 'description': 'element-wise min',
    'input_a': {'shape': [5], 'data': x_mm_a.tolist()},
    'input_b': {'shape': [5], 'data': x_mm_b.tolist()},
    'expected': {'shape': [5], 'data': np.minimum(x_mm_a, x_mm_b).tolist()}
}
tests['max_elementwise'] = {
    'op': 'Max', 'description': 'element-wise max',
    'input_a': {'shape': [5], 'data': x_mm_a.tolist()},
    'input_b': {'shape': [5], 'data': x_mm_b.tolist()},
    'expected': {'shape': [5], 'data': np.maximum(x_mm_a, x_mm_b).tolist()}
}
print(f'  min: {np.minimum(x_mm_a, x_mm_b).tolist()}')
print(f'  max: {np.maximum(x_mm_a, x_mm_b).tolist()}')

# ============================================
# CONSTANTOFSHAPE
# ============================================
print('=== ConstantOfShape ===')

# Fill with zero (default)
tests['constantofshape_zero'] = {
    'op': 'ConstantOfShape', 'description': 'fill [2,3] with 0',
    'input_shape': {'shape': [2], 'data': [2, 3]},
    'attributes': {'value': 0.0},
    'expected': {'shape': [2, 3], 'data': [0.0] * 6}
}

# Fill with -inf (attention mask pattern — the DistilBERT bug!)
tests['constantofshape_neginf'] = {
    'op': 'ConstantOfShape', 'description': 'fill [1,1,6,6] with -inf (attention mask)',
    'input_shape': {'shape': [4], 'data': [1, 1, 6, 6]},
    'attributes': {'value': float('-inf')},
    'expected': {'shape': [1, 1, 6, 6], 'data': [float('-inf')] * 36}
}

# Fill with 1.0
tests['constantofshape_ones'] = {
    'op': 'ConstantOfShape', 'description': 'fill [3,4] with 1.0',
    'input_shape': {'shape': [2], 'data': [3, 4]},
    'attributes': {'value': 1.0},
    'expected': {'shape': [3, 4], 'data': [1.0] * 12}
}
print(f'  zero [2,3], neginf [1,1,6,6], ones [3,4]')

# ============================================
# DEPTH TO SPACE CRD mode
# ============================================
print('=== DepthToSpace CRD ===')

x_d2s_crd = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]],
                         [[9, 10], [11, 12]], [[13, 14], [15, 16]]]], dtype=np.float32)
r_crd = 2
b_crd, c_in_crd, h_crd, w_crd = x_d2s_crd.shape
c_out_crd = c_in_crd // (r_crd * r_crd)
out_d2s_crd = np.zeros((b_crd, c_out_crd, h_crd * r_crd, w_crd * r_crd), dtype=np.float32)
for bx in range(b_crd):
    for cx in range(c_out_crd):
        for hy in range(h_crd):
            for wx in range(w_crd):
                for dy in range(r_crd):
                    for dx in range(r_crd):
                        out_d2s_crd[bx, cx, hy*r_crd+dy, wx*r_crd+dx] = x_d2s_crd[bx, cx*(r_crd*r_crd)+r_crd*dx+dy, hy, wx]
tests['depth_to_space_crd'] = {
    'op': 'DepthToSpace', 'description': '[1,4,2,2] blocksize=2 CRD mode -> [1,1,4,4]',
    'input': {'shape': [1, 4, 2, 2], 'data': x_d2s_crd.flatten().tolist()},
    'attributes': {'blocksize': 2, 'mode': 'CRD'},
    'expected': {'shape': list(out_d2s_crd.shape), 'data': out_d2s_crd.flatten().tolist()}
}
print(f'  CRD mode: {out_d2s_crd.flatten().tolist()}')

# ============================================
# SAVE
# ============================================
out_path = os.path.join(out_dir, 'operator_test_cases.json')
with open(out_path, 'w') as f:
    json.dump(tests, f, indent=2)

print(f'\n{"=" * 60}')
print(f'Saved {len(tests)} total test cases to {out_path}')
print(f'File size: {os.path.getsize(out_path):,} bytes')

# Count new vs existing
original_count = 18  # known from before
new_count = len(tests) - original_count
print(f'Original: {original_count}, New: {new_count}')
