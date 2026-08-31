#!/usr/bin/env python3
"""Generate operator-level test data with known inputs and outputs."""

import numpy as np
import json
import os

out_dir = r'D:\users\tj\Projects\SpawnDev.ILGPU.ML\SpawnDev.ILGPU.ML\SpawnDev.ILGPU.ML.Demo\wwwroot\test-models'
os.makedirs(out_dir, exist_ok=True)

tests = {}

# ============================================
# 1. BROADCAST ADD
# ============================================
print('=== Broadcast Add ===')

# Case A: [2,3,4] + [4] — broadcast last dim
a = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
b = np.array([10, 20, 30, 40], dtype=np.float32)
result = a + b
tests['broadcast_add_last_dim'] = {
    'op': 'Add', 'description': '[2,3,4] + [4] broadcast last dim',
    'input_a': {'shape': [2, 3, 4], 'data': a.flatten().tolist()},
    'input_b': {'shape': [4], 'data': b.tolist()},
    'expected': {'shape': [2, 3, 4], 'data': result.flatten().tolist()}
}
print(f'  [2,3,4]+[4]: first 4 = {result.flatten()[:4].tolist()}')

# Case B: [2,3,4] + [3,1] — broadcast mid dim
b2 = np.array([[100], [200], [300]], dtype=np.float32)
result2 = a + b2
tests['broadcast_add_mid_dim'] = {
    'op': 'Add', 'description': '[2,3,4] + [3,1] broadcast mid dim',
    'input_a': {'shape': [2, 3, 4], 'data': a.flatten().tolist()},
    'input_b': {'shape': [3, 1], 'data': b2.flatten().tolist()},
    'expected': {'shape': [2, 3, 4], 'data': result2.flatten().tolist()}
}
print(f'  [2,3,4]+[3,1]: first 4 = {result2.flatten()[:4].tolist()}')

# Case C: [1,3,1] + [2,1,4] — full broadcast both sides
a3 = np.array([[[1], [2], [3]]], dtype=np.float32)
b3 = np.arange(8, dtype=np.float32).reshape(2, 1, 4)
result3 = a3 + b3
tests['broadcast_add_both_sides'] = {
    'op': 'Add', 'description': '[1,3,1] + [2,1,4] full broadcast',
    'input_a': {'shape': [1, 3, 1], 'data': a3.flatten().tolist()},
    'input_b': {'shape': [2, 1, 4], 'data': b3.flatten().tolist()},
    'expected': {'shape': [2, 3, 4], 'data': result3.flatten().tolist()}
}
print(f'  [1,3,1]+[2,1,4]: shape={list(result3.shape)}')

# ============================================
# 2. BROADCAST MUL — BatchNorm scale pattern
# ============================================
print('=== Broadcast Mul ===')
scale = np.array([[[[2.0]], [[0.5]], [[3.0]]]], dtype=np.float32)  # [1,3,1,1]
x = np.ones((1, 3, 2, 2), dtype=np.float32) * 10
result_mul = x * scale
tests['broadcast_mul_scale'] = {
    'op': 'Mul', 'description': '[1,3,2,2] * [1,3,1,1] channel-wise scale',
    'input_a': {'shape': [1, 3, 2, 2], 'data': x.flatten().tolist()},
    'input_b': {'shape': [1, 3, 1, 1], 'data': scale.flatten().tolist()},
    'expected': {'shape': [1, 3, 2, 2], 'data': result_mul.flatten().tolist()}
}
print(f'  Channel scale: {result_mul.flatten().tolist()}')

# ============================================
# 3. BROADCAST SUB — mean subtraction
# ============================================
print('=== Broadcast Sub ===')
x_sub = np.array([[[[100, 150], [120, 130]], [[80, 90], [85, 95]], [[200, 210], [205, 215]]]], dtype=np.float32)
mean_4d = np.array([[[[100]], [[85]], [[200]]]], dtype=np.float32)
result_sub = x_sub - mean_4d
tests['broadcast_sub_mean'] = {
    'op': 'Sub', 'description': '[1,3,2,2] - [1,3,1,1] mean subtraction',
    'input_a': {'shape': [1, 3, 2, 2], 'data': x_sub.flatten().tolist()},
    'input_b': {'shape': [1, 3, 1, 1], 'data': mean_4d.flatten().tolist()},
    'expected': {'shape': [1, 3, 2, 2], 'data': result_sub.flatten().tolist()}
}
print(f'  Mean sub: {result_sub.flatten().tolist()}')

# ============================================
# 4. AVERAGEPOOL
# ============================================
print('=== AveragePool ===')

# 3x3 stride=1 no-pad
x_pool = np.arange(1, 26, dtype=np.float32).reshape(1, 1, 5, 5)
out_pool = np.zeros((1, 1, 3, 3), dtype=np.float32)
for i in range(3):
    for j in range(3):
        out_pool[0, 0, i, j] = x_pool[0, 0, i:i+3, j:j+3].mean()
tests['avgpool_3x3_s1'] = {
    'op': 'AveragePool', 'description': '[1,1,5,5] k=3x3 s=1 -> [1,1,3,3]',
    'input': {'shape': [1, 1, 5, 5], 'data': x_pool.flatten().tolist()},
    'attributes': {'kernel_shape': [3, 3], 'strides': [1, 1], 'pads': [0, 0, 0, 0]},
    'expected': {'shape': [1, 1, 3, 3], 'data': out_pool.flatten().tolist()}
}
print(f'  3x3 s1: {out_pool.flatten().tolist()}')

# 2x2 stride=2
x_pool2 = np.arange(1, 17, dtype=np.float32).reshape(1, 1, 4, 4)
out_pool2 = np.zeros((1, 1, 2, 2), dtype=np.float32)
for i in range(2):
    for j in range(2):
        out_pool2[0, 0, i, j] = x_pool2[0, 0, i*2:i*2+2, j*2:j*2+2].mean()
tests['avgpool_2x2_s2'] = {
    'op': 'AveragePool', 'description': '[1,1,4,4] k=2x2 s=2 -> [1,1,2,2]',
    'input': {'shape': [1, 1, 4, 4], 'data': x_pool2.flatten().tolist()},
    'attributes': {'kernel_shape': [2, 2], 'strides': [2, 2], 'pads': [0, 0, 0, 0]},
    'expected': {'shape': [1, 1, 2, 2], 'data': out_pool2.flatten().tolist()}
}
print(f'  2x2 s2: {out_pool2.flatten().tolist()}')

# ============================================
# 5. SLICE
# ============================================
print('=== Slice ===')

x_slice = np.arange(24, dtype=np.float32).reshape(1, 4, 6)
result_slice = x_slice[:, 1:3, 2:5]
tests['slice_2d'] = {
    'op': 'Slice', 'description': '[1,4,6][:,1:3,2:5] -> [1,2,3]',
    'input': {'shape': [1, 4, 6], 'data': x_slice.flatten().tolist()},
    'attributes': {'starts': [1, 2], 'ends': [3, 5], 'axes': [1, 2]},
    'expected': {'shape': list(result_slice.shape), 'data': result_slice.flatten().tolist()}
}
print(f'  [:,1:3,2:5]: {result_slice.flatten().tolist()}')

# Slice with stride
result_stride = x_slice[:, ::2, ::3]
tests['slice_strided'] = {
    'op': 'Slice', 'description': '[1,4,6][:,::2,::3] strided',
    'input': {'shape': [1, 4, 6], 'data': x_slice.flatten().tolist()},
    'attributes': {'starts': [0, 0], 'ends': [4, 6], 'axes': [1, 2], 'steps': [2, 3]},
    'expected': {'shape': list(result_stride.shape), 'data': result_stride.flatten().tolist()}
}
print(f'  [:,::2,::3]: {result_stride.flatten().tolist()}')

# ============================================
# 6. DEPTH TO SPACE (pixel shuffle)
# ============================================
print('=== DepthToSpace ===')

x_d2s = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]],
                    [[9, 10], [11, 12]], [[13, 14], [15, 16]]]], dtype=np.float32)
r = 2
b, c_in, h, w = x_d2s.shape
c_out = c_in // (r * r)
out_d2s = np.zeros((b, c_out, h * r, w * r), dtype=np.float32)
for bx in range(b):
    for cx in range(c_out):
        for hy in range(h):
            for wx in range(w):
                for dy in range(r):
                    for dx in range(r):
                        out_d2s[bx, cx, hy*r+dy, wx*r+dx] = x_d2s[bx, cx*(r*r)+dy*r+dx, hy, wx]
tests['depth_to_space_b2'] = {
    'op': 'DepthToSpace', 'description': '[1,4,2,2] blocksize=2 -> [1,1,4,4] pixel shuffle',
    'input': {'shape': [1, 4, 2, 2], 'data': x_d2s.flatten().tolist()},
    'attributes': {'blocksize': 2, 'mode': 'DCR'},
    'expected': {'shape': list(out_d2s.shape), 'data': out_d2s.flatten().tolist()}
}
print(f'  blocksize=2: {out_d2s.flatten().tolist()}')

# ============================================
# 7. ARGMAX
# ============================================
print('=== ArgMax ===')

logits = np.array([[0.1, 0.5, 0.2, 0.9, 0.3]], dtype=np.float32)
tests['argmax_axis1'] = {
    'op': 'ArgMax', 'description': '[1,5] argmax axis=1 -> index 3',
    'input': {'shape': [1, 5], 'data': logits.flatten().tolist()},
    'attributes': {'axis': 1, 'keepdims': 0},
    'expected': {'shape': [1], 'data': [3]}
}
print(f'  axis=1: argmax=3')

x_am = np.array([[1, 5, 3, 2], [4, 2, 6, 1], [3, 3, 1, 7]], dtype=np.float32)
tests['argmax_axis0'] = {
    'op': 'ArgMax', 'description': '[3,4] argmax axis=0 -> [4]',
    'input': {'shape': [3, 4], 'data': x_am.flatten().tolist()},
    'attributes': {'axis': 0, 'keepdims': 0},
    'expected': {'shape': [4], 'data': np.argmax(x_am, axis=0).tolist()}
}
print(f'  axis=0: {np.argmax(x_am, axis=0).tolist()}')

# ============================================
# 8. WHERE (boolean select)
# ============================================
print('=== Where ===')

condition = [[True, False, True], [False, True, False]]
x_w = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
y_w = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.float32)
result_w = np.where(condition, x_w, y_w)
tests['where_2d'] = {
    'op': 'Where', 'description': '[2,3] where(cond, x, y)',
    'condition': {'shape': [2, 3], 'data': [1, 0, 1, 0, 1, 0]},
    'input_x': {'shape': [2, 3], 'data': x_w.flatten().tolist()},
    'input_y': {'shape': [2, 3], 'data': y_w.flatten().tolist()},
    'expected': {'shape': [2, 3], 'data': result_w.flatten().tolist()}
}
print(f'  where: {result_w.flatten().tolist()}')

# ============================================
# 9. REDUCEMAX axis=0
# ============================================
print('=== ReduceMax ===')

x_rm = np.array([[1, 5, 3], [4, 2, 6], [3, 7, 1]], dtype=np.float32)
result_rm = np.max(x_rm, axis=0, keepdims=True)
tests['reducemax_axis0'] = {
    'op': 'ReduceMax', 'description': '[3,3] reducemax axis=0 keepdims -> [1,3]',
    'input': {'shape': [3, 3], 'data': x_rm.flatten().tolist()},
    'attributes': {'axes': [0], 'keepdims': 1},
    'expected': {'shape': list(result_rm.shape), 'data': result_rm.flatten().tolist()}
}
print(f'  axis=0: {result_rm.flatten().tolist()}')

# ReduceMin
result_rmin = np.min(x_rm, axis=0, keepdims=True)
tests['reducemin_axis0'] = {
    'op': 'ReduceMin', 'description': '[3,3] reducemin axis=0 keepdims -> [1,3]',
    'input': {'shape': [3, 3], 'data': x_rm.flatten().tolist()},
    'attributes': {'axes': [0], 'keepdims': 1},
    'expected': {'shape': list(result_rmin.shape), 'data': result_rmin.flatten().tolist()}
}
print(f'  min axis=0: {result_rmin.flatten().tolist()}')

# ============================================
# 10. LEAKYRELU
# ============================================
print('=== LeakyReLU ===')

x_lr = np.array([-3, -1, 0, 1, 3], dtype=np.float32)
alpha = 0.01
result_lr = np.where(x_lr >= 0, x_lr, alpha * x_lr)
tests['leakyrelu_alpha001'] = {
    'op': 'LeakyRelu', 'description': '[-3,-1,0,1,3] alpha=0.01',
    'input': {'shape': [5], 'data': x_lr.tolist()},
    'attributes': {'alpha': alpha},
    'expected': {'shape': [5], 'data': result_lr.tolist()}
}
print(f'  alpha=0.01: {result_lr.tolist()}')

# ============================================
# 11. RESHAPE with -1
# ============================================
print('=== Reshape ===')

x_rs = np.arange(24, dtype=np.float32)
tests['reshape_infer_dim'] = {
    'op': 'Reshape', 'description': '[24] reshape to [2,3,-1] -> [2,3,4]',
    'input': {'shape': [24], 'data': x_rs.tolist()},
    'attributes': {'shape': [2, 3, -1]},
    'expected': {'shape': [2, 3, 4], 'data': x_rs.tolist()}
}
print(f'  [24] -> [2,3,-1] = [2,3,4]')

# ============================================
# 12. CONCAT axis=0 (batch concat)
# ============================================
print('=== Concat axis=0 ===')

a_cat = np.array([[1, 2, 3]], dtype=np.float32)
b_cat = np.array([[4, 5, 6], [7, 8, 9]], dtype=np.float32)
result_cat = np.concatenate([a_cat, b_cat], axis=0)
tests['concat_axis0'] = {
    'op': 'Concat', 'description': '[1,3] + [2,3] axis=0 -> [3,3]',
    'input_a': {'shape': [1, 3], 'data': a_cat.flatten().tolist()},
    'input_b': {'shape': [2, 3], 'data': b_cat.flatten().tolist()},
    'attributes': {'axis': 0},
    'expected': {'shape': list(result_cat.shape), 'data': result_cat.flatten().tolist()}
}
print(f'  axis=0: {result_cat.flatten().tolist()}')

# ============================================
# SAVE
# ============================================
out_path = os.path.join(out_dir, 'operator_test_cases.json')
with open(out_path, 'w') as f:
    json.dump(tests, f, indent=2)
print(f'\nSaved {len(tests)} test cases to {out_path}')
print(f'File size: {os.path.getsize(out_path)} bytes')
