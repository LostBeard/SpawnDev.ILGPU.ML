#!/usr/bin/env python3
"""Generate reference outputs for ILGPU.ML unit tests.

Runs ONNX models through onnxruntime (CPU) and saves the output tensors
as raw float32 binary files. These serve as ground truth for comparing
against ILGPU.ML's GPU inference pipeline.

Usage:
    python gen_references.py

Requires: pip install onnxruntime numpy pillow
"""

import os
import sys
import struct
import numpy as np
import onnxruntime as ort

# Paths relative to this script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, '..', 'SpawnDev.ILGPU.ML')
WWWROOT = os.path.join(PROJECT_ROOT, 'SpawnDev.ILGPU.ML.Demo', 'wwwroot')
MODELS_DIR = os.path.join(WWWROOT, 'models')
SAMPLES_DIR = os.path.join(WWWROOT, 'samples')
REFERENCES_DIR = os.path.join(WWWROOT, 'references')


def load_cat_rgba_bin(path, target_size=(224, 224)):
    """Load cat_rgba.bin (int32 width, int32 height, uint32[] RGBA pixels).
    Returns float32 NCHW [1, 3, H, W] in [0, 255] range."""
    with open(path, 'rb') as f:
        width = struct.unpack('<i', f.read(4))[0]
        height = struct.unpack('<i', f.read(4))[0]
        pixel_bytes = f.read(width * height * 4)

    # Unpack RGBA pixels
    pixels = np.frombuffer(pixel_bytes, dtype=np.uint8).reshape(height, width, 4)
    rgb = pixels[:, :, :3].astype(np.float32)  # Drop alpha, keep RGB

    # Resize to target size using bilinear interpolation
    if (height, width) != target_size:
        from PIL import Image
        img = Image.fromarray(pixels[:, :, :3])
        img = img.resize((target_size[1], target_size[0]), Image.BILINEAR)
        rgb = np.array(img, dtype=np.float32)

    # HWC -> NCHW
    nchw = np.transpose(rgb, (2, 0, 1))[np.newaxis, :, :, :]  # [1, 3, H, W]
    return nchw


def generate_style_reference(style_name, input_nchw):
    """Run a style transfer model and save the reference output."""
    model_path = os.path.join(MODELS_DIR, style_name, 'model.onnx')
    if not os.path.exists(model_path):
        print(f'  SKIP {style_name} — model.onnx not found')
        return

    # Suppress ORT warnings
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    sess = ort.InferenceSession(model_path, opts)

    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    # Run inference
    output = sess.run([output_name], {input_name: input_nchw})[0]

    # Save reference
    ref_dir = os.path.join(REFERENCES_DIR, style_name)
    os.makedirs(ref_dir, exist_ok=True)

    # Save output tensor as raw float32
    output_path = os.path.join(ref_dir, 'cat_output_nchw.bin')
    output.astype(np.float32).tofile(output_path)

    # Save input tensor too (so tests can verify preprocessing matches)
    input_path = os.path.join(ref_dir, 'cat_input_nchw.bin')
    input_nchw.astype(np.float32).tofile(input_path)

    # Save metadata
    meta_path = os.path.join(ref_dir, 'metadata.txt')
    with open(meta_path, 'w') as f:
        f.write(f'model: {style_name}\n')
        f.write(f'input_name: {input_name}\n')
        f.write(f'output_name: {output_name}\n')
        f.write(f'input_shape: {list(input_nchw.shape)}\n')
        f.write(f'output_shape: {list(output.shape)}\n')
        f.write(f'input_range: [{input_nchw.min():.2f}, {input_nchw.max():.2f}]\n')
        f.write(f'output_range: [{output.min():.2f}, {output.max():.2f}]\n')
        f.write(f'output_mean: {output.mean():.4f}\n')
        f.write(f'output_std: {output.std():.4f}\n')
        f.write(f'onnxruntime_version: {ort.__version__}\n')
        f.write(f'numpy_version: {np.__version__}\n')

    size_kb = os.path.getsize(output_path) / 1024
    print(f'  {style_name}: output {list(output.shape)}, '
          f'range [{output.min():.1f}, {output.max():.1f}], '
          f'mean={output.mean():.1f}, {size_kb:.0f} KB')


def main():
    print('ILGPU.ML Reference Generator')
    print('=' * 40)

    # Load cat image
    cat_path = os.path.join(SAMPLES_DIR, 'cat_rgba.bin')
    if not os.path.exists(cat_path):
        print(f'ERROR: {cat_path} not found')
        sys.exit(1)

    print(f'Loading cat image from {cat_path}')
    input_nchw = load_cat_rgba_bin(cat_path, target_size=(224, 224))
    print(f'Input: shape={list(input_nchw.shape)}, range=[{input_nchw.min():.0f}, {input_nchw.max():.0f}]')
    print()

    # Generate references for all style models
    styles = ['style-mosaic', 'style-candy', 'style-rain-princess',
              'style-pointilism', 'style-udnie']

    print('Generating style transfer references:')
    for style in styles:
        generate_style_reference(style, input_nchw)

    print()
    print(f'References saved to: {REFERENCES_DIR}')
    print('Done.')


if __name__ == '__main__':
    main()
