"""Load Lore's raw tap / dump files (little-endian f32 / i64, shape in the file name)."""
import os
import re
import glob
import numpy as np

TAPS = r'D:\users\tj\Projects\_DevComms\anima-run3-taps-2026-11-27'
DUMPS = r'D:\users\tj\Projects\_DevComms\anima-run1-dumps-2026-11-27'


def load_raw(path):
    m = re.search(r'_(f32|i64)_([0-9x]+)\.raw$', os.path.basename(path))
    if not m:
        raise ValueError(path)
    dt = np.float32 if m.group(1) == 'f32' else np.int64
    shape = tuple(int(s) for s in m.group(2).split('x'))
    a = np.fromfile(path, dtype=dt)
    assert a.size == int(np.prod(shape)), (path, a.size, shape)
    a = a.reshape(shape)
    return a.astype(np.float64) if dt == np.float32 else a


def find(folder, prefix):
    """The single file in folder whose name starts with prefix + '_f32_' / '_i64_'."""
    hits = [p for p in glob.glob(os.path.join(folder, prefix + '_*.raw'))
            if re.match(re.escape(prefix) + r'_(f32|i64)_[0-9x]+\.raw$', os.path.basename(p))]
    if len(hits) != 1:
        raise FileNotFoundError(f'{prefix}: {hits}')
    return load_raw(hits[0])


def tap(name):
    """tap('x_embedder_0_out0') -> tap_diffusion_model_x_embedder_0_out0_f32_....raw"""
    return find(TAPS, 'tap_diffusion_model_' + name)


def dump(name, folder=DUMPS):
    return find(folder, name)
