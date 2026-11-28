"""Minimal safetensors reader (no safetensors package needed).

Reads the JSON header once, then memory-maps the file and converts single tensors on demand
(bf16 / f16 / f32 -> float64). Nothing is loaded until asked for, so a 4 GB file costs only
the tensors actually used.
"""
import json
import struct
import numpy as np


class SafeTensors:
    def __init__(self, path):
        self.path = path
        with open(path, 'rb') as f:
            n = struct.unpack('<Q', f.read(8))[0]
            self.header = json.loads(f.read(n).decode('utf-8'))
        self.data_start = 8 + n
        self.meta = self.header.pop('__metadata__', None)
        self.mm = np.memmap(path, dtype=np.uint8, mode='r')

    def names(self):
        return list(self.header.keys())

    def shape(self, name):
        return tuple(self.header[name]['shape'])

    def dtype(self, name):
        return self.header[name]['dtype']

    def get(self, name, dtype=np.float64):
        h = self.header[name]
        a, b = h['data_offsets']
        raw = self.mm[self.data_start + a:self.data_start + b]
        dt = h['dtype']
        if dt == 'BF16':
            u16 = np.frombuffer(raw.tobytes(), dtype=np.uint16)
            out = (u16.astype(np.uint32) << 16).view(np.float32)
        elif dt == 'F16':
            out = np.frombuffer(raw.tobytes(), dtype=np.float16)
        elif dt == 'F32':
            out = np.frombuffer(raw.tobytes(), dtype=np.float32)
        elif dt == 'F64':
            out = np.frombuffer(raw.tobytes(), dtype=np.float64)
        elif dt == 'I64':
            out = np.frombuffer(raw.tobytes(), dtype=np.int64)
        else:
            raise ValueError(f'dtype {dt} not handled')
        return out.reshape(h['shape']).astype(dtype)
