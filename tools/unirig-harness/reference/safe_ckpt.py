"""Load a UniRig / PyTorch-Lightning checkpoint without letting the pickle run anything.

A .ckpt is a zip around a pickle. `torch.load(weights_only=True)` refuses the skin checkpoint because it also pickles a python-box `Box` (the training
config, a dict subclass), and `weights_only=False` would execute whatever the file says. This loader uses an unpickler whose `find_class` allows ONLY the
names the two released UniRig checkpoints import (listed with pickletools, nothing executed - see list_pickle_globals in the notes):

  skeleton: collections.OrderedDict, torch.FloatStorage, torch._utils._rebuild_tensor_v2
  skin:     the same + torch.LongStorage, torch.Tensor, torch._tensor._rebuild_from_type_v2, box.box.Box

Anything else raises. Use `python -I` and keep the checkpoint directory out of the interpreter's path.
"""
import pickle
import torch

ALLOWED = {
    ('collections', 'OrderedDict'),
    ('torch', 'FloatStorage'), ('torch', 'LongStorage'), ('torch', 'HalfStorage'), ('torch', 'BFloat16Storage'),
    ('torch._utils', '_rebuild_tensor_v2'),
    ('torch._tensor', '_rebuild_from_type_v2'),
    ('torch', 'Tensor'),
    ('box.box', 'Box'),
}


class _SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) not in ALLOWED:
            raise pickle.UnpicklingError(f'checkpoint imports {module}.{name}, which is not on the allowlist')
        return super().find_class(module, name)


class _PickleModule:           # torch.load wants a module exposing Unpickler / load
    Unpickler = _SafeUnpickler
    __name__ = 'safe_unirig_pickle'
    HIGHEST_PROTOCOL = pickle.HIGHEST_PROTOCOL
    load = staticmethod(lambda f, **kw: _SafeUnpickler(f, **kw).load())


def load_checkpoint(path):
    """The checkpoint dict (tensors on CPU)."""
    import box.box  # noqa: F401  (the skin checkpoint's Box must be importable)
    return torch.load(path, map_location='cpu', weights_only=False, pickle_module=_PickleModule)


if __name__ == '__main__':
    import sys, collections
    ck = load_checkpoint(sys.argv[1])
    print('top-level keys:', list(ck.keys()))
    for k, v in ck.items():
        if k != 'state_dict':
            print(f'  {k}: {repr(v)[:600]}')
    sd = ck['state_dict']
    print('tensors:', len(sd), ' params:', f"{sum(t.numel() for t in sd.values()) / 1e6:.1f}M", ' dtypes:', dict(collections.Counter(str(t.dtype) for t in sd.values())))
    groups = collections.OrderedDict()
    for name, t in sd.items():
        g = '.'.join(name.split('.')[:int(sys.argv[2]) if len(sys.argv) > 2 else 3])
        n, c = groups.get(g, (0, 0)); groups[g] = (n + t.numel(), c + 1)
    for g, (n, c) in groups.items():
        print(f'  {g:64s} {n / 1e6:9.2f}M  {c:4d}')
