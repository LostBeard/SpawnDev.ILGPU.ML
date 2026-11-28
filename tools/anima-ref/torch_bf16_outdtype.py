"""cuBLAS bf16 GEMM at the DiT shapes with bf16 output (ComfyUI's case) vs fp32 output (ours: activations stay fp32), RTX 4070.
How much of the fast path's gap to cuBLAS is the output width. Run with D:\\Projects\\venv-torchcuda\\Scripts\\python.exe."""
import time

import torch

dev = torch.device('cuda')
T = 4096


def best(fn, reps=10):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    b = 1e9
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); torch.cuda.synchronize(); b = min(b, time.perf_counter() - t0)
    return b


for name, K, N in (('qkv', 1536, 4608), ('mlp-up', 1536, 6144), ('mlp-down', 6144, 1536)):
    a = torch.randn(T, K, device=dev, dtype=torch.bfloat16)
    w = torch.randn(N, K, device=dev, dtype=torch.bfloat16)
    a32 = torch.randn(T, K, device=dev, dtype=torch.float32)
    fl = 2 * T * K * N
    parts = []
    t = best(lambda: a @ w.T)
    parts.append(f'bf16 out {t * 1e3:.2f} ms {fl / t / 1e12:.1f}')
    try:
        t = best(lambda: torch.mm(a, w.T, out_dtype=torch.float32))
        parts.append(f'fp32 out {t * 1e3:.2f} ms {fl / t / 1e12:.1f}')
    except Exception as e:  # noqa: BLE001
        parts.append(f'fp32 out n/a ({type(e).__name__})')
    t = best(lambda: (a @ w.T).float())
    parts.append(f'bf16 out + cast {t * 1e3:.2f} ms {fl / t / 1e12:.1f}')
    t = best(lambda: a32.to(torch.bfloat16) @ w.T)
    parts.append(f'fp32 in (cast) {t * 1e3:.2f} ms {fl / t / 1e12:.1f}')
    t = best(lambda: (a32.to(torch.bfloat16) @ w.T).float())
    parts.append(f'fp32 in + fp32 out (casts) {t * 1e3:.2f} ms {fl / t / 1e12:.1f}')
    print(f'{name:8s} ' + ' | '.join(parts) + '  (TFLOPS)')
