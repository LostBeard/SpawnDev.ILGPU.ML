"""Reference for ClapProbe_FitMatchesNumpyReference (2026-11-03): AssistZ Lab/probe_eval.py's fit_logistic (class-balanced, full batch, l2 / N) on a
small deterministic dataset (an LCG, so C# could regenerate it; printed as literals anyway). Usage: py -3.13 tools/clap-reference/probe_reference.py"""
import numpy as np

state = 7
def u():
    global state
    state = (state * 1103515245 + 12345) & 0x7FFFFFFF
    return state / 1073741824.0 - 1.0

n, d = 24, 8
X = np.array([[u() for _ in range(d)] for _ in range(n)], dtype=np.float64)
y = np.array([1 if (i % 3 == 0) else 0 for i in range(n)], dtype=float)   # 8 positives, 16 negatives: unbalanced on purpose
X[y == 1, 0] += 0.8                                                        # separable-ish along axis 0
X = (X / np.linalg.norm(X, axis=1, keepdims=True)).astype(np.float32).astype(np.float64)   # unit rows, float32-exact like embeddings

w = np.where(y == 1, 0.5 / y.sum(), 0.5 / (1 - y).sum())
a, b = np.zeros(d), 0.0
for _ in range(800):
    p = 1 / (1 + np.exp(-(X @ a + b)))
    a -= 0.5 * (X.T @ (w * (p - y)) + 1.0 / len(y) * a)
    b -= 0.5 * np.sum(w * (p - y))
fmt = lambda v: ", ".join(f"{float(x):.9g}f" for x in v)
print("X = {")
for row in X: print("    new float[] { " + fmt(row) + " },")
print("};")
print("y = { " + ", ".join(str(int(v)) for v in y) + " };")
print("a = { " + fmt(a) + " };")
print(f"b = {b:.9g}f;")
