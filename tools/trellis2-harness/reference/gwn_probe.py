"""How decisive is the generalized winding number as an inside/outside signal on a generated mesh?

usage: python gwn_probe.py mesh.obj|mesh.glb [samples=3000] [eps_rel=0.004]

For sampled faces, evaluates the winding number W(p) = sum of signed solid angles / 4 pi at p = centroid +- eps * n (n from the current winding).
For a closed mesh W is 1 inside and 0 outside. A face wound outward has W(c + eps n) < W(c - eps n); the difference d = W(c + eps n) - W(c - eps n)
is therefore negative for a correct face and positive for a flipped one (|W| is used: a closed mesh wound inside-out has W = -1 inside). Prints how many samples are decisive (|d| > 0.5), and of those how many
say "flipped", so it can be compared with mesh_winding_render.py's red share.
"""
import sys
import numpy as np
import mesh_audit as m


def winding(points, tri):
    """W at each point (P,3) from triangles tri (F,3,3), by chunks of faces (Van Oosterom - Strackee)."""
    w = np.zeros(len(points))
    for lo in range(0, len(tri), 20000):
        t = tri[lo:lo + 20000]
        a = t[None, :, 0, :] - points[:, None, :]
        b = t[None, :, 1, :] - points[:, None, :]
        c = t[None, :, 2, :] - points[:, None, :]
        la, lb, lc = np.linalg.norm(a, axis=2), np.linalg.norm(b, axis=2), np.linalg.norm(c, axis=2)
        num = np.einsum('pfk,pfk->pf', a, np.cross(b, c))
        den = la * lb * lc + np.einsum('pfk,pfk->pf', a, b) * lc + np.einsum('pfk,pfk->pf', b, c) * la + np.einsum('pfk,pfk->pf', c, a) * lb
        w += 2 * np.arctan2(num, den).sum(1)
    return w / (4 * np.pi)


if __name__ == '__main__':
    pos, idx = m.load_glb(sys.argv[1])
    v, f = m.weld(pos, idx)
    n_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
    eps = (float(sys.argv[3]) if len(sys.argv) > 3 else 0.004) * float(np.linalg.norm(v.max(0) - v.min(0)))
    tri = v[f]
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]); area = np.linalg.norm(nrm, axis=1)
    ok = area > 1e-12; nrm[ok] /= area[ok, None]
    cen = tri.mean(1)
    rng = np.random.default_rng(1)
    pick = rng.choice(np.where(ok)[0], size=min(n_samples, int(ok.sum())), replace=False)
    wp = winding(cen[pick] + eps * nrm[pick], tri)
    wm = winding(cen[pick] - eps * nrm[pick], tri)
    wp, wm = np.abs(wp), np.abs(wm)   # |W|: ~1 inside whichever way the mesh is wound (an inverted closed mesh has W = -1 inside)
    d = wp - wm
    dec = np.abs(d) > 0.5
    print(f'{sys.argv[1]}: {len(f)} faces, {len(pick)} sampled, eps = {eps:.5f}')
    print(f'  decisive (|W+ - W-| > 0.5): {100 * dec.mean():.1f}%;  of those wound inward (d > 0): {100 * (d[dec] > 0).mean():.1f}%, outward (d < 0): {100 * (d[dec] < 0).mean():.1f}%')
    print(f'  W at the + side: mean {wp.mean():.2f}, - side: {wm.mean():.2f};  both sides ~0 (|W|<0.1: free space / thin sheet): {100 * ((np.abs(wp) < .1) & (np.abs(wm) < .1)).mean():.1f}%')
    for lo, hi in [(0, .1), (.1, .3), (.3, .5), (.5, .8), (.8, 3)]:
        s = (np.abs(d) >= lo) & (np.abs(d) < hi)
        print(f'    |d| in [{lo}, {hi}): {100 * s.mean():5.1f}% of samples, {100 * (d[s] > 0).mean() if s.any() else 0:5.1f}% of them wound inward')
