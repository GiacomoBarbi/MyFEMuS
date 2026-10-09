"""T13 shared helpers: JSON fields on the [z, y, x] node grid, exact Q2 prolongation, element-Simpson
weights, and the SURFACE norm over a union of faces (sum_f int_{Gamma_f}, face slices of the grid)."""
import json
import numpy as np

SLICE = {"zmin": (slice(0, 1), slice(None), slice(None)), "zmax": (slice(-1, None), slice(None), slice(None)),
         "ymin": (slice(None), slice(0, 1), slice(None)), "ymax": (slice(None), slice(-1, None), slice(None)),
         "xmin": (slice(None), slice(None), slice(0, 1)), "xmax": (slice(None), slice(None), slice(-1, None))}


def load(path):
    d = json.load(open(path))
    xyz = np.round(np.array(d["spatial_coordinates"]), 12)
    f = {k: grid(xyz, np.array([s["values"] for s in d[k]])) for k in ("state", "adjoint", "control_full")}
    return d, f


def grid(xyz, field):
    xs = np.unique(xyz[:, 0]); ys = np.unique(xyz[:, 1]); zs = np.unique(xyz[:, 2])
    ix = np.searchsorted(xs, xyz[:, 0]); iy = np.searchsorted(ys, xyz[:, 1]); iz = np.searchsorted(zs, xyz[:, 2])
    out = np.zeros(field.shape[:-1] + (len(zs), len(ys), len(xs)))
    out[..., iz, iy, ix] = field
    return out


def prolong_1d(nc, r):
    nf = 2 * nc * r + 1
    P = np.zeros((nf, 2 * nc + 1))
    for j in range(nf):
        s = j / (2 * r); e = min(int(s), nc - 1); xi = s - e
        l = [2 * (xi - 0.5) * (xi - 1), -4 * xi * (xi - 1), 2 * xi * (xi - 0.5)]
        for k in range(3):
            P[j, 2 * e + k] += l[k]
    return P


def simpson_1d(n):
    w = np.zeros(2 * n + 1)
    for e in range(n):
        w[2 * e:2 * e + 3] += np.array([1, 4, 1]) / (6.0 * n)
    return w


def vol_sq(n, a):
    """int_Omega a^2 (unit cube, element Simpson) for a [z, y, x] grid array"""
    w = simpson_1d(n)
    return float(np.einsum("i,j,k,ijk->", w, w, w, a * a))


def surf_sq(n, a, faces):
    """sum_{f in faces} int_{Gamma_f} a^2 (face Simpson weights on the face slice)"""
    w = simpson_1d(n)
    W2 = np.einsum("i,j->ij", w, w)
    return float(sum(np.sum(W2 * np.squeeze(a[SLICE[f]], axis={"z": 0, "y": 1, "x": 2}[f[0]]) ** 2) for f in faces))
