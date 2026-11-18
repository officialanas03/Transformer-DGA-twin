"""Surface rebuild of discrete simulation data by incremental-insertion Delaunay triangulation (Luo et al. 2023, Sec. 3.4).

Follows the paper's description: three auxiliary "super triangle" points cover the point set, points are inserted one
by one, the InCircle test finds the triangles that violate the Delaunay property (the re-triangulation of that cavity
is equivalent to the diagonal exchanges the paper mentions), and finally every triangle touching an auxiliary point is
deleted. The paper's step 1 (choosing a row/column spacing of "20-100" points) is not needed for this: it only sets the
resolution of the plotted grid, see `resample`.

This module only changes how a field is DRAWN; it does not change any numbers. It adds the one thing a triangulation
cannot know: triangles that bridge a gap (here, between the LV and HV windings) must be removed with `mask_long_edges`.
"""
import numpy as np

def _circumcircles(P, T):
    a, b, c = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
    d = 2.0 * (a[:, 0] * (b[:, 1] - c[:, 1]) + b[:, 0] * (c[:, 1] - a[:, 1]) + c[:, 0] * (a[:, 1] - b[:, 1]))
    d = np.where(np.abs(d) < 1e-300, 1e-300, d)
    a2, b2, c2 = (a**2).sum(1), (b**2).sum(1), (c**2).sum(1)
    ux = (a2 * (b[:, 1] - c[:, 1]) + b2 * (c[:, 1] - a[:, 1]) + c2 * (a[:, 1] - b[:, 1])) / d
    uy = (a2 * (c[:, 0] - b[:, 0]) + b2 * (a[:, 0] - c[:, 0]) + c2 * (b[:, 0] - a[:, 0])) / d
    cc = np.column_stack([ux, uy])
    return cc, ((a - cc) ** 2).sum(1)

def delaunay(points, super_scale=1e4):
    """Triangles (m, 3) of point indices. Duplicate points are rejected."""
    pts = np.asarray(points, float)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3:
        raise ValueError("need at least 3 points of shape (n, 2)")
    if len(np.unique(pts, axis=0)) != len(pts):
        raise ValueError("duplicate points")
    centre = pts.mean(axis=0)
    scale = float(np.abs(pts - centre).max()) or 1.0
    Q = (pts - centre) / scale                                   # uniform scaling keeps the Delaunay property
    n = len(Q)
    M = super_scale
    P = np.vstack([Q, [[-M, -M], [M, -M], [0.0, M]]])             # the three auxiliary points
    T = np.array([[n, n + 1, n + 2]])
    cc, r2 = _circumcircles(P, T)
    for i in range(n):
        p = P[i]
        bad = ((cc - p) ** 2).sum(1) < r2 * (1.0 - 1e-12)         # InCircle test (strict: cocircular points stay)
        if not bad.any():
            raise RuntimeError("point not inside any circumcircle (numerical problem)")
        tb = T[bad]
        edges = np.sort(np.concatenate([tb[:, [0, 1]], tb[:, [1, 2]], tb[:, [2, 0]]]), axis=1)
        uniq, cnt = np.unique(edges, axis=0, return_counts=True)
        boundary = uniq[cnt == 1]                                 # edges of the cavity
        new = np.column_stack([boundary, np.full(len(boundary), i)])
        ncc, nr2 = _circumcircles(P, new)
        T = np.vstack([T[~bad], new]); cc = np.vstack([cc[~bad], ncc]); r2 = np.concatenate([r2[~bad], nr2])
    return T[(T < n).all(axis=1)]                                  # drop triangles that touch an auxiliary point

def mask_long_edges(points, triangles, max_edge):
    """Keep triangles whose longest edge is <= max_edge (removes triangles that bridge gaps)."""
    P = np.asarray(points, float)
    a, b, c = P[triangles[:, 0]], P[triangles[:, 1]], P[triangles[:, 2]]
    longest = np.maximum.reduce([np.linalg.norm(a - b, axis=1), np.linalg.norm(b - c, axis=1), np.linalg.norm(c - a, axis=1)])
    return triangles[longest <= max_edge]

def resample(x, y, z, triangles, step):
    """Linear interpolation of z over the triangulation onto a regular grid of spacing `step`. NaN outside the triangles."""
    import matplotlib.tri as mtri
    tri = mtri.Triangulation(x, y, triangles=triangles)
    interp = mtri.LinearTriInterpolator(tri, z)
    gx = np.arange(np.min(x), np.max(x) + step / 2, step)
    gy = np.arange(np.min(y), np.max(y) + step / 2, step)
    GX, GY = np.meshgrid(gx, gy)
    return gx, gy, np.ma.filled(interp(GX, GY), np.nan)
