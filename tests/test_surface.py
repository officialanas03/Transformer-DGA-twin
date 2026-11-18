import numpy as np
import pytest
from pathlib import Path
from twin import surface as sf
from twin import ansys_io as ai

EX = Path(__file__).resolve().parent.parent / "examples" / "ansys"

def _empty_circle(P, T):
    cc, r2 = sf._circumcircles(P, T)
    for k in range(len(T)):
        d2 = ((P - cc[k]) ** 2).sum(1)
        d2[T[k]] = np.inf
        if (d2 < r2[k] * (1 - 1e-9)).any():
            return False
    return True

def test_random_points_are_delaunay_and_match_matplotlib():
    import matplotlib.tri as mtri
    rng = np.random.default_rng(0)
    P = rng.random((150, 2))
    T = sf.delaunay(P)
    assert _empty_circle(P, T)
    ref = mtri.Triangulation(P[:, 0], P[:, 1]).triangles
    key = lambda T_: {tuple(sorted(t)) for t in T_}
    assert key(T) == key(ref)

def test_triangulation_covers_the_convex_hull():
    rng = np.random.default_rng(1)
    P = rng.random((80, 2))
    T = sf.delaunay(P)
    a, b, c = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
    area = 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])).sum()
    from math import isclose
    hull = _hull_area(P)
    assert isclose(area, hull, rel_tol=1e-6)

def _hull_area(P):
    pts = sorted(map(tuple, P))
    def half(seq):
        h = []
        for p in seq:
            while len(h) >= 2 and (h[-1][0]-h[-2][0])*(p[1]-h[-2][1]) - (h[-1][1]-h[-2][1])*(p[0]-h[-2][0]) <= 0:
                h.pop()
            h.append(p)
        return h
    hull = half(pts)[:-1] + half(pts[::-1])[:-1]
    x, y = np.array(hull).T
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

def test_input_checks():
    with pytest.raises(ValueError):
        sf.delaunay(np.array([[0, 0], [1, 1]]))
    with pytest.raises(ValueError):
        sf.delaunay(np.array([[0, 0], [1, 1], [0, 0]]))

def test_real_mesh_triangulation_and_gap_masking():
    df = ai.read_snapshot(EX / "T_load0p74.csv")
    P = df[["x_m", "y_m"]].to_numpy()
    T = sf.delaunay(P)
    n = len(P)
    assert len(T) > n                                    # roughly 2n triangles for a planar point set
    kept = sf.mask_long_edges(P, T, 0.02)                # mesh spacing is 0.01 m, the gap between windings is 0.05 m
    side = np.where(P[:, 0] < 0.3, 0, 1)
    assert (side[kept[:, 0]] == side[kept[:, 1]]).all() and (side[kept[:, 1]] == side[kept[:, 2]]).all()
    assert len(kept) < len(T)

def test_resample_reproduces_a_linear_field_and_leaves_gaps_empty():
    df = ai.read_snapshot(EX / "T_load0p74.csv")
    P = df[["x_m", "y_m"]].to_numpy()
    T = sf.mask_long_edges(P, sf.delaunay(P), 0.02)
    z = 3.0 * P[:, 0] + 5.0 * P[:, 1] + 2.0
    gx, gy, Z = sf.resample(P[:, 0], P[:, 1], z, T, 0.005)
    GX, GY = np.meshgrid(gx, gy)
    ok = ~np.isnan(Z)
    assert np.allclose(Z[ok], 3 * GX[ok] + 5 * GY[ok] + 2, atol=1e-9)
    gap = (GX > 0.295) & (GX < 0.335)
    assert np.isnan(Z[gap]).all() and ok.sum() > 0
