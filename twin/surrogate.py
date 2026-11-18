"""Reduced-order (POD) surrogate for the winding temperature field.

Snapshots (one flattened temperature field per parameter set) are compressed with an SVD (proper orthogonal
decomposition); the mode coefficients are regressed on the parameters with a polynomial least-squares fit.
Prediction is a matrix product, so it is fast enough to sit inside a Kalman filter.

Limits (by design, not hidden): the surrogate is only as good as the snapshots. Here the snapshots come from the
ASSUMED placeholder model (ANSYS exports or twin.thermal_fv), so the surrogate reproduces that model, not a real
transformer. Predictions outside the training parameter box are refused.
"""
import itertools
import numpy as np

def _features(Z, degree):
    n, d = Z.shape
    cols = [np.ones(n)]
    for deg in range(1, degree + 1):
        for combo in itertools.combinations_with_replacement(range(d), deg):
            cols.append(np.prod(Z[:, combo], axis=1))
    return np.column_stack(cols)

class PODSurrogate:
    def __init__(self, n_modes=None, energy=1 - 1e-9, degree=3):
        self.n_modes, self.energy, self.degree = n_modes, energy, degree

    def fit(self, params, fields):
        P = np.asarray(params, float); F = np.asarray(fields, float)
        if P.ndim != 2 or F.ndim != 2 or len(P) != len(F):
            raise ValueError("params [n, d] and fields [n, m] must have the same number of rows")
        n_feat = _features(P[:1], self.degree).shape[1]
        if len(P) < n_feat:
            raise ValueError(f"need at least {n_feat} snapshots for degree {self.degree}, got {len(P)}")
        self.lo, self.hi = P.min(axis=0), P.max(axis=0)
        span = np.where(self.hi > self.lo, self.hi - self.lo, 1.0)
        self._span = span
        Z = (P - self.lo) / span * 2 - 1
        self.mean = F.mean(axis=0)
        U, s, Vt = np.linalg.svd((F - self.mean).T, full_matrices=False)
        if self.n_modes is None:
            cum = np.cumsum(s**2) / np.sum(s**2)
            r = int(np.searchsorted(cum, self.energy) + 1)
        else:
            r = self.n_modes
        self.modes = U[:, :r]
        self.singular_values = s
        coef = (F - self.mean) @ self.modes                      # [n, r]
        X = _features(Z, self.degree)
        self.w, *_ = np.linalg.lstsq(X, coef, rcond=None)
        return self

    @property
    def n_modes_used(self):
        return self.modes.shape[1]

    def predict(self, params, allow_extrapolation=False):
        P = np.atleast_2d(np.asarray(params, float))
        if not allow_extrapolation and (np.any(P < self.lo - 1e-12) or np.any(P > self.hi + 1e-12)):
            raise ValueError(f"parameters outside the training box {self.lo} .. {self.hi}")
        Z = (P - self.lo) / self._span * 2 - 1
        return self.mean + (_features(Z, self.degree) @ self.w) @ self.modes.T


class WindingSurrogate:
    """POD surrogate of twin.thermal_fv for inputs (load [pu], film coefficient hf [W/m2K]).

    Inputs are mapped to (load^2, 1/hf) before the polynomial fit: heat generation scales with load^2 (ASSUMED) and
    the film resistance with 1/hf. Without this, a cubic fit in (load, hf) has errors of several kelvin."""

    def __init__(self, degree=3, nz=60, nr_lv=10, nr_hv=14):
        self.pod = PODSurrogate(degree=degree)
        self.nz, self.nr_lv, self.nr_hv = nz, nr_lv, nr_hv

    @staticmethod
    def _z(load, hf):
        load, hf = np.broadcast_arrays(np.asarray(load, float), np.asarray(hf, float))
        return np.column_stack([np.ravel(load) ** 2, 1.0 / np.ravel(hf)])

    def fit_from_model(self, loads, hfs, **model_kw):
        """Generate snapshots with the finite-volume solver on a (loads x hfs) grid and fit."""
        from twin import thermal_fv as tf
        pairs = [(l, h) for l in loads for h in hfs]
        F = []
        for l, h in pairs:
            m = tf.solve_model(l, h, nz=self.nz, nr_lv=self.nr_lv, nr_hv=self.nr_hv, **model_kw)
            F.append(np.concatenate([m["LV"]["T"].ravel(), m["HV"]["T"].ravel()]))
        return self.fit(np.array(pairs), np.array(F))

    def fit(self, load_hf, fields):
        lh = np.asarray(load_hf, float)
        self.pod.fit(self._z(lh[:, 0], lh[:, 1]), fields)
        self.n_lv = self.nz * self.nr_lv
        return self

    def predict_fields(self, load, hf):
        return self.pod.predict(self._z(load, hf))

    def hot_spots(self, load, hf):
        """(LV max, HV max) temperatures for scalar or array inputs."""
        F = self.predict_fields(load, hf)
        return F[:, : self.n_lv].max(axis=1), F[:, self.n_lv:].max(axis=1)

    def lv_hot_spot(self, load, hf):
        return float(self.hot_spots(load, hf)[0][0])
