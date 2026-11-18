"""Readers for the CSV files written by ansys/luo_thermal.inp and ansys/luo_thermal_batch.inp.

Single snapshot:  node,x_m,y_m,T_C            (Fortran-style numbers such as '  1.,  0.25E+00,...')
Batch snapshots:  run,node,x_m,y_m,T_C        (snapshots.csv) plus runs.csv with one summary row per run.
Axis of symmetry is Y; x is the radial position. Winding membership is decided by x (ASSUMED geometry below).
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class Geometry:
    """ASSUMED placeholder geometry (m), the same as in the APDL scripts. Not from Luo et al."""
    lv: tuple = (0.250, 0.290)
    hv: tuple = (0.340, 0.400)
    height: float = 1.2

def _clean(df, required):
    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns {missing}; found {list(df.columns)}")
    df = df[required].apply(pd.to_numeric, errors="raise")
    if df.isna().any().any():
        raise ValueError("file contains empty or non-numeric values")
    if not np.isfinite(df.to_numpy()).all():
        raise ValueError("file contains inf or nan")
    return df

def read_snapshot(path):
    """One load case: DataFrame with node, x_m, y_m, T_C."""
    df = _clean(pd.read_csv(path, skipinitialspace=True), ["node", "x_m", "y_m", "T_C"])
    if df.empty:
        raise ValueError("no rows")
    if df.node.duplicated().any():
        raise ValueError("duplicate node numbers")
    if (df.T_C < -50).any() or (df.T_C > 1000).any():
        raise ValueError("temperatures outside -50..1000 C: wrong units?")
    return df

def read_snapshots(path):
    """Batch file: DataFrame with run, node, x_m, y_m, T_C."""
    df = _clean(pd.read_csv(path, skipinitialspace=True), ["run", "node", "x_m", "y_m", "T_C"])
    df["run"] = df["run"].astype(int)
    return df

def read_runs(path):
    cols = ["run", "load_pu", "HF", "TTOP", "Q_W_m3", "LV_max_C", "LV_max_y_m", "HV_max_C", "HV_max_y_m", "T_min_C"]
    df = _clean(pd.read_csv(path, skipinitialspace=True), cols)
    df["run"] = df["run"].astype(int)
    return df

def winding(df, geom=Geometry(), tol=1e-6):
    """Label nodes 'LV' / 'HV' by radial position."""
    lab = np.full(len(df), "other", dtype=object)
    lab[(df.x_m >= geom.lv[0] - tol) & (df.x_m <= geom.lv[1] + tol)] = "LV"
    lab[(df.x_m >= geom.hv[0] - tol) & (df.x_m <= geom.hv[1] + tol)] = "HV"
    out = df.copy(); out["winding"] = lab
    if (lab == "other").any():
        raise ValueError("some nodes are outside both windings: geometry does not match the file")
    return out

def winding_stats(df, geom=Geometry()):
    """Maximum, minimum and mean temperature and the height of the maximum, per winding."""
    d = winding(df, geom)
    out = {}
    for w, g in d.groupby("winding"):
        i = g.T_C.idxmax()
        out[w] = dict(T_max=float(g.T_C.max()), y_at_max=float(g.y_m[i]), x_at_max=float(g.x_m[i]),
                      y_rel_at_max=float(g.y_m[i] / geom.height), T_min=float(g.T_C.min()), T_mean=float(g.T_C.mean()))
    return out

def axial_profile(df, which, geom=Geometry()):
    """Hottest temperature across the winding thickness at each height: (y, T) arrays sorted by y."""
    d = winding(df, geom); d = d[d.winding == which]
    g = d.groupby(d.y_m.round(6)).T_C.max()
    return g.index.to_numpy(float), g.to_numpy(float)
