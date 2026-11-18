"""Plain-Python helpers that feed the Streamlit twin tab and the figure script (no Streamlit/matplotlib here)."""
import numpy as np
import pandas as pd
from twin import luo_chain as lc
from twin.paper_numbers import load, value
from twin.synthetic import make_run
from twin.kalman import EKF

YEARS = [1, 10, 25, 40, 48]
_ROWS = None

def _paper_row(first):
    global _ROWS
    _ROWS = _ROWS or load()
    return {y: value(f"c{n}", _ROWS) for y, n in zip(YEARS, range(first, first + 5))}

# (label, variant, overload hours, Table 2 first row id, status)
CASES = [
    ("I: temperature only", "none", 0, 67, "FITTED (one rate constant on year 1)"),
    ("II: + moisture, C = Eq.13 total moisture", "eq13", 0, 72, "VARIANT with assumed moisture; printed model impossible"),
    ("II: + moisture, C = 0.70 % constant", "const0p70", 0, 72, "VARIANT with assumed moisture; printed model impossible"),
    ("III: + 24 h overload, C = Eq.13", "eq13", 24, 77, "VARIANT with assumed moisture; printed model impossible"),
    ("III: + 24 h overload, C = 0.70 %", "const0p70", 24, 77, "VARIANT with assumed moisture; printed model impossible"),
]

def luo_table2_frame():
    """Paper vs model for Table 2, one row per case and year, with an honest status column."""
    ks = lc.fit_k_scale(_paper_row(67)[1])
    recs = []
    for label, var, ol, first, status in CASES:
        paper = _paper_row(first)
        got = lc.table2_row(YEARS, var, ks, overload_hours=ol)
        for y in YEARS:
            recs.append(dict(case=label, year=y, paper=paper[y], model=round(got[y], 2),
                             diff=round(got[y] - paper[y], 2), status=status))
    return pd.DataFrame(recs)

def calibration_demo(surrogate, truth, hf_true, hours=24 * 14, qx=1e-3, hf0=94.0, seed=3, sigma=1.0):
    """Run the EKF on a SYNTHETIC sensor stream. `truth` generates the hidden hot-spot; `surrogate` is the filter's model."""
    run = make_run(hours, lambda l, h: truth.lv_hot_spot(l, h), hf_true, seed=seed, sigma=sigma)
    h = lambda x, u: surrogate.lv_hot_spot(u, float(np.exp(x[0])))
    f = EKF([np.log(hf0)], [[0.3 ** 2]], [[qx ** 2]], sigma ** 2, h)
    est, pred = [], []
    for z, u in zip(run.measured_hot_spot, run.load):
        f.step(z, u); est.append(float(np.exp(f.x[0]))); pred.append(h(f.x, u))
    est, pred = np.array(est), np.array(pred)
    uncal = np.array([h([np.log(hf0)], u) for u in run.load])
    clean = run.truth["hot_spot"]
    rm = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
    skip = min(100, hours // 4)
    return dict(run=run, hf_est=est, hf_true=run.truth["hf"], pred=pred, uncal=uncal,
                rmse_filter=rm(pred[skip:], clean[skip:]), rmse_uncal=rm(uncal, clean),
                rmse_sensor=rm(run.measured_hot_spot, clean),
                overload_error=float(h(f.x, 1.3) - truth.lv_hot_spot(1.3, float(run.truth["hf"][-1]))),
                synthetic=True)
