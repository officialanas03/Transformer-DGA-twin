import numpy as np
import pytest
from pathlib import Path
from twin import thermal_fv as tf
from twin import ansys_io as ai
from twin.surrogate import WindingSurrogate, PODSurrogate
from twin.synthetic import make_run, load_profile
from twin.kalman import EKF

EX = Path(__file__).resolve().parent.parent / "examples" / "ansys"
LOADS, HFS = np.linspace(0.3, 1.4, 9), np.linspace(50, 110, 7)

@pytest.fixture(scope="module")
def surr():
    return WindingSurrogate(nz=60).fit_from_model(LOADS, HFS)

@pytest.fixture(scope="module")
def truth_fine():          # finer mesh than the filter's surrogate, so the filter does not see its own model
    return WindingSurrogate(nz=80, nr_lv=12, nr_hv=16).fit_from_model(LOADS, HFS)

@pytest.fixture(scope="module")
def truth_wrong_oil():     # MODEL MISMATCH: top-oil 47 C instead of the assumed 40 C
    return WindingSurrogate(nz=60).fit_from_model(LOADS, HFS, t_top=47.0)

# ---------- finite-volume solver vs the real ANSYS exports ----------
@pytest.mark.parametrize("load,fname", [(0.74, "T_load0p74.csv"), (1.3, "T_load1p30.csv")])
def test_fv_solver_agrees_with_ansys(load, fname):
    ans = ai.winding_stats(ai.read_snapshot(EX / fname))
    m = tf.solve_model(load, 72.0)
    for w in ("LV", "HV"):
        assert m[w]["T_max"] == pytest.approx(ans[w]["T_max"], abs=0.1)
        assert m[w]["y_at_max"] == pytest.approx(ans[w]["y_at_max"], abs=0.02)

def test_fv_mesh_convergence_and_trends():
    a = tf.solve_model(0.74, 72.0, nz=60)["LV"]["T_max"]
    b = tf.solve_model(0.74, 72.0, nz=120)["LV"]["T_max"]
    assert abs(a - b) < 0.1
    assert tf.solve_model(0.74, 100.0, nz=60)["LV"]["T_max"] < a                      # more cooling
    assert tf.solve_model(1.0, 72.0, nz=60)["LV"]["T_max"] > a                        # more load
    assert tf.hot_spot(0.74, 72.0, nz=60)[1] == "HV"                                  # the assumed model puts the hot spot in HV

# ---------- surrogate ----------
def test_surrogate_reproduces_training_and_holdout(surr):
    rng = np.random.default_rng(1)
    q = np.column_stack([rng.uniform(0.3, 1.4, 15), rng.uniform(50, 110, 15)])
    for l, h in q:
        m = tf.solve_model(l, h, nz=60)
        lv, hv = surr.hot_spots(l, h)
        assert abs(lv[0] - m["LV"]["T_max"]) < 0.1 and abs(hv[0] - m["HV"]["T_max"]) < 0.1
    assert surr.pod.n_modes_used <= 6

def test_surrogate_matches_the_ansys_hot_spot(surr):
    assert surr.lv_hot_spot(0.74, 72.0) == pytest.approx(61.38, abs=0.1)

def test_surrogate_refuses_extrapolation_and_too_few_snapshots(surr):
    with pytest.raises(ValueError):
        surr.lv_hot_spot(2.0, 72.0)
    with pytest.raises(ValueError):
        PODSurrogate(degree=3).fit(np.random.rand(3, 2), np.random.rand(3, 10))
    with pytest.raises(ValueError):
        PODSurrogate().fit(np.random.rand(30, 2), np.random.rand(29, 10))

# ---------- synthetic data are labelled ----------
def test_synthetic_runs_are_labelled_and_reproducible():
    k = load_profile(100, seed=2)
    assert np.array_equal(k, load_profile(100, seed=2)) and k.min() >= 0.35 and k.max() <= 1.3
    r = make_run(50, lambda l, h: 50 + 20 * l, 70.0, seed=2)
    assert r.synthetic is True and "SYNTHETIC" in r.note and "hot_spot" in r.truth

# ---------- Kalman filter ----------
def _run_filter(surr, truth, hf_true, qx=1e-3, hf0=94.0, hours=24 * 14, seed=3):
    r = make_run(hours, lambda l, h: truth.lv_hot_spot(l, h), hf_true, seed=seed, sigma=1.0)
    h = lambda x, u: surr.lv_hot_spot(u, float(np.exp(x[0])))
    f = EKF([np.log(hf0)], [[0.3**2]], [[qx**2]], 1.0, h)
    est, pred = [], []
    for z, u in zip(r.measured_hot_spot, r.load):
        f.step(z, u); est.append(float(np.exp(f.x[0]))); pred.append(h(f.x, u))
    return r, f, np.array(est), np.array(pred), h

def test_filter_recovers_constant_parameter(surr, truth_fine):
    r, f, est, pred, h = _run_filter(surr, truth_fine, 72.0)
    assert abs(est[-1] - 72.0) / 72.0 < 0.03
    rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
    assert rmse(pred[100:], r.truth["hot_spot"][100:]) < 0.5 < rmse(r.measured_hot_spot, r.truth["hot_spot"])   # beats the raw sensor noise
    uncal = np.array([h([np.log(94.0)], u) for u in r.load])
    assert rmse(uncal, r.truth["hot_spot"]) > 3.0                                                                # wrong start was bad

def test_filter_tracks_drift(surr, truth_fine):
    hours = 24 * 14
    hf = np.linspace(72, 58, hours)
    r, f, est, pred, h = _run_filter(surr, truth_fine, hf, qx=2e-3, hours=hours)
    assert abs(est[-1] - 58) / 58 < 0.05
    assert f.P[0, 0] < 0.3**2                                                                                     # uncertainty shrank

def test_filter_skips_missing_measurements(surr):
    h = lambda x, u: surr.lv_hot_spot(u, float(np.exp(x[0])))
    f = EKF([np.log(80.0)], [[0.1]], [[1e-6]], 1.0, h)
    x0 = f.x.copy()
    f.step(np.nan, 0.8); f.step(None, 0.8)
    assert np.allclose(f.x, x0)

def test_ekf_rejects_bad_noise():
    with pytest.raises(ValueError):
        EKF([0.0], [[1.0]], [[0.0]], 0.0, lambda x, u: 0.0)

def test_model_mismatch_gives_a_good_fit_but_wrong_parameter(surr, truth_wrong_oil):
    """UNFLATTERING BY DESIGN: if the oil model is wrong the filter hides it in the film coefficient."""
    r, f, est, pred, h = _run_filter(surr, truth_wrong_oil, 72.0)
    assert abs(est[-1] - 72.0) / 72.0 > 0.10                         # parameter clearly wrong
    rmse = float(np.sqrt(np.mean((pred[100:] - r.truth["hot_spot"][100:]) ** 2)))
    assert rmse < 3.0                                                # yet it tracks the sensor reasonably
    overload_error = h(f.x, 1.3) - truth_wrong_oil.lv_hot_spot(1.3, 72.0)
    assert abs(overload_error) > 3.0                                 # and extrapolates badly to overload
