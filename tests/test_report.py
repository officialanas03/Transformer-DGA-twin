import numpy as np
import pytest
from twin import report as rp
from twin.surrogate import WindingSurrogate

def test_table2_frame_structure_and_statuses():
    df = rp.luo_table2_frame()
    assert len(df) == 25 and set(df.year) == set(rp.YEARS)
    first = df[df.case.str.startswith("I:")]
    assert first["diff"].abs().max() < 1.0 and first.status.iloc[0].startswith("FITTED")
    assert all(s.startswith("VARIANT") for s in df[~df.case.str.startswith("I:")].status)
    assert np.allclose(df["diff"], df["model"] - df["paper"], atol=0.011)

def test_calibration_demo_is_flagged_synthetic():
    L, H = np.linspace(0.3, 1.4, 6), np.linspace(50, 110, 5)
    s = WindingSurrogate(nz=40, nr_lv=8, nr_hv=10).fit_from_model(L, H)
    out = rp.calibration_demo(s, s, 72.0, hours=240)
    assert out["synthetic"] is True and out["run"].synthetic is True
    assert out["rmse_filter"] < out["rmse_sensor"] < out["rmse_uncal"]
    assert abs(out["hf_est"][-1] - 72.0) < 5.0
