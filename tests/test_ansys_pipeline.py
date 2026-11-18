import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from twin import ansys_io as ai
from twin import pipeline_dp as pdp
from twin.dingwang import w_star

EX = Path(__file__).resolve().parent.parent / "examples" / "ansys"

def test_reads_real_snapshot_and_matches_independent_parse():
    df = ai.read_snapshot(EX / "T_load0p74.csv")
    assert len(df) == 1452 and df.node.is_unique
    s = ai.winding_stats(df)
    # values from the run log and an independent csv-module parse of the same file
    assert s["LV"]["T_max"] == pytest.approx(61.38, abs=0.01)
    assert s["LV"]["y_at_max"] == pytest.approx(1.13, abs=0.005)
    assert s["HV"]["T_max"] > s["LV"]["T_max"]          # the model's HV winding is hotter: contradicts the paper
    assert min(s["LV"]["T_min"], s["HV"]["T_min"]) == pytest.approx(50.9862, abs=0.01)

def test_overload_snapshot():
    s = ai.winding_stats(ai.read_snapshot(EX / "T_load1p30.csv"))
    assert s["LV"]["T_max"] == pytest.approx(117.07, abs=0.01)

def test_axial_profile_is_sorted_and_peaks_near_top():
    df = ai.read_snapshot(EX / "T_load0p74.csv")
    y, t = ai.axial_profile(df, "LV")
    assert np.all(np.diff(y) > 0) and t.max() == pytest.approx(61.38, abs=0.01)
    assert 0.8 < y[np.argmax(t)] / 1.2 < 1.0

def test_bad_files_rejected(tmp_path):
    p = tmp_path / "a.csv"
    p.write_text("node,x_m,y_m\n1,0.25,0\n")
    with pytest.raises(ValueError):
        ai.read_snapshot(p)
    p.write_text("node,x_m,y_m,T_C\n1,0.25,0,20\n1,0.26,0,20\n")
    with pytest.raises(ValueError):
        ai.read_snapshot(p)
    p.write_text("node,x_m,y_m,T_C\n1,0.25,0,not_a_number\n")
    with pytest.raises(ValueError):
        ai.read_snapshot(p)
    p.write_text("node,x_m,y_m,T_C\n1,0.25,0,5000\n")
    with pytest.raises(ValueError):
        ai.read_snapshot(p)

def test_node_outside_windings_rejected():
    df = pd.DataFrame(dict(node=[1], x_m=[0.31], y_m=[0.0], T_C=[40.0]))
    with pytest.raises(ValueError):
        ai.winding(df)

def test_batch_readers(tmp_path):
    (tmp_path / "runs.csv").write_text(
        "run,load_pu,HF,TTOP,Q_W_m3,LV_max_C,LV_max_y_m,HV_max_C,HV_max_y_m,T_min_C\n"
        "    1.,  0.400,   60.0,   40.0,  0.29200E+05,   45.000,   1.1000,   50.000,   1.0000,   30.000\n")
    (tmp_path / "snapshots.csv").write_text("run,node,x_m,y_m,T_C\n    1.,       1.,  0.250000E+00,  0.000000E+00,  0.300000E+02\n")
    assert ai.read_runs(tmp_path / "runs.csv").run.iloc[0] == 1
    assert ai.read_snapshots(tmp_path / "snapshots.csv").T_C.iloc[0] == 30.0

def test_dp_decreases_with_temperature_and_years_and_respects_floor():
    a = pdp.dp_profile(np.array([40.0, 60.0, 80.0, 100.0]), 20)
    assert np.all(np.diff(a) < 0)
    assert pdp.dp_profile(60.0, 30) < pdp.dp_profile(60.0, 10)
    assert pdp.dp_profile(150.0, 500) >= 1200 * (1 - w_star(0.18)) - 1e-9

def test_dp_at_paper_hot_spot_year1_is_the_fitted_value():
    assert float(pdp.dp_profile(61.29, 1)) == pytest.approx(pdp.TABLE2_I_YEAR1, abs=1e-6)   # a FIT by construction

def test_map_and_hot_spot_from_real_snapshot():
    df = ai.read_snapshot(EX / "T_load0p74.csv")
    m = pdp.dp_map_from_snapshot(df, 48)
    assert m.DP.min() >= 252 and m.DP.max() <= 1200
    assert m.DP[m.T_C.idxmax()] == m.DP.min()
    hs = pdp.hot_spot_dp(df, 48)
    assert hs["LV"]["DP"] > hs["HV"]["DP"]            # hotter HV ages more in this assumed model


# ---------- moisture field and DP per node ----------
def _snaps():
    return (ai.read_snapshot(EX / "T_load0p74.csv"), ai.read_snapshot(EX / "T_load1p30.csv"))

def test_moisture_field_mean_equals_eq13_and_is_wetter_where_cooler():
    from twin.moisture import total_moisture_pct
    a, _ = _snaps()
    for year in (3, 25, 48):
        c = pdp.moisture_field_pct(a, year)
        assert np.sum(c * a.x_m) / np.sum(a.x_m) == pytest.approx(float(total_moisture_pct(year)), rel=1e-9)
        assert np.corrcoef(c, a.T_C)[0, 1] < -0.9          # opposite trend to temperature, as in the paper
    assert pdp.moisture_field_pct(a, 48).max() > pdp.moisture_field_pct(a, 10).max()

def test_moisture_scaling_is_independent_of_the_eq1_prefactor(monkeypatch):
    from twin import moisture as mo
    a, _ = _snaps()
    base = pdp.moisture_field_pct(a, 20)
    orig = mo.equilibrium_paper_moisture_pct
    monkeypatch.setattr(mo, "equilibrium_paper_moisture_pct", lambda *args, **kw: 100.0 * orig(*args, **kw))
    assert np.allclose(pdp.moisture_field_pct(a, 20), base)      # only the temperature shape matters

def test_dp_with_moisture_and_overload_ordering_and_floor():
    a, b = _snaps()
    t_only = pdp.dp_map_from_snapshot(a, 25).DP
    wet = pdp.dp_map_with_moisture(a, 25).DP
    wet_ol = pdp.dp_map_with_moisture(a, 25, df_overload=b, overload_hours=24).DP
    assert (wet < t_only).all() and (wet_ol < wet).all()
    floor = 1200 * (1 - w_star(0.18))
    assert wet_ol.min() >= floor - 1e-9

def test_overload_inputs_are_checked():
    a, b = _snaps()
    with pytest.raises(ValueError):
        pdp.dp_map_with_moisture(a, 5, overload_hours=24)
    with pytest.raises(ValueError):
        pdp.dp_map_with_moisture(a, 5, df_overload=b.iloc[:-1], overload_hours=24)

def test_position_summary_lv_is_lowest_at_top_in_the_assumed_model():
    a, _ = _snaps()
    s = pdp.position_summary(pdp.dp_map_with_moisture(a, 20))
    assert s["LV"]["top"] < s["LV"]["middle"] < s["LV"]["bottom"]

def test_hot_spot_summary_matches_hot_spot_dp():
    a, _ = _snaps()
    s1 = pdp.hot_spot_summary(pdp.dp_map_from_snapshot(a, 30))
    s2 = pdp.hot_spot_dp(a, 30)
    for w in ("LV", "HV"):
        assert s1[w]["DP"] == pytest.approx(s2[w]["DP"]) and s1[w]["T"] == pytest.approx(s2[w]["T"])
