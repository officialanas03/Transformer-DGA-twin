import pytest
from pathlib import Path
from twin import thermal_variants as tv, pipeline_dp as pdp, ansys_io as ai

EX = Path(__file__).resolve().parent.parent / "examples" / "ansys"

@pytest.fixture(scope="module")
def vs():
    return {k: tv.variant(*args) for k, args in {"A": (40, 1.0), "B": (40, 0.6), "D": (60, 0.6)}.items()}

def test_fit_hits_the_printed_hot_spot(vs):
    for v in vs.values():
        assert v["lv_max"] == pytest.approx(tv.HOT_SPOT_TARGET, abs=0.01)

def test_variant_a_matches_the_ansys_based_dp_summary(vs):
    ansys = pdp.position_summary(pdp.dp_map_with_moisture(ai.read_snapshot(EX / "T_load0p74.csv"), 48))
    mine = pdp.position_summary(pdp.dp_map_with_moisture(vs["A"]["base"], 48))
    for w in ("LV", "HV"):
        for k in ("top", "middle", "bottom"):
            assert mine[w][k] == pytest.approx(ansys[w][k], abs=15)      # same model, different mesh and node positions

def test_unfitted_checks_are_reported_not_forced(vs):
    assert not vs["A"]["lv_max"] > vs["A"]["hv_max"]            # equal loss density: HV is hotter (contradicts the paper)
    assert vs["B"]["lv_max"] > vs["B"]["hv_max"]                # HV loss density x0.6: LV hotter, as the paper says
    for k in ("A", "B"):
        assert abs(vs[k]["lv_overload"] - tv.OVERLOAD_TARGET) / tv.OVERLOAD_TARGET < 0.05
    assert vs["D"]["lv_overload"] < 0.75 * tv.OVERLOAD_TARGET   # a wider temperature range breaks the overload check

def test_wider_temperature_range_steepens_the_dp_gradient(vs):
    a = pdp.position_summary(pdp.dp_map_with_moisture(vs["B"]["base"], 48))["LV"]
    d = pdp.position_summary(pdp.dp_map_with_moisture(vs["D"]["base"], 48))["LV"]
    assert d["bottom"] - d["top"] > a["bottom"] - a["top"]

def test_unreachable_target_raises():
    with pytest.raises(ValueError):
        tv.fit_film_coefficient(40, target=500.0)
