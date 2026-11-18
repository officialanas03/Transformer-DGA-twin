import pytest
from twin import luo_chain as lc
from twin.dingwang import w_star, k_dp
from twin.paper_numbers import load, value

ROWS = load()
def row(first):
    return {y: value(f"c{n}", ROWS) for y, n in zip((1, 10, 25, 40, 48), range(first, first + 5))}
T2_I, T2_II, T2_III = row(67), row(72), row(77)
YEARS = [1, 10, 25, 40, 48]

def test_fit_reproduces_year1_and_row_one_within_one_dp():
    ks = lc.fit_k_scale(T2_I[1])
    got = lc.table2_row(YEARS, "none", ks)
    assert got[1] == pytest.approx(T2_I[1], abs=1e-6)               # by construction: a FIT
    for y in YEARS[1:]:
        assert abs(got[y] - T2_I[y]) < 1.0                          # the form reproduces later years

def test_dp_never_below_levelling_off_and_monotonic():
    ks = lc.fit_k_scale(T2_I[1])
    floor = 1200 * (1 - w_star(0.18))
    for variant in ("none", "eq13", "const0p70"):
        vals = list(lc.table2_row(list(range(1, 200)), variant, ks, overload_hours=24).values())
        assert all(a >= b for a, b in zip(vals, vals[1:]))
        assert min(vals) >= floor

def test_moisture_and_overload_speed_up_ageing_in_variants():
    ks = lc.fit_k_scale(T2_I[1])
    base = lc.table2_row(YEARS, "none", ks)
    wet = lc.table2_row(YEARS, "eq13", ks)
    wet_ol = lc.table2_row(YEARS, "eq13", ks, overload_hours=24)
    for y in YEARS:
        assert wet[y] < base[y] and wet_ol[y] < wet[y]

def test_time_to_250_does_not_exist_in_the_printed_model():
    # 250 is below the levelling-off DP (about 252) at 0.18 % moisture, so Eq. 5 never reaches it
    ks = lc.fit_k_scale(T2_I[1])
    assert lc.years_to_reach(250, "none", ks) is None
    assert lc.years_to_reach(300, "none", ks) is not None

def test_seven_point_four_claim_not_reproduced():
    ratio = lc.relative_rate_moisture_only(2.0, 0.2)                 # alpha_C alone
    assert 1.9 < ratio < 2.2
    assert ratio < 7.4

def test_printed_k_dp_makes_moisture_slow_ageing_where_defined():
    assert k_dp(0.5) < k_dp(0.18)
    assert lc.relative_rate_moisture_only(0.5, 0.18, include_k_dp=True) < 1.0

def test_unknown_variant_rejected():
    with pytest.raises(ValueError):
        lc.moisture_for_year("bogus", 1)
