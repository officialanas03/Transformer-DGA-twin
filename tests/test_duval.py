import itertools
import pytest
from dga.rules.duval import classify, classify_pct, ZONE_ORDER

# (CH4, C2H4, C2H2) in ppm, expected zone, source. Each case is a worked example
# published in a paper other than the one the breakpoints came from.
PUBLISHED = [
    ((10, 35, 5), "T3", "Dukarm 2020 (Energies 13:6459), Eq. 5 example"),
    ((140, 120, 0), "T2", "Dukarm 2020, Case 1 (Mansour case)"),
    ((101.2, 158.8, 0), "T3", "Dukarm 2020, Table 2 ABC"),
    ((260.0, 0, 0), "PD", "Dukarm 2020, Table 2 ABD"),
    ((255.2, 63.7, 95.2), "D1", "Dukarm 2020, Table 2 BCE"),
    ((294.5, 0, 55.8), "D1", "Dukarm 2020, Table 2 BDE"),
    ((261.2, 125.8, 75), "D2", "Dukarm 2020, Table 4 'Other'"),
    ((1313, 616, 2908), "D1", "Reinhausen worked example"),
    ((32, 32, 36), "D2", "Dessouky 2016, C2H2=.36 CH4=.32 C2H4=.32"),
]

@pytest.mark.parametrize("gases,expected,source", PUBLISHED)
def test_published_examples(gases, expected, source):
    assert classify(*gases) == expected, source

def test_corners():
    assert classify(100, 0, 0) == "PD"
    assert classify(0, 100, 0) == "T3"
    assert classify(0, 0, 100) == "D1"

def test_all_zero_is_none():
    assert classify(0, 0, 0) is None

def test_every_point_gets_exactly_one_known_zone():
    step = 0.5
    n = int(100 / step)
    seen = set()
    for i, j in itertools.product(range(n + 1), repeat=2):
        c2h4, c2h2 = i * step, j * step
        if c2h4 + c2h2 > 100:
            continue
        zone = classify_pct(100 - c2h4 - c2h2, c2h4, c2h2)
        assert zone in ZONE_ORDER
        seen.add(zone)
    assert seen == set(ZONE_ORDER)   # all seven zones are reachable

def test_breakpoints_sit_on_the_expected_side():
    assert classify_pct(2.0, 20.0, 0.0) == "T2"      # C2H4 = 20 is T2, not T1
    assert classify_pct(81.0, 19.0, 0.0) == "T1"
    assert classify_pct(64.0, 23.0, 13.0) == "D2"    # C2H4 = 23, C2H2 = 13
    assert classify_pct(64.1, 22.9, 13.0) == "D1"
    assert classify_pct(98.0, 1.0, 1.0) == "PD"
