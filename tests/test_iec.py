import numpy as np
import pytest
from dga.rules.iec import classify, ratios, THRESHOLDS

# gases are (H2, CH4, C2H6, C2H4, C2H2); each case is built to give the stated ratios
def test_pd():        assert classify(100, 5, 50, 4, 0) == "PD"         # R1=.05 R5=.08
def test_d1():        assert classify(100, 30, 10, 20, 30) == "D1"      # R2=1.5 R1=.3 R5=2.0 (R5 not > 2, so D1 only)
def test_d2():        assert classify(100, 80, 10, 50, 40) == "D2"      # R2=.8 R1=.8 R5=5
def test_t1():        assert classify(10, 30, 60, 30, 5) == "T1"        # R1=3 R5=.5
def test_t2():        assert classify(10, 30, 15, 30, 1) == "T2"        # R2=.033 R1=3 R5=2
def test_t3():        assert classify(10, 30, 5, 30, 1) == "T3"         # R2=.033 R1=3 R5=6
def test_undetermined():
    assert classify(100, 5, 10, 5, 0) == "undetermined"                 # R1=.05 but R5=.5

def test_overlap_is_reported_not_guessed():
    # R2=1.5, R1=.3, R5=3 satisfies both D1 and D2
    assert classify(100, 30, 10, 30, 45) == "ambiguous:D1/D2"

def test_zero_denominators():
    assert ratios(0, 5, 1, 0, 3)[0] == np.inf                           # C2H2/C2H4 with C2H4 = 0
    assert np.isnan(ratios(0, 0, 1, 1, 1)[1])                           # 0/0 is not evaluable

def test_every_fault_defined():
    assert set(THRESHOLDS) == {"PD", "D1", "D2", "T1", "T2", "T3"}
