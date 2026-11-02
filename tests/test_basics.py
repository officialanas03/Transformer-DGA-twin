import pandas as pd
import pytest
from dga.data.labels import standardize, to_coarse
from dga.data.loader import load_dga
from dga.ml.features import make_features

def test_standardize_maps_all_source_vocabularies():
    s = pd.Series(["No fault", "Low intensity discharge", "High intensity discharge", "Thermal fault",
                   "T1&T2", "D2", "T3+D2", "something odd"])
    assert standardize(s).tolist() == ["Normal", "D1", "D2", "T", "T1&T2", "D2", None, None]

def test_to_coarse_merges_t1_t2():
    assert to_coarse(pd.Series(["T1", "T2", "T3", "D1"])).tolist() == ["T1&T2", "T1&T2", "T3", "D1"]

def test_loader_accepts_missing_co_and_drops_mixed(tmp_path):
    p = tmp_path / "a.csv"
    pd.DataFrame({"H2": [1, 2], "CH4": [1, 2], "C2H6": [1, 2], "C2H4": [1, 2], "C2H2": [1, 2],
                  "fault": ["D2", "T3+D2"]}).to_csv(p, index=False)
    with pytest.warns(UserWarning):
        df = load_dga(p)
    assert len(df) == 1 and "CO" not in df.columns

def test_loader_rejects_negative(tmp_path):
    p = tmp_path / "b.csv"
    pd.DataFrame({"H2": [-1], "CH4": [1], "C2H6": [1], "C2H4": [1], "C2H2": [1], "fault": ["D2"]}).to_csv(p, index=False)
    with pytest.raises(ValueError):
        load_dga(p)

def test_features_zero_total_is_finite():
    df = pd.DataFrame({"H2": [0], "CH4": [0], "C2H6": [0], "C2H4": [0], "C2H2": [0]})
    X = make_features(df)
    assert X.notna().all().all() and X.shape[1] == 6

def test_features_use_co_requires_columns():
    df = pd.DataFrame({"H2": [1], "CH4": [1], "C2H6": [1], "C2H4": [1], "C2H2": [1]})
    with pytest.raises(ValueError):
        make_features(df, use_co=True)
