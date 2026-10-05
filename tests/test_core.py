import numpy as np
import pytest

from credit_risk import config as C
from credit_risk.data import load_raw, make_xy, split, validate
from credit_risk.features import FeatureEngineer, build_preprocessor
from credit_risk.scorecard import best_threshold, expected_cost, pd_to_score, risk_band


def test_data_loads_and_target_flipped():
    df = load_raw()
    X, y = make_xy(df)
    assert len(df) == 1000 and set(y.unique()) == {0, 1}
    assert abs(y.mean() - 0.30) < 1e-9          # 30% bad in German Credit


def test_validate_rejects_bad_data():
    df = load_raw().drop(columns=["age"])
    with pytest.raises(ValueError):
        validate(df)


def test_split_is_stratified():
    X, y = make_xy(load_raw())
    _, _, ytr, yte = split(X, y)
    assert abs(ytr.mean() - yte.mean()) < 0.01


def test_feature_engineer_adds_columns():
    X, _ = make_xy(load_raw())
    out = FeatureEngineer().transform(X)
    assert set(C.ENGINEERED) <= set(out.columns)
    assert (out["monthly_burden"] > 0).all()


def test_preprocessor_no_unsafe_names():
    X, y = make_xy(load_raw())
    Xt = build_preprocessor().fit_transform(X, y)
    assert not any(ch in c for c in Xt.columns for ch in "[]<")
    assert Xt.isna().sum().sum() == 0


def test_score_is_monotonic_decreasing_in_pd():
    s = pd_to_score(np.array([0.01, 0.1, 0.3, 0.6, 0.9]))
    assert (np.diff(s) < 0).all() and s.max() <= 850 and s.min() >= 300


def test_risk_band_edges():
    assert risk_band(700) == "Very Low Risk" and risk_band(400) == "Very High Risk"


def test_threshold_beats_naive_when_fn_is_costly():
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.3, 2000)
    p = np.clip(0.3 + 0.35 * (y - 0.3) + rng.normal(0, 0.15, 2000), 0.01, 0.99)
    t, c = best_threshold(y, p)
    assert t < 0.5 and c <= expected_cost(y, p, 0.5)
