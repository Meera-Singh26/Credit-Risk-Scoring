import json

import numpy as np
import pandas as pd
import pytest

from credit_risk import config as C
from credit_risk import monitor
from credit_risk.data import load_raw, make_xy, split
from credit_risk.fairness import audit

needs_model = pytest.mark.skipif(not C.MODEL_PATH.exists(), reason="train the model first")


def test_psi_identical_is_zero_and_shift_is_positive():
    e = np.array([.1] * 10)
    assert monitor.psi(e, e) == pytest.approx(0, abs=1e-9)
    shifted = np.array([.02] * 5 + [.18] * 5)
    assert monitor.psi(e, shifted) > C.PSI_ALERT


def test_protected_attributes_not_model_features():
    assert not set(C.PROTECTED) & set(C.NUMERIC + C.CATEGORICAL)


def test_fairness_audit_structure(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "FAIRNESS_PATH", tmp_path / "f.json")
    X, y = make_xy(load_raw())
    _, X_te, _, y_te = split(X, y)
    p = np.random.default_rng(0).uniform(0, 1, len(X_te))
    out = audit(X_te, y_te, p, 0.5)
    assert {"sex", "foreign_worker", "age_group"} <= set(out["attributes"])
    for groups in out["attributes"].values():
        assert max(g["disparate_impact"] for g in groups.values()) == pytest.approx(1.0)
    assert json.loads((tmp_path / "f.json").read_text())["rule"]


@needs_model
def test_registry_and_meta_present():
    meta = json.loads(C.META_PATH.read_text())
    assert (C.REGISTRY_DIR / meta["version"] / "credit_model.joblib").exists()
    assert (C.REGISTRY_DIR / meta["version"] / "meta.json").exists()


@needs_model
def test_calibration_improves_brier():
    t = json.loads(C.METRICS_PATH.read_text())["test"]
    assert t["brier"] <= t["brier_uncalibrated"] + 1e-3


def _live(scorer, df):
    p = scorer.calibrator.predict(scorer.pipeline.predict_proba(df[scorer.columns])[:, 1])
    out = df.copy()
    out["credit_score"] = scorer_scores(p)
    out["decision"] = np.where(p >= scorer.threshold, "REJECT", "APPROVE")
    return out


def scorer_scores(p):
    from credit_risk.scorecard import pd_to_score
    return pd_to_score(p)


@needs_model
def test_drift_detects_shift_but_not_stable_traffic(tmp_path, monkeypatch):
    from credit_risk.predict import CreditScorer
    monkeypatch.setattr(C, "DRIFT_PATH", tmp_path / "d.json")
    s = CreditScorer()
    X, y = make_xy(load_raw())
    X = split(X, y)[1]   # unseen applicants, like real traffic
    stable = _live(s, X.sample(400, replace=True, random_state=1))
    assert monitor.drift_report(stable)["status"] in {"ok", "warn"}
    bad = X.sample(400, replace=True, random_state=2).copy()
    bad["amount"] *= 2.2
    bad["duration"] = (bad["duration"] + 18).clip(upper=72)
    rep = monitor.drift_report(_live(s, bad))
    assert rep["status"] == "alert" and rep["features"]["amount"]["status"] == "alert"


def test_insufficient_data_guard():
    assert monitor.drift_report(pd.DataFrame({"a": [1]}))["status"] == "insufficient_data"
