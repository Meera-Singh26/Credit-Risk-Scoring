"""Training pipeline.

1. Repeated stratified CV compares models (train split only).
2. Out-of-fold predictions -> Platt calibrator + cost-optimal threshold (test set untouched).
3. One-shot evaluation on the held-out test set with bootstrap confidence intervals.
4. Fairness audit, drift reference profile, model card.
5. Versioned artifact in models/registry/, tracked in MLflow.
"""
import argparse
import json
import logging
import platform
import shutil
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from scipy.stats import ks_2samp
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, roc_auc_score
from sklearn.model_selection import (
    RandomizedSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from . import config as C
from . import tracking
from .calibration import PlattCalibrator
from .data import data_fingerprint, load_raw, make_xy, split
from .features import build_preprocessor
from .scorecard import best_threshold, expected_cost, pd_to_score

log = logging.getLogger("train")


def candidates(fast: bool = False) -> dict[str, Pipeline]:
    xgb = XGBClassifier(eval_metric="logloss", random_state=C.RANDOM_STATE, n_jobs=1, tree_method="hist")
    return {
        "logreg": Pipeline([("pre", build_preprocessor(scale_numeric=True)),
                            ("clf", LogisticRegression(C=0.3, max_iter=2000))]),
        "random_forest": Pipeline([("pre", build_preprocessor()),
                                   ("clf", RandomForestClassifier(
                                       n_estimators=150 if fast else 500, min_samples_leaf=3,
                                       class_weight="balanced_subsample",
                                       random_state=C.RANDOM_STATE, n_jobs=-1))]),
        "xgboost": Pipeline([("pre", build_preprocessor()), ("clf", xgb)]),
    }


def tune_xgb(pipe, X, y, n_iter):
    space = {"clf__n_estimators": [150, 250, 400], "clf__learning_rate": [0.02, 0.05, 0.1],
             "clf__max_depth": [2, 3, 4], "clf__min_child_weight": [1, 3, 5],
             "clf__subsample": [0.7, 0.85, 1.0], "clf__colsample_bytree": [0.6, 0.8, 1.0],
             "clf__reg_lambda": [1, 3, 10]}
    cv = StratifiedKFold(C.CV_FOLDS, shuffle=True, random_state=C.RANDOM_STATE)
    s = RandomizedSearchCV(pipe, space, n_iter=n_iter, scoring="roc_auc", cv=cv,
                           random_state=C.RANDOM_STATE, n_jobs=-1).fit(X, y)
    log.info("XGB tuned: CV AUC %.4f", s.best_score_)
    return s.best_estimator_, s.best_params_


def compare(models, X, y, repeats):
    cv = RepeatedStratifiedKFold(n_splits=C.CV_FOLDS, n_repeats=repeats, random_state=C.RANDOM_STATE)
    rows = {}
    for name, m in models.items():
        r = cross_validate(m, X, y, cv=cv, n_jobs=-1, scoring={
            "auc": "roc_auc", "pr_auc": "average_precision", "brier": "neg_brier_score"})
        rows[name] = {"cv_auc": r["test_auc"].mean(), "cv_auc_std": r["test_auc"].std(),
                      "cv_pr_auc": r["test_pr_auc"].mean(), "cv_brier": -r["test_brier"].mean()}
        log.info("%-14s AUC %.4f ± %.4f (%d folds)", name, rows[name]["cv_auc"],
                 rows[name]["cv_auc_std"], len(r["test_auc"]))
    return pd.DataFrame(rows).T.sort_values("cv_auc", ascending=False)


def bootstrap_ci(y, p, n, seed=C.RANDOM_STATE):
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    aucs = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            aucs.append(roc_auc_score(y[i], p[i]))
    return [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))]


def evaluate(y_true, p, threshold, n_boot):
    y_true = np.asarray(y_true)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    auc = roc_auc_score(y_true, p)
    cost, base = expected_cost(y_true, p, threshold), expected_cost(y_true, p, 1.01)
    return {"roc_auc": auc, "roc_auc_ci95": bootstrap_ci(y_true, p, n_boot), "gini": 2 * auc - 1,
            "pr_auc": average_precision_score(y_true, p), "brier": brier_score_loss(y_true, p),
            "ks": ks_2samp(p[y_true == 1], p[y_true == 0]).statistic, "threshold": threshold,
            "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
            "recall_defaulters": tp / (tp + fn), "precision_defaulters": tp / max(tp + fp, 1),
            "cost_model": cost, "cost_approve_all": base, "cost_reduction_pct": 100 * (1 - cost / base)}


def _round(o):
    if isinstance(o, float):
        return round(o, 4)
    if isinstance(o, dict):
        return {k: _round(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_round(v) for v in o]
    return o


def main(fast: bool = False) -> dict:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for d in (C.MODEL_DIR, C.REPORT_DIR, C.FIG_DIR, C.REGISTRY_DIR):
        d.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    sha = data_fingerprint()
    version = f"{now:%Y%m%d-%H%M%S}-{sha[:6]}"
    repeats, n_boot = (1, 200) if fast else (C.CV_REPEATS, C.BOOTSTRAPS)

    with tracking.run(f"train-{version}"):
        df = load_raw()
        X, y = make_xy(df)
        X_tr, X_te, y_tr, y_te = split(X, y)
        log.info("Version %s | train %d | test %d | default rate %.1f%%", version, len(X_tr), len(X_te), 100 * y.mean())

        models = candidates(fast)
        models["xgboost"], xgb_params = tune_xgb(models["xgboost"], X_tr, y_tr, 6 if fast else 25)
        leaderboard = compare(models, X_tr, y_tr, repeats)
        best_name = leaderboard.index[0]
        best = models[best_name]
        log.info("Selected: %s", best_name)

        # Out-of-fold raw probabilities -> calibrator + threshold. Test set never touched.
        cv = StratifiedKFold(C.CV_FOLDS, shuffle=True, random_state=C.RANDOM_STATE)
        oof_raw = cross_val_predict(best, X_tr, y_tr, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
        calibrator = PlattCalibrator().fit(oof_raw, y_tr)
        oof = calibrator.predict(oof_raw)
        threshold, _ = best_threshold(y_tr, oof)

        best.fit(X_tr, y_tr)
        p_raw = best.predict_proba(X_te)[:, 1]
        p_te = calibrator.predict(p_raw)
        test = evaluate(y_te, p_te, threshold, n_boot)
        test["brier_uncalibrated"] = brier_score_loss(y_te, p_raw)
        log.info("TEST AUC %.4f CI%s | KS %.3f | Brier %.4f (raw %.4f) | cost -%.1f%%",
                 test["roc_auc"], np.round(test["roc_auc_ci95"], 3), test["ks"], test["brier"],
                 test["brier_uncalibrated"], test["cost_reduction_pct"])

        from .fairness import audit
        fair = audit(X_te, y_te, p_te, threshold)
        log.info("Fairness 4/5 rule all pass: %s", fair["all_pass"])

        from .monitor import build_reference
        build_reference(X_tr, pd_to_score(oof))

        background = best.named_steps["pre"].transform(X_tr).sample(
            min(100, len(X_tr)), random_state=C.RANDOM_STATE)
        meta = {"version": version, "trained_at": now.isoformat(timespec="seconds"),
                "data_sha256_12": sha, "model_name": best_name, "threshold": threshold,
                "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
                "excluded_protected_features": C.PROTECTED,
                "env": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                        "xgboost": xgboost.__version__}}
        artifact = {**meta, "pipeline": best, "calibrator": calibrator, "background": background,
                    "categories": {c: sorted(df[c].unique().tolist()) for c in C.CATEGORICAL},
                    "numeric_ranges": {c: [float(df[c].min()), float(df[c].max())] for c in C.NUMERIC},
                    "input_columns": C.NUMERIC + C.CATEGORICAL}

        vdir = C.REGISTRY_DIR / version
        vdir.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact, vdir / "credit_model.joblib")
        shutil.copy(vdir / "credit_model.joblib", C.MODEL_PATH)   # "current" pointer
        metrics = {"selected_model": best_name, "version": version,
                   "leaderboard": _round(leaderboard.to_dict("index")), "test": _round(test)}
        (vdir / "meta.json").write_text(json.dumps({**meta, "test": metrics["test"]}, indent=2))
        C.METRICS_PATH.write_text(json.dumps(metrics, indent=2))
        C.META_PATH.write_text(json.dumps({**meta, "categories": artifact["categories"],
                                           "numeric_ranges": artifact["numeric_ranges"]}, indent=2))

        from .report import make_figures, write_model_card
        make_figures(best, best_name, X_tr, X_te, y_te, p_te, p_raw, threshold, leaderboard, y_tr, oof)
        write_model_card(meta, metrics, fair)

        tracking.log_params({"model": best_name, "version": version, "data_sha": sha,
                             "threshold": threshold, "cost_fn": C.COST_FN, **{f"xgb_{k}": v for k, v in xgb_params.items()}})
        tracking.log_metrics({k: v for k, v in test.items() if isinstance(v, float)})
        tracking.log_metrics({f"cv_auc_{n}": r["cv_auc"] for n, r in leaderboard.iterrows()})
        tracking.log_artifacts(C.FIG_DIR)
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="smaller search for CI/tests")
    main(ap.parse_args().fast)
