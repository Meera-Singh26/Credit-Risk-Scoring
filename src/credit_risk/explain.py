"""SHAP-based explanations (global + per-applicant reason codes)."""
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression

from . import config as C


def make_explainer(clf, background: pd.DataFrame):
    if isinstance(clf, LogisticRegression):
        return shap.LinearExplainer(clf, background)
    return shap.TreeExplainer(clf)


def shap_values(explainer, Xt: pd.DataFrame) -> np.ndarray:
    sv = explainer.shap_values(Xt)
    if isinstance(sv, list):          # older shap: [class0, class1]
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:                  # (n, features, classes)
        sv = sv[:, :, 1]
    return sv


def pretty(feature: str) -> str:
    """'status_... < 100 DM' -> 'status = ... < 100 DM'."""
    for cat in sorted(C.CATEGORICAL, key=len, reverse=True):
        if feature.startswith(cat + "_"):
            return f"{cat.replace('_', ' ')} = {feature[len(cat) + 1:]}"
    return feature.replace("_", " ")


def reason_codes(sv_row: np.ndarray, x_row: pd.Series, feature_names, k: int = 5) -> list[dict]:
    order = np.argsort(-np.abs(sv_row))[:k]
    return [{"feature": pretty(feature_names[i]),
             "value": None if feature_names[i] not in x_row.index else _py(x_row[feature_names[i]]),
             "impact": round(float(sv_row[i]), 4),
             "direction": "raises risk" if sv_row[i] > 0 else "lowers risk"} for i in order]


def _py(v):
    return v.item() if hasattr(v, "item") else v
