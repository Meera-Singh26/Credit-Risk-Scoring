"""Drift monitoring: PSI of live traffic vs the training reference profile."""
import json
import threading
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import config as C

EPS = 1e-4
_lock = threading.Lock()


def log_prediction(record: dict, result: dict, request_id: str = "") -> None:
    """Append one JSON line per scored application (inputs + outputs) for monitoring/audit."""
    C.LOG_DIR.mkdir(exist_ok=True)
    row = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "request_id": request_id,
           "model_version": result.get("model_version"),
           **{k: record[k] for k in C.NUMERIC + C.CATEGORICAL},
           "default_probability": result["default_probability"],
           "credit_score": result["credit_score"], "decision": result["decision"]}
    with _lock, open(C.PRED_LOG, "a") as f:
        f.write(json.dumps(row) + "\n")


def psi(expected: np.ndarray, actual: np.ndarray) -> float:
    e, a = np.clip(expected, EPS, None), np.clip(actual, EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def build_reference(X_tr: pd.DataFrame, scores: np.ndarray) -> dict:
    """Decile bins for numeric features + the score, category shares for categoricals."""
    ref = {"numeric": {}, "categorical": {}, "n": int(len(X_tr))}
    for c in C.NUMERIC:
        edges = np.unique(np.quantile(X_tr[c], np.linspace(0, 1, 11)))
        edges[0], edges[-1] = -np.inf, np.inf
        share = np.histogram(X_tr[c], edges)[0] / len(X_tr)
        ref["numeric"][c] = {"edges": [float(e) if np.isfinite(e) else None for e in edges],
                             "share": share.tolist()}
    for c in C.CATEGORICAL:
        ref["categorical"][c] = X_tr[c].value_counts(normalize=True).to_dict()
    edges = np.unique(np.quantile(scores, np.linspace(0, 1, 11)))
    edges[0], edges[-1] = -np.inf, np.inf
    ref["score"] = {"edges": [float(e) if np.isfinite(e) else None for e in edges],
                    "share": (np.histogram(scores, edges)[0] / len(scores)).tolist()}
    C.REFERENCE_PATH.write_text(json.dumps(ref))
    return ref


def _edges(lst):
    return np.array([-np.inf if i == 0 else np.inf if i == len(lst) - 1 else v
                     for i, v in enumerate(lst)], dtype=float)


def status(v: float) -> str:
    return "alert" if v >= C.PSI_ALERT else "warn" if v >= C.PSI_WARN else "ok"


def load_log(path=C.PRED_LOG, last: int | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    df = pd.DataFrame(rows)
    return df.tail(last) if last else df


def drift_report(live: pd.DataFrame | None = None) -> dict:
    ref = json.loads(C.REFERENCE_PATH.read_text())
    live = load_log() if live is None else live
    if len(live) < C.MIN_MONITOR_ROWS:
        return {"status": "insufficient_data", "rows": int(len(live)),
                "needed": C.MIN_MONITOR_ROWS}
    feats = {}
    for c, r in ref["numeric"].items():
        share = np.histogram(live[c].astype(float), _edges(r["edges"]))[0] / len(live)
        feats[c] = psi(np.array(r["share"]), share)
    for c, r in ref["categorical"].items():
        cats = sorted(set(r) | set(live[c].unique()))
        e = np.array([r.get(k, 0) for k in cats])
        a = np.array([(live[c] == k).mean() for k in cats])
        feats[c] = psi(e, a)
    sc = np.histogram(live["credit_score"].astype(float), _edges(ref["score"]["edges"]))[0] / len(live)
    score_psi = psi(np.array(ref["score"]["share"]), sc)
    worst = sorted(feats.items(), key=lambda kv: -kv[1])
    overall = max([score_psi] + list(feats.values()))
    rep = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows": int(len(live)), "status": status(overall),
        "score_psi": round(score_psi, 4), "score_status": status(score_psi),
        "reject_rate": float((live["decision"] == "REJECT").mean()),
        "features": {k: {"psi": round(v, 4), "status": status(v)} for k, v in worst},
        "action": {"ok": "No action.", "warn": "Investigate shifted features.",
                   "alert": "Significant drift: review data pipeline and consider retraining."}[status(overall)],
    }
    C.DRIFT_PATH.write_text(json.dumps(rep, indent=2))
    return rep


if __name__ == "__main__":
    print(json.dumps(drift_report(), indent=2))
