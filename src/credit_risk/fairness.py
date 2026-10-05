"""Fairness audit on the held-out set. Protected attributes are NOT model inputs;
this checks whether outcomes still differ across groups (proxy discrimination)."""
import json

import numpy as np
import pandas as pd

from . import config as C

DI_MIN = 0.80  # "four-fifths rule" on approval rates


def _groups(X: pd.DataFrame) -> dict[str, pd.Series]:
    sex = X["personal_status_sex"].map(
        lambda v: "female" if v.startswith("female") else "male")
    return {
        "sex": sex,
        "foreign_worker": X["foreign_worker"].map({"yes": "foreign", "no": "domestic"}),
        "age_group": pd.cut(X["age"], [0, 24, 59, 200], labels=["<25", "25-59", "60+"]).astype(str),
    }


def audit(X_te: pd.DataFrame, y_te, p_te, threshold: float) -> dict:
    y = np.asarray(y_te)
    reject = (np.asarray(p_te) >= threshold).astype(int)
    out = {"rule": f"approval-rate ratio vs best group must be >= {DI_MIN}", "attributes": {}}
    for attr, g in _groups(X_te).items():
        g = g.to_numpy()
        rows = {}
        for grp in sorted(set(g)):
            m = g == grp
            if m.sum() < 10:
                continue
            good, bad = m & (y == 0), m & (y == 1)
            rows[grp] = {
                "n": int(m.sum()),
                "approval_rate": float(1 - reject[m].mean()),
                "false_reject_rate": float(reject[good].mean()) if good.sum() else None,  # good people denied
                "default_catch_rate": float(reject[bad].mean()) if bad.sum() else None,
                "mean_pd": float(np.mean(np.asarray(p_te)[m])),
            }
        best = max(r["approval_rate"] for r in rows.values()) or 1.0
        for r in rows.values():
            r["disparate_impact"] = r["approval_rate"] / best
            r["passes_80_rule"] = bool(r["disparate_impact"] >= DI_MIN)
        out["attributes"][attr] = rows
    out["all_pass"] = all(r["passes_80_rule"] for a in out["attributes"].values() for r in a.values())
    C.FAIRNESS_PATH.write_text(json.dumps(out, indent=2))
    return out
