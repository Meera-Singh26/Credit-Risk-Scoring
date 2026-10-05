"""Inference service shared by the API and the Streamlit app."""
import joblib
import pandas as pd

from . import config as C
from .explain import make_explainer, reason_codes, shap_values
from .scorecard import pd_to_score, risk_band


class ModelNotTrained(RuntimeError):
    pass


class CreditScorer:
    def __init__(self, path=C.MODEL_PATH):
        if not path.exists():
            raise ModelNotTrained(f"{path} not found. Run `python -m credit_risk.train` first.")
        art = joblib.load(path)
        self.pipeline = art["pipeline"]
        self.calibrator = art["calibrator"]
        self.version = art["version"]
        self.trained_at = art["trained_at"]
        self.data_sha = art["data_sha256_12"]
        self.model_name = art["model_name"]
        self.threshold = art["threshold"]
        self.categories = art["categories"]
        self.numeric_ranges = art["numeric_ranges"]
        self.columns = art["input_columns"]
        self._pre = self.pipeline.named_steps["pre"]
        self._explainer = make_explainer(self.pipeline.named_steps["clf"], art["background"])

    def validate(self, record: dict) -> None:
        missing = [c for c in self.columns if c not in record]
        if missing:
            raise ValueError(f"Missing fields: {missing}")
        for c, allowed in self.categories.items():
            if record[c] not in allowed:
                raise ValueError(f"Invalid value for '{c}': {record[c]!r}. Allowed: {allowed}")

    def score(self, record: dict, top_k: int = 5) -> dict:
        self.validate(record)
        X = pd.DataFrame([{c: record[c] for c in self.columns}])
        p = float(self.calibrator.predict(self.pipeline.predict_proba(X)[:, 1])[0])
        score = float(pd_to_score(p))
        Xt = self._pre.transform(X)
        sv = shap_values(self._explainer, Xt)[0]
        return {
            "default_probability": round(p, 4),
            "credit_score": round(score),
            "risk_band": risk_band(score),
            "decision": "REJECT" if p >= self.threshold else "APPROVE",
            "threshold": round(self.threshold, 2),
            "model_version": self.version,
            "reasons": reason_codes(sv, Xt.iloc[0], list(Xt.columns), top_k),
        }

    def info(self) -> dict:
        return {"version": self.version, "model": self.model_name, "trained_at": self.trained_at,
                "data_sha256_12": self.data_sha, "threshold": round(self.threshold, 4)}

    def sample_record(self) -> dict:
        rec = {c: v[0] for c, v in self.categories.items()}
        rec.update({c: int(sum(r) / 2) for c, r in self.numeric_ranges.items()})
        return rec
