"""Platt scaling so predicted probabilities mean what they say (needed for scores & cost maths)."""
import numpy as np
from sklearn.linear_model import LogisticRegression


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p)).reshape(-1, 1)


class PlattCalibrator:
    def fit(self, p_raw, y):
        self.lr_ = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(p_raw), y)
        return self

    def predict(self, p_raw) -> np.ndarray:
        return self.lr_.predict_proba(_logit(p_raw))[:, 1]
