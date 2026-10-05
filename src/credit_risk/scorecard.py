"""Business layer: cost-optimal threshold, credit score, risk bands."""
import numpy as np

from . import config as C


def expected_cost(y_true, p, threshold: float) -> float:
    y_true, p = np.asarray(y_true), np.asarray(p)
    reject = p >= threshold
    fn = ((~reject) & (y_true == 1)).sum()   # approved a defaulter
    fp = (reject & (y_true == 0)).sum()      # rejected a good customer
    return float(C.COST_FN * fn + C.COST_FP * fp)


def best_threshold(y_true, p) -> tuple[float, float]:
    grid = np.linspace(0.05, 0.95, 91)
    costs = [expected_cost(y_true, p, t) for t in grid]
    i = int(np.argmin(costs))
    return float(grid[i]), float(costs[i])


def pd_to_score(p, eps: float = 1e-4):
    """Map P(default) to a 300-850 credit score (higher = safer)."""
    p = np.clip(p, eps, 1 - eps)
    factor = C.PDO / np.log(2)
    offset = C.BASE_SCORE - factor * np.log(C.BASE_ODDS)
    return np.clip(offset + factor * np.log((1 - p) / p), 300, 850)


def risk_band(score: float) -> str:
    if score >= 650: return "Very Low Risk"
    if score >= 600: return "Low Risk"
    if score >= 550: return "Medium Risk"
    if score >= 500: return "High Risk"
    return "Very High Risk"
