"""Generate synthetic live traffic and run drift monitoring.

  python scripts/simulate_traffic.py            # healthy traffic   -> PSI ok
  python scripts/simulate_traffic.py --drift    # shifted economy   -> PSI alert
"""
import argparse
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src")]

import numpy as np  # noqa: E402

from credit_risk import config as C  # noqa: E402
from credit_risk import monitor  # noqa: E402
from credit_risk.data import load_raw, make_xy, split  # noqa: E402
from credit_risk.predict import CreditScorer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=300)
    ap.add_argument("--drift", action="store_true", help="simulate a recession: bigger/longer loans, weaker finances")
    ap.add_argument("--reset", action="store_true", help="delete existing prediction log first")
    a = ap.parse_args()
    if a.reset and C.PRED_LOG.exists():
        C.PRED_LOG.unlink()

    rng = np.random.default_rng(7)
    scorer = CreditScorer()
    # Live traffic = applicants the model has NOT seen (held-out split), as in production.
    X, y = make_xy(load_raw())
    df = split(X, y)[1].sample(a.n, replace=True, random_state=7)
    for _, row in df.iterrows():
        rec = row.to_dict()
        if a.drift:
            rec["amount"] = float(rec["amount"] * rng.uniform(1.6, 2.4))
            rec["duration"] = int(min(72, rec["duration"] + rng.integers(8, 24)))
            if rng.random() < 0.5:
                rec["savings"] = "... < 100 DM"
            if rng.random() < 0.3:
                rec["employment_duration"] = "unemployed"
        monitor.log_prediction(rec, scorer.score(rec))
    rep = monitor.drift_report()
    print(f"status={rep['status']} score_psi={rep['score_psi']} reject_rate={rep['reject_rate']:.0%}")
    for k, v in list(rep["features"].items())[:5]:
        print(f"  {k:<22} PSI {v['psi']:<7} {v['status']}")
    print(rep["action"])


if __name__ == "__main__":
    main()
