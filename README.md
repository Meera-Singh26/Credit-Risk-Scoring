# Credit Risk Scoring: production-style ML system

Predicts **probability of default** for loan applicants, converts it to a **300-850 credit score**, makes a **cost-optimal approve/reject decision**, and explains each decision with **SHAP reason codes**. Includes experiment tracking, model registry, fairness audit, drift monitoring, hardened API, CI and Docker.

**Data:** German Credit (UCI), 1,000 loans, 30% defaults. Bundled in `data/raw` (zero setup).

## What makes it production-style
| Concern | Implementation |
|---|---|
| Reproducibility | fixed seeds, data SHA-256 fingerprint, pinned split, `requirements.txt` |
| Leakage control | preprocessing inside CV; calibrator + threshold fit on **out-of-fold train predictions**; test set touched once |
| Model selection | LogReg vs RandomForest vs tuned XGBoost, **repeated** stratified CV (5x3) |
| Honest uncertainty | bootstrap 95% CI on test AUC |
| Calibration | Platt scaling (Brier 0.1656 -> 0.1554) so scores and cost maths mean something |
| Business decision | threshold minimises cost (approve a defaulter = 5x reject a good customer) |
| Explainability | per-applicant SHAP reason codes + global importance |
| Fairness | `personal_status_sex`, `foreign_worker` **excluded from the model**; outcomes audited by sex / foreign worker / age with the 4/5 rule |
| Tracking | MLflow (SQLite): params, metrics, figures per run |
| Versioning | `models/registry/<timestamp>-<datahash>/` with artifact + meta; version returned in every API response |
| Monitoring | every prediction logged; **PSI drift** per feature and score vs training profile |
| Serving | FastAPI: API-key auth, request IDs, JSON logs, Prometheus `/metrics`, input validation, health probe |
| Governance | auto-generated `reports/MODEL_CARD.md` each training run |
| Quality | 24 tests, ruff, GitHub Actions CI (lint, train, test, drift sim, Docker smoke test) |

## Results (held-out test, 200 loans)
| Metric | Value |
|---|---|
| ROC-AUC | 0.802 (95% CI 0.7324 to 0.869) |
| Gini / KS | 0.604 / 0.5 |
| Defaulter recall @ threshold 0.17 | 88% |
| Cost vs approving everyone | -62% |

CV ROC-AUC: RandomForest 0.7959, XGBoost 0.7862, LogReg 0.7719.

## Quick start
```bash
pip install -r requirements-dev.txt
make all            # lint + EDA + train + tests
make api            # http://localhost:8000/docs   (set API_KEY=... to require X-API-Key)
make app            # Streamlit dashboard
make mlflow         # experiment tracker on :5000
make simulate-drift # fake a recession, watch PSI alert
docker compose up --build
```

```bash
curl -X POST localhost:8000/score -H 'content-type: application/json' -d @example_request.json
curl localhost:8000/monitoring/drift
```

## Layout
```
src/credit_risk/  config data features calibration scorecard train explain predict fairness monitor tracking report eda
api/              FastAPI app, schemas
app/              Streamlit (score, performance, insights, fairness & monitoring)
scripts/          traffic / drift simulator
tests/            data, features, scorecard, API, auth, logging, fairness, drift, registry
.github/          CI workflow
```

## Known issues (read before presenting)
- **Fairness audit flags two groups** (women, applicants under 25) below the 0.80 approval-ratio line, even with protected features excluded. This is proxy effect from legitimate features (status, housing, job) plus an aggressive 5:1 threshold, and groups are small (n=61, n=39). A real deployment would investigate, review the cost ratio, and get legal sign-off. See `reports/MODEL_CARD.md`.
- **Data is tiny.** AUC CI is wide (0.7324 to 0.869); differences between top models are within noise.
- **No reject inference,** and the data is old and German; scorecard scaling is conventional, not calibrated to a real book.
- Drift monitoring is verified on simulated traffic, not real production data.
