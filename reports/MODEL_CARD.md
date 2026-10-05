# Model Card: Credit Risk Scoring

**Version:** `20261005-063049-ac003d`  |  **Trained:** 2026-10-05T06:30:49+00:00  |  **Data hash:** `ac003d1158e4`
**Algorithm:** random_forest + Platt calibration  |  **Train/Test:** 800/200

## Intended use
Rank consumer-loan applicants by probability of default and support an approve/reject decision with reason codes.
**Not** for fully automated adverse decisions without human review. Demonstration model on a public 1,000-row dataset.

## Performance (held-out test set)
| Metric | Value |
|---|---|
| ROC-AUC | 0.802 (95% bootstrap CI 0.732 to 0.869) |
| Gini / KS | 0.604 / 0.500 |
| Brier (calibrated / raw) | 0.155 / 0.166 |
| Threshold | P(default) >= 0.17 -> reject (cost FN:FP = 5:1) |
| Defaulter recall | 88% |
| Cost reduction vs approve-all | 62% |

## Fairness
Protected attributes (personal_status_sex, foreign_worker) are **excluded from the model** and used only for auditing.
Rule: approval-rate ratio vs best group >= 0.80.

| Attribute | Group | n | Approval rate | Disparate impact | 4/5 rule |
|---|---|---|---|---|---|
| sex | female | 61 | 28% | 0.74 | FAIL |
| sex | male | 139 | 37% | 1.00 | pass |
| foreign_worker | foreign | 194 | 34% | 1.00 | pass |
| age_group | 25-59 | 146 | 38% | 0.96 | pass |
| age_group | 60+ | 15 | 40% | 1.00 | pass |
| age_group | <25 | 39 | 18% | 0.45 | FAIL |

Small groups produce noisy ratios; treat results as indicative, not conclusive.

## Limitations
- 1,000 training rows; wide confidence intervals; one geography and era (German credit data).
- Cost ratio 5:1 is an assumption; set it from real lender economics.
- No reject-inference: the data only contains approved loans, so rejected applicants are unseen.
- Age is used as a feature and audited; confirm legal permissibility for your jurisdiction.

## Monitoring
Every prediction is logged; `python -m credit_risk.monitor` computes PSI vs the training profile
(<0.10 stable, 0.10-0.25 investigate, >0.25 retrain).
