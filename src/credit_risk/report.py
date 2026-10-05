"""Figure generation for the README / Streamlit performance tab."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve

from . import config as C
from .explain import make_explainer, pretty, shap_values
from .scorecard import expected_cost, pd_to_score

INK, RED, TEAL, GOLD = "#0f172a", "#e11d48", "#0d9488", "#f59e0b"


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / name, dpi=140)
    plt.close(fig)


def make_figures(pipe, name, X_tr, X_te, y_te, p_te, p_raw, threshold, leaderboard, y_tr, oof):
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                         "font.size": 10})
    y_te = np.asarray(y_te)

    # 1 ROC
    fpr, tpr, _ = roc_curve(y_te, p_te)
    fig, ax = plt.subplots(figsize=(5, 4.2))
    ax.plot(fpr, tpr, color=RED, lw=2.2, label=f"{name}")
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC (test set)")
    ax.legend(); _save(fig, "roc.png")

    # 2 Calibration
    fig, ax = plt.subplots(figsize=(5, 4.2))
    for p, lab, col in ((p_raw, "raw", "#94a3b8"), (p_te, "calibrated", TEAL)):
        frac, mean = calibration_curve(y_te, p, n_bins=6, strategy="quantile")
        ax.plot(mean, frac, "o-", color=col, lw=2, label=lab)
    ax.plot([0, 1], [0, 1], "--", color="grey"); ax.legend()
    ax.set(xlabel="Predicted P(default)", ylabel="Observed default rate", title="Calibration (test)")
    _save(fig, "calibration.png")

    # 3 Cost vs threshold (OOF train curve + chosen threshold)
    grid = np.linspace(0.05, 0.95, 91)
    costs = [expected_cost(y_tr, oof, t) / len(y_tr) for t in grid]
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    ax.plot(grid, costs, color=INK, lw=2)
    ax.axvline(threshold, color=RED, ls="--", label=f"chosen = {threshold:.2f}")
    ax.set(xlabel="Reject if P(default) ≥ t", ylabel="Avg cost / applicant (FN = 5×FP)",
           title="Business-optimal threshold"); ax.legend(); _save(fig, "threshold_cost.png")

    # 4 Model leaderboard
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    lb = leaderboard.sort_values("cv_auc")
    ax.barh(lb.index, lb["cv_auc"], xerr=lb["cv_auc_std"], color=[GOLD if n == name else "#94a3b8" for n in lb.index])
    ax.set(xlim=(0.6, 0.85), xlabel="5-fold CV ROC-AUC", title="Model comparison"); _save(fig, "leaderboard.png")

    # 5 Score distribution by outcome
    sc = pd_to_score(p_te)
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    ax.hist(sc[y_te == 0], bins=20, alpha=.75, color=TEAL, label="Repaid")
    ax.hist(sc[y_te == 1], bins=20, alpha=.75, color=RED, label="Defaulted")
    ax.set(xlabel="Credit score", ylabel="Applicants", title="Score separation (test)"); ax.legend()
    _save(fig, "score_distribution.png")

    # 6 SHAP global importance
    pre, clf = pipe.named_steps["pre"], pipe.named_steps["clf"]
    Xt = pre.transform(X_te)
    sv = shap_values(make_explainer(clf, pre.transform(X_tr).sample(100, random_state=C.RANDOM_STATE)), Xt)
    imp = np.abs(sv).mean(0)
    top = np.argsort(imp)[-12:]
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.barh([pretty(Xt.columns[i]) for i in top], imp[top], color=RED)
    ax.set(xlabel="mean |SHAP|", title="What drives default risk"); _save(fig, "shap_importance.png")
    plt.figure(); shap.summary_plot(sv, Xt, show=False, max_display=12, plot_size=(7, 5))
    plt.tight_layout(); plt.savefig(C.FIG_DIR / "shap_beeswarm.png", dpi=140); plt.close()


def write_model_card(meta: dict, metrics: dict, fair: dict) -> None:
    t = metrics["test"]
    lo, hi = t["roc_auc_ci95"]
    rows = []
    for attr, groups in fair["attributes"].items():
        for g, r in groups.items():
            rows.append(f"| {attr} | {g} | {r['n']} | {r['approval_rate']:.0%} | "
                        f"{r['disparate_impact']:.2f} | {'pass' if r['passes_80_rule'] else 'FAIL'} |")
    card = f"""# Model Card: Credit Risk Scoring

**Version:** `{meta['version']}`  |  **Trained:** {meta['trained_at']}  |  **Data hash:** `{meta['data_sha256_12']}`
**Algorithm:** {meta['model_name']} + Platt calibration  |  **Train/Test:** {meta['n_train']}/{meta['n_test']}

## Intended use
Rank consumer-loan applicants by probability of default and support an approve/reject decision with reason codes.
**Not** for fully automated adverse decisions without human review. Demonstration model on a public 1,000-row dataset.

## Performance (held-out test set)
| Metric | Value |
|---|---|
| ROC-AUC | {t['roc_auc']:.3f} (95% bootstrap CI {lo:.3f} to {hi:.3f}) |
| Gini / KS | {t['gini']:.3f} / {t['ks']:.3f} |
| Brier (calibrated / raw) | {t['brier']:.3f} / {t['brier_uncalibrated']:.3f} |
| Threshold | P(default) >= {t['threshold']:.2f} -> reject (cost FN:FP = {C.COST_FN:.0f}:{C.COST_FP:.0f}) |
| Defaulter recall | {t['recall_defaulters']:.0%} |
| Cost reduction vs approve-all | {t['cost_reduction_pct']:.0f}% |

## Fairness
Protected attributes ({', '.join(meta['excluded_protected_features'])}) are **excluded from the model** and used only for auditing.
Rule: approval-rate ratio vs best group >= 0.80.

| Attribute | Group | n | Approval rate | Disparate impact | 4/5 rule |
|---|---|---|---|---|---|
""" + "\n".join(rows) + f"""

Small groups produce noisy ratios; treat results as indicative, not conclusive.

## Limitations
- 1,000 training rows; wide confidence intervals; one geography and era (German credit data).
- Cost ratio {C.COST_FN:.0f}:{C.COST_FP:.0f} is an assumption; set it from real lender economics.
- No reject-inference: the data only contains approved loans, so rejected applicants are unseen.
- Age is used as a feature and audited; confirm legal permissibility for your jurisdiction.

## Monitoring
Every prediction is logged; `python -m credit_risk.monitor` computes PSI vs the training profile
(<0.10 stable, 0.10-0.25 investigate, >0.25 retrain).
"""
    (C.REPORT_DIR / "MODEL_CARD.md").write_text(card)
