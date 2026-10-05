"""Credit Risk Studio - run: streamlit run app/streamlit_app.py"""
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt
import streamlit as st

from credit_risk import config as C
from credit_risk.predict import CreditScorer, ModelNotTrained

st.set_page_config(page_title="Credit Risk Studio", page_icon="🏦", layout="wide")

st.markdown("""
<style>
.stApp{background:#0b1020;color:#e5e7eb}
h1,h2,h3{letter-spacing:-.02em}
.hero{padding:28px 32px;border-radius:20px;margin-bottom:18px;
 background:linear-gradient(120deg,#e11d48 0%,#7c3aed 55%,#0ea5e9 100%)}
.hero h1{margin:0;color:#fff;font-size:2.3rem}.hero p{margin:4px 0 0;color:#ffe4ec}
.card{background:#141a30;border:1px solid #232b4a;border-radius:18px;padding:22px;text-align:center}
.big{font-size:3.6rem;font-weight:800;line-height:1}
.pill{display:inline-block;padding:6px 16px;border-radius:99px;font-weight:700;margin-top:10px}
.ok{background:#052e26;color:#34d399;border:1px solid #34d399}
.no{background:#3b0a18;color:#fb7185;border:1px solid #fb7185}
.lbl{color:#94a3b8;font-size:.8rem;text-transform:uppercase;letter-spacing:.12em}
</style>""", unsafe_allow_html=True)

st.markdown('<div class="hero"><h1>🏦 Credit Risk Studio</h1>'
            '<p>Default probability · 300-850 credit score · explainable decisions</p></div>',
            unsafe_allow_html=True)


@st.cache_resource
def load():
    return CreditScorer()


try:
    scorer = load()
except ModelNotTrained as e:
    st.error(str(e))
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(["🎯 Score an applicant", "📈 Model performance", "🔎 Data insights", "🛡️ Fairness & Monitoring"])
st.sidebar.markdown(f"**Model** `{scorer.model_name}`  \n**Version** `{scorer.version}`  \n**Trained** {scorer.trained_at[:10]}")

with tab1:
    d = scorer.sample_record()
    cats, rng = scorer.categories, scorer.numeric_ranges
    c1, c2, c3 = st.columns(3)
    rec = {}
    with c1:
        st.subheader("Loan")
        rec["duration"] = st.slider("Duration (months)", 4, 72, 24)
        rec["amount"] = st.number_input("Amount", 250, 50000, 5000, step=250)
        rec["installment_rate"] = st.slider("Installment rate (% of income)", 1, 4, 3)
        rec["purpose"] = st.selectbox("Purpose", cats["purpose"], index=cats["purpose"].index("car (new)"))
        rec["other_installment_plans"] = st.selectbox("Other installment plans", cats["other_installment_plans"], index=1)
    with c2:
        st.subheader("Applicant")
        rec["age"] = st.slider("Age", 18, 75, 32)
        rec["job"] = st.selectbox("Job", cats["job"], index=1)
        rec["employment_duration"] = st.selectbox("Employment duration", cats["employment_duration"], index=3)
        rec["housing"] = st.selectbox("Housing", cats["housing"], index=1)
        rec["present_residence"] = st.slider("Years at residence", 1, 4, 2)
    with c3:
        st.subheader("Financial standing")
        rec["status"] = st.selectbox("Checking account", cats["status"], index=2)
        rec["savings"] = st.selectbox("Savings", cats["savings"], index=0)
        rec["credit_history"] = st.selectbox("Credit history", cats["credit_history"], index=3)
        rec["property"] = st.selectbox("Property", cats["property"], index=1)
        rec["other_debtors"] = st.selectbox("Other debtors", cats["other_debtors"], index=2)
        rec["number_credits"] = st.slider("Existing credits", 1, 4, 1)
        rec["people_liable"] = st.slider("People liable", 1, 2, 1)
        rec["telephone"] = st.selectbox("Telephone", cats["telephone"], index=0)

    res = scorer.score(rec, top_k=8)
    color = "#34d399" if res["decision"] == "APPROVE" else "#fb7185"
    a, b, c = st.columns([1, 1, 1.4])
    a.markdown(f'<div class="card"><div class="lbl">Credit score</div>'
               f'<div class="big" style="color:{color}">{res["credit_score"]}</div>'
               f'<div class="lbl">{res["risk_band"]}</div></div>', unsafe_allow_html=True)
    b.markdown(f'<div class="card"><div class="lbl">Default probability</div>'
               f'<div class="big">{res["default_probability"]*100:.1f}%</div>'
               f'<span class="pill {"ok" if res["decision"]=="APPROVE" else "no"}">{res["decision"]}</span></div>',
               unsafe_allow_html=True)
    with c:
        fig, ax = plt.subplots(figsize=(5.4, 3.2), facecolor="#0b1020")
        r = res["reasons"][::-1]
        ax.barh([x["feature"][:38] for x in r], [x["impact"] for x in r],
                color=["#fb7185" if x["impact"] > 0 else "#34d399" for x in r])
        ax.set_facecolor("#0b1020"); ax.axvline(0, color="#64748b", lw=.8)
        ax.tick_params(colors="#cbd5e1", labelsize=8); ax.set_title("Why this score (SHAP)", color="#e5e7eb", fontsize=10)
        for s in ax.spines.values(): s.set_visible(False)
        buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=160, transparent=True, bbox_inches='tight')
        plt.close(fig); st.image(buf)
    st.caption(f"Reject threshold: P(default) ≥ {res['threshold']} · chosen to minimise cost where approving "
               "a defaulter costs 5× rejecting a good customer. Red bars raise risk, green bars lower it.")

with tab2:
    m = json.loads(C.METRICS_PATH.read_text())
    t = m["test"]
    k = st.columns(5)
    k[0].metric("ROC-AUC", t["roc_auc"]); k[1].metric("Gini", t["gini"]); k[2].metric("KS", t["ks"])
    k[3].metric("Defaulter recall", f'{t["recall_defaulters"]:.0%}')
    k[4].metric("Cost saved vs approve-all", f'{t["cost_reduction_pct"]:.0f}%')
    st.caption(f"Selected model: **{m['selected_model']}** · evaluated once on a held-out 20% test set.")
    g = st.columns(3)
    for col, f in zip(g * 2, ["roc", "calibration", "threshold_cost", "leaderboard",
                              "score_distribution", "shap_importance"], strict=False):
        col.image(str(C.FIG_DIR / f"{f}.png"), width='stretch')
    st.image(str(C.FIG_DIR / "shap_beeswarm.png"))

with tab3:
    g = st.columns(2)
    for i, f in enumerate(["eda_status", "eda_savings", "eda_credit_history", "eda_purpose"]):
        g[i % 2].image(str(C.FIG_DIR / f"{f}.png"), width='stretch')
    st.image(str(C.FIG_DIR / "eda_numeric.png"))

with tab4:
    import pandas as pd

    from credit_risk import monitor
    st.subheader("Fairness audit (held-out set)")
    st.caption("Protected attributes are NOT model inputs. This checks whether outcomes still differ by group (4/5 rule).")
    fair = json.loads(C.FAIRNESS_PATH.read_text())
    rows = [{"attribute": a, "group": g, "n": r["n"], "approval rate": f"{r['approval_rate']:.0%}",
             "disparate impact": round(r["disparate_impact"], 2), "4/5 rule": "✅" if r["passes_80_rule"] else "⚠️ review"}
            for a, grp in fair["attributes"].items() for g, r in grp.items()]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.subheader("Live drift monitor (PSI)")
    live = monitor.load_log()
    if len(live) < C.MIN_MONITOR_ROWS:
        st.info(f"{len(live)} logged predictions. Need {C.MIN_MONITOR_ROWS}+. Run `python scripts/simulate_traffic.py`.")
    else:
        rep = monitor.drift_report(live)
        st.metric("Overall status", rep["status"].upper(), f"score PSI {rep['score_psi']}")
        st.dataframe(pd.DataFrame([{"feature": k, **v} for k, v in rep["features"].items()]).head(10),
                     hide_index=True, width="stretch")
        st.caption(rep["action"])
