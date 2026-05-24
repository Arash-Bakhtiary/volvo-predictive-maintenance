"""
Volvo Bus Predictive Maintenance — Fleet Dashboard App
Streamlit app connecting to Databricks SQL warehouse via connector.
"""
import os
import streamlit as st
import pandas as pd
from databricks import sql as dbsql

st.set_page_config(
    page_title="Volvo Fleet — Predictive Maintenance",
    page_icon="🚌",
    layout="wide",
)

WAREHOUSE_ID = os.environ.get("DATABRICKS_WAREHOUSE_ID", "5b1ca4f1d21b522a")
HOST         = os.environ.get("DATABRICKS_HOST", "").replace("https://", "")
TOKEN        = os.environ.get("DATABRICKS_TOKEN", "")


@st.cache_data(ttl=300)
def query(sql: str) -> pd.DataFrame:
    with dbsql.connect(
        server_hostname=HOST,
        http_path=f"/sql/1.0/warehouses/{WAREHOUSE_ID}",
        access_token=TOKEN,
    ) as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            return cur.fetchall_arrow().to_pandas()


# ── Header ────────────────────────────────────────────────────────────────────
st.title("🚌 Volvo Bus Fleet — Predictive Maintenance Dashboard")
st.caption("Data: `volvo_poc.gold` | Model: `bus_failure_predictor v2` | AUC-ROC 0.94")

# ── KPI Row ───────────────────────────────────────────────────────────────────
scores = query("SELECT * FROM volvo_poc.gold.fleet_failure_scores")
candidates = query("SELECT * FROM volvo_poc.gold.maintenance_candidates")

total_buses  = len(scores)
high_risk    = (scores["risk_tier"] == "HIGH").sum()
medium_risk  = (scores["risk_tier"] == "MEDIUM").sum()
avg_prob     = scores["failure_probability"].astype(float).mean()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Buses",       f"{total_buses}")
c2.metric("HIGH Risk",         f"{high_risk}",   delta=f"{high_risk/total_buses:.0%} of fleet", delta_color="inverse")
c3.metric("MEDIUM Risk",       f"{medium_risk}")
c4.metric("Avg Failure Prob",  f"{avg_prob:.1%}")

st.divider()

# ── Charts Row ────────────────────────────────────────────────────────────────
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Failure Probability by Bus Model")
    model_avg = (scores.groupby("bus_model")["failure_probability"]
                       .apply(lambda x: pd.to_numeric(x).mean())
                       .sort_values(ascending=False)
                       .reset_index())
    model_avg.columns = ["bus_model", "avg_failure_probability"]
    st.bar_chart(model_avg.set_index("bus_model"))

with col_right:
    st.subheader("Risk Tier Distribution")
    tier_counts = scores["risk_tier"].value_counts().reset_index()
    tier_counts.columns = ["risk_tier", "count"]
    st.bar_chart(tier_counts.set_index("risk_tier"))

st.divider()

# ── Maintenance Candidates Table ──────────────────────────────────────────────
st.subheader("🔴 Buses Requiring Immediate Action (HIGH Risk)")
high = scores[scores["risk_tier"] == "HIGH"].copy()
high["failure_probability"] = pd.to_numeric(high["failure_probability"]).round(3)
high = high.sort_values("failure_probability", ascending=False)
st.dataframe(
    high[["bus_id","bus_model","depot_region","failure_probability","recommended_action","scored_at"]],
    use_container_width=True, hide_index=True,
)

st.divider()

# ── Failure probability histogram ─────────────────────────────────────────────
st.subheader("Failure Probability Distribution Across Fleet")
scores["failure_probability"] = pd.to_numeric(scores["failure_probability"])
st.bar_chart(scores["failure_probability"].round(1).value_counts().sort_index())

st.caption("Refreshes every 5 minutes | Powered by Databricks Free Edition")
