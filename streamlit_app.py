"""
streamlit_app.py — RUL Predictive Maintenance Dashboard
Author: Person B — Data Processing & ML
Project: RUL Predictive Maintenance System
Hosted on: Snowflake Streamlit

Live Demo:
https://app.snowflake.com/uae-north.azure/ea22229/#/streamlit-apps/RUL_DB.PUBLIC."RUL-Dashboard"
"""

import streamlit as st
import pandas as pd

session = st.connection("snowflake").session()

st.set_page_config(page_title="RUL Predictive Maintenance", layout="wide")
st.title("🔧 Predictive Maintenance — Engine RUL Dashboard")
st.markdown("**NASA C-MAPSS FD001 | Random Forest Model | RMSE: 18.87**")
st.divider()

# Load predictions
pred_df = session.sql("SELECT * FROM RUL_DB.PUBLIC.PREDICTIONS ORDER BY UNIT").to_pandas()

# ── Row 1: Key metrics ──────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Engines",     len(pred_df))
col2.metric("Avg Predicted RUL", f"{pred_df['PREDICTED_RUL'].mean():.1f} cycles")
col3.metric("Model RMSE",        "18.87")
col4.metric("🔴 Critical Engines", len(pred_df[pred_df['PREDICTED_RUL'] < 40]))

st.divider()

# ── Row 2: Engine inspector ─────────────────────────────────
st.subheader("🔍 Engine Inspector")
selected_unit = st.selectbox("Select Engine:", pred_df['UNIT'].tolist())

row       = pred_df[pred_df['UNIT'] == selected_unit].iloc[0]
predicted = row['PREDICTED_RUL']
actual    = row['ACTUAL_RUL']
error     = row['ABS_ERROR']

if predicted > 80:   status = "🟢 Healthy"
elif predicted > 40: status = "🟡 Monitor"
else:                status = "🔴 Critical"

c1, c2, c3, c4 = st.columns(4)
c1.metric("Predicted RUL",  f"{predicted:.1f} cycles")
c2.metric("Actual RUL",     f"{actual} cycles")
c3.metric("Absolute Error", f"{error:.1f} cycles")
c4.metric("Status",         status)

st.divider()

# ── Row 3: All engines bar chart ────────────────────────────
st.subheader("📊 All Engines — Predicted RUL")
st.bar_chart(pred_df.set_index('UNIT')['PREDICTED_RUL'])

st.divider()

# ── Row 4: Fleet health summary ─────────────────────────────
st.subheader("🏥 Fleet Health Summary")
h1, h2, h3 = st.columns(3)
healthy  = len(pred_df[pred_df['PREDICTED_RUL'] > 80])
monitor  = len(pred_df[(pred_df['PREDICTED_RUL'] >= 40) & (pred_df['PREDICTED_RUL'] <= 80)])
critical = len(pred_df[pred_df['PREDICTED_RUL'] < 40])
h1.metric("🟢 Healthy  (RUL > 80)",   f"{healthy} engines")
h2.metric("🟡 Monitor  (RUL 40–80)",  f"{monitor} engines")
h3.metric("🔴 Critical (RUL < 40)",   f"{critical} engines")

st.divider()

# ── Row 5: Full predictions table ───────────────────────────
st.subheader("📋 Full Predictions Table")
st.dataframe(
    pred_df[['UNIT','PREDICTED_RUL','ACTUAL_RUL','ABS_ERROR']].style.format({
        'PREDICTED_RUL': '{:.1f}',
        'ACTUAL_RUL':    '{:.0f}',
        'ABS_ERROR':     '{:.1f}'
    }),
    use_container_width=True
)
