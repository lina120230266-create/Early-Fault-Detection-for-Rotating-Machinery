"""
prepare_for_tableau.py — Generate clean CSVs for Person C (Tableau)
Author: Person B — Data Processing & ML
Project: RUL Predictive Maintenance System

Generates:
    train_with_headers.csv  ← sensor readings + RUL labels (for degradation charts)

Usage:
    python prepare_for_tableau.py
"""

import pandas as pd
import numpy as np

# ── Column names (CMAPSS has no header row) ──────────────────
cols = (
    ["unit", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)

print("Loading train_FD001.txt...")
df = pd.read_csv("train_FD001.txt", sep=r"\s+", header=None, names=cols)

# ── Compute RUL (capped at 125) ──────────────────────────────
max_cycle    = df.groupby("unit")["cycle"].transform("max")
df["rul"]    = np.minimum(max_cycle - df["cycle"], 125)

# ── Add health status column ─────────────────────────────────
def health_status(rul):
    if rul > 80:  return "Healthy"
    elif rul > 40: return "Monitor"
    else:          return "Critical"

df["health_status"] = df["rul"].apply(health_status)

# ── Save ─────────────────────────────────────────────────────
df.to_csv("train_with_headers.csv", index=False)
print(f"✅ train_with_headers.csv saved ({len(df)} rows)")
print("\nSend to Person C:")
print("  - train_with_headers.csv  → degradation trend charts in Tableau")
print("  - predictions.csv         → RUL predictions dashboard")
