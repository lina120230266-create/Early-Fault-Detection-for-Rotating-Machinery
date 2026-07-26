"""
upload.py — Upload NASA C-MAPSS files to Snowflake Stage
Author: Person B — Data Processing & ML
Project: RUL Predictive Maintenance System

Usage:
    python upload.py
"""

import snowflake.connector
import os

# ── Connection ──────────────────────────────────────────────
conn = snowflake.connector.connect(
    user      = 'LINATAMER',
    password  = os.environ.get('SNOWFLAKE_PASSWORD', 'YOUR_PASSWORD_HERE'),
    account   = 'clzwxvn-wy35353',
    warehouse = 'RUL_WH',
    database  = 'RUL_DB',
    schema    = 'PUBLIC'
)

cs = conn.cursor()

# ── Upload raw .txt files ────────────────────────────────────
BASE_PATH = r'C:/Users/lina2/OneDrive/Desktop/RUL_Project'

files = [
    'train_FD001.txt',
    'test_FD001.txt',
    'RUL_FD001.txt'
]

print("Uploading NASA C-MAPSS files to Snowflake stage...\n")
for f in files:
    full_path = f"{BASE_PATH}/{f}"
    print(f"Uploading {f}...")
    cs.execute(f"PUT file://{full_path} @TURBOFAN_STAGE AUTO_COMPRESS=FALSE OVERWRITE=TRUE")
    print(f"  ✅ {f} uploaded\n")

print("All files uploaded!")
print("Run LIST @TURBOFAN_STAGE in Snowflake to verify.\n")

# ── Upload predictions.csv back to Snowflake ─────────────────
from snowflake.connector.pandas_tools import write_pandas
import pandas as pd

pred_path = f"{BASE_PATH}/predictions.csv"
if os.path.exists(pred_path):
    print("Uploading predictions.csv to PREDICTIONS table...")
    pred_df = pd.read_csv(pred_path)
    pred_df.columns = [c.upper() for c in pred_df.columns]
    write_pandas(conn, pred_df, 'PREDICTIONS')
    print(f"  ✅ {len(pred_df)} predictions loaded into Snowflake!\n")
else:
    print("predictions.csv not found — run train_rul_model.py first.")

conn.close()
print("Done!")
