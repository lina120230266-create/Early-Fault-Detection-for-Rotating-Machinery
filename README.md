# 🔧 Predictive Maintenance — Engine RUL Estimation

**Egypt Japan University of Science and Technology (E-JUST)**  
Data Engineering Capstone Project

---

## 📌 Project Overview

An end-to-end cloud data pipeline that predicts the **Remaining Useful Life (RUL)** of aircraft turbofan engines using the NASA C-MAPSS FD001 dataset.

> RUL = how many flight cycles does an engine have left before it needs maintenance?

---

## 🏗️ Architecture

```
NASA C-MAPSS Dataset
        ↓
Azure Blob Storage          ← Person A (Ingestion)
        ↓
Azure Data Factory (ETL)    ← Person A (Orchestration)
        ↓
Snowflake                   ← Person B (Processing + ML)
  → Raw tables
  → Clean tables
  → RUL labels
  → Feature engineering
  → ML_READY tables
        ↓
Python ML Model             ← Person B
  → Random Forest
  → XGBoost
  → predictions.csv
        ↓
Tableau + Streamlit         ← Person C (Visualization)
```

---

## 📁 Repository Structure

```
├── snowflake_pipeline.sql     # Full Snowflake processing pipeline (SQL)
├── train_rul_model.py         # ML model training + evaluation
├── upload.py                  # Upload files to Snowflake stage
├── prepare_for_tableau.py     # Generate clean CSVs for Tableau
└── README.md
```

---

## 📊 Dataset

**NASA C-MAPSS FD001**
- 100 training engines, 100 test engines
- 21 sensor measurements per cycle
- 1 operating condition, 1 fault mode (HPC degradation)
- Train rows: 20,631 | Test rows: 13,096

Download: [NASA Prognostics Data Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)

---

## ⚙️ Snowflake Pipeline (Person B — Phase 3)

### Tables created:
| Table | Rows | Description |
|---|---|---|
| RAW_TRAIN_FD001 | 20,631 | Raw training data, 26 columns |
| RAW_TEST_FD001 | 13,096 | Raw test data, 26 columns |
| RAW_RUL_FD001 | 100 | Ground truth RUL values |
| CLEAN_TRAIN_FD001 | 20,631 | 7 flat sensors removed |
| CLEAN_TEST_FD001 | 13,096 | 7 flat sensors removed |
| TRAIN_WITH_RUL | 20,631 | RUL labels added (cap=125) |
| ML_READY_TRAIN | 20,631 | Rolling avg features added |
| ML_READY_TEST | 13,096 | Rolling avg features added |
| PREDICTIONS | 100 | Final model predictions |

### Key processing steps:
1. **Sensor filtering** — dropped sensors 1, 5, 6, 10, 16, 18, 19 (near-zero variance)
2. **RUL computation** — `RUL = max_cycle - current_cycle`, capped at 125
3. **Rolling features** — 5-cycle trailing average per sensor per engine

---

## 🧠 Machine Learning (Person B — Phase 4)

### Models trained:
| Model | Validation RMSE | Test RMSE |
|---|---|---|
| Random Forest | 19.17 | **18.87** ✅ |
| XGBoost | 19.42 | — |

**Winner: Random Forest** (lower RMSE on validation set)

### Key decisions:
- **Group K-Fold CV** — split by engine unit, not by row (prevents data leakage)
- **RUL cap at 125** — early-life sensor readings carry no degradation signal
- **Rolling window = 5 cycles** — smooths noise, reveals trends

### Evaluation metrics:
- **RMSE: 18.87** (average prediction error ~19 cycles)
- **MAE: 13.73** (median prediction error ~14 cycles)
- **NASA S-score: 1009** (< 2000 = good; penalizes late predictions harder)

---

## 🚀 Live Demo

**Streamlit Dashboard (Snowflake):**
```
https://app.snowflake.com/uae-north.azure/ea22229/#/streamlit-apps/RUL_DB.PUBLIC."RUL-Dashboard"
```

Features:
- Engine health status (🟢 Healthy / 🟡 Monitor / 🔴 Critical)
- Per-engine RUL inspector (dropdown for all 100 engines)
- Fleet health summary
- Full predictions table

---

## 🛠️ Setup & Usage

### Prerequisites
```bash
pip install pandas numpy scikit-learn xgboost snowflake-connector-python matplotlib
pip install "snowflake-connector-python[pandas]"
```

### Run locally (no Snowflake needed)
```bash
python train_rul_model.py --source csv \
  --train_path train_FD001.txt \
  --test_path test_FD001.txt \
  --rul_path RUL_FD001.txt
```

### Upload to Snowflake
```bash
python upload.py
```

### Run with Snowflake connection
```bash
export SNOWFLAKE_PASSWORD=your_password
python train_rul_model.py --source snowflake
```

---

## 👥 Team

| Person | Role | Phases |
|---|---|---|
| Person A | Data Engineering & Infrastructure | 1, 2, 6 |
| Person B (Lina Tamer) | Data Processing & ML | 3, 4 |
| Person C | Visualization & Delivery | 5, 7 |

---

## 📚 References

- [NASA C-MAPSS Dataset](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)
- Saxena, A. et al. (2008). Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation. PHM 2008.
- [Snowflake Documentation](https://docs.snowflake.com)
- [Scikit-learn RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
