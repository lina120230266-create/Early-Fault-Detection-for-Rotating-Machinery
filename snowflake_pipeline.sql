/* ============================================================
   RUL Predictive Maintenance — Snowflake Processing Pipeline
   Dataset : NASA C-MAPSS FD001
   Author  : Person B — Data Processing & ML
   Project : End-to-End Predictive Maintenance System
   ============================================================

   PIPELINE STAGES:
   1. Environment setup (warehouse, database, schema)
   2. Raw landing tables
   3. File format + stage
   4. Load data from stage
   5. Clean tables (drop flat sensors)
   6. RUL label computation
   7. Feature engineering (rolling averages)
   8. Predictions write-back table
   9. Verification queries
   ============================================================ */


-- ============================================================
-- 1. ENVIRONMENT SETUP
-- ============================================================
CREATE WAREHOUSE IF NOT EXISTS RUL_WH
  WAREHOUSE_SIZE = 'XSMALL'
  AUTO_SUSPEND   = 60
  AUTO_RESUME    = TRUE;

CREATE DATABASE IF NOT EXISTS RUL_DB;
CREATE SCHEMA   IF NOT EXISTS RUL_DB.PUBLIC;

USE WAREHOUSE RUL_WH;
USE DATABASE  RUL_DB;
USE SCHEMA    PUBLIC;


-- ============================================================
-- 2. RAW LANDING TABLES
--    Receives data from Azure Blob via Azure Data Factory.
--    26 columns: unit, cycle, 3 op settings, 21 sensors.
-- ============================================================
CREATE OR REPLACE TABLE RAW_TRAIN_FD001 (
    unit         NUMBER,
    cycle        NUMBER,
    op_setting_1 FLOAT, op_setting_2 FLOAT, op_setting_3 FLOAT,
    sensor_1     FLOAT, sensor_2  FLOAT, sensor_3  FLOAT,
    sensor_4     FLOAT, sensor_5  FLOAT, sensor_6  FLOAT,
    sensor_7     FLOAT, sensor_8  FLOAT, sensor_9  FLOAT,
    sensor_10    FLOAT, sensor_11 FLOAT, sensor_12 FLOAT,
    sensor_13    FLOAT, sensor_14 FLOAT, sensor_15 FLOAT,
    sensor_16    FLOAT, sensor_17 FLOAT, sensor_18 FLOAT,
    sensor_19    FLOAT, sensor_20 FLOAT, sensor_21 FLOAT
);

CREATE OR REPLACE TABLE RAW_TEST_FD001 LIKE RAW_TRAIN_FD001;

CREATE OR REPLACE TABLE RAW_RUL_FD001 (
    true_rul NUMBER
);


-- ============================================================
-- 3. FILE FORMAT + INTERNAL STAGE
--    CMAPSS files are space-delimited with no header row.
-- ============================================================
CREATE OR REPLACE FILE FORMAT CMAPSS_FORMAT
  TYPE                          = 'CSV'
  FIELD_DELIMITER               = ' '
  SKIP_HEADER                   = 0
  TRIM_SPACE                    = TRUE
  ERROR_ON_COLUMN_COUNT_MISMATCH = FALSE;

CREATE OR REPLACE STAGE TURBOFAN_STAGE
  FILE_FORMAT = CMAPSS_FORMAT;

-- Upload files via Python upload.py, then verify:
-- LIST @TURBOFAN_STAGE;


-- ============================================================
-- 4. LOAD DATA FROM STAGE INTO RAW TABLES
-- ============================================================
COPY INTO RAW_TRAIN_FD001
  FROM @TURBOFAN_STAGE/train_FD001.txt
  FILE_FORMAT = (FORMAT_NAME = CMAPSS_FORMAT)
  FORCE = TRUE;

COPY INTO RAW_TEST_FD001
  FROM @TURBOFAN_STAGE/test_FD001.txt
  FILE_FORMAT = (FORMAT_NAME = CMAPSS_FORMAT)
  FORCE = TRUE;

COPY INTO RAW_RUL_FD001
  FROM @TURBOFAN_STAGE/RUL_FD001.txt
  FILE_FORMAT = (FORMAT_NAME = CMAPSS_FORMAT)
  FORCE = TRUE;


-- ============================================================
-- 5. SENSOR VARIANCE CHECK
--    Identify near-constant sensors to drop.
--    Sensors with STDDEV near 0 carry no predictive signal.
--    For FD001: sensors 1,5,6,10,16,18,19 are near-constant.
-- ============================================================
SELECT
    ROUND(STDDEV(sensor_1),4)  AS s1,  ROUND(STDDEV(sensor_2),4)  AS s2,
    ROUND(STDDEV(sensor_3),4)  AS s3,  ROUND(STDDEV(sensor_4),4)  AS s4,
    ROUND(STDDEV(sensor_5),4)  AS s5,  ROUND(STDDEV(sensor_6),4)  AS s6,
    ROUND(STDDEV(sensor_7),4)  AS s7,  ROUND(STDDEV(sensor_8),4)  AS s8,
    ROUND(STDDEV(sensor_9),4)  AS s9,  ROUND(STDDEV(sensor_10),4) AS s10,
    ROUND(STDDEV(sensor_11),4) AS s11, ROUND(STDDEV(sensor_12),4) AS s12,
    ROUND(STDDEV(sensor_13),4) AS s13, ROUND(STDDEV(sensor_14),4) AS s14,
    ROUND(STDDEV(sensor_15),4) AS s15, ROUND(STDDEV(sensor_16),4) AS s16,
    ROUND(STDDEV(sensor_17),4) AS s17, ROUND(STDDEV(sensor_18),4) AS s18,
    ROUND(STDDEV(sensor_19),4) AS s19, ROUND(STDDEV(sensor_20),4) AS s20,
    ROUND(STDDEV(sensor_21),4) AS s21
FROM RAW_TRAIN_FD001;


-- ============================================================
-- 6. CLEAN TABLES — DROP FLAT SENSORS
--    Keeping 14 informative sensors: 2,3,4,7,8,9,11,12,13,14,15,17,20,21
--    Dropping: 1,5,6,10,16,18,19 (near-zero variance in FD001)
-- ============================================================
CREATE OR REPLACE TABLE CLEAN_TRAIN_FD001 AS
SELECT
    unit, cycle,
    op_setting_1, op_setting_2, op_setting_3,
    sensor_2, sensor_3, sensor_4, sensor_7, sensor_8,
    sensor_9, sensor_11, sensor_12, sensor_13, sensor_14,
    sensor_15, sensor_17, sensor_20, sensor_21
FROM RAW_TRAIN_FD001;

CREATE OR REPLACE TABLE CLEAN_TEST_FD001 AS
SELECT
    unit, cycle,
    op_setting_1, op_setting_2, op_setting_3,
    sensor_2, sensor_3, sensor_4, sensor_7, sensor_8,
    sensor_9, sensor_11, sensor_12, sensor_13, sensor_14,
    sensor_15, sensor_17, sensor_20, sensor_21
FROM RAW_TEST_FD001;


-- ============================================================
-- 7. RUL LABEL COMPUTATION
--    RUL = max_cycle_for_unit - current_cycle
--    Capped at 125: early-life sensor readings are flat and
--    uninformative, so we don't try to distinguish RUL > 125.
-- ============================================================
CREATE OR REPLACE TABLE TRAIN_WITH_RUL AS
SELECT
    t.*,
    LEAST(m.max_cycle - t.cycle, 125) AS rul
FROM CLEAN_TRAIN_FD001 t
JOIN (
    SELECT unit, MAX(cycle) AS max_cycle
    FROM CLEAN_TRAIN_FD001
    GROUP BY unit
) m ON t.unit = m.unit;


-- ============================================================
-- 8. FEATURE ENGINEERING — ROLLING WINDOW AVERAGES
--    5-cycle trailing mean per sensor per engine.
--    Smooths noise and reveals degradation trends.
--    Window: current cycle + 4 preceding cycles.
-- ============================================================
CREATE OR REPLACE TABLE ML_READY_TRAIN AS
SELECT
    unit, cycle, rul,
    op_setting_1, op_setting_2, op_setting_3,
    sensor_2, sensor_3, sensor_4, sensor_7, sensor_8,
    sensor_9, sensor_11, sensor_12, sensor_13, sensor_14,
    sensor_15, sensor_17, sensor_20, sensor_21,
    AVG(sensor_2)  OVER (PARTITION BY unit ORDER BY cycle ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) AS s2_avg,
    AVG(sensor_3)  OVER (PARTITION BY unit ORDER BY cycle ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) AS s3_avg,
    AVG(sensor_4)  OVER (PARTITION BY unit ORDER BY cycle ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) AS s4_avg,
    AVG(sensor_7)  OVER (PARTITION BY unit ORDER BY cycle ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) AS s7_avg,
    AVG(sensor_8)  OVER (PARTITION BY unit ORDER BY cycle ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) AS s8_avg,
    AVG(sensor_9)  OVER (PARTITION BY unit ORDER BY cycle ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) AS s9_avg,
    AVG(sensor_11) OVER (PARTITION BY unit ORDER BY cycle ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) AS s11_avg,
    AVG(sensor_12) OVER (PARTITION BY unit ORDER BY cycle ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) AS s12_avg,
    AVG(sensor_13) OVER (PARTITION BY unit ORDER BY cycle ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) AS s13_avg,
    AVG(sensor_14) OVER (PARTITION BY unit ORDER BY cycle ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) AS s14_avg,
    AVG(sensor_15) OVER (PARTITION BY unit ORDER BY cycle ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) AS s15_avg,
    AVG(sensor_17) OVER (PARTITION BY unit ORDER BY cycle ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) AS s17_avg,
    AVG(sensor_20) OVER (PARTITION BY unit ORDER BY cycle ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) AS s20_avg,
    AVG(sensor_21) OVER (PARTITION BY unit ORDER BY cycle ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) AS s21_avg
FROM TRAIN_WITH_RUL;

CREATE OR REPLACE TABLE ML_READY_TEST AS
SELECT
    unit, cycle,
    op_setting_1, op_setting_2, op_setting_3,
    sensor_2, sensor_3, sensor_4, sensor_7, sensor_8,
    sensor_9, sensor_11, sensor_12, sensor_13, sensor_14,
    sensor_15, sensor_17, sensor_20, sensor_21,
    AVG(sensor_2)  OVER (PARTITION BY unit ORDER BY cycle ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) AS s2_avg,
    AVG(sensor_3)  OVER (PARTITION BY unit ORDER BY cycle ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) AS s3_avg,
    AVG(sensor_4)  OVER (PARTITION BY unit ORDER BY cycle ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) AS s4_avg,
    AVG(sensor_7)  OVER (PARTITION BY unit ORDER BY cycle ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) AS s7_avg,
    AVG(sensor_8)  OVER (PARTITION BY unit ORDER BY cycle ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) AS s8_avg,
    AVG(sensor_9)  OVER (PARTITION BY unit ORDER BY cycle ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) AS s9_avg,
    AVG(sensor_11) OVER (PARTITION BY unit ORDER BY cycle ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) AS s11_avg,
    AVG(sensor_12) OVER (PARTITION BY unit ORDER BY cycle ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) AS s12_avg,
    AVG(sensor_13) OVER (PARTITION BY unit ORDER BY cycle ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) AS s13_avg,
    AVG(sensor_14) OVER (PARTITION BY unit ORDER BY cycle ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) AS s14_avg,
    AVG(sensor_15) OVER (PARTITION BY unit ORDER BY cycle ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) AS s15_avg,
    AVG(sensor_17) OVER (PARTITION BY unit ORDER BY cycle ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) AS s17_avg,
    AVG(sensor_20) OVER (PARTITION BY unit ORDER BY cycle ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) AS s20_avg,
    AVG(sensor_21) OVER (PARTITION BY unit ORDER BY cycle ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) AS s21_avg
FROM CLEAN_TEST_FD001;


-- ============================================================
-- 9. PREDICTIONS TABLE
--    Populated by Python train_rul_model.py via upload.py
-- ============================================================
CREATE OR REPLACE TABLE PREDICTIONS (
    unit          NUMBER,
    predicted_rul FLOAT,
    actual_rul    NUMBER,
    model_name    VARCHAR,
    abs_error     FLOAT
);


-- ============================================================
-- 10. PERSON C ACCESS
--     Read-only Snowflake user for Tableau connection
-- ============================================================
CREATE USER IF NOT EXISTS PERSON_C
  PASSWORD            = 'ProjectRUL2026'
  DEFAULT_WAREHOUSE   = RUL_WH
  DEFAULT_NAMESPACE   = RUL_DB.PUBLIC
  DEFAULT_ROLE        = PUBLIC;

GRANT USAGE  ON WAREHOUSE RUL_WH                    TO USER PERSON_C;
GRANT USAGE  ON DATABASE  RUL_DB                    TO USER PERSON_C;
GRANT USAGE  ON SCHEMA    RUL_DB.PUBLIC             TO USER PERSON_C;
GRANT SELECT ON ALL TABLES IN SCHEMA RUL_DB.PUBLIC  TO USER PERSON_C;


-- ============================================================
-- 11. VERIFICATION QUERIES
-- ============================================================
-- Row counts
SELECT COUNT(*) AS raw_train   FROM RAW_TRAIN_FD001;   -- 20631
SELECT COUNT(*) AS raw_test    FROM RAW_TEST_FD001;    -- 13096
SELECT COUNT(*) AS raw_rul     FROM RAW_RUL_FD001;     -- 100
SELECT COUNT(*) AS clean_train FROM CLEAN_TRAIN_FD001; -- 20631
SELECT COUNT(*) AS ml_train    FROM ML_READY_TRAIN;    -- 20631
SELECT COUNT(*) AS ml_test     FROM ML_READY_TEST;     -- 13096
SELECT COUNT(*) AS predictions FROM PREDICTIONS;       -- 100

-- RUL distribution
SELECT MIN(rul)         AS min_rul,
       MAX(rul)         AS max_rul,
       ROUND(AVG(rul),2) AS avg_rul
FROM TRAIN_WITH_RUL;
-- Expected: 0 | 125 | ~86

-- Prediction accuracy
SELECT ROUND(AVG(ABS_ERROR),2) AS avg_error,
       ROUND(MIN(ABS_ERROR),2) AS best_error,
       ROUND(MAX(ABS_ERROR),2) AS worst_error
FROM PREDICTIONS;

-- Fleet health summary
SELECT
    COUNT(CASE WHEN PREDICTED_RUL > 80  THEN 1 END) AS healthy,
    COUNT(CASE WHEN PREDICTED_RUL BETWEEN 40 AND 80 THEN 1 END) AS monitor,
    COUNT(CASE WHEN PREDICTED_RUL < 40  THEN 1 END) AS critical
FROM PREDICTIONS;
