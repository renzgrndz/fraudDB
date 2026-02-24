# 🔍 Fraud Detection System
> A production-style fraud detection pipeline built with PostgreSQL, scikit-learn, and Streamlit.

---

## 📌 Project Overview

This system detects fraudulent financial transactions using a full end-to-end data pipeline — from raw data ingestion to a live prediction interface. It was built as a final SQL project and extended into data engineering, machine learning, and a web application.

**Dataset:** PaySim — a synthetic mobile money transaction simulator (sampled to 100,000 rows)  
**Fraud Rate:** ~0.13% (highly imbalanced — this shapes every design decision)

---

## 🏗️ Architecture

```
Raw CSV
   │
   ▼
┌─────────────────────────────────────────┐
│           PostgreSQL Database           │
│                                         │
│  staging.*       ← raw ingestion        │
│  core.*          ← normalized model     │
│  analytics.*     ← features & views     │
└─────────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────────────┐
│         ML Pipeline (Python)            │
│                                         │
│  Load features → Train → Evaluate       │
│  Save artifact (fraud_model.joblib)     │
└─────────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────────────┐
│         Streamlit Application           │
│                                         │
│  Analytics Dashboard                    │
│  Model Performance                      │
│  Live Prediction Interface              │
└─────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
FDS_R1/
├── artifacts/                  # Saved model files (auto-created)
│   ├── fraud_model.joblib      # Best model artifact
│   ├── lr_pipeline.joblib      # Logistic Regression pipeline
│   └── rf_pipeline.joblib      # Random Forest pipeline
├── figures/                    # Evaluation plots (auto-created)
├── app.py                      # Streamlit web application
├── train_model.py              # ML training pipeline
├── schema.sql                  # Database schema (3-layer)
├── sample_data_load.sql        # ETL: staging → core promotion
├── queries.sql                 # Analytical & validation queries
├── paysim_100k.csv             # Sampled dataset (100k rows)
├── requirements.txt            # Python dependencies
├── .env                        # Database credentials
└── README.md                   # This file
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.10+
- PostgreSQL 14+
- pgAdmin 4
- PaySim dataset from [Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Database Credentials

Create a `.env` file in your project root:

```
DB_URL=postgresql://postgres:yourpassword@127.0.0.1:5432/fraud_db
```

### 3. Create the Database

In pgAdmin Query Tool, connected to `fraud_db`, open and run `schema.sql`.

### 4. Prepare the Dataset

Sample the full PaySim CSV down to 100,000 rows:

```python
import pandas as pd
df = pd.read_csv("PS_20174392719_1491204439457_log.csv").sample(100_000, random_state=42)
df.to_csv("paysim_100k.csv", index=False)
```

### 5. Load Data into PostgreSQL

In pgAdmin: right-click `staging.raw_transactions` → **Import/Export Data**
- Format: `csv`
- Header: **ON**
- File: path to `paysim_100k.csv`
- Columns: `step, type, amount, name_orig, old_balance_orig, new_balance_orig, name_dest, old_balance_dest, new_balance_dest, is_fraud, is_flagged_fraud`

Then promote staging → core by running `sample_data_load.sql` in pgAdmin Query Tool.

### 6. Verify Data Load

```sql
SELECT COUNT(*) FROM staging.raw_transactions;   -- 100,000
SELECT COUNT(*) FROM core.accounts;              -- ~6,000
SELECT COUNT(*) FROM core.transactions;          -- 100,000
SELECT COUNT(*) FROM analytics.mv_ml_features;  -- 100,000
```

### 7. Train the Model

```bash
python train_model.py
```

Training produces ROC-AUC and PR-AUC scores for both models, a classification report, saved artifacts in `artifacts/`, and evaluation plots in `figures/`.

### 8. Launch the App

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## 🗄️ Database Layer

### 3-Schema Design

| Schema | Purpose | Objects |
|---|---|---|
| `staging` | Raw CSV ingestion, no transformation | `raw_transactions` |
| `core` | Normalized 3NF model with constraints | `transactions`, `accounts`, `transaction_types` |
| `analytics` | Feature layer for ML and dashboards | `mv_ml_features`, `mv_sender_features`, views |

### Why Normalization?

The raw CSV stores "TRANSFER" as text in every row. The normalized model stores an integer ID instead, with a lookup table. This reduces storage, enforces consistency, and enables referential integrity — the database physically cannot contain a transaction pointing to a non-existent account.

### Constraints Applied

```sql
amount >= 0                          -- no negative transactions
account_type IN ('C', 'M')           -- only valid account types
sender_id → core.accounts            -- foreign key
recipient_id → core.accounts         -- foreign key
type_id → core.transaction_types     -- foreign key
```

### Indexing Strategy

| Index | Columns | Purpose |
|---|---|---|
| `idx_txn_sender` | `sender_id` | Sender behavioral aggregations |
| `idx_txn_type_fraud` | `(type_id, is_fraud)` | Fraud-by-type dashboard queries |
| `idx_txn_sender_step` | `(sender_id, step)` | Temporal window functions |
| `idx_accounts_ref` | `account_ref` | ETL normalization lookups |

---

## 🧮 Feature Engineering

All features are computed in SQL using window functions and stored in `analytics.mv_ml_features`.

| Feature | Type | Fraud Signal |
|---|---|---|
| `sender_balance_drop` | Numeric | Large drop → account draining |
| `dest_balance_mismatch` | Numeric | Amount ≠ balance change → layering |
| `sender_drained` | Binary | Balance hits $0 → full drain fraud |
| `dest_was_empty` | Binary | Recipient was empty → mule account |
| `sender_amount_zscore` | Numeric | Unusual amount vs personal history |
| `sender_txns_same_step` | Numeric | High velocity → automated fraud |
| `sender_unique_recipients` | Numeric | Fan-out pattern → money scattering |
| `sender_historical_fraud_rate` | Numeric | Past fraud rate by this sender |

### Window Function Example

```sql
-- Z-score: how unusual is this amount for this specific sender?
(amount - AVG(amount) OVER (PARTITION BY sender_id))
/ NULLIF(STDDEV(amount) OVER (PARTITION BY sender_id), 0)
AS sender_amount_zscore
```

---

## 🤖 Machine Learning

### Why Not Accuracy?

With a 0.13% fraud rate, a model predicting "legitimate" for every transaction achieves **99.87% accuracy** while catching zero fraud. Accuracy is meaningless on imbalanced data.

### Metrics We Use

| Metric | What It Measures |
|---|---|
| **Precision** | Of flagged fraud, how many were real? (false alarm cost) |
| **Recall** | Of real fraud, how many did we catch? (missed fraud cost) |
| **ROC-AUC** | Discrimination ability across all thresholds |
| **PR-AUC** | Primary metric — precision-recall tradeoff on positive class only |

### Class Imbalance Strategy

- **SMOTE** — generates synthetic fraud examples by interpolating between real ones
- **class_weight="balanced"** — increases training penalty for missing fraud
- **Stratified split** — preserves fraud rate in both train and test sets

### Models Trained

**Logistic Regression** — fast, interpretable, well-calibrated probabilities. Cannot capture nonlinear feature interactions. Best for explainability.

**Random Forest** — captures nonlinear interactions, robust to outliers, native feature importance. Best for raw performance.

The model with higher PR-AUC is automatically saved as the production artifact.

### Threshold Tuning

The model outputs a probability between 0 and 1. The decision threshold is adjustable:

- **Lower threshold** → higher recall, more false alarms
- **Higher threshold** → higher precision, more missed fraud

The Streamlit app includes a live slider to tune this based on business priorities.

---

## 📊 Streamlit Application

### Page 1 — Fraud Analytics Dashboard
- KPI metrics: total transactions, fraud count, fraud rate, fraud volume
- Fraud rate by transaction type (bar chart)
- Fraud trend over time (time series)
- Top risky senders ranking

### Page 2 — Model Performance
- ROC-AUC and PR-AUC scores
- Confusion matrix, ROC curve, Precision-Recall curve
- Feature importance chart

### Page 3 — Live Prediction
- Input form for transaction details and sender context
- Real-time fraud probability score
- Risk threshold slider
- Risk signal breakdown showing which factors triggered

### Caching Strategy

```python
@st.cache_resource        # DB engine + model — initialized once
@st.cache_data(ttl=300)   # Query results — refreshed every 5 minutes
```

---

## 🔬 Key SQL Queries

### Fraud Rate by Transaction Type
```sql
SELECT * FROM analytics.v_fraud_by_type;
```

### Fraud Trend Over Time
```sql
SELECT * FROM analytics.v_fraud_trend ORDER BY step;
```

### Top Risky Senders
```sql
SELECT * FROM analytics.v_risky_senders LIMIT 20;
```

### Anomalous Transactions (3+ std deviations above sender mean)
```sql
WITH sender_stats AS (
    SELECT sender_id, AVG(amount) AS mean, STDDEV(amount) AS std
    FROM core.transactions GROUP BY sender_id
)
SELECT t.transaction_id, a.account_ref, t.amount,
       ROUND((t.amount - ss.mean) / NULLIF(ss.std, 0), 2) AS z_score,
       t.is_fraud
FROM core.transactions t
JOIN sender_stats ss ON t.sender_id = ss.sender_id
JOIN core.accounts a ON t.sender_id = a.account_id
WHERE (t.amount - ss.mean) / NULLIF(ss.std, 0) > 3
ORDER BY z_score DESC;
```

---

## 🚀 Scaling Roadmap

| Phase | Stack |
|---|---|
| **Now (batch)** | PostgreSQL + scikit-learn + Streamlit |
| **Phase 2** | dbt for feature pipeline + MLflow for experiment tracking |
| **Phase 3** | Kafka ingestion + Flink for real-time feature computation |
| **Phase 4** | Feature store (Feast) + model registry + CI/CD deployment |

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `psycopg2-binary` | PostgreSQL driver |
| `sqlalchemy` | Database connection management |
| `pandas` | Data manipulation |
| `scikit-learn` | ML models and evaluation |
| `imbalanced-learn` | SMOTE for class imbalance |
| `joblib` | Model serialization |
| `matplotlib` | Visualization |
| `streamlit` | Web application |
| `python-dotenv` | Environment variable management |

---

## 👤 Author

**Renz Granadozo**  
Final Project — SQL & Data Engineering  
Built with PostgreSQL · scikit-learn · Streamlit
