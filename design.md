# 🏗️ System Design — Fraud Detection System

## Architecture Overview

```
FDS_R1/
├── app.py                  ← Streamlit entry point (root for easy run)
├── schema.sql              ← Database schema (root for easy access)
├── queries.sql             ← Analytical queries (root for easy access)
├── design.md               ← This file
│
├── data/                   ← Raw and sampled datasets
├── ml/                     ← Machine learning pipeline
├── sql/                    ← ETL and data loading scripts
├── artifacts/              ← Trained model files
├── figures/                ← Evaluation plots
└── docs/                   ← Documentation
```

---

## Database Design (3-Layer Schema)

### Why 3 Schemas?

| Layer | Schema | Responsibility |
|---|---|---|
| Ingestion | `staging` | Land raw CSV exactly as-is. No transformation. Preserves source fidelity. |
| Storage | `core` | Normalized 3NF tables. FK constraints. Business rules enforced here. |
| Serving | `analytics` | Views and materialized views. Decouples query logic from storage. |

Each layer has one job. If the ETL breaks, staging is untouched and reprocessable. If the ML feature set changes, only the analytics layer is modified — core stays stable.

### Normalization Decisions

- `transaction_types` extracted to a lookup table — "TRANSFER" stored once, not 30,000 times
- `accounts` unified from both sender and recipient columns — single source of truth for account data
- `transactions` references both by FK — referential integrity enforced at DB level, not application level

### Materialized Views vs Regular Views

`analytics.mv_ml_features` is materialized because it computes window functions over 100,000 rows. A regular view would recompute this on every ML training run (~seconds of overhead). The materialized view computes once and is refreshed explicitly after new data loads.

---

## ML Design Decisions

### Why PR-AUC as Primary Metric?

Fraud rate is ~0.13%. ROC-AUC is optimistic on imbalanced data because it includes the massive true-negative denominator. PR-AUC focuses only on the positive (fraud) class — it directly measures what we care about: how well the model ranks fraud above legitimate transactions.

### Why SMOTE + class_weight Together?

SMOTE generates new synthetic fraud samples by interpolating between existing ones — giving the model more signal to learn from. `class_weight="balanced"` additionally penalizes misclassifying fraud during training. Both together produce significantly better recall than either alone.

### Why Two Models?

Logistic Regression is the interpretable baseline — coefficients directly show feature influence, probabilities are well-calibrated, and it's fast to retrain. Random Forest captures nonlinear interactions that LR misses (e.g. sender_drained AND dest_was_empty together). Both are trained; the one with higher PR-AUC is saved as the production artifact.

### Pipeline Object

Wrapping preprocessing + model in a single `Pipeline` object prevents data leakage. SMOTE runs only during `fit()`, not `predict_proba()`. Scaling is fit on training data and applied consistently at inference. The entire inference logic ships as one portable `.joblib` file.

---

## Application Design Decisions

### Why Streamlit?

Streamlit turns a Python script into a web app with minimal boilerplate. For a project where the audience is analysts and stakeholders (not developers), it delivers a working UI without building a separate frontend. The tradeoff is that it reruns the entire script on every interaction — solved by caching.

### Caching Strategy

```python
@st.cache_resource   # Engine + model: expensive to init, stateless once created
@st.cache_data(ttl=300)  # Query results: fresh enough at 5-minute TTL
```

Without caching, every slider move would requery PostgreSQL and reload the model — making the app unusable.

### Threshold Slider

The model outputs P(fraud) between 0 and 1. The threshold determines the operating point on the precision-recall curve. This is exposed as a slider because the right threshold depends on business context — a bank minimizing losses sets it low (high recall), a bank minimizing customer complaints sets it high (high precision).

---

## Folder Structure Rationale

| Location | Reason |
|---|---|
| `app.py` at root | `streamlit run app.py` works from project root without path flags |
| `schema.sql` at root | Run directly in pgAdmin without navigating subfolders |
| `queries.sql` at root | Frequently referenced during development and presentation |
| `ml/train_model.py` | Separates ML concerns from app and DB layers |
| `sql/sample_data_load.sql` | Groups ETL scripts separate from schema definition |
| `data/` | Keeps raw files out of root, gitignored in production |
| `artifacts/` | Model outputs separate from source code |
| `docs/` | Documentation separate from executable code |
