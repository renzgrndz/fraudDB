"""
app.py — Fraud Detection Streamlit Application
===============================================
Three-page dashboard:
  1. Fraud Analytics   — aggregate insights from PostgreSQL
  2. Model Performance — confusion matrix, ROC, feature importance
  3. Live Prediction   — single transaction risk scoring

Run:
    streamlit run app/app.py
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from sklearn.metrics import (
    RocCurveDisplay,
    PrecisionRecallDisplay,
    ConfusionMatrixDisplay,
    roc_auc_score,
    average_precision_score,
)

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
DB_URL        = os.getenv("DB_URL")
ARTIFACTS_DIR = Path(__file__).parent / "artifacts"

st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CACHED RESOURCES
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_engine():
    return create_engine(DB_URL, pool_pre_ping=True)


@st.cache_resource
def load_model():
    path = ARTIFACTS_DIR / "fraud_model.joblib"
    if path.exists():
        return joblib.load(path)
    return None


@st.cache_data(ttl=300)
def query_fraud_by_type():
    with get_engine().connect() as conn:
        return pd.read_sql(text("SELECT * FROM analytics.v_fraud_by_type"), conn)


@st.cache_data(ttl=300)
def query_fraud_trend():
    with get_engine().connect() as conn:
        return pd.read_sql(text("SELECT * FROM analytics.v_fraud_trend ORDER BY step"), conn)


@st.cache_data(ttl=300)
def query_risky_senders(limit: int = 20):
    with get_engine().connect() as conn:
        return pd.read_sql(
            text(f"SELECT * FROM analytics.v_risky_senders LIMIT {limit}"), conn
        )


@st.cache_data(ttl=600)
def query_ml_features_sample(n: int = 5000):
    """Load a stratified sample for model eval display."""
    with get_engine().connect() as conn:
        return pd.read_sql(
            text(f"""
                (SELECT * FROM analytics.mv_ml_features WHERE is_fraud = TRUE  LIMIT {n // 2})
                UNION ALL
                (SELECT * FROM analytics.mv_ml_features WHERE is_fraud = FALSE LIMIT {n // 2})
            """),
            conn,
        )


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "amount", "sender_balance_pre", "sender_balance_post",
    "recip_balance_pre", "recip_balance_post",
    "sender_balance_drop", "recipient_balance_gain", "dest_balance_mismatch",
    "sender_drained", "dest_was_empty", "sender_txns_same_step",
    "sender_amount_zscore", "sender_total_txns", "sender_avg_amount",
    "sender_unique_recipients",
]
TYPE_OPTIONS = ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"]


def build_input_vector(inputs: dict, feature_names: list) -> pd.DataFrame:
    """Build a single-row DataFrame matching training feature schema."""
    row = {f: 0.0 for f in feature_names}
    for k, v in inputs.items():
        if k in row:
            row[k] = v
    # One-hot type
    type_col = f"type_{inputs.get('type_name', 'PAYMENT')}"
    if type_col in row:
        row[type_col] = 1.0
    return pd.DataFrame([row])


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.title("🔍 Fraud Detection")
st.sidebar.markdown("Production fraud analytics & ML system.")
page = st.sidebar.radio(
    "Navigate",
    ["📊 Fraud Analytics", "🤖 Model Performance", "⚡ Live Prediction"],
)
st.sidebar.markdown("---")
st.sidebar.caption("Data: PaySim 100k | Model: Random Forest + LR")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1: FRAUD ANALYTICS DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
if page == "📊 Fraud Analytics":
    st.title("📊 Fraud Analytics Dashboard")

    # ── KPI Row
    try:
        fraud_type_df = query_fraud_by_type()
        trend_df      = query_fraud_trend()

        total_txns   = fraud_type_df["total_transactions"].sum()
        total_fraud  = fraud_type_df["fraud_count"].sum()
        total_volume = fraud_type_df["total_volume"].sum()
        fraud_volume = fraud_type_df["fraud_volume"].fillna(0).sum()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Transactions", f"{total_txns:,.0f}")
        col2.metric("Fraud Transactions", f"{total_fraud:,.0f}")
        col3.metric(
            "Fraud Rate",
            f"{total_fraud / total_txns * 100:.3f}%" if total_txns > 0 else "0.000%",
            help="% of transactions flagged as fraud"
        )
        col4.metric("Fraud Volume ($)", f"${fraud_volume:,.0f}")

        st.markdown("---")

        # ── Fraud Rate by Transaction Type
        st.subheader("Fraud Rate by Transaction Type")
        col_a, col_b = st.columns([2, 1])

        with col_a:
            fig, ax = plt.subplots(figsize=(8, 4))
            colors = ["#e74c3c" if r > 5 else "#3498db" for r in fraud_type_df["fraud_rate_pct"]]
            bars = ax.bar(fraud_type_df["type_name"], fraud_type_df["fraud_rate_pct"], color=colors)
            ax.set_ylabel("Fraud Rate (%)")
            ax.set_title("Fraud Rate by Transaction Type")
            ax.bar_label(bars, fmt="%.2f%%", padding=3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col_b:
            display_df = fraud_type_df[["type_name", "total_transactions", "fraud_count", "fraud_rate_pct"]].copy()
            display_df.columns = ["Type", "Total", "Fraud", "Fraud %"]
            st.dataframe(display_df, width="stretch", hide_index=True)

        st.markdown("---")

        # ── Fraud Trend Over Time
        st.subheader("Fraud Trend Over Time (by Step/Hour)")
        fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

        ax1.fill_between(trend_df["step"], trend_df["total_txns"],
                         alpha=0.3, color="#3498db", label="Total Txns")
        ax1.fill_between(trend_df["step"], trend_df["fraud_txns"],
                         alpha=0.7, color="#e74c3c", label="Fraud Txns")
        ax1.set_ylabel("Transaction Count")
        ax1.legend()
        ax1.set_title("Transaction Volume vs Fraud Volume")

        ax2.plot(trend_df["step"], trend_df["fraud_rate_pct"],
                 color="#e74c3c", linewidth=1.2)
        ax2.set_ylabel("Fraud Rate (%)")
        ax2.set_xlabel("Step (Hour)")
        ax2.set_title("Fraud Rate Over Time")

        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

        st.markdown("---")

        # ── Risky Senders
        st.subheader("⚠️ Top Risky Senders")
        risky_df = query_risky_senders(20)
        risky_display = risky_df[[
            "account_ref", "total_txn_count", "fraud_count",
            "fraud_rate", "total_amount", "unique_recipients"
        ]].copy()
        risky_display["fraud_rate"] = (risky_display["fraud_rate"] * 100).round(2).astype(str) + "%"
        risky_display["total_amount"] = risky_display["total_amount"].map("${:,.0f}".format)
        risky_display.columns = [
            "Account", "Total Txns", "Fraud Txns", "Fraud Rate", "Total Volume", "Unique Recipients"
        ]
        st.dataframe(risky_display, width="stretch", hide_index=True)

    except Exception as e:
        st.error(f"Database connection error: {e}")
        st.info("Ensure PostgreSQL is running and DB_URL is set correctly.")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2: MODEL PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🤖 Model Performance":
    st.title("🤖 Model Performance")

    artifact = load_model()
    if artifact is None:
        st.warning("No model artifact found. Run `python ml/train_model.py` first.")
        st.stop()

    pipeline      = artifact["pipeline"]
    feature_names = artifact["feature_names"]
    model_name    = artifact["model_name"]

    st.info(f"**Active model:** {model_name}")

    # Load evaluation data
    with st.spinner("Loading evaluation data from PostgreSQL ..."):
        try:
            df = query_ml_features_sample(5000)
        except Exception as e:
            st.error(f"Could not load data: {e}")
            st.stop()

    # Preprocess
    type_dummies = pd.get_dummies(df["type_name"], prefix="type")
    df = pd.concat([df, type_dummies], axis=1)
    all_features = feature_names
    df[all_features] = df[all_features].fillna(0)

    X = df[all_features].astype(float)
    y = df["is_fraud"].astype(int)

    y_proba = pipeline.predict_proba(X)[:, 1]
    y_pred  = pipeline.predict(X)

    roc_auc = roc_auc_score(y, y_proba)
    pr_auc  = average_precision_score(y, y_proba)

    # Metrics
    col1, col2 = st.columns(2)
    col1.metric("ROC-AUC", f"{roc_auc:.4f}")
    col2.metric("PR-AUC (primary)", f"{pr_auc:.4f}")

    st.markdown("---")

    # Plots
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay.from_predictions(
            y, y_pred, display_labels=["Legit", "Fraud"],
            ax=ax, colorbar=False, cmap="Blues"
        )
        st.pyplot(fig)
        plt.close()

    with col_right:
        st.subheader("ROC Curve")
        fig, ax = plt.subplots(figsize=(5, 4))
        RocCurveDisplay.from_predictions(y, y_proba, ax=ax)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
        st.pyplot(fig)
        plt.close()

    st.subheader("Precision-Recall Curve")
    fig, ax = plt.subplots(figsize=(10, 4))
    PrecisionRecallDisplay.from_predictions(y, y_proba, ax=ax)
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # Feature importance
    st.subheader("Feature Importance")
    model = pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        title = "Random Forest — Gini Importance"
    else:
        importances = np.abs(model.coef_[0])
        title = "Logistic Regression — |Coefficient|"

    n_show = min(15, len(importances))
    indices = np.argsort(importances)[::-1][:n_show]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(
        [feature_names[i] for i in indices[::-1]],
        importances[indices[::-1]],
        color="steelblue"
    )
    ax.set_title(title)
    ax.set_xlabel("Importance")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3: LIVE PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
elif page == "⚡ Live Prediction":
    st.title("⚡ Live Transaction Risk Scoring")

    artifact = load_model()
    if artifact is None:
        st.warning("No model artifact found. Run `python ml/train_model.py` first.")
        st.stop()

    pipeline      = artifact["pipeline"]
    feature_names = artifact["feature_names"]

    st.markdown("Enter transaction details below to get a real-time fraud probability score.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Transaction Details")
        txn_type          = st.selectbox("Transaction Type", TYPE_OPTIONS)
        amount            = st.number_input("Amount ($)", min_value=0.0, value=10000.0, step=100.0)
        sender_bal_pre    = st.number_input("Sender Balance (Before)", min_value=0.0, value=15000.0)
        sender_bal_post   = st.number_input("Sender Balance (After)", min_value=0.0, value=5000.0)
        recip_bal_pre     = st.number_input("Recipient Balance (Before)", min_value=0.0, value=0.0)
        recip_bal_post    = st.number_input("Recipient Balance (After)", min_value=0.0, value=10000.0)

    with col2:
        st.subheader("Sender Context")
        sender_total_txns       = st.slider("Sender Total Past Transactions", 1, 500, 10)
        sender_avg_amount       = st.number_input("Sender Avg Transaction Amount", min_value=0.0, value=5000.0)
        sender_unique_recip     = st.slider("Sender Unique Recipients (Lifetime)", 1, 100, 3)
        sender_txns_same_step   = st.slider("Sender Transactions This Hour", 1, 20, 1)
        sender_amount_zscore    = st.slider("Amount Z-Score vs Sender History", -5.0, 10.0, 0.5)

    st.markdown("---")
    threshold = st.slider(
        "🎚️ Risk Threshold",
        min_value=0.01, max_value=0.99, value=0.50, step=0.01,
        help="Lower threshold = catch more fraud, more false alarms. Higher = fewer alarms, more misses."
    )

    if st.button("🔍 Score Transaction", type="primary"):
        sender_balance_drop     = sender_bal_pre - sender_bal_post
        recipient_balance_gain  = recip_bal_post - recip_bal_pre
        dest_balance_mismatch   = abs(amount - recipient_balance_gain)
        sender_drained          = 1 if sender_bal_post == 0 else 0
        dest_was_empty          = 1 if recip_bal_pre == 0 else 0

        inputs = {
            "amount":                  amount,
            "sender_balance_pre":      sender_bal_pre,
            "sender_balance_post":     sender_bal_post,
            "recip_balance_pre":       recip_bal_pre,
            "recip_balance_post":      recip_bal_post,
            "sender_balance_drop":     sender_balance_drop,
            "recipient_balance_gain":  recipient_balance_gain,
            "dest_balance_mismatch":   dest_balance_mismatch,
            "sender_drained":          sender_drained,
            "dest_was_empty":          dest_was_empty,
            "sender_txns_same_step":   sender_txns_same_step,
            "sender_amount_zscore":    sender_amount_zscore,
            "sender_total_txns":       sender_total_txns,
            "sender_avg_amount":       sender_avg_amount,
            "sender_unique_recipients": sender_unique_recip,
            "type_name":               txn_type,
        }

        X_input = build_input_vector(inputs, feature_names)
        proba   = pipeline.predict_proba(X_input)[0, 1]
        label   = "🚨 FRAUD" if proba >= threshold else "✅ LEGITIMATE"
        color   = "red" if proba >= threshold else "green"

        st.markdown("---")
        st.subheader("Prediction Result")

        col_r1, col_r2, col_r3 = st.columns(3)
        col_r1.metric("Fraud Probability", f"{proba:.4f}")
        col_r2.metric("Threshold", f"{threshold:.2f}")
        col_r3.metric("Decision", label)

        # Probability gauge
        fig, ax = plt.subplots(figsize=(8, 1.5))
        ax.barh(["Risk"], [proba], color=color, height=0.5)
        ax.barh(["Risk"], [1 - proba], left=[proba], color="#ecf0f1", height=0.5)
        ax.axvline(threshold, color="orange", linewidth=2, linestyle="--", label=f"Threshold={threshold}")
        ax.set_xlim(0, 1)
        ax.set_xlabel("Fraud Probability")
        ax.legend(loc="upper right")
        ax.set_title(f"Transaction Risk Score: {proba:.2%}")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Key risk signals
        st.subheader("Risk Signal Breakdown")
        signals = {
            "Sender balance drained to $0":     sender_drained == 1,
            "Recipient was empty account":       dest_was_empty == 1,
            "Balance mismatch detected":         dest_balance_mismatch > amount * 0.05,
            "High velocity (multiple txns/hour)": sender_txns_same_step > 3,
            "Amount anomalous vs history":        sender_amount_zscore > 2.5,
        }
        for signal, triggered in signals.items():
            icon = "🔴" if triggered else "🟢"
            st.markdown(f"{icon} {signal}")
