"""
train_model.py — Fraud Detection ML Pipeline
=============================================
Loads features from PostgreSQL analytics layer, trains Logistic Regression
and Random Forest classifiers, handles class imbalance, evaluates with
fraud-appropriate metrics, and saves model artifacts.

Usage:
    python ml/train_model.py

Environment variables (or .env file):
    DB_URL = postgresql://user:password@localhost:5432/fraud_db
"""

import os
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,  # PR-AUC
    RocCurveDisplay,
    PrecisionRecallDisplay,
    ConfusionMatrixDisplay,
)
from sklearn.inspection import permutation_importance
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

DB_URL       = os.getenv("DB_URL")
ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"
FIGURES_DIR   = Path(__file__).parent.parent / "figures"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Features used for training
# Selected for fraud detection signal and low collinearity
FEATURE_COLS = [
    "amount",
    "sender_balance_pre",
    "sender_balance_post",
    "recip_balance_pre",
    "recip_balance_post",
    "sender_balance_drop",       # pre - post for sender
    "recipient_balance_gain",    # post - pre for recipient
    "dest_balance_mismatch",     # |amount - recipient gain| → laundering signal
    "sender_drained",            # binary: sender balance hit 0
    "dest_was_empty",            # binary: recipient had 0 balance before
    "sender_txns_same_step",     # velocity: txns in same hour
    "sender_amount_zscore",      # how unusual is this amount for this sender
    "sender_total_txns",         # sender history volume
    "sender_avg_amount",         # sender baseline
    "sender_unique_recipients",  # fan-out: many recipients = suspicious
]

TARGET_COL = "is_fraud"

# One-hot encoded transaction types (added during preprocessing)
TYPE_DUMMIES = ["type_TRANSFER", "type_CASH_OUT", "type_PAYMENT", "type_CASH_IN", "type_DEBIT"]


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
def load_features(engine) -> pd.DataFrame:
    """Load ML feature set from PostgreSQL analytics materialized view."""
    log.info("Loading features from analytics.mv_ml_features ...")
    query = text("""
        SELECT
            transaction_id,
            type_name,
            amount,
            sender_balance_pre,
            sender_balance_post,
            recip_balance_pre,
            recip_balance_post,
            sender_balance_drop,
            recipient_balance_gain,
            dest_balance_mismatch,
            sender_drained,
            dest_was_empty,
            sender_txns_same_step,
            sender_amount_zscore,
            sender_total_txns,
            sender_avg_amount,
            sender_unique_recipients,
            is_fraud
        FROM analytics.mv_ml_features
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    log.info(f"Loaded {len(df):,} rows. Fraud rate: {df['is_fraud'].mean():.4%}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
def preprocess(df: pd.DataFrame):
    """
    - One-hot encode transaction type
    - Fill nulls (lag features are null for first sender txn)
    - Return X, y, feature_names
    """
    # One-hot encode transaction type
    type_dummies = pd.get_dummies(df["type_name"], prefix="type")
    df = pd.concat([df, type_dummies], axis=1)

    # Collect all feature columns present
    all_features = FEATURE_COLS + [c for c in type_dummies.columns]

    # Fill NaN in lag/window features with 0 (first occurrence per sender)
    df[all_features] = df[all_features].fillna(0)

    X = df[all_features].astype(float)
    y = df[TARGET_COL].astype(int)

    log.info(f"Feature matrix: {X.shape}, Class balance: {y.value_counts().to_dict()}")
    return X, y, all_features


# ─────────────────────────────────────────────────────────────────────────────
# MODEL TRAINING
# ─────────────────────────────────────────────────────────────────────────────
def build_lr_pipeline():
    """
    Logistic Regression with SMOTE oversampling.
    
    WHY LOGISTIC REGRESSION:
    - Fast, interpretable coefficients
    - Good baseline for probability calibration
    - Works well with scaled features
    - Gives P(fraud) not just class, enabling threshold tuning
    
    WHY SMOTE over class_weight:
    - Generates synthetic minority samples vs just reweighting
    - Better recall on highly imbalanced data (~0.1% fraud rate)
    """
    return ImbPipeline([
        ("smote",    SMOTE(random_state=42, k_neighbors=5)),
        ("scaler",   StandardScaler()),
        ("model",    LogisticRegression(
            max_iter=1000,
            C=0.1,              # stronger regularization for high-dim
            solver="lbfgs",
            class_weight="balanced",  # double insurance with SMOTE
            random_state=42,
        )),
    ])


def build_rf_pipeline():
    """
    Random Forest — no scaling needed, handles nonlinearity.
    
    WHY RANDOM FOREST:
    - Captures nonlinear interactions (e.g., sender_drained AND dest_was_empty)
    - Native feature importance
    - Robust to outliers (tree-based)
    - Handles mixed scales without normalization
    - Typically outperforms LR on tabular fraud data
    
    Tradeoff: slower training, less interpretable than LR.
    """
    return ImbPipeline([
        ("smote",  SMOTE(random_state=42, k_neighbors=5)),
        ("model",  RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42,
        )),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(name: str, pipeline, X_test, y_test):
    """
    WHY NOT ACCURACY:
    With 0.1% fraud rate, a model predicting ALL transactions as legitimate
    achieves 99.9% accuracy — useless. We care about:
    
    - Precision: of all flagged frauds, how many are real? (false alarm cost)
    - Recall:    of all real frauds, how many did we catch? (missed fraud cost)
    - ROC-AUC:   discrimination ability across all thresholds
    - PR-AUC:    precision-recall area — more informative than ROC on imbalanced data
    """
    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc  = average_precision_score(y_test, y_proba)

    log.info(f"\n{'='*50}")
    log.info(f"Model: {name}")
    log.info(f"ROC-AUC:  {roc_auc:.4f}")
    log.info(f"PR-AUC:   {pr_auc:.4f}  ← primary metric for fraud")
    log.info(f"\n{classification_report(y_test, y_pred, target_names=['Legit', 'Fraud'])}")

    return {"roc_auc": roc_auc, "pr_auc": pr_auc, "y_proba": y_proba, "y_pred": y_pred}


def plot_results(name: str, pipeline, X_test, y_test, feature_names: list):
    """Generate confusion matrix, ROC, PR curve, and feature importance plots."""
    safe_name = name.lower().replace(" ", "_")
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred  = pipeline.predict(X_test)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"{name} — Evaluation", fontsize=14)

    # Confusion matrix
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=["Legit", "Fraud"],
        ax=axes[0], colorbar=False, cmap="Blues"
    )
    axes[0].set_title("Confusion Matrix")

    # ROC curve
    RocCurveDisplay.from_predictions(y_test, y_proba, ax=axes[1])
    axes[1].set_title("ROC Curve")
    axes[1].plot([0,1],[0,1], "k--", alpha=0.4)

    # PR curve
    PrecisionRecallDisplay.from_predictions(y_test, y_proba, ax=axes[2])
    axes[2].set_title("Precision-Recall Curve")

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / f"{safe_name}_eval.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # Feature importance (RF native or permutation for LR)
    try:
        model = pipeline.named_steps["model"]
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            title = "RF Feature Importances (Gini)"
        else:
            result = permutation_importance(
                pipeline, X_test, y_test, n_repeats=10, random_state=42, scoring="average_precision"
            )
            importances = result.importances_mean
            title = "LR Permutation Importances (PR-AUC)"

        indices = np.argsort(importances)[::-1][:15]
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.barh(
            [feature_names[i] for i in indices[::-1]],
            importances[indices[::-1]],
            color="steelblue"
        )
        ax2.set_title(title)
        ax2.set_xlabel("Importance")
        plt.tight_layout()
        fig2.savefig(FIGURES_DIR / f"{safe_name}_importance.png", dpi=120, bbox_inches="tight")
        plt.close(fig2)
        log.info(f"Figures saved to {FIGURES_DIR}/")
    except Exception as e:
        log.warning(f"Feature importance plot failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def main():
    log.info("Connecting to PostgreSQL ...")
    engine = create_engine(DB_URL, pool_pre_ping=True)

    # Load & preprocess
    df = load_features(engine)
    X, y, feature_names = preprocess(df)

    # Stratified split — preserves fraud rate in both splits
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    log.info(f"Train: {len(X_train):,} | Test: {len(X_test):,}")
    log.info(f"Train fraud: {y_train.sum():,} | Test fraud: {y_test.sum():,}")

    results = {}

    # ── Logistic Regression
    log.info("Training Logistic Regression ...")
    lr_pipeline = build_lr_pipeline()
    lr_pipeline.fit(X_train, y_train)
    results["Logistic Regression"] = evaluate("Logistic Regression", lr_pipeline, X_test, y_test)
    plot_results("Logistic Regression", lr_pipeline, X_test, y_test, feature_names)

    # ── Random Forest
    log.info("Training Random Forest ...")
    rf_pipeline = build_rf_pipeline()
    rf_pipeline.fit(X_train, y_train)
    results["Random Forest"] = evaluate("Random Forest", rf_pipeline, X_test, y_test)
    plot_results("Random Forest", rf_pipeline, X_test, y_test, feature_names)

    # ── Select best model by PR-AUC (fraud-appropriate metric)
    best_name = max(results, key=lambda k: results[k]["pr_auc"])
    best_pipeline = lr_pipeline if best_name == "Logistic Regression" else rf_pipeline
    log.info(f"\nBest model: {best_name} (PR-AUC={results[best_name]['pr_auc']:.4f})")

    # ── Save artifacts
    artifact = {
        "pipeline":      best_pipeline,
        "feature_names": feature_names,
        "model_name":    best_name,
        "metrics":       {k: {m: v for m, v in v.items() if m != "y_proba"} for k, v in results.items()},
    }
    joblib.dump(artifact, ARTIFACTS_DIR / "fraud_model.joblib")
    joblib.dump(lr_pipeline, ARTIFACTS_DIR / "lr_pipeline.joblib")
    joblib.dump(rf_pipeline, ARTIFACTS_DIR / "rf_pipeline.joblib")

    log.info(f"Model artifacts saved to {ARTIFACTS_DIR}/")
    log.info("Training complete.")


if __name__ == "__main__":
    main()
