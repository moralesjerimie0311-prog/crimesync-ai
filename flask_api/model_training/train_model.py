"""
CrimeSync AI — Model Training Script (v2)
==========================================
Trains:
  1. Random Forest Classifier  → barangay risk prediction
  2. Evaluates with accuracy, precision, recall, F1-score
  3. Saves model + encoders + scaler to models/

Usage:
  python train_model.py --db_host localhost --db_user root --db_pass "" --db_name incident_system

Requirements:
  pip install scikit-learn pandas numpy mysql-connector-python joblib imbalanced-learn

Changes from v1:
  - Dynamic quantile-based risk classification (no hardcoded thresholds)
  - Crash-safe evaluation that works with 1, 2, or 3 classes
  - Graceful stop when fewer than 2 classes exist after labeling
  - Stratified split and cross-validation only when statistically valid
  - Richer diagnostic output throughout the pipeline
"""

import argparse
import os
import logging
import json
import numpy as np
import pandas as pd
import joblib

from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score
)
from collections import Counter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Output directory
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "flask_api", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Risk class label map (index → display name)
RISK_LABELS = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}


# ══════════════════════════════════════════════════════════════
#  STEP 1 — LOAD DATA FROM MYSQL
# ══════════════════════════════════════════════════════════════

def load_data_from_mysql(host: str, user: str, password: str, database: str) -> pd.DataFrame:
    """
    Pulls all incident records from the 'incidents' table.
    Handles flexible date columns (date_committed / date_reported / incident_date).
    """
    import mysql.connector

    logger.info(f"Connecting to MySQL: {host}/{database}")
    conn = mysql.connector.connect(
        host=host, user=user, password=password, database=database
    )

    query = """
        SELECT
            incident_id,
            TRIM(COALESCE(barangay, ''))                     AS barangay,
            COALESCE(incident_type, type, 'Unknown')         AS incident_type,
            latitude,
            longitude,
            COALESCE(
                NULLIF(date_committed, '0000-00-00'),
                NULLIF(date_reported,  '0000-00-00'),
                NULLIF(incident_date,  '0000-00-00'),
                DATE(created_at)
            )                                                 AS incident_date,
            COALESCE(priority, 'low')                        AS priority,
            COALESCE(status, 'pending')                      AS status
        FROM incidents
        WHERE barangay IS NOT NULL
          AND TRIM(barangay) != ''
        ORDER BY incident_date ASC
    """

    df = pd.read_sql(query, conn)
    conn.close()

    unique_barangays = df["barangay"].nunique() if "barangay" in df.columns else 0
    logger.info(f"✅ Loaded {len(df)} incident records across {unique_barangays} unique barangay(s)")
    return df


# ══════════════════════════════════════════════════════════════
#  STEP 2 — FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates 6 features for Random Forest:
      1. barangay_enc        — label-encoded barangay name
      2. crime_type_enc      — label-encoded incident type
      3. month               — month of incident (1–12)
      4. day_of_week         — 0=Monday, 6=Sunday
      5. total_incidents     — cumulative incidents in that barangay up to this date
      6. days_since_last     — days since previous incident in the same barangay

    Target label (risk_class) — DYNAMIC QUANTILE APPROACH:
      Thresholds are computed from the actual data distribution using the 33rd and 66th
      percentiles of each barangay's average monthly incident rate, so all three classes
      are always represented when the data has sufficient variation.

      0 = Low Risk    (avg monthly rate ≤ 33rd percentile)
      1 = Medium Risk (avg monthly rate ≤ 66th percentile)
      2 = High Risk   (avg monthly rate >  66th percentile)
    """
    logger.info("Engineering features…")

    df = df.copy()
    df["incident_date"] = pd.to_datetime(df["incident_date"], errors="coerce")
    df = df.dropna(subset=["incident_date"])
    df = df.sort_values("incident_date")

    logger.info(f"  Records after date parsing: {len(df)}")

    # ── Temporal features
    df["month"]       = df["incident_date"].dt.month
    df["day_of_week"] = df["incident_date"].dt.dayofweek

    # ── Cumulative incident count per barangay
    df["total_incidents"] = df.groupby("barangay").cumcount() + 1

    # ── Days since last incident in same barangay
    df["prev_date"]       = df.groupby("barangay")["incident_date"].shift(1)
    df["days_since_last"] = (df["incident_date"] - df["prev_date"]).dt.days.fillna(30)
    df["days_since_last"] = df["days_since_last"].clip(0, 999)

    # ── Monthly incident rate per barangay (for labelling)
    df["year_month"] = df["incident_date"].dt.to_period("M")
    monthly_rate_df = (
        df.groupby(["barangay", "year_month"])
          .size()
          .reset_index(name="monthly_count")
    )
    avg_rate_per_barangay = (
        monthly_rate_df.groupby("barangay")["monthly_count"]
        .mean()
        .reset_index(name="avg_monthly_rate")
    )
    df = df.merge(avg_rate_per_barangay, on="barangay", how="left")

    # ── DIAGNOSTIC: inspect the distribution of avg monthly rate
    print("\n" + "─"*60)
    print("  DIAGNOSTIC — avg_monthly_rate distribution per record:")
    print(df["avg_monthly_rate"].describe().to_string())
    print("\n  Per-barangay avg_monthly_rate:")
    print(df.groupby("barangay")["avg_monthly_rate"].first().to_string())
    print("─"*60 + "\n")

    # ── DYNAMIC QUANTILE CLASSIFICATION
    #    Compute thresholds from the actual barangay-level rate distribution.
    #    Using the unique per-barangay values avoids letting high-incident barangays
    #    (which have many rows) dominate the quantile calculation.
    per_barangay_rates = df.groupby("barangay")["avg_monthly_rate"].first()
    q33 = per_barangay_rates.quantile(0.33)
    q66 = per_barangay_rates.quantile(0.66)

    logger.info(f"  Dynamic risk thresholds → Low ≤ {q33:.2f}, Medium ≤ {q66:.2f}, High > {q66:.2f}")

    def classify_risk_dynamic(rate: float) -> int:
        """Assign risk class using data-driven quantile boundaries."""
        if rate > q66:
            return 2   # High Risk
        elif rate > q33:
            return 1   # Medium Risk
        return 0       # Low Risk

    df["risk_class"] = df["avg_monthly_rate"].apply(classify_risk_dynamic)

    # ── DIAGNOSTIC: class distribution
    class_dist = Counter(df["risk_class"])
    print("  Risk class distribution (record level):")
    for cls_id, count in sorted(class_dist.items()):
        print(f"    {RISK_LABELS.get(cls_id, cls_id):15s} (class {cls_id}): {count} records")
    print(f"\n  risk_class value_counts:\n{df['risk_class'].value_counts().to_string()}\n")

    logger.info(f"Class distribution: {class_dist}")
    return df


# ══════════════════════════════════════════════════════════════
#  STEP 3 — ENCODE + SCALE
# ══════════════════════════════════════════════════════════════

def encode_and_scale(df: pd.DataFrame):
    """
    Label-encode categorical columns, standard-scale numeric features.
    Returns: X (features), y (labels), encoders dict, scaler
    """
    encoders = {}

    le_bgy = LabelEncoder()
    df["barangay_enc"] = le_bgy.fit_transform(df["barangay"].astype(str))
    encoders["barangay"] = le_bgy

    le_type = LabelEncoder()
    df["crime_type_enc"] = le_type.fit_transform(df["incident_type"].astype(str))
    encoders["crime_type"] = le_type

    feature_cols = [
        "barangay_enc",
        "crime_type_enc",
        "month",
        "day_of_week",
        "total_incidents",
        "days_since_last",
    ]

    X = df[feature_cols].values.astype(float)
    y = df["risk_class"].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    logger.info(f"  Feature matrix shape: X={X.shape}, unique classes={sorted(np.unique(y).tolist())}")
    return X, y, encoders, scaler, feature_cols


# ══════════════════════════════════════════════════════════════
#  STEP 4 — TRAIN RANDOM FOREST
# ══════════════════════════════════════════════════════════════

def train_random_forest(X_train, y_train) -> RandomForestClassifier:
    """
    Trains a Random Forest with class_weight='balanced' to handle
    imbalanced datasets (more Low/Medium than High Risk incidents).
    """
    logger.info("Training Random Forest…")

    model = RandomForestClassifier(
        n_estimators=200,         # 200 trees for stable predictions
        max_depth=12,             # limit overfitting
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",  # handles class imbalance automatically
        random_state=42,
        n_jobs=-1,                # use all CPU cores
    )
    model.fit(X_train, y_train)

    logger.info(f"✅ Random Forest trained  (train size={len(y_train)})")
    return model


# ══════════════════════════════════════════════════════════════
#  STEP 5 — EVALUATE MODEL
# ══════════════════════════════════════════════════════════════

def evaluate_model(model, X_test, y_test, feature_names: list | None = None) -> dict:
    """
    Crash-safe evaluation that works with 1, 2, or 3 classes present.

    Dynamically builds target_names from the classes that actually appear
    in y_test, so classification_report never raises a ValueError.

    Returns a dict with accuracy, precision, recall, F1, per-class breakdown,
    confusion matrix, and feature importances.
    """
    logger.info("Evaluating model…")
    y_pred = model.predict(X_test)

    # ── Detect which classes are present in the test set
    present_classes   = sorted(np.unique(np.concatenate([y_test, y_pred])).tolist())
    present_labels    = [RISK_LABELS.get(c, f"Class {c}") for c in present_classes]

    logger.info(f"  Classes present in test set: {present_classes} → {present_labels}")

    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0,
                                labels=present_classes)
    recall    = recall_score(y_test, y_pred, average="weighted", zero_division=0,
                             labels=present_classes)
    f1        = f1_score(y_test, y_pred, average="weighted", zero_division=0,
                         labels=present_classes)

    # Per-class metrics — only for classes actually present
    report = classification_report(
        y_test, y_pred,
        labels=present_classes,          # ← key fix: pass only present classes
        target_names=present_labels,     # ← matches present_classes 1-to-1
        output_dict=True,
        zero_division=0,
    )

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=present_classes).tolist()

    # Feature importances
    importances = model.feature_importances_.tolist()
    feat_imp = dict(zip(feature_names, importances)) if feature_names else importances

    metrics = {
        "accuracy":                round(accuracy * 100, 2),
        "precision_weighted":      round(precision * 100, 2),
        "recall_weighted":         round(recall * 100, 2),
        "f1_weighted":             round(f1 * 100, 2),
        "per_class":               {k: v for k, v in report.items() if k in present_labels},
        "confusion_matrix":        cm,
        "confusion_matrix_labels": present_labels,
        "feature_importances":     feat_imp,
        "test_set_size":           len(y_test),
        "class_distribution_test": dict(Counter(y_test.tolist())),
        "present_classes":         present_classes,
    }

    # Thesis-ready summary
    col_w = max(len(lbl) for lbl in present_labels) + 2
    print("\n" + "="*60)
    print("  CrimeSync AI — Model Evaluation Report")
    print("="*60)
    print(f"  Accuracy:             {metrics['accuracy']}%")
    print(f"  Precision (weighted): {metrics['precision_weighted']}%")
    print(f"  Recall (weighted):    {metrics['recall_weighted']}%")
    print(f"  F1-Score (weighted):  {metrics['f1_weighted']}%")
    print(f"\n  Per-Class Breakdown:  ({len(present_classes)} class(es) detected)")
    for lbl in present_labels:
        if lbl in report:
            r = report[lbl]
            print(f"    {lbl:{col_w}s}: P={r['precision']:.2f}  R={r['recall']:.2f}"
                  f"  F1={r['f1-score']:.2f}  N={r['support']:.0f}")
    print(f"\n  Confusion Matrix (rows=actual, cols=predicted):")
    header = "".join(f"{lbl[:5]:>7}" for lbl in present_labels)
    print(f"  {'':>{col_w}}  {header}")
    for i, row in enumerate(cm):
        row_str = "".join(f"{v:>7}" for v in row)
        print(f"  {present_labels[i]:{col_w}s}  {row_str}")
    print("="*60 + "\n")

    return metrics


def cross_validate(model, X, y) -> dict:
    """
    Stratified k-fold cross-validation.

    Automatically reduces the number of folds if there are not enough
    samples in the smallest class, and skips entirely if validation is
    not statistically meaningful.
    """
    class_counts = Counter(y.tolist())
    min_class_count = min(class_counts.values())
    n_classes = len(class_counts)

    # Need at least 2 samples per class per fold for stratified CV
    max_folds = min(5, min_class_count)

    if max_folds < 2:
        logger.warning(
            f"  ⚠️  Skipping cross-validation: smallest class has only "
            f"{min_class_count} sample(s). Need at least 2 per fold."
        )
        return {
            "cv_skipped": True,
            "cv_skip_reason": (
                f"Smallest class has {min_class_count} sample(s); "
                f"need ≥ 2 per fold for stratified CV."
            ),
        }

    n_splits = max_folds
    logger.info(f"  Running {n_splits}-fold stratified cross-validation…")

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="f1_weighted")
    result = {
        "cv_skipped":    False,
        "cv_n_folds":    n_splits,
        "cv_f1_scores":  [round(s * 100, 2) for s in scores],
        "cv_f1_mean":    round(scores.mean() * 100, 2),
        "cv_f1_std":     round(scores.std() * 100, 2),
    }
    print(f"  Cross-Validation ({n_splits}-fold) F1: {result['cv_f1_mean']}%"
          f" ± {result['cv_f1_std']}%")
    return result


# ══════════════════════════════════════════════════════════════
#  STEP 6 — SAVE MODELS
# ══════════════════════════════════════════════════════════════

def save_models(model, encoders, scaler) -> None:
    joblib.dump(model,    os.path.join(MODEL_DIR, "rf_risk_model.joblib"))
    joblib.dump(encoders, os.path.join(MODEL_DIR, "label_encoders.joblib"))
    joblib.dump(scaler,   os.path.join(MODEL_DIR, "feature_scaler.joblib"))
    logger.info(f"✅ Models saved to {MODEL_DIR}/")


# ══════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ══════════════════════════════════════════════════════════════

def main(args) -> None:

    # ── 1. Load data
    df = load_data_from_mysql(args.db_host, args.db_user, args.db_pass, args.db_name)

    if len(df) < 50:
        logger.error(
            f"❌ Not enough data to train. "
            f"Loaded {len(df)} records but need at least 50. "
            "Collect more incident data and re-run."
        )
        return

    # ── 2. Feature engineering (includes dynamic risk labelling)
    df_feat = engineer_features(df)

    # ── 3. Class-count safeguard
    n_classes = df_feat["risk_class"].nunique()
    if n_classes < 2:
        logger.error(
            f"❌ Only {n_classes} risk class found after labelling. "
            "Training requires at least 2 distinct classes.\n"
            "\n"
            "Why this happens:\n"
            "  • All barangays have nearly identical incident rates, so the\n"
            "    quantile boundaries collapse into a single bin.\n"
            "\n"
            "Suggested fixes:\n"
            "  1. Collect more incident data spanning a longer time period.\n"
            "  2. Ensure data includes barangays with meaningfully different\n"
            "     incident frequencies.\n"
            "  3. Lower the minimum record threshold (currently 50) if your\n"
            "     dataset is intentionally small.\n"
        )
        return

    # ── 4. Encode and scale
    feature_names = ["barangay_enc", "crime_type_enc", "month", "day_of_week",
                     "total_incidents", "days_since_last"]
    X, y, encoders, scaler, _ = encode_and_scale(df_feat)

    logger.info(f"Dataset → X={X.shape}, y classes={sorted(np.unique(y).tolist())}, "
                f"n_samples={len(y)}")

    # ── 5. Train/test split — stratified only when valid
    class_counts = Counter(y.tolist())
    min_count = min(class_counts.values())

    if min_count < 2:
        logger.error(
            f"❌ Cannot stratify: the smallest class has only {min_count} sample(s). "
            "Stratified splitting requires ≥ 2 samples per class. "
            "Collect more data for under-represented barangays."
        )
        return

    use_stratify = n_classes >= 2
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y if use_stratify else None,
    )

    logger.info(f"  Train size: {len(y_train)},  Test size: {len(y_test)}")
    logger.info(f"  Train class dist: {Counter(y_train.tolist())}")
    logger.info(f"  Test  class dist: {Counter(y_test.tolist())}")

    # ── 6. Train model
    model = train_random_forest(X_train, y_train)

    # ── 7. Evaluate
    metrics    = evaluate_model(model, X_test, y_test, feature_names)
    cv_metrics = cross_validate(model, X, y)

    # ── 8. Save models
    save_models(model, encoders, scaler)

    # ── 9. Save evaluation report (JSON for thesis documentation)
    report = {
        **metrics,
        **cv_metrics,
        "trained_at":   datetime.now().isoformat(),
        "n_samples":    len(df),
        "n_barangays":  int(df["barangay"].nunique()),
        "n_classes":    int(n_classes),
        "features":     feature_names,
    }
    report_path = os.path.join(MODEL_DIR, "evaluation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"✅ Evaluation report saved: {report_path}")

    print("\n✅ Training complete. Models saved to flask_api/models/")
    print(f"   Records loaded:  {len(df)}")
    print(f"   Unique barangays: {df['barangay'].nunique()}")
    print(f"   Risk classes:    {n_classes}")
    print(f"   Accuracy:        {metrics['accuracy']}%")
    print(f"   F1-Score:        {metrics['f1_weighted']}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CrimeSync AI Model Training")
    parser.add_argument("--db_host", default="localhost")
    parser.add_argument("--db_user", default="root")
    parser.add_argument("--db_pass", default="")
    parser.add_argument("--db_name", default="incident_system")
    args = parser.parse_args()
    main(args)