"""
Baseline training script for ML pipeline.

Trains a simple logistic regression model on gold training examples data.
Produces predictions with time-based train/test split for temporal validation.

Usage:
    python train_baseline.py --training-examples-csv ./tmp/snapshots/gold_run_1/training_examples.csv
"""

import argparse
import csv
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
import warnings

from config import LOG_LEVEL
from logging_utils import setup_logging

try:
    import joblib
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score, classification_report
    from sklearn.preprocessing import StandardScaler
except ImportError as e:
    print(f"Required packages not available: {e}")
    print("Install with: pip install pandas numpy scikit-learn joblib")
    sys.exit(1)

logger = logging.getLogger(__name__)

# Set random state for reproducibility
RANDOM_STATE = 42


def validate_csv_file(file_path, label_col="label_responded_within_7d"):
    """Validate CSV file exists and has required columns."""
    if not file_path.exists():
        raise FileNotFoundError(f"Training examples CSV not found: {file_path}")

    if not file_path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")

    # Check required columns
    required_cols = [
        'org_id', 'enrollment_id', 'ghl_contact_id', 'decision_ts_utc',
        label_col,
        'attempts_sms_24h', 'attempts_email_24h', 'attempts_voice_no_voicemail_24h', 'voicemail_drops_24h'
    ]

    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        headers = set(reader.fieldnames or [])
        missing_cols = set(required_cols) - headers
        if missing_cols:
            raise ValueError(f"Missing required columns in {file_path.name}: {missing_cols}")

    return True


def load_and_prepare_data(csv_path, label_col="label_responded_within_7d"):
    """Load training data and prepare features/target."""
    logger.info("Loading data from %s...", csv_path)

    # Load CSV
    df = pd.read_csv(csv_path)

    # Parse decision timestamp
    df['decision_ts_utc'] = pd.to_datetime(df['decision_ts_utc'])

    # Define feature columns (no labels)
    feature_cols = [
        'attempts_sms_24h', 'attempts_email_24h', 'attempts_voice_no_voicemail_24h', 'voicemail_drops_24h'
    ]

    # Extract features and target
    X = df[feature_cols].copy()
    y = df[label_col].copy()

    # Handle missing values (fill all with 0 - counts and binary ints from SQL)
    for col in feature_cols:
        X[col] = X[col].fillna(0)

    # Convert target to int (fill missing labels with 0)
    y = y.fillna(0).astype(int)

    logger.info("Loaded %d rows with %d features", len(df), len(feature_cols))
    logger.info("Target distribution: %s", y.value_counts().to_dict())

    return df, X, y, feature_cols


def create_time_split(df, test_size=0.2):
    """Create train/test split based on decision timestamp (earliest 80% train, latest 20% test)."""
    # Sort by decision timestamp
    df_sorted = df.sort_values('decision_ts_utc').reset_index(drop=True)

    # Calculate split point
    n_total = len(df_sorted)
    n_train = int(n_total * (1 - test_size))

    # Create split indicator
    split_labels = ['train'] * n_train + ['test'] * (n_total - n_train)
    df_sorted['split'] = split_labels

    train_idx = df_sorted['split'] == 'train'
    test_idx = df_sorted['split'] == 'test'

    logger.info("Time-based split: %d train (%.1f%%), %d test (%.1f%%)",
                n_train, n_train/n_total*100, n_total-n_train, (n_total-n_train)/n_total*100)

    if n_train > 0:
        train_date_range = f"{df_sorted[train_idx]['decision_ts_utc'].min()} to {df_sorted[train_idx]['decision_ts_utc'].max()}"
        logger.info("Train date range: %s", train_date_range)

    if n_total - n_train > 0:
        test_date_range = f"{df_sorted[test_idx]['decision_ts_utc'].min()} to {df_sorted[test_idx]['decision_ts_utc'].max()}"
        logger.info("Test date range: %s", test_date_range)

    return df_sorted, train_idx, test_idx


def train_baseline_model(X_train, y_train, X_test, y_test):
    """Train logistic regression baseline model."""
    if len(X_train) == 0:
        raise ValueError("No training data available")

    logger.info("Training logistic regression on %d samples...", len(X_train))

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test) if len(X_test) > 0 else np.array([]).reshape(0, X_train.shape[1])

    # Train model
    model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000, class_weight="balanced")
    model.fit(X_train_scaled, y_train)

    # Generate predictions
    train_scores = model.predict_proba(X_train_scaled)[:, 1] if len(X_train_scaled) > 0 else np.array([])
    test_scores = model.predict_proba(X_test_scaled)[:, 1] if len(X_test_scaled) > 0 else np.array([])

    logger.info("Model trained successfully")

    return model, scaler, train_scores, test_scores


def compute_metrics(y_true, scores, split_name):
    """Compute and log evaluation metrics."""
    if len(y_true) == 0 or len(scores) == 0:
        logger.info("%s: No data for evaluation", split_name)
        return {}

    metrics = {}

    # Basic stats
    n_samples = len(y_true)
    n_positive = sum(y_true)
    positive_rate = n_positive / n_samples if n_samples > 0 else 0

    metrics['n_samples'] = n_samples
    metrics['n_positive'] = n_positive
    metrics['positive_rate'] = positive_rate

    logger.info("%s: %d samples, %d positive (%.3f rate)", split_name, n_samples, n_positive, positive_rate)

    # Skip advanced metrics for very small datasets or edge cases
    if n_samples < 5 or n_positive == 0 or n_positive == n_samples:
        logger.info("%s: Skipping advanced metrics (insufficient or homogeneous data)", split_name)
        return metrics

    try:
        # PR-AUC (preferred)
        pr_auc = average_precision_score(y_true, scores)
        metrics['pr_auc'] = pr_auc
        logger.info("%s: PR-AUC = %.4f", split_name, pr_auc)

        # ROC-AUC (fallback)
        if len(np.unique(y_true)) > 1:  # Need both classes for ROC-AUC
            roc_auc = roc_auc_score(y_true, scores)
            metrics['roc_auc'] = roc_auc
            logger.info("%s: ROC-AUC = %.4f", split_name, roc_auc)

    except Exception as e:
        logger.warning("%s: Could not compute advanced metrics: %s", split_name, e)

    return metrics


def create_predictions_output(df_with_split, train_scores, test_scores, output_path, label_col="label_responded_within_7d"):
    """Create predictions CSV with scores and metadata."""
    # Copy dataframe to avoid modifying original
    df_out = df_with_split.copy()

    # Assign scores directly using boolean masks (safer than enumerate/iterrows)
    train_idx = df_out['split'] == 'train'
    test_idx = df_out['split'] == 'test'

    df_out.loc[train_idx, 'score'] = train_scores if len(train_scores) > 0 else 0.0
    df_out.loc[test_idx, 'score'] = test_scores if len(test_scores) > 0 else 0.0

    # Add split_order for proper sorting (test=0 first, train=1 second)
    df_out['split_order'] = df_out['split'].map({'test': 0, 'train': 1})

    # Select output columns
    output_cols = [
        'org_id', 'enrollment_id', 'ghl_contact_id', 'decision_ts_utc',
        'score', label_col, 'split'
    ]
    pred_df = df_out[output_cols + ['split_order']].copy()
    pred_df.rename(columns={label_col: 'y_true'}, inplace=True)

    # Sort by split_order (test first), then score desc within each split
    pred_df_sorted = pred_df.sort_values(['split_order', 'score'], ascending=[True, False])

    # Drop split_order column before output
    pred_df_sorted = pred_df_sorted.drop(columns=['split_order'])

    # Write to CSV
    pred_df_sorted.to_csv(output_path, index=False)
    logger.info("Predictions written to %s", output_path)

    return pred_df_sorted


def compute_precision_at_k(pred_df, ks=(10, 20, 50)):
    """Compute precision@k from the test split of predictions, skipping k > n_test."""
    test_df = pred_df[pred_df['split'] == 'test'].copy()
    n_test = len(test_df)
    # pred_df is already sorted by score desc within splits
    results = []
    for k in ks:
        if k > n_test:
            continue
        top_k = test_df.head(k)
        n_positive = int(top_k['y_true'].sum())
        results.append({
            "k": k,
            "n": n_test,
            "n_positive": n_positive,
            "precision": n_positive / k,
        })
    return results


def _json_default(obj):
    """Handle numpy types for JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def write_metrics_json(metrics, output_path):
    """Write metrics dict to JSON with deterministic formatting."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, sort_keys=True, ensure_ascii=False, default=_json_default)
        f.write('\n')
    logger.info("Metrics written to %s", output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Train baseline logistic regression model on gold training examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--training-examples-csv',
        required=True,
        type=Path,
        help='Path to training_examples.csv file'
    )

    parser.add_argument(
        '--label-col',
        default='label_responded_within_7d',
        help='Name of the label column to use as target (default: label_responded_within_7d)'
    )

    parser.add_argument(
        '--log-level',
        default=None,
        help='Log level: DEBUG, INFO, WARNING, ERROR (default: from LOG_LEVEL env var)'
    )

    args = parser.parse_args()

    setup_logging(args.log_level or LOG_LEVEL)

    logger.info("Input CSV: %s, label_col: %s", args.training_examples_csv, args.label_col)

    # Validate input file
    try:
        validate_csv_file(args.training_examples_csv, label_col=args.label_col)
    except (FileNotFoundError, ValueError) as e:
        logger.error("Error: %s", e)
        sys.exit(1)

    # Determine output path
    predictions_path = args.training_examples_csv.parent / "predictions.csv"

    try:
        # Load and prepare data
        df, X, y, feature_cols = load_and_prepare_data(args.training_examples_csv, label_col=args.label_col)

        # Check for tiny dataset
        if len(df) < 20:
            logger.warning("Small dataset (%d rows). Metrics may be unreliable.", len(df))

        # Create time-based split
        df_with_split, train_idx, test_idx = create_time_split(df)

        # Rebuild X and y from sorted dataframe to align with split indices
        X = df_with_split[feature_cols].copy()
        y = df_with_split[args.label_col].fillna(0).astype(int).copy()

        X_train = X[train_idx]
        y_train = y[train_idx]
        X_test = X[test_idx]
        y_test = y[test_idx]

        # Early-exit if training split has only one class
        if y_train.nunique() < 2:
            logger.warning("Cannot train: training split has only one class (all 0s or all 1s). "
                           "Expand date range or change label.")
            sys.exit(0)

        # Train model
        model, scaler, train_scores, test_scores = train_baseline_model(X_train, y_train, X_test, y_test)

        # Save model and scaler artifacts
        model_path = args.training_examples_csv.parent / "model.joblib"
        scaler_path = args.training_examples_csv.parent / "scaler.joblib"
        joblib.dump(model, model_path)
        logger.info("Model saved to %s", model_path)
        joblib.dump(scaler, scaler_path)
        logger.info("Scaler saved to %s", scaler_path)

        # Compute and log metrics
        logger.info("=" * 60)
        logger.info("MODEL EVALUATION")
        logger.info("=" * 60)

        train_metrics = compute_metrics(y_train.values, train_scores, "TRAIN")
        test_metrics = compute_metrics(y_test.values, test_scores, "TEST")

        # Create predictions output
        logger.info("=" * 60)
        logger.info("OUTPUT GENERATION")
        logger.info("=" * 60)

        pred_df = create_predictions_output(df_with_split, train_scores, test_scores, predictions_path, label_col=args.label_col)

        # Compute precision@k from test split
        precision_at_k = compute_precision_at_k(pred_df)

        # Write metrics.json
        metrics_path = args.training_examples_csv.parent / "metrics.json"
        metrics_payload = {
            "run_timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "input_training_examples_csv": str(args.training_examples_csv),
            "predictions_csv": str(predictions_path),
            "label_col": args.label_col,
            "feature_cols": feature_cols,
            "train": train_metrics,
            "test": test_metrics,
            "precision_at_k": precision_at_k,
            "model_path": str(model_path),
            "scaler_path": str(scaler_path),
        }
        write_metrics_json(metrics_payload, metrics_path)

        # Final summary
        logger.info("=" * 60)
        logger.info("BASELINE TRAINING COMPLETED")
        logger.info("=" * 60)
        logger.info("Input: %s", args.training_examples_csv)
        logger.info("Output: %s", predictions_path)
        logger.info("Features used: %d", len(feature_cols))
        logger.info("Total samples: %d", len(df))
        logger.info("Train samples: %d", sum(train_idx))
        logger.info("Test samples: %d", sum(test_idx))

        if 'pr_auc' in test_metrics:
            logger.info("Test PR-AUC: %.4f", test_metrics['pr_auc'])

    except Exception as e:
        logger.error("Training failed: %s", e)
        sys.exit(1)


if __name__ == '__main__':
    main()
