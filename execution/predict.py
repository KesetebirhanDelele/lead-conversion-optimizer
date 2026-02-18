"""
Predict-only script for ML pipeline.

Loads a previously trained model (model.joblib) and scaler (scaler.joblib)
from the same directory as the input CSV, scores every row, and writes
predictions.csv and metrics.json.

Usage (PowerShell):
    python execution/predict.py `
        --training-examples-csv ./tmp/snapshots/gold_run_1/training_examples.csv

    python execution/predict.py `
        --training-examples-csv ./tmp/snapshots/gold_run_1/training_examples.csv `
        --label-col label_responded_within_14d `
        --predictions-out ./tmp/out/predictions.csv `
        --metrics-out ./tmp/out/metrics.json
"""

import argparse
import csv
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from config import LOG_LEVEL
from logging_utils import setup_logging

try:
    import joblib
    import pandas as pd
    import numpy as np
    from sklearn.metrics import average_precision_score, roc_auc_score
except ImportError as e:
    print(f"Required packages not available: {e}")
    print("Install with: pip install pandas numpy scikit-learn joblib")
    sys.exit(1)

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    'attempts_sms_24h',
    'attempts_email_24h',
    'attempts_voice_no_voicemail_24h',
    'voicemail_drops_24h',
]

ID_COLS = ['org_id', 'enrollment_id', 'ghl_contact_id', 'decision_ts_utc']


def validate_inputs(csv_path, model_path, scaler_path):
    """Validate that the CSV, model, and scaler files exist and CSV has required columns."""
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not csv_path.is_file():
        raise ValueError(f"Path is not a file: {csv_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler artifact not found: {scaler_path}")

    required_cols = set(ID_COLS + FEATURE_COLS)
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        headers = set(reader.fieldnames or [])
        missing = required_cols - headers
        if missing:
            raise ValueError(f"Missing required columns in {csv_path.name}: {missing}")

    return True


def load_and_score(csv_path, model, scaler):
    """Load CSV, prepare features, and generate scores."""
    logger.info("Loading data from %s...", csv_path)
    df = pd.read_csv(csv_path)
    df['decision_ts_utc'] = pd.to_datetime(df['decision_ts_utc'])

    X = df[FEATURE_COLS].copy()
    for col in FEATURE_COLS:
        X[col] = X[col].fillna(0)

    X_scaled = scaler.transform(X)
    scores = model.predict_proba(X_scaled)[:, 1]
    df['score'] = scores

    logger.info("Scored %d rows", len(df))
    return df


def build_predictions_df(df, label_col):
    """Build the output predictions dataframe, sorted by score descending."""
    output_cols = list(ID_COLS) + ['score']

    has_label = label_col in df.columns
    if has_label:
        df = df.copy()
        df['y_true'] = df[label_col].fillna(0).astype(int)
        output_cols.append('y_true')

    pred_df = df[output_cols].copy()
    pred_df = pred_df.sort_values('score', ascending=False).reset_index(drop=True)
    return pred_df, has_label


def compute_score_quantiles(scores):
    """Compute p0/p10/p50/p90/p100 quantiles from score array."""
    return {
        "p0": float(np.percentile(scores, 0)),
        "p10": float(np.percentile(scores, 10)),
        "p50": float(np.percentile(scores, 50)),
        "p90": float(np.percentile(scores, 90)),
        "p100": float(np.percentile(scores, 100)),
    }


def compute_precision_at_k(y_true, scores_desc, ks=(10, 20, 50)):
    """Compute precision@k from y_true aligned with scores sorted descending."""
    n = len(y_true)
    results = []
    for k in ks:
        if k > n:
            continue
        top_k_labels = y_true[:k]
        n_positive = int(top_k_labels.sum())
        results.append({
            "k": k,
            "n": n,
            "n_positive": n_positive,
            "precision": n_positive / k,
        })
    return results


def build_metrics(args, pred_df, has_label, model_path, scaler_path, predictions_out):
    """Build the metrics payload dict."""
    scores = pred_df['score'].values
    metrics = {
        "run_timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "input_training_examples_csv": str(args.training_examples_csv),
        "predictions_csv": str(predictions_out),
        "label_col": args.label_col,
        "feature_cols": list(FEATURE_COLS),
        "n_samples": len(pred_df),
        "score_quantiles": compute_score_quantiles(scores),
        "model_path": str(model_path),
        "scaler_path": str(scaler_path),
    }

    if has_label:
        y_true = pred_df['y_true'].values
        n_positive = int(y_true.sum())
        n_samples = len(y_true)
        metrics["n_positive"] = n_positive
        metrics["positive_rate"] = n_positive / n_samples if n_samples > 0 else 0.0

        if len(np.unique(y_true)) > 1:
            metrics["pr_auc"] = float(average_precision_score(y_true, scores))
            metrics["roc_auc"] = float(roc_auc_score(y_true, scores))
        else:
            metrics["pr_auc"] = None
            metrics["roc_auc"] = None

        metrics["precision_at_k"] = compute_precision_at_k(y_true, scores)

    return metrics


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
        description="Score data using a previously trained model and scaler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        '--training-examples-csv', required=True, type=Path,
        help='Path to CSV file to score',
    )
    parser.add_argument(
        '--label-col', default='label_responded_within_7d',
        help='Label column name (used for optional evaluation metrics)',
    )
    parser.add_argument(
        '--predictions-out', type=Path, default=None,
        help='Output path for predictions CSV (default: <csv_dir>/predictions.csv)',
    )
    parser.add_argument(
        '--metrics-out', type=Path, default=None,
        help='Output path for metrics JSON (default: <csv_dir>/metrics.json)',
    )
    parser.add_argument(
        '--log-level', default=None,
        help='Log level: DEBUG, INFO, WARNING, ERROR (default: from LOG_LEVEL env var)',
    )

    args = parser.parse_args()

    setup_logging(args.log_level or LOG_LEVEL)

    csv_dir = args.training_examples_csv.parent
    model_path = csv_dir / "model.joblib"
    scaler_path = csv_dir / "scaler.joblib"
    predictions_out = args.predictions_out or (csv_dir / "predictions.csv")
    metrics_out = args.metrics_out or (csv_dir / "metrics.json")

    # Validate inputs
    try:
        validate_inputs(args.training_examples_csv, model_path, scaler_path)
    except (FileNotFoundError, ValueError) as e:
        logger.error("Error: %s", e)
        sys.exit(1)

    try:
        # Load artifacts
        logger.info("Loading model from %s...", model_path)
        model = joblib.load(model_path)
        logger.info("Model loaded")
        logger.info("Loading scaler from %s...", scaler_path)
        scaler = joblib.load(scaler_path)
        logger.info("Scaler loaded")

        # Score
        df = load_and_score(args.training_examples_csv, model, scaler)

        # Build predictions output
        pred_df, has_label = build_predictions_df(df, args.label_col)
        pred_df.to_csv(predictions_out, index=False)
        logger.info("Predictions written to %s", predictions_out)

        # Build and write metrics
        metrics = build_metrics(args, pred_df, has_label, model_path, scaler_path, predictions_out)
        write_metrics_json(metrics, metrics_out)

        # Summary
        logger.info("=" * 60)
        logger.info("PREDICT COMPLETED")
        logger.info("=" * 60)
        logger.info("Input: %s", args.training_examples_csv)
        logger.info("Predictions: %s", predictions_out)
        logger.info("Metrics: %s", metrics_out)
        logger.info("Rows scored: %d", len(pred_df))
        if has_label and metrics.get("pr_auc") is not None:
            logger.info("PR-AUC: %.4f", metrics['pr_auc'])

    except Exception as e:
        logger.error("Prediction failed: %s", e)
        sys.exit(1)


if __name__ == '__main__':
    main()
