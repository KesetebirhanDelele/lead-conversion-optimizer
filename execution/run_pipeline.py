"""
End-to-end pipeline: extract snapshot then train or predict, optionally persist scores.

Orchestration-only — no business logic. Invokes extract_snapshot.py,
then either train_baseline.py or predict.py, then optionally
write_scores_to_sql.py as subprocesses.

Usage (train — default):
    python run_pipeline.py \
        --since 2024-01-01 --until 2024-02-01 \
        --target booked_call_within_7d \
        --out-root ./tmp/runs \
        --outcomes-query-file ./sql/outcomes.sql \
        --training-examples-query-file ./sql/training_examples.sql

Usage (predict with external artifacts):
    python run_pipeline.py --mode predict \
        --artifacts-dir ./tmp/runs/previous_run \
        --since 2024-01-01 --until 2024-02-01 \
        --target booked_call_within_7d \
        --out-root ./tmp/runs \
        --outcomes-query-file ./sql/outcomes.sql \
        --training-examples-query-file ./sql/training_examples.sql

Usage (skip score persistence):
    python run_pipeline.py --no-persist-scores \
        --since 2024-01-01 --until 2024-02-01 \
        --target booked_call_within_7d \
        --out-root ./tmp/runs \
        --outcomes-query-file ./sql/outcomes.sql \
        --training-examples-query-file ./sql/training_examples.sql
"""

import argparse
import csv
import logging
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from config import LOG_LEVEL
from logging_utils import setup_logging


logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent


def build_run_folder_name(since, until, target):
    """Build a deterministic run folder name with a UTC timestamp suffix."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"run_{since}_{until}_{target}_{ts}"


def build_extract_cmd(args, run_dir):
    """Build the subprocess command list for extract_snapshot.py."""
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "extract_snapshot.py"),
        "--since", args.since,
        "--until", args.until,
        "--target", args.target,
        "--out", str(run_dir),
        "--outcomes-query-file", str(args.outcomes_query_file),
        "--training-examples-query-file", str(args.training_examples_query_file),
    ]
    return cmd


def build_train_cmd(training_csv, label_col=None):
    """Build the subprocess command list for train_baseline.py."""
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "train_baseline.py"),
        "--training-examples-csv", str(training_csv),
    ]
    if label_col is not None:
        cmd.extend(["--label-col", label_col])
    return cmd


def build_predict_cmd(training_csv):
    """Build the subprocess command list for predict.py."""
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "predict.py"),
        "--training-examples-csv", str(training_csv),
    ]
    return cmd


def build_write_scores_cmd(predictions_csv, metrics_json, table_name="dbo.lead_scores"):
    """Build the subprocess command list for write_scores_to_sql.py."""
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "write_scores_to_sql.py"),
        "--predictions-csv", str(predictions_csv),
        "--metrics-json", str(metrics_json),
        "--table-name", table_name,
    ]
    return cmd


def _csv_has_data_rows(path):
    """Return True if CSV file has at least one data row beyond the header."""
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        return next(reader, None) is not None


def main():
    parser = argparse.ArgumentParser(
        description="Run extract + train pipeline end-to-end",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--mode", choices=["train", "predict"], default="train", help="Pipeline mode: train (default) or predict")
    parser.add_argument("--since", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--until", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--target", required=True, help="Target variable name (pass-through to extract)")
    parser.add_argument("--out-root", required=True, type=Path, help="Root directory for run folders")
    parser.add_argument("--outcomes-query-file", required=True, type=Path, help="SQL file for outcomes extraction")
    parser.add_argument("--training-examples-query-file", required=True, type=Path, help="SQL file for training examples extraction")
    parser.add_argument("--label-col", default=None, help="Label column (pass-through to train_baseline.py)")
    parser.add_argument("--persist-scores", action="store_true", default=True, help="Persist scores to SQL after training/prediction (default: True)")
    parser.add_argument("--no-persist-scores", action="store_false", dest="persist_scores", help="Skip score persistence to SQL")
    parser.add_argument("--scores-table-name", default="dbo.lead_scores", help="Target SQL table for scores (default: dbo.lead_scores)")
    parser.add_argument("--artifacts-dir", type=Path, default=None, help="Directory containing model.joblib/scaler.joblib for predict mode (takes precedence over --registry-dir)")
    parser.add_argument("--registry-dir", type=Path, default=None, help="Model registry directory; predict mode loads from registry-dir/active/ when --artifacts-dir not provided")
    parser.add_argument("--log-level", default=None, help="Log level: DEBUG, INFO, WARNING, ERROR (default: from LOG_LEVEL env var)")

    args = parser.parse_args()

    setup_logging(args.log_level or LOG_LEVEL)

    logger.info("Pipeline mode=%s, since=%s, until=%s, target=%s", args.mode, args.since, args.until, args.target)

    # Create run folder
    run_name = build_run_folder_name(args.since, args.until, args.target)
    run_dir = args.out_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Run folder: %s", run_dir)

    # Step 1: Extract snapshot
    logger.info("=" * 60)
    logger.info("STEP 1: EXTRACT SNAPSHOT")
    logger.info("=" * 60)

    extract_cmd = build_extract_cmd(args, run_dir)
    logger.info("Running: %s", ' '.join(extract_cmd))
    result = subprocess.run(extract_cmd)
    if result.returncode != 0:
        logger.error("Extraction failed with exit code %d", result.returncode)
        sys.exit(result.returncode)

    # Fail fast if training_examples.csv is missing
    training_csv = run_dir / "training_examples.csv"
    if not training_csv.exists():
        logger.error("training_examples.csv not found in %s", run_dir)
        sys.exit(1)

    # Check if training_examples.csv has any data rows
    has_data = _csv_has_data_rows(training_csv)

    # Step 2: Train or Predict
    if args.mode == "train":
        logger.info("=" * 60)
        logger.info("STEP 2: TRAIN BASELINE MODEL")
        logger.info("=" * 60)

        train_cmd = build_train_cmd(training_csv, label_col=args.label_col)
        logger.info("Running: %s", ' '.join(train_cmd))
        result = subprocess.run(train_cmd)
        if result.returncode != 0:
            logger.error("Training failed with exit code %d", result.returncode)
            sys.exit(result.returncode)

    elif args.mode == "predict":
        logger.info("=" * 60)
        logger.info("STEP 2: PREDICT (score only)")
        logger.info("=" * 60)

        if not has_data:
            logger.warning("training_examples.csv has 0 data rows; skipping artifact copy and prediction.")
        else:
            # Determine artifact source: --artifacts-dir takes precedence, then --registry-dir/active/
            artifact_source = None
            if args.artifacts_dir is not None:
                artifact_source = args.artifacts_dir
            elif args.registry_dir is not None:
                artifact_source = args.registry_dir / "active"

            if artifact_source is not None:
                if not artifact_source.is_dir():
                    logger.error("Artifact source does not exist or is not a directory: %s", artifact_source)
                    sys.exit(1)
                for artifact in ("model.joblib", "scaler.joblib"):
                    src = artifact_source / artifact
                    if not src.exists():
                        logger.error("%s not found in %s", artifact, artifact_source)
                        sys.exit(1)
                    shutil.copy2(src, run_dir / artifact)
                    logger.info("Copied %s from %s to %s", artifact, artifact_source, run_dir)

            # Verify model artifacts exist before calling predict.py
            model_path = run_dir / "model.joblib"
            scaler_path = run_dir / "scaler.joblib"
            if not model_path.exists():
                logger.error("model.joblib not found in %s. Promote a model (promote_model.py), pass --registry-dir, or pass --artifacts-dir.", run_dir)
                sys.exit(1)
            if not scaler_path.exists():
                logger.error("scaler.joblib not found in %s. Promote a model (promote_model.py), pass --registry-dir, or pass --artifacts-dir.", run_dir)
                sys.exit(1)

        predict_cmd = build_predict_cmd(training_csv)
        logger.info("Running: %s", ' '.join(predict_cmd))
        result = subprocess.run(predict_cmd)
        if result.returncode != 0:
            logger.error("Prediction failed with exit code %d", result.returncode)
            sys.exit(result.returncode)

    # Step 3: Persist scores (optional)
    if args.persist_scores:
        predictions_csv = run_dir / "predictions.csv"
        metrics_json = run_dir / "metrics.json"

        if not predictions_csv.exists():
            logger.error("predictions.csv not found in %s", run_dir)
            sys.exit(1)
        if not metrics_json.exists():
            logger.error("metrics.json not found in %s", run_dir)
            sys.exit(1)

        # Skip persistence when predictions have 0 data rows
        if not _csv_has_data_rows(predictions_csv):
            logger.warning("predictions.csv has 0 data rows; skipping score persistence.")
        else:
            logger.info("=" * 60)
            logger.info("STEP 3: PERSIST SCORES TO SQL")
            logger.info("=" * 60)

            write_cmd = build_write_scores_cmd(predictions_csv, metrics_json, table_name=args.scores_table_name)
            logger.info("Running: %s", ' '.join(write_cmd))
            result = subprocess.run(write_cmd)
            if result.returncode != 0:
                logger.error("Score persistence failed with exit code %d", result.returncode)
                sys.exit(result.returncode)

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETED")
    logger.info("=" * 60)
    logger.info("Run folder: %s", run_dir)


if __name__ == "__main__":
    main()
