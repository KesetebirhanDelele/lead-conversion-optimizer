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
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


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
    parser.add_argument("--artifacts-dir", type=Path, default=None, help="Directory containing model.joblib/scaler.joblib for predict mode (overrides run_dir)")

    args = parser.parse_args()

    # Create run folder
    run_name = build_run_folder_name(args.since, args.until, args.target)
    run_dir = args.out_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run folder: {run_dir}")

    # Step 1: Extract snapshot
    print("\n" + "=" * 60)
    print("STEP 1: EXTRACT SNAPSHOT")
    print("=" * 60)

    extract_cmd = build_extract_cmd(args, run_dir)
    print(f"Running: {' '.join(extract_cmd)}")
    result = subprocess.run(extract_cmd)
    if result.returncode != 0:
        print(f"Extraction failed with exit code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)

    # Fail fast if training_examples.csv is missing
    training_csv = run_dir / "training_examples.csv"
    if not training_csv.exists():
        print(f"training_examples.csv not found in {run_dir}", file=sys.stderr)
        sys.exit(1)

    # Step 2: Train or Predict
    if args.mode == "train":
        print("\n" + "=" * 60)
        print("STEP 2: TRAIN BASELINE MODEL")
        print("=" * 60)

        train_cmd = build_train_cmd(training_csv, label_col=args.label_col)
        print(f"Running: {' '.join(train_cmd)}")
        result = subprocess.run(train_cmd)
        if result.returncode != 0:
            print(f"Training failed with exit code {result.returncode}", file=sys.stderr)
            sys.exit(result.returncode)

    elif args.mode == "predict":
        print("\n" + "=" * 60)
        print("STEP 2: PREDICT (score only)")
        print("=" * 60)

        # If --artifacts-dir is provided, copy model artifacts into run_dir
        if args.artifacts_dir is not None:
            if not args.artifacts_dir.is_dir():
                print(f"--artifacts-dir does not exist or is not a directory: {args.artifacts_dir}", file=sys.stderr)
                sys.exit(1)
            for artifact in ("model.joblib", "scaler.joblib"):
                src = args.artifacts_dir / artifact
                if not src.exists():
                    print(f"{artifact} not found in --artifacts-dir {args.artifacts_dir}", file=sys.stderr)
                    sys.exit(1)
                shutil.copy2(src, run_dir / artifact)
                print(f"Copied {artifact} from {args.artifacts_dir} to {run_dir}")

        # Verify model artifacts exist before calling predict.py
        model_path = run_dir / "model.joblib"
        scaler_path = run_dir / "scaler.joblib"
        if not model_path.exists():
            print(f"model.joblib not found in {run_dir}. Run training first or copy artifacts into the run folder.", file=sys.stderr)
            sys.exit(1)
        if not scaler_path.exists():
            print(f"scaler.joblib not found in {run_dir}. Run training first or copy artifacts into the run folder.", file=sys.stderr)
            sys.exit(1)

        predict_cmd = build_predict_cmd(training_csv)
        print(f"Running: {' '.join(predict_cmd)}")
        result = subprocess.run(predict_cmd)
        if result.returncode != 0:
            print(f"Prediction failed with exit code {result.returncode}", file=sys.stderr)
            sys.exit(result.returncode)

    # Step 3: Persist scores (optional)
    if args.persist_scores:
        predictions_csv = run_dir / "predictions.csv"
        metrics_json = run_dir / "metrics.json"

        if not predictions_csv.exists():
            print(f"predictions.csv not found in {run_dir} — skipping score persistence", file=sys.stderr)
            sys.exit(1)
        if not metrics_json.exists():
            print(f"metrics.json not found in {run_dir} — skipping score persistence", file=sys.stderr)
            sys.exit(1)

        print("\n" + "=" * 60)
        print("STEP 3: PERSIST SCORES TO SQL")
        print("=" * 60)

        write_cmd = build_write_scores_cmd(predictions_csv, metrics_json, table_name=args.scores_table_name)
        print(f"Running: {' '.join(write_cmd)}")
        result = subprocess.run(write_cmd)
        if result.returncode != 0:
            print(f"Score persistence failed with exit code {result.returncode}", file=sys.stderr)
            sys.exit(result.returncode)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED")
    print("=" * 60)
    print(f"Run folder: {run_dir}")


if __name__ == "__main__":
    main()
