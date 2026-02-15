"""
End-to-end pipeline: extract snapshot then train baseline model.

Orchestration-only — no business logic. Invokes extract_snapshot.py
and train_baseline.py as subprocesses in sequence.

Usage:
    python run_pipeline.py \
        --since 2024-01-01 --until 2024-02-01 \
        --target booked_call_within_7d \
        --out-root ./tmp/runs \
        --outcomes-query-file ./sql/outcomes.sql \
        --training-examples-query-file ./sql/training_examples.sql
"""

import argparse
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


def main():
    parser = argparse.ArgumentParser(
        description="Run extract + train pipeline end-to-end",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--since", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--until", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--target", required=True, help="Target variable name (pass-through to extract)")
    parser.add_argument("--out-root", required=True, type=Path, help="Root directory for run folders")
    parser.add_argument("--outcomes-query-file", required=True, type=Path, help="SQL file for outcomes extraction")
    parser.add_argument("--training-examples-query-file", required=True, type=Path, help="SQL file for training examples extraction")
    parser.add_argument("--label-col", default=None, help="Label column (pass-through to train_baseline.py)")

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

    # Step 2: Train baseline model
    print("\n" + "=" * 60)
    print("STEP 2: TRAIN BASELINE MODEL")
    print("=" * 60)

    train_cmd = build_train_cmd(training_csv, label_col=args.label_col)
    print(f"Running: {' '.join(train_cmd)}")
    result = subprocess.run(train_cmd)
    if result.returncode != 0:
        print(f"Training failed with exit code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED")
    print("=" * 60)
    print(f"Run folder: {run_dir}")


if __name__ == "__main__":
    main()
