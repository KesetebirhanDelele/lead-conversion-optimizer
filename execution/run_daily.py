"""
Daily scoring helper — thin wrapper around run_pipeline.py.

Computes --since/--until from --days-back (default 90) and invokes
run_pipeline.py as a subprocess with persist-scores enabled.

Usage (predict with 90-day window, registry model):
    python execution/run_daily.py `
        --registry-dir ./tmp/model_registry `
        --outcomes-query-file ./queries/outcomes.sql `
        --training-examples-query-file ./queries/gold_training_examples_proxy_v2.sql

Usage (predict with explicit artifacts):
    python execution/run_daily.py `
        --artifacts-dir ./tmp/runs/latest_train_run `
        --outcomes-query-file ./queries/outcomes.sql `
        --training-examples-query-file ./queries/gold_training_examples_proxy_v2.sql

Usage (train with 120-day window):
    python execution/run_daily.py --mode train --days-back 120 `
        --outcomes-query-file ./queries/outcomes.sql `
        --training-examples-query-file ./queries/gold_training_examples_proxy_v2.sql
"""

import argparse
import logging
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from config import LOG_LEVEL
from logging_utils import setup_logging


logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent


def compute_date_range(days_back):
    """Return (since, until) as YYYY-MM-DD strings based on days_back from today UTC."""
    today = datetime.now(timezone.utc).date()
    since = today - timedelta(days=days_back)
    return since.isoformat(), today.isoformat()


def build_pipeline_cmd(args, since, until):
    """Build the subprocess command list for run_pipeline.py."""
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "run_pipeline.py"),
        "--mode", args.mode,
        "--since", since,
        "--until", until,
        "--target", args.target,
        "--out-root", str(args.out_root),
        "--outcomes-query-file", str(args.outcomes_query_file),
        "--training-examples-query-file", str(args.training_examples_query_file),
        "--persist-scores",
        "--scores-table-name", args.scores_table_name,
    ]
    if args.artifacts_dir is not None:
        cmd.extend(["--artifacts-dir", str(args.artifacts_dir)])
    if args.registry_dir is not None:
        cmd.extend(["--registry-dir", str(args.registry_dir)])
    if args.label_col is not None:
        cmd.extend(["--label-col", args.label_col])
    if args.no_persist_scores:
        # Override: remove --persist-scores and add --no-persist-scores
        cmd.remove("--persist-scores")
        cmd.append("--no-persist-scores")
    return cmd


def main():
    parser = argparse.ArgumentParser(
        description="Daily scoring job — wraps run_pipeline.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--days-back", type=int, default=90,
                        help="Number of days to look back for --since (default: 90)")
    parser.add_argument("--mode", choices=["train", "predict"], default="predict",
                        help="Pipeline mode (default: predict)")
    parser.add_argument("--target", default="booked_call_within_7d",
                        help="Target variable (default: booked_call_within_7d)")
    parser.add_argument("--out-root", type=Path, default=Path("tmp/runs"),
                        help="Root directory for run folders (default: tmp/runs)")
    parser.add_argument("--outcomes-query-file", required=True, type=Path,
                        help="SQL file for outcomes extraction")
    parser.add_argument("--training-examples-query-file", required=True, type=Path,
                        help="SQL file for training examples extraction")
    parser.add_argument("--artifacts-dir", type=Path, default=None,
                        help="Directory containing model.joblib/scaler.joblib (predict mode)")
    parser.add_argument("--registry-dir", type=Path, default=None,
                        help="Model registry directory; predict mode loads from registry-dir/active/ when --artifacts-dir not provided")
    parser.add_argument("--label-col", default=None,
                        help="Label column (pass-through to run_pipeline.py)")
    parser.add_argument("--scores-table-name", default="dbo.lead_scores",
                        help="Target SQL table for scores (default: dbo.lead_scores)")
    parser.add_argument("--no-persist-scores", action="store_true", default=False,
                        help="Skip score persistence to SQL")
    parser.add_argument("--log-level", default=None,
                        help="Log level: DEBUG, INFO, WARNING, ERROR (default: from LOG_LEVEL env var)")

    args = parser.parse_args()

    setup_logging(args.log_level or LOG_LEVEL)

    since, until = compute_date_range(args.days_back)
    logger.info("Date range: %s to %s (%d days back)", since, until, args.days_back)

    cmd = build_pipeline_cmd(args, since, until)
    logger.info("Running: %s", ' '.join(cmd))

    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
