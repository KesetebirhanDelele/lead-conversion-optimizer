"""
Promote a trained model run to the model registry.

Copies model.joblib, scaler.joblib, and metrics.json into a versioned
registry folder and atomically updates the active/ symlink directory.

Usage (PowerShell):
    python execution/promote_model.py `
        --run-dir ./tmp/runs/<completed_run>

    python execution/promote_model.py `
        --run-dir ./tmp/runs/<completed_run> `
        --registry-dir ./tmp/model_registry `
        --min-pr-auc 0.15 `
        --notes "first promoted model"
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from config import LOG_LEVEL, DEFAULT_REGISTRY_DIR
from logging_utils import setup_logging

logger = logging.getLogger(__name__)

REQUIRED_ARTIFACTS = ["model.joblib", "scaler.joblib", "metrics.json"]


def validate_run_dir(run_dir):
    """Check that run_dir contains model.joblib, scaler.joblib, metrics.json."""
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
    for artifact in REQUIRED_ARTIFACTS:
        path = run_dir / artifact
        if not path.exists():
            raise FileNotFoundError(f"Required artifact missing: {path}")


def load_metrics(run_dir):
    """Load and return metrics.json from a run directory."""
    metrics_path = run_dir / "metrics.json"
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def check_pr_auc_gate(metrics, min_pr_auc):
    """Check whether test PR-AUC meets the minimum threshold.

    Returns:
        (passed, actual_value) where passed is bool and actual_value is
        the PR-AUC float or None if the key was not present.
    """
    if min_pr_auc is None:
        return True, None

    test_metrics = metrics.get("test", {})
    actual = test_metrics.get("pr_auc")
    if actual is None:
        # No test PR-AUC available — cannot enforce gate
        return True, None

    return actual >= min_pr_auc, actual


def _get_git_commit():
    """Return the current git commit hash, or None on failure."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def build_model_card(run_dir, metrics, notes=None):
    """Build a model card dict for the promoted model."""
    return {
        "promoted_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_run_dir": str(run_dir),
        "git_commit": _get_git_commit(),
        "label_col": metrics.get("label_col"),
        "feature_cols": metrics.get("feature_cols"),
        "metrics_snippet": {
            "train": metrics.get("train"),
            "test": metrics.get("test"),
        },
        "notes": notes,
    }


def promote(run_dir, registry_dir, force=False, min_pr_auc=None, notes=None):
    """Promote a trained run to the model registry.

    1. Validate required files exist in run_dir
    2. Check PR-AUC gate (unless --force)
    3. Copy artifacts into registry/versions/<timestamp>/
    4. Atomically update registry/active/
    5. Write model_card.json
    """
    run_dir = Path(run_dir).resolve()
    registry_dir = Path(registry_dir).resolve()

    # Step 1: validate
    validate_run_dir(run_dir)
    metrics = load_metrics(run_dir)
    logger.info("Validated run directory: %s", run_dir)

    # Step 2: PR-AUC gate
    if not force and min_pr_auc is not None:
        passed, actual = check_pr_auc_gate(metrics, min_pr_auc)
        if not passed:
            logger.error(
                "PR-AUC gate failed: test PR-AUC %.4f < threshold %.4f. "
                "Use --force to override.",
                actual, min_pr_auc,
            )
            sys.exit(1)
        if actual is not None:
            logger.info("PR-AUC gate passed: %.4f >= %.4f", actual, min_pr_auc)

    # Step 3: create versioned copy
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    version_dir = registry_dir / "versions" / ts
    version_dir.mkdir(parents=True, exist_ok=True)

    for artifact in REQUIRED_ARTIFACTS:
        shutil.copy2(run_dir / artifact, version_dir / artifact)
    logger.info("Artifacts copied to version: %s", version_dir)

    # Build and write model card
    model_card = build_model_card(run_dir, metrics, notes=notes)
    card_path = version_dir / "model_card.json"
    with open(card_path, "w", encoding="utf-8") as f:
        json.dump(model_card, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")
    logger.info("Model card written to %s", card_path)

    # Step 4: atomically update active/ (Windows-safe)
    active_dir = registry_dir / "active"
    tmp_dir = registry_dir / f"_active_tmp_{os.getpid()}"
    old_dir = registry_dir / f"_active_old_{os.getpid()}"

    # Clean up any leftover temp dirs from previous failed runs
    for leftover in (tmp_dir, old_dir):
        if leftover.exists():
            shutil.rmtree(leftover)

    # Copy version files into temp staging dir
    shutil.copytree(version_dir, tmp_dir)

    # Swap: move current active aside, rename temp to active
    if active_dir.exists():
        active_dir.rename(old_dir)
    tmp_dir.rename(active_dir)

    # Clean up old active
    if old_dir.exists():
        shutil.rmtree(old_dir)

    logger.info("Active model updated: %s", active_dir)
    logger.info("Promotion complete. Version: %s", ts)

    return version_dir


def main():
    parser = argparse.ArgumentParser(
        description="Promote a trained model to the model registry",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--run-dir", required=True, type=Path,
                        help="Path to completed training run directory")
    parser.add_argument("--registry-dir", type=Path, default=DEFAULT_REGISTRY_DIR,
                        help=f"Registry root directory (default: {DEFAULT_REGISTRY_DIR})")
    parser.add_argument("--force", action="store_true",
                        help="Allow promotion without PR-AUC gate check")
    parser.add_argument("--min-pr-auc", type=float, default=None,
                        help="Minimum test PR-AUC required for promotion")
    parser.add_argument("--notes", type=str, default=None,
                        help="Free-text notes to include in model card")
    parser.add_argument("--log-level", default=None,
                        help="Log level: DEBUG, INFO, WARNING, ERROR (default: from LOG_LEVEL env var)")

    args = parser.parse_args()

    setup_logging(args.log_level or LOG_LEVEL)

    logger.info("Promoting run: %s -> %s", args.run_dir, args.registry_dir)

    try:
        promote(
            run_dir=args.run_dir,
            registry_dir=args.registry_dir,
            force=args.force,
            min_pr_auc=args.min_pr_auc,
            notes=args.notes,
        )
    except FileNotFoundError as e:
        logger.error("%s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
