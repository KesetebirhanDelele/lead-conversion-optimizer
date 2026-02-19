"""Tests for execution/promote_model.py."""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "execution"))
from promote_model import validate_run_dir, check_pr_auc_gate, build_model_card, promote


# ---------- helpers ----------

def _create_run_dir(tmp_path, pr_auc=0.25):
    """Create a fake run directory with required artifacts."""
    run_dir = tmp_path / "run_1"
    run_dir.mkdir()
    (run_dir / "model.joblib").write_text("fake_model", encoding="utf-8")
    (run_dir / "scaler.joblib").write_text("fake_scaler", encoding="utf-8")
    metrics = {
        "label_col": "label_responded_within_7d",
        "feature_cols": ["attempts_sms_24h", "attempts_email_24h",
                         "attempts_voice_no_voicemail_24h", "voicemail_drops_24h"],
        "train": {"n_samples": 100, "pr_auc": 0.30},
        "test": {"n_samples": 25, "pr_auc": pr_auc},
        "model_path": str(run_dir / "model.joblib"),
        "scaler_path": str(run_dir / "scaler.joblib"),
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    return run_dir, metrics


# ---------- validate_run_dir ----------

class TestValidateRunDir:
    def test_passes_with_all_files(self, tmp_path):
        run_dir, _ = _create_run_dir(tmp_path)
        validate_run_dir(run_dir)  # should not raise

    def test_fails_when_model_missing(self, tmp_path):
        run_dir, _ = _create_run_dir(tmp_path)
        (run_dir / "model.joblib").unlink()
        with pytest.raises(FileNotFoundError, match="model.joblib"):
            validate_run_dir(run_dir)

    def test_fails_when_metrics_missing(self, tmp_path):
        run_dir, _ = _create_run_dir(tmp_path)
        (run_dir / "metrics.json").unlink()
        with pytest.raises(FileNotFoundError, match="metrics.json"):
            validate_run_dir(run_dir)

    def test_fails_when_dir_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="does not exist"):
            validate_run_dir(tmp_path / "nonexistent")


# ---------- check_pr_auc_gate ----------

class TestCheckPrAucGate:
    def test_passes_when_above_threshold(self):
        metrics = {"test": {"pr_auc": 0.25}}
        passed, actual = check_pr_auc_gate(metrics, min_pr_auc=0.20)
        assert passed is True
        assert actual == 0.25

    def test_fails_when_below_threshold(self):
        metrics = {"test": {"pr_auc": 0.10}}
        passed, actual = check_pr_auc_gate(metrics, min_pr_auc=0.20)
        assert passed is False
        assert actual == 0.10

    def test_skips_when_no_test_pr_auc(self):
        metrics = {"test": {"n_samples": 25}}
        passed, actual = check_pr_auc_gate(metrics, min_pr_auc=0.20)
        assert passed is True
        assert actual is None

    def test_skips_when_no_test_key(self):
        metrics = {"train": {"pr_auc": 0.30}}
        passed, actual = check_pr_auc_gate(metrics, min_pr_auc=0.20)
        assert passed is True
        assert actual is None

    def test_passes_when_min_pr_auc_is_none(self):
        metrics = {"test": {"pr_auc": 0.01}}
        passed, actual = check_pr_auc_gate(metrics, min_pr_auc=None)
        assert passed is True
        assert actual is None


# ---------- build_model_card ----------

class TestBuildModelCard:
    def test_contains_expected_keys(self, tmp_path):
        run_dir, metrics = _create_run_dir(tmp_path)
        card = build_model_card(run_dir, metrics, notes="test note")
        assert "promoted_at_utc" in card
        assert card["source_run_dir"] == str(run_dir)
        assert card["label_col"] == "label_responded_within_7d"
        assert len(card["feature_cols"]) == 4
        assert card["notes"] == "test note"
        assert "metrics_snippet" in card
        assert "train" in card["metrics_snippet"]
        assert "test" in card["metrics_snippet"]


# ---------- promote (full integration) ----------

class TestPromote:
    def test_creates_version_and_active_dirs(self, tmp_path):
        run_dir, _ = _create_run_dir(tmp_path)
        registry = tmp_path / "registry"

        version_dir = promote(run_dir, registry)

        # Version directory created with artifacts
        assert version_dir.exists()
        assert (version_dir / "model.joblib").exists()
        assert (version_dir / "scaler.joblib").exists()
        assert (version_dir / "metrics.json").exists()
        assert (version_dir / "model_card.json").exists()

        # Active directory created
        active = registry / "active"
        assert active.is_dir()
        assert (active / "model.joblib").exists()
        assert (active / "scaler.joblib").exists()
        assert (active / "model_card.json").exists()

    def test_model_card_contents(self, tmp_path):
        run_dir, _ = _create_run_dir(tmp_path)
        registry = tmp_path / "registry"

        promote(run_dir, registry, notes="my notes")

        card_path = registry / "active" / "model_card.json"
        card = json.loads(card_path.read_text(encoding="utf-8"))
        assert card["label_col"] == "label_responded_within_7d"
        assert len(card["feature_cols"]) == 4
        assert card["notes"] == "my notes"
        assert card["promoted_at_utc"] is not None
        assert card["source_run_dir"] == str(run_dir.resolve())

    def test_active_content_matches_source(self, tmp_path):
        run_dir, _ = _create_run_dir(tmp_path)
        registry = tmp_path / "registry"

        promote(run_dir, registry)

        # Active model.joblib should have the same content as source
        active_model = (registry / "active" / "model.joblib").read_text(encoding="utf-8")
        assert active_model == "fake_model"

    def test_second_promote_replaces_active(self, tmp_path):
        run_dir1, _ = _create_run_dir(tmp_path)
        registry = tmp_path / "registry"

        # Use distinct timestamps so version dirs don't collide
        with patch("promote_model.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            promote(run_dir1, registry)

        # Create second run with different content
        run_dir2 = tmp_path / "run_2"
        run_dir2.mkdir()
        (run_dir2 / "model.joblib").write_text("model_v2", encoding="utf-8")
        (run_dir2 / "scaler.joblib").write_text("scaler_v2", encoding="utf-8")
        metrics2 = {"label_col": "custom_label", "feature_cols": ["a"], "test": {}, "train": {}}
        (run_dir2 / "metrics.json").write_text(json.dumps(metrics2), encoding="utf-8")

        with patch("promote_model.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2024, 6, 15, 13, 0, 0, tzinfo=timezone.utc)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            promote(run_dir2, registry)

        active_model = (registry / "active" / "model.joblib").read_text(encoding="utf-8")
        assert active_model == "model_v2"

        # Should have 2 versions
        versions = list((registry / "versions").iterdir())
        assert len(versions) == 2

    def test_min_pr_auc_blocks_promotion(self, tmp_path):
        run_dir, _ = _create_run_dir(tmp_path, pr_auc=0.05)
        registry = tmp_path / "registry"

        with pytest.raises(SystemExit) as exc_info:
            promote(run_dir, registry, min_pr_auc=0.20)
        assert exc_info.value.code == 1

        # Active should NOT exist
        assert not (registry / "active").exists()

    def test_force_skips_pr_auc_check(self, tmp_path):
        run_dir, _ = _create_run_dir(tmp_path, pr_auc=0.05)
        registry = tmp_path / "registry"

        # With --force, promotion should succeed despite low PR-AUC
        version_dir = promote(run_dir, registry, force=True, min_pr_auc=0.20)
        assert version_dir.exists()
        assert (registry / "active" / "model.joblib").exists()
