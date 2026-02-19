"""Tests for execution/run_daily.py."""

import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "execution"))
from run_daily import compute_date_range, build_pipeline_cmd, SCRIPT_DIR


# ---------- compute_date_range ----------

class TestComputeDateRange:
    def test_default_90_days(self):
        with patch("run_daily.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2024, 6, 15, tzinfo=timezone.utc)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            since, until = compute_date_range(90)

        assert until == "2024-06-15"
        assert since == "2024-03-17"

    def test_custom_days_back(self):
        with patch("run_daily.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2024, 6, 15, tzinfo=timezone.utc)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            since, until = compute_date_range(30)

        assert until == "2024-06-15"
        assert since == "2024-05-16"

    def test_returns_iso_format_strings(self):
        since, until = compute_date_range(90)
        # Both should be YYYY-MM-DD format
        assert len(since) == 10
        assert len(until) == 10
        assert since[4] == "-"
        assert until[4] == "-"


# ---------- build_pipeline_cmd ----------

class TestBuildPipelineCmd:
    def _make_args(self, tmp_path, **overrides):
        defaults = dict(
            mode="predict",
            target="booked_call_within_7d",
            out_root=tmp_path / "runs",
            outcomes_query_file=tmp_path / "outcomes.sql",
            training_examples_query_file=tmp_path / "training.sql",
            scores_table_name="dbo.lead_scores",
            artifacts_dir=None,
            registry_dir=None,
            label_col=None,
            no_persist_scores=False,
        )
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    def test_contains_required_flags(self, tmp_path):
        args = self._make_args(tmp_path)
        cmd = build_pipeline_cmd(args, "2024-01-01", "2024-04-01")

        assert sys.executable == cmd[0]
        assert str(SCRIPT_DIR / "run_pipeline.py") in cmd
        assert "--mode" in cmd
        assert "predict" in cmd
        assert "--since" in cmd
        assert "2024-01-01" in cmd
        assert "--until" in cmd
        assert "2024-04-01" in cmd
        assert "--target" in cmd
        assert "booked_call_within_7d" in cmd
        assert "--persist-scores" in cmd

    def test_includes_registry_dir_when_provided(self, tmp_path):
        registry = tmp_path / "my_registry"
        args = self._make_args(tmp_path, registry_dir=registry)
        cmd = build_pipeline_cmd(args, "2024-01-01", "2024-04-01")

        assert "--registry-dir" in cmd
        idx = cmd.index("--registry-dir")
        assert cmd[idx + 1] == str(registry)

    def test_excludes_registry_dir_when_none(self, tmp_path):
        args = self._make_args(tmp_path, registry_dir=None)
        cmd = build_pipeline_cmd(args, "2024-01-01", "2024-04-01")

        assert "--registry-dir" not in cmd

    def test_includes_artifacts_dir_when_provided(self, tmp_path):
        artifacts = tmp_path / "my_artifacts"
        args = self._make_args(tmp_path, artifacts_dir=artifacts)
        cmd = build_pipeline_cmd(args, "2024-01-01", "2024-04-01")

        assert "--artifacts-dir" in cmd
        idx = cmd.index("--artifacts-dir")
        assert cmd[idx + 1] == str(artifacts)

    def test_no_persist_scores_override(self, tmp_path):
        args = self._make_args(tmp_path, no_persist_scores=True)
        cmd = build_pipeline_cmd(args, "2024-01-01", "2024-04-01")

        assert "--no-persist-scores" in cmd
        assert "--persist-scores" not in cmd

    def test_includes_label_col_when_provided(self, tmp_path):
        args = self._make_args(tmp_path, label_col="custom_label")
        cmd = build_pipeline_cmd(args, "2024-01-01", "2024-04-01")

        assert "--label-col" in cmd
        idx = cmd.index("--label-col")
        assert cmd[idx + 1] == "custom_label"

    def test_custom_scores_table_name(self, tmp_path):
        args = self._make_args(tmp_path, scores_table_name="dbo.custom_scores")
        cmd = build_pipeline_cmd(args, "2024-01-01", "2024-04-01")

        idx = cmd.index("--scores-table-name")
        assert cmd[idx + 1] == "dbo.custom_scores"
