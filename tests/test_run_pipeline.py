"""Tests for execution/run_pipeline.py."""

import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "execution"))
from run_pipeline import build_run_folder_name, build_extract_cmd, build_train_cmd, SCRIPT_DIR


# ---------- run folder naming ----------

class TestBuildRunFolderName:
    def test_contains_since_until_target(self):
        name = build_run_folder_name("2024-01-01", "2024-02-01", "booked_call_within_7d")
        assert name.startswith("run_2024-01-01_2024-02-01_booked_call_within_7d_")

    def test_ends_with_utc_timestamp(self):
        name = build_run_folder_name("2024-01-01", "2024-02-01", "booked_call_within_7d")
        # Expect suffix like _20240115T143218Z
        suffix = name.split("booked_call_within_7d_")[1]
        assert re.match(r"^\d{8}T\d{6}Z$", suffix), f"Unexpected timestamp format: {suffix}"

    def test_different_targets_produce_different_names(self):
        a = build_run_folder_name("2024-01-01", "2024-02-01", "alpha")
        b = build_run_folder_name("2024-01-01", "2024-02-01", "beta")
        # Strip timestamps for comparison
        assert a.rsplit("_", 1)[0] != b.rsplit("_", 1)[0]


# ---------- subprocess command construction ----------

class TestBuildExtractCmd:
    def _make_args(self, tmp_path):
        return SimpleNamespace(
            since="2024-01-01",
            until="2024-02-01",
            target="booked_call_within_7d",
            outcomes_query_file=tmp_path / "outcomes.sql",
            training_examples_query_file=tmp_path / "training.sql",
        )

    def test_extract_cmd_contains_required_flags(self, tmp_path):
        args = self._make_args(tmp_path)
        run_dir = tmp_path / "run_test"
        cmd = build_extract_cmd(args, run_dir)

        assert sys.executable == cmd[0]
        assert str(SCRIPT_DIR / "extract_snapshot.py") in cmd
        assert "--since" in cmd
        assert "2024-01-01" in cmd
        assert "--until" in cmd
        assert "2024-02-01" in cmd
        assert "--target" in cmd
        assert "booked_call_within_7d" in cmd
        assert "--out" in cmd
        assert str(run_dir) in cmd
        assert "--outcomes-query-file" in cmd
        assert "--training-examples-query-file" in cmd

    def test_extract_cmd_uses_correct_script_path(self, tmp_path):
        args = self._make_args(tmp_path)
        cmd = build_extract_cmd(args, tmp_path)
        script_path = Path(cmd[1])
        assert script_path.name == "extract_snapshot.py"


class TestBuildTrainCmd:
    def test_train_cmd_without_label_col(self, tmp_path):
        csv_path = tmp_path / "training_examples.csv"
        cmd = build_train_cmd(csv_path)
        assert "--training-examples-csv" in cmd
        assert str(csv_path) in cmd
        assert "--label-col" not in cmd

    def test_train_cmd_with_label_col(self, tmp_path):
        csv_path = tmp_path / "training_examples.csv"
        cmd = build_train_cmd(csv_path, label_col="label_responded_within_14d")
        assert "--label-col" in cmd
        idx = cmd.index("--label-col")
        assert cmd[idx + 1] == "label_responded_within_14d"


# ---------- fail-fast when training_examples.csv missing ----------

class TestMissingTrainingCsv:
    def test_exits_nonzero_when_csv_missing(self, tmp_path):
        """Pipeline must fail if extract succeeds but training_examples.csv is absent."""
        script = str(Path(__file__).resolve().parent.parent / "execution" / "run_pipeline.py")

        # Create dummy SQL files so argparse doesn't complain
        outcomes_sql = tmp_path / "outcomes.sql"
        outcomes_sql.write_text("SELECT 1", encoding="utf-8")
        training_sql = tmp_path / "training.sql"
        training_sql.write_text("SELECT 1", encoding="utf-8")

        # Mock extract_snapshot.py: create a fake script that exits 0 but writes NO csv
        fake_extract = tmp_path / "fake_extract.py"
        fake_extract.write_text("import sys; sys.exit(0)", encoding="utf-8")

        # We patch SCRIPT_DIR at the module level so run_pipeline uses our fake
        # Instead, run via subprocess and mock extract by patching the command
        # Simpler: use subprocess with a modified extract that does nothing
        with patch("run_pipeline.subprocess.run") as mock_run:
            # First call (extract) succeeds
            mock_run.return_value = MagicMock(returncode=0)

            from run_pipeline import main
            out_root = tmp_path / "runs"

            test_args = [
                "--since", "2024-01-01",
                "--until", "2024-02-01",
                "--target", "booked_call_within_7d",
                "--out-root", str(out_root),
                "--outcomes-query-file", str(outcomes_sql),
                "--training-examples-query-file", str(training_sql),
            ]

            with patch("sys.argv", ["run_pipeline.py"] + test_args):
                with pytest.raises(SystemExit) as exc_info:
                    main()
                assert exc_info.value.code == 1


# ---------- full pipeline with mocked subprocesses ----------

class TestPipelineOrchestration:
    def test_calls_extract_then_train(self, tmp_path):
        """Pipeline calls extract, then train when training_examples.csv exists."""
        from run_pipeline import main

        out_root = tmp_path / "runs"
        outcomes_sql = tmp_path / "outcomes.sql"
        outcomes_sql.write_text("SELECT 1", encoding="utf-8")
        training_sql = tmp_path / "training.sql"
        training_sql.write_text("SELECT 1", encoding="utf-8")

        call_log = []

        def fake_subprocess_run(cmd, **kwargs):
            call_log.append(cmd)
            # After "extract" call, create training_examples.csv in the run dir
            if "extract_snapshot.py" in cmd[1]:
                # Find the --out value
                out_idx = cmd.index("--out")
                run_dir = Path(cmd[out_idx + 1])
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "training_examples.csv").write_text("header\n", encoding="utf-8")
            return MagicMock(returncode=0)

        test_args = [
            "--since", "2024-01-01",
            "--until", "2024-02-01",
            "--target", "booked_call_within_7d",
            "--out-root", str(out_root),
            "--outcomes-query-file", str(outcomes_sql),
            "--training-examples-query-file", str(training_sql),
        ]

        with patch("run_pipeline.subprocess.run", side_effect=fake_subprocess_run):
            with patch("sys.argv", ["run_pipeline.py"] + test_args):
                main()

        assert len(call_log) == 2, f"Expected 2 subprocess calls, got {len(call_log)}"
        assert "extract_snapshot.py" in call_log[0][1]
        assert "train_baseline.py" in call_log[1][1]
