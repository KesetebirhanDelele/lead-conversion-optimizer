"""Tests for execution/run_pipeline.py."""

import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "execution"))
from run_pipeline import build_run_folder_name, build_extract_cmd, build_train_cmd, build_predict_cmd, build_write_scores_cmd, _csv_has_data_rows, SCRIPT_DIR


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
                (run_dir / "training_examples.csv").write_text("header\nrow\n", encoding="utf-8")
            return MagicMock(returncode=0)

        test_args = [
            "--since", "2024-01-01",
            "--until", "2024-02-01",
            "--target", "booked_call_within_7d",
            "--out-root", str(out_root),
            "--outcomes-query-file", str(outcomes_sql),
            "--training-examples-query-file", str(training_sql),
            "--no-persist-scores",
        ]

        with patch("run_pipeline.subprocess.run", side_effect=fake_subprocess_run):
            with patch("sys.argv", ["run_pipeline.py"] + test_args):
                main()

        assert len(call_log) == 2, f"Expected 2 subprocess calls, got {len(call_log)}"
        assert "extract_snapshot.py" in call_log[0][1]
        assert "train_baseline.py" in call_log[1][1]


# ---------- build_predict_cmd ----------

class TestBuildPredictCmd:
    def test_predict_cmd_uses_correct_script(self, tmp_path):
        csv_path = tmp_path / "training_examples.csv"
        cmd = build_predict_cmd(csv_path)
        assert sys.executable == cmd[0]
        script_path = Path(cmd[1])
        assert script_path.name == "predict.py"

    def test_predict_cmd_includes_csv_flag(self, tmp_path):
        csv_path = tmp_path / "training_examples.csv"
        cmd = build_predict_cmd(csv_path)
        assert "--training-examples-csv" in cmd
        assert str(csv_path) in cmd

    def test_predict_cmd_does_not_include_label_col(self, tmp_path):
        csv_path = tmp_path / "training_examples.csv"
        cmd = build_predict_cmd(csv_path)
        assert "--label-col" not in cmd


# ---------- predict mode orchestration ----------

def _base_args(tmp_path):
    """Build base CLI args list for pipeline tests."""
    outcomes_sql = tmp_path / "outcomes.sql"
    outcomes_sql.write_text("SELECT 1", encoding="utf-8")
    training_sql = tmp_path / "training.sql"
    training_sql.write_text("SELECT 1", encoding="utf-8")
    return [
        "--since", "2024-01-01",
        "--until", "2024-02-01",
        "--target", "booked_call_within_7d",
        "--out-root", str(tmp_path / "runs"),
        "--outcomes-query-file", str(outcomes_sql),
        "--training-examples-query-file", str(training_sql),
    ]


class TestPredictModeOrchestration:
    def test_calls_extract_then_predict(self, tmp_path):
        """In predict mode, pipeline calls extract then predict.py (not train)."""
        from run_pipeline import main

        call_log = []

        def fake_subprocess_run(cmd, **kwargs):
            call_log.append(cmd)
            if "extract_snapshot.py" in cmd[1]:
                out_idx = cmd.index("--out")
                run_dir = Path(cmd[out_idx + 1])
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "training_examples.csv").write_text("h\nr\n", encoding="utf-8")
                (run_dir / "model.joblib").write_text("fake", encoding="utf-8")
                (run_dir / "scaler.joblib").write_text("fake", encoding="utf-8")
            return MagicMock(returncode=0)

        test_args = _base_args(tmp_path) + ["--mode", "predict", "--no-persist-scores"]

        with patch("run_pipeline.subprocess.run", side_effect=fake_subprocess_run):
            with patch("sys.argv", ["run_pipeline.py"] + test_args):
                main()

        assert len(call_log) == 2
        assert "extract_snapshot.py" in call_log[0][1]
        assert "predict.py" in call_log[1][1]
        assert "train_baseline.py" not in call_log[1][1]


class TestPredictModeMissingCsv:
    def test_exits_nonzero_when_csv_missing(self, tmp_path):
        """Predict mode must fail if training_examples.csv absent after extraction."""
        from run_pipeline import main

        def fake_subprocess_run(cmd, **kwargs):
            if "extract_snapshot.py" in cmd[1]:
                out_idx = cmd.index("--out")
                run_dir = Path(cmd[out_idx + 1])
                run_dir.mkdir(parents=True, exist_ok=True)
                # Do NOT create training_examples.csv
            return MagicMock(returncode=0)

        test_args = _base_args(tmp_path) + ["--mode", "predict"]

        with patch("run_pipeline.subprocess.run", side_effect=fake_subprocess_run):
            with patch("sys.argv", ["run_pipeline.py"] + test_args):
                with pytest.raises(SystemExit) as exc_info:
                    main()
                assert exc_info.value.code == 1


class TestPredictModeMissingArtifacts:
    def test_exits_nonzero_when_model_missing(self, tmp_path):
        """Predict mode must fail if model.joblib is absent (no predict.py call)."""
        from run_pipeline import main

        call_log = []

        def fake_subprocess_run(cmd, **kwargs):
            call_log.append(cmd)
            if "extract_snapshot.py" in cmd[1]:
                out_idx = cmd.index("--out")
                run_dir = Path(cmd[out_idx + 1])
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "training_examples.csv").write_text("h\nr\n", encoding="utf-8")
                # model.joblib missing, scaler present
                (run_dir / "scaler.joblib").write_text("fake", encoding="utf-8")
            return MagicMock(returncode=0)

        test_args = _base_args(tmp_path) + ["--mode", "predict"]

        with patch("run_pipeline.subprocess.run", side_effect=fake_subprocess_run):
            with patch("sys.argv", ["run_pipeline.py"] + test_args):
                with pytest.raises(SystemExit) as exc_info:
                    main()
                assert exc_info.value.code == 1

        # predict.py must NOT have been called
        assert len(call_log) == 1
        assert "extract_snapshot.py" in call_log[0][1]

    def test_exits_nonzero_when_scaler_missing(self, tmp_path):
        """Predict mode must fail if scaler.joblib is absent."""
        from run_pipeline import main

        call_log = []

        def fake_subprocess_run(cmd, **kwargs):
            call_log.append(cmd)
            if "extract_snapshot.py" in cmd[1]:
                out_idx = cmd.index("--out")
                run_dir = Path(cmd[out_idx + 1])
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "training_examples.csv").write_text("h\nr\n", encoding="utf-8")
                # scaler.joblib missing, model present
                (run_dir / "model.joblib").write_text("fake", encoding="utf-8")
            return MagicMock(returncode=0)

        test_args = _base_args(tmp_path) + ["--mode", "predict"]

        with patch("run_pipeline.subprocess.run", side_effect=fake_subprocess_run):
            with patch("sys.argv", ["run_pipeline.py"] + test_args):
                with pytest.raises(SystemExit) as exc_info:
                    main()
                assert exc_info.value.code == 1

        assert len(call_log) == 1


# ---------- build_write_scores_cmd ----------

class TestBuildWriteScoresCmd:
    def test_uses_correct_script(self, tmp_path):
        cmd = build_write_scores_cmd(tmp_path / "predictions.csv", tmp_path / "metrics.json")
        assert sys.executable == cmd[0]
        script_path = Path(cmd[1])
        assert script_path.name == "write_scores_to_sql.py"

    def test_includes_required_flags(self, tmp_path):
        preds = tmp_path / "predictions.csv"
        metrics = tmp_path / "metrics.json"
        cmd = build_write_scores_cmd(preds, metrics, table_name="dbo.custom_table")
        assert "--predictions-csv" in cmd
        assert str(preds) in cmd
        assert "--metrics-json" in cmd
        assert str(metrics) in cmd
        assert "--table-name" in cmd
        assert "dbo.custom_table" in cmd

    def test_default_table_name(self, tmp_path):
        cmd = build_write_scores_cmd(tmp_path / "p.csv", tmp_path / "m.json")
        idx = cmd.index("--table-name")
        assert cmd[idx + 1] == "dbo.lead_scores"


# ---------- persist-scores orchestration ----------

class TestPersistScoresOrchestration:
    def test_three_step_pipeline_train(self, tmp_path):
        """Train mode: extract -> train -> write_scores (3 subprocess calls)."""
        from run_pipeline import main

        call_log = []

        def fake_subprocess_run(cmd, **kwargs):
            call_log.append(cmd)
            if "extract_snapshot.py" in cmd[1]:
                out_idx = cmd.index("--out")
                run_dir = Path(cmd[out_idx + 1])
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "training_examples.csv").write_text("h\nr\n", encoding="utf-8")
            elif "train_baseline.py" in cmd[1]:
                csv_idx = cmd.index("--training-examples-csv")
                csv_path = Path(cmd[csv_idx + 1])
                run_dir = csv_path.parent
                (run_dir / "predictions.csv").write_text("h\nr\n", encoding="utf-8")
                (run_dir / "metrics.json").write_text("{}", encoding="utf-8")
            return MagicMock(returncode=0)

        test_args = _base_args(tmp_path) + ["--persist-scores"]

        with patch("run_pipeline.subprocess.run", side_effect=fake_subprocess_run):
            with patch("sys.argv", ["run_pipeline.py"] + test_args):
                main()

        assert len(call_log) == 3, f"Expected 3 subprocess calls, got {len(call_log)}"
        assert "extract_snapshot.py" in call_log[0][1]
        assert "train_baseline.py" in call_log[1][1]
        assert "write_scores_to_sql.py" in call_log[2][1]

    def test_three_step_pipeline_predict(self, tmp_path):
        """Predict mode: extract -> predict -> write_scores (3 subprocess calls)."""
        from run_pipeline import main

        call_log = []

        def fake_subprocess_run(cmd, **kwargs):
            call_log.append(cmd)
            if "extract_snapshot.py" in cmd[1]:
                out_idx = cmd.index("--out")
                run_dir = Path(cmd[out_idx + 1])
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "training_examples.csv").write_text("h\nr\n", encoding="utf-8")
                (run_dir / "model.joblib").write_text("fake", encoding="utf-8")
                (run_dir / "scaler.joblib").write_text("fake", encoding="utf-8")
            elif "predict.py" in cmd[1]:
                csv_idx = cmd.index("--training-examples-csv")
                csv_path = Path(cmd[csv_idx + 1])
                run_dir = csv_path.parent
                (run_dir / "predictions.csv").write_text("h\nr\n", encoding="utf-8")
                (run_dir / "metrics.json").write_text("{}", encoding="utf-8")
            return MagicMock(returncode=0)

        test_args = _base_args(tmp_path) + ["--mode", "predict"]

        with patch("run_pipeline.subprocess.run", side_effect=fake_subprocess_run):
            with patch("sys.argv", ["run_pipeline.py"] + test_args):
                main()

        assert len(call_log) == 3
        assert "extract_snapshot.py" in call_log[0][1]
        assert "predict.py" in call_log[1][1]
        assert "write_scores_to_sql.py" in call_log[2][1]

    def test_no_persist_scores_skips_step3(self, tmp_path):
        """--no-persist-scores skips write_scores_to_sql.py call."""
        from run_pipeline import main

        call_log = []

        def fake_subprocess_run(cmd, **kwargs):
            call_log.append(cmd)
            if "extract_snapshot.py" in cmd[1]:
                out_idx = cmd.index("--out")
                run_dir = Path(cmd[out_idx + 1])
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "training_examples.csv").write_text("h\nr\n", encoding="utf-8")
            return MagicMock(returncode=0)

        test_args = _base_args(tmp_path) + ["--no-persist-scores"]

        with patch("run_pipeline.subprocess.run", side_effect=fake_subprocess_run):
            with patch("sys.argv", ["run_pipeline.py"] + test_args):
                main()

        assert len(call_log) == 2
        assert "extract_snapshot.py" in call_log[0][1]
        assert "train_baseline.py" in call_log[1][1]
        # No write_scores_to_sql.py call
        assert all("write_scores_to_sql.py" not in c[1] for c in call_log)

    def test_custom_scores_table_name_passed_through(self, tmp_path):
        """--scores-table-name value appears in write_scores_to_sql.py command."""
        from run_pipeline import main

        call_log = []

        def fake_subprocess_run(cmd, **kwargs):
            call_log.append(cmd)
            if "extract_snapshot.py" in cmd[1]:
                out_idx = cmd.index("--out")
                run_dir = Path(cmd[out_idx + 1])
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "training_examples.csv").write_text("h\nr\n", encoding="utf-8")
            elif "train_baseline.py" in cmd[1]:
                csv_idx = cmd.index("--training-examples-csv")
                run_dir = Path(cmd[csv_idx + 1]).parent
                (run_dir / "predictions.csv").write_text("h\nr\n", encoding="utf-8")
                (run_dir / "metrics.json").write_text("{}", encoding="utf-8")
            return MagicMock(returncode=0)

        test_args = _base_args(tmp_path) + ["--scores-table-name", "dbo.custom_scores"]

        with patch("run_pipeline.subprocess.run", side_effect=fake_subprocess_run):
            with patch("sys.argv", ["run_pipeline.py"] + test_args):
                main()

        write_cmd = call_log[2]
        assert "write_scores_to_sql.py" in write_cmd[1]
        idx = write_cmd.index("--table-name")
        assert write_cmd[idx + 1] == "dbo.custom_scores"

    def test_persist_fails_when_predictions_missing(self, tmp_path):
        """Pipeline exits 1 if predictions.csv missing before Step 3."""
        from run_pipeline import main

        def fake_subprocess_run(cmd, **kwargs):
            if "extract_snapshot.py" in cmd[1]:
                out_idx = cmd.index("--out")
                run_dir = Path(cmd[out_idx + 1])
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "training_examples.csv").write_text("h\nr\n", encoding="utf-8")
            # train_baseline.py succeeds but does NOT produce predictions.csv
            return MagicMock(returncode=0)

        test_args = _base_args(tmp_path)

        with patch("run_pipeline.subprocess.run", side_effect=fake_subprocess_run):
            with patch("sys.argv", ["run_pipeline.py"] + test_args):
                with pytest.raises(SystemExit) as exc_info:
                    main()
                assert exc_info.value.code == 1


# ---------- artifacts-dir ----------

class TestArtifactsDir:
    def test_copies_artifacts_and_calls_predict(self, tmp_path):
        """--artifacts-dir copies model/scaler into run_dir before predict.py."""
        from run_pipeline import main

        # Create an artifacts directory with fake model files
        artifacts = tmp_path / "artifacts"
        artifacts.mkdir()
        (artifacts / "model.joblib").write_text("model_data", encoding="utf-8")
        (artifacts / "scaler.joblib").write_text("scaler_data", encoding="utf-8")

        call_log = []

        def fake_subprocess_run(cmd, **kwargs):
            call_log.append(cmd)
            if "extract_snapshot.py" in cmd[1]:
                out_idx = cmd.index("--out")
                run_dir = Path(cmd[out_idx + 1])
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "training_examples.csv").write_text("h\nr\n", encoding="utf-8")
            elif "predict.py" in cmd[1]:
                csv_idx = cmd.index("--training-examples-csv")
                run_dir = Path(cmd[csv_idx + 1]).parent
                # Verify artifacts were copied into run_dir
                assert (run_dir / "model.joblib").exists()
                assert (run_dir / "scaler.joblib").exists()
                assert (run_dir / "model.joblib").read_text(encoding="utf-8") == "model_data"
                (run_dir / "predictions.csv").write_text("h\nr\n", encoding="utf-8")
                (run_dir / "metrics.json").write_text("{}", encoding="utf-8")
            return MagicMock(returncode=0)

        test_args = _base_args(tmp_path) + [
            "--mode", "predict",
            "--artifacts-dir", str(artifacts),
        ]

        with patch("run_pipeline.subprocess.run", side_effect=fake_subprocess_run):
            with patch("sys.argv", ["run_pipeline.py"] + test_args):
                main()

        assert len(call_log) == 3
        assert "predict.py" in call_log[1][1]
        assert "write_scores_to_sql.py" in call_log[2][1]

    def test_exits_when_artifacts_dir_missing(self, tmp_path):
        """--artifacts-dir pointing to nonexistent path exits 1."""
        from run_pipeline import main

        def fake_subprocess_run(cmd, **kwargs):
            if "extract_snapshot.py" in cmd[1]:
                out_idx = cmd.index("--out")
                run_dir = Path(cmd[out_idx + 1])
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "training_examples.csv").write_text("h\nr\n", encoding="utf-8")
            return MagicMock(returncode=0)

        test_args = _base_args(tmp_path) + [
            "--mode", "predict",
            "--artifacts-dir", str(tmp_path / "nonexistent"),
        ]

        with patch("run_pipeline.subprocess.run", side_effect=fake_subprocess_run):
            with patch("sys.argv", ["run_pipeline.py"] + test_args):
                with pytest.raises(SystemExit) as exc_info:
                    main()
                assert exc_info.value.code == 1

    def test_exits_when_model_missing_in_artifacts_dir(self, tmp_path):
        """--artifacts-dir with missing model.joblib exits 1."""
        from run_pipeline import main

        artifacts = tmp_path / "artifacts"
        artifacts.mkdir()
        # Only scaler, no model
        (artifacts / "scaler.joblib").write_text("fake", encoding="utf-8")

        def fake_subprocess_run(cmd, **kwargs):
            if "extract_snapshot.py" in cmd[1]:
                out_idx = cmd.index("--out")
                run_dir = Path(cmd[out_idx + 1])
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "training_examples.csv").write_text("h\nr\n", encoding="utf-8")
            return MagicMock(returncode=0)

        test_args = _base_args(tmp_path) + [
            "--mode", "predict",
            "--artifacts-dir", str(artifacts),
        ]

        with patch("run_pipeline.subprocess.run", side_effect=fake_subprocess_run):
            with patch("sys.argv", ["run_pipeline.py"] + test_args):
                with pytest.raises(SystemExit) as exc_info:
                    main()
                assert exc_info.value.code == 1


# ---------- registry-dir predict mode ----------

class TestRegistryPredict:
    def test_predict_uses_registry_active_by_default(self, tmp_path):
        """--registry-dir copies model/scaler from active/ into run_dir."""
        from run_pipeline import main

        # Create a fake registry with active/ dir
        registry = tmp_path / "registry"
        active = registry / "active"
        active.mkdir(parents=True)
        (active / "model.joblib").write_text("registry_model", encoding="utf-8")
        (active / "scaler.joblib").write_text("registry_scaler", encoding="utf-8")

        call_log = []

        def fake_subprocess_run(cmd, **kwargs):
            call_log.append(cmd)
            if "extract_snapshot.py" in cmd[1]:
                out_idx = cmd.index("--out")
                run_dir = Path(cmd[out_idx + 1])
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "training_examples.csv").write_text("h\nr\n", encoding="utf-8")
            elif "predict.py" in cmd[1]:
                csv_idx = cmd.index("--training-examples-csv")
                run_dir = Path(cmd[csv_idx + 1]).parent
                # Verify registry artifacts were copied into run_dir
                assert (run_dir / "model.joblib").exists()
                assert (run_dir / "model.joblib").read_text(encoding="utf-8") == "registry_model"
                (run_dir / "predictions.csv").write_text("h\nr\n", encoding="utf-8")
                (run_dir / "metrics.json").write_text("{}", encoding="utf-8")
            return MagicMock(returncode=0)

        test_args = _base_args(tmp_path) + [
            "--mode", "predict",
            "--registry-dir", str(registry),
            "--no-persist-scores",
        ]

        with patch("run_pipeline.subprocess.run", side_effect=fake_subprocess_run):
            with patch("sys.argv", ["run_pipeline.py"] + test_args):
                main()

        assert len(call_log) == 2
        assert "extract_snapshot.py" in call_log[0][1]
        assert "predict.py" in call_log[1][1]

    def test_predict_errors_when_registry_active_missing(self, tmp_path):
        """--registry-dir pointing to dir without active/ exits 1."""
        from run_pipeline import main

        # Registry exists but no active/ subdirectory
        registry = tmp_path / "registry"
        registry.mkdir()

        def fake_subprocess_run(cmd, **kwargs):
            if "extract_snapshot.py" in cmd[1]:
                out_idx = cmd.index("--out")
                run_dir = Path(cmd[out_idx + 1])
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "training_examples.csv").write_text("h\nr\n", encoding="utf-8")
            return MagicMock(returncode=0)

        test_args = _base_args(tmp_path) + [
            "--mode", "predict",
            "--registry-dir", str(registry),
        ]

        with patch("run_pipeline.subprocess.run", side_effect=fake_subprocess_run):
            with patch("sys.argv", ["run_pipeline.py"] + test_args):
                with pytest.raises(SystemExit) as exc_info:
                    main()
                assert exc_info.value.code == 1

    def test_artifacts_dir_takes_precedence_over_registry(self, tmp_path):
        """When both --artifacts-dir and --registry-dir provided, artifacts-dir wins."""
        from run_pipeline import main

        # Create registry with active/
        registry = tmp_path / "registry"
        active = registry / "active"
        active.mkdir(parents=True)
        (active / "model.joblib").write_text("registry_model", encoding="utf-8")
        (active / "scaler.joblib").write_text("registry_scaler", encoding="utf-8")

        # Create artifacts-dir with different content
        artifacts = tmp_path / "artifacts"
        artifacts.mkdir()
        (artifacts / "model.joblib").write_text("artifacts_model", encoding="utf-8")
        (artifacts / "scaler.joblib").write_text("artifacts_scaler", encoding="utf-8")

        call_log = []

        def fake_subprocess_run(cmd, **kwargs):
            call_log.append(cmd)
            if "extract_snapshot.py" in cmd[1]:
                out_idx = cmd.index("--out")
                run_dir = Path(cmd[out_idx + 1])
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "training_examples.csv").write_text("h\nr\n", encoding="utf-8")
            elif "predict.py" in cmd[1]:
                csv_idx = cmd.index("--training-examples-csv")
                run_dir = Path(cmd[csv_idx + 1]).parent
                # Verify artifacts-dir content was used, not registry
                assert (run_dir / "model.joblib").read_text(encoding="utf-8") == "artifacts_model"
                (run_dir / "predictions.csv").write_text("h\nr\n", encoding="utf-8")
                (run_dir / "metrics.json").write_text("{}", encoding="utf-8")
            return MagicMock(returncode=0)

        test_args = _base_args(tmp_path) + [
            "--mode", "predict",
            "--artifacts-dir", str(artifacts),
            "--registry-dir", str(registry),
            "--no-persist-scores",
        ]

        with patch("run_pipeline.subprocess.run", side_effect=fake_subprocess_run):
            with patch("sys.argv", ["run_pipeline.py"] + test_args):
                main()

        assert len(call_log) == 2
        assert "predict.py" in call_log[1][1]


# ---------- empty-snapshot predict mode ----------

class TestEmptySnapshotPredict:
    def test_predict_zero_rows_skips_artifact_copy_and_succeeds(self, tmp_path):
        """Predict mode with 0-row training_examples.csv should skip artifact copy,
        call predict.py, and succeed without model artifacts."""
        from run_pipeline import main

        call_log = []

        def fake_subprocess_run(cmd, **kwargs):
            call_log.append(cmd)
            if "extract_snapshot.py" in cmd[1]:
                out_idx = cmd.index("--out")
                run_dir = Path(cmd[out_idx + 1])
                run_dir.mkdir(parents=True, exist_ok=True)
                # Header-only CSV (0 data rows)
                (run_dir / "training_examples.csv").write_text(
                    "org_id,enrollment_id,ghl_contact_id,decision_ts_utc,score\n",
                    encoding="utf-8",
                )
                # No model.joblib or scaler.joblib — not needed for 0 rows
            elif "predict.py" in cmd[1]:
                # predict.py should handle 0 rows and write empty outputs
                csv_idx = cmd.index("--training-examples-csv")
                run_dir = Path(cmd[csv_idx + 1]).parent
                (run_dir / "predictions.csv").write_text(
                    "org_id,enrollment_id,ghl_contact_id,decision_ts_utc,score\n",
                    encoding="utf-8",
                )
                (run_dir / "metrics.json").write_text('{"n_samples": 0}', encoding="utf-8")
            return MagicMock(returncode=0)

        test_args = _base_args(tmp_path) + [
            "--mode", "predict",
            "--no-persist-scores",
        ]

        with patch("run_pipeline.subprocess.run", side_effect=fake_subprocess_run):
            with patch("sys.argv", ["run_pipeline.py"] + test_args):
                main()

        # extract + predict called; no artifact copy errors
        assert len(call_log) == 2
        assert "extract_snapshot.py" in call_log[0][1]
        assert "predict.py" in call_log[1][1]

    def test_predict_zero_rows_skips_persist(self, tmp_path):
        """Predict mode with 0-row CSV: Step 3 persist is skipped when
        predictions.csv has 0 data rows."""
        from run_pipeline import main

        call_log = []

        def fake_subprocess_run(cmd, **kwargs):
            call_log.append(cmd)
            if "extract_snapshot.py" in cmd[1]:
                out_idx = cmd.index("--out")
                run_dir = Path(cmd[out_idx + 1])
                run_dir.mkdir(parents=True, exist_ok=True)
                # Header-only CSV (0 data rows)
                (run_dir / "training_examples.csv").write_text(
                    "org_id,enrollment_id,ghl_contact_id,decision_ts_utc,score\n",
                    encoding="utf-8",
                )
            elif "predict.py" in cmd[1]:
                csv_idx = cmd.index("--training-examples-csv")
                run_dir = Path(cmd[csv_idx + 1]).parent
                # predict.py writes header-only predictions.csv
                (run_dir / "predictions.csv").write_text(
                    "org_id,enrollment_id,ghl_contact_id,decision_ts_utc,score\n",
                    encoding="utf-8",
                )
                (run_dir / "metrics.json").write_text('{"n_samples": 0}', encoding="utf-8")
            return MagicMock(returncode=0)

        # persist-scores is ON (default)
        test_args = _base_args(tmp_path) + [
            "--mode", "predict",
            "--persist-scores",
        ]

        with patch("run_pipeline.subprocess.run", side_effect=fake_subprocess_run):
            with patch("sys.argv", ["run_pipeline.py"] + test_args):
                main()

        # extract + predict called; write_scores_to_sql.py NOT called
        assert len(call_log) == 2
        assert "extract_snapshot.py" in call_log[0][1]
        assert "predict.py" in call_log[1][1]
        assert all("write_scores_to_sql.py" not in c[1] for c in call_log)

    def test_train_mode_still_calls_train_with_empty_csv(self, tmp_path):
        """Train mode with 0-row CSV still calls train_baseline.py (train handles its own errors)."""
        from run_pipeline import main

        call_log = []

        def fake_subprocess_run(cmd, **kwargs):
            call_log.append(cmd)
            if "extract_snapshot.py" in cmd[1]:
                out_idx = cmd.index("--out")
                run_dir = Path(cmd[out_idx + 1])
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "training_examples.csv").write_text("h\nr\n", encoding="utf-8")
            return MagicMock(returncode=0)

        test_args = _base_args(tmp_path) + ["--no-persist-scores"]

        with patch("run_pipeline.subprocess.run", side_effect=fake_subprocess_run):
            with patch("sys.argv", ["run_pipeline.py"] + test_args):
                main()

        assert len(call_log) == 2
        assert "train_baseline.py" in call_log[1][1]


# ---------- _csv_has_data_rows ----------

class TestCsvHasDataRows:
    def test_returns_true_for_csv_with_data(self, tmp_path):
        from run_pipeline import _csv_has_data_rows
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("col1,col2\na,b\n", encoding="utf-8")
        assert _csv_has_data_rows(csv_path) is True

    def test_returns_false_for_header_only_csv(self, tmp_path):
        from run_pipeline import _csv_has_data_rows
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("col1,col2\n", encoding="utf-8")
        assert _csv_has_data_rows(csv_path) is False

    def test_returns_false_for_empty_file(self, tmp_path):
        from run_pipeline import _csv_has_data_rows
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("", encoding="utf-8")
        assert _csv_has_data_rows(csv_path) is False
