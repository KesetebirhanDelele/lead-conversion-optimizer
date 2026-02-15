"""Tests for execution/train_baseline.py."""

import csv
import subprocess
import sys
from pathlib import Path

import pytest

# Import the module under test
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "execution"))
from train_baseline import validate_csv_file, compute_precision_at_k


BASE_HEADERS = [
    "org_id", "enrollment_id", "ghl_contact_id", "decision_ts_utc",
    "attempts_sms_24h", "attempts_email_24h",
    "attempts_voice_no_voicemail_24h", "voicemail_drops_24h",
]


def _write_csv(path, headers, rows):
    """Helper: write a CSV with given headers and rows."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in rows:
            writer.writerow(row)


# ---------- validate_csv_file ----------

class TestValidateCsvFile:
    def test_passes_with_default_label_col(self, tmp_path):
        csv_path = tmp_path / "ok.csv"
        headers = BASE_HEADERS + ["label_responded_within_7d"]
        _write_csv(csv_path, headers, [["org1", "e1", "c1", "2024-01-01", 1, 0, 0, 0, 1]])
        assert validate_csv_file(csv_path) is True

    def test_passes_with_custom_label_col(self, tmp_path):
        csv_path = tmp_path / "ok.csv"
        headers = BASE_HEADERS + ["label_responded_within_14d"]
        _write_csv(csv_path, headers, [["org1", "e1", "c1", "2024-01-01", 1, 0, 0, 0, 1]])
        assert validate_csv_file(csv_path, label_col="label_responded_within_14d") is True

    def test_fails_if_label_col_missing(self, tmp_path):
        csv_path = tmp_path / "missing_label.csv"
        # CSV has base headers but NOT the requested label column
        _write_csv(csv_path, BASE_HEADERS, [["org1", "e1", "c1", "2024-01-01", 1, 0, 0, 0]])
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_csv_file(csv_path, label_col="label_responded_within_14d")

    def test_fails_if_default_label_col_missing(self, tmp_path):
        csv_path = tmp_path / "missing_default.csv"
        _write_csv(csv_path, BASE_HEADERS, [["org1", "e1", "c1", "2024-01-01", 1, 0, 0, 0]])
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_csv_file(csv_path)


# ---------- single-class early exit ----------

class TestSingleClassEarlyExit:
    def test_exits_cleanly_when_all_zeros(self, tmp_path):
        """When every label is 0, script should exit 0 with warning and no predictions.csv."""
        csv_path = tmp_path / "all_zeros.csv"
        headers = BASE_HEADERS + ["label_responded_within_7d"]
        # 10 rows, all label=0, sequential dates so time split works
        rows = [
            [f"org{i}", f"e{i}", f"c{i}", f"2024-01-{i+1:02d}", i % 3, 0, 0, 0, 0]
            for i in range(10)
        ]
        _write_csv(csv_path, headers, rows)

        script = str(Path(__file__).resolve().parent.parent / "execution" / "train_baseline.py")
        result = subprocess.run(
            [sys.executable, script, "--training-examples-csv", str(csv_path)],
            capture_output=True, timeout=30,
            env={**__import__("os").environ, "PYTHONUTF8": "1"},
            encoding="utf-8",
        )

        assert result.returncode == 0, f"Expected exit 0, got {result.returncode}.\nstderr: {result.stderr}"
        assert "Cannot train" in result.stdout
        assert not (tmp_path / "predictions.csv").exists()

    def test_exits_cleanly_when_all_ones(self, tmp_path):
        """When every label is 1, script should exit 0 with warning and no predictions.csv."""
        csv_path = tmp_path / "all_ones.csv"
        headers = BASE_HEADERS + ["label_responded_within_7d"]
        rows = [
            [f"org{i}", f"e{i}", f"c{i}", f"2024-01-{i+1:02d}", i % 3, 0, 0, 0, 1]
            for i in range(10)
        ]
        _write_csv(csv_path, headers, rows)

        script = str(Path(__file__).resolve().parent.parent / "execution" / "train_baseline.py")
        result = subprocess.run(
            [sys.executable, script, "--training-examples-csv", str(csv_path)],
            capture_output=True, timeout=30,
            env={**__import__("os").environ, "PYTHONUTF8": "1"},
            encoding="utf-8",
        )

        assert result.returncode == 0, f"Expected exit 0, got {result.returncode}.\nstderr: {result.stderr}"
        assert "Cannot train" in result.stdout
        assert not (tmp_path / "predictions.csv").exists()


# ---------- compute_precision_at_k ----------

class TestComputePrecisionAtK:
    def _make_pred_df(self, labels):
        """Build a minimal pred_df with test split sorted by score desc."""
        import pandas as pd
        rows = []
        for i, label in enumerate(labels):
            rows.append({
                "score": 1.0 - i * 0.01,
                "y_true": label,
                "split": "test",
            })
        # add a train row so the function filters correctly
        rows.append({"score": 0.5, "y_true": 1, "split": "train"})
        return pd.DataFrame(rows)

    def test_skips_k_greater_than_n_test(self):
        # 5 test rows — k=10,20,50 should all be skipped
        df = self._make_pred_df([1, 0, 1, 0, 1])
        result = compute_precision_at_k(df, ks=(10, 20, 50))
        assert result == []

    def test_precision_values(self):
        # 15 test rows: first 10 are [1,1,1,0,0,0,0,0,0,0], rest 0
        labels = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        df = self._make_pred_df(labels)
        result = compute_precision_at_k(df, ks=(10,))
        assert len(result) == 1
        assert result[0]["k"] == 10
        assert result[0]["n"] == 15
        assert result[0]["n_positive"] == 3
        assert result[0]["precision"] == pytest.approx(0.3)


# ---------- metrics.json integration ----------

def _make_two_class_csv(csv_path, n=30):
    """Create a CSV with both classes so training succeeds."""
    headers = BASE_HEADERS + ["label_responded_within_7d"]
    rows = []
    for i in range(n):
        label = 1 if i % 3 == 0 else 0
        rows.append([
            f"org{i}", f"e{i}", f"c{i}", f"2024-01-{(i % 28) + 1:02d}",
            i % 4, i % 2, 0, 0, label,
        ])
    _write_csv(csv_path, headers, rows)


class TestMetricsJson:
    def _run_training(self, tmp_path):
        csv_path = tmp_path / "training_examples.csv"
        _make_two_class_csv(csv_path)
        script = str(Path(__file__).resolve().parent.parent / "execution" / "train_baseline.py")
        result = subprocess.run(
            [sys.executable, script, "--training-examples-csv", str(csv_path)],
            capture_output=True, timeout=60,
            env={**__import__("os").environ, "PYTHONUTF8": "1"},
            encoding="utf-8",
        )
        return result, tmp_path

    def test_metrics_json_created(self, tmp_path):
        result, out_dir = self._run_training(tmp_path)
        assert result.returncode == 0, f"Training failed.\nstderr: {result.stderr}"
        metrics_path = out_dir / "metrics.json"
        assert metrics_path.exists(), "metrics.json was not created"

    def test_metrics_json_required_keys(self, tmp_path):
        result, out_dir = self._run_training(tmp_path)
        assert result.returncode == 0, f"stderr: {result.stderr}"
        import json
        metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
        required_top = {
            "run_timestamp_utc", "input_training_examples_csv",
            "predictions_csv", "label_col", "feature_cols",
            "train", "test", "precision_at_k",
        }
        assert required_top.issubset(metrics.keys()), f"Missing keys: {required_top - metrics.keys()}"
        for split in ("train", "test"):
            assert "n_samples" in metrics[split]
            assert "n_positive" in metrics[split]
            assert "positive_rate" in metrics[split]

    def test_precision_at_k_respects_n_test(self, tmp_path):
        result, out_dir = self._run_training(tmp_path)
        assert result.returncode == 0, f"stderr: {result.stderr}"
        import json
        metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
        n_test = metrics["test"]["n_samples"]
        for entry in metrics["precision_at_k"]:
            assert entry["k"] <= n_test, f"k={entry['k']} exceeds n_test={n_test}"
