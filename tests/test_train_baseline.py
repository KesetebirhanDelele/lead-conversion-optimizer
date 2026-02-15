"""Tests for execution/train_baseline.py."""

import csv
import subprocess
import sys
from pathlib import Path

import pytest

# Import the module under test
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "execution"))
from train_baseline import validate_csv_file


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
