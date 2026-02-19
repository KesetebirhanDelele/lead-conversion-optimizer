"""Tests for execution/predict.py."""

import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "execution"))
from predict import (
    FEATURE_COLS,
    ID_COLS,
    _write_empty_outputs,
    compute_score_quantiles,
    compute_precision_at_k,
    validate_inputs,
)


# ---------- helpers ----------

HEADER_COLS = ID_COLS + FEATURE_COLS + ["label_responded_within_7d"]


def _make_csv(path, rows=None):
    """Write a CSV with the standard header and optional data rows."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER_COLS)
        if rows:
            for row in rows:
                writer.writerow(row)


def _make_empty_csv(path):
    """Write a header-only CSV (0 data rows)."""
    _make_csv(path, rows=None)


# ---------- validate_inputs ----------

class TestValidateInputs:
    def test_passes_with_all_files(self, tmp_path):
        csv_path = tmp_path / "training_examples.csv"
        _make_csv(csv_path)
        (tmp_path / "model.joblib").write_text("m", encoding="utf-8")
        (tmp_path / "scaler.joblib").write_text("s", encoding="utf-8")

        assert validate_inputs(csv_path, tmp_path / "model.joblib", tmp_path / "scaler.joblib") is True

    def test_raises_when_csv_missing(self, tmp_path):
        (tmp_path / "model.joblib").write_text("m", encoding="utf-8")
        (tmp_path / "scaler.joblib").write_text("s", encoding="utf-8")
        with pytest.raises(FileNotFoundError):
            validate_inputs(tmp_path / "missing.csv", tmp_path / "model.joblib", tmp_path / "scaler.joblib")

    def test_raises_when_model_missing(self, tmp_path):
        csv_path = tmp_path / "training_examples.csv"
        _make_csv(csv_path)
        (tmp_path / "scaler.joblib").write_text("s", encoding="utf-8")
        with pytest.raises(FileNotFoundError):
            validate_inputs(csv_path, tmp_path / "model.joblib", tmp_path / "scaler.joblib")

    def test_raises_when_columns_missing(self, tmp_path):
        csv_path = tmp_path / "training_examples.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["wrong_col"])
        (tmp_path / "model.joblib").write_text("m", encoding="utf-8")
        (tmp_path / "scaler.joblib").write_text("s", encoding="utf-8")
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_inputs(csv_path, tmp_path / "model.joblib", tmp_path / "scaler.joblib")


# ---------- _write_empty_outputs ----------

class TestWriteEmptyOutputs:
    def _make_args(self, tmp_path, label_col="label_responded_within_7d"):
        from types import SimpleNamespace
        return SimpleNamespace(
            training_examples_csv=tmp_path / "training_examples.csv",
            label_col=label_col,
        )

    def test_writes_header_only_predictions(self, tmp_path):
        args = self._make_args(tmp_path)
        predictions_out = tmp_path / "predictions.csv"
        metrics_out = tmp_path / "metrics.json"
        model_path = tmp_path / "model.joblib"
        scaler_path = tmp_path / "scaler.joblib"
        csv_headers = set(HEADER_COLS)

        _write_empty_outputs(args, csv_headers, predictions_out, metrics_out, model_path, scaler_path)

        # predictions.csv should have header only
        with open(predictions_out, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            assert "score" in header
            assert "y_true" in header  # label_col is in csv_headers
            data = list(reader)
            assert len(data) == 0

    def test_writes_metrics_with_zero_samples(self, tmp_path):
        args = self._make_args(tmp_path)
        predictions_out = tmp_path / "predictions.csv"
        metrics_out = tmp_path / "metrics.json"
        model_path = tmp_path / "model.joblib"
        scaler_path = tmp_path / "scaler.joblib"
        csv_headers = set(HEADER_COLS)

        _write_empty_outputs(args, csv_headers, predictions_out, metrics_out, model_path, scaler_path)

        with open(metrics_out, "r", encoding="utf-8") as f:
            metrics = json.load(f)

        assert metrics["n_samples"] == 0
        assert metrics["n_positive"] == 0
        assert metrics["positive_rate"] == 0.0
        assert metrics["precision_at_k"] == []
        assert metrics["score_quantiles"] is None

    def test_omits_y_true_when_label_not_in_headers(self, tmp_path):
        args = self._make_args(tmp_path, label_col="nonexistent_label")
        predictions_out = tmp_path / "predictions.csv"
        metrics_out = tmp_path / "metrics.json"
        csv_headers = set(HEADER_COLS)  # doesn't contain "nonexistent_label"

        _write_empty_outputs(args, csv_headers, predictions_out, metrics_out,
                             tmp_path / "model.joblib", tmp_path / "scaler.joblib")

        with open(predictions_out, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            assert "y_true" not in header


# ---------- predict.py main() empty-data guard (subprocess) ----------

class TestPredictMainEmptyGuard:
    def test_zero_row_csv_exits_zero_and_writes_outputs(self, tmp_path):
        """predict.py with a 0-row CSV should exit 0 and write empty outputs,
        even without model/scaler artifacts."""
        import subprocess

        csv_path = tmp_path / "training_examples.csv"
        _make_empty_csv(csv_path)
        # Intentionally do NOT create model.joblib or scaler.joblib

        script = str(Path(__file__).resolve().parent.parent / "execution" / "predict.py")
        result = subprocess.run(
            [sys.executable, script, "--training-examples-csv", str(csv_path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            env={**__import__("os").environ, "PYTHONUTF8": "1"},
        )

        assert result.returncode == 0, f"stderr: {result.stderr}"

        predictions_out = tmp_path / "predictions.csv"
        metrics_out = tmp_path / "metrics.json"

        assert predictions_out.exists()
        assert metrics_out.exists()

        with open(metrics_out, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        assert metrics["n_samples"] == 0

    def test_zero_row_csv_does_not_require_model(self, tmp_path):
        """0-row CSV should succeed without model.joblib existing."""
        import subprocess

        csv_path = tmp_path / "training_examples.csv"
        _make_empty_csv(csv_path)
        # No model.joblib, no scaler.joblib

        script = str(Path(__file__).resolve().parent.parent / "execution" / "predict.py")
        result = subprocess.run(
            [sys.executable, script, "--training-examples-csv", str(csv_path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            env={**__import__("os").environ, "PYTHONUTF8": "1"},
        )

        assert result.returncode == 0


# ---------- compute helpers ----------

class TestComputeScoreQuantiles:
    def test_returns_expected_keys(self):
        import numpy as np
        scores = np.array([0.1, 0.5, 0.9])
        result = compute_score_quantiles(scores)
        assert set(result.keys()) == {"p0", "p10", "p50", "p90", "p100"}

    def test_values_are_floats(self):
        import numpy as np
        scores = np.array([0.2, 0.8])
        result = compute_score_quantiles(scores)
        assert all(isinstance(v, float) for v in result.values())


class TestComputePrecisionAtK:
    def test_basic_precision(self):
        import numpy as np
        y_true = np.array([1, 0, 1, 0, 0, 1, 0, 0, 0, 0])
        results = compute_precision_at_k(y_true, y_true, ks=(10,))
        assert len(results) == 1
        assert results[0]["k"] == 10
        assert results[0]["n_positive"] == 3
        assert results[0]["precision"] == 0.3

    def test_skips_k_larger_than_n(self):
        import numpy as np
        y_true = np.array([1, 0, 1])
        results = compute_precision_at_k(y_true, y_true, ks=(10, 20, 50))
        assert len(results) == 0
