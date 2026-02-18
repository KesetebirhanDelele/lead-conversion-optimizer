"""Tests for execution/write_scores_to_sql.py — no real DB connections."""

import csv
import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock, call

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "execution"))
from write_scores_to_sql import (
    load_metrics,
    load_predictions,
    build_upsert_rows,
    upsert_rows,
    MERGE_SQL,
)


# ---------- helpers ----------

def _write_csv(path, headers, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in rows:
            writer.writerow(row)


PRED_HEADERS = [
    "org_id", "enrollment_id", "ghl_contact_id",
    "decision_ts_utc", "score", "y_true", "split",
]


def _sample_predictions(tmp_path, rows=None):
    """Write a sample predictions.csv and return its path."""
    csv_path = tmp_path / "predictions.csv"
    if rows is None:
        rows = [
            ["orgA", "e1", "c1", "2024-01-01 00:00:00", 0.9, 1, "test"],
            ["orgA", "e2", "c2", "2024-01-02 00:00:00", 0.7, 0, "train"],
            ["orgB", "e3", "c3", "2024-01-03 00:00:00", 0.5, 1, "test"],
        ]
    _write_csv(csv_path, PRED_HEADERS, rows)
    return csv_path


def _sample_metrics(tmp_path):
    """Write a sample metrics.json and return its path."""
    metrics_path = tmp_path / "metrics.json"
    payload = {
        "label_col": "label_responded_within_7d",
        "model_path": "/models/model.joblib",
        "scaler_path": "/models/scaler.joblib",
        "run_timestamp_utc": "2024-06-15T12:00:00Z",
    }
    metrics_path.write_text(json.dumps(payload), encoding="utf-8")
    return metrics_path


# ---------- load_metrics ----------

class TestLoadMetrics:
    def test_loads_valid_json(self, tmp_path):
        path = _sample_metrics(tmp_path)
        m = load_metrics(path)
        assert m["label_col"] == "label_responded_within_7d"
        assert m["model_path"] == "/models/model.joblib"

    def test_raises_when_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_metrics(tmp_path / "nonexistent.json")


# ---------- load_predictions ----------

class TestLoadPredictions:
    def test_loads_all_rows(self, tmp_path):
        csv_path = _sample_predictions(tmp_path)
        df = load_predictions(csv_path, split_filter="all")
        assert len(df) == 3

    def test_filters_test_only(self, tmp_path):
        csv_path = _sample_predictions(tmp_path)
        df = load_predictions(csv_path, split_filter="test")
        assert len(df) == 2
        assert set(df["split"]) == {"test"}

    def test_filters_train_only(self, tmp_path):
        csv_path = _sample_predictions(tmp_path)
        df = load_predictions(csv_path, split_filter="train")
        assert len(df) == 1
        assert df.iloc[0]["enrollment_id"] == "e2"

    def test_raises_when_csv_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_predictions(tmp_path / "missing.csv")

    def test_raises_when_required_columns_missing(self, tmp_path):
        csv_path = tmp_path / "bad.csv"
        _write_csv(csv_path, ["org_id", "enrollment_id"], [["a", "b"]])
        with pytest.raises(ValueError, match="Missing required columns"):
            load_predictions(csv_path)

    def test_raises_when_split_filter_but_no_split_column(self, tmp_path):
        csv_path = tmp_path / "nosplit.csv"
        headers = ["org_id", "enrollment_id", "ghl_contact_id", "decision_ts_utc", "score"]
        _write_csv(csv_path, headers, [["a", "b", "c", "2024-01-01", 0.5]])
        with pytest.raises(ValueError, match="split"):
            load_predictions(csv_path, split_filter="test")


# ---------- build_upsert_rows ----------

class TestBuildUpsertRows:
    def test_row_shape_and_values(self, tmp_path):
        import pandas as pd
        csv_path = _sample_predictions(tmp_path)
        df = load_predictions(csv_path, split_filter="all")
        metrics = json.loads(_sample_metrics(tmp_path).read_text(encoding="utf-8"))
        rows = build_upsert_rows(df, metrics, csv_path)

        assert len(rows) == 3
        # Each row is a 10-tuple
        assert len(rows[0]) == 10

        # Check first row content
        org_id, enroll_id, contact_id, dt, label, score, model, scaler, run_ts, src = rows[0]
        assert org_id == "orgA"
        assert enroll_id == "e1"
        assert label == "label_responded_within_7d"
        assert score == pytest.approx(0.9)
        assert model == "/models/model.joblib"
        assert scaler == "/models/scaler.joblib"
        assert run_ts == "2024-06-15T12:00:00Z"
        assert str(csv_path) in src

    def test_decision_ts_formatted(self, tmp_path):
        import pandas as pd
        csv_path = _sample_predictions(tmp_path)
        df = load_predictions(csv_path)
        metrics = json.loads(_sample_metrics(tmp_path).read_text(encoding="utf-8"))
        rows = build_upsert_rows(df, metrics, csv_path)
        # decision_ts_utc should be formatted as YYYY-MM-DD HH:MM:SS
        assert rows[0][3] == "2024-01-01 00:00:00"

    def test_label_col_from_metrics(self, tmp_path):
        csv_path = _sample_predictions(tmp_path)
        df = load_predictions(csv_path)
        metrics = {"label_col": "custom_label_14d"}
        rows = build_upsert_rows(df, metrics, csv_path)
        # label_col is at index 4
        assert all(r[4] == "custom_label_14d" for r in rows)


# ---------- upsert_rows ----------

class TestUpsertRows:
    def test_calls_executemany_with_parameterized_sql(self):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        rows = [
            ("org", "e1", "c1", "2024-01-01 00:00:00", "lbl", 0.9, "m", "s", "ts", "src"),
            ("org", "e2", "c2", "2024-01-02 00:00:00", "lbl", 0.7, "m", "s", "ts", "src"),
        ]

        n = upsert_rows(mock_conn, rows, "dbo.lead_scores", batch_size=500)

        assert n == 2
        mock_cursor.executemany.assert_called_once()
        sql_arg = mock_cursor.executemany.call_args[0][0]
        # SQL must contain parameterized placeholders, not string-interpolated values
        assert "?" in sql_arg
        assert "dbo.lead_scores" in sql_arg
        mock_conn.commit.assert_called_once()
        mock_cursor.close.assert_called_once()

    def test_batches_correctly(self):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        rows = [("o", f"e{i}", "c", "d", "l", 0.5, "m", "s", "t", "p") for i in range(5)]
        upsert_rows(mock_conn, rows, "tbl", batch_size=2)

        # 5 rows with batch_size=2 => 3 executemany calls
        assert mock_cursor.executemany.call_count == 3
        assert mock_conn.commit.call_count == 3


# ---------- dry-run via main ----------

class TestDryRun:
    def test_dry_run_does_not_connect(self, tmp_path):
        csv_path = _sample_predictions(tmp_path)
        metrics_path = _sample_metrics(tmp_path)

        test_args = [
            "write_scores_to_sql.py",
            "--predictions-csv", str(csv_path),
            "--metrics-json", str(metrics_path),
            "--dry-run",
        ]

        with patch("write_scores_to_sql.pyodbc") as mock_pyodbc:
            with patch("sys.argv", test_args):
                from write_scores_to_sql import main
                with pytest.raises(SystemExit) as exc_info:
                    main()
                assert exc_info.value.code == 0

            # pyodbc.connect must NOT have been called
            mock_pyodbc.connect.assert_not_called()
