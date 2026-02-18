"""
Push predictions back into SQL Server (write-back stage).

Reads predictions.csv and upserts into dbo.lead_scores (or another table).
Designed for "SQL table → sync later to CRM" workflow.

Usage:
  python execution/push_scores_to_db.py ^
    --predictions-csv tmp/runs/<run>/predictions.csv ^
    --metrics-json tmp/runs/<run>/metrics.json ^
    --table dbo.lead_scores
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

try:
    import pandas as pd
    import pyodbc
except ImportError as e:
    print(f"❌ Required packages not available: {e}")
    print("Install with: pip install pandas pyodbc")
    sys.exit(1)

# Reuse your existing connection builder if available
try:
    from execution.extract_snapshot import build_connection_string
except Exception:
    build_connection_string = None


MERGE_SQL = """
MERGE {table} AS tgt
USING (SELECT ? AS org_id,
              ? AS enrollment_id,
              ? AS ghl_contact_id,
              ? AS decision_ts_utc,
              ? AS label_col,
              ? AS score,
              ? AS model_path,
              ? AS scaler_path,
              ? AS run_timestamp_utc,
              ? AS source_predictions_csv) AS src
ON (tgt.org_id = src.org_id AND tgt.enrollment_id = src.enrollment_id AND tgt.label_col = src.label_col)
WHEN MATCHED THEN
  UPDATE SET
    tgt.ghl_contact_id = src.ghl_contact_id,
    tgt.decision_ts_utc = src.decision_ts_utc,
    tgt.score = src.score,
    tgt.model_path = src.model_path,
    tgt.scaler_path = src.scaler_path,
    tgt.run_timestamp_utc = src.run_timestamp_utc,
    tgt.source_predictions_csv = src.source_predictions_csv,
    tgt.updated_at_utc = SYSUTCDATETIME()
WHEN NOT MATCHED THEN
  INSERT (org_id, enrollment_id, ghl_contact_id, decision_ts_utc, label_col, score,
          model_path, scaler_path, run_timestamp_utc, source_predictions_csv,
          created_at_utc, updated_at_utc)
  VALUES (src.org_id, src.enrollment_id, src.ghl_contact_id, src.decision_ts_utc, src.label_col, src.score,
          src.model_path, src.scaler_path, src.run_timestamp_utc, src.source_predictions_csv,
          SYSUTCDATETIME(), SYSUTCDATETIME());
"""


def load_metrics(metrics_json: Path) -> dict:
    if not metrics_json.exists():
        return {}
    return json.loads(metrics_json.read_text(encoding="utf-8"))


def connect():
    if build_connection_string is None:
        raise RuntimeError("build_connection_string() not available. Import from execution.extract_snapshot failed.")
    conn_str = build_connection_string()
    return pyodbc.connect(conn_str)


def main():
    p = argparse.ArgumentParser(description="Upsert predictions into SQL Server table for CRM sync later")
    p.add_argument("--predictions-csv", required=True, type=Path)
    p.add_argument("--metrics-json", required=False, type=Path, default=None)
    p.add_argument("--table", required=False, default="dbo.lead_scores")
    p.add_argument("--batch-size", type=int, default=500)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if not args.predictions_csv.exists():
        print(f"❌ predictions.csv not found: {args.predictions_csv}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.predictions_csv)
    required = {"org_id", "enrollment_id", "ghl_contact_id", "decision_ts_utc", "score", "split"}
    missing = required - set(df.columns)
    if missing:
        print(f"❌ predictions.csv missing columns: {sorted(missing)}", file=sys.stderr)
        sys.exit(1)

    metrics = load_metrics(args.metrics_json) if args.metrics_json else {}
    label_col = metrics.get("label_col") or "label_responded_within_7d"
    model_path = metrics.get("model_path")
    scaler_path = metrics.get("scaler_path")
    run_ts = metrics.get("run_timestamp_utc") or datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    # push only test+train? Usually push ALL scored rows. Keep it simple: push all.
    df["label_col"] = label_col
    df["run_timestamp_utc"] = run_ts
    df["model_path"] = model_path
    df["scaler_path"] = scaler_path
    df["source_predictions_csv"] = str(args.predictions_csv)

    # normalize timestamps for pyodbc
    df["decision_ts_utc"] = pd.to_datetime(df["decision_ts_utc"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")

    rows = df[["org_id", "enrollment_id", "ghl_contact_id", "decision_ts_utc",
               "label_col", "score", "model_path", "scaler_path", "run_timestamp_utc",
               "source_predictions_csv"]].to_records(index=False)

    if args.dry_run:
        print(f"✅ DRY RUN: would upsert {len(df)} rows into {args.table}")
        sys.exit(0)

    conn = connect()
    cur = conn.cursor()

    sql = MERGE_SQL.format(table=args.table)

    total = len(rows)
    print(f"Upserting {total} rows into {args.table} ...")

    try:
        # executemany batches for speed
        for start in range(0, total, args.batch_size):
            batch = rows[start:start + args.batch_size]
            cur.executemany(sql, list(batch))
            conn.commit()
        print(f"✅ Upsert complete: {total} rows")
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    main()
