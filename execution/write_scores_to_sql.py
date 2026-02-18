"""
Write prediction scores into SQL Server table dbo.lead_scores.

Reads predictions.csv and metrics.json from a run folder,
then upserts rows into the target table via parameterized MERGE.

Usage (PowerShell):
    python execution/write_scores_to_sql.py `
        --predictions-csv tmp/runs/<run>/predictions.csv `
        --metrics-json tmp/runs/<run>/metrics.json

    python execution/write_scores_to_sql.py `
        --predictions-csv tmp/runs/<run>/predictions.csv `
        --metrics-json tmp/runs/<run>/metrics.json `
        --split test --dry-run
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    import pandas as pd
    import pyodbc
except ImportError as e:
    print(f"❌ Required packages not available: {e}")
    print("Install with: pip install pandas pyodbc")
    sys.exit(1)


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


def build_connection_string():
    """Build SQL Server connection string from environment variables."""
    host = os.getenv('CORY_SQL_HOST')
    port = os.getenv('CORY_SQL_PORT', '1433')
    database = os.getenv('CORY_SQL_DATABASE')
    auth_mode = os.getenv('CORY_SQL_AUTH_MODE')
    driver = os.getenv('CORY_SQL_ODBC_DRIVER', 'ODBC Driver 18 for SQL Server')
    encrypt = os.getenv('CORY_SQL_ENCRYPT', 'yes')
    trust_server_cert = os.getenv('CORY_SQL_TRUST_SERVER_CERT', 'true')

    conn_str_parts = [
        f"DRIVER={{{driver}}}",
        f"SERVER={host},{port}",
        f"DATABASE={database}",
    ]

    if auth_mode == 'integrated':
        conn_str_parts.append("Trusted_Connection=yes")
    elif auth_mode == 'sqlauth':
        user = os.getenv('CORY_SQL_USER')
        password = os.getenv('CORY_SQL_PASSWORD')
        conn_str_parts.extend([f"UID={user}", f"PWD={password}"])

    conn_str_parts.append(f"Encrypt={encrypt}")
    trust_cert_value = trust_server_cert.lower() in ['true', '1', 'yes']
    conn_str_parts.append(f"TrustServerCertificate={'yes' if trust_cert_value else 'no'}")
    conn_str_parts.append("Connection Timeout=30")

    return ';'.join(conn_str_parts)


def load_metrics(metrics_json_path):
    """Load metrics.json and return the parsed dict."""
    if not metrics_json_path.exists():
        raise FileNotFoundError(f"metrics.json not found: {metrics_json_path}")
    return json.loads(metrics_json_path.read_text(encoding="utf-8"))


def load_predictions(csv_path, split_filter="all"):
    """Load predictions.csv into a DataFrame, optionally filtering by split."""
    if not csv_path.exists():
        raise FileNotFoundError(f"predictions.csv not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required = {"org_id", "enrollment_id", "ghl_contact_id", "decision_ts_utc", "score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if split_filter != "all":
        if "split" not in df.columns:
            raise ValueError(f"--split {split_filter} requested but 'split' column not in CSV")
        df = df[df["split"] == split_filter].copy()

    return df


def build_upsert_rows(df, metrics, predictions_csv_path):
    """Build list of tuples ready for parameterized MERGE execution."""
    label_col = metrics.get("label_col", "label_responded_within_7d")
    model_path = metrics.get("model_path")
    scaler_path = metrics.get("scaler_path")
    run_ts = metrics.get("run_timestamp_utc",
                         datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))

    # Normalize decision_ts_utc to datetime2(0)-compatible string
    df = df.copy()
    df["decision_ts_utc"] = pd.to_datetime(
        df["decision_ts_utc"], errors="coerce"
    ).dt.strftime("%Y-%m-%d %H:%M:%S")

    rows = []
    for _, r in df.iterrows():
        rows.append((
            str(r["org_id"]),
            str(r["enrollment_id"]),
            str(r["ghl_contact_id"]),
            r["decision_ts_utc"],
            label_col,
            float(r["score"]),
            model_path,
            scaler_path,
            run_ts,
            str(predictions_csv_path),
        ))
    return rows


def upsert_rows(conn, rows, table_name, batch_size=500):
    """Execute parameterized MERGE in batches. Returns total rows processed."""
    sql = MERGE_SQL.format(table=table_name)
    cur = conn.cursor()
    try:
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            cur.executemany(sql, batch)
            conn.commit()
    finally:
        cur.close()
    return len(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Upsert prediction scores into SQL Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--predictions-csv", required=True, type=Path,
                        help="Path to predictions.csv")
    parser.add_argument("--metrics-json", required=True, type=Path,
                        help="Path to metrics.json")
    parser.add_argument("--table-name", default="dbo.lead_scores",
                        help="Target SQL table (default: dbo.lead_scores)")
    parser.add_argument("--split", choices=["all", "test", "train"], default="all",
                        help="Which split to write (default: all)")
    parser.add_argument("--batch-size", type=int, default=500,
                        help="Rows per batch (default: 500)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print row count without writing to DB")

    args = parser.parse_args()

    # Load inputs
    try:
        metrics = load_metrics(args.metrics_json)
        df = load_predictions(args.predictions_csv, split_filter=args.split)
    except (FileNotFoundError, ValueError) as e:
        print(f"❌ {e}", file=sys.stderr)
        sys.exit(1)

    rows = build_upsert_rows(df, metrics, args.predictions_csv)
    print(f"Prepared {len(rows)} rows (split={args.split}) for {args.table_name}")

    if args.dry_run:
        print(f"✅ DRY RUN: would upsert {len(rows)} rows into {args.table_name}")
        sys.exit(0)

    # Connect and upsert
    try:
        conn_str = build_connection_string()
        conn = pyodbc.connect(conn_str)
    except Exception as e:
        print(f"❌ Database connection failed: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        n = upsert_rows(conn, rows, args.table_name, batch_size=args.batch_size)
        print(f"✅ Upserted {n} rows into {args.table_name}")
    except Exception as e:
        print(f"❌ Upsert failed: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
