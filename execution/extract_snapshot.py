"""
Data extraction script for ML pipeline dataset snapshots.

SQL Server (Agent Cory DB) serves as the single source of truth.
This script produces timestamped, immutable dataset snapshots for reproducible ML training.

Usage:
    python extract_snapshot.py --since 2024-01-01 --until 2024-02-01 --out ./data --target booked_call_within_7d
"""

import argparse
import csv
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, date, timezone
from pathlib import Path

from config import LOG_LEVEL
from logging_utils import setup_logging

try:
    import pyodbc
except ImportError:
    print("pyodbc not available")
    sys.exit(1)

logger = logging.getLogger(__name__)

# Fixed deterministic seed for reproducible runs
DETERMINISTIC_SEED = 42


def get_git_commit():
    """Get current git commit hash if available."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def build_connection_string():
    """Build SQL Server connection string from environment variables."""
    host = os.getenv('CORY_SQL_HOST')
    port = os.getenv('CORY_SQL_PORT', '1433')
    database = os.getenv('CORY_SQL_DATABASE')
    auth_mode = os.getenv('CORY_SQL_AUTH_MODE')
    driver = os.getenv('CORY_SQL_ODBC_DRIVER', 'ODBC Driver 18 for SQL Server')
    encrypt = os.getenv('CORY_SQL_ENCRYPT', 'yes')
    trust_server_cert = os.getenv('CORY_SQL_TRUST_SERVER_CERT', 'true')

    # Base connection string
    conn_str_parts = [
        f"DRIVER={{{driver}}}",
        f"SERVER={host},{port}",
        f"DATABASE={database}",
    ]

    # Authentication configuration
    if auth_mode == 'integrated':
        conn_str_parts.append("Trusted_Connection=yes")
    elif auth_mode == 'sqlauth':
        user = os.getenv('CORY_SQL_USER')
        password = os.getenv('CORY_SQL_PASSWORD')
        conn_str_parts.extend([
            f"UID={user}",
            f"PWD={password}"
        ])

    # Connection security and timeout settings
    conn_str_parts.append(f"Encrypt={encrypt}")

    # Handle TrustServerCertificate setting
    trust_cert_value = trust_server_cert.lower() in ['true', '1', 'yes']
    trust_cert_str = 'yes' if trust_cert_value else 'no'
    conn_str_parts.append(f"TrustServerCertificate={trust_cert_str}")

    conn_str_parts.append("Connection Timeout=30")

    return ';'.join(conn_str_parts)


def validate_date(date_str):
    """Validate date format YYYY-MM-DD."""
    try:
        datetime.strptime(date_str, '%Y-%m-%d')
        return date_str
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid date format: {date_str}. Use YYYY-MM-DD.")


def validate_target(target):
    """Validate target selection."""
    valid_targets = ['booked_call_within_7d', 'qualified_within_14d', 'conversion_within_30d']
    if target not in valid_targets:
        raise argparse.ArgumentTypeError(f"Invalid target: {target}. Choose from {valid_targets}")
    return target


def validate_sql_file(file_path):
    """Validate SQL query file exists and is readable."""
    path = Path(file_path)
    if not path.exists():
        raise argparse.ArgumentTypeError(f"SQL file does not exist: {file_path}")
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"Path is not a file: {file_path}")
    if not path.suffix.lower() == '.sql':
        raise argparse.ArgumentTypeError(f"File must have .sql extension: {file_path}")
    return str(path.absolute())


def extract_outcomes(args, output_dir):
    """Extract outcomes data from SQL Server."""
    logger.info("Extracting outcomes data...")

    # Read SQL query file
    with open(args.outcomes_query_file, 'r') as f:
        query = f.read().strip()

    if not query:
        raise ValueError(f"SQL file is empty: {args.outcomes_query_file}")

    # Connect to database and execute query
    conn_str = build_connection_string()
    outcomes_file = Path(output_dir) / "outcomes.csv"
    row_count = 0

    try:
        with pyodbc.connect(conn_str) as conn:
            cursor = conn.cursor()

            logger.info("Executing query from: %s", args.outcomes_query_file)
            if '?' in query:
                cursor.execute(query, args.since, args.until)
            else:
                cursor.execute(query)

            # Get column names
            columns = [desc[0] for desc in cursor.description]

            # Write to CSV
            with open(outcomes_file, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)

                # Write headers
                writer.writerow(columns)

                # Write data rows
                while True:
                    rows = cursor.fetchmany(1000)  # Process in batches
                    if not rows:
                        break

                    for row in rows:
                        writer.writerow(row)
                        row_count += 1

    except pyodbc.Error as e:
        logger.error("Database error during extraction: %s", e)
        raise
    except Exception as e:
        logger.error("Error during extraction: %s", e)
        raise

    logger.info("Extracted %d outcomes records to %s", row_count, outcomes_file.name)
    return row_count


def extract_action_log(args, output_dir):
    """Extract outcomes action log data from SQL Server."""
    if not hasattr(args, 'outcomes_action_log_query_file') or not args.outcomes_action_log_query_file:
        return 0

    logger.info("Extracting outcomes action log data...")

    # Read SQL query file
    with open(args.outcomes_action_log_query_file, 'r') as f:
        query = f.read().strip()

    if not query:
        raise ValueError(f"SQL file is empty: {args.outcomes_action_log_query_file}")

    # Connect to database and execute query
    conn_str = build_connection_string()
    action_log_file = Path(output_dir) / "outcomes_action_log.csv"
    row_count = 0

    try:
        with pyodbc.connect(conn_str) as conn:
            cursor = conn.cursor()

            logger.info("Executing action log query from: %s", args.outcomes_action_log_query_file)
            if '?' in query:
                cursor.execute(query, args.since, args.until)
            else:
                cursor.execute(query)

            # Get column names
            columns = [desc[0] for desc in cursor.description]

            # Write to CSV
            with open(action_log_file, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)

                # Write headers
                writer.writerow(columns)

                # Write data rows
                while True:
                    rows = cursor.fetchmany(1000)  # Process in batches
                    if not rows:
                        break

                    for row in rows:
                        writer.writerow(row)
                        row_count += 1

    except pyodbc.Error as e:
        logger.error("Database error during action log extraction: %s", e)
        raise
    except Exception as e:
        logger.error("Error during action log extraction: %s", e)
        raise

    logger.info("Extracted %d action log records to %s", row_count, action_log_file.name)
    return row_count


def extract_engagement_logs(args, output_dir):
    """Extract engagement logs data from SQL Server."""
    if not hasattr(args, 'engagement_logs_query_file') or not args.engagement_logs_query_file:
        return 0

    logger.info("Extracting engagement logs data...")

    # Read SQL query file
    with open(args.engagement_logs_query_file, 'r') as f:
        query = f.read().strip()

    if not query:
        raise ValueError(f"SQL file is empty: {args.engagement_logs_query_file}")

    # Connect to database and execute query
    conn_str = build_connection_string()
    engagement_logs_file = Path(output_dir) / "engagement_logs.csv"
    row_count = 0

    try:
        with pyodbc.connect(conn_str) as conn:
            cursor = conn.cursor()

            logger.info("Executing engagement logs query from: %s", args.engagement_logs_query_file)
            if '?' in query:
                cursor.execute(query, args.since, args.until)
            else:
                cursor.execute(query)

            # Get column names
            columns = [desc[0] for desc in cursor.description]

            # Write to CSV
            with open(engagement_logs_file, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)

                # Write headers
                writer.writerow(columns)

                # Write data rows
                while True:
                    rows = cursor.fetchmany(1000)  # Process in batches
                    if not rows:
                        break

                    for row in rows:
                        writer.writerow(row)
                        row_count += 1

    except pyodbc.Error as e:
        logger.error("Database error during engagement logs extraction: %s", e)
        raise
    except Exception as e:
        logger.error("Error during engagement logs extraction: %s", e)
        raise

    logger.info("Extracted %d engagement logs records to %s", row_count, engagement_logs_file.name)
    return row_count


def extract_training_examples(args, output_dir):
    """Extract training examples data from SQL Server."""
    if not hasattr(args, 'training_examples_query_file') or not args.training_examples_query_file:
        return 0

    logger.info("Extracting training examples data...")

    # Read SQL query file
    with open(args.training_examples_query_file, 'r') as f:
        query = f.read().strip()

    if not query:
        raise ValueError(f"SQL file is empty: {args.training_examples_query_file}")

    # Connect to database and execute query
    conn_str = build_connection_string()
    training_examples_file = Path(output_dir) / "training_examples.csv"
    row_count = 0

    try:
        with pyodbc.connect(conn_str) as conn:
            cursor = conn.cursor()

            logger.info("Executing training examples query from: %s", args.training_examples_query_file)
            if '?' in query:
                cursor.execute(query, args.since, args.until)
            else:
                cursor.execute(query)

            # Get column names
            columns = [desc[0] for desc in cursor.description]

            # Write to CSV
            with open(training_examples_file, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)

                # Write headers
                writer.writerow(columns)

                # Write data rows
                while True:
                    rows = cursor.fetchmany(1000)  # Process in batches
                    if not rows:
                        break

                    for row in rows:
                        writer.writerow(row)
                        row_count += 1

    except pyodbc.Error as e:
        logger.error("Database error during training examples extraction: %s", e)
        raise
    except Exception as e:
        logger.error("Error during training examples extraction: %s", e)
        raise

    logger.info("Extracted %d training examples records to %s", row_count, training_examples_file.name)
    return row_count


def create_run_manifest(args, output_dir, row_count=0, action_log_row_count=0, engagement_logs_row_count=0, training_examples_row_count=0):
    """Create JSON manifest file documenting the extraction run."""
    run_timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    git_commit = get_git_commit()

    manifest = {
        "run_timestamp": run_timestamp,
        "mode": "READ_ONLY_EXTRACT",
        "args": {
            "since": args.since,
            "until": args.until,
            "output_directory": args.out,
            "target": args.target,
            "outcomes_query_file": getattr(args, 'outcomes_query_file', None),
            "outcomes_action_log_query_file": getattr(args, 'outcomes_action_log_query_file', None),
            "engagement_logs_query_file": getattr(args, 'engagement_logs_query_file', None),
            "training_examples_query_file": getattr(args, 'training_examples_query_file', None)
        },
        "git_commit": git_commit,
        "deterministic_seed": DETERMINISTIC_SEED,
        "entities_to_extract": [
            "contacts",
            "campaign_steps",
            "engagement_logs",
            "outcomes"
        ],
        "row_counts": {
            "outcomes": row_count
        },
        "files": {
            "outcomes": "outcomes.csv"
        }
    }

    # Add action log entries only if query file is provided
    if getattr(args, "outcomes_action_log_query_file", None):
        manifest["row_counts"]["outcomes_action_log"] = action_log_row_count
        manifest["files"]["outcomes_action_log"] = "outcomes_action_log.csv"

    # Add engagement logs entries only if query file is provided
    if getattr(args, "engagement_logs_query_file", None):
        manifest["row_counts"]["engagement_logs"] = engagement_logs_row_count
        manifest["files"]["engagement_logs"] = "engagement_logs.csv"

    # Add training examples entries only if query file is provided
    if getattr(args, "training_examples_query_file", None):
        manifest["row_counts"]["training_examples"] = training_examples_row_count
        manifest["files"]["training_examples"] = "training_examples.csv"

    manifest_path = Path(output_dir) / f"run_manifest_{run_timestamp[:19].replace(':', '-')}.json"

    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    logger.info("Run manifest created: %s", manifest_path)
    return manifest_path


def main():
    parser = argparse.ArgumentParser(
        description="Extract dataset snapshot for ML pipeline training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--since',
        required=True,
        type=validate_date,
        help='Start date for data extraction (YYYY-MM-DD)'
    )

    parser.add_argument(
        '--until',
        required=True,
        type=validate_date,
        help='End date for data extraction (YYYY-MM-DD)'
    )

    parser.add_argument(
        '--out',
        required=True,
        help='Output directory for dataset files'
    )

    parser.add_argument(
        '--target',
        required=True,
        type=validate_target,
        help='Target variable for ML training (booked_call_within_7d|qualified_within_14d|conversion_within_30d)'
    )

    parser.add_argument(
        '--outcomes-query-file',
        required=True,
        type=validate_sql_file,
        help='Path to .sql file containing outcomes extraction query'
    )

    parser.add_argument(
        '--outcomes-action-log-query-file',
        required=False,
        type=validate_sql_file,
        help='Path to .sql file containing outcomes action log extraction query'
    )

    parser.add_argument(
        '--engagement-logs-query-file',
        required=False,
        type=validate_sql_file,
        help='Path to .sql file containing engagement logs extraction query'
    )

    parser.add_argument(
        '--training-examples-query-file',
        required=False,
        type=validate_sql_file,
        help='Path to .sql file containing training examples extraction query'
    )

    parser.add_argument(
        '--log-level',
        default=None,
        help='Log level: DEBUG, INFO, WARNING, ERROR (default: from LOG_LEVEL env var)'
    )

    args = parser.parse_args()

    setup_logging(args.log_level or LOG_LEVEL)

    # Validate date range
    since_date = datetime.strptime(args.since, '%Y-%m-%d').date()
    until_date = datetime.strptime(args.until, '%Y-%m-%d').date()

    if since_date >= until_date:
        logger.error("--since date must be before --until date")
        sys.exit(1)

    # Create output directory if it doesn't exist
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", output_dir.absolute())

    # Perform outcomes extraction
    try:
        row_count = extract_outcomes(args, output_dir)
        action_log_row_count = extract_action_log(args, output_dir)
        engagement_logs_row_count = extract_engagement_logs(args, output_dir)
        training_examples_row_count = extract_training_examples(args, output_dir)

        # Create run manifest with results
        manifest_path = create_run_manifest(args, output_dir, row_count, action_log_row_count, engagement_logs_row_count, training_examples_row_count)

        # Print status message
        logger.info("=" * 60)
        if row_count > 0 or action_log_row_count > 0 or engagement_logs_row_count > 0 or training_examples_row_count > 0:
            logger.info("READ-ONLY extraction completed successfully.")
        else:
            logger.info("READ-ONLY extraction completed with no data.")
        logger.info("=" * 60)
        logger.info("Target: %s", args.target)
        logger.info("Date range: %s to %s", args.since, args.until)
        logger.info("Outcomes extracted: %d rows", row_count)
        if hasattr(args, 'outcomes_action_log_query_file') and args.outcomes_action_log_query_file:
            logger.info("Action log extracted: %d rows", action_log_row_count)
        if hasattr(args, 'engagement_logs_query_file') and args.engagement_logs_query_file:
            logger.info("Engagement logs extracted: %d rows", engagement_logs_row_count)
        if hasattr(args, 'training_examples_query_file') and args.training_examples_query_file:
            logger.info("Training examples extracted: %d rows", training_examples_row_count)
        logger.info("Query file: %s", Path(args.outcomes_query_file).name)
        logger.info("Manifest: %s", manifest_path.name)

    except Exception as e:
        logger.error("Extraction failed: %s", e)
        sys.exit(1)


if __name__ == '__main__':
    main()
