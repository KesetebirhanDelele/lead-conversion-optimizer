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
import os
import subprocess
import sys
from datetime import datetime, date, timezone
from pathlib import Path

try:
    import pyodbc
except ImportError:
    print("❌ pyodbc not available")
    sys.exit(1)

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
    print("Extracting outcomes data...")
    
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
            
            print(f"Executing query from: {args.outcomes_query_file}")
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
        print(f"❌ Database error during extraction: {e}")
        raise
    except Exception as e:
        print(f"❌ Error during extraction: {e}")
        raise
    
    print(f"✅ Extracted {row_count} outcomes records to {outcomes_file.name}")
    return row_count


def extract_action_log(args, output_dir):
    """Extract outcomes action log data from SQL Server."""
    if not hasattr(args, 'outcomes_action_log_query_file') or not args.outcomes_action_log_query_file:
        return 0
    
    print("Extracting outcomes action log data...")
    
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
            
            print(f"Executing action log query from: {args.outcomes_action_log_query_file}")
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
        print(f"❌ Database error during action log extraction: {e}")
        raise
    except Exception as e:
        print(f"❌ Error during action log extraction: {e}")
        raise
    
    print(f"✅ Extracted {row_count} action log records to {action_log_file.name}")
    return row_count


def extract_engagement_logs(args, output_dir):
    """Extract engagement logs data from SQL Server."""
    if not hasattr(args, 'engagement_logs_query_file') or not args.engagement_logs_query_file:
        return 0
    
    print("Extracting engagement logs data...")
    
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
            
            print(f"Executing engagement logs query from: {args.engagement_logs_query_file}")
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
        print(f"❌ Database error during engagement logs extraction: {e}")
        raise
    except Exception as e:
        print(f"❌ Error during engagement logs extraction: {e}")
        raise
    
    print(f"✅ Extracted {row_count} engagement logs records to {engagement_logs_file.name}")
    return row_count


def create_run_manifest(args, output_dir, row_count=0, action_log_row_count=0, engagement_logs_row_count=0):
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
            "engagement_logs_query_file": getattr(args, 'engagement_logs_query_file', None)
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
    
    manifest_path = Path(output_dir) / f"run_manifest_{run_timestamp[:19].replace(':', '-')}.json"
    
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"Run manifest created: {manifest_path}")
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
    
    args = parser.parse_args()
    
    # Validate date range
    since_date = datetime.strptime(args.since, '%Y-%m-%d').date()
    until_date = datetime.strptime(args.until, '%Y-%m-%d').date()
    
    if since_date >= until_date:
        print("Error: --since date must be before --until date", file=sys.stderr)
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir.absolute()}")
    
    # Perform outcomes extraction
    try:
        row_count = extract_outcomes(args, output_dir)
        action_log_row_count = extract_action_log(args, output_dir)
        engagement_logs_row_count = extract_engagement_logs(args, output_dir)
        
        # Create run manifest with results
        manifest_path = create_run_manifest(args, output_dir, row_count, action_log_row_count, engagement_logs_row_count)
        
        # Print status message
        print("\n" + "="*60)
        if row_count > 0 or action_log_row_count > 0 or engagement_logs_row_count > 0:
            print("✅ READ-ONLY extraction completed successfully.")
        else:
            print("ℹ️  READ-ONLY extraction completed with no data.")
        print("="*60)
        print(f"Target: {args.target}")
        print(f"Date range: {args.since} to {args.until}")
        print(f"Outcomes extracted: {row_count} rows")
        if hasattr(args, 'outcomes_action_log_query_file') and args.outcomes_action_log_query_file:
            print(f"Action log extracted: {action_log_row_count} rows")
        if hasattr(args, 'engagement_logs_query_file') and args.engagement_logs_query_file:
            print(f"Engagement logs extracted: {engagement_logs_row_count} rows")
        print(f"Query file: {Path(args.outcomes_query_file).name}")
        print(f"Manifest: {manifest_path.name}")
        
    except Exception as e:
        print(f"\n❌ Extraction failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()