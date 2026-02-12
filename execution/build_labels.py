"""
Label building script for ML pipeline.

Builds labels from existing snapshot CSVs without requiring database connections.
Produces deterministic labels based on timing between first engagement and booking outcomes.

Usage:
    python build_labels.py --snapshot-dir ./tmp/snapshots/run1 --out ./tmp/labels --label-window-days 7
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


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


def parse_iso_timestamp(timestamp_str):
    """Parse ISO timestamp string with Z suffix to datetime."""
    if not timestamp_str or timestamp_str.strip() == '':
        return None
    
    try:
        # Handle format: YYYY-MM-DDTHH:MM:SSZ
        if timestamp_str.endswith('Z'):
            timestamp_str = timestamp_str[:-1] + '+00:00'
        return datetime.fromisoformat(timestamp_str).replace(tzinfo=timezone.utc)
    except ValueError as e:
        raise ValueError(f"Invalid timestamp format: {timestamp_str}") from e


def format_iso_timestamp(dt):
    """Format datetime to ISO string with Z suffix."""
    if dt is None:
        return ""
    return dt.replace(microsecond=0).isoformat().replace('+00:00', 'Z')


def validate_csv_file(file_path, required_columns):
    """Validate CSV file exists and has required columns."""
    if not file_path.exists():
        raise FileNotFoundError(f"Required CSV file not found: {file_path}")
    
    if not file_path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")
    
    # Check if file has required columns
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        headers = set(reader.fieldnames or [])
        missing_cols = set(required_columns) - headers
        if missing_cols:
            raise ValueError(f"Missing required columns in {file_path.name}: {missing_cols}")
    
    return True


def load_engagement_data(engagement_file):
    """Load engagement data and compute first engagement per contact."""
    engagement_data = {}
    row_count = 0
    
    with open(engagement_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            row_count += 1
            contact_id = row['ghl_contact_id']
            if not contact_id:
                continue
                
            try:
                event_ts = parse_iso_timestamp(row['event_ts'])
                if event_ts is None:
                    continue
                    
                # Track first engagement timestamp per contact
                if contact_id not in engagement_data or event_ts < engagement_data[contact_id]:
                    engagement_data[contact_id] = event_ts
                    
            except ValueError:
                # Skip rows with invalid timestamps
                continue
    
    return engagement_data, row_count


def load_outcome_data(outcomes_file):
    """Load outcome data and compute first booking per contact."""
    outcome_data = {}
    row_count = 0
    
    with open(outcomes_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            row_count += 1
            contact_id = row['ghl_contact_id']
            if not contact_id:
                continue
                
            try:
                created_at = parse_iso_timestamp(row['created_at'])
                if created_at is None:
                    continue
                    
                # Track first booking timestamp per contact
                if contact_id not in outcome_data or created_at < outcome_data[contact_id]:
                    outcome_data[contact_id] = created_at
                    
            except ValueError:
                # Skip rows with invalid timestamps
                continue
    
    return outcome_data, row_count


def build_labels(engagement_data, outcome_data, label_window_days):
    """Build labels by joining engagement and outcome data."""
    labels = []
    
    # Left join: all contacts from engagement data
    for contact_id, first_engagement_ts in engagement_data.items():
        booking_created_at = outcome_data.get(contact_id)
        
        # Compute booked_call_within_window
        booked_call_within_window = 0
        if booking_created_at is not None:
            window_end = first_engagement_ts + timedelta(days=label_window_days)
            if (booking_created_at >= first_engagement_ts and 
                booking_created_at < window_end):
                booked_call_within_window = 1
        
        labels.append({
            'ghl_contact_id': contact_id,
            'first_engagement_ts': format_iso_timestamp(first_engagement_ts),
            'booking_created_at': format_iso_timestamp(booking_created_at),
            'booked_call_within_7d': booked_call_within_window
        })
    
    # Sort by ghl_contact_id for deterministic output
    labels.sort(key=lambda x: x['ghl_contact_id'])
    
    return labels


def create_labels_manifest(args, engagement_row_count, outcome_row_count, label_row_count):
    """Create JSON manifest file documenting the label building run."""
    run_timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    git_commit = get_git_commit()
    
    manifest = {
        "run_timestamp": run_timestamp,
        "mode": "LABEL_BUILD",
        "args": {
            "snapshot_dir": str(args.snapshot_dir),
            "output_directory": str(args.out),
            "label_window_days": args.label_window_days
        },
        "git_commit": git_commit,
        "inputs": {
            "engagement_logs": "engagement_logs.csv",
            "outcomes": "outcomes.csv"
        },
        "outputs": {
            "labels": "labels.csv"
        },
        "row_counts": {
            "engagement_rows": engagement_row_count,
            "outcome_rows": outcome_row_count,
            "label_rows": label_row_count
        },
        "label_window_days": args.label_window_days
    }
    
    manifest_path = Path(args.out) / f"labels_manifest_{run_timestamp[:19].replace(':', '-')}.json"
    
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"Labels manifest created: {manifest_path.name}")
    return manifest_path


def main():
    parser = argparse.ArgumentParser(
        description="Build labels from engagement logs and outcomes data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--snapshot-dir',
        required=True,
        type=Path,
        help='Directory containing engagement_logs.csv and outcomes.csv'
    )
    
    parser.add_argument(
        '--out',
        required=True,
        type=Path,
        help='Output directory for labels.csv and manifest'
    )
    
    parser.add_argument(
        '--label-window-days',
        type=int,
        default=7,
        help='Number of days for booking window after first engagement (default: 7)'
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    if not args.snapshot_dir.exists() or not args.snapshot_dir.is_dir():
        print(f"❌ Error: Snapshot directory not found: {args.snapshot_dir}", file=sys.stderr)
        sys.exit(1)
    
    # Create output directory
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {args.out.absolute()}")
    
    # Define required files
    engagement_file = args.snapshot_dir / "engagement_logs.csv"
    outcomes_file = args.snapshot_dir / "outcomes.csv"
    
    try:
        # Validate required files and columns
        validate_csv_file(engagement_file, ['ghl_contact_id', 'event_ts'])
        validate_csv_file(outcomes_file, ['ghl_contact_id', 'created_at'])
        
        # Load data
        print("Loading engagement logs data...")
        engagement_data, engagement_row_count = load_engagement_data(engagement_file)
        print(f"✅ Loaded {len(engagement_data)} unique contacts from {engagement_row_count} engagement rows")
        
        print("Loading outcomes data...")
        outcome_data, outcome_row_count = load_outcome_data(outcomes_file)
        print(f"✅ Loaded {len(outcome_data)} unique contacts from {outcome_row_count} outcome rows")
        
        # Build labels
        print(f"Building labels with {args.label_window_days}-day window...")
        labels = build_labels(engagement_data, outcome_data, args.label_window_days)
        
        # Write labels CSV
        labels_file = args.out / "labels.csv"
        with open(labels_file, 'w', newline='', encoding='utf-8') as f:
            if labels:
                writer = csv.DictWriter(f, fieldnames=labels[0].keys())
                writer.writeheader()
                writer.writerows(labels)
        
        print(f"✅ Created {len(labels)} labels in {labels_file.name}")
        
        # Create manifest
        manifest_path = create_labels_manifest(args, engagement_row_count, outcome_row_count, len(labels))
        
        # Print summary
        print("\n" + "="*60)
        print("✅ Label building completed successfully.")
        print("="*60)
        print(f"Snapshot directory: {args.snapshot_dir}")
        print(f"Label window: {args.label_window_days} days")
        print(f"Engagement rows processed: {engagement_row_count}")
        print(f"Outcome rows processed: {outcome_row_count}")
        print(f"Labels generated: {len(labels)}")
        positive_labels = sum(1 for label in labels if label['booked_call_within_7d'] == 1)
        print(f"Positive labels (booked): {positive_labels}")
        print(f"Negative labels (not booked): {len(labels) - positive_labels}")
        print(f"Labels file: {labels_file.name}")
        print(f"Manifest: {manifest_path.name}")
        
    except FileNotFoundError as e:
        print(f"❌ File error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"❌ Validation error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error during label building: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()