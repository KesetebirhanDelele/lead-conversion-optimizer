"""
Data extraction script for ML pipeline dataset snapshots.

SQL Server (Agent Cory DB) serves as the single source of truth.
This script produces timestamped, immutable dataset snapshots for reproducible ML training.

Usage:
    python extract_snapshot.py --since 2024-01-01 --until 2024-02-01 --out ./data --target booked_call_within_7d
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, date
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


def create_run_manifest(args, output_dir):
    """Create JSON manifest file documenting the extraction run."""
    timestamp = datetime.now().isoformat()
    git_commit = get_git_commit()
    
    # Deterministic seed based on run parameters
    seed_string = f"{args.since}_{args.until}_{args.target}_{timestamp[:10]}"
    deterministic_seed = hash(seed_string) % (2**32)  # 32-bit positive integer
    
    manifest = {
        "timestamp": timestamp,
        "args": {
            "since": args.since,
            "until": args.until,
            "output_directory": args.out,
            "target": args.target
        },
        "git_commit": git_commit,
        "deterministic_seed": deterministic_seed,
        "entities_to_extract": [
            "contacts",
            "campaign_steps", 
            "engagement_logs",
            "outcomes"
        ]
    }
    
    manifest_path = Path(output_dir) / f"run_manifest_{timestamp[:19].replace(':', '-')}.json"
    
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
    
    # Create run manifest
    manifest_path = create_run_manifest(args, output_dir)
    
    # Print status message
    print("\n" + "="*60)
    print("READ-ONLY mode; no DB operations implemented yet.")
    print("="*60)
    print(f"Target: {args.target}")
    print(f"Date range: {args.since} to {args.until}")
    print(f"Manifest: {manifest_path.name}")


if __name__ == '__main__':
    main()