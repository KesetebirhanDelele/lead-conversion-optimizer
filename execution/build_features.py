"""
Feature building script for ML pipeline.

Builds features from engagement logs and joins with labels for training data.
Produces deterministic features based on engagement patterns per contact.

Usage:
    python build_features.py --snapshot-dir ./tmp/snapshots/run1 --labels-file ./tmp/labels/run1/labels.csv --out ./tmp/features
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
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


def load_engagement_features(engagement_file):
    """Load engagement data and compute features per contact."""
    contact_features = {}
    row_count = 0
    
    with open(engagement_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            row_count += 1
            contact_id = row['ghl_contact_id']
            if not contact_id:
                continue
            
            # Initialize contact features if not seen before
            if contact_id not in contact_features:
                contact_features[contact_id] = {
                    'engagement_count': 0,
                    'unique_channels': set(),
                    'inbound_count': 0,
                    'outbound_count': 0,
                    'has_subject': False
                }
            
            features = contact_features[contact_id]
            
            # Update features
            features['engagement_count'] += 1
            
            # Track unique channels
            channel = row.get('channel', '').strip()
            if channel:
                features['unique_channels'].add(channel)
            
            # Count direction
            direction = row.get('direction', '').strip().lower()
            if direction == 'inbound':
                features['inbound_count'] += 1
            elif direction == 'outbound':
                features['outbound_count'] += 1
            
            # Check for subject
            subject = row.get('subject', '').strip()
            if subject and not features['has_subject']:
                features['has_subject'] = True
    
    # Convert sets to counts for serialization
    for contact_id, features in contact_features.items():
        features['unique_channels'] = len(features['unique_channels'])
        features['has_subject'] = 1 if features['has_subject'] else 0
    
    return contact_features, row_count


def load_action_log_features(action_log_file):
    """Load action log data and compute additional features per contact."""
    contact_features = {}
    row_count = 0
    
    if not action_log_file.exists():
        return contact_features, row_count
    
    with open(action_log_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        # Check if required columns exist
        if 'appointment_id' not in reader.fieldnames:
            return contact_features, row_count
        
        for row in reader:
            row_count += 1
            appointment_id = row.get('appointment_id', '').strip()
            if not appointment_id:
                continue
            
            # For now, just count action log entries
            # Future: could extract more sophisticated features
            if appointment_id not in contact_features:
                contact_features[appointment_id] = {
                    'action_log_count': 0
                }
            
            contact_features[appointment_id]['action_log_count'] += 1
    
    return contact_features, row_count


def load_labels(labels_file):
    """Load labels data."""
    labels_data = {}
    row_count = 0
    
    with open(labels_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            row_count += 1
            contact_id = row['ghl_contact_id']
            if not contact_id:
                continue
            
            labels_data[contact_id] = {
                'first_engagement_ts': row.get('first_engagement_ts', ''),
                'booking_created_at': row.get('booking_created_at', ''),
                'booked_call_within_7d': int(row.get('booked_call_within_7d', 0))
            }
    
    return labels_data, row_count


def build_features_dataset(engagement_features, action_log_features, labels_data):
    """Build final features dataset by joining engagement features with labels."""
    features_dataset = []
    
    # Join engagement features with labels
    for contact_id, features in engagement_features.items():
        # Get label data for this contact
        label_data = labels_data.get(contact_id, {
            'first_engagement_ts': '',
            'booking_created_at': '',
            'booked_call_within_7d': 0
        })
        
        # Combine features and labels
        feature_row = {
            'ghl_contact_id': contact_id,
            'engagement_count': features['engagement_count'],
            'unique_channels': features['unique_channels'],
            'inbound_count': features['inbound_count'],
            'outbound_count': features['outbound_count'],
            'has_subject': features['has_subject'],
            'first_engagement_ts': label_data['first_engagement_ts'],
            'booking_created_at': label_data['booking_created_at'],
            'booked_call_within_7d': label_data['booked_call_within_7d']
        }
        
        features_dataset.append(feature_row)
    
    # Sort by contact_id for deterministic output
    features_dataset.sort(key=lambda x: x['ghl_contact_id'])
    
    return features_dataset


def create_features_manifest(args, engagement_row_count, action_log_row_count, labels_row_count, features_row_count):
    """Create JSON manifest file documenting the feature building run."""
    run_timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    git_commit = get_git_commit()
    
    manifest = {
        "run_timestamp": run_timestamp,
        "mode": "FEATURE_BUILD",
        "args": {
            "snapshot_dir": str(args.snapshot_dir),
            "labels_file": str(args.labels_file),
            "output_directory": str(args.out)
        },
        "git_commit": git_commit,
        "inputs": {
            "engagement_logs": "engagement_logs.csv",
            "outcomes_action_log": "outcomes_action_log.csv",
            "labels": str(Path(args.labels_file).name)
        },
        "outputs": {
            "features": "features.csv"
        },
        "row_counts": {
            "engagement_rows": engagement_row_count,
            "action_log_rows": action_log_row_count,
            "label_rows": labels_row_count,
            "feature_rows": features_row_count
        },
        "features_generated": [
            "engagement_count",
            "unique_channels", 
            "inbound_count",
            "outbound_count",
            "has_subject"
        ]
    }
    
    manifest_path = Path(args.out) / f"features_manifest_{run_timestamp[:19].replace(':', '-')}.json"
    
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"Features manifest created: {manifest_path.name}")
    return manifest_path


def main():
    parser = argparse.ArgumentParser(
        description="Build features from engagement logs and join with labels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--snapshot-dir',
        required=True,
        type=Path,
        help='Directory containing engagement_logs.csv and optionally outcomes_action_log.csv'
    )
    
    parser.add_argument(
        '--labels-file',
        required=True,
        type=Path,
        help='Path to labels.csv file'
    )
    
    parser.add_argument(
        '--out',
        required=True,
        type=Path,
        help='Output directory for features.csv and manifest'
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    if not args.snapshot_dir.exists() or not args.snapshot_dir.is_dir():
        print(f"❌ Error: Snapshot directory not found: {args.snapshot_dir}", file=sys.stderr)
        sys.exit(1)
    
    # Validate labels file
    if not args.labels_file.exists():
        print(f"❌ Error: Labels file not found: {args.labels_file}", file=sys.stderr)
        sys.exit(1)
    
    # Create output directory
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {args.out.absolute()}")
    
    # Define input files
    engagement_file = args.snapshot_dir / "engagement_logs.csv"
    action_log_file = args.snapshot_dir / "outcomes_action_log.csv"
    
    try:
        # Validate required files and columns
        validate_csv_file(engagement_file, ['ghl_contact_id', 'channel', 'direction', 'subject'])
        validate_csv_file(args.labels_file, ['ghl_contact_id', 'booked_call_within_7d'])
        
        # Load engagement features
        print("Loading engagement features...")
        engagement_features, engagement_row_count = load_engagement_features(engagement_file)
        print(f"✅ Generated features for {len(engagement_features)} contacts from {engagement_row_count} engagement rows")
        
        # Load action log features (optional)
        print("Loading action log features...")
        action_log_features, action_log_row_count = load_action_log_features(action_log_file)
        if action_log_row_count > 0:
            print(f"✅ Processed {action_log_row_count} action log rows")
        else:
            print("ℹ️  No action log data found (optional)")
        
        # Load labels
        print("Loading labels...")
        labels_data, labels_row_count = load_labels(args.labels_file)
        print(f"✅ Loaded labels for {len(labels_data)} contacts from {labels_row_count} label rows")
        
        # Build features dataset
        print("Building features dataset...")
        features_dataset = build_features_dataset(engagement_features, action_log_features, labels_data)
        
        # Write features CSV
        features_file = args.out / "features.csv"
        with open(features_file, 'w', newline='', encoding='utf-8') as f:
            if features_dataset:
                writer = csv.DictWriter(f, fieldnames=features_dataset[0].keys())
                writer.writeheader()
                writer.writerows(features_dataset)
        
        print(f"✅ Created {len(features_dataset)} feature rows in {features_file.name}")
        
        # Create manifest
        manifest_path = create_features_manifest(args, engagement_row_count, action_log_row_count, labels_row_count, len(features_dataset))
        
        # Print summary
        print("\n" + "="*60)
        print("✅ Feature building completed successfully.")
        print("="*60)
        print(f"Snapshot directory: {args.snapshot_dir}")
        print(f"Labels file: {args.labels_file.name}")
        print(f"Engagement rows processed: {engagement_row_count}")
        print(f"Action log rows processed: {action_log_row_count}")
        print(f"Label rows processed: {labels_row_count}")
        print(f"Feature rows generated: {len(features_dataset)}")
        positive_labels = sum(1 for row in features_dataset if row['booked_call_within_7d'] == 1)
        print(f"Positive samples: {positive_labels}")
        print(f"Negative samples: {len(features_dataset) - positive_labels}")
        print(f"Features file: {features_file.name}")
        print(f"Manifest: {manifest_path.name}")
        
    except FileNotFoundError as e:
        print(f"❌ File error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"❌ Validation error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error during feature building: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()