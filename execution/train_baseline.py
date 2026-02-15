"""
Baseline training script for ML pipeline.

Trains a simple logistic regression model on gold training examples data.
Produces predictions with time-based train/test split for temporal validation.

Usage:
    python train_baseline.py --training-examples-csv ./tmp/snapshots/gold_run_1/training_examples.csv
"""

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path
import warnings

try:
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score, classification_report
    from sklearn.preprocessing import StandardScaler
except ImportError as e:
    print(f"❌ Required packages not available: {e}")
    print("Install with: pip install pandas numpy scikit-learn")
    sys.exit(1)

# Set random state for reproducibility
RANDOM_STATE = 42


def validate_csv_file(file_path, label_col="label_responded_within_7d"):
    """Validate CSV file exists and has required columns."""
    if not file_path.exists():
        raise FileNotFoundError(f"Training examples CSV not found: {file_path}")

    if not file_path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")

    # Check required columns
    required_cols = [
        'org_id', 'enrollment_id', 'ghl_contact_id', 'decision_ts_utc',
        label_col,
        'attempts_sms_24h', 'attempts_email_24h', 'attempts_voice_no_voicemail_24h', 'voicemail_drops_24h'
    ]
    
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        headers = set(reader.fieldnames or [])
        missing_cols = set(required_cols) - headers
        if missing_cols:
            raise ValueError(f"Missing required columns in {file_path.name}: {missing_cols}")
    
    return True


def load_and_prepare_data(csv_path, label_col="label_responded_within_7d"):
    """Load training data and prepare features/target."""
    print(f"Loading data from {csv_path}...")

    # Load CSV
    df = pd.read_csv(csv_path)

    # Parse decision timestamp
    df['decision_ts_utc'] = pd.to_datetime(df['decision_ts_utc'])

    # Define feature columns (no labels)
    feature_cols = [
        'attempts_sms_24h', 'attempts_email_24h', 'attempts_voice_no_voicemail_24h', 'voicemail_drops_24h'
    ]

    # Extract features and target
    X = df[feature_cols].copy()
    y = df[label_col].copy()

    # Handle missing values (fill all with 0 - counts and binary ints from SQL)
    for col in feature_cols:
        X[col] = X[col].fillna(0)

    # Convert target to int (fill missing labels with 0)
    y = y.fillna(0).astype(int)
    
    print(f"✅ Loaded {len(df)} rows with {len(feature_cols)} features")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    return df, X, y, feature_cols


def create_time_split(df, test_size=0.2):
    """Create train/test split based on decision timestamp (earliest 80% train, latest 20% test)."""
    # Sort by decision timestamp
    df_sorted = df.sort_values('decision_ts_utc').reset_index(drop=True)
    
    # Calculate split point
    n_total = len(df_sorted)
    n_train = int(n_total * (1 - test_size))
    
    # Create split indicator
    split_labels = ['train'] * n_train + ['test'] * (n_total - n_train)
    df_sorted['split'] = split_labels
    
    train_idx = df_sorted['split'] == 'train'
    test_idx = df_sorted['split'] == 'test'
    
    print(f"Time-based split: {n_train} train ({n_train/n_total*100:.1f}%), {n_total-n_train} test ({(n_total-n_train)/n_total*100:.1f}%)")
    
    if n_train > 0:
        train_date_range = f"{df_sorted[train_idx]['decision_ts_utc'].min()} to {df_sorted[train_idx]['decision_ts_utc'].max()}"
        print(f"Train date range: {train_date_range}")
    
    if n_total - n_train > 0:
        test_date_range = f"{df_sorted[test_idx]['decision_ts_utc'].min()} to {df_sorted[test_idx]['decision_ts_utc'].max()}"
        print(f"Test date range: {test_date_range}")
    
    return df_sorted, train_idx, test_idx


def train_baseline_model(X_train, y_train, X_test, y_test):
    """Train logistic regression baseline model."""
    if len(X_train) == 0:
        raise ValueError("No training data available")
    
    print(f"Training logistic regression on {len(X_train)} samples...")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test) if len(X_test) > 0 else np.array([]).reshape(0, X_train.shape[1])
    
    # Train model
    model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000, class_weight="balanced")
    model.fit(X_train_scaled, y_train)
    
    # Generate predictions
    train_scores = model.predict_proba(X_train_scaled)[:, 1] if len(X_train_scaled) > 0 else np.array([])
    test_scores = model.predict_proba(X_test_scaled)[:, 1] if len(X_test_scaled) > 0 else np.array([])
    
    print(f"✅ Model trained successfully")
    
    return model, scaler, train_scores, test_scores


def compute_metrics(y_true, scores, split_name):
    """Compute and print evaluation metrics."""
    if len(y_true) == 0 or len(scores) == 0:
        print(f"{split_name}: No data for evaluation")
        return {}
    
    metrics = {}
    
    # Basic stats
    n_samples = len(y_true)
    n_positive = sum(y_true)
    positive_rate = n_positive / n_samples if n_samples > 0 else 0
    
    metrics['n_samples'] = n_samples
    metrics['n_positive'] = n_positive
    metrics['positive_rate'] = positive_rate
    
    print(f"{split_name}: {n_samples} samples, {n_positive} positive ({positive_rate:.3f} rate)")
    
    # Skip advanced metrics for very small datasets or edge cases
    if n_samples < 5 or n_positive == 0 or n_positive == n_samples:
        print(f"{split_name}: Skipping advanced metrics (insufficient or homogeneous data)")
        return metrics
    
    try:
        # PR-AUC (preferred)
        pr_auc = average_precision_score(y_true, scores)
        metrics['pr_auc'] = pr_auc
        print(f"{split_name}: PR-AUC = {pr_auc:.4f}")
        
        # ROC-AUC (fallback)
        if len(np.unique(y_true)) > 1:  # Need both classes for ROC-AUC
            roc_auc = roc_auc_score(y_true, scores)
            metrics['roc_auc'] = roc_auc
            print(f"{split_name}: ROC-AUC = {roc_auc:.4f}")
    
    except Exception as e:
        print(f"{split_name}: Could not compute advanced metrics: {e}")
    
    return metrics


def create_predictions_output(df_with_split, train_scores, test_scores, output_path, label_col="label_responded_within_7d"):
    """Create predictions CSV with scores and metadata."""
    # Copy dataframe to avoid modifying original
    df_out = df_with_split.copy()

    # Assign scores directly using boolean masks (safer than enumerate/iterrows)
    train_idx = df_out['split'] == 'train'
    test_idx = df_out['split'] == 'test'

    df_out.loc[train_idx, 'score'] = train_scores if len(train_scores) > 0 else 0.0
    df_out.loc[test_idx, 'score'] = test_scores if len(test_scores) > 0 else 0.0

    # Add split_order for proper sorting (test=0 first, train=1 second)
    df_out['split_order'] = df_out['split'].map({'test': 0, 'train': 1})

    # Select output columns
    output_cols = [
        'org_id', 'enrollment_id', 'ghl_contact_id', 'decision_ts_utc',
        'score', label_col, 'split'
    ]
    pred_df = df_out[output_cols + ['split_order']].copy()
    pred_df.rename(columns={label_col: 'y_true'}, inplace=True)
    
    # Sort by split_order (test first), then score desc within each split
    pred_df_sorted = pred_df.sort_values(['split_order', 'score'], ascending=[True, False])
    
    # Drop split_order column before output
    pred_df_sorted = pred_df_sorted.drop(columns=['split_order'])
    
    # Write to CSV
    pred_df_sorted.to_csv(output_path, index=False)
    print(f"✅ Predictions written to {output_path}")
    
    return pred_df_sorted


def main():
    parser = argparse.ArgumentParser(
        description="Train baseline logistic regression model on gold training examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--training-examples-csv',
        required=True,
        type=Path,
        help='Path to training_examples.csv file'
    )

    parser.add_argument(
        '--label-col',
        default='label_responded_within_7d',
        help='Name of the label column to use as target (default: label_responded_within_7d)'
    )

    args = parser.parse_args()

    # Validate input file
    try:
        validate_csv_file(args.training_examples_csv, label_col=args.label_col)
    except (FileNotFoundError, ValueError) as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Determine output path
    predictions_path = args.training_examples_csv.parent / "predictions.csv"
    
    try:
        # Load and prepare data
        df, X, y, feature_cols = load_and_prepare_data(args.training_examples_csv, label_col=args.label_col)
        
        # Check for tiny dataset
        if len(df) < 20:
            print(f"⚠️  Warning: Small dataset ({len(df)} rows). Metrics may be unreliable.")
        
        # Create time-based split
        df_with_split, train_idx, test_idx = create_time_split(df)
        
        # Rebuild X and y from sorted dataframe to align with split indices
        X = df_with_split[feature_cols].copy()
        y = df_with_split[args.label_col].fillna(0).astype(int).copy()

        X_train = X[train_idx]
        y_train = y[train_idx]
        X_test = X[test_idx]
        y_test = y[test_idx]

        # Early-exit if training split has only one class
        if y_train.nunique() < 2:
            print("⚠️ Cannot train: training split has only one class (all 0s or all 1s). "
                  "Expand date range or change label.")
            sys.exit(0)

        # Train model
        model, scaler, train_scores, test_scores = train_baseline_model(X_train, y_train, X_test, y_test)
        
        # Compute and print metrics
        print("\n" + "="*60)
        print("MODEL EVALUATION")
        print("="*60)
        
        train_metrics = compute_metrics(y_train.values, train_scores, "TRAIN")
        test_metrics = compute_metrics(y_test.values, test_scores, "TEST")
        
        # Create predictions output
        print("\n" + "="*60)
        print("OUTPUT GENERATION")
        print("="*60)
        
        pred_df = create_predictions_output(df_with_split, train_scores, test_scores, predictions_path, label_col=args.label_col)
        
        # Final summary
        print("\n" + "="*60)
        print("✅ BASELINE TRAINING COMPLETED")
        print("="*60)
        print(f"Input: {args.training_examples_csv}")
        print(f"Output: {predictions_path}")
        print(f"Features used: {len(feature_cols)}")
        print(f"Total samples: {len(df)}")
        print(f"Train samples: {sum(train_idx)}")
        print(f"Test samples: {sum(test_idx)}")
        
        if 'pr_auc' in test_metrics:
            print(f"Test PR-AUC: {test_metrics['pr_auc']:.4f}")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()