# Runbook

All commands are PowerShell-safe (no `&&`). Run from the repo root.

## Prerequisites

```powershell
pip install pandas numpy scikit-learn joblib pyodbc
```

Set environment variables (see `.env.example`):

```powershell
$env:CORY_SQL_HOST = "your-sql-server"
$env:CORY_SQL_DATABASE = "AgentCoryDB"
$env:CORY_SQL_AUTH_MODE = "integrated"
$env:PYTHONUTF8 = "1"
```

## Run Tests

```powershell
python -m pytest tests/ -v
```

## Train a Model (end-to-end)

Extract data, train baseline logistic regression, persist scores:

```powershell
python execution/run_pipeline.py `
    --since 2024-01-01 --until 2024-06-01 `
    --target booked_call_within_7d `
    --out-root ./tmp/runs `
    --outcomes-query-file ./queries/outcomes.sql `
    --training-examples-query-file ./queries/gold_training_examples_proxy_v2.sql
```

Skip score persistence:

```powershell
python execution/run_pipeline.py `
    --since 2024-01-01 --until 2024-06-01 `
    --target booked_call_within_7d `
    --out-root ./tmp/runs `
    --outcomes-query-file ./queries/outcomes.sql `
    --training-examples-query-file ./queries/gold_training_examples_proxy_v2.sql `
    --no-persist-scores
```

## Promote a Model to the Registry

After a successful training run, promote the model to make it the active version:

```powershell
python execution/promote_model.py `
    --run-dir ./tmp/runs/<completed_train_run>
```

With a PR-AUC quality gate:

```powershell
python execution/promote_model.py `
    --run-dir ./tmp/runs/<completed_train_run> `
    --min-pr-auc 0.15 `
    --notes "v2 with voicemail features"
```

Force promotion (skip PR-AUC check):

```powershell
python execution/promote_model.py `
    --run-dir ./tmp/runs/<completed_train_run> `
    --force
```

## Model Registry Layout

```
tmp/model_registry/
  active/                    # always points to the latest promoted version
    model.joblib
    scaler.joblib
    metrics.json
    model_card.json
  versions/
    20240615T120000Z/        # timestamped version snapshots
      model.joblib
      scaler.joblib
      metrics.json
      model_card.json
    20240701T080000Z/
      ...
```

## Score New Leads (predict mode)

Using the model registry (recommended):

```powershell
python execution/run_pipeline.py --mode predict `
    --registry-dir ./tmp/model_registry `
    --since 2024-06-01 --until 2024-07-01 `
    --target booked_call_within_7d `
    --out-root ./tmp/runs `
    --outcomes-query-file ./queries/outcomes.sql `
    --training-examples-query-file ./queries/gold_training_examples_proxy_v2.sql
```

Using an explicit artifacts directory (overrides registry):

```powershell
python execution/run_pipeline.py --mode predict `
    --artifacts-dir ./tmp/runs/<previous_train_run> `
    --since 2024-06-01 --until 2024-07-01 `
    --target booked_call_within_7d `
    --out-root ./tmp/runs `
    --outcomes-query-file ./queries/outcomes.sql `
    --training-examples-query-file ./queries/gold_training_examples_proxy_v2.sql
```

## Daily Scoring Job

Using the model registry:

```powershell
python execution/run_daily.py `
    --registry-dir ./tmp/model_registry `
    --outcomes-query-file ./queries/outcomes.sql `
    --training-examples-query-file ./queries/gold_training_examples_proxy_v2.sql
```

Using an explicit artifacts directory:

```powershell
python execution/run_daily.py `
    --artifacts-dir ./tmp/runs/<latest_train_run> `
    --outcomes-query-file ./queries/outcomes.sql `
    --training-examples-query-file ./queries/gold_training_examples_proxy_v2.sql
```

Weekly retrain (120-day window):

```powershell
python execution/run_daily.py --mode train --days-back 120 `
    --outcomes-query-file ./queries/outcomes.sql `
    --training-examples-query-file ./queries/gold_training_examples_proxy_v2.sql
```

### PowerShell Task Scheduler

Daily predict using registry (06:00 UTC):

```powershell
schtasks /create /tn "LeadScoring_Daily" /tr `
    "C:\Python312\python.exe C:\path\to\execution\run_daily.py --registry-dir C:\path\to\tmp\model_registry --outcomes-query-file C:\path\to\queries\outcomes.sql --training-examples-query-file C:\path\to\queries\gold_training_examples_proxy_v2.sql" `
    /sc daily /st 06:00
```

Weekly retrain (Sunday 02:00):

```powershell
schtasks /create /tn "LeadScoring_WeeklyRetrain" /tr `
    "C:\Python312\python.exe C:\path\to\execution\run_daily.py --mode train --days-back 120 --outcomes-query-file C:\path\to\queries\outcomes.sql --training-examples-query-file C:\path\to\queries\gold_training_examples_proxy_v2.sql" `
    /sc weekly /d SUN /st 02:00
```

## Individual Scripts

### Extract snapshot only

```powershell
python execution/extract_snapshot.py `
    --since 2024-01-01 --until 2024-06-01 `
    --target booked_call_within_7d `
    --out ./tmp/snapshots/my_snapshot `
    --outcomes-query-file ./queries/outcomes.sql `
    --training-examples-query-file ./queries/gold_training_examples_proxy_v2.sql
```

### Train only (from existing CSV)

```powershell
python execution/train_baseline.py `
    --training-examples-csv ./tmp/runs/<run>/training_examples.csv
```

With a custom label column:

```powershell
python execution/train_baseline.py `
    --training-examples-csv ./tmp/runs/<run>/training_examples.csv `
    --label-col label_responded_within_14d
```

### Predict only (from existing CSV + artifacts)

```powershell
python execution/predict.py `
    --training-examples-csv ./tmp/runs/<run>/training_examples.csv
```

### Persist scores to SQL

Dry run first (always recommended):

```powershell
python execution/write_scores_to_sql.py `
    --predictions-csv ./tmp/runs/<run>/predictions.csv `
    --metrics-json ./tmp/runs/<run>/metrics.json `
    --dry-run
```

Write test-split rows only:

```powershell
python execution/write_scores_to_sql.py `
    --predictions-csv ./tmp/runs/<run>/predictions.csv `
    --metrics-json ./tmp/runs/<run>/metrics.json `
    --split test
```

Write all rows:

```powershell
python execution/write_scores_to_sql.py `
    --predictions-csv ./tmp/runs/<run>/predictions.csv `
    --metrics-json ./tmp/runs/<run>/metrics.json
```

## Output Artifacts

Each run creates a folder under `--out-root` containing:

| File | Description |
|---|---|
| `training_examples.csv` | Extracted feature/label data |
| `outcomes.csv` | Raw outcome data |
| `manifest.json` | Extraction metadata |
| `model.joblib` | Trained model (train mode only) |
| `scaler.joblib` | Fitted scaler (train mode only) |
| `predictions.csv` | Scored predictions |
| `metrics.json` | Evaluation metrics and run metadata |

## Logging

All scripts accept `--log-level {DEBUG,INFO,WARNING,ERROR}`. Default comes from the `LOG_LEVEL` environment variable (fallback: `INFO`).

```powershell
$env:LOG_LEVEL = "DEBUG"
python execution/train_baseline.py --training-examples-csv ./tmp/runs/<run>/training_examples.csv
```

Or per-invocation:

```powershell
python execution/train_baseline.py `
    --training-examples-csv ./tmp/runs/<run>/training_examples.csv `
    --log-level DEBUG
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `UnicodeEncodeError` on Windows | Set `$env:PYTHONUTF8 = "1"` |
| `ModuleNotFoundError: config` | Run from repo root or add `execution/` to `PYTHONPATH` |
| `pyodbc.Error: ... driver not found` | Install [ODBC Driver 18 for SQL Server](https://learn.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-server) |
| `Cannot train: training split has only one class` | Expand date range or use a different label column |
| `model.joblib not found` | Run training first, promote a model (`promote_model.py`), or provide `--artifacts-dir` / `--registry-dir` |
