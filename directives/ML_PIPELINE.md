# ML Pipeline Contract
**Lead Conversion Optimization System**

## Purpose
Improve lead conversion rates by learning from campaign and lead interaction data to recommend data-driven prompt and campaign design changes.

## Data Source
**SQL Server (Agent Cory DB)** serves as the single source of truth.

### Required Entities (High-Level)
- **Contacts** - Lead demographics, source attribution, qualification status
- **Campaign Steps** - Sequence definitions, prompt templates, decision points
- **Engagement Logs** - Interaction timestamps, response content, engagement metrics
- **Outcomes** - Conversion events, call bookings, qualification results

## Targets & Evaluation Metrics

### Targets (Choose up to 3)
- **T1: booked_call_within_7d** - 1 if a call is booked within 7 days of first engagement, else 0
- **T2: qualified_within_14d** - 1 if qualified within 14 days, else 0  
- **T3: conversion_within_30d** - 1 if customer conversion within 30 days, else 0

### Metrics + Gates
- **Primary** - PR-AUC (due to class imbalance) with minimum improvement vs baseline (e.g., +5% relative)
- **Secondary** - Calibration (Brier score or calibration curve), and lift@topK (e.g., lift in top 10% scored leads)

### Baseline
- Simple baseline model (logistic regression or rules-based)
- Current campaign performance comparator

## Pipeline Stages

### 1. Extract
- Pull data from Agent Cory DB with read-only access
- Create timestamped dataset snapshots
- Validate data completeness and quality gates

### 2. Clean
- Handle missing values, duplicates, and outliers
- Standardize formats and encoding
- Apply business rules for data consistency

### 3. Feature Engineering
- Generate interaction patterns, timing features, content analysis
- Calculate engagement scores and conversion indicators
- Create campaign performance metrics

### 4. Train
- Use fixed random seeds for reproducibility
- Train conversion prediction models
- Generate feature importance rankings

### 5. Evaluate
- Apply holdout validation with metric thresholds
- Compare against baseline performance
- Generate model performance reports

### 6. Recommend
- Identify underperforming campaign elements
- Suggest prompt optimizations and step sequence changes
- Prioritize recommendations by impact potential

### 7. Publish
- Store results in model registry with versioning
- Generate human-readable recommendation reports
- Create reproducible experiment documentation

## Determinism & Gating

### Reproducibility
- **Fixed Seeds** - All random operations use consistent seeds
- **Dataset Snapshots** - Immutable, timestamped training datasets
- **Model Registry** - Versioned artifacts with lineage tracking
- **Containerized Runs** - Consistent execution environments

### Quality Gates
- **Metric Thresholds** - Minimum performance standards before deployment
- **Data Quality Checks** - Automated validation of input data integrity
- **Model Drift Detection** - Alert on performance degradation

## Output Artifacts

### Operational
- **Scored Leads** - Conversion probability rankings for active leads
- **Feature Attribution** - Key factors driving conversion predictions

### Strategic
- **Recommended Prompt Edits** - Specific language optimizations
- **Recommended Step Sequencing** - Campaign flow improvements
- **Experiment Plan** - Proposed A/B tests for validation

## Safety Constraints

### Database Access
- **Read-Only by Default** - No write permissions to production systems
- **Sandbox Testing** - All experiments run against dev/test environments
- **Access Logging** - Audit trail of all data access

### Change Management
- **Human Approval Required** - No automated campaign modifications
- **Staged Rollouts** - Gradual deployment of recommendations
- **Rollback Capability** - Ability to revert changes quickly

### Data Protection
- **PII Handling** - Anonymization of sensitive contact information
- **Retention Policies** - Automatic cleanup of temporary datasets
- **Access Controls** - Role-based permissions for pipeline components

---
*This directive defines the contract for ML-driven lead conversion optimization. Updates must be approved and tested before implementation.*