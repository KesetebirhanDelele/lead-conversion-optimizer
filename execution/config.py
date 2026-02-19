"""
Centralized defaults and environment variable documentation for the ML pipeline.

This module holds constants and defaults used across execution scripts.
Secrets are NOT loaded here — connection string builders read them at runtime.
"""

import os
from pathlib import Path


# ---------- pipeline defaults ----------

DEFAULT_MODE = "train"
DEFAULT_LABEL_COL = "label_responded_within_7d"
DEFAULT_SCORES_TABLE = "dbo.lead_scores"
MODEL_FILENAME = "model.joblib"
SCALER_FILENAME = "scaler.joblib"
RANDOM_STATE = 42
DEFAULT_REGISTRY_DIR = Path("tmp/model_registry")

# ---------- logging ----------

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# ---------- feature columns ----------

FEATURE_COLS = [
    "attempts_sms_24h",
    "attempts_email_24h",
    "attempts_voice_no_voicemail_24h",
    "voicemail_drops_24h",
]

# ---------- DB connection env var names and non-secret defaults ----------
# These document which env vars the connection string builders expect.
# Secrets (CORY_SQL_USER, CORY_SQL_PASSWORD) are read at runtime only.

DB_ENV_VARS = {
    "CORY_SQL_HOST":              "(required) SQL Server hostname or IP",
    "CORY_SQL_PORT":              "(optional) TCP port, default 1433",
    "CORY_SQL_DATABASE":          "(required) Database name",
    "CORY_SQL_AUTH_MODE":         "(required) 'integrated' or 'sqlauth'",
    "CORY_SQL_ODBC_DRIVER":      "(optional) ODBC driver name, default 'ODBC Driver 18 for SQL Server'",
    "CORY_SQL_ENCRYPT":          "(optional) 'yes' or 'no', default 'yes'",
    "CORY_SQL_TRUST_SERVER_CERT": "(optional) 'true' or 'false', default 'true'",
    "CORY_SQL_USER":             "(required for sqlauth) SQL login username",
    "CORY_SQL_PASSWORD":         "(required for sqlauth) SQL login password",
}

DEFAULT_SQL_PORT = "1433"
DEFAULT_SQL_ODBC_DRIVER = "ODBC Driver 18 for SQL Server"
DEFAULT_SQL_ENCRYPT = "yes"
DEFAULT_SQL_TRUST_SERVER_CERT = "true"
DEFAULT_SQL_TIMEOUT = 30
