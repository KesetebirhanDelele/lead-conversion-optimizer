# SQL Server Connection Contract
**Agent Cory DB Integration**

## Purpose
Define connection requirements and configuration for SQL Server (Agent Cory DB), which serves as the single source of truth for lead conversion data.

## Data Access Policy
- **Read-Only Permissions** - Pipeline components access data in read-only mode by default
- **Source of Truth** - Agent Cory DB contains authoritative lead, campaign, and engagement data
- **No Direct Writes** - ML pipeline does not modify production data

## Environment Variables

### Required
- **CORY_SQL_HOST** - SQL Server hostname or IP address
- **CORY_SQL_DATABASE** - Target database name
- **CORY_SQL_AUTH_MODE** - Authentication method (`integrated` or `sqlauth`)

### Optional
- **CORY_SQL_PORT** - SQL Server port (defaults to 1433 if not specified)
- **CORY_SQL_USER** - Username for SQL authentication (required when `CORY_SQL_AUTH_MODE=sqlauth`)
- **CORY_SQL_PASSWORD** - Password for SQL authentication (required when `CORY_SQL_AUTH_MODE=sqlauth`)
- **CORY_SQL_ODBC_DRIVER** - ODBC driver name (defaults to "ODBC Driver 18 for SQL Server")
- **CORY_SQL_ENCRYPT** - Enable connection encryption (`yes` or `no`, defaults to `yes`)
- **CORY_SQL_TRUST_SERVER_CERT** - Trust server certificate without validation (`true` or `false`, defaults to `true`)

## Authentication Guidance

### Development Environment
- **Preferred**: Use `CORY_SQL_AUTH_MODE=integrated` if Windows integrated authentication is available
- **Alternative**: Use `CORY_SQL_AUTH_MODE=sqlauth` with dedicated read-only service account credentials

### Production Environment
- Use dedicated service account with minimal read-only permissions
- Rotate credentials according to security policy
- Never commit credentials to version control

## Connection Security
- Use encrypted connections (TLS/SSL) when available
- Limit network access to database server via firewall rules
- Use principle of least privilege for database permissions
- Audit all data access for compliance

## Validation

### Check Environment Configuration
Run this command to validate required environment variables:

```powershell
python -c "import os; missing = [var for var in ['CORY_SQL_HOST', 'CORY_SQL_DATABASE', 'CORY_SQL_AUTH_MODE'] if not os.getenv(var)]; print('Missing required env vars:', missing) if missing else print('All required SQL Server env vars are set')"
```

### Authentication Mode Validation
- If `CORY_SQL_AUTH_MODE=sqlauth`, also check for `CORY_SQL_USER` and `CORY_SQL_PASSWORD`
- If `CORY_SQL_AUTH_MODE=integrated`, ensure running user has database access

---
*This directive defines SQL Server connection requirements for the ML pipeline. Updates require security review.*