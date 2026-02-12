"""
SQL Server connection verification script.

Tests database connectivity to Agent Cory DB using environment configuration.
Validates environment variables first, then attempts database connection.
"""

import os
import sys

try:
    import pyodbc
except ImportError:
    print("❌ pyodbc not available")
    sys.exit(1)

from execution.validate_env import validate_environment


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


def sanitize_error_message(error_msg):
    """Remove sensitive information from error messages."""
    # Remove any potential password leaks
    sensitive_patterns = [
        os.getenv('CORY_SQL_PASSWORD', ''),
        'PWD=',
        'Password='
    ]
    
    sanitized = str(error_msg)
    for pattern in sensitive_patterns:
        if pattern and pattern in sanitized:
            sanitized = sanitized.replace(pattern, '[REDACTED]')
    
    return sanitized


def test_connection():
    """Test SQL Server database connection."""
    print("Testing SQL Server connection...")
    
    try:
        # Build connection string
        conn_str = build_connection_string()
        
        # Attempt connection
        print(f"Connecting to: {os.getenv('CORY_SQL_HOST')}:{os.getenv('CORY_SQL_PORT', '1433')}")
        print(f"Database: {os.getenv('CORY_SQL_DATABASE')}")
        print(f"Auth mode: {os.getenv('CORY_SQL_AUTH_MODE')}")
        
        with pyodbc.connect(conn_str) as conn:
            cursor = conn.cursor()
            
            # Test query
            cursor.execute("SELECT 1 as test_value")
            result = cursor.fetchone()
            
            if result and result[0] == 1:
                print("✅ SELECT 1 query successful")
                return True
            else:
                print("❌ Unexpected query result")
                return False
                
    except pyodbc.Error as e:
        error_msg = sanitize_error_message(str(e))
        print(f"❌ Database connection failed: {error_msg}")
        return False
    except Exception as e:
        error_msg = sanitize_error_message(str(e))
        print(f"❌ Unexpected error: {error_msg}")
        return False


def main():
    """Main entry point for connection testing."""
    print("SQL Server Connection Test")
    print("=" * 40)
    
    # First validate environment
    if not validate_environment():
        print("\n❌ Environment validation failed - cannot proceed with connection test")
        sys.exit(1)
    
    print("\n" + "=" * 40)
    
    # Test database connection
    if test_connection():
        print("\n🎉 Connection OK")
        sys.exit(0)
    else:
        print("\n❌ Connection test failed")
        sys.exit(1)


if __name__ == "__main__":
    main()