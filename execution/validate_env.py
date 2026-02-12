"""
Environment validation script for SQL Server connection configuration.

Validates required environment variables for connecting to Agent Cory DB
according to the SQLSERVER_CONNECTION directive.
"""

import os
import sys


def validate_environment():
    """Validate SQL Server connection environment variables."""
    errors = []
    
    # Required environment variables
    required_vars = ['CORY_SQL_HOST', 'CORY_SQL_DATABASE', 'CORY_SQL_AUTH_MODE']
    
    # Check required variables exist
    for var in required_vars:
        if not os.getenv(var):
            errors.append(f"Missing required environment variable: {var}")
    
    # Validate auth mode if present
    auth_mode = os.getenv('CORY_SQL_AUTH_MODE')
    if auth_mode:
        valid_auth_modes = ['integrated', 'sqlauth']
        if auth_mode not in valid_auth_modes:
            errors.append(f"Invalid CORY_SQL_AUTH_MODE: '{auth_mode}'. Must be one of: {valid_auth_modes}")
        
        # Check sqlauth requirements
        if auth_mode == 'sqlauth':
            sqlauth_required = ['CORY_SQL_USER', 'CORY_SQL_PASSWORD']
            for var in sqlauth_required:
                if not os.getenv(var):
                    errors.append(f"Missing required environment variable for sqlauth: {var}")
    
    # Optional variables (just for informational purposes)
    optional_vars = ['CORY_SQL_PORT', 'CORY_SQL_USER', 'CORY_SQL_PASSWORD']
    present_optional = [var for var in optional_vars if os.getenv(var)]
    
    # Report results
    if errors:
        print("Environment validation FAILED:")
        for error in errors:
            print(f"  ❌ {error}")
        return False
    else:
        print("Environment validation PASSED:")
        print("  ✅ All required SQL Server environment variables are set")
        print(f"  ✅ Auth mode: {auth_mode}")
        if present_optional:
            print(f"  ℹ️  Optional variables present: {', '.join(present_optional)}")
        return True


def main():
    """Main entry point for environment validation."""
    success = validate_environment()
    
    if success:
        print("\n🎉 Environment OK")
        sys.exit(0)
    else:
        print("\n❌ Environment validation failed")
        sys.exit(1)


if __name__ == "__main__":
    main()