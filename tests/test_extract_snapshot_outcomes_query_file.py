import unittest
import subprocess
import tempfile
from pathlib import Path
import sys
import os


class TestExtractSnapshotOutcomesQueryFile(unittest.TestCase):
    
    def test_nonexistent_query_file_exits_nonzero(self):
        """Test that extract_snapshot.py exits with non-zero code for non-existent SQL file."""
        # Create temporary directory for test output
        with tempfile.TemporaryDirectory() as temp_dir:
            # Path to the script
            script_path = Path(__file__).parent.parent / "execution" / "extract_snapshot.py"
            
            # Set up environment with ODBC driver to avoid driver issues
            env = os.environ.copy()
            env['CORY_SQL_ODBC_DRIVER'] = 'ODBC Driver 17 for SQL Server'
            
            # Run the script with non-existent query file
            result = subprocess.run([
                sys.executable, str(script_path),
                "--since", "2024-01-01",
                "--until", "2024-01-31",
                "--out", temp_dir,
                "--target", "booked_call_within_7d",
                "--outcomes-query-file", "queries/does_not_exist.sql"
            ], capture_output=True, text=True, env=env)
            
            # Assert script failed
            self.assertNotEqual(result.returncode, 0, "Script should exit with non-zero code for non-existent SQL file")
            
            # Assert error message contains expected phrase
            combined_output = result.stdout + result.stderr
            self.assertIn("SQL file does not exist", combined_output, "Error output should contain 'SQL file does not exist'")


if __name__ == '__main__':
    unittest.main()