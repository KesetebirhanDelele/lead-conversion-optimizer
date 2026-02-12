import unittest
import subprocess
import json
import tempfile
from pathlib import Path
import sys
from types import SimpleNamespace

from execution.extract_snapshot import create_run_manifest


class TestExtractSnapshotManifest(unittest.TestCase):
    
    def test_manifest_creation(self):
        """Test that create_run_manifest creates a valid manifest file."""
        # Create temporary directory for test output
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create temporary SQL file
            sql_file = Path(temp_dir) / "outcomes.sql"
            with open(sql_file, 'w') as f:
                f.write("SELECT 1 AS test_value;")
            
            # Mock args object
            args = SimpleNamespace(
                since="2024-01-01",
                until="2024-01-31",
                out=temp_dir,
                target="booked_call_within_7d",
                outcomes_query_file=str(sql_file)
            )
            
            # Test with mock row count (simulating successful extraction)
            row_count = 1
            manifest_path = create_run_manifest(args, temp_dir, row_count)
            
            # Load and validate manifest
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            
            # Assert required keys exist
            required_keys = ["run_timestamp", "mode", "args", "git_commit", "deterministic_seed", "entities_to_extract", "row_counts", "files"]
            for key in required_keys:
                self.assertIn(key, manifest, f"Manifest should contain key: {key}")
            
            # Assert specific values
            self.assertEqual(manifest["mode"], "READ_ONLY_EXTRACT")
            self.assertEqual(manifest["deterministic_seed"], 42)
            self.assertEqual(manifest["args"]["target"], "booked_call_within_7d")
            
            # Assert entities_to_extract is a list with expected entities
            expected_entities = ["contacts", "campaign_steps", "engagement_logs", "outcomes"]
            self.assertEqual(manifest["entities_to_extract"], expected_entities)
            
            # Assert row_counts and files structure
            self.assertEqual(manifest["row_counts"]["outcomes"], 1)
            self.assertEqual(manifest["files"]["outcomes"], "outcomes.csv")
            
            # Assert args structure
            self.assertEqual(manifest["args"]["since"], "2024-01-01")
            self.assertEqual(manifest["args"]["until"], "2024-01-31")
            self.assertEqual(Path(manifest["args"]["output_directory"]).resolve(), Path(temp_dir).resolve())
            self.assertEqual(manifest["args"]["outcomes_query_file"], str(sql_file))
    
    def test_invalid_target_exits_nonzero(self):
        """Test that extract_snapshot.py exits with non-zero code for invalid target."""
        # Create temporary directory for test output
        with tempfile.TemporaryDirectory() as temp_dir:
            # Path to the script
            script_path = Path(__file__).parent.parent / "execution" / "extract_snapshot.py"
            
            # Run the script with invalid target
            result = subprocess.run([
                sys.executable, str(script_path),
                "--since", "2024-01-01",
                "--until", "2024-01-31",
                "--out", temp_dir,
                "--target", "not_a_real_target"
            ], capture_output=True, text=True)
            
            # Assert script failed
            self.assertNotEqual(result.returncode, 0, "Script should exit with non-zero code for invalid target")
            
            # Assert error message contains expected phrase
            combined_output = result.stdout + result.stderr
            self.assertIn("Invalid target", combined_output, "Error output should contain 'Invalid target'")


if __name__ == '__main__':
    unittest.main()