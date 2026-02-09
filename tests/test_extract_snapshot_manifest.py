import unittest
import subprocess
import json
import tempfile
from pathlib import Path
import sys


class TestExtractSnapshotManifest(unittest.TestCase):
    
    def test_manifest_creation(self):
        """Test that extract_snapshot.py creates a valid manifest file."""
        # Create temporary directory for test output
        with tempfile.TemporaryDirectory() as temp_dir:
            # Path to the script
            script_path = Path(__file__).parent.parent / "execution" / "extract_snapshot.py"
            
            # Run the script
            result = subprocess.run([
                sys.executable, str(script_path),
                "--since", "2024-01-01",
                "--until", "2024-01-31", 
                "--out", temp_dir,
                "--target", "booked_call_within_7d"
            ], capture_output=True, text=True)
            
            # Assert script ran successfully
            self.assertEqual(result.returncode, 0, f"Script failed with output: {result.stderr}")
            
            # Find the manifest file(s)
            manifest_files = list(Path(temp_dir).glob("run_manifest_*.json"))
            self.assertGreaterEqual(len(manifest_files), 1, "Should create at least one manifest file")
            
            # Select the newest manifest file by modification time
            newest_manifest = max(manifest_files, key=lambda p: p.stat().st_mtime)
            
            # Load and validate manifest
            with open(newest_manifest, 'r') as f:
                manifest = json.load(f)
            
            # Assert required keys exist
            required_keys = ["run_timestamp", "mode", "args", "git_commit", "deterministic_seed", "entities_to_extract"]
            for key in required_keys:
                self.assertIn(key, manifest, f"Manifest should contain key: {key}")
            
            # Assert specific values
            self.assertEqual(manifest["mode"], "READ_ONLY_SCAFFOLD")
            self.assertEqual(manifest["deterministic_seed"], 42)
            self.assertEqual(manifest["args"]["target"], "booked_call_within_7d")
            
            # Assert entities_to_extract is a list with expected entities
            expected_entities = ["contacts", "campaign_steps", "engagement_logs", "outcomes"]
            self.assertEqual(manifest["entities_to_extract"], expected_entities)
            
            # Assert args structure
            self.assertEqual(manifest["args"]["since"], "2024-01-01")
            self.assertEqual(manifest["args"]["until"], "2024-01-31")
            self.assertEqual(Path(manifest["args"]["output_directory"]).resolve(), Path(temp_dir).resolve())


if __name__ == '__main__':
    unittest.main()