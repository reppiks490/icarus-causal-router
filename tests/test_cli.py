import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class CLITests(unittest.TestCase):
    def call(self, *args):
        return subprocess.run([sys.executable, "-m", "causal_router.cli", *map(str, args)],
                              capture_output=True, text=True)

    def test_demo_runs_and_verifies_identical_recovery(self):
        with tempfile.TemporaryDirectory() as root:
            out = Path(root) / "demo"
            result = self.call("demo", "--output", out, "--steps", "40", "--seed", "7")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((out / "report.json").is_file())
            report = json.loads((out / "report.json").read_text())
            self.assertTrue(report["synthetic"])
            self.assertFalse(report["execution_allowed"])
            self.assertEqual(report["summary"]["metrics"]["settled"], 40)
            verify = self.call("verify", "--db", out / "research.db", "--policy", out / "policy.json")
            self.assertEqual(verify.returncode, 0, verify.stderr)
            self.assertEqual(json.loads(verify.stdout)["summary"], report["summary"])

    def test_unknown_command_is_an_error(self):
        self.assertNotEqual(self.call("trade").returncode, 0)

    def test_verify_missing_database_does_not_create_one(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "missing.db"
            self.assertNotEqual(self.call("verify", "--db", path, "--policy", "missing.json").returncode, 0)
            self.assertFalse(path.exists())

    def test_existing_demo_directory_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as root:
            r = self.call("demo", "--output", root, "--steps", "10")
            self.assertNotEqual(r.returncode, 0)
            self.assertEqual(list(Path(root).iterdir()), [])


if __name__ == "__main__": unittest.main()
