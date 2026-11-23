"""M0 specification tests: no model, GPU, sandbox or candidate execution."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from jsonschema import ValidationError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from spec_reward_oracle import InvalidOutcome, specification_reward
from validate_repository import (config_issues, load, report_path, source_snapshot,
                                 validate_problem, validate_repository)


class RewardSpecificationTests(unittest.TestCase):
    def test_acceptance_table(self):
        cases = load(ROOT / "verification/reward-cases.json")["cases"]
        self.assertGreaterEqual(len(cases), 12)
        for case in cases:
            with self.subTest(case=case["name"]):
                if case.get("expected_error"):
                    with self.assertRaises(InvalidOutcome):
                        specification_reward(case["input"])
                else:
                    self.assertAlmostEqual(specification_reward(case["input"]), case["expected_reward"])

    def test_regression_never_gets_positive_reward(self):
        for fixed in range(5):
            result = specification_reward(dict(status="OK", compiled=True,
                fixed_failed=fixed, total_failed=4, preserved_passed=1,
                total_passed=2, all_passed=False))
            self.assertEqual(result, 0)

    def test_reward_monotone_without_regression(self):
        rewards = [specification_reward(dict(status="OK", compiled=True,
            fixed_failed=n, total_failed=4, preserved_passed=2,
            total_passed=2, all_passed=n == 4)) for n in range(5)]
        self.assertEqual(rewards, sorted(rewards))
        self.assertEqual(rewards[-1], 1)

    def test_boolean_counts_are_not_integers(self):
        with self.assertRaises(InvalidOutcome):
            specification_reward(dict(status="OK", compiled=True,
                fixed_failed=True, total_failed=4, preserved_passed=2,
                total_passed=2, all_passed=False))

    def test_unknown_status_fails_closed(self):
        with self.assertRaises(InvalidOutcome):
            specification_reward({"status": "PRINTED_PASS"})


class RepositoryContractTests(unittest.TestCase):
    def setUp(self):
        self.row = json.loads((ROOT / "examples/problem.jsonl").read_text())
        self.schema = load(ROOT / "schemas/problem.schema.json")

    def test_repository_contracts(self):
        issues, checks = validate_repository(ROOT)
        self.assertEqual(issues, [])
        self.assertGreaterEqual(len(checks), 6)

    def test_example_is_explicitly_unverified(self):
        validate_problem(self.row, self.schema)
        self.assertEqual(self.row["verification"]["status"], "NOT_RUN")

    def test_altered_test_bundle_rejected(self):
        self.row["tests"][0]["expected"] = 99
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            validate_problem(self.row, self.schema)

    def test_duplicate_test_id_rejected(self):
        self.row["tests"][1]["test_id"] = self.row["tests"][0]["test_id"]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_problem(self.row, self.schema)

    def test_claimed_verification_needs_evidence(self):
        self.row["verification"]["status"] = "VERIFIED"
        with self.assertRaises(ValidationError):
            validate_problem(self.row, self.schema)

    def test_partial_grpo_group_rejected(self):
        configs = [load(ROOT / f"configs/{name}.json") for name in ("sft", "grpo", "evaluation")]
        configs[1]["trainer"]["generation_batch_size"] = 15
        self.assertTrue(any("complete groups" in i for i in config_issues(*configs)))

    def test_changed_thinking_mode_rejected(self):
        configs = [load(ROOT / f"configs/{name}.json") for name in ("sft", "grpo", "evaluation")]
        configs[0]["enable_thinking"] = True
        self.assertTrue(any("non-thinking" in i for i in config_issues(*configs)))


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.repo = Path(self.tmp.name)
        subprocess.run(["git", "init", "-q", str(self.repo)], check=True)
        (self.repo / ".gitignore").write_text(".ai-native/\n")
        (self.repo / "spec.md").write_text("original\n")

    def git(self, *args):
        subprocess.run(["git", "-C", str(self.repo), *args], check=True, capture_output=True)

    def test_unborn_snapshot_and_untracked_edit(self):
        before = source_snapshot(self.repo)
        self.assertIsNone(before["commit"])
        (self.repo / "spec.md").write_text("changed\n")
        self.assertNotEqual(before["fingerprint"], source_snapshot(self.repo)["fingerprint"])

    def test_staged_then_reversed_edit_changes_fingerprint(self):
        self.git("add", ".")
        self.git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "initial")
        before = source_snapshot(self.repo)
        (self.repo / "spec.md").write_text("staged change\n")
        self.git("add", "spec.md")
        (self.repo / "spec.md").write_text("original\n")
        self.assertNotEqual(before["fingerprint"], source_snapshot(self.repo)["fingerprint"])

    def test_runtime_evidence_does_not_change_source_fingerprint(self):
        before = source_snapshot(self.repo)
        runtime = self.repo / ".ai-native"
        runtime.mkdir()
        (runtime / "evidence.json").write_text('{}')
        self.assertEqual(before["fingerprint"], source_snapshot(self.repo)["fingerprint"])

    def test_report_cannot_overwrite_source(self):
        with self.assertRaises(ValueError):
            report_path(self.repo, "spec.md")

    def test_report_symlink_escape_rejected(self):
        (self.repo / ".ai-native").mkdir()
        (self.repo / ".ai-native" / "escape").symlink_to(self.repo, target_is_directory=True)
        with self.assertRaises(ValueError):
            report_path(self.repo, ".ai-native/escape/result.json")

    def test_safe_report_path(self):
        self.assertEqual(report_path(self.repo, ".ai-native/static.json"), self.repo / ".ai-native/static.json")


if __name__ == "__main__":
    unittest.main()
