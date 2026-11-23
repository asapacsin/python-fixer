"""Audit-tool tests. Rows are marked VERIFIED with synthetic test-double evidence:
this checks audit logic only and is not sandbox or data-quality evidence."""
import json
from pathlib import Path
import tempfile
import unittest

from python_fixer.common import digest
from python_fixer.data.audit import build_audit, mechanical_checks, sample
from python_fixer.data.build import fixture_seeds, plan_rows, write_dataset
from python_fixer.data.models import mark_verified


def synthetic_verified(row):
    ids = [t["test_id"] for t in row["tests"]]
    return mark_verified(row, {"status": "VERIFIED", "baseline_passed_ids": ids[:1],
                               "baseline_failed_ids": ids[1:], "image_digest": "python@sha256:" + "a" * 64,
                               "policy_sha256": "0" * 64, "evidence_sha256": "1" * 64})


class AuditTests(unittest.TestCase):
    def setUp(self):
        seeds = fixture_seeds("authored-fixtures-v2")
        self.rows = [synthetic_verified(r) for r in plan_rows(seeds, source="authored-fixtures-v2")]
        self.tmp = tempfile.TemporaryDirectory()
        self.directory = Path(self.tmp.name) / "data"
        config = {"source": "authored-fixtures-v2", "schema_version": 1}
        self.manifest = write_dataset(self.directory, self.rows, {}, [{"problem_id": "fixture-gcd-v2-x", "reason": "r"}], config)

    def tearDown(self):
        self.tmp.cleanup()

    def test_sample_is_deterministic_seeded_and_sized(self):
        first = [r["problem_id"] for r in sample(self.rows, 17)]
        self.assertEqual(first, [r["problem_id"] for r in sample(list(reversed(self.rows)), 17)])
        self.assertEqual(len(first), 50)
        self.assertEqual(len(set(first)), 50)
        self.assertNotEqual(first, [r["problem_id"] for r in sample(self.rows, 18)])
        with self.assertRaises(ValueError):
            sample(self.rows[:49], 17)

    def test_report_binds_manifest_and_leaves_manual_review_pending(self):
        report = build_audit(self.directory, 17, digest(self.manifest))
        self.assertEqual(report["manifest_sha256"], digest(self.manifest))
        self.assertEqual(report["manual_review_status"], "PENDING")
        self.assertTrue(all(c["manual_review"]["verdict"] == "PENDING" and c["manual_review"]["reviewer"] is None
                            for c in report["cases"]))
        self.assertEqual(report["attrition"]["rejected_rows"], 1)
        self.assertEqual(report["attrition"]["planned_rows"], len(self.rows) + 1)
        with self.assertRaises(ValueError):
            build_audit(self.directory, 17, "0" * 64)

    def test_tampered_dataset_is_rejected(self):
        path = self.directory / "problems.jsonl"
        path.write_bytes(path.read_bytes().replace(b"Return", b"Retunr", 1))
        with self.assertRaises(ValueError):
            build_audit(self.directory, 17)

    def test_mechanical_checks_flag_reformatted_mutant(self):
        row = next(r for r in self.rows if r["source_function_id"] == "fixture-binary_search-v2")
        self.assertTrue(all(mechanical_checks(row).values()))
        bad = json.loads(json.dumps(row))
        bad["provenance"]["mutation_operator"] = "ast-10-ops-0-Lt"
        bad["buggy_code"] = bad["reference_code"].replace("lo, hi = 0, len(xs) - 1", "lo, hi = (0, len(xs) - 1)").replace("lo <= hi", "lo < hi")
        self.assertFalse(mechanical_checks(bad)["mutant_is_single_line_edit"])

    def test_leakage_checks_fail_when_view_leaks(self):
        from unittest import mock
        import python_fixer.data.audit as audit
        row = next(r for r in self.rows if r["source_function_id"] == "fixture-merge_sorted-v2")
        real = audit.model_view
        leaks = {
            "evaluation_view_excludes_reference": lambda v, r: v["prompt"][1]["content"] + "\n" + r["reference_code"],
            "evaluation_view_excludes_test_records": lambda v, r: v["prompt"][1]["content"] + json.dumps(r["tests"]),
            "evaluation_view_excludes_provenance": lambda v, r: v["prompt"][1]["content"] + r["test_suite_sha256"],
        }
        for check, inject in leaks.items():
            def leaky(r, role, inject=inject):
                view = real(r, role)
                view["prompt"][1]["content"] = inject(view, r)
                return view
            with self.subTest(check=check), mock.patch.object(audit, "model_view", leaky):
                self.assertFalse(mechanical_checks(row)[check])
        def extra_field(r, role):
            view = real(r, role)
            view["tests"] = r["tests"]
            return view
        with mock.patch.object(audit, "model_view", extra_field):
            self.assertFalse(mechanical_checks(row)["evaluation_view_only_prompt_and_id"])
        self.assertTrue(all(mechanical_checks(row).values()))

    def test_planner_rejects_seeds_outside_source_and_labels_mutants(self):
        with self.assertRaises(ValueError):
            plan_rows(fixture_seeds("authored-fixtures-v2"))  # v2 seeds under default v1 source
        rows = plan_rows(fixture_seeds("authored-fixtures-v2"), source="authored-fixtures-v2",
                         mutations_per_train_function=2)
        mutants = [r for r in rows if r["provenance"]["mutation_operator"] != "authored-bug-v1"]
        self.assertTrue(mutants)
        self.assertTrue(all(r["difficulty"] == "easy" for r in mutants))

    def test_behavioural_duplicates_are_dropped_authored_first(self):
        from python_fixer.data.build import dedupe_by_behaviour
        rows = [r for r in plan_rows(fixture_seeds("authored-fixtures-v2"), source="authored-fixtures-v2",
                                     mutations_per_train_function=2)
                if r["source_function_id"] == "fixture-median-v2"]
        self.assertGreaterEqual(len(rows), 2)
        evidence = {r["problem_id"]: {"runs": [{"buggy": {"outcome_sha256": "same"}}]} for r in rows}
        kept, dropped = dedupe_by_behaviour(rows, evidence)
        self.assertEqual([r["provenance"]["mutation_operator"] for r in kept], ["authored-bug-v1"])
        self.assertEqual(len(dropped), len(rows) - 1)
        self.assertTrue(all(d["origin"] == "mutant" and d["bug_category"] for d in dropped))


if __name__ == "__main__":
    unittest.main()
