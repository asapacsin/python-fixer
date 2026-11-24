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
    @classmethod
    def setUpClass(cls):
        # Built once: write_dataset is slow and every test only reads it, except the
        # tamper test, which works on its own copy.
        seeds = fixture_seeds("authored-fixtures-v2")
        cls.rows = [synthetic_verified(r) for r in plan_rows(seeds, source="authored-fixtures-v2")]
        cls.tmp = tempfile.TemporaryDirectory()
        cls.directory = Path(cls.tmp.name) / "data"
        config = {"source": "authored-fixtures-v2", "schema_version": 1}
        cls.manifest = write_dataset(cls.directory, cls.rows, {}, [{"problem_id": "fixture-gcd-v2-x", "reason": "r"}], config)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

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
        import shutil
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp) / "data"
            shutil.copytree(self.directory, copy)
            path = copy / "problems.jsonl"
            path.write_bytes(path.read_bytes().replace(b"Return", b"Retunr", 1))
            with self.assertRaises(ValueError):
                build_audit(copy, 17)
        build_audit(self.directory, 17)  # the shared original is untouched

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
        natural = {
            "evaluation_view_excludes_reference": lambda v, r: v["prompt"][1]["content"] + "\n" + r["reference_code"].replace("    ", "\t").replace(", ", " , "),
            "evaluation_view_excludes_test_records": lambda v, r: v["prompt"][1]["content"] + "".join(
                f"\n>>> merge_sorted({t['args'][0]!r}, {t['args'][1]!r})\n{t['expected']!r}" for t in r["tests"]),
        }
        for check, inject in natural.items():
            def leaky(r, role, inject=inject):
                view = real(r, role)
                view["prompt"][1]["content"] = inject(view, r)
                return view
            with self.subTest(check="natural-" + check), mock.patch.object(audit, "model_view", leaky):
                self.assertFalse(mechanical_checks(row)[check])
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

    def test_crash_only_rule_applies_to_every_variant(self):
        from python_fixer.data.build import crash_only_mutants
        rows = [r for r in plan_rows(fixture_seeds("authored-fixtures-v2"), source="authored-fixtures-v2",
                                     mutations_per_train_function=2)
                if r["source_function_id"] == "fixture-merge_sorted-v2"]
        ids = [t["test_id"] for t in rows[0]["tests"]]
        crash = {"passed_ids": ids[2:], "returned_ids": ids[2:]}        # failing tests all raised
        wrong = {"passed_ids": ids[2:], "returned_ids": ids}            # failing tests returned values
        self.assertIn("authored-bug-v1", {r["provenance"]["mutation_operator"] for r in rows})
        # Rule changed after review round 3: authored bugs are no longer exempt.
        for buggy, kept_expected in ((crash, 0), (wrong, len(rows))):
            evidence = {r["problem_id"]: {"runs": [{"buggy": buggy}]} for r in rows}
            kept, dropped = crash_only_mutants(rows, evidence)
            self.assertEqual(len(kept), kept_expected)
            self.assertEqual(len(dropped), len(rows) - kept_expected)
            self.assertTrue(all("raising" in d["reason"] for d in dropped))

    def test_manual_exclusion_must_match_review_hash(self):
        from python_fixer.data.build import main
        import hashlib
        # Self-contained: does not depend on ignored runtime evidence being present.
        review = Path(self.tmp.name) / "review.json"
        review.write_text('{"verdict": "REJECT"}')
        config = json.loads(Path("configs/data-v4.json").read_text())
        config["manual_exclusions"][0]["review"] = str(review)
        for digest_value, expected_exit in (("0" * 64, 3), (hashlib.sha256(review.read_bytes()).hexdigest(), 0)):
            config["manual_exclusions"][0]["review_sha256"] = digest_value
            path = Path(self.tmp.name) / "config.json"
            path.write_text(json.dumps(config))
            with self.subTest(expected_exit=expected_exit):
                self.assertEqual(main(["--config", str(path), "--plan-only"]), expected_exit)

    def test_spec_stated_waiver_is_hash_bound(self):
        import hashlib
        from python_fixer.data.audit import load_waivers
        row = next(r for r in self.rows if r["source_function_id"] == "fixture-fibonacci-v2")
        self.assertFalse(mechanical_checks(row)["evaluation_view_excludes_test_records"])
        review = Path(self.tmp.name) / "waiver-review.json"
        review.write_text('{"decision": "ACCEPTABLE"}')
        waiver = {"test_suite_sha256": row["test_suite_sha256"], "test_ids": ["1", "2"],
                  "check": "evaluation_view_excludes_test_records", "reason": "spec-stated convention",
                  "review": str(review), "review_sha256": hashlib.sha256(review.read_bytes()).hexdigest()}
        path = Path(self.tmp.name) / "waivers.json"
        path.write_text(json.dumps([waiver]))
        checks = mechanical_checks(row, load_waivers(path))
        self.assertTrue(checks["evaluation_view_excludes_test_records"])
        self.assertTrue(checks["at_least_four_tests"])  # 7 tests, 2 spec-stated
        other = dict(waiver, test_suite_sha256="0" * 64)
        self.assertFalse(mechanical_checks(row, [other])["evaluation_view_excludes_test_records"])
        path.write_text(json.dumps([dict(waiver, review_sha256="0" * 64)]))
        with self.assertRaises(ValueError):
            load_waivers(path)


if __name__ == "__main__":
    unittest.main()
