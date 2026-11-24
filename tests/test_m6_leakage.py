"""M6 audits on the frozen v7 data, plus M7 artifact/link checks. No torch needed."""
import importlib.util
import json
import unittest
from pathlib import Path

from python_fixer.data.models import model_view
from python_fixer.evaluation import leakage

ROOT = Path(__file__).resolve().parents[1]
V7 = ROOT / "verification/evidence/m1-v7/m1-fixtures-v7"


def load(name):
    return [json.loads(l) for l in (V7 / name).read_text().splitlines() if l]


class LeakageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train, cls.val, cls.test = load("train.jsonl"), load("validation.jsonl"), load("test.jsonl")
        views = [model_view(r, "sft") for r in cls.train]
        cls.inputs = [json.dumps(v["prompt"]) + v["completion"] for v in views]

    def test_frozen_v7_training_views_are_clean(self):
        self.assertEqual(leakage.data_access_audit(self.train, self.val + self.test, self.inputs), [])

    def test_detects_injected_heldout_reference_and_identity(self):
        leaked = self.inputs + [self.test[0]["reference_code"]]
        self.assertTrue(leakage.data_access_audit(self.train, self.test, leaked))
        self.assertTrue(leakage.data_access_audit(self.train + [self.test[0]], self.test, self.inputs))

    def test_ngram_overlap(self):
        ref = self.test[0]["reference_code"]
        self.assertEqual(leakage.ngram_overlap(ref, [ref], n=5), 1.0)
        self.assertEqual(leakage.ngram_overlap(ref, ["unrelated text " * 10], n=5), 0.0)
        report = leakage.contamination_report(self.test, [ref], n=5)
        self.assertIn(self.test[0]["problem_id"], report["flagged"])


class ArtifactCheckTests(unittest.TestCase):
    def test_committed_checksums_and_doc_links(self):
        spec = importlib.util.spec_from_file_location("check_artifacts", ROOT / "scripts/check_artifacts.py")
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        self.assertEqual(mod.checksum_issues(), [])
        self.assertEqual(mod.link_issues(), [])


if __name__ == "__main__":
    unittest.main()
