import tempfile
import unittest
from pathlib import Path

from python_fixer.common import digest
from python_fixer.data.artifacts import load_verified_partition
from python_fixer.data.build import plan_rows, fixture_seeds, write_dataset
from python_fixer.sandbox.gateway import verify_problem
from python_fixer.sandbox.policy import Policy
from scripts.reverify_train_subset import reverify_train_subset
from test_m1_core import FakeSandbox


class ReverifySubsetTests(unittest.TestCase):
    def test_subset_preserves_parent_pin_and_loads_under_new_policy(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            destination = root / "extended"
            row = next(r for r in plan_rows(fixture_seeds()) if r["source_function_id"] == "fixture-maximum-v1")
            row["split"] = "train"
            strict = FakeSandbox()
            strict.reference = row["reference_code"]
            verified, evidence = verify_problem(strict, row)
            parent = write_dataset(source, [verified], {row["problem_id"]: evidence}, [], {})
            parent_pin = digest(parent)

            class ExtendedFake(FakeSandbox):
                policy = Policy.load(Path("configs/sandbox-validation-extended.json"))

            extended = ExtendedFake()
            extended.reference = row["reference_code"]
            extended.available = lambda: None
            result = reverify_train_subset(source, destination, parent_pin, Path("configs/sandbox-validation-extended.json"),
                                           {row["problem_id"]}, sandbox=extended)
            rows, manifest = load_verified_partition(destination, "train",
                                                       expected_manifest_sha256=result["manifest_sha256"])

            self.assertEqual(len(rows), 1)
            self.assertEqual(manifest["parent_manifest_sha256"], parent_pin)
            self.assertEqual(manifest["split_namespace"], parent["split_namespace"])
            self.assertEqual(rows[0]["verification"]["policy_sha256"], extended.policy.sha256)
            self.assertEqual(load_verified_partition(source, "train", expected_manifest_sha256=parent_pin)[0], [verified])
            with self.assertRaisesRegex(ValueError, "overwrite"):
                reverify_train_subset(source, destination, parent_pin, Path("configs/sandbox-validation-extended.json"),
                                      {row["problem_id"]}, sandbox=extended)

            nested = source / "nested-output"
            with self.assertRaisesRegex(ValueError, "outside"):
                reverify_train_subset(source, nested, parent_pin,
                                      Path("configs/sandbox-validation-extended.json"),
                                      {row["problem_id"]}, sandbox=extended)

            failed = ExtendedFake()
            failed.available = lambda: (_ for _ in ()).throw(RuntimeError("sandbox unavailable"))
            failed_destination = root / "failed-output"
            with self.assertRaisesRegex(RuntimeError, "unavailable"):
                reverify_train_subset(source, failed_destination, parent_pin,
                                      Path("configs/sandbox-validation-extended.json"),
                                      {row["problem_id"]}, sandbox=failed)
            self.assertFalse(failed_destination.exists())
            self.assertEqual(list(root.glob("failed-output.staging-*")), [])


if __name__ == "__main__":
    unittest.main()
