import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from python_fixer.common import digest, write_json
from python_fixer.data.artifacts import load_verified_partition
from python_fixer.data.build import plan_rows, fixture_seeds, write_dataset
from python_fixer.sandbox.docker import DockerSandbox
from python_fixer.sandbox.policy import Policy, SandboxUnavailable
from python_fixer.sandbox.gateway import verify_problem
from test_m1_core import FakeSandbox


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.destination=Path(self.tmp.name)/"dataset"
        row=next(r for r in plan_rows(fixture_seeds()) if r["source_function_id"]=="fixture-maximum-v1")
        # Test-only registry with a known train split, before evidence binding.
        row["split"]="train"
        sandbox=FakeSandbox()
        sandbox.reference=row["reference_code"]
        self.row,self.evidence=verify_problem(sandbox,row)
        self.manifest=write_dataset(self.destination,[self.row],{row["problem_id"]:self.evidence},[],{"scope":"protocol-double"})

    def test_frozen_partition_and_repeated_evidence_load(self):
        rows,manifest=load_verified_partition(self.destination,"train",expected_manifest_sha256=digest(self.manifest))
        self.assertEqual(rows,[self.row])

    def test_wrong_operator_manifest_pin_fails(self):
        with self.assertRaisesRegex(ValueError,"operator-pinned"):
            load_verified_partition(self.destination,"train",expected_manifest_sha256="0"*64)

    def test_tampered_baseline_and_rehashed_row_still_rejects_old_evidence(self):
        changed=copy.deepcopy(self.row)
        v=changed["verification"]
        v["baseline_passed_ids"],v["baseline_failed_ids"]=v["baseline_failed_ids"],v["baseline_passed_ids"]
        self.manifest["problem_hashes"][changed["problem_id"]]=digest(changed)
        from python_fixer.common import canonical_bytes
        import hashlib
        payload=canonical_bytes(changed)+b"\n"
        (self.destination/"train.jsonl").write_bytes(payload)
        self.manifest["partition_sha256"]["train"]=hashlib.sha256(payload).hexdigest()
        write_json(self.destination/"manifest.json",self.manifest)
        with self.assertRaisesRegex(ValueError,"baseline"):
            load_verified_partition(self.destination,"train")

    def test_modified_evidence_rejected(self):
        path=self.destination/"evidence"/(self.row["problem_id"]+".json")
        path.write_text("{}")
        with self.assertRaisesRegex(ValueError,"evidence hash"):
            load_verified_partition(self.destination,"train")

    def test_immutable_build_does_not_overwrite(self):
        with self.assertRaisesRegex(ValueError,"overwrite"):
            write_dataset(self.destination,[],{},[],{})

    def test_train_export_contains_no_heldout_test_bundle(self):
        from python_fixer.data.export import export_train
        target=Path(self.tmp.name)/"worker-registry"
        result=export_train(self.destination,target,digest(self.manifest))
        self.assertFalse(result["heldout_tests_exported"])
        self.assertFalse((target/"test.jsonl").exists())
        self.assertFalse((target/"validation.jsonl").exists())
        loaded,_=load_verified_partition(target,"train",expected_manifest_sha256=digest(self.manifest))
        self.assertEqual(loaded,[self.row])
        public=json.loads((target/"rl-view.jsonl").read_text())
        self.assertEqual(set(public),{"prompt","problem_id"})


class RuntimePolicyTests(unittest.TestCase):
    def test_known_non_enforcing_runtime_blocks_before_execution(self):
        sandbox=DockerSandbox(Policy("python@sha256:"+"a"*64))
        daemon=SimpleNamespace(returncode=0,stdout=json.dumps({"runsc":{"runtimeArgs":["--ignore-cgroups"]}}))
        with patch("python_fixer.sandbox.docker.subprocess.run",return_value=daemon):
            with self.assertRaisesRegex(SandboxUnavailable,"ignores cgroups"):
                sandbox.invoke("never executed","f(x)",{"args":[1],"kwargs":{}})
