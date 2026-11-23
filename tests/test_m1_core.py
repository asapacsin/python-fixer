import copy
import json
from pathlib import Path
import tempfile
import unittest
import uuid

from python_fixer.common import digest, strict_loads, text_digest
from python_fixer.data.build import fixture_seeds, plan_rows
from python_fixer.data.grouping import clone_shape, group_families, split_for_family
from python_fixer.data.models import validate_problem, model_view, load_problems
from python_fixer.data.mutations import mutate
from python_fixer.rewards.behavior import InvalidOutcome, behavior_reward
from python_fixer.sandbox.gateway import Gateway, exact_equal, verify_problem
from python_fixer.sandbox.policy import Policy
from python_fixer.sandbox.docker import DockerSandbox

ROOT = Path(__file__).resolve().parents[1]
IMAGE = "python@sha256:" + "a"*64


class MetadataTests(unittest.TestCase):
    def test_deterministic_build_and_order(self):
        seeds = fixture_seeds()
        self.assertEqual(plan_rows(seeds), plan_rows(list(reversed(seeds))))
        self.assertEqual(len(plan_rows(seeds)), 16)

    def test_renamed_clone_is_grouped(self):
        seed = fixture_seeds()[0]
        clone = copy.deepcopy(seed)
        clone["source_function_id"] = "renamed"
        clone["source_family_id"] = "unrelated-claim"
        clone["reference_code"] = seed["reference_code"].replace("maximum", "choose")
        groups = group_families([seed, clone])
        self.assertEqual(groups[seed["source_function_id"]], groups["renamed"])

    def test_split_is_family_owned(self):
        rows = plan_rows(fixture_seeds(), mutations_per_train_function=2)
        groups = {}
        for row in rows:
            groups.setdefault(row["source_family_id"], set()).add(row["split"])
        self.assertTrue(all(len(splits)==1 for splits in groups.values()))

    def test_unknown_license_and_duplicate_assertion_rejected(self):
        seeds = fixture_seeds()
        seeds[0]["license"] = "Unknown"
        with self.assertRaisesRegex(ValueError, "license"):
            plan_rows(seeds)
        seeds = fixture_seeds()
        seeds[0]["tests"][1] = copy.deepcopy(seeds[0]["tests"][0])
        with self.assertRaisesRegex(ValueError, "duplicate test input"):
            plan_rows(seeds)

    def test_hash_and_baseline_validation(self):
        row = plan_rows(fixture_seeds())[0]
        row["tests"][0]["expected"] = "tampered"
        with self.assertRaisesRegex(ValueError, "hash"):
            validate_problem(row)

    def test_host_metadata_mutation_is_deterministic(self):
        source = fixture_seeds()[0]["reference_code"]
        self.assertEqual(mutate(source), mutate(source))
        self.assertTrue(mutate(source))

    def test_strict_json(self):
        for bad in ('{"a":1,"a":2}', '{"a":NaN}'):
            with self.assertRaises(ValueError):
                strict_loads(bad)

    def test_no_boolean_numeric_coercion(self):
        self.assertFalse(exact_equal(True, 1))
        self.assertFalse(exact_equal(1, 1.0))
        self.assertTrue(exact_equal({"a":1,"b":2}, {"b":2,"a":1}))


class RewardTests(unittest.TestCase):
    def test_production_reward_matches_independent_acceptance_table(self):
        cases = json.loads((ROOT/"verification/reward-cases.json").read_text())["cases"]
        for case in cases:
            with self.subTest(case=case["name"]):
                if case.get("expected_error"):
                    with self.assertRaises(InvalidOutcome):
                        behavior_reward(case["input"])
                else:
                    self.assertEqual(behavior_reward(case["input"]), case["expected_reward"])

    def test_penalty_cannot_bypass_binding_check(self):
        with self.assertRaisesRegex(InvalidOutcome, "binding"):
            behavior_reward({"status":"TIMEOUT","candidate_sha256":"wrong"}, expected_bindings={"candidate_sha256":"expected"})


class FakeSandbox:
    """Protocol-only test double, never a claim of containment or execution."""
    policy = Policy(IMAGE)
    calls = 0

    def invoke(self, source, signature, test, **kwargs):
        self.calls += 1
        assert "expected" not in test
        value = max(test["args"]) if source == self.reference else min(test["args"])
        return {"status":"OK","compiled":True,"returned":True,"value":value,"cpu_ms":1,"peak_memory_bytes":100}


class GatewayTests(unittest.TestCase):
    def setUp(self):
        self.sandbox = FakeSandbox()
        self.row = next(r for r in plan_rows(fixture_seeds()) if r["source_function_id"] == "fixture-maximum-v1")
        self.sandbox.reference = self.row["reference_code"]
        self.verified, self.evidence = verify_problem(self.sandbox, self.row)
        self.verified["split"] = "train"  # Explicit test fixture, not a dataset export.

    def request(self, source):
        return {"schema_version":1,"request_id":str(uuid.uuid4()),"run_id":"unit-test",
                "problem_id":self.row["problem_id"],"candidate_sha256":text_digest(source),
                "candidate_source":source,"suite_role":"train_reward","limits_profile":"python-pure-v1"}

    def test_verification_partition_and_evidence(self):
        self.assertEqual(self.verified["verification"]["status"], "VERIFIED")
        self.assertEqual(self.verified["verification"]["evidence_sha256"], digest(self.evidence))
        self.assertEqual(len(self.evidence["runs"]), 3)

    def test_projection_drops_private_fields(self):
        for role in ("sft", "rl", "evaluation"):
            view = model_view(self.verified, role)
            serialized = json.dumps(view)
            for key in ("tests", "verification", "source_uri", "reference_code", "test_suite_sha256"):
                self.assertNotIn('"'+key+'"', serialized)
            if role != "sft":
                self.assertNotIn(self.verified["reference_code"], serialized)

    def test_heldout_cannot_become_reward_or_sft(self):
        self.verified["split"] = "test"
        for role in ("sft", "rl"):
            with self.assertRaises(ValueError):
                model_view(self.verified, role)
        with self.assertRaises(ValueError):
            Gateway(self.sandbox, [self.verified])

    def test_full_reward_cache_and_request_binding(self):
        gateway = Gateway(self.sandbox, [self.verified])
        request = self.request(self.row["reference_code"])
        result = gateway.handle(request)
        calls = self.sandbox.calls
        self.assertEqual(behavior_reward(result), 1)
        request["request_id"] = str(uuid.uuid4())
        repeated = gateway.handle(request)
        self.assertEqual(self.sandbox.calls, calls)
        self.assertEqual(repeated["request_id"], request["request_id"])

    def test_hash_unknown_problem_and_role_rejected(self):
        gateway = Gateway(self.sandbox, [self.verified])
        for field, value in (("candidate_sha256","bad"),("problem_id","heldout"),("suite_role","heldout_evaluation")):
            request = self.request(self.row["reference_code"])
            request[field] = value
            with self.assertRaises(ValueError):
                gateway.handle(request)

    def test_resource_command_has_no_mounts_or_fallback(self):
        command = DockerSandbox(Policy(IMAGE)).command("pf-test", 2)
        for flag in ("--runtime=runsc", "--network=none", "--read-only", "--user=65532:65532", "--cap-drop=ALL"):
            self.assertIn(flag, command)
        self.assertNotIn("--privileged", command)
        self.assertNotIn("-v", command)
        self.assertNotIn("--mount", command)
        with self.assertRaises(ValueError):
            Policy("python:latest")

    def test_candidate_and_runtime_process_limits_are_distinct_and_bounded(self):
        command = DockerSandbox(Policy(IMAGE)).command("pf-test", 2)
        self.assertIn("--ulimit=nproc=32:32", command)  # candidate cap inside gVisor
        self.assertIn("--pids-limit=128", command)      # whole-sandbox host cgroup bound
        for pids, runtime in ((32, 32), (32, 257), (33, 128)):
            with self.subTest(pids=pids, runtime=runtime), self.assertRaises(ValueError):
                Policy(IMAGE, pids=pids, runtime_pids=runtime)


if __name__ == "__main__":
    unittest.main()
