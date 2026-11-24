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

    def test_v2_source_extends_v1_without_changing_it(self):
        v1, v2 = fixture_seeds(), fixture_seeds("authored-fixtures-v2")
        self.assertEqual(v2[:len(v1)], v1)
        self.assertGreaterEqual(len(v2), 50)
        ids = [s["source_function_id"] for s in v2]
        self.assertEqual(len(ids), len(set(ids)))
        rows = plan_rows(v2, source="authored-fixtures-v2", mutations_per_train_function=2)
        self.assertEqual(rows, plan_rows(list(reversed(v2)), source="authored-fixtures-v2",
                                         mutations_per_train_function=2))
        splits = {}
        for row in rows:
            validate_problem(row)
            self.assertEqual(row["provenance"]["revision"], "fixtures-v2")
            expected = "fixtures.json" if row["source_function_id"].endswith("-v1") else "fixtures_v2.json"
            self.assertTrue(row["provenance"]["path"].endswith(expected))
            splits.setdefault(row["source_family_id"], set()).add(row["split"])
            if row["split"] != "train":
                self.assertEqual(row["provenance"]["mutation_operator"], "authored-bug-v1")
        self.assertTrue(all(len(s) == 1 for s in splits.values()))
        by_function = {}
        for row in rows:
            by_function.setdefault(row["source_function_id"], []).append(row)
            if row["provenance"]["mutation_operator"] != "authored-bug-v1":
                kind = row["provenance"]["mutation_operator"].split("-")[2]
                self.assertEqual(row["bug_category"], {"ops": "comparison", "op": "operator"}[kind])
        for function, variants in by_function.items():
            buggy = [r["buggy_code"] for r in variants]
            self.assertEqual(len(buggy), len(set(buggy)), function)
            self.assertLessEqual(len(variants), 3)
        self.assertEqual({r["split"] for r in rows}, {"train", "validation", "test"})
        with self.assertRaises(ValueError):
            fixture_seeds("unknown-source")

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

    def test_mutation_is_one_token_and_preserves_formatting(self):
        source = ("def f(xs, n):\n    lo, hi = 0, len(xs) - 1\n    s = 'é' + 'x'\n"
                  "    return lo <= hi and n > (1 + 2) * 3\n")
        variants = dict(mutate(source))
        self.assertTrue(variants)
        for name, text in variants.items():
            changed = [(a, b) for a, b in zip(source.splitlines(), text.splitlines()) if a != b]
            with self.subTest(name=name):
                self.assertEqual(len(source.splitlines()), len(text.splitlines()))
                self.assertEqual(len(changed), 1)
                self.assertNotIn("(0, len(xs) - 1)", text)
        self.assertIn("    lo, hi = 0, len(xs) + 1\n", "".join(variants.values()))
        self.assertTrue(any("lo < hi" in t for t in variants.values()))
        self.assertTrue(any("n >= (1 + 2) * 3" in t for t in variants.values()))
        self.assertTrue(any("(1 - 2) * 3" in t for t in variants.values()))
        self.assertTrue(any("(1 + 2) + 3" in t for t in variants.values()))
        self.assertTrue(any("'é' - 'x'" in t for t in variants.values()))  # non-ASCII offsets

    def test_mutation_offsets_ignore_non_tokenizer_line_breaks(self):
        source = "def f(a, b):\n    s = 'x\x0cy\u2028z'\n    return a < b\n"
        variants = [t for _, t in mutate(source)]
        self.assertIn(source.replace("a < b", "a <= b"), variants)

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

    def test_unencodable_candidate_value_is_a_scored_failure_not_infrastructure(self):
        class Surrogate(FakeSandbox):
            def invoke(self, source, signature, test, **kwargs):
                return {"status":"OK","compiled":True,"returned":True,"value":"\ud800","cpu_ms":1,"peak_memory_bytes":1}
        from python_fixer.sandbox.gateway import execute_suite
        result = execute_suite(Surrogate(), self.row, "def maximum(a, b):\n    return '?'\n")
        self.assertEqual(result["status"], "OK")
        self.assertEqual(result["passed_ids"], [])
        self.assertRegex(result["outcome_sha256"], "^[0-9a-f]{64}$")

    def test_child_rejects_oversized_int_and_maps_signal_deaths(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("pf_child", ROOT / "src/python_fixer/sandbox/child.py")
        child = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(child)  # defines functions only; main() is guarded
        self.assertEqual(child.json_value(10 ** 100), 10 ** 100)
        with self.assertRaises(ValueError):
            child.json_value(10 ** 5000)
        # Behavioural: run the trusted shim's suite branch with a trusted runner that
        # kills itself with SIGSEGV (test code only; no candidate code runs on the host).
        import subprocess, sys as _sys
        request = {"tests": [{"args": [1], "kwargs": {}}], "wall_seconds": 5, "cpu_seconds": 2,
                   "runner_source": "import os, signal\nos.kill(os.getpid(), signal.SIGSEGV)\n",
                   "source": "", "signature": "repair(x)", "imports": [], "memory_mib": 256}
        done = subprocess.run([_sys.executable, "-I", str(ROOT / "src/python_fixer/sandbox/child.py")],
                              input=json.dumps(request), capture_output=True, text=True, timeout=30)
        result = json.loads(done.stdout)["results"][0]
        self.assertEqual((result["status"], result["returned"], result["compiled"]), ("OK", False, True))

    def test_container_crash_exit_is_candidate_failure(self):
        from unittest import mock
        import python_fixer.sandbox.docker as docker_module
        sandbox = DockerSandbox(Policy(IMAGE))
        class Done:
            returncode = 0
            stderr = b""
        def fake_popen(*args, **kwargs):
            class Process:
                returncode = None
                def poll(self):
                    self.returncode = code
                    return code
                def kill(self): pass
                def wait(self, timeout=None): pass
            return Process()
        for code, expected in ((139, "OK"), (134, "OK"), (137, "RESOURCE_LIMIT"), (135, "INTERNAL_ERROR"),
                               (132, "INTERNAL_ERROR"), (1, "INTERNAL_ERROR")):
            with self.subTest(code=code), mock.patch.object(sandbox, "available"), \
                    mock.patch.object(docker_module.subprocess, "run", return_value=Done()), \
                    mock.patch.object(docker_module.subprocess, "Popen", side_effect=fake_popen):
                result = sandbox.invoke("def repair(x):\n    return x\n", "repair(x)", {"args": [1], "kwargs": {}})
                self.assertEqual(result["status"], expected)
                self.assertIsNot(result.get("returned"), True)
                if expected == "OK":
                    self.assertIs(result["compiled"], True)

    def test_crashing_candidate_scores_zero_end_to_end(self):
        # A crash on every test must yield a scalar reward of 0, like a wrong answer.
        class Crash(FakeSandbox):
            def invoke(self, source, signature, test, **kwargs):
                if source == self.reference:
                    return super().invoke(source, signature, test)
                return {"status":"OK","compiled":True,"returned":False,"cpu_ms":1,"peak_memory_bytes":1}
        sandbox = Crash()
        sandbox.reference = self.row["reference_code"]
        gateway = Gateway(sandbox, [self.verified])
        response = gateway.handle(self.request("def maximum(a, b):\n    return maximum(a, b)\n"))
        self.assertEqual(response["status"], "OK")
        self.assertEqual(behavior_reward(response), 0.0)

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
        for pids, runtime in ((32, 32), (32, 257), (32, 129), (32, 256), (33, 128)):
            with self.subTest(pids=pids, runtime=runtime), self.assertRaises(ValueError):
                Policy(IMAGE, pids=pids, runtime_pids=runtime)


if __name__ == "__main__":
    unittest.main()
