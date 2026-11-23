"""Trusted test ownership, comparisons, hash bindings and isolated execution."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
import copy
import math
import threading
import time
import uuid
from python_fixer.common import canonical_bytes, digest, text_digest
from python_fixer.data.models import validate_problem, mark_verified
from .policy import SandboxUnavailable


def exact_equal(actual, expected) -> bool:
    # No Python coercion of True==1; dictionary order does not matter.
    try:
        return canonical_bytes(actual) == canonical_bytes(expected)
    except (ValueError, TypeError):
        return False


def execute_suite(sandbox, row: dict, source: str) -> dict:
    started = time.monotonic()
    cpu_ms = peak = 0
    passed = []
    observed = []  # trusted-side behaviour signature; never sent to the candidate
    compiled = False
    if hasattr(sandbox,"invoke_suite"):
        results = sandbox.invoke_suite(source,row["signature"],
                    [{"args":t["args"],"kwargs":t["kwargs"]} for t in row["tests"]])
    else:
        results = [sandbox.invoke(source,row["signature"],{"args":t["args"],"kwargs":t["kwargs"]}) for t in row["tests"]]
    for test, outcome in zip(row["tests"],results):
        remaining = sandbox.policy.wall_seconds - (time.monotonic()-started)
        cpu_left = sandbox.policy.cpu_seconds - cpu_ms/1000
        if remaining <= 0 or cpu_left <= 0:
            return {"status": "TIMEOUT", "compiled": compiled, "passed_ids": [],
                    "cpu_ms": cpu_ms, "peak_memory_bytes": peak}
        # Pass inputs only, never expected values or other tests.
        compiled |= outcome.get("compiled") is True
        cpu_ms += outcome.get("cpu_ms", 0)
        peak = max(peak, outcome.get("peak_memory_bytes", 0))
        if outcome["status"] != "OK":
            return {"status": outcome["status"], "compiled": compiled, "passed_ids": [],
                    "cpu_ms": cpu_ms, "peak_memory_bytes": peak}
        if cpu_ms > sandbox.policy.cpu_seconds*1000:
            return {"status": "TIMEOUT", "compiled": compiled, "passed_ids": [],
                    "cpu_ms": cpu_ms, "peak_memory_bytes": peak}
        observed.append([test["test_id"], outcome.get("returned") is True,
                         outcome.get("value") if outcome.get("returned") is True else None])
        if outcome.get("returned") is True and exact_equal(outcome.get("value"), test["expected"]):
            passed.append(test["test_id"])
    if len(results) != len(row["tests"]):
        return {"status":"INTERNAL_ERROR","compiled":compiled,"passed_ids":[],
                "cpu_ms":cpu_ms,"peak_memory_bytes":peak}
    return {"status": "OK", "compiled": compiled, "passed_ids": passed,
            "outcome_sha256": digest(observed), "cpu_ms": cpu_ms, "peak_memory_bytes": peak}


def verify_problem(sandbox, row: dict, repeats: int = 3) -> tuple[dict, dict]:
    validate_problem(row)
    if repeats < 3:
        raise ValueError("verification requires three repeated reference/buggy runs")
    observed = []
    for _ in range(repeats):
        reference = execute_suite(sandbox, row, row["reference_code"])
        buggy = execute_suite(sandbox, row, row["buggy_code"])
        if reference["status"] == "INTERNAL_ERROR" or buggy["status"] == "INTERNAL_ERROR":
            raise SandboxUnavailable("worker infrastructure failed during verification")
        all_ids = {t["test_id"] for t in row["tests"]}
        passed = set(buggy["passed_ids"])
        if reference["status"] != "OK" or set(reference["passed_ids"]) != all_ids:
            raise ValueError("reference did not pass every test")
        if buggy["status"] != "OK" or not buggy["compiled"] or not passed or passed == all_ids:
            raise ValueError("bug must compile, fail semantic tests and preserve regression tests")
        observed.append({"reference": reference, "buggy": buggy})
    baseline = observed[0]["buggy"]["passed_ids"]
    if any(run["buggy"]["passed_ids"] != baseline for run in observed):
        raise ValueError("flaky buggy outcome")
    if len({run["buggy"].get("outcome_sha256") for run in observed}) != 1:
        raise ValueError("flaky buggy outcome")
    evidence = {"image_digest": sandbox.policy.image, "policy_sha256": sandbox.policy.sha256,
                "problem_sha256": digest(row), "runs": observed}
    verified = mark_verified(row, {"status": "VERIFIED", "baseline_passed_ids": sorted(baseline),
        "baseline_failed_ids": sorted(all_ids-set(baseline)), "image_digest": sandbox.policy.image,
        "policy_sha256": sandbox.policy.sha256, "evidence_sha256": digest(evidence)})
    return verified, evidence


class Gateway:
    def __init__(self, sandbox, rows: list[dict], *, suite_role: str = "train_reward", concurrency: int = 4):
        if suite_role not in {"train_reward", "heldout_evaluation"}:
            raise ValueError("unrecognized gateway role")
        self.sandbox = sandbox
        self.suite_role = suite_role
        self.rows = {}
        for row in rows:
            validate_problem(row)
            wanted_split = "train" if suite_role == "train_reward" else "test"
            if row["split"] != wanted_split or row["verification"]["status"] != "VERIFIED":
                raise ValueError("registry contains unverified or wrong-split data")
            if row["verification"]["image_digest"] != sandbox.policy.image or row["verification"]["policy_sha256"] != sandbox.policy.sha256:
                raise ValueError("verification does not match current worker identity")
            if row["problem_id"] in self.rows:
                raise ValueError("duplicate registry problem")
            self.rows[row["problem_id"]] = copy.deepcopy(row)
        if not 1 <= concurrency <= 4:
            raise ValueError("concurrency exceeds policy")
        self.semaphore = threading.BoundedSemaphore(concurrency)
        self.cache = {}
        self.lock = threading.Lock()

    def handle(self, request: dict) -> dict:
        fields = {"schema_version", "request_id", "run_id", "problem_id", "candidate_sha256",
                  "candidate_source", "suite_role", "limits_profile"}
        if set(request) != fields or request["schema_version"] != 1:
            raise ValueError("invalid request schema")
        uuid.UUID(request["request_id"])
        if not isinstance(request["run_id"], str) or not request["run_id"] or len(request["run_id"]) > 100:
            raise ValueError("invalid run ID")
        if len(canonical_bytes(request)) > 131072 or not isinstance(request["candidate_source"], str):
            raise ValueError("oversized request")
        source = request["candidate_source"]
        if len(source.encode()) > 65536 or request["candidate_sha256"] != text_digest(source):
            raise ValueError("candidate digest/size mismatch")
        if request["suite_role"] != self.suite_role or request["limits_profile"] != "python-pure-v1":
            raise ValueError("request cannot change permissions or limits")
        row = self.rows.get(request["problem_id"])
        if row is None:
            raise ValueError("problem is outside the authorized registry")
        key = digest({"candidate": text_digest(source), "suite": row["test_suite_sha256"],
                      "baseline": row["verification"], "policy": self.sandbox.policy.sha256,
                      "image": self.sandbox.policy.image, "signature": row["signature"]})
        with self.lock:
            cached = copy.deepcopy(self.cache.get(key))
        if cached is not None:
            return {**cached, "request_id": request["request_id"]}
        if not self.semaphore.acquire(timeout=10):
            raise SandboxUnavailable("bounded worker queue is full")
        began = time.monotonic()
        try:
            result = None
            for attempt in range(3):
                result = execute_suite(self.sandbox, row, source)
                if result["status"] != "INTERNAL_ERROR":
                    break
                if attempt < 2:
                    time.sleep(0.05 * 2**attempt)
            failed = set(row["verification"]["baseline_failed_ids"])
            passed = set(row["verification"]["baseline_passed_ids"])
            now = set(result["passed_ids"])
            good = result["status"] == "OK"
            response = {"schema_version": 1, "request_id": request["request_id"],
                "candidate_sha256": text_digest(source), "suite_sha256": row["test_suite_sha256"],
                "image_digest": self.sandbox.policy.image, "policy_sha256": self.sandbox.policy.sha256,
                "status": result["status"], "compiled": result["compiled"],
                "fixed_failed": len(now & failed) if good else 0, "total_failed": len(failed),
                "preserved_passed": len(now & passed) if good else 0, "total_passed": len(passed),
                "all_passed": good and now == failed | passed,
                "wall_ms": int((time.monotonic()-began)*1000),
                "cpu_ms": result["cpu_ms"], "peak_memory_bytes": result["peak_memory_bytes"],
                "truncated_output": False, "runner_version": "python-pure-v1", "diagnostics": ""}
            if good:
                with self.lock:
                    if len(self.cache) >= 4096:
                        self.cache.clear()
                    self.cache[key] = copy.deepcopy(response)
            return response
        finally:
            self.semaphore.release()
