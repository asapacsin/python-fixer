"""Score paired greedy validation completions inside the isolated gVisor worker."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from python_fixer.evaluation.extraction import extract_module
from python_fixer.sandbox.docker import DockerSandbox
from python_fixer.sandbox.gateway import execute_suite
from python_fixer.sandbox.policy import Policy


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--sft", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists():
        raise SystemExit(f"output exists: {output}")
    data_path = Path(args.data)
    rows = read_jsonl(data_path)
    if not rows or any(row.get("split") != "validation" for row in rows):
        raise SystemExit("input must contain validation rows only")
    by_id = {row["problem_id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise SystemExit("duplicate validation problem_id")
    arms = {"baseline": read_jsonl(Path(args.base)), "sft": read_jsonl(Path(args.sft))}
    expected = set(by_id)
    for name, records in arms.items():
        ids = [item["problem_id"] for item in records]
        if len(ids) != len(expected) or set(ids) != expected or len(set(ids)) != len(ids):
            raise SystemExit(f"{name} candidates do not exactly match the validation grid")
        for item in records:
            raw = item["completion"].encode("utf-8")
            if hashlib.sha256(raw).hexdigest() != item["candidate_sha256"]:
                raise SystemExit(f"{name} candidate hash mismatch: {item['problem_id']}")

    sandbox = DockerSandbox(Policy.load(Path(args.policy)))
    sandbox.available()
    details = []
    for problem_id, row in by_id.items():
        record = {"problem_id": problem_id}
        for arm_name in ("baseline", "sft"):
            candidate = next(x for x in arms[arm_name] if x["problem_id"] == problem_id)
            extracted = extract_module(candidate["completion"])
            if extracted.status != "VALID":
                outcome = {"status": "INVALID_OUTPUT", "solved": False,
                           "reason": extracted.reason, "passed_tests": 0,
                           "total_tests": len(row["tests"])}
            else:
                suite = execute_suite(sandbox, row, extracted.module.decode("utf-8"))
                passed = len(suite.get("passed_ids", []))
                outcome = {
                    "status": suite["status"],
                    "solved": suite["status"] == "OK" and passed == len(row["tests"]),
                    "passed_tests": passed,
                    "total_tests": len(row["tests"]),
                }
            record[arm_name] = {
                **outcome,
                "candidate_sha256": candidate["candidate_sha256"],
                "generated_tokens": candidate["generated_tokens"],
                "generation_ms": candidate["generation_ms"],
            }
        details.append(record)
        print(problem_id, record["baseline"]["solved"], record["sft"]["solved"], flush=True)

    n = len(details)
    base_solved = sum(item["baseline"]["solved"] for item in details)
    sft_solved = sum(item["sft"]["solved"] for item in details)
    improved = sum(not item["baseline"]["solved"] and item["sft"]["solved"] for item in details)
    regressed = sum(item["baseline"]["solved"] and not item["sft"]["solved"] for item in details)
    report = {
        "evidence_kind": "EXPLORATORY_PAIRED_GREEDY_VALIDATION",
        "split": "validation",
        "problems": n,
        "dataset_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "policy_sha256": sandbox.policy.sha256,
        "image_digest": sandbox.policy.image,
        "baseline_solved": base_solved,
        "baseline_pass_at_1": base_solved / n,
        "sft_solved": sft_solved,
        "sft_pass_at_1": sft_solved / n,
        "absolute_gain": (sft_solved - base_solved) / n,
        "improved_problems": improved,
        "regressed_problems": regressed,
        "results": details,
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "results"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
