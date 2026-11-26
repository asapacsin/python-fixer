"""Score one greedy baseline and multiple SFT seeds on the same validation grid."""
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


def validate_candidates(name: str, records: list[dict], expected: set[str]) -> dict[str, dict]:
    ids = [item["problem_id"] for item in records]
    if len(ids) != len(expected) or set(ids) != expected or len(set(ids)) != len(ids):
        raise SystemExit(f"{name} candidates do not exactly match the validation grid")
    result = {}
    for item in records:
        raw = item["completion"].encode("utf-8")
        if hashlib.sha256(raw).hexdigest() != item["candidate_sha256"]:
            raise SystemExit(f"{name} candidate hash mismatch: {item['problem_id']}")
        result[item["problem_id"]] = item
    return result


def score_one(sandbox: DockerSandbox, row: dict, candidate: dict) -> dict:
    extracted = extract_module(candidate["completion"])
    if extracted.status != "VALID":
        outcome = {"status": "INVALID_OUTPUT", "solved": False,
                   "reason": extracted.reason, "passed_tests": 0,
                   "total_tests": len(row["tests"])}
    else:
        suite = execute_suite(sandbox, row, extracted.module.decode("utf-8"))
        passed = len(suite.get("passed_ids", []))
        outcome = {"status": suite["status"],
                   "solved": suite["status"] == "OK" and passed == len(row["tests"]),
                   "passed_tests": passed, "total_tests": len(row["tests"])}
    return {**outcome, "candidate_sha256": candidate["candidate_sha256"],
            "generated_tokens": candidate["generated_tokens"],
            "generation_ms": candidate["generation_ms"]}


def summarize(details: list[dict], arm: str) -> dict:
    solved = sum(row[arm]["solved"] for row in details)
    return {"solved": solved, "problems": len(details),
            "pass_at_1": solved / len(details),
            "timeouts": sum(row[arm]["status"] == "TIMEOUT" for row in details),
            "invalid_outputs": sum(row[arm]["status"] == "INVALID_OUTPUT" for row in details)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--sft", action="append", required=True, metavar="SEED=JSONL")
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
    expected = set(by_id)
    arms = {"baseline": validate_candidates("baseline", read_jsonl(Path(args.base)), expected)}
    seeds = {}
    for spec in args.sft:
        seed, sep, path = spec.partition("=")
        if not sep or not seed.isdigit() or seed in seeds:
            raise SystemExit("each --sft must be a unique SEED=JSONL")
        seeds[seed] = validate_candidates(f"seed{seed}", read_jsonl(Path(path)), expected)

    sandbox = DockerSandbox(Policy.load(Path(args.policy)))
    sandbox.available()
    details = []
    for problem_id, row in by_id.items():
        record = {"problem_id": problem_id,
                  "baseline": score_one(sandbox, row, arms["baseline"][problem_id]),
                  "seeds": {seed: score_one(sandbox, row, records[problem_id])
                            for seed, records in sorted(seeds.items(), key=lambda pair: int(pair[0]))}}
        details.append(record)
        print(problem_id, record["baseline"]["solved"],
              {seed: outcome["solved"] for seed, outcome in record["seeds"].items()}, flush=True)

    baseline_summary = summarize(details, "baseline")
    seed_summaries = {}
    for seed in seeds:
        wins = sum(not row["baseline"]["solved"] and row["seeds"][seed]["solved"] for row in details)
        losses = sum(row["baseline"]["solved"] and not row["seeds"][seed]["solved"] for row in details)
        scored = summarize([{**row, "sft": row["seeds"][seed]} for row in details], "sft")
        seed_summaries[seed] = {**scored, "wins": wins, "regressions": losses,
                                "baseline_solved": baseline_summary["solved"],
                                "absolute_gain": (scored["solved"] - baseline_summary["solved"]) / len(details)}
    report = {"evidence_kind": "EXPLORATORY_PAIRED_GREEDY_VALIDATION",
              "split": "validation", "problems": len(details),
              "dataset_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
              "policy_sha256": sandbox.policy.sha256, "image_digest": sandbox.policy.image,
              "wall_timeout_seconds": sandbox.policy.wall_seconds,
              "cpu_timeout_seconds": sandbox.policy.cpu_seconds,
              "baseline": baseline_summary, "seeds": seed_summaries, "results": details}
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "results"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
