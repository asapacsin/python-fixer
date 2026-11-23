"""Spec 02 pre-scaling audit: attrition, mutation survival and a 50-case review sample.

Reads a published dataset directory (manifest + problems.jsonl). It never executes
or compiles fixture code; differences are computed on source text only. The
worksheet's human verdict fields start PENDING: an author's mechanical pass is not
the manual/independent review that spec 02 and the M1 contract require.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import random
import sys
from collections import Counter
from pathlib import Path

import json
from python_fixer.common import canonical_bytes, digest, strict_loads, write_json
from .models import model_view, validate_problem

AUTHORED = "authored-bug-v1"
SAMPLE_SIZE = 50


def load_dataset(directory: Path, manifest_sha256: str | None = None) -> tuple[dict, list[dict]]:
    raw_manifest = (directory / "manifest.json").read_bytes()
    manifest = strict_loads(raw_manifest.decode())
    if manifest_sha256 is not None and digest(manifest) != manifest_sha256:
        raise ValueError("manifest digest does not match the pin")
    payload = (directory / "problems.jsonl").read_bytes()
    if hashlib.sha256(payload).hexdigest() != manifest["dataset_sha256"]:
        raise ValueError("problems.jsonl does not match manifest dataset_sha256")
    rows = [strict_loads(line) for line in payload.decode().splitlines() if line]
    if len(rows) != manifest["row_count"]:
        raise ValueError("row count does not match manifest")
    for row in rows:
        validate_problem(row)
        if row["verification"]["status"] != "VERIFIED":
            raise ValueError("audit only covers verified rows")
    return manifest, rows


def changed_lines(reference: str, buggy: str) -> int:
    diff = difflib.ndiff(reference.splitlines(), buggy.splitlines())
    return sum(1 for line in diff if line.startswith("- "))


def mechanical_checks(row: dict) -> dict:
    tests = row["tests"]
    passed = set(row["verification"]["baseline_passed_ids"])
    failed = set(row["verification"]["baseline_failed_ids"])
    inputs = {digest({"args": t["args"], "kwargs": t["kwargs"]}) for t in tests}
    view = model_view(row, "evaluation")
    # Search the raw model-facing strings (not a repr, which escapes newlines).
    text = "\n".join(m["content"] for m in view["prompt"])
    canonical = canonical_bytes
    # Bare expected values ("True", "Fizz") legitimately occur in behavior text or
    # buggy code, so a leak is a whole test record, its canonical JSON, or a
    # non-trivial expected value that also appears together with its arguments.
    def leaked(test):
        expected = canonical(test["expected"]).decode()
        args = canonical(test["args"]).decode()
        return (canonical(test).decode() in text or json.dumps(test) in text
                or (len(expected) > 3 and expected in text and args in text))
    leaked_tests = [t for t in tests if leaked(t)]
    checks = {
        "at_least_four_tests": len(tests) >= 4,
        "distinct_test_inputs": len(inputs) == len(tests),
        "baseline_has_failure": bool(failed),
        "baseline_has_regression_pass": bool(passed),
        "baseline_partition_complete": passed | failed == {t["test_id"] for t in tests} and not passed & failed,
        "mutant_is_single_line_edit": (row["provenance"]["mutation_operator"] == AUTHORED
                                       or changed_lines(row["reference_code"], row["buggy_code"]) == 1),
        "behavior_nonempty": bool(row["behavior"].strip()),
        "bug_differs_from_reference": row["buggy_code"] != row["reference_code"],
        "evaluation_view_only_prompt_and_id": set(view) == {"prompt", "problem_id"},
        # The buggy module is meant to be in the prompt and may contain the reference as
        # a prefix (`value in xs` vs `value in xs[1:]`); search the rest of the text.
        "evaluation_view_excludes_reference": (row["reference_code"].strip()
                                               not in text.replace(row["buggy_code"].strip(), "", 1)
                                               and row["reference_code"].strip() != row["buggy_code"].strip()),
        "evaluation_view_excludes_test_records": not leaked_tests,
        "evaluation_view_excludes_provenance": (row["provenance"]["source_sha256"] not in text
                                                and row["test_suite_sha256"] not in text),
    }
    return checks


def attrition(manifest: dict, rows: list[dict]) -> dict:
    accepted = Counter((r["bug_category"], r["provenance"]["mutation_operator"] == AUTHORED) for r in rows)
    rejected_ids = [r["problem_id"] for r in manifest["rejected"]]
    # Rejections carry their own category/origin (same rules as accepted rows).
    rejected = Counter(f"{r.get('bug_category', 'unrecorded')}|{r.get('origin', 'unrecorded')}"
                       for r in manifest["rejected"])
    survival = {}
    for category in sorted({r["bug_category"] for r in rows if r["provenance"]["mutation_operator"] != AUTHORED}
                           | {r.get("bug_category") for r in manifest["rejected"] if r.get("origin") == "mutant"}):
        kept = sum(1 for r in rows if r["bug_category"] == category and r["provenance"]["mutation_operator"] != AUTHORED)
        lost = sum(1 for r in manifest["rejected"] if r.get("origin") == "mutant" and r.get("bug_category") == category)
        survival[category] = {"accepted": kept, "rejected": lost}
    mutants_accepted = sum(1 for r in rows if r["provenance"]["mutation_operator"] != AUTHORED)
    authored_accepted = len(rows) - mutants_accepted
    reasons = Counter(r["reason"] for r in manifest["rejected"])
    planned = len(rows) + len(rejected_ids)
    return {
        "planned_rows": planned,
        "accepted_rows": len(rows),
        "rejected_rows": len(rejected_ids),
        "acceptance_rate": round(len(rows) / planned, 4) if planned else None,
        "authored_bugs_accepted": authored_accepted,
        "mutants_accepted": mutants_accepted,
        "rejection_reasons": dict(sorted(reasons.items())),
        "rejected_by_category_and_origin": dict(sorted(rejected.items())),
        "mutant_survival_by_category": survival,
        "accepted_by_category": dict(sorted(Counter(r["bug_category"] for r in rows).items())),
        "accepted_by_category_and_origin": {f"{cat}|{'authored' if authored else 'mutant'}": n
                                             for (cat, authored), n in sorted(accepted.items())},
        "note": "Accepted and rejected rows use the same category rule: authored label for authored "
                "bugs, edited construct for mutants.",
    }


def sample(rows: list[dict], seed: int, size: int = SAMPLE_SIZE) -> list[dict]:
    if len(rows) < size:
        raise ValueError(f"need at least {size} accepted cases, found {len(rows)}")
    ordered = sorted(rows, key=lambda r: r["problem_id"])
    return sorted(random.Random(seed).sample(ordered, size), key=lambda r: r["problem_id"])


def build_audit(directory: Path, seed: int, manifest_sha256: str | None = None) -> dict:
    manifest, rows = load_dataset(directory, manifest_sha256)
    chosen = sample(rows, seed)
    cases = []
    for row in chosen:
        checks = mechanical_checks(row)
        cases.append({
            "problem_id": row["problem_id"], "split": row["split"], "bug_category": row["bug_category"],
            "difficulty": row["difficulty"], "origin": "authored" if row["provenance"]["mutation_operator"] == AUTHORED else "mutant",
            "mutation_operator": row["provenance"]["mutation_operator"],
            "signature": row["signature"], "behavior": row["behavior"],
            "changed_lines": changed_lines(row["reference_code"], row["buggy_code"]),
            "diff": list(difflib.unified_diff(row["reference_code"].splitlines(), row["buggy_code"].splitlines(),
                                              "reference", "buggy", lineterm="", n=0))[2:],
            "baseline_passed_ids": row["verification"]["baseline_passed_ids"],
            "baseline_failed_ids": row["verification"]["baseline_failed_ids"],
            "mechanical_checks": checks, "mechanical_pass": all(checks.values()),
            "manual_review": {"reviewer": None, "verdict": "PENDING", "notes": None,
                              "questions": ["Behavior text unambiguous and matches reference?",
                                            "Tests cover boundary, ordinary and invariant cases?",
                                            "Bug realistic and not an equivalent mutant?",
                                            "Difficulty label follows the fixed human rule?"]},
        })
    return {"schema_version": 1, "scope": "spec02-pre-scaling-audit",
            "dataset": str(directory), "manifest_sha256": digest(manifest),
            "dataset_sha256": manifest["dataset_sha256"], "sample_seed": seed, "sample_size": len(cases),
            "attrition": attrition(manifest, rows),
            "mechanical_summary": {"cases_passing_all_checks": sum(c["mechanical_pass"] for c in cases),
                                   "failures_by_check": dict(sorted(Counter(k for c in cases for k, v in c["mechanical_checks"].items() if not v).items()))},
            "manual_review_status": "PENDING",
            "limitations": ["mechanical checks are not the manual review required by spec 02",
                            "no fixture code was executed or compiled by this audit"],
            "cases": cases}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        output = Path(args.output)
        if output.exists():
            raise ValueError("refuse to overwrite an existing audit")
        report = build_audit(Path(args.dataset), args.seed, args.manifest_sha256)
        write_json(output, report)
        summary = report["mechanical_summary"]
        print(f"AUDIT SAMPLE: {report['sample_size']} cases; mechanical pass "
              f"{summary['cases_passing_all_checks']}/{report['sample_size']}; manual review PENDING")
        return 0
    except Exception as exc:
        print("AUDIT BLOCKED:", exc, file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
