from __future__ import annotations
from pathlib import Path
import copy
from jsonschema import Draft202012Validator
from python_fixer.common import digest, text_digest, strict_loads, resource_path

ROOT = Path(__file__).resolve().parents[3]


def validate_problem(row: dict) -> None:
    schema = strict_loads(resource_path("schemas/problem.schema.json").read_text())
    Draft202012Validator(schema).validate(row)
    if row["test_suite_sha256"] != digest(row["tests"]):
        raise ValueError("test bundle hash mismatch")
    if row["provenance"]["source_sha256"] != text_digest(row["reference_code"]):
        raise ValueError("reference hash mismatch")
    ids = [t["test_id"] for t in row["tests"]]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate test IDs")
    for test in row["tests"]:
        digest(test)  # Reject NaN/Infinity recursively, including expected values.
    if row["verification"]["status"] == "VERIFIED":
        failed = set(row["verification"]["baseline_failed_ids"])
        passed = set(row["verification"]["baseline_passed_ids"])
        if failed & passed or failed | passed != set(ids):
            raise ValueError("baseline IDs must partition the complete suite")


def load_problems(path: Path) -> list[dict]:
    rows = [strict_loads(line) for line in path.read_text().splitlines() if line.strip()]
    ids, families, functions = set(), {}, {}
    for row in rows:
        validate_problem(row)
        if row["problem_id"] in ids:
            raise ValueError("duplicate problem ID")
        ids.add(row["problem_id"])
        for ledger, key in ((families, "source_family_id"), (functions, "source_function_id")):
            identity = row[key]
            if identity in ledger and ledger[identity] != row["split"]:
                raise ValueError("cross-split source identity: " + key)
            ledger[identity] = row["split"]
    return rows


def messages(row: dict) -> list[dict]:
    # Construct from an allowlist: never copy a canonical row into model input.
    template = resource_path("configs/prompts/repair.txt").read_text()
    system, user = template.removeprefix("SYSTEM:\n").split("\nUSER:\n",1)
    return [{"role":"system","content":system.strip()},
            {"role":"user","content":user.format(behavior=row["behavior"],signature=row["signature"],
                                                   buggy_code=row["buggy_code"]).strip()}]


def model_view(row: dict, role: str) -> dict:
    validate_problem(row)
    if row["verification"]["status"] != "VERIFIED":
        raise ValueError("unverified rows cannot enter model datasets")
    if role in {"sft", "rl"} and row["split"] != "train":
        raise ValueError("held-out and validation rows cannot enter training")
    prompt = messages(row)
    if role == "sft":
        return {"prompt": prompt, "completion": row["reference_code"]}
    if role in {"rl", "evaluation"}:
        return {"prompt": prompt, "problem_id": row["problem_id"]}
    raise ValueError("unknown view")


def mark_verified(row: dict, evidence: dict) -> dict:
    result = copy.deepcopy(row)
    result["verification"] = evidence
    validate_problem(result)
    return result
