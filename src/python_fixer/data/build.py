"""Build immutable, sandbox-verified fixture datasets (no model or GPU needed)."""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from importlib.resources import files
from pathlib import Path
import subprocess
import sys
from python_fixer.common import canonical_bytes, digest, strict_loads, text_digest, atomic_write, write_json
from .grouping import group_families, split_for_family
from .models import validate_problem
from .mutations import mutate


def fixture_seeds() -> list[dict]:
    return strict_loads(files("python_fixer.data").joinpath("fixtures.json").read_text())


def plan_rows(seeds: list[dict], *, split_seed: int = 17, clone_threshold: float = .92,
              mutations_per_train_function: int = 0) -> list[dict]:
    if not 0 < clone_threshold <= 1 or mutations_per_train_function < 0:
        raise ValueError("invalid grouping/mutation configuration")
    identities = [s["source_function_id"] for s in seeds]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate source function")
    for seed in seeds:
        if seed["license"] not in {"CC0-1.0", "MIT", "Apache-2.0", "BSD-3-Clause"}:
            raise ValueError("source license is not approved")
        test_inputs = [digest({"args": t["args"], "kwargs": t["kwargs"]}) for t in seed["tests"]]
        if len(set(test_inputs)) != len(test_inputs):
            raise ValueError("duplicate test input adds no independent assertion")
    families = group_families(seeds, clone_threshold)
    rows = []
    for seed in sorted(seeds, key=lambda s: s["source_function_id"]):
        function = seed["source_function_id"]
        family = families[function]
        split = split_for_family(family, split_seed)
        variants = [("authored-bug-v1", seed["buggy_code"])]
        if split == "train":
            variants += mutate(seed["reference_code"])[:mutations_per_train_function]
        for operator, buggy in variants:
            row = {"schema_version": 1, "problem_id": function + "-" + text_digest(operator)[:12],
                "source_function_id": function, "source_family_id": family, "split": split,
                "signature": seed["signature"], "behavior": seed["behavior"],
                "reference_code": seed["reference_code"], "buggy_code": buggy,
                "bug_category": seed["bug_category"], "difficulty": seed["difficulty"],
                "test_suite_id": function + "-tests-v1", "test_suite_sha256": digest(seed["tests"]),
                "tests": seed["tests"], "verification": {"status": "NOT_RUN"},
                "provenance": {"source_uri": "authored://python-fixer/fixtures-v1", "revision": "fixtures-v1",
                    "path": "src/python_fixer/data/fixtures.json", "license": seed["license"],
                    "source_sha256": text_digest(seed["reference_code"]), "mutation_operator": operator,
                    "mutation_seed": split_seed}}
            validate_problem(row)
            rows.append(row)
    return sorted(rows, key=lambda r: r["problem_id"])


def write_dataset(destination: Path, rows: list[dict], evidence: dict, rejected: list[dict], config: dict) -> dict:
    if destination.exists():
        raise ValueError("refuse to overwrite an immutable dataset; choose a new output directory")
    payload = b"".join(canonical_bytes(row)+b"\n" for row in rows)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    manifest = {"schema_version": 1, "stage": "CPU_VERIFIED_PRE_TOKEN_FILTER",
        "builder_commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "builder_source_sha256": digest({str(p.relative_to(Path(__file__).parents[1])):text_digest(p.read_text())
            for p in Path(__file__).parents[1].rglob("*.py")}),
        "content_plan_sha256": digest([{**r,"verification":{"status":"NOT_RUN"}} for r in rows]),
        "config_sha256": digest(config), "source_sha256": digest(fixture_seeds()),
        "row_count": len(rows), "family_count": len({r["source_family_id"] for r in rows}),
        "split_counts": dict(Counter(r["split"] for r in rows)),
        "category_counts": dict(Counter(r["bug_category"] for r in rows)),
        "difficulty_counts": dict(Counter(r["difficulty"] for r in rows)),
        "split_membership_sha256": digest({r["problem_id"]: [r["split"], r["source_family_id"]] for r in rows}),
        "dataset_sha256": __import__("hashlib").sha256(payload).hexdigest(),
        "partition_sha256": {split:__import__("hashlib").sha256(b"".join(canonical_bytes(r)+b"\n" for r in rows if r["split"]==split)).hexdigest()
                             for split in ("train","validation","test")},
        "problem_hashes": {r["problem_id"]: digest(r) for r in rows},
        "evidence_hashes": {key: digest(value) for key, value in evidence.items()},
        "rejected": rejected, "license_ledger": {"authored-fixtures-v1": "CC0-1.0"},
        "limitations": ["16 authored fixtures, not a research corpus", "token eligibility not applied",
                         "50-case independent/manual data audit not performed"]}
    # Publish manifest last. Partial directories have no manifest and are not datasets.
    destination.mkdir(parents=True)
    atomic_write(destination / "problems.jsonl", payload)
    for split in ("train","validation","test"):
        atomic_write(destination / (split+".jsonl"),b"".join(canonical_bytes(r)+b"\n" for r in rows if r["split"]==split))
    for key, value in evidence.items():
        write_json(destination / "evidence" / (key + ".json"), value)
    write_json(destination / "manifest.json", manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--plan-only", action="store_true", help="print metadata plan; never claims verification")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    try:
        config = strict_loads(Path(args.config).read_text())
        if config["source"] != "authored-fixtures-v1" or config["schema_version"] != 1:
            raise ValueError("unsupported source/config")
        rows = plan_rows(fixture_seeds(), split_seed=config["split_seed"], clone_threshold=config["clone_threshold"],
                         mutations_per_train_function=config["mutations_per_train_function"])
        if args.plan_only:
            print(canonical_bytes({"status": "NOT_VERIFIED", "planned_rows": len(rows),
                  "split_counts": dict(Counter(r["split"] for r in rows)), "plan_sha256": digest(rows)}).decode())
            return 0
        from python_fixer.sandbox.docker import DockerSandbox
        from python_fixer.sandbox.policy import Policy
        from python_fixer.sandbox.gateway import verify_problem
        sandbox = DockerSandbox(Policy.load(Path(config["sandbox_config"])))
        sandbox.available()
        if not 1 <= config["workers"] <= 4:
            raise ValueError("workers must be within policy")
        destination = Path(args.output or config["output"])
        if destination.exists():
            raise ValueError("output already exists")
        def verify(row):
            try:
                checked, evidence = verify_problem(sandbox, row, config["verification_repeats"])
                return checked, evidence, None
            except ValueError as exc:
                return None, None, {"problem_id": row["problem_id"], "reason": str(exc)}
        results = list(ThreadPoolExecutor(config["workers"]).map(verify, rows))
        accepted = [r for r, _, _ in results if r is not None]
        evidence = {r["problem_id"]: e for r, e, _ in results if r is not None}
        rejected = [reason for _, _, reason in results if reason is not None]
        if not accepted:
            raise ValueError("zero verified rows; dataset cannot be published: "+str(Counter(r["reason"] for r in rejected)))
        manifest = write_dataset(destination, accepted, evidence, rejected, config)
        print(canonical_bytes({"status": "CPU_VERIFIED", "output": str(destination),
              "accepted": len(accepted), "rejected": len(rejected), "manifest_sha256": digest(manifest)}).decode())
        return 0 if not rejected else 2
    except Exception as exc:
        print("BUILD BLOCKED:", str(exc), file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
