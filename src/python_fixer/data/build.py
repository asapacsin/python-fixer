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


# Each source is an immutable, ordered list of authored CC0 seed files.
SOURCES = {"authored-fixtures-v1": ("fixtures.json",),
           "authored-fixtures-v2": ("fixtures.json", "fixtures_v2.json")}


def seed_paths(source: str = "authored-fixtures-v1") -> dict[str, str]:
    if source not in SOURCES:
        raise ValueError("unsupported source")
    return {seed["source_function_id"]: "src/python_fixer/data/" + name
            for name in SOURCES[source]
            for seed in strict_loads(files("python_fixer.data").joinpath(name).read_text())}


def fixture_seeds(source: str = "authored-fixtures-v1") -> list[dict]:
    if source not in SOURCES:
        raise ValueError("unsupported source")
    return [seed for name in SOURCES[source]
            for seed in strict_loads(files("python_fixer.data").joinpath(name).read_text())]


def plan_rows(seeds: list[dict], *, split_seed: int = 17, clone_threshold: float = .92,
              mutations_per_train_function: int = 0, source: str = "authored-fixtures-v1") -> list[dict]:
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
    paths = seed_paths(source)
    missing = [s["source_function_id"] for s in seeds if s["source_function_id"] not in paths]
    if missing:
        raise ValueError("seeds do not belong to source " + source + ": " + ", ".join(sorted(missing)[:5]))
    revision = source.removeprefix("authored-")
    families = group_families(seeds, clone_threshold)
    rows = []
    for seed in sorted(seeds, key=lambda s: s["source_function_id"]):
        function = seed["source_function_id"]
        family = families[function]
        split = split_for_family(family, split_seed)
        variants = [("authored-bug-v1", seed["buggy_code"])]
        if split == "train":
            # Discard mutants identical to an earlier variant (e.g. the authored bug):
            # duplicates add no new repair case and would double-count one program.
            seen = {seed["buggy_code"]}
            for operator, buggy in mutate(seed["reference_code"]):
                if len(variants) > mutations_per_train_function:
                    break
                if buggy not in seen:
                    seen.add(buggy)
                    variants.append((operator, buggy))
        for operator, buggy in variants:
            # Fixed rule for mutants: the edited construct defines the category
            # (Compare op -> comparison, BinOp -> operator), not the seed's authored bug.
            category = (seed["bug_category"] if operator == "authored-bug-v1"
                        else "comparison" if operator.split("-")[2] == "ops" else "operator")
            row = {"schema_version": 1, "problem_id": function + "-" + text_digest(operator)[:12],
                "source_function_id": function, "source_family_id": family, "split": split,
                "signature": seed["signature"], "behavior": seed["behavior"],
                "reference_code": seed["reference_code"], "buggy_code": buggy,
                "bug_category": category,
                # Fixed rule: a mutant is one local token edit, hence "easy"; authored
                # bugs keep the author's label (one local edit / interacting branches / algorithmic).
                "difficulty": seed["difficulty"] if operator == "authored-bug-v1" else "easy",
                "test_suite_id": function + "-tests-v1", "test_suite_sha256": digest(seed["tests"]),
                "tests": seed["tests"], "verification": {"status": "NOT_RUN"},
                "provenance": {"source_uri": "authored://python-fixer/" + revision, "revision": revision,
                    "path": paths[function], "license": seed["license"],
                    "source_sha256": text_digest(seed["reference_code"]), "mutation_operator": operator,
                    "mutation_seed": split_seed}}
            validate_problem(row)
            rows.append(row)
    return sorted(rows, key=lambda r: r["problem_id"])


def rejection(row: dict, reason: str) -> dict:
    operator = row["provenance"]["mutation_operator"]
    return {"problem_id": row["problem_id"], "reason": reason, "bug_category": row["bug_category"],
            "mutation_operator": operator, "origin": "authored" if operator == "authored-bug-v1" else "mutant"}


def crash_only_mutants(rows: list[dict], evidence: dict) -> tuple[list[dict], list[dict]]:
    """Fixed rule: a buggy variant must fail at least one test by *returning a wrong value*.
    Mutants whose every failing test raises are the runtime analogue of a syntax-error
    mutant (spec 02) and are rejected. Applies to every variant, authored bugs included."""
    kept, dropped = [], []
    for row in rows:
        buggy = evidence[row["problem_id"]]["runs"][0]["buggy"]
        failing = {t["test_id"] for t in row["tests"]} - set(buggy["passed_ids"])
        if not failing & set(buggy["returned_ids"]):
            dropped.append(rejection(row, "bug fails only by raising; no wrong returned value"))
        else:
            kept.append(row)
    return kept, dropped


def dedupe_by_behaviour(rows: list[dict], evidence: dict) -> tuple[list[dict], list[dict]]:
    """Drop variants whose buggy program behaves identically, on every test, to an
    earlier variant of the same function (authored bug first, then plan order)."""
    order = sorted(rows, key=lambda r: (r["source_function_id"],
                   r["provenance"]["mutation_operator"] != "authored-bug-v1", r["problem_id"]))
    seen, kept, dropped = set(), [], []
    for row in order:
        key = (row["source_function_id"], evidence[row["problem_id"]]["runs"][0]["buggy"]["outcome_sha256"])
        if key in seen:
            dropped.append(rejection(row, "behaviourally identical to another variant of the same function"))
        else:
            seen.add(key)
            kept.append(row)
    return sorted(kept, key=lambda r: r["problem_id"]), dropped


def write_dataset(destination: Path, rows: list[dict], evidence: dict, rejected: list[dict], config: dict,
                  seeds: list[dict] | None = None) -> dict:
    if destination.exists():
        raise ValueError("refuse to overwrite an immutable dataset; choose a new output directory")
    payload = b"".join(canonical_bytes(row)+b"\n" for row in rows)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    manifest = {"schema_version": 1, "stage": "CPU_VERIFIED_PRE_TOKEN_FILTER",
        "builder_commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "builder_source_sha256": digest({str(p.relative_to(Path(__file__).parents[1])):text_digest(p.read_text())
            for p in Path(__file__).parents[1].rglob("*.py")}),
        "content_plan_sha256": digest([{**r,"verification":{"status":"NOT_RUN"}} for r in rows]),
        "config_sha256": digest(config), "source_sha256": digest(seeds if seeds is not None else fixture_seeds(config.get("source", "authored-fixtures-v1"))),
        # Splits are only comparable within one namespace; adding seeds can regroup
        # families (v1 -> v2 moved count_equal/count_positive from train to validation).
        "split_namespace": f"{config.get('source', 'authored-fixtures-v1')}:split-seed-{config.get('split_seed', 17)}",
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
        "rejected": rejected, "license_ledger": {config.get("source", "authored-fixtures-v1"): "CC0-1.0"},
        "limitations": [f"{len({r['source_function_id'] for r in rows})} authored fixture functions, not a research corpus", "token eligibility not applied",
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
        if config["source"] not in SOURCES or config["schema_version"] != 1:
            raise ValueError("unsupported source/config")
        seeds = fixture_seeds(config["source"])
        rows = plan_rows(seeds, source=config["source"], split_seed=config["split_seed"], clone_threshold=config["clone_threshold"],
                         mutations_per_train_function=config["mutations_per_train_function"])
        exclusions = {e["problem_id"]: e for e in config.get("manual_exclusions", [])}
        for entry in exclusions.values():
            # Bind each exclusion to the exact review evidence it cites.
            review = Path(entry["review"])
            if not review.is_file() or __import__("hashlib").sha256(review.read_bytes()).hexdigest() != entry["review_sha256"]:
                raise ValueError("manual exclusion review evidence missing or changed: " + entry["review"])
        unknown = set(exclusions) - {r["problem_id"] for r in rows}
        if unknown:
            raise ValueError("manual exclusion names an unplanned problem: " + ", ".join(sorted(unknown)))
        excluded = [rejection(r, "manual review: " + exclusions[r["problem_id"]]["reason"])
                    for r in rows if r["problem_id"] in exclusions]
        rows = [r for r in rows if r["problem_id"] not in exclusions]
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
                return None, None, rejection(row, str(exc))
        results = list(ThreadPoolExecutor(config["workers"]).map(verify, rows))
        accepted = [r for r, _, _ in results if r is not None]
        evidence = {r["problem_id"]: e for r, e, _ in results if r is not None}
        rejected = excluded + [reason for _, _, reason in results if reason is not None]
        if accepted:
            accepted, crashing = crash_only_mutants(accepted, evidence)
            accepted, duplicates = dedupe_by_behaviour(accepted, evidence)
            duplicates += crashing
            evidence = {r["problem_id"]: evidence[r["problem_id"]] for r in accepted}
            rejected += duplicates
        rejected.sort(key=lambda r: r["problem_id"])
        if not accepted:
            raise ValueError("zero verified rows; dataset cannot be published: "+str(Counter(r["reason"] for r in rejected)))
        manifest = write_dataset(destination, accepted, evidence, rejected, config, seeds)
        print(canonical_bytes({"status": "CPU_VERIFIED", "output": str(destination),
              "accepted": len(accepted), "rejected": len(rejected), "manifest_sha256": digest(manifest)}).decode())
        return 0 if not rejected else 2
    except Exception as exc:
        print("BUILD BLOCKED:", str(exc), file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
