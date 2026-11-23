#!/usr/bin/env python3
"""Validate M0 contracts only. Never loads models or executes dataset programs."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import tomllib

try:
    from jsonschema import Draft202012Validator
except ImportError:
    raise SystemExit("Install CPU checks: python3 -m pip install -r requirements-dev.txt")

ROOT = Path(__file__).resolve().parents[1]
SPEC_NAMES = [
    "00-project-overview", "01-system-architecture", "02-dataset-design",
    "03-sft-training", "04-grpo-training", "05-reward-engine",
    "06-evaluation-protocol", "07-cloud-infrastructure", "08-security-and-risks",
    "09-experiment-plan",
]


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate_problem(row: dict, schema: dict) -> None:
    Draft202012Validator(schema).validate(row)
    tests = row["tests"]
    ids = [t["test_id"] for t in tests]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate test IDs")
    expected_hash = sha(json.dumps(tests, sort_keys=True, separators=(",", ":")).encode())
    if row["test_suite_sha256"] != expected_hash:
        raise ValueError("test suite hash mismatch")
    if row["provenance"]["source_sha256"] != sha(row["reference_code"].encode()):
        raise ValueError("authored example source hash mismatch")
    verified = row["verification"]
    if verified["status"] == "VERIFIED":
        failed, passed = set(verified["baseline_failed_ids"]), set(verified["baseline_passed_ids"])
        if failed & passed or failed | passed != set(ids):
            raise ValueError("baseline sets must partition the suite")


def config_issues(sft: dict, grpo: dict, evaluation: dict) -> list[str]:
    issues = []
    s, g = sft["trainer"], grpo["trainer"]
    if sft["enable_thinking"] or grpo["enable_thinking"] or evaluation["enable_thinking"]:
        issues.append("primary study requires explicit non-thinking mode")
    if not s["completion_only_loss"] or s["assistant_only_loss"] or s["packing"]:
        issues.append("SFT prompt-completion mask contract changed")
    if g["num_generations"] < 2 or g["generation_batch_size"] % g["num_generations"]:
        issues.append("GRPO generation batch must contain complete groups")
    if "steps_per_generation" in g:
        issues.append("do not configure both generation_batch_size and steps_per_generation")
    if g["generation_batch_size"] != g["per_device_train_batch_size"] * g["gradient_accumulation_steps"]:
        issues.append("primary one-device generation and update batch mismatch")
    if g["loss_type"] != "dr_grpo" or g["scale_rewards"] != "group" or g["beta"] != 0:
        issues.append("primary objective differs from preregistered design")
    if g["max_prompt_length"] + g["max_completion_length"] != s["max_length"]:
        issues.append("sequence budgets differ")
    if g["max_completion_length"] != evaluation["primary"]["max_new_tokens"]:
        issues.append("primary completion budget differs")
    if evaluation["primary"]["do_sample"]:
        issues.append("primary pass@1 must be greedy; sampled pass@1 is secondary")
    if any(k > evaluation["secondary"]["n"] for k in evaluation["secondary"]["k"]):
        issues.append("pass@k requires n >= k")
    for name, value in (("SFT", sft), ("GRPO", grpo)):
        if value["status"] != "UNVERIFIED_INITIAL_CONFIGURATION":
            issues.append(name + " GPU settings must remain marked unverified in M0")
    return issues


def check_links(root: Path) -> list[str]:
    issues = []
    for path in sorted(source_paths(root)):
        if path.suffix != ".md":
            continue
        if any(p.startswith(".") for p in path.relative_to(root).parts):
            continue
        for target in re.findall(r"\[[^\]]*\]\(([^\s)]+)\)", path.read_text(encoding="utf-8")):
            if re.match(r"^[a-zA-Z][\w+.-]*:", target) or target.startswith("#"):
                continue
            destination = (path.parent / target.split("#", 1)[0]).resolve()
            if not destination.is_relative_to(root.resolve()) or not destination.is_file():
                issues.append(f"broken local link: {path.relative_to(root)} -> {target}")
    return issues


def source_paths(root: Path) -> list[Path]:
    raw = subprocess.check_output(["git","-C",str(root),"ls-files","--cached","--others","--exclude-standard","-z"])
    return [root/os.fsdecode(rel) for rel in set(raw.split(b"\0")) if rel and (root/os.fsdecode(rel)).is_file()]


def validate_repository(root: Path) -> tuple[list[str], list[str]]:
    issues, checks = [], []
    required = ["README.md", "AGENTS.md", "pyproject.toml", "docs/implementation-roadmap.md",
                "docs/adr/001-training-and-execution.md", "docs/ai-native-practice.md"]
    required += [f"specs/{name}.md" for name in SPEC_NAMES]
    for name in required:
        if not (root / name).is_file():
            issues.append("missing required file: " + name)
    checks.append("specification-files")
    issues.extend(check_links(root))
    checks.append("local-document-links")
    for path in source_paths(root):
        if path.suffix != ".json":
            continue
        if not any(p.startswith(".") for p in path.relative_to(root).parts):
            load(path)
    tomllib.loads((root / "pyproject.toml").read_text())
    for path in (root / "schemas").glob("*.schema.json"):
        Draft202012Validator.check_schema(load(path))
    checks.append("json-toml-and-schema-syntax")
    schema = load(root / "schemas/problem.schema.json")
    Draft202012Validator.check_schema(schema)
    seen = set()
    for line in (root / "examples/problem.jsonl").read_text().splitlines():
        row = json.loads(line)
        validate_problem(row, schema)
        if row["problem_id"] in seen:
            issues.append("duplicate example problem ID")
        seen.add(row["problem_id"])
        if row["verification"]["status"] != "NOT_RUN":
            issues.append("M0 example must not claim sandbox verification")
    checks.append("problem-schema-and-example-hashes")
    issues += config_issues(*(load(root / f"configs/{name}.json") for name in ("sft", "grpo", "evaluation")))
    candidate = load(root / "configs/training-candidate.json")
    if candidate["status"] != "UNVERIFIED_CANDIDATE_NOT_A_LOCKFILE":
        issues.append("candidate dependency tuple is not a verified lock")
    if candidate["packages"]["trl"] != "0.23.1":
        issues.append("update version-specific specifications before changing TRL")
    checks.append("experiment-config-consistency")
    requirements = load(root / "verification/requirements.json")["requirements"]
    by_id = {r["id"]: r for r in requirements}
    if len(by_id) != len(requirements):
        issues.append("duplicate requirement IDs")
    contract = load(root / "verification/contracts/m1.json")
    Draft202012Validator(load(root / "schemas/task-contract.schema.json")).validate(contract)
    gate_ids = {g["id"]: g for g in contract["gates"]}
    for rid in contract["required_requirement_ids"]:
        requirement = by_id[rid]
        gate = gate_ids.get(requirement["gate_id"])
        if not gate or gate["level"] != requirement["level"] or not gate["required"]:
            issues.append("missing required evidence level: " + rid)
    if contract["budgets"]["max_attempts"] > 3 or contract["budgets"]["max_total_seconds"] <= 0:
        issues.append("task budget missing or excessive")
    if contract["independent_review"]["status"] != "NOT_VERIFIED":
        issues.append("M1 must not claim a review before implementation")
    checks.append("task-requirements-and-budget")
    return issues, checks


def git_bytes(root: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.PIPE)


def source_snapshot(root: Path) -> dict:
    head = subprocess.run(["git", "-C", str(root), "rev-parse", "--verify", "HEAD"],
                          capture_output=True, text=True)
    if head.returncode:
        # Null means a positively identified unborn symbolic branch, not arbitrary Git failure.
        branch = git_bytes(root, "symbolic-ref", "HEAD").decode().strip()
        exists = subprocess.run(["git", "-C", str(root), "show-ref", "--verify", "--quiet", branch])
        if exists.returncode != 1:
            raise ValueError("cannot establish valid HEAD or unborn branch")
    files = {}
    for raw in set(git_bytes(root, "ls-files", "--cached", "--others", "--exclude-standard", "-z").split(b"\0")):
        if not raw:
            continue
        rel = os.fsdecode(raw)
        path = root / rel
        if path.is_symlink():
            files[rel] = "symlink:" + sha(os.fsencode(os.readlink(path)))
        else:
            files[rel] = sha(path.read_bytes()) if path.is_file() else "DELETED"
    payload = {"commit": head.stdout.strip() if head.returncode == 0 else None,
               "files": files,
               "unstaged_sha256": sha(git_bytes(root, "diff", "--no-ext-diff", "--no-textconv", "--binary")),
               "staged_sha256": sha(git_bytes(root, "diff", "--cached", "--no-ext-diff", "--no-textconv", "--binary"))}
    payload["fingerprint"] = sha(json.dumps(payload, sort_keys=True).encode())
    return payload


def report_path(root: Path, supplied: str) -> Path:
    runtime = root / ".ai-native"
    if runtime.is_symlink():
        raise ValueError("runtime directory cannot be a symlink")
    path = (root / supplied).resolve()
    if not path.is_relative_to(runtime.resolve()) or path == runtime.resolve() or path.suffix != ".json":
        raise ValueError("reports must be JSON under ignored .ai-native/")
    ignored = subprocess.run(["git", "-C", str(root), "check-ignore", "-q", str(path)])
    if ignored.returncode != 0:
        raise ValueError("report destination must be gitignored")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", help="write static-check evidence under .ai-native/")
    args = parser.parse_args()
    try:
        issues, checks = validate_repository(ROOT)
        if args.report:
            destination = report_path(ROOT, args.report)
            result = {"schema_version": 1, "scope": "M0-static-contract-checks-only",
                      "status": "FAIL" if issues else "PASS", "exit_code": 1 if issues else 0,
                      "command": "python3 scripts/validate_repository.py",
                      "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                      "checks": checks, "issues": issues, "snapshot": source_snapshot(ROOT),
                      "independent_review": "NOT_VERIFIED", "gpu_and_sandbox": "NOT_RUN"}
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode="w", dir=destination.parent, delete=False) as stream:
                temporary = Path(stream.name)
                try:
                    json.dump(result, stream, indent=2, sort_keys=True)
                    stream.write("\n")
                    stream.close()
                    temporary.replace(destination)
                finally:
                    temporary.unlink(missing_ok=True)
        for issue in issues:
            print("FAIL:", issue, file=sys.stderr)
        if issues:
            return 1
        print(f"PASS: {len(checks)} M0 specification checks; no GPU/sandbox/model review executed")
        return 0
    except Exception as exc:
        print(f"VALIDATION ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
