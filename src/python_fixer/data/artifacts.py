"""Load verified registries only through immutable data + execution evidence."""
from __future__ import annotations
import copy
import hashlib
from pathlib import Path
from python_fixer.common import digest, strict_loads
from .models import load_problems


def load_verified_partition(directory: Path, split: str, *, expected_manifest_sha256: str | None = None) -> tuple[list[dict], dict]:
    if split not in {"train","validation","test"}:
        raise ValueError("invalid split")
    directory = directory.resolve()
    manifest = strict_loads((directory/"manifest.json").read_text())
    if expected_manifest_sha256 is not None and digest(manifest)!=expected_manifest_sha256:
        raise ValueError("manifest does not match operator-pinned hash")
    path = directory/(split+".jsonl")
    if path.is_symlink() or not path.resolve().is_relative_to(directory):
        raise ValueError("partition cannot escape immutable dataset directory")
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["partition_sha256"][split]:
        raise ValueError("partition hash mismatch")
    rows = load_problems(path)
    for row in rows:
        identifier = row["problem_id"]
        if row["split"]!=split or digest(row)!=manifest["problem_hashes"][identifier]:
            raise ValueError("row does not match frozen manifest")
        # Never resolve a caller-provided ID as an arbitrary host path.
        evidence_path = directory/"evidence"/(identifier+".json")
        if not evidence_path.resolve().is_relative_to(directory/"evidence") or evidence_path.is_symlink():
            raise ValueError("invalid evidence path")
        evidence = strict_loads(evidence_path.read_text())
        expected = manifest["evidence_hashes"][identifier]
        if digest(evidence)!=expected or row["verification"]["evidence_sha256"]!=expected:
            raise ValueError("execution evidence hash mismatch")
        original = copy.deepcopy(row)
        original["verification"]={"status":"NOT_RUN"}
        if digest(original)!=evidence["problem_sha256"]:
            raise ValueError("execution evidence belongs to another problem")
        verification = row["verification"]
        if verification["status"]!="VERIFIED" or evidence["image_digest"]!=verification["image_digest"] or evidence["policy_sha256"]!=verification["policy_sha256"]:
            raise ValueError("runner identity mismatch")
        if len(evidence["runs"])<3:
            raise ValueError("missing repeated execution evidence")
        all_ids = {t["test_id"] for t in row["tests"]}
        baseline = set(verification["baseline_passed_ids"])
        for run in evidence["runs"]:
            if run["reference"]["status"]!="OK" or set(run["reference"]["passed_ids"])!=all_ids:
                raise ValueError("reference execution evidence failed")
            if run["buggy"]["status"]!="OK" or not run["buggy"]["compiled"] or set(run["buggy"]["passed_ids"])!=baseline:
                raise ValueError("buggy baseline differs from execution evidence")
    if len(rows)!=manifest["split_counts"].get(split,0):
        raise ValueError("partition count mismatch")
    return rows,manifest
