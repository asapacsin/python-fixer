"""Reverify selected frozen train rows under a new bounded sandbox policy."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import os
from pathlib import Path
import shutil
import sys
import tempfile

from python_fixer.common import atomic_write, canonical_bytes, digest, write_json
from python_fixer.data.artifacts import load_verified_partition
from python_fixer.sandbox.docker import DockerSandbox
from python_fixer.sandbox.gateway import verify_problem
from python_fixer.sandbox.policy import Policy


def publish_subset(staging: Path, destination: Path) -> None:
    """Publish on the Linux worker or Windows without replacing a concurrent destination."""
    if os.name == "nt":
        os.rename(staging, destination)
        return
    if sys.platform != "linux":
        raise OSError("atomic no-replace publication requires Linux or Windows")
    import ctypes
    libc = ctypes.CDLL(None, use_errno=True)
    rename = libc.renameat2
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    at_fdcwd, rename_noreplace = -100, 1
    if rename(at_fdcwd, os.fsencode(staging), at_fdcwd, os.fsencode(destination), rename_noreplace):
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(destination))


def reverify_train_subset(source: Path, destination: Path, parent_pin: str,
                          policy_path: Path, problem_ids: set[str], sandbox=None) -> dict:
    rows, parent = load_verified_partition(source, "train", expected_manifest_sha256=parent_pin)
    source = source.resolve()
    destination = destination.resolve()
    if destination.exists():
        raise ValueError("refuse to overwrite immutable output")
    if destination.is_relative_to(source):
        raise ValueError("output must be outside the parent dataset")
    if not problem_ids or not problem_ids <= {row["problem_id"] for row in rows}:
        raise ValueError("selection must contain existing train problem IDs")
    selected = [row for row in rows if row["problem_id"] in problem_ids]
    policy = Policy.load(policy_path)
    sandbox = sandbox or DockerSandbox(policy)
    if sandbox.policy.sha256 != policy.sha256:
        raise ValueError('sandbox does not match requested policy')
    sandbox.available()
    verified, proofs = [], {}
    for row in selected:
        candidate = dict(row)
        candidate["verification"] = {"status": "NOT_RUN"}
        checked, proof = verify_problem(sandbox, candidate, repeats=3)
        verified.append(checked)
        proofs[row["problem_id"]] = proof
    payload = b"".join(canonical_bytes(row) + b"\n" for row in verified)
    manifest = {
        "schema_version": parent["schema_version"],
        "stage": parent["stage"],
        "builder_commit": parent["builder_commit"],
        "builder_source_sha256": parent["builder_source_sha256"],
        "source_sha256": parent["source_sha256"],
        "split_namespace": parent["split_namespace"],
        "parent_source_manifest_sha256": parent_pin,
        "parent_manifest_sha256": parent_pin,
        "reverification": {"selection": sorted(problem_ids), "selected_count": len(verified),
                            "parent_train_count": len(rows), "policy_sha256": policy.sha256,
                            "image_digest": policy.image, "verification_repeats": 3},
        "scope": "SELECTED_TRAIN_ROWS_ONLY", "row_count": len(verified),
        "family_count": len({row["source_family_id"] for row in verified}),
        "split_counts": {"train": len(verified)},
        "category_counts": dict(Counter(row["bug_category"] for row in verified)),
        "difficulty_counts": dict(Counter(row["difficulty"] for row in verified)),
        "split_membership_sha256": digest({row["problem_id"]: [row["split"], row["source_family_id"]]
                                            for row in verified}),
        "dataset_sha256": hashlib.sha256(payload).hexdigest(),
        "partition_sha256": {"train": hashlib.sha256(payload).hexdigest()},
        "problem_hashes": {row["problem_id"]: digest(row) for row in verified},
        "evidence_hashes": {key: digest(value) for key, value in proofs.items()},
    }
    manifest["config_sha256"] = digest({"parent_manifest_sha256": parent_pin,
                                        "policy_sha256": policy.sha256,
                                        "verification_repeats": 3})
    staging = Path(tempfile.mkdtemp(prefix=destination.name + ".staging-", dir=destination.parent))
    try:
        atomic_write(staging / "problems.jsonl", payload)
        atomic_write(staging / "train.jsonl", payload)
        (staging / "evidence").mkdir()
        for key, proof in proofs.items():
            write_json(staging / "evidence" / (key + ".json"), proof)
        write_json(staging / "manifest.json", manifest)
        loaded, _ = load_verified_partition(staging, "train", expected_manifest_sha256=digest(manifest))
        if loaded != verified:
            raise ValueError("readback differs from verified selection")
        publish_subset(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {"status": "PASS", "selected_count": len(loaded), "parent_train_count": len(rows),
            "parent_manifest_sha256": parent_pin, "manifest_sha256": digest(manifest),
            "policy_sha256": policy.sha256}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--parent-manifest-sha256", required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--problem-id", action="append", required=True)
    args = parser.parse_args(argv)
    try:
        print(canonical_bytes(reverify_train_subset(args.dataset, args.destination,
              args.parent_manifest_sha256, args.policy, set(args.problem_id))).decode())
        return 0
    except Exception as exc:
        print("REVERIFY BLOCKED:", str(exc), file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
