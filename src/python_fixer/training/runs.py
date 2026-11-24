"""Run-state rules for M3-M5 (spec 03/04/09): budgets, verified checkpoints,
validation-only selection and experiment-manifest completeness."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

PRE_RUN_FIELDS = ("study_id", "run_id", "arm", "seed", "hypothesis", "environment_lock_sha256",
                  "model_sha", "tokenizer_sha", "parent_adapter_sha256", "dataset_manifest_sha256",
                  "prompt_sha256", "parser_sha256", "reward_version", "sandbox_policy_sha256",
                  "trainer_args", "max_steps", "budget", "selection_rule", "operator")
POST_RUN_FIELDS = ("completed_steps", "candidates", "tokens", "loss_trace", "reward_trace", "status",
                   "stop_reason", "hardware", "cost", "checkpoints", "resume_history", "selected_checkpoint")
ARMS = {"E0", "E0-P", "E1", "E2", "E3"}
MANIFEST_NAME = "checkpoint-manifest.json"


class BudgetExceeded(RuntimeError):
    pass


class CandidateBudget:
    """Counts every generated candidate (valid, invalid, retried) against a hard cap."""

    def __init__(self, max_candidates: int, used: int = 0):
        if max_candidates <= 0 or used < 0 or used > max_candidates:
            raise ValueError("invalid candidate budget")
        self.max_candidates, self.used = max_candidates, used

    def consume(self, n: int) -> None:
        if n <= 0:
            raise ValueError("consume a positive count")
        if self.used + n > self.max_candidates:
            raise BudgetExceeded(f"candidate budget {self.max_candidates} would be exceeded "
                                 f"({self.used}+{n})")
        self.used += n

    def state(self) -> dict:
        return {"max_candidates": self.max_candidates, "used": self.used}


def planned_candidates(max_steps: int, generation_batch_size: int, gradient_accumulation_steps: int,
                       per_device_train_batch_size: int, world_size: int = 1) -> int:
    """Candidates generated over a run. One generation batch feeds steps_per_generation micro-steps."""
    per_update = per_device_train_batch_size * gradient_accumulation_steps * world_size
    if generation_batch_size % (per_device_train_batch_size * world_size):
        raise ValueError("generation batch must divide by microbatch x world size")
    return max_steps * per_update


def select_checkpoint(evaluations: list[dict]) -> dict:
    """Highest validation pass@1; ties go to the earliest step. Held-out scores are refused."""
    if not evaluations:
        raise ValueError("no validation evaluations")
    steps = [e["step"] for e in evaluations]
    if len(set(steps)) != len(steps):
        raise ValueError("duplicate checkpoint steps")
    for e in evaluations:
        if e.get("split") != "validation":
            raise ValueError("selection may only use validation results")
        if not (isinstance(e["pass_at_1"], (int, float)) and math.isfinite(e["pass_at_1"])
                and 0 <= e["pass_at_1"] <= 1):
            raise ValueError("invalid pass@1")
    best = max(e["pass_at_1"] for e in evaluations)
    return min((e for e in evaluations if e["pass_at_1"] == best), key=lambda e: e["step"])


def _file_hashes(directory: Path) -> dict[str, str]:
    out = {}
    for path in sorted(p for p in directory.rglob("*") if p.is_file() and p.name != MANIFEST_NAME):
        out[path.relative_to(directory).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def seal_checkpoint(directory: Path, step: int) -> dict:
    """Write the manifest last, atomically: a directory without it is partial."""
    directory = Path(directory)
    manifest = {"step": step, "files": _file_hashes(directory)}
    if not manifest["files"]:
        raise ValueError("empty checkpoint")
    tmp = directory / (MANIFEST_NAME + ".tmp")
    tmp.write_text(json.dumps(manifest, sort_keys=True))
    os.replace(tmp, directory / MANIFEST_NAME)
    return manifest


def verify_checkpoint(directory: Path) -> bool:
    path = Path(directory) / MANIFEST_NAME
    if not path.is_file():
        return False
    try:
        manifest = json.loads(path.read_text())
    except ValueError:
        return False
    return manifest.get("files") == _file_hashes(Path(directory))


def latest_verified_checkpoint(run_root: Path) -> Path | None:
    candidates = []
    for d in Path(run_root).glob("checkpoint-*"):
        suffix = d.name.split("-", 1)[1]
        if d.is_dir() and suffix.isdigit() and verify_checkpoint(d):
            candidates.append((int(suffix), d))
    return max(candidates)[1] if candidates else None


def validate_manifest(manifest: dict, *, completed: bool = False) -> list[str]:
    """Return missing/invalid fields. Unknown costs must be null, never invented."""
    problems = [f for f in PRE_RUN_FIELDS if f not in manifest]
    if manifest.get("arm") not in ARMS:
        problems.append("arm")
    if manifest.get("arm") == "E2" and not manifest.get("parent_adapter_sha256"):
        problems.append("parent_adapter_sha256 (E2 starts from the selected E1 adapter)")
    if completed:
        problems += [f for f in POST_RUN_FIELDS if f not in manifest]
        if manifest.get("status") == "COMPLETED" and problems:
            problems.append("status COMPLETED with missing fields")
    return problems
