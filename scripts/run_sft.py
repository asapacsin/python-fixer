"""Run the frozen single-seed SFT experiment after the full GPU preflight.

All inputs and outputs stay outside Git. This command refuses to train unless
the exact preflight report is a full Spec 07 PASS, the frozen model and train
split are verified, and the run root matches the recorded persistent volume.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
TRAIN_RELATIVE = Path("verification/evidence/m1-v9/m1-fixtures-v9/train.jsonl")
MODEL_FILES = {
    "model.safetensors": "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b",
    "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
}
SEED = 17
USER_MAX_COST_YUAN = 170.0
USER_MAX_ELAPSED_HOURS = 12.0


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_preflight(path: Path, sft_only: bool = False) -> dict:
    report = load_json(path)
    expected_scope = "SFT_ONLY_GPU_PREFLIGHT" if sft_only else "FULL_SPEC07_SFT_PREFLIGHT"
    if report.get("status") != "PASS" or report.get("scope") != expected_scope:
        raise ValueError(f"preflight must have status PASS and scope {expected_scope}")
    required = {"model", "train_data_sha256", "persistent_storage", "gpu", "sft_smoke", "identity"}
    required |= {"checkpoint_readback", "optimizer_rng_resume"} if sft_only else {"sandbox", "checkpoint_recovery"}
    if not required.issubset(report):
        raise ValueError("preflight report is incomplete")
    if report["model"].get("revision") != MODEL_REVISION:
        raise ValueError("preflight used a different base-model revision")
    expected_hashes = report["model"].get("sha256", {})
    if any(expected_hashes.get(name) != value for name, value in MODEL_FILES.items()):
        raise ValueError("preflight report does not identify the pinned model and tokenizer hashes")
    smoke = report["sft_smoke"]
    if not report["gpu"].get("name") or not 20 <= smoke.get("steps", 0) <= 50:
        raise ValueError("preflight lacks a GPU-bound 20–50-step SFT overfit check")
    gpu_name = report["gpu"].get("name", "")
    if (report["gpu"].get("role") != "training" or "推理" in gpu_name
            or "inference" in gpu_name.casefold()):
        raise ValueError("SFT preflight is not bound to a training-class GPU")
    if (smoke.get("lora_optimizer_steps_with_nonzero_grad") != smoke.get("steps")
            or smoke.get("sampled_base_tensors_unchanged") is not True
            or smoke.get("memory_headroom_bytes", 0) < 2 * 1024**3):
        raise ValueError("preflight lacks LoRA-gradient, frozen-base, or GPU-headroom evidence")
    if sft_only:
        if report.get("full_spec07_preflight") != "NOT_VERIFIED":
            raise ValueError("SFT-only report must retain the unverified full Spec 07 status")
        if report.get("checkpoint_readback", {}).get("status") != "PASS":
            raise ValueError("SFT-only report lacks current persistent checkpoint hash readback")
        resume = report.get("optimizer_rng_resume", {})
        if resume.get("status") != "PASS" or resume.get("train_data_sha256", "").lower() != report["train_data_sha256"].lower():
            raise ValueError("SFT-only report lacks matching fresh-process optimizer/RNG resume evidence")
    else:
        sandbox = report["sandbox"]
        if sandbox.get("status") != "PASS" or sandbox.get("sandbox") != "gvisor" or not sandbox.get("worker_id"):
            raise ValueError("preflight lacks PASS evidence from the separate enforcing sandbox worker")
    storage = report["persistent_storage"]
    if (not storage.get("path") or not storage.get("mount_source")
            or (not sft_only and not storage.get("volume_id"))):
        raise ValueError("preflight lacks verified persistent checkpoint storage identity")
    if not sft_only:
        recovery = report["checkpoint_recovery"]
        if recovery.get("status") != "PASS" or recovery.get("fresh_process_optimizer_rng_resume") is not True:
            raise ValueError("preflight lacks fresh-process optimizer/RNG checkpoint recovery evidence")
    identity = report.get("identity", {})
    if not identity.get("run_id") or not identity.get("report_sha256"):
        raise ValueError("preflight lacks run identity or report hash")
    return {"path": str(path.resolve()), "sha256": digest(path), "report": report}


def verify_inputs(model_dir: Path, train_jsonl: Path, train_sha256: str) -> tuple[list[dict], dict]:
    from python_fixer.data.models import load_problems, model_view

    revision_file = model_dir / "PINNED_REVISION"
    if not model_dir.is_dir() or not revision_file.is_file():
        raise ValueError("pinned local model directory is required")
    if revision_file.read_text(encoding="ascii").strip() != MODEL_REVISION:
        raise ValueError("model directory revision is not the frozen Qwen3-0.6B commit")
    model_hashes = {}
    for name, expected in MODEL_FILES.items():
        path = model_dir / name
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f"pinned model file missing or hash mismatch: {name}")
        model_hashes[name] = expected
    if not train_jsonl.is_file() or "validation" in train_jsonl.name.lower() or "test" in train_jsonl.name.lower():
        raise ValueError("explicit frozen train-only JSONL is required")
    actual = digest(train_jsonl)
    if actual.lower() != train_sha256.lower():
        raise ValueError("training JSONL hash mismatch")
    sums_path = ROOT / "verification/evidence/m1-v9/SHA256SUMS"
    train_rel = "./m1-fixtures-v9/train.jsonl"
    frozen_hash = next((line.split()[0] for line in sums_path.read_text().splitlines()
                        if line.endswith("  " + train_rel)), None)
    if not frozen_hash or actual != frozen_hash:
        raise ValueError("train JSONL does not match the committed v9 SHA256SUMS")
    rows = load_problems(train_jsonl)
    if not rows or any(row.get("split") != "train" for row in rows):
        raise ValueError("training input includes a non-train split")
    for row in rows:
        model_view(row, "sft")
    return rows, {"path": str(train_jsonl.resolve()), "sha256": actual, "rows": len(rows),
                  "model_sha256": model_hashes}


def verify_run_root(path: Path, expected_source: str, preflight: dict) -> dict:
    if not path.is_dir():
        raise ValueError("run root must already be mounted and provisioned")
    mounted = subprocess.run(["findmnt", "--noheadings", "--output", "SOURCE,FSTYPE",
                              "--target", str(path)], check=True, capture_output=True, text=True)
    fields = mounted.stdout.strip().split()
    if len(fields) != 2 or fields[0] != expected_source:
        raise ValueError("run-root mount does not match the expected provider mount source")
    if fields[0] in {"/", "overlay"} or fields[1] in {"overlay", "rootfs"}:
        raise ValueError("ephemeral root/overlay storage is not allowed")
    recorded = preflight["report"]["persistent_storage"]
    if recorded.get("mount_source") != fields[0]:
        raise ValueError("run-root mount differs from the preflight mount")
    if Path(recorded["path"]).resolve() != path.resolve():
        raise ValueError("run-root path differs from the path verified by preflight")
    free = shutil.disk_usage(path).free
    if free < 8 * 1024**3:
        raise ValueError("run root has less than 8 GiB free")
    return {"path": str(path.resolve()), "mount_source": fields[0], "filesystem": fields[1],
            "free_bytes_before": free}


def budget_window(args: argparse.Namespace) -> dict:
    values = (args.max_cost_yuan, args.max_hours, args.already_billed_yuan,
              args.billing_unit_yuan)
    if any(value is None for value in values) or args.billing_unit_minutes is None or args.current_unit_minutes_left is None or args.node_elapsed_minutes is None:
        raise ValueError("complete provider billing inputs are required for a full Spec 07 run")
    if any(not math.isfinite(value) for value in values):
        raise ValueError("budget values must be finite numbers")
    if args.max_cost_yuan <= 0 or args.max_hours <= 0 or args.billing_unit_yuan <= 0:
        raise ValueError("budget limits and billing unit cost must be positive")
    if args.max_cost_yuan > USER_MAX_COST_YUAN or args.max_hours > USER_MAX_ELAPSED_HOURS:
        raise ValueError("SFT limits cannot exceed the user's ¥170 or 12-hour caps")
    if args.already_billed_yuan < 0 or args.already_billed_yuan > args.max_cost_yuan:
        raise ValueError("already-billed amount must be between zero and the total cost cap")
    if args.already_billed_yuan + 1e-9 < args.billing_unit_yuan:
        raise ValueError("current billing interval must be confirmed paid before SFT can start")
    if args.billing_unit_minutes < 1 or not 0 <= args.current_unit_minutes_left <= args.billing_unit_minutes:
        raise ValueError("billing interval and remaining minutes are inconsistent")
    if args.node_elapsed_minutes < 0:
        raise ValueError("node elapsed time cannot be negative")
    if args.shutdown_reserve_seconds < 60:
        raise ValueError("at least 60 seconds must be reserved for checkpoint finalization")

    future_units = math.floor((args.max_cost_yuan - args.already_billed_yuan + 1e-9)
                              / args.billing_unit_yuan)
    billing_window_seconds = (args.current_unit_minutes_left * 60
                              + future_units * args.billing_unit_minutes * 60)
    wall_cap_seconds = args.max_hours * 3600 - args.node_elapsed_minutes * 60
    allowed_seconds = min(wall_cap_seconds, billing_window_seconds)
    training_seconds = allowed_seconds - args.shutdown_reserve_seconds
    if training_seconds <= 0:
        raise ValueError("remaining cost/time window cannot cover training plus checkpoint reserve")
    return {
        "max_cost_yuan": args.max_cost_yuan,
        "already_billed_yuan": args.already_billed_yuan,
        "billing_unit_yuan": args.billing_unit_yuan,
        "billing_unit_minutes": args.billing_unit_minutes,
        "current_unit_minutes_left": args.current_unit_minutes_left,
        "future_units_affordable": future_units,
        "max_wall_hours": args.max_hours,
        "node_elapsed_minutes_before_command": args.node_elapsed_minutes,
        "allowed_seconds_from_training_start": allowed_seconds,
        "shutdown_reserve_seconds": args.shutdown_reserve_seconds,
        "training_deadline_seconds": training_seconds,
    }


def main(argv: list[str] | None = None) -> int:
    command_started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--train-sha256", required=True)
    parser.add_argument("--preflight-report", type=Path, required=True)
    parser.add_argument("--sft-only-preflight", action="store_true",
                        help="run the short SFT pilot with GPU smoke and persistent checkpoint readback; full Spec 07 remains NOT_VERIFIED")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--expected-mount-source", required=True)
    parser.add_argument("--study-id", required=True)
    parser.add_argument("--max-cost-yuan", type=float, default=170.0)
    parser.add_argument("--max-hours", type=float, default=12.0)
    parser.add_argument("--already-billed-yuan", type=float,
                        help="provider spend already charged for this node, including preflight")
    parser.add_argument("--billing-unit-yuan", type=float,
                        help="price charged at the next billing interval, from the provider panel")
    parser.add_argument("--billing-unit-minutes", type=int,
                        help="provider billing interval in minutes (for example 60 hourly or 1440 daily)")
    parser.add_argument("--current-unit-minutes-left", type=int,
                        help="minutes remaining in the already-paid current billing interval")
    parser.add_argument("--node-elapsed-minutes", type=int,
                        help="minutes since this GPU training node began billing")
    parser.add_argument("--sft-deadline-minutes", type=int, default=20,
                        help="elapsed-time ceiling for the explicitly requested SFT-only pilot")
    parser.add_argument("--shutdown-reserve-seconds", type=int, default=300)
    args = parser.parse_args(argv)
    try:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", args.study_id):
            raise ValueError("study ID must be a simple path component using letters, digits, dot, underscore, or hyphen")
        if args.sft_only_preflight:
            if not 2 <= args.sft_deadline_minutes <= 20:
                raise ValueError("SFT-only deadline must be between 2 and 20 minutes")
            budget = {"max_elapsed_minutes": args.sft_deadline_minutes,
                      "training_deadline_seconds": args.sft_deadline_minutes * 60 - 60,
                      "shutdown_reserve_seconds": 60,
                      "cost_status": "NOT_VERIFIED_FROM_PROVIDER_BILLING"}
        else:
            budget = budget_window(args)
        preflight = verify_preflight(args.preflight_report, sft_only=args.sft_only_preflight)
        rows, data = verify_inputs(args.model_dir, args.train_jsonl, args.train_sha256)
        storage = verify_run_root(args.run_root, args.expected_mount_source, preflight)
        if preflight["report"]["train_data_sha256"].lower() != data["sha256"].lower():
            raise ValueError("preflight was run against a different training-data hash")
        for name, expected in data["model_sha256"].items():
            if preflight["report"]["model"]["sha256"].get(name) != expected:
                raise ValueError(f"preflight used a different pinned model file: {name}")
        observed_gpu = subprocess.run(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
                                      check=True, capture_output=True, text=True).stdout.splitlines()
        expected_gpu = preflight["report"]["identity"].get(
            "current_gpu_uuid" if args.sft_only_preflight else "replacement_gpu_uuid")
        if len(observed_gpu) != 1 or observed_gpu[0].strip() != expected_gpu:
            raise ValueError("current GPU UUID does not match the assembled preflight replacement node")

        run_id = f"{args.study_id}-E1-seed{SEED}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
        final_dir = args.run_root / run_id
        staging_dir = args.run_root / ("." + run_id + ".staging")
        if final_dir.exists() or staging_dir.exists():
            raise ValueError("run ID already exists")
        staging_dir.mkdir(mode=0o700)
        config_path = ROOT / "configs/sft.json"
        cfg = load_json(config_path)
        if cfg.get("seed") != SEED:
            raise ValueError("configs/sft.json seed must remain 17 for E1 seed 17")
        if cfg.get("model_revision") not in (None, MODEL_REVISION):
            raise ValueError("SFT config model revision mismatch")

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from python_fixer.training.sft import build_examples, check_collated_labels, make_trainer

        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
        if tokenizer.chat_template is None:
            raise ValueError("pinned tokenizer is missing its chat template")
        examples, quarantine = build_examples(rows, tokenizer, cfg["trainer"]["max_length"])
        if not examples:
            raise ValueError("no eligible SFT rows after token-length filtering")

        model = AutoModelForCausalLM.from_pretrained(args.model_dir, local_files_only=True,
                                                     torch_dtype=torch.bfloat16,
                                                     attn_implementation="sdpa")
        model.config.use_cache = False
        trainer = make_trainer(model, tokenizer, examples, cfg, str(staging_dir / "trainer"),
                               seed=SEED, eval_strategy="no", save_steps=50, save_total_limit=3)
        check_collated_labels(next(iter(trainer.get_train_dataloader())), tokenizer)
        training_started = time.monotonic()
        measured_step_seconds = (preflight["report"]["sft_smoke"].get("wall_seconds", 0)
                                 / max(1, preflight["report"]["sft_smoke"].get("steps", 1)))
        expected_step_seconds = (max(1.5, 2.0 * measured_step_seconds) if args.sft_only_preflight
                                 else max(30.0, 2.0 * measured_step_seconds))
        budget_stop = BudgetStopCallback(command_started, budget["training_deadline_seconds"],
                                         expected_step_seconds)
        trainer.add_callback(RunEvidenceCallback(staging_dir / "metrics.jsonl", training_started))
        trainer.add_callback(budget_stop)
        start_time = datetime.now(timezone.utc)
        result = trainer.train()
        elapsed = time.monotonic() - training_started
        command_elapsed = time.monotonic() - command_started
        torch.cuda.synchronize()
        trainer._save_checkpoint(trainer.model, trial=None)
        trainer.save_model(str(staging_dir / "adapter"))
        tokenizer.save_pretrained(staging_dir / "adapter")
        trainer.save_state()
        if not (staging_dir / "adapter").is_dir():
            raise RuntimeError("SFT adapter export is missing")
        training_complete = result.global_step == trainer.state.max_steps

        hashes = {}
        for file in sorted(staging_dir.rglob("*")):
            if file.is_file():
                with file.open("rb") as handle:
                    os.fsync(handle.fileno())
                hashes[str(file.relative_to(staging_dir))] = digest(file)
        manifest = {
            "status": "COMPLETED_TRAINING_NOT_EVALUATED" if training_complete else "BUDGET_STOPPED_INCOMPLETE",
            "run_id": run_id, "arm": "E1", "seed": SEED,
            "started_at": start_time.isoformat(), "finished_at": datetime.now(timezone.utc).isoformat(),
            "wall_seconds": elapsed, "steps": result.global_step, "train_loss": result.metrics.get("train_loss"),
            "model_id": "Qwen/Qwen3-0.6B", "model_revision": MODEL_REVISION,
            "model_dir": str(args.model_dir.resolve()), "data": data,
            "config_sha256": digest(config_path), "preflight": {"path": preflight["path"], "sha256": preflight["sha256"],
                "scope": preflight["report"]["scope"],
                "full_spec07_preflight": preflight["report"].get("full_spec07_preflight", "PASS")},
            "budget": {**budget, "conservative_step_seconds": expected_step_seconds,
                       "training_elapsed_seconds": elapsed,
                       "command_elapsed_seconds": command_elapsed,
                       "estimate_includes_shutdown_reserve": True,
                       "stop_reason": budget_stop.stop_reason,
                       "estimated_total_spend_yuan": (None if args.sft_only_preflight else
                       args.already_billed_yuan +
                       max(0, math.ceil(max(0, command_elapsed + args.shutdown_reserve_seconds
                                             - args.current_unit_minutes_left * 60)
                                        / (args.billing_unit_minutes * 60))) * args.billing_unit_yuan)},
            "sandbox_preflight": preflight["report"]["sandbox"], "persistent_storage": storage,
            "gpu": {"name": torch.cuda.get_device_name(),
                    "memory_total_bytes": torch.cuda.mem_get_info(torch.cuda.current_device())[1],
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved()},
            "software": {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
                         "cuda_runtime": torch.version.cuda},
            "eligible_train_rows": len(examples), "quarantined_long_rows": quarantine,
            "checkpoint_reload": "NOT_VERIFIED", "validation_evaluation": "NOT_RUN", "artifacts": hashes,
        }
        (staging_dir / "run-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        for directory, _, _ in os.walk(staging_dir, topdown=False):
            fd = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        staging_dir.rename(final_dir)
        print(json.dumps({"status": manifest["status"], "run_dir": str(final_dir),
                          "steps": result.global_step, "wall_seconds": elapsed,
                          "stop_reason": budget_stop.stop_reason}, indent=2))
        return 0 if training_complete else 3
    except Exception as exc:
        print(json.dumps({"status": "BLOCKED", "reason": str(exc)}, indent=2), file=sys.stderr)
        return 2


class RunEvidenceCallback:
    """Append observed trainer metrics without storing prompts or generated text."""

    def __new__(cls, path: Path, start: float):
        from transformers import TrainerCallback

        class Callback(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                record = {"step": state.global_step, "elapsed_seconds": time.monotonic() - start,
                          "logs": logs or {}, "learning_rate": getattr(args, "learning_rate", None)}
                with path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, allow_nan=False) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                return control

        return Callback()


class BudgetStopCallback:
    """Stop before starting an update that could cross the budget window."""

    def __new__(cls, started: float, deadline_seconds: float, estimated_step_seconds: float):
        from transformers import TrainerCallback

        class Callback(TrainerCallback):
            def __init__(self):
                self.started = started
                self.deadline_seconds = deadline_seconds
                self.estimated_step_seconds = estimated_step_seconds
                self.stop_reason = None

            def on_step_begin(self, args, state, control, **kwargs):
                elapsed = time.monotonic() - self.started
                if elapsed + self.estimated_step_seconds >= self.deadline_seconds:
                    self.stop_reason = "budget_window_reserve_reached"
                    control.should_training_stop = True
                return control

        return Callback()


if __name__ == "__main__":
    raise SystemExit(main())
