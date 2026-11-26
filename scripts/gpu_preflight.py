"""Partial SFT GPU smoke check for the pinned Qwen3-0.6B model.

This command verifies local inputs and performs a short real SFT smoke run.
It never reads the sealed test split and never executes generated code. A
successful run is SMOKE_PASS only. The complete spec 07 preflight, including
checkpoint reload/equivalence and replacement-node recovery, remains
NOT_VERIFIED.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "Qwen/Qwen3-0.6B"
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
MODEL_FILES = {
    "model.safetensors": "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b",
    "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fail(message: str) -> None:
    raise ValueError(message)


def verify_model(path: Path) -> dict:
    if not path.is_dir():
        fail("model path must be a local directory; network model IDs are rejected")
    revision = path / "PINNED_REVISION"
    if not revision.is_file() or revision.read_text().strip() != MODEL_REVISION:
        fail(f"model directory must contain PINNED_REVISION with {MODEL_REVISION}")
    hashes = {}
    for name, expected in MODEL_FILES.items():
        file_path = path / name
        if not file_path.is_file():
            fail(f"pinned model file is missing: {name}")
        actual = sha256(file_path)
        if actual != expected:
            fail(f"pinned model file hash mismatch: {name}")
        hashes[name] = actual
    return {"repository": MODEL_ID, "revision": MODEL_REVISION, "sha256": hashes}


def load_train_data(path: Path, expected_hash: str) -> tuple[list[dict], str]:
    if not path.is_file() or "test" in path.name.lower() or "validation" in path.name.lower():
        fail("SFT data must be an explicit local train-only JSONL file")
    actual = sha256(path)
    if actual != expected_hash:
        fail("train data SHA256 does not match the supplied frozen manifest hash")
    from python_fixer.data.models import load_problems, model_view

    rows = load_problems(path)
    if not rows or any(row["split"] != "train" for row in rows):
        fail("SFT input must contain train split rows only")
    for row in rows:
        model_view(row, "sft")
    return rows, actual


def verify_sandbox_evidence(path: Path, expected_sha256: str, policy_sha256: str) -> dict:
    if not path.is_file():
        fail("real sandbox preflight evidence is required")
    evidence = json.loads(path.read_text(encoding="utf-8"))
    actual_sha256 = sha256(path)
    if actual_sha256 != expected_sha256.lower():
        fail("worker probe report does not match operator-pinned SHA256")
    if (evidence.get("status") != "PASS"
            or evidence.get("scope") != "SEPARATE_MTLS_GVISOR_WORKER_PROBE_ONLY"
            or evidence.get("full_spec07_preflight") != "NOT_VERIFIED"):
        fail("worker report must be a passing, explicitly scoped separate-worker probe")
    observed = evidence.get("observed_scores")
    if not isinstance(observed, list) or not observed:
        fail("worker probe report has no observed candidate scores")
    for row in observed:
        if (row.get("reference_reward") != 1.0 or row.get("buggy_reward") != 0.0
                or not row.get("reference_binding_match") or not row.get("buggy_binding_match")):
            fail("worker probe lacks matching expected scores and bound suite/policy/image")
    inputs = evidence.get("inputs", {})
    if inputs.get("policy_sha256") != policy_sha256:
        fail("worker probe policy differs from the trainer sandbox policy")
    return {"scope": evidence["scope"], "report_path": str(path.resolve()),
            "operator_pinned_sha256": actual_sha256, "probe_run_id": evidence.get("probe_run_id"),
            "worker_endpoint_host": evidence.get("worker_endpoint_host"),
            "observed_scores": observed, "policy_sha256": policy_sha256,
            "train_data_sha256": inputs.get("train_jsonl_sha256")}


def verify_persistent_root(path: Path, expected_source: str) -> dict:
    if not path.is_dir():
        fail("run root must already exist on the mounted persistent filesystem")
    result = subprocess.run(
        ["findmnt", "--noheadings", "--output", "SOURCE,FSTYPE", "--target", str(path)],
        check=True, capture_output=True, text=True,
    )
    fields = result.stdout.strip().split()
    if len(fields) != 2:
        fail("findmnt did not return a mount source and filesystem type")
    source, filesystem = fields
    if source != expected_source:
        fail("run-root mount source does not match --expected-mount-source")
    if source in {"/", "overlay"} or filesystem in {"overlay", "rootfs"}:
        fail("root/overlay filesystems cannot be used as persistent run storage")
    probe = path / (".preflight-write-" + str(os.getpid()))
    payload = os.urandom(64)
    try:
        probe.write_bytes(payload)
        with probe.open("rb") as stream:
            os.fsync(stream.fileno())
        if probe.read_bytes() != payload:
            fail("persistent run-root write/readback check failed")
    finally:
        probe.unlink(missing_ok=True)
    free = shutil.disk_usage(path).free
    if free < 8 * 1024**3:
        fail("persistent run root needs at least 8 GiB free for model/cache/checkpoint artifacts")
    return {"path": str(path.resolve()), "mount_source": source, "filesystem": filesystem,
            "free_bytes": free}


def verify_gpu(expected_name: str) -> dict:
    import torch

    if not expected_name or not expected_name.strip():
        fail("an exact training-GPU label from the provider is required")
    if not torch.cuda.is_available():
        fail("CUDA is unavailable")
    if not torch.cuda.is_bf16_supported():
        fail("GPU does not support BF16")
    device = torch.cuda.current_device()
    uuid_result = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader", "-i", str(device)],
        check=True, capture_output=True, text=True,
    )
    gpu_uuid = uuid_result.stdout.strip()
    if not gpu_uuid or "\n" in gpu_uuid:
        fail("could not establish a single actual GPU UUID")
    name = torch.cuda.get_device_name(device)
    if "推理" in name or "inference" in name.casefold():
        fail("the active GPU is designated for inference and cannot be used for SFT")
    if name.strip() != expected_name.strip():
        fail("observed GPU name differs from --expected-training-gpu-name")
    memory_free_bytes, memory_total_bytes = torch.cuda.mem_get_info(device)
    if memory_total_bytes < 20 * 1024**3:
        fail("GPU has less than 20 GiB VRAM")
    if memory_free_bytes < 4 * 1024**3:
        fail("GPU has less than 4 GiB free before loading the pinned model")
    a = torch.randn((256, 256), device="cuda", dtype=torch.bfloat16)
    result = a @ a
    torch.cuda.synchronize()
    if not torch.isfinite(result).all().item():
        fail("BF16 CUDA matrix check produced non-finite output")
    return {"name": name, "role": "training", "device": device, "uuid": gpu_uuid,
            "total_memory_bytes": memory_total_bytes,
            "memory_free_before_bytes": memory_free_bytes,
            "driver": torch.cuda.driver_version() if hasattr(torch.cuda, "driver_version") else None,
            "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated()}


def run_short_sft(model_path: Path, rows: list[dict], output_path: Path, steps: int, seed: int) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
    from python_fixer.training.sft import build_examples, check_collated_labels, make_trainer, parameter_hashes

    cfg = json.loads((ROOT / "configs/sft.json").read_text(encoding="utf-8"))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.chat_template is None:
        fail("pinned tokenizer has no chat template")
    examples, quarantine = build_examples(rows[:16], tokenizer, cfg["trainer"]["max_length"])
    if quarantine or len(examples) != 16:
        fail("preflight requires 16 eligible verified train examples")
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True,
                                                  torch_dtype=torch.bfloat16,
                                                  attn_implementation="sdpa")
    model.config.use_cache = False
    trainer = make_trainer(model, tokenizer, examples, cfg, str(output_path), eval_examples=examples, seed=seed,
                           max_steps=steps, num_train_epochs=1, save_steps=steps,
                           save_total_limit=2, eval_strategy="no", logging_steps=1)
    batch = next(iter(trainer.get_train_dataloader()))
    check_collated_labels(batch, tokenizer)
    trainable = [name for name, param in trainer.model.named_parameters() if param.requires_grad]
    if not trainable or any("lora_" not in name for name in trainable):
        fail("SFT smoke must train LoRA parameters only")

    def is_sampled_base_weight(name: str) -> bool:
        return "lora_" not in name and any(f"layers.{index}." in name for index in (0, 27)) and name.endswith(".weight")

    before = parameter_hashes(trainer.model, lambda name: "lora_" in name)
    base_before = parameter_hashes(trainer.model, is_sampled_base_weight)
    if not base_before:
        fail("could not hash sampled frozen base-model layers")
    before_eval = trainer.evaluate(metric_key_prefix="toy_train_before")
    before_loss = before_eval.get("toy_train_before_loss")
    if not isinstance(before_loss, (int, float)) or not math.isfinite(before_loss):
        fail("preflight toy-set loss before training is missing or non-finite")

    class LoraGradientEvidence(TrainerCallback):
        def __init__(self):
            self.optimizer_steps = 0
            self.nonzero_gradient_steps = 0
            self.max_lora_grad_norm = 0.0

        def on_pre_optimizer_step(self, callback_args, state, control, **kwargs):
            grads = [parameter.grad.detach().float() for name, parameter in trainer.model.named_parameters()
                     if "lora_" in name and parameter.grad is not None]
            if grads:
                grad_norm = torch.linalg.vector_norm(torch.stack([grad.norm() for grad in grads])).item()
                self.optimizer_steps += 1
                if math.isfinite(grad_norm) and grad_norm > 0:
                    self.nonzero_gradient_steps += 1
                    self.max_lora_grad_norm = max(self.max_lora_grad_norm, grad_norm)
            return control

    gradient_evidence = LoraGradientEvidence()
    trainer.add_callback(gradient_evidence)
    started = time.monotonic()
    result = trainer.train()
    elapsed = time.monotonic() - started
    after = parameter_hashes(trainer.model, lambda name: "lora_" in name)
    base_after = parameter_hashes(trainer.model, is_sampled_base_weight)
    metrics = result.metrics
    if result.global_step != steps or not before or before == after:
        fail("SFT smoke did not complete requested steps or change LoRA tensors")
    if not torch.isfinite(torch.tensor(metrics.get("train_loss", float("nan")))).item():
        fail("SFT smoke loss is non-finite")
    if gradient_evidence.nonzero_gradient_steps != steps or gradient_evidence.optimizer_steps != steps:
        fail("SFT smoke did not observe finite nonzero LoRA gradients at every optimizer step")
    if base_before != base_after:
        fail("sampled frozen base-model tensors changed during LoRA training")
    after_eval = trainer.evaluate(metric_key_prefix="toy_train_after")
    after_loss = after_eval.get("toy_train_after_loss")
    if not isinstance(after_loss, (int, float)) or not math.isfinite(after_loss) or after_loss >= before_loss:
        fail("overfit smoke did not reduce finite loss on its 16 train-only examples")
    trainer.save_model(str(output_path / "adapter"))
    tokenizer.save_pretrained(output_path / "adapter")
    trainer.save_state()
    first = examples[0]["prompt"]
    expected = greedy_output(trainer.model, tokenizer, first)
    expected_path = output_path / "expected-generation.json"
    expected_path.write_text(json.dumps({"prompt": first, "completion": expected}), encoding="utf-8")
    reload = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--verify-adapter-reload",
         "--model-dir", str(model_path), "--adapter-dir", str(output_path / "adapter"),
         "--expected-generation", str(expected_path)],
        check=True, capture_output=True, text=True,
    )
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    _, memory_total_bytes = torch.cuda.mem_get_info(torch.cuda.current_device())
    memory_headroom = memory_total_bytes - peak_reserved
    if memory_headroom < 2 * 1024**3:
        fail("SFT smoke left less than 2 GiB of device-memory headroom")
    del trainer, model
    torch.cuda.empty_cache()
    return {"steps": result.global_step, "train_loss": metrics["train_loss"], "wall_seconds": elapsed,
            "toy_train_loss_before": before_loss, "toy_train_loss_after": after_loss,
            "lora_optimizer_steps_with_nonzero_grad": gradient_evidence.nonzero_gradient_steps,
            "max_lora_grad_norm": gradient_evidence.max_lora_grad_norm,
            "sampled_base_tensors_unchanged": base_before == base_after,
            "peak_allocated_bytes": peak_allocated, "peak_reserved_bytes": peak_reserved,
            "memory_headroom_bytes": memory_headroom,
            "lora_tensor_count": len(after), "adapter_changed": before != after,
            "fresh_process_adapter_reload": json.loads(reload.stdout),
            "trainer_state_saved": (output_path / "trainer_state.json").is_file(),
            }


def resume_worker(args: argparse.Namespace) -> dict:
    """Run one leg of the deterministic resume comparison in its own process."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
    from python_fixer.training.sft import build_examples, make_trainer, parameter_hashes

    rows, _ = load_train_data(args.train_jsonl, args.train_sha256)
    cfg = json.loads((ROOT / "configs/sft.json").read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    examples, quarantine = build_examples(rows[:16], tokenizer, cfg["trainer"]["max_length"])
    if quarantine or len(examples) != 16:
        fail("resume check requires 16 eligible verified train examples")

    class StopAfterCheckpoint(TrainerCallback):
        def on_step_end(self, current_args, state, control, **kwargs):
            if args.stop_at and state.global_step >= args.stop_at:
                control.should_training_stop = True
            return control

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, local_files_only=True,
                                                  torch_dtype=torch.bfloat16,
                                                  attn_implementation="sdpa")
    model.config.use_cache = False
    trainer = make_trainer(
        model, tokenizer, examples, cfg, str(args.output_dir), seed=args.seed,
        max_steps=3, num_train_epochs=1, save_steps=2, save_total_limit=2,
        eval_strategy="no", logging_steps=1, save_strategy="steps",
    )
    if args.resume_checkpoint:
        result = trainer.train(resume_from_checkpoint=str(args.resume_checkpoint))
    else:
        trainer.add_callback(StopAfterCheckpoint())
        result = trainer.train()
    trainer.save_state()
    step_losses = {str(int(item["step"])): float(item["loss"]) for item in trainer.state.log_history
                   if "step" in item and "loss" in item}
    if any(not math.isfinite(loss) for loss in step_losses.values()):
        fail("resume run logged a non-finite loss")
    lora_hashes = parameter_hashes(trainer.model, lambda name: "lora_" in name)
    if not lora_hashes:
        fail("resume run produced no LoRA tensors")
    return {"global_step": result.global_step, "step_losses": step_losses,
            "lora_hashes": lora_hashes,
            "pid": os.getpid(), "output_dir": str(args.output_dir)}


def verify_optimizer_resume(model_dir: Path, train_jsonl: Path, train_sha256: str,
                            work_dir: Path, seed: int, provider_node_id: str,
                            expected_training_gpu_name: str) -> dict:
    """Compare a fresh-process checkpoint resume with the same uninterrupted schedule."""
    work_dir.mkdir(parents=True, exist_ok=False)
    script = str(Path(__file__).resolve())

    def invoke(leg: str, output: Path, checkpoint: Path | None = None) -> dict:
        result_path = output.parent / (output.name + "-result.json")
        command = [sys.executable, script, "--resume-worker", "--model-dir", str(model_dir),
                   "--train-jsonl", str(train_jsonl), "--train-sha256", train_sha256,
                   "--resume-output", str(output), "--seed", str(seed), "--resume-leg", leg,
                   "--result-file", str(result_path)]
        if checkpoint:
            command.extend(["--resume-checkpoint", str(checkpoint)])
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode:
            fail(f"{leg} resume worker failed ({completed.returncode}): {completed.stderr[-2000:]} "
                 f"{completed.stdout[-2000:]}")
        return json.loads(result_path.read_text(encoding="utf-8"))

    reference = invoke("reference", work_dir / "reference")
    interrupted = invoke("interrupt", work_dir / "interrupted")
    checkpoint = work_dir / "interrupted" / "checkpoint-2"
    required_files = ("optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json")
    checkpoint_files = (*required_files, "adapter_model.safetensors", "adapter_config.json")
    files = {}
    for name in checkpoint_files:
        candidate = checkpoint / name
        if not candidate.is_file():
            fail(f"resume checkpoint is missing required state: {name}")
        files[name] = sha256(candidate)
    resumed = invoke("resume", work_dir / "resumed", checkpoint)
    reference_loss = reference["step_losses"].get("3")
    resumed_loss = resumed["step_losses"].get("3")
    if reference["global_step"] != 3 or interrupted["global_step"] != 2 or resumed["global_step"] != 3:
        fail("reference/interrupted/resumed runs did not stop at expected update counts")
    if resumed["pid"] in {reference["pid"], interrupted["pid"]}:
        fail("resume step did not run in a distinct fresh process")
    if reference_loss is None or resumed_loss is None:
        fail("step-3 loss missing from Trainer logs")
    tolerance = max(1e-4, abs(reference_loss) * 0.01)
    loss_delta = abs(reference_loss - resumed_loss)
    reference_hashes, resumed_hashes = reference["lora_hashes"], resumed["lora_hashes"]
    adapter_match = reference_hashes == resumed_hashes
    if loss_delta > tolerance or not adapter_match:
        fail("fresh-process resume diverged from uninterrupted run")
    return {"status": "PASS", "seed": seed, "train_data_sha256": train_sha256,
            "provider_node_id": provider_node_id,
            "checkpoint_path": str(checkpoint.resolve()),
            "checkpoint_state_sha256": files,
            "reference_gpu_uuid": verify_gpu(expected_training_gpu_name)["uuid"],
            "reference_global_step": reference["global_step"],
            "interrupted_global_step": interrupted["global_step"],
            "resumed_global_step": resumed["global_step"],
            "reference_next_step_loss": reference_loss, "resumed_next_step_loss": resumed_loss,
            "loss_delta": loss_delta, "loss_tolerance": tolerance, "adapter_hashes_match": adapter_match,
            "process_ids": [reference["pid"], interrupted["pid"], resumed["pid"]],
            "reference_lora_hashes": reference_hashes}


def verify_replacement_recovery(record_path: Path, model_dir: Path, train_jsonl: Path,
                                train_sha256: str, output_dir: Path, run_root: Path,
                                expected_mount_source: str, provider_node_id: str,
                                expected_training_gpu_name: str) -> dict:
    parent_record_sha256 = sha256(record_path)
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if record.get("status") != "PASS" or record.get("train_data_sha256") != train_sha256:
        fail("original resume evidence is missing or uses different train data")
    current_gpu = verify_gpu(expected_training_gpu_name)["uuid"]
    if current_gpu == record.get("reference_gpu_uuid"):
        fail("replacement recovery requires a different GPU UUID from the original run")
    if not record.get("provider_node_id") or provider_node_id == record["provider_node_id"]:
        fail("replacement recovery requires a different operator-observed provider node ID")
    checkpoint = Path(record["checkpoint_path"])
    persistent = verify_persistent_root(run_root, expected_mount_source)
    persistent["node_id"] = provider_node_id
    resolved_checkpoint = checkpoint.resolve()
    try:
        resolved_checkpoint.relative_to(run_root.resolve())
    except ValueError:
        fail("original checkpoint must reside beneath the verified persistent run root")
    allowed_files = {"optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json",
                     "adapter_model.safetensors", "adapter_config.json"}
    if set(record.get("checkpoint_state_sha256", {})) - allowed_files:
        fail("original evidence contains unrecognized checkpoint file names")
    for name, expected in record.get("checkpoint_state_sha256", {}).items():
        file_path = checkpoint / name
        if not file_path.is_file() or sha256(file_path) != expected:
            fail(f"saved interrupted checkpoint hash mismatch: {name}")
    required = allowed_files
    if set(record.get("checkpoint_state_sha256", {})) != required:
        fail("original evidence does not hash every required checkpoint state file")
    output_dir.mkdir(parents=True, exist_ok=False)
    result_file = output_dir / "replacement-result.json"
    command = [sys.executable, str(Path(__file__).resolve()), "--resume-worker",
               "--model-dir", str(model_dir), "--train-jsonl", str(train_jsonl),
               "--train-sha256", train_sha256, "--resume-output", str(output_dir / "run"),
               "--seed", str(record["seed"]), "--resume-leg", "resume",
               "--resume-checkpoint", str(checkpoint), "--result-file", str(result_file)]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode:
        fail(f"replacement resume failed ({completed.returncode}): {completed.stderr[-2000:]}")
    resumed = json.loads(result_file.read_text(encoding="utf-8"))
    reference_loss = record.get("reference_next_step_loss")
    resumed_loss = resumed.get("step_losses", {}).get("3")
    if resumed.get("global_step") != 3 or resumed.get("pid") == os.getpid():
        fail("replacement resume did not complete step 3 in a fresh process")
    tolerance = record.get("loss_tolerance")
    delta = abs(reference_loss - resumed_loss) if reference_loss is not None and resumed_loss is not None else None
    adapter_match = resumed.get("lora_hashes") == record.get("reference_lora_hashes")
    if delta is None or tolerance is None or delta > tolerance or not adapter_match:
        fail("replacement-node recovery differs from original uninterrupted reference")
    return {"status": "PASS", "scope": "REPLACEMENT_RECOVERY_ONLY",
            "full_spec07_preflight": "NOT_VERIFIED",
            "parent_resume_record_sha256": parent_record_sha256,
            "original_gpu_uuid": record["reference_gpu_uuid"], "replacement_gpu_uuid": current_gpu,
            "original_node_id": record["provider_node_id"], "replacement_node_id": provider_node_id,
            "replacement_persistent_storage": persistent,
            "checkpoint_state_sha256_verified": record["checkpoint_state_sha256"],
            "resumed_step": resumed["global_step"], "loss_delta": delta,
            "loss_tolerance": tolerance, "adapter_hashes_match": adapter_match,
            "replacement_pid": resumed["pid"]}


def greedy_output(model, tokenizer, prompt: str) -> str:
    import torch

    model.eval()
    encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        generated = model.generate(**encoded, do_sample=False, max_new_tokens=64)
    return tokenizer.decode(generated[0, encoded["input_ids"].shape[1]:], skip_special_tokens=False)


def verify_adapter_reload(model_path: Path, adapter_path: Path, expected_path: Path) -> dict:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, local_files_only=True)
    base = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True,
                                                torch_dtype=torch.bfloat16,
                                                attn_implementation="sdpa").to("cuda")
    model = PeftModel.from_pretrained(base, adapter_path).eval()
    actual = greedy_output(model, tokenizer, expected["prompt"])
    if actual != expected["completion"]:
        fail("fresh-process adapter reload greedy generation differs")
    return {"status": "PASS", "comparison": "exact_greedy_generation",
            "completion_sha256": hashlib.sha256(actual.encode()).hexdigest()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-adapter-reload", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--resume-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--replacement-recovery", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--train-jsonl", type=Path)
    parser.add_argument("--train-sha256")
    parser.add_argument("--sandbox-evidence", type=Path)
    parser.add_argument("--sandbox-sha256")
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--expected-mount-source")
    parser.add_argument("--provider-node-id")
    parser.add_argument("--expected-training-gpu-name",
                        help="exact GPU label from the provider; inference-labeled GPUs are always rejected")
    parser.add_argument("--steps", type=int, default=20,
                        help="train-only 16-example overfit preflight, from 20 through 50 optimizer steps")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--adapter-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--expected-generation", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--resume-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--resume-checkpoint", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--resume-leg", choices=("reference", "interrupt", "resume"), help=argparse.SUPPRESS)
    parser.add_argument("--stop-at", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--result-file", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--original-resume-evidence", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not 20 <= args.steps <= 50:
        parser.error("--steps must be between 20 and 50 to satisfy the SFT preflight specification")
    if args.verify_adapter_reload:
        try:
            result = verify_adapter_reload(args.model_dir, args.adapter_dir, args.expected_generation)
            print(json.dumps(result))
            return 0
        except Exception as exc:
            print(json.dumps({"status": "FAIL", "reason": str(exc)}), file=sys.stderr)
            return 2
    if args.replacement_recovery:
        try:
            result = verify_replacement_recovery(args.original_resume_evidence, args.model_dir,
                                                 args.train_jsonl, args.train_sha256, args.resume_output,
                                                 args.run_root, args.expected_mount_source, args.provider_node_id,
                                                 args.expected_training_gpu_name)
            print(json.dumps(result, indent=2))
            return 0
        except Exception as exc:
            print(json.dumps({"status": "FAIL", "scope": "REPLACEMENT_RECOVERY_ONLY",
                              "reason": str(exc)}, indent=2), file=sys.stderr)
            return 2
    if args.resume_worker:
        try:
            args.output_dir = args.resume_output
            args.stop_at = 2 if args.resume_leg == "interrupt" else None
            result = resume_worker(args)
            if args.result_file is None:
                fail("resume worker requires --result-file")
            args.result_file.write_text(json.dumps(result), encoding="utf-8")
            return 0
        except Exception as exc:
            print(json.dumps({"status": "FAIL", "reason": str(exc)}), file=sys.stderr)
            return 2
    if not all((args.train_jsonl, args.train_sha256, args.sandbox_evidence, args.sandbox_sha256, args.run_root,
                args.expected_mount_source, args.provider_node_id, args.expected_training_gpu_name)):
        parser.error("normal smoke mode requires train data/hash, worker evidence, persistent root, node ID, and expected training GPU name")
    if len(args.train_sha256) != 64:
        parser.error("--train-sha256 must be a SHA256 hex digest")
    try:
        model = verify_model(args.model_dir)
        rows, data_hash = load_train_data(args.train_jsonl, args.train_sha256)
        from python_fixer.sandbox.policy import Policy
        policy_path = ROOT / "configs/sandbox.json"
        sandbox_policy_sha256 = Policy.load(policy_path).sha256
        sandbox = verify_sandbox_evidence(args.sandbox_evidence, args.sandbox_sha256, sandbox_policy_sha256)
        if sandbox.get("train_data_sha256") != data_hash:
            fail("worker probe report uses different train data")
        persistent = verify_persistent_root(args.run_root, args.expected_mount_source)
        persistent["node_id"] = args.provider_node_id
        gpu = verify_gpu(args.expected_training_gpu_name)
        run_id = f"sft-preflight-seed{args.seed}-{int(time.time())}"
        output = args.run_root / run_id
        output.mkdir(mode=0o700)
        sft = run_short_sft(args.model_dir, rows, output / "preflight", args.steps, args.seed)
        resume = verify_optimizer_resume(args.model_dir, args.train_jsonl, data_hash,
                                        output / "resume-verification", args.seed, args.provider_node_id,
                                        args.expected_training_gpu_name)
        resume_path = output / "optimizer-resume.json"
        resume_path.write_text(json.dumps(resume, indent=2) + "\n", encoding="utf-8")
        resume_sha256 = sha256(resume_path)
        report = {
            "status": "SMOKE_PASS",
            "scope": "SFT_GPU_SMOKE_ONLY",
            "full_spec07_preflight": "NOT_VERIFIED",
            "not_verified": ["replacement-node recovery", "full spec07 external worker integration"],
            "model": model,
            "train_data_sha256": data_hash,
            "train_examples": len(rows),
            "sandbox": sandbox,
            "persistent_storage": persistent,
            "gpu": gpu,
            "sft_smoke": sft,
            "optimizer_rng_resume": resume,
            "optimizer_rng_resume_path": str(resume_path.resolve()),
            "optimizer_rng_resume_sha256": resume_sha256,
            "run_id": run_id,
            "python": sys.version,
            "platform": platform.platform(),
        }
        report_path = output / "preflight-report.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({"status": "SMOKE_PASS", "full_spec07_preflight": "NOT_VERIFIED",
                          "report": str(report_path), "run_id": run_id}, indent=2))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "BLOCKED", "reason": str(exc)}, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
