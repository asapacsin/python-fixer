"""Assemble full SFT readiness from hash-pinned observed evidence.

This command performs no probes or training. It verifies the original GPU smoke,
live worker probe, operator-pinned enforcement report, replacement recovery and
current persistent-volume readback before producing FULL_SPEC07_SFT_PREFLIGHT.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
MODEL_FILES = {
    "model.safetensors": "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b",
    "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
}
REQUIRED_WORKER_CHECKS = {
    "normal-return", "forbidden-import", "forbidden-read", "introspection", "signature-change",
    "syntax", "loop", "memory", "forged-pass", "network_denied", "rootfs_denied",
    "sentinel_absent", "secret_env_absent", "gpu_absent", "docker_absent", "nonroot",
    "host-sentinel-unchanged", "output-cap", "fresh-workspace", "container-memory-cap",
    "process-cap", "thread-storm-bounded", "dns_denied", "descendant-bounded", "exception",
    "recursion", "nested-return", "expected-values-absent-suite", "expected-values-absent-per-test",
    "expected-probe-positive-control", "segfault-candidate-failure", "deep-source-rewardable",
    "big-int-return", "surrogate-return", "no-container-leaks",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_pinned(path: Path, expected: str, label: str) -> tuple[dict, str]:
    if not path.is_file():
        raise ValueError(f"{label} file is missing: {path}")
    actual = sha256(path)
    if actual.lower() != expected.lower():
        raise ValueError(f"{label} SHA256 mismatch")
    return json.loads(path.read_text(encoding="utf-8")), actual


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-smoke", type=Path, required=True)
    parser.add_argument("--gpu-smoke-sha256", required=True)
    parser.add_argument("--worker-probe", type=Path, required=True)
    parser.add_argument("--worker-probe-sha256", required=True)
    parser.add_argument("--worker-enforcement", type=Path, required=True)
    parser.add_argument("--worker-enforcement-sha256", required=True,
                        help="operator-observed hash from a separate trusted channel")
    parser.add_argument("--worker-id", required=True, help="operator-confirmed distinct CPU worker identity")
    parser.add_argument("--worker-host", required=True, help="operator-confirmed HTTPS worker host")
    parser.add_argument("--original-node-id", required=True)
    parser.add_argument("--replacement-node-id", required=True)
    parser.add_argument("--replacement-gpu-uuid", required=True,
                        help="operator-observed replacement GPU UUID, checked against nvidia-smi")
    parser.add_argument("--replacement-recovery", type=Path, required=True)
    parser.add_argument("--replacement-recovery-sha256", required=True)
    parser.add_argument("--train-sha256", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--expected-mount-source", required=True)
    parser.add_argument("--expected-volume-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise ValueError("output already exists")
        if not re.fullmatch(r"[0-9a-fA-F]{64}", args.train_sha256):
            raise ValueError("train SHA256 must be a hex digest")

        smoke, smoke_hash = read_pinned(args.gpu_smoke, args.gpu_smoke_sha256, "GPU smoke")
        worker, worker_hash = read_pinned(args.worker_probe, args.worker_probe_sha256, "worker probe")
        enforcement, enforcement_hash = read_pinned(
            args.worker_enforcement, args.worker_enforcement_sha256, "worker enforcement")
        recovery, recovery_hash = read_pinned(
            args.replacement_recovery, args.replacement_recovery_sha256, "replacement recovery")

        if smoke.get("status") != "SMOKE_PASS" or smoke.get("scope") != "SFT_GPU_SMOKE_ONLY":
            raise ValueError("GPU input is not an observed SFT smoke report")
        model = smoke.get("model", {})
        if model.get("revision") != MODEL_REVISION or model.get("sha256") != MODEL_FILES:
            raise ValueError("GPU smoke model revision or file hashes do not match pinned Qwen3-0.6B")
        if smoke.get("train_data_sha256", "").lower() != args.train_sha256.lower():
            raise ValueError("GPU smoke used a different train dataset")
        smoke_steps = smoke.get("sft_smoke", {}).get("steps", 0)
        if not smoke.get("run_id") or not isinstance(smoke_steps, int) or not 20 <= smoke_steps <= 50:
            raise ValueError("GPU smoke must verify a 20–50-step train-only overfit run")
        smoke_evidence = smoke.get("sft_smoke", {})
        before_loss = smoke_evidence.get("toy_train_loss_before")
        after_loss = smoke_evidence.get("toy_train_loss_after")
        if (not isinstance(before_loss, (int, float)) or not math.isfinite(before_loss)
                or not isinstance(after_loss, (int, float)) or not math.isfinite(after_loss)
                or after_loss >= before_loss):
            raise ValueError("GPU smoke lacks evidence of decreasing 16-example train-only loss")
        if (smoke_evidence.get("lora_optimizer_steps_with_nonzero_grad") != smoke_steps
                or smoke_evidence.get("sampled_base_tensors_unchanged") is not True):
            raise ValueError("GPU smoke lacks nonzero LoRA-gradient and frozen-base evidence")
        gpu = smoke.get("gpu", {})
        gpu_name = gpu.get("name", "")
        if gpu.get("role") != "training" or "推理" in gpu_name or "inference" in gpu_name.casefold():
            raise ValueError("SFT smoke did not use an explicitly identified training GPU")
        if gpu.get("memory_free_before_bytes", 0) < 4 * 1024**3:
            raise ValueError("GPU smoke started with less than 4 GiB of free device memory")
        if smoke_evidence.get("memory_headroom_bytes", 0) < 2 * 1024**3:
            raise ValueError("GPU smoke failed to preserve 2 GiB of device-memory headroom")
        original_uuid = gpu.get("uuid")
        if not gpu.get("name") or not original_uuid:
            raise ValueError("GPU smoke lacks observed GPU identity")

        if worker.get("status") != "PASS" or worker.get("scope") != "SEPARATE_MTLS_GVISOR_WORKER_PROBE_ONLY":
            raise ValueError("separate mTLS worker probe did not pass")
        if worker.get("full_spec07_preflight") != "NOT_VERIFIED":
            raise ValueError("worker probe has an unexpected full-readiness claim")
        worker_inputs = worker.get("inputs", {})
        if (worker_inputs.get("train_jsonl_sha256", "").lower() != args.train_sha256.lower()
                or worker_inputs.get("probed_rows", 0) < 1):
            raise ValueError("worker probe input identity or observed row count mismatch")
        scores = worker.get("observed_scores", [])
        if len(scores) != worker_inputs["probed_rows"] or any(
                row.get("reference_reward") != 1.0 or row.get("buggy_reward") != 0.0
                or not row.get("reference_binding_match") or not row.get("buggy_binding_match")
                for row in scores):
            raise ValueError("worker probe expected rewards or frozen bindings failed")
        if not worker.get("tls", {}).get("server_certificate_verification") or not worker.get("tls", {}).get("client_certificate_present"):
            raise ValueError("worker probe lacks observed two-way TLS identity checks")
        smoke_sandbox = smoke.get("sandbox", {})
        if (smoke_sandbox.get("operator_pinned_sha256") != worker_hash
                or smoke_sandbox.get("policy_sha256") != worker_inputs.get("policy_sha256")
                or smoke_sandbox.get("worker_endpoint_host") != worker.get("worker_endpoint_host")):
            raise ValueError("GPU smoke is not pinned to this exact live worker probe/policy/host")

        if enforcement.get("status") != "PASS" or enforcement.get("scope") != "real-runsc-cpu-preflight":
            raise ValueError("actual runsc worker enforcement report is missing or not PASS")
        checks = enforcement.get("checks", [])
        if not checks or any(item.get("passed") is not True for item in checks):
            raise ValueError("worker enforcement report contains a failed check")
        check_ids = {item.get("id") for item in checks}
        if not REQUIRED_WORKER_CHECKS.issubset(check_ids):
            raise ValueError("worker enforcement report is missing required current preflight checks")
        if (enforcement.get("image_digest") != worker_inputs.get("sandbox_image_digest")
                or enforcement.get("policy_sha256") != worker_inputs.get("policy_sha256")):
            raise ValueError("worker probe and enforcing runtime image/policy bindings differ")
        if args.worker_host != worker.get("worker_endpoint_host"):
            raise ValueError("operator-confirmed worker host differs from live mTLS probe")
        if args.worker_id in {args.original_node_id, args.replacement_node_id}:
            raise ValueError("CPU worker identity must be distinct from original and replacement GPU nodes")

        original_root = Path(smoke.get("persistent_storage", {}).get("path", "")).resolve()
        if not original_root.is_dir():
            raise ValueError("original smoke persistent run root is unavailable")
        if smoke.get("persistent_storage", {}).get("node_id") != args.original_node_id:
            raise ValueError("operator-confirmed original node ID differs from smoke evidence")
        resume_path = Path(smoke.get("optimizer_rng_resume_path", ""))
        resume_hash = smoke.get("optimizer_rng_resume_sha256")
        if not resume_path.is_file() or not resume_hash or sha256(resume_path).lower() != resume_hash.lower():
            raise ValueError("original optimizer/RNG resume record missing or hash mismatch")
        resume = json.loads(resume_path.read_text(encoding="utf-8"))
        if (resume.get("status") != "PASS" or resume.get("seed") != 17
                or resume.get("train_data_sha256", "").lower() != args.train_sha256.lower()
                or resume.get("reference_gpu_uuid") != original_uuid
                or resume.get("provider_node_id") != args.original_node_id
                or smoke.get("optimizer_rng_resume") != resume):
            raise ValueError("original resume record does not match this seed/data/GPU smoke")

        if recovery.get("status") != "PASS" or recovery.get("scope") != "REPLACEMENT_RECOVERY_ONLY":
            raise ValueError("replacement recovery report is missing or not PASS")
        if recovery.get("original_gpu_uuid") != original_uuid:
            raise ValueError("replacement recovery parent GPU differs from original smoke")
        if (recovery.get("original_node_id") != args.original_node_id
                or recovery.get("replacement_node_id") != args.replacement_node_id):
            raise ValueError("recovery original/replacement node identities do not match operator inputs")
        replacement_uuid = recovery.get("replacement_gpu_uuid")
        if not replacement_uuid or replacement_uuid == original_uuid:
            raise ValueError("replacement recovery did not use a distinct GPU UUID")
        observed_gpus = subprocess.run(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
                                       check=True, capture_output=True, text=True).stdout.splitlines()
        if len(observed_gpus) != 1 or observed_gpus[0].strip() != replacement_uuid:
            raise ValueError("recovery GPU UUID differs from the live replacement node")
        if replacement_uuid != args.replacement_gpu_uuid:
            raise ValueError("operator-observed replacement GPU UUID differs from recovery report")
        if recovery.get("parent_resume_record_sha256") != resume_hash:
            raise ValueError("replacement recovery used a different optimizer/RNG resume record")
        original_state_hashes = resume.get("checkpoint_state_sha256", {})
        verified_state_hashes = recovery.get("checkpoint_state_sha256_verified", {})
        if verified_state_hashes != original_state_hashes:
            raise ValueError("replacement did not reverify the exact original checkpoint state")
        if recovery.get("resumed_step") != 3 or recovery.get("adapter_hashes_match") is not True:
            raise ValueError("replacement resume step or adapter agreement failed")
        loss_delta = recovery.get("loss_delta")
        loss_tolerance = recovery.get("loss_tolerance")
        if (not isinstance(loss_delta, (int, float)) or not math.isfinite(loss_delta) or loss_delta < 0
                or not isinstance(loss_tolerance, (int, float)) or not math.isfinite(loss_tolerance)
                or loss_tolerance <= 0 or loss_delta > loss_tolerance):
            raise ValueError("replacement next-step loss differs beyond tolerance")
        if args.replacement_node_id == args.original_node_id:
            raise ValueError("original and replacement node IDs must differ")

        run_root = args.run_root.resolve()
        mount = subprocess.run(["findmnt", "--noheadings", "--output", "SOURCE,FSTYPE",
                                "--target", str(run_root)], check=True, capture_output=True, text=True)
        fields = mount.stdout.strip().split()
        if len(fields) != 2 or fields[0] != args.expected_mount_source or fields[1] in {"overlay", "rootfs"}:
            raise ValueError("current findmnt does not verify the expected persistent volume")
        replacement_storage = recovery.get("replacement_persistent_storage", {})
        if (replacement_storage.get("mount_source") != fields[0]
                or replacement_storage.get("filesystem") != fields[1]
                or Path(replacement_storage.get("path", "")).resolve() != run_root
                or replacement_storage.get("node_id") != args.replacement_node_id):
            raise ValueError("recovery report mount identity differs from live replacement mount")
        checkpoint_path = Path(resume.get("checkpoint_path", "")).resolve()
        relative_checkpoint = checkpoint_path.relative_to(original_root)
        readback = run_root / relative_checkpoint
        readback_hashes = {}
        for name, expected in original_state_hashes.items():
            if name not in {"optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json",
                            "adapter_model.safetensors", "adapter_config.json"}:
                raise ValueError(f"unexpected recovery checkpoint file: {name}")
            current = readback / name
            if not current.is_file() or sha256(current).lower() != expected.lower():
                raise ValueError(f"persistent mount checkpoint readback mismatch: {name}")
            readback_hashes[str(current.resolve())] = expected
        if set(original_state_hashes) != {"optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json",
                                          "adapter_model.safetensors", "adapter_config.json"}:
            raise ValueError("original checkpoint evidence does not include all six expected state files")

        report = {
            "status": "PASS", "scope": "FULL_SPEC07_SFT_PREFLIGHT",
            "identity": {"run_id": smoke["run_id"], "report_sha256": smoke_hash,
                         "original_node_id": args.original_node_id, "replacement_node_id": args.replacement_node_id,
                         "original_gpu_uuid": original_uuid, "replacement_gpu_uuid": replacement_uuid},
            "model": model, "train_data_sha256": args.train_sha256.lower(),
            "sandbox": {"status": "PASS", "sandbox": "gvisor", "worker_id": args.worker_id,
                        "worker_host": args.worker_host, "probe_sha256": worker_hash,
                        "enforcement_sha256": enforcement_hash, "image_digest": enforcement["image_digest"],
                        "policy_sha256": enforcement["policy_sha256"]},
            "persistent_storage": {"path": str(run_root), "mount_source": fields[0],
                                   "volume_id": args.expected_volume_id, "node_id": args.replacement_node_id},
            "gpu": gpu, "sft_smoke": smoke["sft_smoke"],
            "checkpoint_recovery": {"status": "PASS", "fresh_process_optimizer_rng_resume": True,
                                     "checkpoint_id": str(checkpoint_path),
                                     "resume_record_sha256": resume_hash, "replacement_report_sha256": recovery_hash},
            "evidence": {"gpu_smoke_sha256": smoke_hash, "optimizer_rng_resume_sha256": resume_hash,
                         "worker_probe_sha256": worker_hash, "worker_enforcement_sha256": enforcement_hash,
                         "replacement_recovery_sha256": recovery_hash, "checkpoint_readback_files": readback_hashes},
            "assembled_at": datetime.now(timezone.utc).isoformat(),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": report["status"], "scope": report["scope"],
                          "output": str(args.output.resolve()), "sha256": sha256(args.output)}, indent=2))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "NOT_VERIFIED", "scope": "FULL_SPEC07_SFT_PREFLIGHT",
                          "reason": str(exc)}, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
