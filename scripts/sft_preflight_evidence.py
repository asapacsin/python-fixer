"""Probe an explicitly configured, separate mTLS gVisor reward worker.

This records only a bounded train-reference/buggy-candidate integration probe.
It does not test the full attack suite, checkpoint recovery, GPU identity,
persistent storage, or replacement-node behavior and never claims full readiness.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import ssl
import sys
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True, help="HTTPS mTLS endpoint of the separate CPU worker")
    parser.add_argument("--train-jsonl", type=Path, required=True, help="frozen train-only JSONL")
    parser.add_argument("--train-sha256", required=True)
    parser.add_argument("--manifest-sha256", required=True, help="frozen dataset manifest digest")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--policy", type=Path, default=ROOT / "configs/sandbox.json")
    parser.add_argument("--ca", type=Path, required=True)
    parser.add_argument("--certificate", type=Path, required=True)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="fresh report path outside Git")
    parser.add_argument("--count", type=int, default=3, choices=(1, 2, 3, 4))
    args = parser.parse_args(argv)

    try:
        from python_fixer.common import text_digest
        from python_fixer.data.models import load_problems
        from python_fixer.sandbox.client import RewardClient
        from python_fixer.sandbox.policy import Policy

        endpoint = urllib.parse.urlsplit(args.endpoint)
        if endpoint.scheme != "https" or not endpoint.hostname:
            raise ValueError("worker endpoint must be HTTPS with a hostname")
        for name, path in (("CA", args.ca), ("client certificate", args.certificate),
                           ("client key", args.key)):
            if not path.is_file():
                raise ValueError(f"{name} file is missing")
        if args.output.exists():
            raise ValueError("report output already exists")
        if not re.fullmatch(r"[0-9a-fA-F]{64}", args.train_sha256):
            raise ValueError("train SHA256 must be a hex digest")
        if not re.fullmatch(r"[0-9a-fA-F]{64}", args.manifest_sha256):
            raise ValueError("manifest SHA256 must be a hex digest")
        if sha256(args.train_jsonl).lower() != args.train_sha256.lower():
            raise ValueError("frozen train JSONL hash mismatch")
        if sha256(args.manifest).lower() != args.manifest_sha256.lower():
            raise ValueError("frozen dataset manifest hash mismatch")

        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        if manifest.get("stage") != "CPU_VERIFIED_PRE_TOKEN_FILTER" or manifest.get("split_counts", {}).get("test", 0) < 1:
            raise ValueError("dataset manifest is not the expected frozen source manifest")
        rows = load_problems(args.train_jsonl)
        if not rows or any(row.get("split") != "train" for row in rows):
            raise ValueError("worker probe accepts train rows only")
        policy = Policy.load(args.policy)
        policy_hash = policy.sha256
        image_digest = policy.image
        chosen = rows[:args.count]
        if len(chosen) != args.count:
            raise ValueError("not enough frozen training rows")

        bindings = {}
        for row in chosen:
            verified = row["verification"]
            if verified["image_digest"] != image_digest or verified["policy_sha256"] != policy_hash:
                raise ValueError("frozen row image/policy binding differs from current approved policy")
            bindings[row["problem_id"]] = {
                "suite_sha256": row["test_suite_sha256"],
                "image_digest": image_digest,
                "policy_sha256": policy_hash,
            }

        certificate_fingerprint = hashlib.sha256(args.certificate.read_bytes()).hexdigest()
        run_id = "sft-worker-probe-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        client = RewardClient(args.endpoint, ca=str(args.ca), certificate=str(args.certificate),
                              key=str(args.key), bindings=bindings, run_id=run_id)
        observed = []
        passed = True
        for row in chosen:
            ref_reward, ref = client.score(row["problem_id"], row["reference_code"])
            bug_reward, bug = client.score(row["problem_id"], row["buggy_code"])
            observed.append({
                "problem_id": row["problem_id"],
                "reference_reward": ref_reward,
                "buggy_reward": bug_reward,
                "reference_status": ref["status"],
                "buggy_status": bug["status"],
                "reference_wall_ms": ref.get("wall_ms"),
                "buggy_wall_ms": bug.get("wall_ms"),
                "reference_cpu_ms": ref.get("cpu_ms"),
                "buggy_cpu_ms": bug.get("cpu_ms"),
                "reference_binding_match": all(ref[k] == bindings[row["problem_id"]][key]
                                                for k, key in (("suite_sha256", "suite_sha256"),
                                                               ("image_digest", "image_digest"),
                                                               ("policy_sha256", "policy_sha256"))),
                "buggy_binding_match": all(bug[k] == bindings[row["problem_id"]][key]
                                            for k, key in (("suite_sha256", "suite_sha256"),
                                                           ("image_digest", "image_digest"),
                                                           ("policy_sha256", "policy_sha256"))),
            })
            passed &= ref_reward == 1.0 and bug_reward == 0.0
            passed &= observed[-1]["reference_binding_match"] and observed[-1]["buggy_binding_match"]

        context = ssl.create_default_context(cafile=str(args.ca))
        context.load_cert_chain(str(args.certificate), str(args.key))
        report = {
            "status": "PASS" if passed else "FAIL",
            "scope": "SEPARATE_MTLS_GVISOR_WORKER_PROBE_ONLY",
            "full_spec07_preflight": "NOT_VERIFIED",
            "probe_run_id": run_id,
            "observed_at": datetime.now(timezone.utc).isoformat(),
            "worker_endpoint_host": endpoint.hostname,
            "tls": {"minimum_version": "TLSv1.2", "server_certificate_verification": True,
                    "client_certificate_present": True, "client_certificate_sha256": certificate_fingerprint},
            "inputs": {"train_jsonl_sha256": args.train_sha256.lower(),
                       "manifest_sha256": args.manifest_sha256.lower(),
                       "manifest_path": str(args.manifest.resolve()), "policy_sha256": policy_hash,
                       "sandbox_image_digest": image_digest, "rows": len(rows), "probed_rows": len(chosen)},
            "observed_scores": observed,
            "not_verified": ["full worker attack suite", "GPU preflight", "persistent checkpoint storage",
                             "fresh-process optimizer/RNG resume", "replacement-node recovery"],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        report["report_sha256"] = sha256(args.output)
        print(json.dumps(report, indent=2))
        return 0 if passed else 1
    except Exception as exc:
        print(json.dumps({"status": "NOT_VERIFIED", "scope": "SEPARATE_MTLS_GVISOR_WORKER_PROBE_ONLY",
                          "full_spec07_preflight": "NOT_VERIFIED", "reason": str(exc)}, indent=2),
              file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
