# 07 — Disposable GPU operations and recovery

Workflow: GitHub revision -> disposable Linux GPU node -> staged frozen dataset ->
training -> external persistent checkpoints -> sealed evaluation -> report.
The node is replaceable; no unique artifact may live only on its root disk.

## Environment identity and compatibility

Target RTX 4090 24 GB, Linux x86_64, Python 3.11. Proposed version tuple is
[training-candidate.json](../configs/training-candidate.json). Pin all transitive
packages and wheel hashes during implementation; record Python build, OS image
digest, package lock, torch/CUDA/cuDNN versions and driver in every run. The
candidate stack was selected from versioned APIs, not resolved/installed here.
M0 tooling runs on Python 3.12 without ML dependencies.

For proposed cu128 wheels, prefer a driver meeting CUDA 12.8 native requirements
(Linux >=570.26); verify against the exact NVIDIA release notes and provider.
Do not assume system `nvcc` equals PyTorch's runtime CUDA. Preflight checks
`nvidia-smi`, torch.cuda availability/device capability, BF16 support, a CUDA
matrix operation, available memory and driver/runtime compatibility. Install
trusted signed/checksummed artifacts with TLS validation. No insecure-index,
disabled certificate checks or arbitrary downloaded install scripts.

The GPU node never gets a Docker socket or permission to run candidate code.
Provisioning a separate gVisor CPU worker and approving spend are outside M0.
Storage credentials are least privilege to the current run prefix; held-out
evaluator identity is separate. Inspect existing bindings by presence only before
requesting missing credentials. Never write secrets into configs/commands/logs.

## Storage and version pins

Nonsecret runtime paths: `PF_RUN_ROOT` (mounted persistent filesystem),
`PF_DATA_ROOT` (read-only train/validation manifests), `HF_HOME` (external cache),
`PF_SANDBOX_ENDPOINT` (authenticated gateway), `PF_STUDY_ID`. Provider-specific
secret bindings are configured only after provider selection; no dummy token.
Check device/mount identity and write/read-back persistence before loading models.
`/tmp`, root disk and a local directory named persistent do not prove persistence.

Hub acquisition records resolved repository SHA, tokenizer SHA and every file hash.
Dataset acquisition uses approved licenses and immutable source revisions.
Never use `revision=main` in a paid run. Checkpoint names are immutable
`<study>/<arm>/<seed>/<run_id>/step-N/`. Manifest includes input/config/code hashes,
parent adapter, completed steps, token/candidate counts, RNG state, software and
hardware. Base weights are referenced by SHA; do not duplicate them per adapter.

Write checkpoint to a staging directory, fsync where supported, compute hashes,
upload/sync objects, read back and verify, then atomically publish COMMITTED.json.
Object stores do not have POSIX rename; publish a final small manifest after all
objects exist. LATEST points only to committed verified checkpoints. Retain at
least the newest two committed checkpoints plus selected best. Never delete the
last good checkpoint before verifying its replacement.

## Preflight before full training

Implement `scripts/gpu_preflight.py` in M3; until then training is blocked.
Before expensive steps, require a budget cap, pinned manifests and working CPU
sandbox. The preflight report must contain actual assertions and artifact paths:

1. Versions/GPU/BF16 and storage write/read-back checks; resource headroom.
2. Pinned model/tokenizer load, non-thinking template/mask/EOS test and short
   generation. Generation returns a candidate string without host execution.
3. 16 verified train examples through the real collator; no hidden test fields.
4. Real remote sandbox reference-pass/buggy-fail and attack suite; score known
   outcomes with reward v1. Check network/secret isolation and resource cleanup.
5. 2–5 SFT or GRPO updates for the selected stage; finite gradients/weight changes,
   reward grouping for GRPO. M3 may run SFT stage; M4 must add GRPO stage before RL.
6. Save full optimizer/RNG checkpoint, terminate process, reload on a new process
   and verify next-step loss/logits within documented precision tolerances.
7. Synchronize manifest, read hashes back through persistent storage, then recover
   once on a replacement node before trusting long runs.

A zero exit without expected step counts, changed weights, complete logs and
verified artifacts is FAIL. Preflight GPU memory target is <22 GiB allocated peak
with no OOM; record reserved and device-wide usage too. End-to-end sandbox
concurrency/latency must keep the 16-candidate window within the operator-approved
step-time budget, initially <=120 seconds on 16 small problems (estimate to test).

## Logging, budget and shutdown

Append JSONL metrics locally on persistent storage; optional external experiment
tracker only with permission and no private test payloads. Run manifest has start,
end, interrupted duration, provider GPU billing hours, sandbox CPU hours, prices
and currency/date. Estimate cost as `GPU_hours*observed_rate + CPU + storage +
egress`; do not invent provider prices. Profile 50 steps, extrapolate planned
tokens/steps, then obtain authorized budget before a full run. Stop/checkpoint
before reaching the budget cap; maintain explicit cleanup ownership.

On interruption: terminate generation safely, cancel sandbox requests, save
checkpoint if possible and mark INTERRUPTED; if save fails, retain last committed
checkpoint. Resume validates exact data/config/parent hashes and restores RNG,
optimizer and sampler. Config changes create a new run linked to its predecessor.

Shutdown: stop scheduling; finish/cancel in-flight requests; commit checkpoint;
verify storage manifest and metrics; record remaining work and last good step;
revoke node-scoped credentials; stop the paid instance and verify provider billing
state. Do not shut down an unrelated user's instance or delete persistent storage.
