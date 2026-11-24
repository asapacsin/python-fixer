# GPU work: what it needs and how to price it (for operator approval)

Status (2026-11-24): **estimate of workload, not of money.** Spec 07 forbids inventing
provider prices or throughput, so dollar figures are left to the operator's quote and
the 50-step GPU profile. Nothing here is authorized yet (AGENTS rule 13).

## Workload from frozen configs (exact arithmetic)

| Arm | Seeds | Work per seed | Total |
|---|---|---|---|
| E0, E0-P (baseline eval) | 1 | greedy + sampled on validation/test | small |
| E1 SFT | 3 | v9: 391 train rows x 2 epochs = 782 examples (~49 updates at 16/update) | ~147 updates |
| E2 GRPO | 3 | 500 updates x 16 candidates = 8,000 candidates, <=512 new tokens each | 24,000 candidates |
| E3 GRPO | 3 | same as E2 | 24,000 candidates |

RL generation upper bound: 48,000 candidates x 512 tokens = 24.6M generated tokens
(plus prompts <=1,536 tokens). `runs.planned_candidates` reproduces the 8,000 figure
from `configs/grpo.json`, and `runs.CandidateBudget` enforces it.

v9 (299 authored functions, 62 held-out) supersedes the 67-case v7 set. Held-out power
is still limited, so it should be estimated from validation in M2. The rest of this note was written for v7:
E1 and the evaluation arms there were too small to be a
meaningful benchmark (spec 02). A research run needs the larger licensed dataset first.
That is CPU/data work, but it needs a source licensing decision from the owner.

## Measured inputs available now

- Sandbox scoring: one full suite on real runsc took **680-800 ms CPU** in this session
  (5 repeated runs). For 48,000 RL candidates that is about **9-11 CPU-hours** on the
  separate worker, before parallelism. Measured here, not on the target worker.
- CPU training smoke: tiny random model only, so no GPU throughput can be inferred from it.

## Unknown until the 50-step GPU profile (spec 07)

- Generated tokens/s for Qwen3-0.6B BF16 with HF generate (no vLLM) on the chosen GPU.
- Peak memory (gate: <22 GiB allocated on a 24 GB card).
- Provider hourly rate, storage and egress.

Cost formula (spec 07): `GPU_hours * observed_rate + sandbox_CPU_hours * rate + storage + egress`,
where `GPU_hours = generated_tokens / measured_tokens_per_second / 3600` plus training,
evaluation and preflight time.

## Proposed approval steps

1. Approve one small GPU node (24 GB class) for the **preflight only**: pin and download
   Qwen3-0.6B weights and tokenizer by commit, install `requirements-train.lock`, run
   the M3 overfit-16 test and a 50-step GRPO profile against the separate sandbox worker.
2. From that profile, compute the full-study GPU hours and price. Approve or reject the
   full study separately.
3. Separate CPU worker machine for the sandbox (ADR-002) before any GPU trainer runs.
