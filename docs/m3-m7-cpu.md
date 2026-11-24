# M3-M7 CPU slice: what is done without a GPU

Status (2026-11-24): **CPU parts implemented and tested; no GPU, weights or model run.**
Contract: [m3-m7-cpu.json](../verification/contracts/m3-m7-cpu.json). Evidence:
`verification/evidence/m3-m7-cpu/` (SHA256SUMS). Tests use a tiny Qwen3-architecture
model with random weights and a locally built ChatML tokenizer
(`tests/training_fixtures.py`). They prove library wiring, not model quality or real Qwen3
tokenizer behaviour.

## Done (CPU)

| Roadmap item | Where | Evidence |
|---|---|---|
| M3 SFTTrainer/PEFT adapter, label masks, checkpoint/resume | `training/sft.py` | `test_m3_sft.py` (8 tests): prompt prefix, one EOS, masked prompt/pad labels on collated batches, only LoRA trainable, base hashes unchanged, finite falling toy loss, resume from step 3 equals uninterrupted step 6, adapter reload logits |
| M3 hash-locked environment | `requirements-train.lock` | torch 2.8.0, transformers 4.56.2, trl 0.23.1, peft 0.17.1 (64 packages) install with `--require-hashes` (Python 3.12, linux x86_64) |
| M4 reward callback, metadata/group alignment | `training/grpo.py` | `test_m4_grpo.py` (9 tests incl. num_iterations guard): order kept, invalid output -0.5, faults abort, mixed groups abort, degenerate-group ledger |
| M4 GRPOTrainer adapter, Dr. GRPO settings | `training/grpo.py` | equal rewards give no LoRA change at beta=0; mixed rewards change LoRA; base frozen |
| M4 real-sandbox integration (CPU) | `scripts/m4_cpu_smoke.py` | reference 1.0, buggy 0.0, malformed -0.5 through real runsc + mTLS; 2 GRPO updates; random model groups all-fail (zero advantage, as specified) |
| M5 budget, verified checkpoints, earliest-tie selection, manifests | `training/runs.py` | `test_m5_runs.py`: 8,000-candidate arithmetic, hard cap before scoring, partial/tampered checkpoints skipped |
| M6 data-access and contamination audits | `evaluation/leakage.py` | `test_m6_leakage.py`: frozen v7 clean; injected leaks detected |
| M7 checksum and doc-link checks | `scripts/check_artifacts.py` | all committed SHA256SUMS and relative links verify |
| GPU workload/cost method | [gpu-cost-estimate.md](gpu-cost-estimate.md) | counts exact; prices and throughput left to the GPU profile |

## Still needs a GPU or an owner decision

- Pin and download Qwen3-0.6B weights/tokenizer, test the real chat template and EOS.
- M2 E0/E0-P baseline generation; M3 overfit-16 and E1 seeds; M4 real preflight; M5 runs;
  M6 held-out scoring; M7 reproduction on a replacement node.
- A larger licensed dataset for a meaningful benchmark (v7 is a 67-case pipeline set).
- Separate CPU sandbox worker before any GPU trainer (ADR-002).
- Read-only LLM review of this slice: PASS, two non-blocking findings fixed (advisory, not independent per AGENTS rule 7).
