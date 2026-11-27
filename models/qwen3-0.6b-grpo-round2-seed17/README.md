# Qwen3-0.6B GRPO seed 17, round 2

This directory contains the final LoRA adapter from the three-hour GRPO continuation. It is an adapter, not a full copy of Qwen3-0.6B or a resumable optimizer checkpoint.

- Base: `Qwen/Qwen3-0.6B`, revision `c1899de289a04d12100db370d81485cdf75e47ca`
- Training lineage: SFT seed 17 → GRPO round 1, step 30 → three-hour GRPO continuation
- Final saved global step: 310 (trainer reached step 311 before the time limit)
- Adapter SHA-256: `c3353037b7061d81c5e7e49f286bfc43b5d88f05dfef7551dfe2da65d3c8c9a8`
- Evaluation: three stochastic decoding seeds on 32 validation cases; results in `reports/evaluations/seed17-sft-grpo-rounds-three-seed-summary-20261127.md`

Load the pinned base model with Transformers, then load this directory with PEFT. Do not use this adapter with a different base revision.
