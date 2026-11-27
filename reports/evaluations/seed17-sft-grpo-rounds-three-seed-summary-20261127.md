# SFT → GRPO round 1 → GRPO round 2: three-seed validation

Date: 2026-11-27 (Asia/Shanghai). The same 32 validation problems were sampled with decoding seeds 17, 29, and 43 for each checkpoint. All runs used temperature 0.7, top-p 0.95, top-k 0, a 512-token cap, and the same gVisor policy (15 seconds / 5 CPU seconds). These are three stochastic decoding repeats of the same SFT and GRPO training lineage, not three independently trained model seeds.

| Sampling seed | SFT | GRPO round 1 (step 30) | GRPO round 2 (3-hour continuation; saved step 310) |
| ---: | ---: | ---: | ---: |
| 17 | 16/32 (50.00%) | 17/32 (53.13%) | 17/32 (53.13%) |
| 29 | 17/32 (53.13%) | 17/32 (53.13%) | 16/32 (50.00%) |
| 43 | 15/32 (46.88%) | 18/32 (56.25%) | 18/32 (56.25%) |
| **Mean** | **16.00/32 (50.00%)** | **17.33/32 (54.17%)** | **17.00/32 (53.13%)** |

## Readout

- Round 1 vs SFT: +1.33 solved problems on average (+4.17 percentage points).
- Round 2 vs SFT: +1.00 solved problem on average (+3.13 percentage points).
- Round 2 vs round 1: −0.33 solved problem on average (−1.04 percentage points). The second training round did not improve the mean validation score over round 1.
- Round 2 paired differences vs SFT by sampling seed were +1, −1, and +3 problems. This is a small exploratory validation comparison, not evidence of a reliable generalization gain.

## Scope and evidence

- Round 1 was the 30-step GRPO pilot; round 2 resumed its step-30 checkpoint and ran for three hours, finishing at global step 311 with the last resumable checkpoint at step 310.
- Both GRPO rounds used only four verified train-only pilot rows, not the full 391-row training set. At the end of round 2, reward mean was 1.0 with zero reward variance and zero gradient norm, indicating saturation on that small pilot.
- Validation scoring used the same dataset SHA-256 `5cc7e23ba5ae87b14a5e0eaf0e1e5ea89fb6c572b1d36e4d20b3fee543015578`, policy SHA-256 `700ee53009dc39ac2b80b4faa43db7bf874c73e2fa4e578f591ff3996569ad3a`, and pinned gVisor image `python@sha256:07d4824036af339fa1b6ef769cc99d9c10a141769538b52b3b597c84f7a7b535`.
- Detailed round-2 paired results are in `reports/evaluations/seed17-grpo-round2-seed{17,29,43}-paired-20261127.json`. The final LoRA adapter is in `models/qwen3-0.6b-grpo-round2-seed17/`; the full resumable checkpoint is saved locally under `outputs/grpo-seed17-three-hour-20261127/trainer/checkpoint-310/`.
