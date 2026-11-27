# Seed 17 GRPO three-hour continuation

Status: RUNNING as of 2026-11-27 13:39 CST; trainer step 57. Checkpoints 35 and 50 are copied to the PC and all manifest hashes verified.

## Performance recorded before continuation

Fixed SFT seed 17 and GRPO seed 17 checkpoints were compared on the same 32 validation cases under the same 15 s / 5 CPU gVisor policy. Three paired stochastic decoding seeds produced:

| Sampling seed | SFT | GRPO | Difference |
| ---: | ---: | ---: | ---: |
| 17 | 16/32 (50.00%) | 17/32 (53.13%) | +1 |
| 29 | 17/32 (53.13%) | 17/32 (53.13%) | 0 |
| 43 | 15/32 (46.88%) | 18/32 (56.25%) | +3 |
| Mean | 16/32 (50.00%) | 17.33/32 (54.17%) | +1.33/32 (+4.17 pp) |

This is a small exploratory gain, not a confirmed improvement. Full per-problem evidence is in `seed17-sft-vs-grpo-three-sampling-repeats-20261127.json`.

## Continuation setup

- Process launched: 2026-11-27 13:20:17 CST. The training timer is three hours of optimizer runtime, starting after model and trainer setup.
- Resume source: step 30 checkpoint at `/.platform/state/python-fixer/grpo-runs/seed17-one-hour-20261127T100435/trainer/checkpoint-30`.
- Resume adapter SHA-256: `b7cbc62020215f986af776bcbce2c0cc7469106c1c8e93b1d14f3d0f3c45fdad`.
- Training scope: four verified train-only pilot rows, file SHA-256 `bfdb50465da27c200dcc48d89cde3ea7c1353be7bf0ea798d3cff1ee64ffeb2c`. The 391-row dataset has not completed re-verification, so this is not full-data GRPO.
- Worker preflight: mTLS reference score returned `OK`, reward 1.0.
- Settings: seed 17, same GRPO settings as the pilot, three-hour trainer time cap, maximum global step 500, checkpoint every five steps with manifest verification.
- Output: `/.platform/state/python-fixer/grpo-runs/seed17-three-hour-20261127T131732` on persistent platform storage. Final adapter and checkpoint will be copied to the PC after completion.

The continuation may overfit the four-row pilot; its post-run validation will be reported separately from the pre-run score.

- Local recovery checkpoint: outputs/grpo-seed17-three-hour-20261127/trainer/checkpoint-35/ (step 35; all manifest hashes verified).


- Local recovery checkpoint: outputs/grpo-seed17-three-hour-20261127/trainer/checkpoint-50/ (step 50; all manifest hashes verified).


- Local recovery checkpoint: outputs/grpo-seed17-three-hour-20261127/trainer/checkpoint-70/ (step 70; all 15 manifest SHA-256 entries verified locally).
- Status at 2026-11-27 13:53 CST: trainer at approximately step 79; later batches have returned reward mean 1.0, reward std 0, and gradient norm 0. Training remains active under the three-hour cap.

- Local recovery checkpoint: outputs/grpo-seed17-three-hour-20261127/trainer/checkpoint-80/ (step 80; all 15 manifest SHA-256 entries verified locally).
- Status at 2026-11-27 13:57 CST: trainer active at global step 85.

- Local recovery checkpoint: outputs/grpo-seed17-three-hour-20261127/trainer/checkpoint-90/ (step 90; all 15 manifest SHA-256 entries verified locally).
- Status: training process remains active after checkpoint 90.

- Local recovery checkpoint: outputs/grpo-seed17-three-hour-20261127/trainer/checkpoint-100/ (step 100; all 15 manifest SHA-256 entries verified locally).

- Local recovery checkpoint: outputs/grpo-seed17-three-hour-20261127/trainer/checkpoint-120/ (step 120; all 15 manifest SHA-256 entries verified locally).
- Status at roughly 2026-11-27 14:21 CST: 122/500 steps; trainer elapsed about 59:37 and process is active.

- Local recovery checkpoint: outputs/grpo-seed17-three-hour-20261127/trainer/checkpoint-140/ (step 140; all 15 manifest SHA-256 entries verified locally).
- Status: trainer still active; step 140 was reached at approximately 72 minutes of runtime.

- Local recovery checkpoint: outputs/grpo-seed17-three-hour-20261127/trainer/checkpoint-160/ (step 160; all 15 manifest SHA-256 entries verified locally).
- Status: process active at approximately 86 minutes of trainer runtime.

- Local recovery checkpoint: outputs/grpo-seed17-three-hour-20261127/trainer/checkpoint-180/ (step 180; all 15 manifest SHA-256 entries verified locally).
- Status at approximately 2026-11-27 14:59 CST: trainer active at step 180, elapsed about 98 minutes.

- Local recovery checkpoint: outputs/grpo-seed17-three-hour-20261127/trainer/checkpoint-200/ (step 200; all 15 manifest SHA-256 entries verified locally).
- Status: trainer active at step 206 and about 115 minutes elapsed.

- Local recovery checkpoint: outputs/grpo-seed17-three-hour-20261127/trainer/checkpoint-220/ (step 220; all 15 manifest SHA-256 entries verified locally).
- Status at approximately 2026-11-27 15:27 CST: trainer active, elapsed about 126 minutes.

- Local recovery checkpoint: outputs/grpo-seed17-three-hour-20261127/trainer/checkpoint-240/ (step 240; all 15 manifest SHA-256 entries verified locally).
- Status: trainer active at step 248 and approximately 141 minutes elapsed.

- Local recovery checkpoint: outputs/grpo-seed17-three-hour-20261127/trainer/checkpoint-260/ (step 260; all 15 manifest SHA-256 entries verified locally).
- Status: trainer active at approximately step 268.

- Local recovery checkpoint: outputs/grpo-seed17-three-hour-20261127/trainer/checkpoint-280/ (step 280; all 15 manifest SHA-256 entries verified locally).
- Status: trainer active at approximately 161 minutes elapsed.

## Completion

- Trainer exit code: 0; three-hour timer completed at 2026-11-27 16:21:36 CST.
- Final resumable checkpoint copied to outputs/grpo-seed17-three-hour-20261127/trainer/checkpoint-310/; checkpoint manifest verified for all 15 files.
- Final adapter SHA-256: c3353037b7061d81c5e7e49f286bfc43b5d88f05dfef7551dfe2da65d3c8c9a8.
- Training reached step 311 before stopping; latest saved checkpoint is step 310.
- At the last saved steps, reward mean was 1.0, reward standard deviation 0, and gradient norm 0. The four-row pilot had saturated; this three-hour run does not establish a broader accuracy gain.
- Post-continuation three-seed validation is complete. Results and the round-by-round comparison are in [seed17-sft-grpo-rounds-three-seed-summary-20261127.md](seed17-sft-grpo-rounds-three-seed-summary-20261127.md).
