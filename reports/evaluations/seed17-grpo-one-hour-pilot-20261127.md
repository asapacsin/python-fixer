# Seed 17 GRPO one-hour pilot

Run date: 2026-11-27 (Asia/Shanghai). This was an exploratory single-seed pilot, not the full 391-row training run.

## Same-policy SFT vs GRPO

Both adapters were evaluated greedily on the same 32 validation problems with the same 15 s / 5 CPU gVisor policy and the same generation script.

| Model | Solved | Pass@1 |
| --- | ---: | ---: |
| SFT seed 17 | 18/32 | 56.25% |
| GRPO seed 17 | 19/32 | 59.38% |

GRPO gained **1 problem out of 32 (+3.13 percentage points)** over SFT: 1 improved, 0 regressed. GRPO started from the SFT seed 17 adapter (`a9e7892efa7f3a74a4b32eee9375beed66ced49d9af9e8f314c7ca7438710b90`).

The earlier **2/32** figure was the unadapted base model under this 15 s / 5 CPU policy, not SFT. The earlier SFT **10/32** report used a different 5 s / 2 CPU policy; it is not directly comparable.

## Training record

- Completed 30 optimizer steps, resuming from step 15 after a transient reward-worker HTTP disconnect.
- Initial segment: 10:04:30–10:16:44 CST; stopped at step 15 with exit code 1 after the worker disconnect.
- Resumed segment: 503 seconds; completed at step 30 with exit code 0.
- Approximate active training time: 20 min 37 sec. The scheduled stop signal ran at 10:41:30 CST after training and GPU validation generation had finished. The container restarted; SSH still worked at 10:46 CST and the GPU was idle (0% utilization, 0 MiB allocated). Provider-level shutdown is unverified.
- All 392 LoRA adapter tensors changed during the resumed segment. The step 30 checkpoint manifest verified locally.
- GRPO adapter SHA-256: `b7cbc62020215f986af776bcbce2c0cc7469106c1c8e93b1d14f3d0f3c45fdad`.

The adapter and resumable checkpoint are on the training PC under the ignored `outputs/grpo-seed17-one-hour-20261127/` directory. They are intentionally kept out of Git. Machine-readable paired results and hashes are in the companion JSON and in `seed17-sft-vs-grpo-same-policy-20261127.json`.

This is an exploratory result: GRPO training used four verified train-only rows, a single seed, and 30 optimizer steps. It is not a full-data or multi-seed result.

