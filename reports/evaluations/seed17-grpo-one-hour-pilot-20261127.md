# Seed 17 GRPO one-hour pilot

Run date: 2026-11-27 (Asia/Shanghai). This was an exploratory single-seed pilot, not the full 391-row training run.

## Model lineage and result

GRPO was initialized from the **SFT seed 17 adapter** (`a9e7892efa7f3a74a4b32eee9375beed66ced49d9af9e8f314c7ca7438710b90`). The 2/32 arm below was the **unadapted Qwen3-0.6B base model**, not the SFT model.

| Arm | Solved | Pass@1 |
| --- | ---: | ---: |
| Unadapted Qwen3-0.6B base | 2/32 | 6.25% |
| GRPO seed 17 (initialized from SFT seed 17) | 19/32 | 59.38% |

This is not a direct GRPO-versus-SFT comparison. The earlier SFT seed 17 report showed 10/32 (31.25%) using a 5 s / 2 CPU scoring policy; this pilot used 15 s / 5 CPU. Those scores are not directly comparable. The current result shows GRPO's outcome against the unadapted base under this pilot policy; it does not isolate the improvement from SFT versus the additional GRPO updates.

GRPO improved 17 base-model cases and regressed 0. This is an exploratory single-seed result: training used only four verified rows, with one greedy completion per validation problem. Training reward saturated often, so train reward alone was not used as the performance measure.

## Training record

- Completed 30 optimizer steps, resuming from step 15 after a transient reward-worker HTTP disconnect.
- Initial segment: 10:04:30–10:16:44 CST; stopped at step 15 with exit code 1 after the worker disconnect.
- Resumed segment: 503 seconds; completed at step 30 with exit code 0.
- Approximate active training time: 20 min 37 sec. The scheduled stop signal ran at 10:41:30 CST after training and GPU validation generation had finished. The container restarted; SSH still worked at 10:46 CST and the GPU was idle (0% utilization, 0 MiB allocated). Provider-level shutdown is unverified.
- All 392 LoRA adapter tensors changed during the resumed segment. The step 30 checkpoint manifest verified locally.
- GRPO adapter SHA-256: `b7cbc62020215f986af776bcbce2c0cc7469106c1c8e93b1d14f3d0f3c45fdad`.

The adapter and resumable checkpoint are on the training PC under the ignored `outputs/grpo-seed17-one-hour-20261127/` directory. They are intentionally kept out of Git. Machine-readable paired results and hashes are in the companion JSON.

Run a broader, multi-seed evaluation and score the SFT and GRPO adapters under the exact same policy before claiming a direct improvement over SFT.

