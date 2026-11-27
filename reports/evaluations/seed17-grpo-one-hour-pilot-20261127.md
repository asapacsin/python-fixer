# Seed 17 GRPO one-hour pilot

Run date: 2026-11-27 (Asia/Shanghai). This was an exploratory single-seed pilot, not the full 391-row training run.

## Result

Paired greedy validation on the same 32 validation problems and isolated 15 s / 5 CPU test policy:

| Arm | Solved | Pass@1 |
| --- | ---: | ---: |
| SFT seed 17 baseline | 2/32 | 6.25% |
| GRPO seed 17 | 19/32 | 59.38% |

GRPO improved 17 problems and regressed 0. This is a promising single run, not a generalization claim: the policy was trained on only four verified train rows, with one seed and one greedy completion per validation problem. The run reward saturated often, so train reward alone was not used as the performance measure.

## Training record

- Completed 30 optimizer steps, resuming from step 15 after a transient reward-worker HTTP disconnect.
- Initial segment: 10:04:30–10:16:44 CST; stopped at step 15 with exit code 1 after the worker disconnect.
- Resumed segment: 503 seconds; completed at step 30 with exit code 0.
- Approximate active training time: 20 min 37 sec. The scheduled stop signal ran at 10:41:30 CST after training and GPU validation generation had finished. The timed signal restarted the container; SSH still worked at 10:46 CST and the GPU was idle (0% utilization, 0 MiB allocated). Provider-level shutdown is unverified.
- All 392 LoRA adapter tensors changed during the resumed segment. The step 30 checkpoint manifest verified locally.
- Adapter SHA-256: `b7cbc62020215f986af776bcbce2c0cc7469106c1c8e93b1d14f3d0f3c45fdad`.

The adapter and resumable checkpoint are on the training PC under the ignored `outputs/grpo-seed17-one-hour-20261127/` directory. They are intentionally kept out of Git. Machine-readable paired results and hashes are in the companion JSON.

Validation is exploratory. Run a broader, multi-seed evaluation on the full verified train split before treating this as a stable estimate.



