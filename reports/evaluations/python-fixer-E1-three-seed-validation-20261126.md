# Qwen3-0.6B three-seed SFT comparison

Exploratory greedy evaluation on the same 32-row validation split. The held-out test split was not used. All three SFT runs completed 50 steps with the same pinned model, data, and training configuration.

| Seed | Train loss | SFT time | Baseline solved | SFT solved | Improved | Regressed | Baseline timeouts | SFT timeouts |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 17 | 0.07285 | 291.4s | 1/32 | 10/32 | 9 | 0 | 6 | 14 |
| 29 | 0.07606 | 297.1s | 2/32 | 7/32 | 5 | 0 | 5 | 17 |
| 43 | 0.07079 | 295.0s | 1/32 | 4/32 | 3 | 0 | 9 | 25 |

Adapters are saved locally under outputs/sft-e1-seed17, outputs/sft-e1-seed29, and outputs/sft-e1-seed43. Each run manifest, metric log, adapter SHA-256, and detailed per-problem score are recorded in this repository.

## Interpretation

These are small exploratory validation results. The sandbox produced many timeouts, especially for seed 43, so pass counts are provisional and should not be treated as a stable estimate of generalization. No claim is made from training loss alone. A follow-up should first stabilize CPU sandbox timing, then rerun the same fixed candidate set before using a held-out test set.

Provider billing was not accessible from the training host, so actual spend was not verified. No GRPO run was performed.

Descriptive mean SFT solved count: 7/32 (range 4–10 across seeds). The baseline counts also varied slightly across scoring runs (1–2/32), which reinforces that timeout-related results are provisional.
