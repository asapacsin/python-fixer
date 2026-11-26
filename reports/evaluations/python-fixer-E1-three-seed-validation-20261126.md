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

Under the strict limit, the descriptive SFT mean was 7/32 (range 4–10). The baseline was re-executed separately for each seed, and its score varied between 1 and 2 solved; the relaxed comparison below instead uses one shared baseline pass.

## Rerun with a longer checker limit

The earlier checker stopped a candidate after 5 seconds wall time or 2 seconds CPU time. The separate validation profile raises this to 15 seconds / 5 CPU seconds; the normal sandbox profile remains unchanged. Isolation preflight passed 35/35 checks under the updated profile.

All four arms were scored in one run against one shared baseline pass:

| Arm | Solved | Timeouts | Improved over baseline | Regressed |
|---|---:|---:|---:|---:|
| Baseline | 2/32 | 0 | — | — |
| Seed 17 | 15/32 | 4 | 13 | 0 |
| Seed 29 | 15/32 | 2 | 13 | 0 |
| Seed 43 | 16/32 | 1 | 14 | 0 |

The seed SFT mean is 15.3/32 solved (range 15–16). Compared with the earlier strict-limit runs, SFT timeouts fell from 14/17/25 to 4/2/1. So the short checker limit was a major cause of the earlier failures. Remaining failures include wrong answers as well as a few timeouts.

This remains exploratory validation, not held-out test evidence. Details and the exact policy/preflight hashes are in the relaxed evaluation JSON and preflight report. The H200 was not used for this rerun.

## Extended checker limit

For a final timeout check, all arms were scored again at 60 seconds wall time / 30 CPU seconds, with the same shared baseline. Sandbox isolation preflight again passed 35/35 checks.

| Arm | Solved | Timeouts | Wins over baseline | Regressions |
|---|---:|---:|---:|---:|
| Baseline | 2/32 | 0 | — | — |
| Seed 17 | 18/32 | 0 | 16 | 0 |
| Seed 29 | 17/32 | 0 | 15 | 0 |
| Seed 43 | 17/32 | 0 | 15 | 0 |

No answer hit the 60-second / 30-CPU-second cap. The extra time let a few more answers pass than the 15/5 profile. Some answers still fail because their outputs do not pass the tests. This is still exploratory validation, not a held-out test result. The H200 was not used.
