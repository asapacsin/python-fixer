# Evaluation and run records

| Report | What it covers |
| --- | --- |
| [Base model vs. three SFT seeds](evaluations/python-fixer-E1-three-seed-validation-20261126.md) | Greedy validation results and the effect of the sandbox time limit |
| [SFT vs. both GRPO rounds](evaluations/seed17-sft-grpo-rounds-three-seed-summary-20261127.md) | Paired validation results on 32 problems with three matched decoding seeds |
| [SFT vs. GRPO round 1](evaluations/seed17-sft-vs-grpo-three-sampling-repeats-20261127.md) | Per-seed comparisons and uncertainty for the first-round gain |
| [Three-hour GRPO continuation](evaluations/seed17-grpo-three-hour-continuation-20261127.md) | Training conditions, checkpoint recovery, reward saturation, and final adapter hash |
| [Training run records](runs/) | SFT run manifests and training metrics |

The results above are exploratory validation. Use the exact denominators, identify the model and scoring policy, and distinguish training seeds from decoding seeds when reporting them. Preserve inconclusive and interrupted runs and record measured costs when available. Full resumable checkpoints, private tests, and candidate payloads belong in controlled storage. See the [evaluation protocol](../specs/06-evaluation-protocol.md) and [published adapters](../models/README.md).
