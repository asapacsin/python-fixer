# Python Fixer

**A 0.6B model for repairing Python functions, trained and evaluated with executable tests.**

Supervised fine-tuning raised mean validation pass@1 from **6.25% to 54.17%** across three independently trained LoRA adapters. A separate GRPO pilot on one SFT checkpoint showed a **4.17 percentage point** gain in its first round. Three more hours of GRPO did not improve on that round.

[Evaluation reports](reports/README.md) · [Model adapters](models/README.md) · [System design](specs/01-system-architecture.md)

## Performance

Pass@1 is the share of problems solved by the model's first proposed fix. A fix passes when it repairs the failing behavior and preserves the behavior that already passed its tests.

### Supervised fine-tuning vs. base model

Greedy decoding on the same **32 validation problems**, with one shared Qwen3-0.6B baseline and three independently trained SFT adapters. Each candidate had a 60-second wall-clock and 30-second CPU limit. All four models recorded zero timeouts under this policy.

| Model | Problems solved | Pass@1 |
| --- | ---: | ---: |
| Qwen3-0.6B base | 2/32 | 6.25% |
| SFT seed 17 | 18/32 | 56.25% |
| SFT seed 29 | 17/32 | 53.13% |
| SFT seed 43 | 17/32 | 53.13% |
| **SFT mean** | **17.33/32** | **54.17%** |

The mean gain was **47.92 percentage points**. Each SFT seed ran for 50 optimizer steps, with about five minutes of training per seed. [SFT report and timeout analysis](reports/evaluations/python-fixer-E1-three-seed-validation-20261126.md) · [Per-problem scores](reports/evaluations/python-fixer-E1-three-seed-validation-extended-20261126.json)

### GRPO vs. its SFT starting point

The three checkpoints below share one SFT seed 17 training lineage. Each was evaluated on the same 32 validation problems with three matched decoding seeds and a 15-second wall-clock and 5-second CPU limit. The decoding repeats measure sampling variation within this single training lineage.

| Checkpoint | Seed 17 | Seed 29 | Seed 43 | Mean pass@1 |
| --- | ---: | ---: | ---: | ---: |
| SFT seed 17 | 16/32 | 17/32 | 15/32 | 50.00% |
| **GRPO round 1, step 30** | **17/32** | **17/32** | **18/32** | **54.17%** |
| GRPO round 2, step 310 | 17/32 | 16/32 | 18/32 | 53.13% |

GRPO used four verified training examples. Round 1 had the highest observed mean, and the three-hour continuation finished **1.04 percentage points below** it. The final training logs recorded equal rewards across sampled candidates and zero gradient norm, consistent with saturation on those four examples. [Paired evaluation](reports/evaluations/seed17-sft-grpo-rounds-three-seed-summary-20261127.md) · [Continuation log](reports/evaluations/seed17-grpo-three-hour-continuation-20261127.md)

Both comparisons are exploratory validation results. The SFT and GRPO tables use different decoding settings and sandbox limits. The planned sealed test evaluation remains open.

## How it works

1. **Build repair cases.** Group related functions before splitting and verify that the buggy function fails the relevant tests and the reference fix passes. [Dataset design](specs/02-dataset-design.md)
2. **Train LoRA adapters.** Teach Qwen3-0.6B from verified fixes with supervised fine-tuning (SFT). In the GRPO pilot, compare multiple candidate fixes for each training prompt using rewards from test execution. [SFT](specs/03-sft-training.md) · [GRPO](specs/04-grpo-training.md)
3. **Run tests on a separate CPU worker.** Use gVisor to isolate candidate execution from the GPU trainer. Check both repaired behavior and regressions. Separate training rewards from validation scoring. [Reward design](specs/05-reward-engine.md) · [Evaluation protocol](specs/06-evaluation-protocol.md)
4. **Save evidence alongside the scores.** The repository includes per-problem results, run manifests, and adapter weights. The GRPO continuation also produced a verified resumable checkpoint. [Reports and run records](reports/README.md)

## Explore the artifacts

- [SFT seed 17](models/qwen3-0.6b-sft-e1-seed17) · [SFT seed 29](models/qwen3-0.6b-sft-e1-seed29) · [SFT seed 43](models/qwen3-0.6b-sft-e1-seed43)
- [Published GRPO round 2 adapter](models/qwen3-0.6b-grpo-round2-seed17)
- [Adapter loading notes and pinned base revision](models/README.md)
- [Experiment plan](specs/09-experiment-plan.md) · [Engineering roadmap](docs/implementation-roadmap.md)

### Local repository checks

With Python 3.11 or 3.12 in an activated virtual environment, run the following from the repository root. These checks cover repository contracts and CPU tooling.

~~~bash
python -m pip install -r requirements-dev.txt
python -m pip install --no-deps --no-build-isolation -e .
python scripts/validate_repository.py
python -m unittest discover -s tests -v
~~~
