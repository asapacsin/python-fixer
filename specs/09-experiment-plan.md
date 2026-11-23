# 09 — Experiment schedule and preregistration

## Frozen core comparison

| Arm | Initialization | Updates | Data | Selection |
|---|---|---|---|---|
| E0 | pinned Qwen3-0.6B | 0 | no training | fixed base |
| E0-P | same base | 0 | fixed stronger prompt | no tuning on test |
| E1 | same base + new LoRA | 2 SFT epochs | verified SFT train | best validation, earliest tie |
| E2 | selected matching E1 adapter | 500 GRPO steps | frozen RL train | best validation, earliest tie |
| E3 | same base + new LoRA | 500 identical GRPO steps | same RL train order | same rule |

Seeds 17,29,43 for trained arms. E2 and E3 use same RL prompts, grouping,
temperature, reward version, max completion length, batch shape and maximum
generated candidates. At 16 completions/update and 500 updates the planned
budget is 8,000 candidates per seed, excluding labeled preflight and validation.
Actual trainer counts must confirm this arithmetic; record repeated/resumed
generations, variable EOS tokens, failed groups and retries. Equal update count
does not imply equal GPU time/tokens; publish those separately. E2 includes the
extra SFT compute, so E2-E3 alone does not establish equal-total-cost superiority.

## Execution order

1. M1: freeze source licensing, family split, tests, schemas and sandbox evidence.
2. M2: implement evaluation, estimate power from validation, seal test protocol;
   generate E0/E0-P candidates once locked. Do not inspect sealed comparative
   outcomes until model choices are final; an evaluator can retain raw artifacts.
3. M3: run SFT preflight then E1 seeds, choose checkpoints using validation only.
4. M4: implement reward adapter/GRPO integration; pass CPU security tests and
   real GPU preflight. Freeze RL settings after validation-only pilot.
5. M5: execute paired E2/E3 seeds; keep failed/interrupted runs in the ledger.
6. M6: independent evaluator scores all frozen candidates/models together;
   calculate contrasts, confidence intervals, subgroup errors and audit leakage.
7. M7: reproduce one run from a replacement node and package report/artifact index.

## Pilot, ablations and spending hierarchy

Run the cheapest falsification first: M1 synthetic fixture through real sandbox;
M2 base on validation subset; M3 overfit 16 train pairs; M4 two-prompt reward
group and short updates. Stop full-study launch if these fail. These are smoke
tests, not evidence of model improvement.

Primary study has priority over optional ablations. After measuring costs and
obtaining spending authorization, consider G=8 vs4, binary vs partial reward,
scale_rewards=none vs group, beta=.01 vs0 and QLoRA vs BF16 LoRA. Change one factor
at a time, maintain fixed validation tuning budget, and label exploratory results.
Qwen3-1.7B is a separate size-scaling study with its own preflight and budget;
do not mix its scores into the 0.6B primary contrasts.

## Decision thresholds

Use [evaluation protocol](06-evaluation-protocol.md) exactly: >=5 percentage-point
absolute gain, positive simultaneous lower interval, corrected statistical test
and <=2-point regression increase for meaningful primary improvement. A result
may show SFT gains but no RL gains, RL regressions, or advantage of prompting;
all are reportable. Do not extend steps until a desired test score appears.

Matched RL-budget analysis tests warm-up effects. Equal-total-cost sensitivity
uses validation learning curves at a predeclared measured compute/time budget
and explicitly labels any unmatched candidate/token counts. Do not claim a
causal SFT benefit from uncontrolled extra compute alone.

## Required experiment manifest

Before launch: study/run/arm/seed IDs, hypothesis, immutable source and environment
lock, model/tokenizer/parent adapter hashes, data split/test manifest, prompt and
parser hashes, reward/sandbox versions, complete trainer arguments, max steps,
candidate/token/time/spend budget, validation selection rule and operator authority.
After run: completed steps/candidates/tokens, losses and reward traces, status,
stop reason, actual hardware/rates/costs, checkpoints and hashes, resume history,
selected checkpoint rationale and evaluator artifacts. Missing fields prevent
COMPLETED/EVALUATED status; unknown costs remain unknown, never invented.

## Report outline

Problem and scope; related methods and source citations; data/license/attrition;
security and reward protocol; four-arm settings and cost; main confidence intervals;
prompt baseline; category/difficulty errors; zero-variance/stability diagnostics;
negative results; contamination and power limits; exact reproduction instructions;
artifact index with restricted/private access distinctions. No expected result
table is prepopulated with numeric performance. See [reports policy](../reports/README.md).
