# 06 — Frozen evaluation and statistical claims

## Seal before training

Freeze dataset/model/tokenizer/prompt/parser/sandbox hashes, generation settings,
comparison plan and success thresholds before E0. Test bundle and references
are accessible only to an independent evaluator identity. Training logs never
contain test code or feedback. Use validation for checkpoint selection/hyperparameters;
test evaluation occurs only after all arm configurations are locked. No test-time
repair loop, self-consistency selection or choosing best candidate using test
feedback in pass@1. Repair failures are zeroes, not exclusions.

## Inference conditions

Primary held-out repair pass@1: one **greedy** candidate per problem, no sampling,
non-thinking mode, same raw-code template, max_new_tokens=512, batch size fixed
after preflight (start 1), same BF16 model path, same EOS/parser and hardware.
Vendor recommends stochastic non-thinking decoding; greedy here is a deliberate
controlled primary protocol, not a vendor recommendation. Include a secondary
sampled comparison so the conclusion is not specific to greedy decoding. Failed
compilation, invalid output, timeouts and any failing required hidden test count
as failure. Never retry a candidate because its score was poor. Infrastructure
failure retries the same candidate; unresolved outage makes the run incomplete.

Secondary sampling: n=20 independent completions/problem for k=1,5,10 at
temperature .7, top_p=.8, top_k=20, max_new_tokens=512; seeds 101/202/303.
Store every outcome; for c successes among n compute the unbiased estimator
`pass@k = 1 - C(n-c,k)/C(n,k)`, equal to 1 if n-c<k. Compute stably through products,
not factorial overflow. Report sampled pass@1 separately from primary greedy
pass@1. Repeated sampling seeds quantify inference variation, not independent
training runs or extra independent problems.

E0-P uses [stronger.txt](../configs/prompts/stronger.txt), a fixed diagnostic
checklist within the same total context and 512 completion-token budget. No
few-shot answers from evaluation families. Other arms keep the original prompt.
Report this separately; a gain over E0 alone may be attributable to prompting.

## Metrics and result records

Report macro problem pass@1; family-weighted sensitivity; pass@k; compilation rate;
invalid/timeout/policy rates; regression rate (fraction of candidates failing at
least one previously passing test, plus per-test regression fraction); breakdowns
by fixed bug category/difficulty with counts; generated tokens; inference
latency median/p95; training GPU hours and wall time; sandbox CPU hours; measured
cloud/storage/egress cost and total cost per additional correctly repaired problem.
If denominator <=0, cost per gain is undefined, not zero.

Latency includes tokenize/generate/decode after one excluded warm-up, with CUDA
synchronization around timings; separately report sandbox and end-to-end latency.
Record device, driver, precision, batch size, GPU utilization and concurrent load.
No comparison across changed hardware without explicit qualification.

Candidate JSONL has `study_id`, `arm`, `train_seed` (null E0/E0-P), `checkpoint_sha256`,
`model_revision`, `tokenizer_sha256`, `dataset_manifest_sha256`, `prompt_sha256`,
`generation_config_sha256`, `problem_id`, `source_family_id`, `sampling_seed`,
`sample_index`, `candidate_sha256`, `finish_reason`, `generated_tokens`,
`generation_ms`, `sandbox_report_sha256`, `compiled`, `all_tests_pass`,
`regression`, `status`. Candidate source and private test outcomes reside in
restricted artifacts, not public Git logs. Aggregate JSON includes exact
denominators, inclusion/exclusion ledger, missing records, confidence method,
cost ledger hashes and software/hardware manifests. A validator rejects duplicate,
missing or unrun attempts before computing metrics.
The per-candidate shape is [evaluation-result.schema.json](../schemas/evaluation-result.schema.json).

## Repetition, uncertainty and success

Three training seeds (17/29/43) for E1/E2/E3; E2 seed links to its E1 parent.
E0/E0-P greedy results are deterministic controls repeated only for reproducibility
checks, not counted as new independent baselines. Report each trained seed and
mean, including failed training runs. Primary comparisons are E1-E0 and E2-E1.
Use paired family-cluster bootstrap (10,000 replicates, seed 987): sample families
with replacement, retain all their problems and paired arm outcomes. For trained
arms resample training-seed pairs as an additional level and report seed ranges;
three seeds give limited uncertainty estimates. Report ordinary 95% intervals and
97.5% simultaneous Bonferroni intervals for the two primary contrasts; calculate
paired two-sided permutation p-values on family-level differences and Holm-adjust
the two tests at family alpha .05. Document exact implementation before evaluation.

Predeclared meaningful improvement for a primary contrast: absolute pass@1 gain
>=.05, simultaneous interval lower bound >0, Holm-adjusted p<.05, and regression
increase <=.02. Compare E2 against E0-P as a separately labeled secondary test;
claim gains beyond stronger prompting only if its paired interval excludes 0.
E2-E3 reports warm-up effect at equal RL budget and separately equal-total-cost
sensitivity. Category/subgroup tests are descriptive unless preregistered.

300–500 problems may have low power for five-point differences, especially with
few independent families. Estimate paired variance on validation only and publish
a power/sensitivity analysis before test sealing. Do not enlarge the holdout
after inspecting results. If uncertainty includes zero, report inconclusive;
absence of statistical significance is not equivalence. Learning curves and
train reward cannot replace held-out inference results.

## Contamination and independence

Qwen pretraining data is not fully auditable; public benchmarks may be memorized.
Report QuixBugs separately and emphasize authored/new source-family holdouts.
Clone checks and source grouping reduce constructed leakage but cannot prove no
pretraining contamination. Independent reviewer checks split manifests, data
access controls, exact candidate counts, exclusion ledger and evaluation scripts
before final report. If test output informed another model revision, retire the
confirmatory status of that holdout.

## CPU implementation addendum (before any model results)

The independent CPU slice is documented in
[m2-cpu-analysis.md](../docs/m2-cpu-analysis.md). `evaluation.run` currently analyzes
recorded results only; generation remains unimplemented. Frozen plan and result
schemas bind one common BF16/hardware/batch/software profile, explicit per-arm
checkpoints and per-problem prompt/test-suite hashes and baseline failing/passing
counts. E1/E3 must reference the exact base; E2 must reference the exact
selected E1 seed checkpoint. Observed records require private candidate/report
artifacts and an operator-pinned canonical plan hash. These checks prove integrity,
not actual model execution or independent reviewer identity.

Algorithm details fixed now: quantile interpolation is type7; paired bootstrap
resamples families and paired seed IDs independently, keeping problems within
each sampled family and reporting the original problem-weighted mean. Paired
permutation flips the sign of each family aggregate weighted by its problem
count. Enumerate all signs for <=16 families; otherwise use 10,000 Monte Carlo
permutations with seed988 and `(exceedances+1)/(replicates+1)`. Bootstrap uses
10,000 replicates with seed987. Secondary contrasts retain unadjusted descriptive
intervals; only the prespecified two primary contrasts enter Holm correction.

Result-v1 `regression` denotes a regression actually observed in a complete test
execution. For incomplete executions it must be false rather than fabricate
per-test observations. Publish observed rate and a conservative rate counting
all incomplete executions as failure to preserve behavior; apply the <=.02
increase criterion to the conservative rate. Records also contain explicit
`regressed_passed_count` (null for incomplete execution); compute per-test rates
against frozen baseline counts and publish the observed denominator and conservative
bound. Reject changed suite denominators and truncated-output full-pass reports.
Cost remains unavailable without an audited ledger and must be null, never estimated
from missing evidence. This clarification precedes all results.
