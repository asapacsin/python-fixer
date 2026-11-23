# M2 independent CPU analysis slice

Status: model-free analytics implemented; **real E0/E0-P evaluation NOT_RUN**.
M1 sandbox security remains blocked. This slice is independently authorized CPU
work under `verification/contracts/m2-cpu.json`, not a claim that M1 or M2 passed.
No weights, GPU, model generation or candidate execution are needed by these tools.

## Interfaces

- `evaluation.plan`: validate an operator-frozen public-metadata plan. It includes
  exact base/checkpoint/parent links (E1/E3 from base; E2 from paired E1), common model/tokenizer/data/generation/inference
  profile hashes, per-problem prompt/suite hashes and the entire expected grid.
- `evaluation.records`: validate shape, complete attempts, provenance and semantic
  consistency. Observed-model data requires exact-byte candidate and sandbox-report
  artifacts; hash-valid reports are checked against fixed suite/image/policy and
  their trusted full-pass/regression counts. No Python source is executed or parsed.
- `evaluation.metrics`: unbiased pass@k, compilation/failure/regression rates,
  family weighting, category/difficulty strata and generation token/latency metrics.
- `evaluation.statistics`: paired two-level family/seed bootstrap, paired cluster
  permutation tests and Holm correction. Details are in spec06's implementation addendum.
- `evaluation.run`: `validate` or `analyze` saved results, with an external plan pin
  and immutable new output artifact. This is not an inference command.

The schemas are [evaluation-plan.schema.json](../schemas/evaluation-plan.schema.json)
and [evaluation-result.schema.json](../schemas/evaluation-result.schema.json).
There is intentionally no populated real-model plan or result table: model/data
revisions and baseline artifacts have not been acquired or run.

## Run CPU tests now

```bash
source /workspace/.venvs/python-fixer/bin/activate
cd /workspace/python-fixer
python -m unittest discover -s tests -p 'test_m2_*.py' -v
python scripts/run_cpu_checks.py
```

Test metadata/results are generated in temporary directories and explicitly
labeled synthetic. Some integrity tests emulate observed-record interfaces using
test doubles; they do not establish any real model/hardware run or improvement.

## Analyze future recorded results

First pin an actual model/tokenizer, sealed dataset, inference profile, checkpoint
parents, generation and all problem/suite/prompt hashes in a plan. Freeze it before
observing test outcomes and retain its canonical JSON SHA-256 in operator-controlled
state. Hashing conventions: plan/generation/inference hashes use sorted compact
UTF-8 JSON via `python_fixer.common.digest`; file/candidate/report hashes refer to
**exact stored bytes**, including newlines. Do not confuse the two.

```bash
python -m python_fixer.evaluation.run validate \
  --plan /private/study/plan.json --plan-sha256 <operator-pin> \
  --results /private/study/results.jsonl --evidence-root /private/study/evidence
python -m python_fixer.evaluation.run analyze \
  --plan /private/study/plan.json --plan-sha256 <operator-pin> \
  --results /private/study/results.jsonl --evidence-root /private/study/evidence \
  --output /private/study/new-analysis.json
```

Evidence files live at `candidates/<raw-sha256>.py` and
`sandbox/<raw-sha256>.json` under the provided private root. Candidate files are
read only as bounded byte strings. Symlink escapes, missing/mismatched artifacts,
unexpected/duplicate/missing rows, changed seeds/profiles and inconsistent results
abort before metrics/output publication. The CLI accepts at most one million
planned records and at most 64 KiB per record/artifact; validated records are
held in memory, so the full sampled study may require several GB of CPU RAM.

## Statistical and evidence limits

Greedy primary and sampled secondary analyses require separate frozen plans.
Sampling repeats are averaged within a problem; different trained checkpoints
are never pooled to estimate pass@k. Bootstrap resamples paired training seeds
and source families, retaining each family's problems. Small-family and seed
warnings remain visible. Baseline copies are controls, not new independent runs.

Incomplete execution does not invent an observed regression count. Report both
the observed lower-bound rate and a conservative rate treating every incomplete
execution as failure to preserve behavior; use the conservative rate for the
preregistered regression-increase ceiling. Per-test regressions use explicit
`regressed_passed_count` values and frozen baseline denominators. Incomplete
executions carry null counts; publish observed denominators and conservative
bounds rather than inventing observations. Compute cost remains null until an
audited ledger exists; latency is not a cost ledger.

Only a complete, observed, confirmatory **primary** grid is eligible for the
numeric gain criterion. Synthetic/exploratory/sample-only analyses cannot set it.
Even eligible numeric criteria are not independent execution authentication or
scientific approval. The report explicitly retains that distinction. Public
benchmark contamination, test quality and real model provenance still need M6 review.

Remaining M2 work: Qwen generation/parser integration, sealed evaluator deployment,
validation-based power analysis, actual E0/E0-P inference/timing/cost and independent
protocol/run audit. No M2 model-experiment completion is claimed here.
