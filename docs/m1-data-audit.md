# M1 pre-scaling data audit (spec 02)

Status (2026-11-23): **v3 frozen candidate built and mechanically audited 50/50;
manual review by the user-designated agent reviewer covers the v2 sample, with v3
deltas pending that reviewer's second pass.** No fixture code was executed or
compiled by the audit tool; all execution evidence comes from the runsc build.

Reviewer identity, stated plainly: the user designated a fresh Opus 5.5 agent at
high effort, run read-only with no authorship context. It is the same model family
as the author and **not a human reviewer**.

## Frozen candidate: v3

- Source `authored-fixtures-v2` (16 v1 + 37 v2 authored CC0 functions), config
  `configs/data-v3.json` (split seed 17, up to 2 mutants per train function,
  3 verification repeats, one manual exclusion).
- `artifacts/data/m1-fixtures-v3` (ignored, external to Git). Manifest pin
  `d68a564d28ac640461b2e09495016e752f31a48f452b1adef4312ec46ea94f1b`; split namespace
  `authored-fixtures-v2:split-seed-17`; policy `2dc7000f…5fde`.
- A second build (`artifacts/data/m1-fixtures-v3-rebuild`, retained) gave identical
  content-plan, split-membership, source, config, split/category/difficulty counts and
  rejected set. It was produced by the same author, not independently.

| | count |
|---|---|
| planned rows | 86 (one manually excluded before verification) |
| accepted | 71 (train 57 / validation 10 / test 4; 48 families) |
| rejected: no regression baseline or no failing test | 12 mutants |
| rejected: behaviourally identical to another variant | 2 (`median` `a+b`/`b+a`, `sign` mutant equal to authored bug) |
| rejected: manual review | 1 (`rotate_left` `xs[k:] - xs[:k]`, TypeError on every non-empty input) |
| authored bugs accepted | 53 / 53 |

Mutant survival by category (edited-construct rule): comparison 6 accepted / 10
rejected, operator 12 / 5. Difficulty: easy 62, medium 9 (mutants are `easy` by rule).

Split shift between namespaces: adding v2 seeds moved `count_equal-v1` and
`count_positive-v1` from train (v1) to validation (v2). Never mix v1 and v2 splits.

## Review history (kept as negative results)

1. **Author pass on v2 (before any review).** Found and fixed: a reference using
   unavailable builtins, five bugs with no regression baseline, whole-module
   reformatting by the mutator, text-duplicate mutants and inherited mutant
   categories. Tests and bugs were therefore edited *after* reading sampled cases and
   sandbox outcomes, so the later samples are not untouched estimates.
2. **Manual review of the v2 sample** (`.ai-native/m1-manual-review-v2.json`, bound
   to worksheet `09379cf5…9c35`, manifest `230a246f…44ad`): 27 ACCEPT, 22 ACCEPT_WITH_NOTE,
   1 REJECT. No wrong expected value found in ~210 hand-recomputed tests.
3. **Applied in v3:** the REJECT became a manual exclusion; mutants are labelled `easy`;
   behavioural duplicate detection (trusted outcome digest) replaced text-only dedup.
4. **Code review round 1** (`.ai-native/m1-code-review-r1.md`): REWORK. Fixed: audit
   leakage checks could never fail (searched a repr), provenance fallback, roadmap
   overclaim, missing pids evidence, undocumented split shift, mixed category systems
   in attrition, stale docs, loose pids range, mutator newline offsets, extraction
   control tokens.

## Not yet addressed (from the manual review, before scaling further)

- Authored `bug_category` labels are free-form; several are loose (constant edits and
  operand swaps labelled `operator`, a comparison flip labelled `branch`).
- Many suites sit at the four-test minimum; some have one discriminating test and
  untested stated boundaries (clamp bounds, `safe_divide` with b==0, `prefix` n=0).
- Some behaviour texts leave preconditions implicit (chunk size>0, caesar k sign).

## v3 worksheet

`.ai-native/m1-audit-v3.json` (sha256 `e5ad3d37…a615`): 50 seeded cases, mechanical
50/50. 36 have a carried-over v2 manual verdict on byte-identical content; 14 need
the reviewer's second pass.

```bash
python -m python_fixer.data.audit --dataset artifacts/data/m1-fixtures-v3 \
  --manifest-sha256 d68a564d28ac640461b2e09495016e752f31a48f452b1adef4312ec46ea94f1b \
  --output .ai-native/<new-audit>.json
```
