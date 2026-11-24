# M1 pre-scaling data audit (spec 02)

Status (2026-11-24): **v7 frozen for the M1 pipeline-fixture scope.** Built from
committed code `e5cf562`, deterministic on rebuild in content-plan, split-membership, source and config hashes (row/partition hashes embed runtime evidence and vary), mechanical audit 50/50 (with one
reviewed waiver), manual review recommends FREEZE. Freeze record
`.ai-native/m1-freeze-v7.json` (sha256 `bf107516…f06b`) binds every file below by hash.

Reviewer identity, stated plainly: the user designated a fresh Opus 5.5 agent at
high effort, run read-only with no authorship context. It is the same model family
as the author and **not a human reviewer**; under AGENTS rule 7 this is an LLM review,
and a human spot-check is advisable before scaling beyond fixtures.

## Frozen dataset: v7

- Config `configs/data-v7.json`; `artifacts/data/m1-fixtures-v7`; manifest pin
  `299e7adf74a25a1ae80943a68a44047bff22f8521cf90ea27efecb4f06bf0737`; split namespace
  `authored-fixtures-v2:7a1547a7002afe7a:split-seed-17`; policy `5bea6490…096d`;
  preflight `.ai-native/m1-preflight-v7.json` (35/35).
- Rebuild `artifacts/data/m1-fixtures-v7-rebuild` (retained, same author) matches on
  content plan, split membership, source, config, builder source, counts and rejected set.
- 67 accepted (train 53 / validation 10 / test 4; 48 families); 19 rejected: 12 with no
  failing or no regression test, 4 crash-only, 2 behavioural duplicates, 1 manual
  exclusion (bound to `.ai-native/m1-manual-review-v2.json` by SHA-256). Authored bugs
  accepted 53 / 53.
- v5 → v7 changed only verification evidence (policy fixes for crash scoring and
  pre-compilation failures); all row content, the rejected set and the 50-case sample
  are identical, so the pass-4 FREEZE verdict applies.
- Worksheet `.ai-native/m1-audit-v7.json` (sha256 `cd8c8f1e…9683`): 50/50. The two
  `fibonacci` rows pass via the waiver in `configs/audit-waivers.json` (tests 1-2 are
  the stated indexing convention; bound to suite hash, behaviour digest, function id
  and the pass-3 review); tests 5-7 were added as private tests.

## Manual review passes

| Pass | File | Scope | Outcome |
|---|---|---|---|
| 1 | `m1-manual-review-v2.json` | 50 v2 cases | 27 ACCEPT, 22 with note, 1 REJECT (now excluded) |
| 2 | `m1-manual-review-v3.json` | 14 new v3 cases + 36 carried | no REJECT |
| 3 | `m1-manual-review-v4-pass3.json` | 6 new v4 cases, fibonacci flag | ACCEPTABLE with conditions |
| 4 | `m1-manual-review-v5-pass4.json` | 3 changed cases, conditions | conditions met, FREEZE |

Carried verdicts were matched strictly on the reviewed content including labels.

## Use limits (from the reviews)

- Pipeline fixtures, not a research benchmark; the test split has 4 rows.
- Authored difficulty/category labels are inconsistent with the fixed rule: do not make
  difficulty-stratified claims; recompute before scaling.
- Several suites have trivial or coincidental regression passes; some preconditions
  are implicit; the test-leak check is a heuristic.

## Superseded: v4

- Config `configs/data-v4.json` (v3 plus the exclusion bound to review SHA-256);
  `artifacts/data/m1-fixtures-v4`, manifest pin
  `63a2ac62b060d6d74a886b284d0a71a7b8499accd24c45c88927d68be0a331d6`, builder commit
  `8814885` (tree matches `builder_source_sha256`), policy `186bc39b…9d25`.
- Rebuild `artifacts/data/m1-fixtures-v4-rebuild` (retained, same author) matches on
  content plan, split membership, source, config, builder source, all counts and the
  rejected set.
- 67 accepted (train 53 / validation 10 / test 4; 48 families). 19 rejected: 12
  mutants with no failing or no regression test, 4 crash-only mutants (fixed rule),
  2 behavioural duplicates, 1 manual exclusion. Authored bugs accepted: 53 / 53.
- Worksheet `.ai-native/m1-audit-v4.json`: 50 cases; 44 carry a manual verdict on
  identical content (including labels); 6 are new. Mechanical 48/50: both
  `fibonacci` rows are flagged because the behaviour text states
  `fibonacci(0) == 0 and fibonacci(1) == 1`, which are tests 1-2. This is the
  indexing convention, but it is also a verbatim test statement; left for the manual
  reviewer rather than reworded to satisfy the check.

v3 (below) is superseded: current code no longer builds `configs/data-v3.json`
(its exclusion lacks `review_sha256`).

## Superseded: v3

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

## v8: expanded authored dataset (2026-11-24)

- Source `authored-fixtures-v3` = v1 (16) + v2 (37) + **116 new authored CC0 functions**
  (`src/python_fixer/data/fixtures_v3.json`). Expected values are hand-written, not
  produced by running the reference. Each authored bug was hand-checked to fail at least
  one test and pass at least one.
- Config `configs/data-v8.json`; frozen copy and checksums in `verification/evidence/m1-v8/`.
  Build: accepted 247 / rejected 75 (all 169 authored pairs accepted; rejections are
  mutants plus the existing v2 manual exclusion). Splits: train 211 / validation 17 /
  test 19; 159 families. Held-out difficulty: easy 18, medium 11, hard 7.
- Fixes found by the real sandbox and audits before freezing: (1) 9 functions used `_` as
  a variable, which the sandbox policy forbids, so they were renamed and none of the
  expected values changed; (2) the 13-gram audit found `rotate_matrix` (train) to be a
  near-clone of `transpose` (test), so they were declared one family and the split was
  regenerated (spec 02).
- Leakage audit (`leakage-audit.json`): no data-access issues; max held-out/train
  13-gram overlap 0.18, none flagged.
- Limits: still small for a confirmatory benchmark (36 held-out problems). Same author
  for functions, tests and bugs, so a human spot-check of the new functions is advisable.
  Manual review of v8 is NOT_VERIFIED.
