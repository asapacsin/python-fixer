# M1 pre-scaling data audit (spec 02)

Status: **mechanical audit 50/50 PASS; manual review PENDING (not independent)**.
Recorded 2026-11-23. No fixture code was executed or compiled by the audit tool;
all execution evidence comes from the runsc sandbox build.

## Dataset under audit

- Source `authored-fixtures-v2`: the 16 v1 seeds (`src/python_fixer/data/fixtures.json`)
  plus 37 new authored CC0 seeds (`src/python_fixer/data/fixtures_v2.json`), 53 functions.
- Config `configs/data-v2.json`: split seed 17, up to 2 deterministic mutants per
  train function, 3 verification repeats.
- Output `artifacts/data/m1-fixtures-v2` (ignored, external to Git). Manifest pin
  (canonical digest, as used by `--manifest-sha256`):
  `230a246f8fd5b53ca428a3f0b30d2ed2fbe47ce6b5587cfea4330105b40d44ad`.
- Policy sha256 `7740fea4ecd45ab3979d7f4d164d2d508f7520e3c479e46a4c10a545df41648b`,
  image `python@sha256:07d48240…b535`, runsc release-20261109.0, cgroups enforced.
- An independent second build gave identical content-plan, split-membership, source,
  config, split/category counts and the same rejected set.

## Attrition and mutation survival

| | count |
|---|---|
| planned rows | 86 |
| accepted | 74 (86.0%) |
| rejected | 12, all mutants: "bug must compile, fail semantic tests and preserve regression tests" |
| authored bugs accepted | 53 / 53 |
| mutants accepted | 21 / 33 |

Splits: train 60, validation 10, test 4; 48 families. Accepted categories:
operator 25, comparison 18, boundary 15, return 6, branch 4, algorithm 3,
initialization 3. The test split is small because families are assigned by an
80/10/10 hash; these are pipeline fixtures, not a held-out benchmark.

## Author findings (fixed before freezing)

Found by reading the sampled cases, not by the mechanical checks:

1. `caesar_shift` reference used `chr`/`ord`, which the restricted sandbox builtins do
   not provide; rewritten with the allowed `string` module.
2. Five authored bugs failed every test (no regression baseline) and were correctly
   rejected; their bugs or tests were adjusted so each bug fails some and passes
   at least one test.
3. Mutants were emitted with `ast.unparse`, which reformatted unrelated lines
   (`lo, hi = 0, n` → `lo, hi = (0, n)`). The mutator now edits only the operator
   token in the original text and keeps the variant only if it parses to exactly
   the intended AST.
4. Mutants identical to the authored bug were kept as separate cases (4 duplicates in
   the first sample); duplicates are now discarded.
5. Mutants inherited the seed's bug category (e.g. a comparison change labelled
   `return`); the category now follows the edited construct.

Noted, not changed: some accepted mutants fail by raising (e.g. `xs[k:] - xs[:k]`,
`i <= len(a)` → IndexError) rather than returning a wrong value. Spec 02 does not
exclude runtime-error bugs; the manual reviewer should decide.

## What remains

The worksheet `.ai-native/m1-audit-v2.json` (sha256
`09379cf5a907ccaa2c35566cd8706ed82c8836b6c74181a3d9cf0810b4759c35`) lists the
50 seeded-random cases with diffs, baselines, mechanical checks and per-case
`manual_review` fields set to `PENDING`. A human reviewer who did not author these
fixtures must fill in reviewer, verdict and notes. Until then the spec 02 manual
review and the M1 independent review are NOT_VERIFIED.

Regenerate the worksheet:

```bash
python -m python_fixer.data.audit --dataset artifacts/data/m1-fixtures-v2 \
  --manifest-sha256 230a246f8fd5b53ca428a3f0b30d2ed2fbe47ce6b5587cfea4330105b40d44ad \
  --output .ai-native/<new-audit>.json
```
