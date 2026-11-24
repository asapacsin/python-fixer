# ADR-004: Held-out size and pre-registered claims after the measured E0 baseline

Status: ACCEPTED (2026-11-24). Inputs: `verification/evidence/e0-cpu-validation/` (E0 greedy 2/32 = 6.25%),
frozen v9 (62 held-out problems). Power is computed by simulating an exact two-sided McNemar test at alpha 0.05,
4,000 runs per point (3,000 for the GRPO-vs-SFT grid).

## Power at N = 62 (measured E0 rate 6.25%, no regressions assumed)

| E1 held-out pass rate | 15% | 20% | 25% | 30% |
|---|---|---|---|---|
| power, E1 vs E0 | 0.47 | 0.87 | 0.98 | >0.99 |

GRPO vs SFT (E2 vs E1), assuming 20% of problems discordant:

| held-out N | 62 | 150 | 300 | 500 | 700 |
|---|---|---|---|---|---|
| 5pp effect | 0.09 | 0.22 | 0.44 | 0.67 | 0.82 |
| 10pp effect | 0.33 | 0.74 | 0.97 | 1.0 | 1.0 |

## Decision

1. **Confirmatory (pre-registered):** H1, E1 > E0 on test at N = 62. This is adequately powered if E1 reaches about 20%. Fences cause 66% of E0
   failures, and SFT teaches the format.
2. **Exploratory only:** E2/E3 vs E1. At N = 62 power is <=0.33. Report effect sizes with CIs, and make no significance claim.
3. **No dataset expansion now.** A confirmatory E2-vs-E1 claim needs about 170 held-out problems for a 10pp effect at 80% power.
   At the frozen split ratio (about 14% held-out), that means about 1,300 more authored functions. The real requirement also depends on the
   discordant rate and E1 level, and both are known only after the GPU preflight. Revisit after the E1 validation run: if E1 is below 20%
   or the owner wants E2 to be confirmatory, expand validation and test (new families only, test still sealed) before any test
   evaluation. Changing the split ratio would be a separate spec change.
