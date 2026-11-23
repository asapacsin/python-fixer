# 05 — Behavior reward v1

Rewards are trusted runner outcomes, never model self-scores, printed pytest
summaries or similarity to a reference implementation. The reference is an
oracle for dataset verification, not the required candidate text. Equivalent
valid implementations earn equal rewards. The reward engine only accepts a
response whose request, candidate, suite, image and policy hashes match.

## Definition and precedence

For a verified training problem, let F be the nonempty set of tests failing on
the buggy program, P the nonempty set passing on it. Let f be the count from F
passing under the candidate, and p the count from P still passing. M1 rejects
problems with either baseline set empty. Run all required tests with fixed
weights and stable IDs; candidate exceptions count as test failures.

Define `a = 1[f=|F| and p=|P|]`. For a complete trusted execution:

`R = 0` if `p < |P|`; otherwise `R = 0.5*f/|F| + 0.5*a`.

Apply failure precedence *before* this expression:

| Outcome | Reward / action |
|---|---|
| INTERNAL_ERROR, missing/hash-mismatched/contradictory report | No reward; bounded retry then abort step BLOCKED |
| POLICY_VIOLATION, TIMEOUT, RESOURCE_LIMIT | -1 |
| INVALID_OUTPUT, COMPILE_ERROR | -0.5 |
| OK, compiled, all tests completed | Formula above |

No separate positive compile reward: compilation is necessary to earn behavior
reward and is measured separately. No fix with no regression scores 0; half of
failing tests repaired without regressions scores .25; full repair scores 1.
Any newly failing previously passing test scores 0. A timed-out attempt cannot
retain partial credit from tests that happened to finish first.

Partial credit is appropriate only for distinct substantive assertions, not
duplicate tests inflating f/|F|. The primary formula is fixed before training.
Binary full-pass reward is an optional ablation. Do not change reward weights
based on held-out results or grant reference-text similarity bonuses.

## Output and policy checks

Accept exactly one UTF-8 module, <=64 KiB, no Markdown fences/prose, same public
function name and signature; reject extra top-level executable statements beyond
imports, permitted definitions and literal constants. Candidate must not redefine
the harness/protocol, import runtime inspection/OS/network modules, read files,
spawn processes or modify tests. AST/import allowlisting is defense in depth.
All parsing and compilation of untrusted candidates occurs in the approved
worker boundary. M0's reward oracle processes counts only and never compiles code.

The worker's invocation shim runs a single function call per isolated process,
returns a bounded typed JSON value, and does not receive expected values. The
trusted gateway compares outputs to private expectations outside the candidate
process. Test expectations and runner decision channels are never candidate
stdout, importable modules or writable files in its namespace. For future
property tests, invoke the candidate as an untrusted callable endpoint and keep
assertions in the trusted controller. Do not run pytest in the same interpreter
as untrusted code and accept its summary as authoritative.

## Test integrity and caching

Test bundles are immutable, versioned and selected only by the gateway's problem
mapping. Candidate request cannot specify paths or commands. Pin comparator,
test seeds and runner image. Cache key includes candidate/suite/policy/image
hashes and limits; no cross-suite reward reuse. Detect output limits, test count
mismatch, unexecuted tests and cancellation; these never produce a valid score.
Periodically audit random accepted repairs with independent extra train-only
tests. New tests create a new reward/data version; never silently change rewards
mid-run. Inspect high-reward cases for hard-coded lookup behavior and weak tests.

## Executable specification

[reward-cases.json](../verification/reward-cases.json) is the acceptance table.
`python3 -m unittest discover -s tests -v` runs the specification oracle over
these cases, including precedence conflicts, regressions and malformed counts.
This proves formula consistency only. M1 must apply the same table to real
sandbox responses and add adversarial execution tests. Production reward now lives
in `src/python_fixer/rewards/behavior.py` and does not import the oracle. M1 security
integration status is recorded in; see [implementation status](../docs/m1-implementation.md).

M4 adds trainer integration tests: reward output length equals generated
completion count; repeated IDs align; no NaN/None rewards; zero-variance groups
produce zero advantages; infrastructure failure aborts rather than poisoning
an optimizer update. Log reward components separately from the scalar.
