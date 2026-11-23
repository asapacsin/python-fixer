# AI-native engineering integration review

Reviewed repository: https://github.com/asapacsin/ai-native-engineering

Reviewed commit: `1a161b849e6faaca9678c6e35c7e730d33e12ac0` (2026-11-23).
Read README, AGENTS, AI_BUILDER_BIBLE, INTEGRATION, PRACTICE_COVERAGE, controller,
GRPO-independent governance/frontier interfaces and process-integration tests.
The upstream audit explicitly says practice coverage is incomplete. Its procedural
approval strings and CLI role names are not authenticated role separation.

## Applied here

| Practice | Local artifact | Enforcement / evidence boundary |
|---|---|---|
| Spec-first, one owner, bounded scope | specs 00–09, ADR-001, AGENTS | Maintainer/agent workflow; no production trainer yet |
| Task goal, acceptance, paths, dependencies and budgets | verification/contracts/m1.json | CPU validator checks shape and gates; not an approval |
| Source-owned required evidence | verification/requirements.json | Validator prevents accidental missing IDs/levels |
| Executable deterministic checks | scripts/validate_repository.py, tests, CI | Executed CPU checks; workflow file alone is not hosted CI evidence |
| Evidence bound to actual source | validator --report | HEAD + staged/unstaged + untracked content, config hashes |
| Train/reward/evaluation ownership | specs 01/02/06/08 | Required M1/M6 access tests, NOT implemented security |
| Honest proof levels | README, reports policy | No GPU, sandbox or model review claimed by doc gates |
| Bounded continuation and blockers | roadmap task dependencies and AGENTS | Continue independent CPU work; no unattended supervisor installed |
| Independent end-state review | M1 contract, AGENTS | Required before merge, missing reviewer remains NOT_VERIFIED |

## Deliberate scope

Do not vendor the upstream controller wholesale into M0. This task is a research
specification, and importing its multi-agent supervisor, optional Claude hooks and
model-routing defaults would add unneeded runtime and billing configuration.
The deterministic local validator and future task contract implement the practices
needed now. If later automating execution, consume a pinned upstream revision via
a thin adapter, run its tests and verify authenticated read-only reviewer behavior.
Do not copy sample PASS states, fake review commands or invent task approvals.

Not installed/verified: independent model sessions, exact-tip merge controller,
required GitHub branch rules, pre-tool hooks, automatic frontier discovery,
supervisor checkpoints or cost-based model routing. Local gates can be bypassed;
CI/branch rules require separate repository administration. No external messages,
pushes, merges or paid model calls were performed for this setup.

## Evidence report protocol

`python3 scripts/validate_repository.py --report .ai-native/m0-validation.json`
records only specification checks: commit (null for this unborn repository),
git status, hashes of index/diff and all nonignored files, requirement IDs,
command, exit status and UTC timestamp. It is not an independent review. Any
change after capture invalidates the fingerprint; rerun before review. A report
must never live in tracked source or overwrite arbitrary paths. Artifacts and
private evaluation logs remain external. This repository has no commit until
the maintainer records its first revision; do not fabricate a base SHA.
