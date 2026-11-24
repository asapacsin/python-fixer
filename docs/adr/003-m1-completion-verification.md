# ADR-003: M1 completion by automated dataset verification

- Status: Accepted
- Date: 2026-11-24
- Decided by: the user (project owner), recorded by the implementing agent

## Context

The M1 contract asks for an "independent read-only review" before M1 is complete.
AGENTS.md rule 7 says self-review does not count. The owner decided that for M1,
whose output is pipeline fixtures and an isolated reward, the verification is the
automated, dataset-based comparison of methods: no human review is required.

## Decision

M1 is complete when every result below comes from real execution and is bound by hash:

1. Every accepted fixture: reference passes, bug fails, three repeated runs on real runsc.
   A v7 rebuild matches the frozen manifest on content-plan, split-membership, source
   and config hashes, counts and rejected set.
2. Through the deployed operator-mTLS gateway, the reward ranks the methods correctly:
   reference 1.0 > buggy 0.0 > attack -1.0, and an unauthorized client is refused.
3. Preflight runs on real runsc with cgroups enforced (35/35). Hosted CI and the unit
   suite pass.

Fresh read-only LLM reviews (rounds 1-7) are advisory. They are recorded as LLM
reviews, not as human or independent approval.

## Evidence

`verification/evidence/m1-v7/` (frozen v7 and its freeze record) and
`verification/evidence/m1-final/` (rebuild manifest, preflight report, mTLS deployment),
each with SHA256SUMS.

## Consequences

- M1 is complete in scope. `verification/contracts/m1.json` stays a specification, as
  its schema requires.
- Before M3/M4 the worker moves to a separate machine (ADR-002).
- A human spot-check is still advisable before scaling past fixtures. It is not an M1 gate.
