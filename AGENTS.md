# Engineering rules

1. Read the relevant `specs/`, `docs/implementation-roadmap.md`, active task
   contract and git status before editing. Implement one milestone at a time.
   M0 is specifications; the next authorized implementation scope is M1.
2. Prefer PyTorch, Transformers, PEFT and TRL over custom training algorithms.
   Keep adapters thin; one owner for split membership, rewards and run state.
3. Make incremental, testable changes. Add meaningful behavioral tests; preserve
   existing assertions. Update specifications and decision records when decisions
   change. Record why a departure is needed before changing experimental scope.
4. Each implementation task needs goal, acceptance, approved paths, required gates,
   evidence level, dependencies and bounded retries/time in a task contract.
   `verification/contracts/m1.json` is an implementation specification, not a
   completed run or an independent approval.
5. Use existing isolated cloud checkouts; do not create a Git worktree unless the
   user explicitly requests one. Preserve existing files and user changes.
6. Run deterministic checks, then bind evidence to HEAD, staged/unstaged diff and
   untracked source content, config/data hashes, commands, exit codes and logs.
   Re-run affected gates when source/config changes. Missing or stale evidence
   is NOT_VERIFIED. Runtime evidence belongs under ignored `.ai-native/`.
7. Request an independent, read-only reviewer only at a reviewable end state.
   Review actual diffs and acceptance evidence. Never self-approve or fabricate
   reviewer identity. No configured independent reviewer means review NOT_VERIFIED.
   Local tests are not an LLM review, GPU preflight, hosted CI or branch protection.
8. Use PASS / REWORK / BLOCKED / NOT_VERIFIED with explicit scope. Retry at most
   three times under the task budget; diagnose after two failed fixes. Continue
   authorized independent CPU work when GPU or external dependencies are blocked.
9. Never fabricate runs, metrics, cost, hardware checks or model improvement.
   Pre-register hypotheses before held-out evaluation. Keep negative results.
10. No held-out tests, references or outputs in SFT examples, RL rewards, prompts
    or model-selection loops. Split by source function / clone family before
    mutation. Keep the sealed evaluator separate from training workers.
11. Never execute generated code on the host or GPU node. Use the approved remote
    gVisor/container boundary from specs 01/08; AST filtering is not isolation.
    Never mount credentials, model storage, host sockets or held-out data there.
12. Keep checkpoints, datasets, caches, generated outputs and secrets outside Git.
    Use immutable manifests, atomic checkpoints and tested resume procedures.
13. Do not download weights, provision GPU nodes or launch expensive training
    without explicit authorization. Estimate cost and pass preflight first.
14. Reproduce official version-specific behavior; pin and hash-lock dependencies
    during implementation. Candidate values marked unverified are not readiness.

Default gates: `python3 scripts/validate_repository.py` and
`python3 -m unittest discover -s tests -v`. Later milestone gates are specified in
the roadmap and must be implemented, not replaced by documentation checks.
