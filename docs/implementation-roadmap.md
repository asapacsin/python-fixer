# Implementation roadmap

Implement one milestone at a time. `[ ]` means unimplemented/unverified, not
authorized paid execution. No model weights, cloud rental or long training job
is authorized by this roadmap. CPU work can proceed through the user's task scope;
paid work requires a measured cost estimate and explicit authorization.

## M0 — Repository, specification and environment contracts

- Objective: make M1 implementable without redesign.
- Dependencies: empty checkout; Python 3.11/3.12 and small validation dependency.
- Tasks: author specs 00–09, architecture decision, fixed prompts, schemas and
  sample row, reward cases, configs, M1 contract, source references and CI.
- Deliverables: files in this repository and current CPU validation evidence.
- Automated tests: documentation/link/config checks; JSON schema validation;
  reward-oracle table, invalid cases, source evidence integrity tests.
- Acceptance: all checks pass; candidate GPU values flagged; unresolved runtime
  requirements explicit; no experiment numbers or trainer implementation.
- Risks: API drift, inconsistent batch arithmetic or split/reward definitions.
- Completion: specification gates pass. Independent review remains a distinct
  requirement; M0 never certifies model training or sandbox behavior.

## M1 — Dataset pipeline, verification and sandbox

- Objective: a safe, deterministic CPU vertical slice before GPU use.
- Dependencies: M0 specs 01/02/05/08; runsc-capable dedicated CPU VM for integration.
- Exact tasks (see `verification/contracts/m1.json`):
  - [x] Implement typed problem/manifest loading and standard JSON Schema checks.
  - [x] Add authored pure-function seeds, license ledger and deterministic mutation operators.
  - [x] Implement source-family grouping, clone audit and seeded group splits before mutation.
  - [x] Implement train/validation/evaluation views that drop prohibited fields.
  - [x] Pin the official Python image and implement a narrow runsc gateway; keep expected values outside candidate processes.
  - [x] Implement requests, limits, cleanup, hash verification, idempotent cache and bounded infrastructure retries.
  - [x] Implement reference-pass / buggy-fail verification and flakiness quarantine.
  - [x] Implement production reward v1 independently and run the shared acceptance table.
  - [x] Pass the real-runsc security preflight on an enforcing runtime (35/35, 2026-11-24, session container per ADR-002); row-by-row coverage and remaining double-only rows in [m1-security-coverage.md](m1-security-coverage.md).
  - [x] Freeze the first data manifest, attrition ledger and 50-case review (v7: 67 verified fixture cases, pin `299e7adf…0737`; LLM manual review, not human; see [m1-data-audit.md](m1-data-audit.md)).
- Deliverables: `src/python_fixer/data`, `sandbox`, `rewards`, CPU CLIs and tests;
  external verified corpus/manifests; worker image digest and isolation report.
- Automated tests: schema roundtrip; duplicate/clone-family leakage; deterministic
  rebuild; all reference tests pass; bug fails relevant test; regression baseline
  nonempty; forged reports; attack table in spec08; network/host sentinel checks;
  cleanup after killed jobs; rewards match fixtures and malformed reports abort.
- Acceptance: at least 16 fixture problems through real sandbox, then 50 accepted
  cases audited before scaling. All generated code executes only in sandbox;
  every accepted row has baseline counts and evidence hashes; no split overlap.
- Risks: runtime unavailable, weak tests, equivalent mutations, source licensing.
- Completion: required unit + real isolation integration gates pass, data manifest
  reproducible and independent read-only review recorded. Missing runsc is BLOCKED;
  metadata/loader tests alone do not complete M1.

Current implementation and historical execution evidence are recorded in
[m1-implementation.md](m1-implementation.md). Checked tasks above mean code is
implemented; the milestone remains BLOCKED until the remaining gates pass.

## M2 — Baseline evaluation

- Objective: implement trustworthy metrics and a frozen inference/evaluation protocol.
- Dependencies: M1 verified dataset/sandbox; licensed/pinned Qwen metadata;
  explicit GPU/inference authorization before model download/run.
- Tasks:
  - [ ] Implement prompt rendering, strict extraction and candidate/result schemas.
  - [ ] Implement immutable evaluation manifest, sealed test access and exact-attempt completeness checks.
  - [ ] Implement greedy pass@1, unbiased pass@k, regressions, cluster bootstrap and paired tests.
  - [ ] Estimate power from validation; lock primary comparisons before evaluation.
  - [ ] Run E0/E0-P with frozen settings; retain sealed outputs and timing/cost ledger.
- Deliverables: evaluation CLI, metric tests, frozen protocol and external baseline artifacts.
- Automated tests: known metric vectors (c=0,n and k edges); paired bootstrap
  determinism; missing/duplicate attempts fail; no hidden fields in generation;
  strict output parser; equivalence of repeated frozen base generation.
- Acceptance: correct denominators and confidence procedure, data/model/template
  hashes present, no baseline scores invented; GPU preflight required for inference.
- Risks: pretraining contamination, weak power, nonidentical templates/decoding.
- Completion: independent audit of protocol and real E0/E0-P artifacts; no training yet.

## M3 — SFT implementation and experiment

- Objective: prove adapter training and evaluate the supervised arm.
- Dependencies: M1/M2; approved GPU budget; resolved training lock; persistent storage.
- Tasks:
  - [ ] Resolve candidate package tuple and produce hash-locked environment/image.
  - [ ] Pin model/tokenizer SHAs, test non-thinking prompt/EOS and label masks.
  - [ ] Implement SFTTrainer/PEFT config adapter, data collator and full checkpoint/resume.
  - [ ] Implement stage-specific GPU preflight and replacement-node storage recovery.
  - [ ] Overfit 16 examples, then run E1 seeds under measured budget.
  - [ ] Select checkpoints using validation and export adapter/manifest.
- Deliverables: SFT CLI, environment lock, verified run manifests and E1 checkpoints.
- Automated tests: masked prompt/pad labels; no target truncation; only LoRA grads;
  tensor changes; base frozen; finite loss; save/reload logits; next-step resume.
- Acceptance: real expected steps/weight changes and verified persistent artifacts;
  measured peak memory/cost; no claims from training loss alone.
- Risks: CUDA/API compatibility, OOM, wrong masks, storage loss.
- Completion: three seeds or clearly labeled budget-limited pilot, validation
  selection evidence and passed GPU preflight. Held-out gain not required to complete training.

## M4 — GRPO and reward integration

- Objective: connect established GRPOTrainer to trusted behavior rewards.
- Dependencies: M1 security/reward gates; M3 environment and selected E1 adapter.
- Tasks:
  - [ ] Implement reward callback with metadata/group alignment and train-only gateway identity.
  - [ ] Implement E2/E3 adapter initialization and optimizer reset rules.
  - [ ] Test actual TRL batch shape and completion count; log zero-variance groups.
  - [ ] Test Dr. GRPO objective settings, masks and optional reference-policy semantics.
  - [ ] Extend GPU preflight to real sampled groups, optimizer steps and resume.
- Deliverables: GRPO CLI/config adapter, trusted reward traces and short smoke artifacts.
- Automated tests: all equal rewards => zero advantages; malformed report => abort;
  failed-regression score0; forged PASS denied; request timeouts bounded; mixed
  rewards yield finite nonzero gradient; save/reload policy state.
- Acceptance: real two-prompt × four-candidate integration plus 2–5 updates,
  correct counts and policy tensor changes without held-out access.
- Risks: no learning signal, worker latency, PEFT reference mismatch.
- Completion: security and real GPU integration gates pass, reviewed evidence;
  a synthetic reviewer or mocked worker is insufficient.

## M5 — Controlled RL experiments and ablations

- Objective: run paired E2/E3 and optional prioritized ablations.
- Dependencies: M4 and approved measured spend; frozen RL and evaluation configs.
- Tasks:
  - [ ] Freeze paired seeds/data order and run 500-step budgets with exact candidate counters.
  - [ ] Monitor preregistered stops, checkpoint/recover failures and record all runs.
  - [ ] Select validation-best checkpoints with earliest tie rule.
  - [ ] If authorized budget remains, execute one-factor ablations from spec09.
- Deliverables: E2/E3 checkpoints, trajectories, stop ledger, actual GPU/CPU cost.
- Automated tests: budget enforcement; hashes/parents; resumption accounting;
  expected steps/tokens/candidates; no unrecorded dropped prompts.
- Acceptance: matched RL budget and disclosed total compute; failed seeds retained;
  no tuning from test outcomes and no unsupported improvement claim.
- Risks: zero-variance collapse, RL regressions, compute imbalance, budget overrun.
- Completion: required arms have complete artifacts or explicit failed-run records;
  incomplete study remains incomplete even if one favorable arm succeeds.

## M6 — Independent evaluation and error analysis

- Objective: answer the hypotheses with sealed, paired evidence.
- Dependencies: locked E0/E0-P/E1/E2/E3 artifacts and M2 protocol.
- Tasks:
  - [ ] Independently verify split/license/clone manifests and data-access audit.
  - [ ] Generate/score exact frozen held-out candidates with evaluator-only identity.
  - [ ] Compute intervals, Holm tests, regressions, strata and cost comparisons.
  - [ ] Categorize errors and audit high rewards for test weaknesses.
- Deliverables: sealed candidate/result JSONL, aggregate metrics, uncertainty/error report.
- Automated tests: complete attempt grid; matching hashes; deterministic metrics;
  contamination checks and immutable exclusion ledger.
- Acceptance: exact denominators, qualified claims against prespecified thresholds;
  negative/inconclusive outcomes retained; independent reviewer inspects actual artifacts.
- Risks: low power, public contamination, post-selection leakage.
- Completion: review and frozen analysis hash; no retrospective holdout tuning.

## M7 — Reproducible report and portfolio

- Objective: make the research auditable and professionally presentable.
- Dependencies: M6 analysis; permissible redistribution and artifact retention.
- Tasks:
  - [ ] Reproduce selected run/evaluation on a replacement node from locks/manifests.
  - [ ] Publish methods, measured results, confidence/cost plots and negative findings.
  - [ ] Create artifact index/model card/dataset card with access and license boundaries.
  - [ ] Document resume, teardown, known risks and a short portfolio walkthrough.
- Deliverables: report, reproduction commands, external artifact index and demo.
- Automated tests: index checksum verification; reproduction result tolerances;
  no committed weights/private tests/secrets; documentation links pass.
- Acceptance: another authorized operator can locate inputs and reconstruct the
  reported comparison; incomplete runs and unavailable artifacts plainly marked.
- Risks: broken links, storage expiry, license restrictions, overstated generality.
- Completion: independent read-only report/reproduction review and persistent
  artifact verification. Publishing/merging remains a separately authorized action.
