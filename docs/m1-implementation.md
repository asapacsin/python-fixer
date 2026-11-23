# M1 CPU implementation and observed readiness

Status: **real runsc security preflight PASS (22/22) and 16-fixture dataset CPU_VERIFIED
on 2026-11-23; 50-case audit and independent review NOT_VERIFIED**.
No GPU, model weights, model generation or training was used.

## Implemented

- Installable package with packaged schemas and canonical prompt resources.
- Strict JSON/schema/hash validation; 16 (v1) + 37 (v2) original authored fixture seeds;
  deterministic clone-family grouping and splits before AST mutation.
- SFT/RL/evaluation projections that keep tests/references/provenance out of prompts
  and reject non-train rows for training.
- gVisor-only Docker driver, no host fallback, mounts, secrets, sockets or GPU.
  One container per candidate suite; fresh interpreter per test; trusted comparisons
  against expected values absent from candidate processes.
- Production behavior-v1 reward, independent of the M0 oracle; candidate/suite/
  image/policy bindings; bounded concurrency, retries, cache and mTLS transport.
- Manifest-pinned dataset loading verifies partition, row and repeated execution
  evidence. Changed baselines cannot reuse old evidence.
- Atomic dataset writes, manifest published last, no overwrite of existing outputs.
  Plan/split IDs are deterministic; runtime metrics/evidence hashes naturally vary.

## 2026-11-23 enforcing-runtime run

Runtime: gVisor `runsc` release-20261109.0 (official release tarball, SHA-512
verified), `--platform=systrap`, **cgroups enforced** (no `--ignore-cgroups`),
cgroup v1, Docker 29.10.0, pinned image from `configs/sandbox.json`. Worker: the
ephemeral cloud session container, not yet the dedicated `separate_cpu_worker`.

- Preflight: **PASS 22/22**, including the aggregate container-memory cap that failed
  previously. Policy sha256 `7740fea4…48b`.
- Fix required: the host `--pids-limit=32` also counted gVisor Sentry threads, so a
  4-child fork storm killed the sandbox (exit 2, INTERNAL_ERROR). The candidate cap
  stays 32 via in-sandbox `RLIMIT_NPROC`; a separate host bound
  `runtime_host_pids_limit=128` contains the whole sandbox (64 was insufficient).
  Rationale recorded in spec 08; covered by a unit test.
- Dataset `artifacts/data/m1-approved-v1` (ignored): accepted 16, rejected 0,
  train 10 / validation 3 / test 3, manifest sha256 `44e98389…693f`. A rebuild gave
  identical content-plan, split-membership, source and config hashes; row/partition
  hashes differ only through embedded runtime evidence.
- Evidence: `.ai-native/m1-approved-worker.json`, `.ai-native/m1-evidence-20261123.json`.

Follow-up the same day: source `authored-fixtures-v2` (53 authored CC0 functions,
`configs/data-v2.json`) built to `artifacts/data/m1-fixtures-v2` with 74 accepted /
12 rejected (all rejections are mutants), deterministic on rebuild. Attrition ledger,
mutation survival and a seeded 50-case worksheet are in
[m1-data-audit.md](m1-data-audit.md); mechanical checks 50/50, manual review PENDING.

Round-1 review fixes then produced v3 (`configs/data-v3.json`, 71 accepted,
manifest pin `d68a564d…4f1b`) and a 29/29 preflight under policy `2dc7000f…5fde`; see
[m1-data-audit.md](m1-data-audit.md) and [m1-security-coverage.md](m1-security-coverage.md).

Worker decision (user, 2026-11-23): for M1, which has no GPU trainer, the session
container is accepted as the CPU worker. Spec 01/08's "separate worker" is about
isolating generated code from the trainer, model weights, credentials and held-out
data; none of these exist in this session. Before any GPU trainer exists (M3/M4) the
worker must be a separate machine, and the operator-mTLS deployment is still unexercised.

Still open: the manual review's second pass on 14 v3 cases,
deployment on the dedicated worker with operator mTLS, and independent read-only
review (NOT_VERIFIED; self-review does not count).

## Earlier observations

The runtime uses `runsc --platform=systrap --ignore-cgroups`. The initial
functional build at ignored `artifacts/data/m1-second` verified all 16 authored
reference/bug pairs, three times each, rejected=0. Splits: train10/validation3/test3.
These are pipeline fixtures, not a research benchmark or model-quality result.
Their evidence predates later runner hardening and is historical, not currently
approved production data.

A broader real preflight observed **21/22** checks passing: non-root, network
denial, read-only root, absent host sentinel/secrets/devices/socket, compilation/
policy failures, loop/memory/output limits, process limits, fresh workspaces and
cleanup. **Aggregate container-memory-cap failed.** Per-process RLIMIT_AS and
Docker flags do not prove the required 256 MiB whole-job bound when cgroups
are ignored. That raw report was produced in an earlier container and is **not
retained** in this checkout; treat it as historical narrative only.

Code rejects an `--ignore-cgroups` runtime *before candidate execution* (preflight
exit 3 BLOCKED). Do not weaken the check or substitute runc/host execution. Default unit tests include real TLS
with disposable certificates plus clearly labeled execution-protocol doubles;
those doubles do not certify containment. Hosted CI and production deployment
have not been exercised.

## CPU development

```bash
cd /workspace/python-fixer
source /workspace/.venvs/python-fixer/bin/activate
python -m pip install -r requirements-dev.txt
python -m pip install --no-deps --no-build-isolation -e .
python scripts/validate_repository.py
python -m unittest discover -s tests -v
python scripts/run_cpu_checks.py
python -m python_fixer.data.build --config configs/data.json --plan-only
```

Planning never executes or verifies programs. See [fixture provenance](fixture-provenance.md).
Unknown licenses and repeated identical input assertions are rejected.

## Resume real execution

Use a dedicated CPU worker with runsc **and actual aggregate memory/cgroup
enforcement**. Install the pinned Python image from `configs/sandbox.json` through
normal registry TLS/checksum verification, then run:

```bash
python -m python_fixer.sandbox.preflight --config configs/sandbox.json \
  --report .ai-native/m1-approved-worker.json
python -m python_fixer.data.build --config configs/data.json \
  --output artifacts/data/m1-approved-v1
```

Both must succeed. Inspect counts and policy/image hashes; save the dataset
manifest digest outside Executor-writable state. The current managed daemon
cannot be repaired by editing repository code. Do not restart it, modify unrelated
containers or rent another machine automatically. No GPU is needed for M1.

Deploy the separate CPU reward service only with operator-provisioned mTLS
certificates; test certificates are not production credentials:

```bash
python -m python_fixer.sandbox.server --config configs/sandbox.json \
  --dataset /persistent/data/m1-approved-v1 --manifest-sha256 <pinned-manifest-hash> \
  --ca /secure/ca.crt --certificate /secure/worker.crt --key /secure/worker.key \
  --client-name authorized-trainer
```

The worker certificate must match the HTTPS hostname. Only train partition and
its evidence/manifests belong on the worker; keep evaluator storage separate.
`python -m python_fixer.data.export --dataset <approved-data> --destination
<new-train-only-directory> --manifest-sha256 <pin>` prepares that subset and
SFT/RL model views without copying held-out test bundles.
Gateway rejects non-train data and caller-selected tests/commands/paths. Protect
runtime reports and manifest pins through permissions; hashes prove consistency,
not the identity of the actor who executed tests.

Remaining M1 gates: enforcing runtime, final complete attack/integration rerun,
current verified dataset and reviewed corpus acquisition/50-case audit before
scaling beyond fixtures. Independent review required further fixes and retained
the security blocker. Later model/research milestones remain unexecuted.

Independent read-only review found and rechecked fixes for manifest loading,
mandatory reward bindings, cumulative CPU budgets, clean CI installation,
completed-output bounds and packaging. Its final mutable-binding concern was
fixed by copying validated caller mappings, with a regression test. Review remains
REWORK for overall M1 because the deployment gate is blocked. The CPU check
report includes exact source fingerprint and nonzero executed test counts;
it never converts that blocker into an approved security verdict.
