# ADR-002: Session container as the M1 CPU worker

- Status: Accepted by the user for M1 only
- Date: 2026-11-23
- Decided by: the user (project owner), recorded by the implementing agent

## Context

Spec 08 asks for gVisor `runsc` on a dedicated disposable CPU VM, separate from
the GPU node (ADR-001 option B). Its purpose is to keep generated code away from
GPU credentials, checkpoints, model storage, held-out tests and other tenants.
In M1 none of these exist: there is no GPU node, trainer, model or secret in this
session.

## Decision

For M1 only, the ephemeral cloud session container runs the runsc worker
(release-20261109.0, systrap, cgroups enforced) and is accepted as the M1 CPU worker.

## Consequences

- M1 sandbox evidence (preflight, dataset verification) comes from this container.
  It is ephemeral, so the runtime install is not persistent.
- Before any GPU trainer, model weights or credentials exist (M3/M4), the worker
  must move to a separate machine as in ADR-001. The operator-mTLS gateway
  deployment (`python -m python_fixer.sandbox.server`) is still unexercised.
- `configs/sandbox.json` keeps `"location": "separate_cpu_worker"` as the target
  for later milestones. This ADR is the documented M1 exception.
