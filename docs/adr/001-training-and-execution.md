# ADR-001: Modular TRL pipeline and isolated CPU execution

- Status: Accepted for specification; GPU/security feasibility unverified
- Date: 2026-11-23
- Owner: Project maintainer (implementation review pending)

## Context and decision

Test whether post-training a 0.6B model improves Python repair on one 24 GB GPU
without exposing training credentials or hidden tests to generated programs.
No existing application code or deployment stack exists. Favor reproducible
batch experiments, explicit contracts and disposable compute over a service platform.

| Option | Effort / maintenance | Performance / cost | Reliability / security | Portability / reversal |
|---|---|---|---|---|
| A: TRL and local hardened execution on the GPU VM | Simplest; one machine | Low transport overhead, CPU contention (estimates) | Candidate escape shares credential/checkpoint host; fails required boundary | Easy to move, unsafe initial authority scope |
| B: TRL GPU job + separate gVisor CPU gateway | Moderate adapter and two hosts | Remote queue latency, small CPU cost; unmeasured | Secrets/evaluation separate; bounded jobs and recoverable manifests | Plain Python/JSON contracts; swap runtime independently |
| C: Distributed trainer + vLLM service + microVM worker pool | Highest operational effort | Possible throughput gain; idle nodes and rollout complexity | More components and state; strongest worker isolation when configured | Expensive migration/maintenance; unnecessary initial scale |

Choose B. A is rejected by host-isolation requirements, despite lower effort.
C is deferred until measured reward or generation throughput warrants it. No
numeric benchmark or cost advantage is claimed. Use established GRPOTrainer and
PEFT; write only data/reward/sandbox adapters. No custom GRPO optimizer, Kubernetes
or workflow framework. JSON manifests own run state; model libraries own training.

## Consequences and challenge

The CPU worker is a hidden operational prerequisite: gVisor may be unavailable
on the selected provider, and network-bound rewards can stall the GPU. Worker
runtime must be tested before renting the GPU. If gVisor cannot pass containment
tests, use an independently reviewed microVM runner, not host execution. Private
expected values stay in the gateway, not the candidate interpreter. IAM and
host permissions, not role labels, enforce data separation.

BF16 LoRA avoids quantization confounds and is the default. QLoRA and 1.7B are
optional measured extensions. TRL 0.23.1 is an intentionally pinned API baseline;
dependency and security updates require revalidation, not indefinite use of an
old package. Current compatibility is documented-source evidence only.

## Verification and revisit

M1 CPU slice: 16 authored verified functions, four candidate variants each,
real runsc worker, attack suite. Observe latency p50/p95, resource use, leak tests
and all candidate outcomes. Pass only if isolation checks hold and trusted
counts/reward match the oracle. M3/M4: profile 16-candidate windows on a 4090;
target <=120 s/window and <22 GiB peak allocated memory, both provisional safety
budgets. If transport dominates step time, batch requests/cache immutable outcomes
before considering vLLM or distributed workers. If the worker cannot protect
secrets/tests or recovery fails, stop GPU launch. Results: NOT_RUN.
