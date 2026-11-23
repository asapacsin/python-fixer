# 08 — Security boundaries and research risks

## Threat model

Every candidate is hostile Python capable of introspection, infinite loops,
fork/memory bombs, filesystem/network probes, protocol forgery and exception abuse.
Training data can contain malicious source. Trust only the gateway/controller,
signed/versioned runtime image, test manifests and host kernel/runtime. Protect
GPU credentials, checkpoints, private tests, other tenants and the worker host.
Containers share a kernel and are not a complete boundary against kernel exploits;
use gVisor `runsc` on a dedicated disposable CPU VM, patched and separate from
the GPU node. Firecracker is a future stronger boundary if gVisor tests/compatibility
fail. Plain Docker without runtime hardening is not an approved fallback.

## Required isolation policy

[sandbox.json](../configs/sandbox.json) is a policy specification, not a daemon.
Each invocation uses a pinned minimal Python image by digest, runsc runtime,
non-root UID/GID 65532, read-only rootfs, network none, all capabilities dropped,
no-new-privileges, default-or-stricter seccomp with gVisor compatibility reviewed,
and no host PID/IPC/network namespaces. No privileged containers, host sockets,
devices or GPU access. No training/evaluation secrets, no inherited host env.
Mount only candidate module and a trusted invocation shim read-only. Private
expected values stay outside the container; no bundle of hidden tests is mounted.

Limits: 1 CPU, 256 MiB memory, no additional swap, 32 processes, 2 CPU seconds
per candidate suite and 5 seconds wall time, 16 MiB noexec/nosuid/nodev tmpfs,
64 KiB total stdout/stderr, 128 KiB request envelope. The gateway enforces a
**suite-wide** budget across per-call processes; per-test resets cannot multiply
the allowed runtime. Kill the entire job cgroup/container on limit, verify no
orphaned descendants, remove workspace and release concurrency slot. No test
completion counts survive forced termination for reward purposes.

Fresh candidate process per test call prevents state carryover. The outer
gateway owns expected values and comparison; candidate outputs are untrusted data
with strict JSON size/type/depth limits. A candidate cannot obtain success by
printing JSON fields such as `all_tests_pass`. AST restrictions reduce the attack
surface but are bypassable and never counted as the security boundary. Check
import allowlist, top-level statements, signature preservation and source length
inside the untrusted boundary before invocation.

## Minimum M1 security acceptance suite

| Attempt | Required observed behavior |
|---|---|
| Outbound HTTP/DNS and metadata IP | No successful connection; gateway records denial |
| Read host sentinel / training token / evaluator path | Unavailable; no secret in output or mounted paths |
| Write candidate/harness/test files | Read-only rejection, original host hashes unchanged |
| Infinite loop / sleeping / spawned descendants | Bounded wall/CPU termination, no surviving processes |
| Memory exhaustion / fork storm / output flood | Limits enforced; bounded response; worker remains healthy |
| Import os/subprocess/socket or introspection bypass | Policy rejection or containment, never host access |
| Print forged PASS / mutate assertion library | Parent comparator still determines result; no false PASS |
| Read expected test answers | Expected values absent from child namespace/files/environment |
| Path traversal / symlink / oversized nested JSON | Gateway rejects before host path resolution or execution |
| Exception / interpreter crash / malformed return | Candidate failure, no untrusted report accepted |
| Worker crash / stale cache / missing tests | INTERNAL_ERROR, no scalar reward, bounded retry |
| Cross-candidate tmp file | Fresh workspace; unavailable to next candidate |

Run malicious test programs only inside this boundary after checking runtime
configuration. The test harness must inspect the actual runtime settings and
host sentinels; an AST rejection alone does not prove network or filesystem
isolation. If nested sandbox/runtime support is unavailable locally, mark the
integration suite BLOCKED and keep metadata work moving; no host-run fallback.

## Research and operations risk register

| Risk | Detection / mitigation | Owner / blocking stage |
|---|---|---|
| Weak tests yield reward hacking | baseline verification, held-out train-only audit tests, manual sample | Data owner / M1,M4 |
| Cross-split family leakage | grouped split + clone audit + access separation | Data/evaluation owner / M1,M6 |
| Public benchmark contamination | separate reporting, authored holdouts, limitation statement | Research owner / M6 |
| SFT mask or Qwen template mismatch | collator label tests and decoded prefixes | Training owner / M3 |
| Zero-variance RL collapse | log group distribution, preregistered pause | Training owner / M4,M5 |
| GPU OOM / slow rewards | measured microbenchmark, bounded concurrency | Infrastructure owner / preflight |
| Lost checkpoints / phantom persistence | atomic manifests, read-back and replacement-node recovery | Infrastructure owner / paid runs |
| Dependency/API drift | candidate version tuple then resolved hash lock | Training owner / M3 |
| Underpowered comparisons | validation-based sensitivity analysis, qualified claims | Evaluation owner / M2,M6 |
| License violation | per-source/per-file license manifest, quarantine unknowns | Data owner / M1 |
| Hidden cost overrun | measured rate/budget cap and shutdown evidence | Operator / paid runs |

These roles are responsibilities, not claims that independent people/agents have
already signed off. The same implementation author cannot supply a fictitious
independent approval. Report missing separation and evidence explicitly.
