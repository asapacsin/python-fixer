# M1 security suite coverage (spec 08 table)

Real = a check in `python -m python_fixer.sandbox.preflight` executed against runsc
(release-20261109.0, systrap, cgroups enforced). Double = a unit test with an
execution-protocol double: it checks gateway logic but **does not certify
containment**. Gap = not exercised. Last real run: see the evidence file named in
[m1-data-audit.md](m1-data-audit.md) (32 checks at the time of writing). Unless noted,
probes use the single-call path; rows marked *suite* also exercise `invoke_suite`,
the path production verification and reward use.

| Spec 08 attempt | Coverage | Checks / tests |
|---|---|---|
| Outbound HTTP/DNS and metadata IP | Real | `network_denied` (TCP to 169.254.169.254), `dns_denied` (getaddrinfo) |
| Read host sentinel / training token / evaluator path | Real | `sentinel_absent`, `secret_env_absent`, `gpu_absent`, `docker_absent`, `forbidden-read` |
| Write candidate/harness/test files | Real | `rootfs_denied`, `host-sentinel-unchanged` |
| Infinite loop / sleeping / spawned descendants | Real | `loop`, `descendant-bounded` (setsid sleeper outlives parent; call returns, container removed), `no-container-leaks` |
| Memory exhaustion / fork storm / output flood | Real | `memory`, `container-memory-cap`, `process-cap` (refused at 31), `thread-storm-bounded` (refused at 31), `output-cap`; host-pids evidence in `.ai-native/m1-pids-limit-negative.json` (that file predates recording probe source and policy hash; the probe is the fork/thread loop in this commit's history) |
| Import os/subprocess/socket or introspection bypass | Real | `forbidden-import`, `introspection`, `signature-change`; probes that bypass the AST filter still hit the OS boundary above |
| Print forged PASS / mutate assertion library | Real (forged PASS); by design (assertions) | `forged-pass`; comparisons run in the trusted parent (`gateway.execute_suite`), no assertion library exists in the candidate process |
| Read expected test answers | Real (*suite* + per-test) | `expected-values-absent-suite`, `expected-values-absent-per-test`: the probe returns the sentinel if it is visible in stdin, environment, `/` or `/tmp`, so a leak would turn the test into a pass; `expected-probe-positive-control` runs in every preflight and requires the probe to report a sentinel deliberately placed in a request field |
| Path traversal / symlink / oversized nested JSON | Double + Real | Gateway accepts no caller paths/commands (`test_hash_unknown_problem_and_role_rejected`, request schema in `Gateway.handle`); oversized request envelope (128 KiB) in `DockerSandbox.invoke`; `nested-return` (200-deep return) is Real. Symlink handling of dataset/evidence roots: `evaluation.records` tests (M2). |
| Exception / interpreter crash / malformed return | Real, partial | `exception`, `recursion`, `nested-return`, `syntax`; *suite*: `big-int-return` (10**5000, over the 4300-digit JSON limit) and `surrogate-return` are candidate failures, not INTERNAL_ERROR. `segfault-candidate-failure` (pure-code segfault via a 10^6-deep tuple hash) is a candidate failure on both the single-call path (container exit 128+signal) and the suite path (per-test interpreter signal); both mappings also have behavioural unit tests. A malformed report from the trusted shim itself remains INTERNAL_ERROR (no report accepted). |
| Worker crash / stale cache / missing tests | Double | `test_full_reward_cache_and_request_binding`, `test_missing_frozen_binding_rejected_before_tls`, transport tests; bounded retry in `Gateway.handle` |
| Cross-candidate tmp file | Real | `fresh-workspace` |

Gaps that remain: a real worker crash mid-request (killed daemon) and a real stale-cache
scenario are only exercised with doubles; kernel-escape resistance is out of scope; the
dedicated worker and operator mTLS deployment have not been exercised.
